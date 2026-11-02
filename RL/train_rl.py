"""Collect fixed rollout buffers, then perform multiple single-step GRPO updates."""
import argparse, copy, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rl.rollout_engine import RolloutEngine
from rl.grpo_trainer import GRPOTrainer, _sequence_logps
from verifier.math_evaluator import rule_reward

def collect_buffer(model, ref_model, tokenizer, engine, prompts, references, generations):
    groups = engine.generate(prompts, generations)
    flat_prompts = [p for p, group in zip(prompts, groups) for _ in group]
    completions = [c for group in groups for c in group]
    flat_refs = [r for r, group in zip(references, groups) for _ in group]
    rewards = torch.tensor([rule_reward(c, r)[0] for c, r in zip(completions, flat_refs)], dtype=torch.float32)
    grouped = rewards.view(len(prompts), generations)
    advantages = ((grouped-grouped.mean(-1,keepdim=True)) /
                  (grouped.std(-1,keepdim=True,unbiased=False)+1e-4)).flatten()
    device = next(model.parameters()).device
    encoded = tokenizer([p+c for p,c in zip(flat_prompts,completions)], padding=True,
                       truncation=True, return_tensors="pt")
    prompts_encoded = tokenizer(flat_prompts, padding=True, truncation=True, return_tensors="pt")
    ids, mask = encoded.input_ids.to(device), encoded.attention_mask.to(device)
    # Left padding is required so prompt lengths identify the response boundary in concatenated inputs.
    prompt_lens = prompts_encoded.attention_mask.sum(-1).to(device)
    with torch.no_grad():
        old = _sequence_logps(model, ids, mask, prompt_lens)
        ref = _sequence_logps(ref_model, ids, mask, prompt_lens)
    return {"input_ids":ids, "attention_mask":mask, "prompt_lens":prompt_lens,
            "old_logps":old.detach(), "ref_logps":ref.detach(),
            "advantages":advantages.to(device), "rewards":rewards}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--data",required=True)
    ap.add_argument("--steps",type=int,default=100); ap.add_argument("--batch-size",type=int,default=4)
    ap.add_argument("--generations",type=int,default=4); ap.add_argument("--update-epochs",type=int,default=4)
    ap.add_argument("--lr",type=float,default=1e-6); ap.add_argument("--no-vllm",action="store_true")
    ap.add_argument("--output",default="checkpoints/grpo")
    a=ap.parse_args(); tok=AutoTokenizer.from_pretrained(a.model)
    tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype="auto",device_map="auto")
    ref=copy.deepcopy(model).eval()
    for p in ref.parameters(): p.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
    with open(a.data,encoding="utf-8") as f: rows=[json.loads(x) for x in f if x.strip()]
    if not rows: raise ValueError("GRPO dataset is empty")
    prompts=[r["prompt"] for r in rows]; refs=[r.get("reference_answer",r.get("answer")) for r in rows]
    if any(x is None for x in refs): raise ValueError("GRPO records need reference_answer or answer")
    engine=RolloutEngine(a.model,tok,use_vllm=not a.no_vllm,max_tokens=512,temperature=.8,top_p=.95)
    trainer=GRPOTrainer(model,ref,tok,opt)
    for step in range(a.steps):
        start=(step*a.batch_size)%len(rows); ix=[(start+j)%len(rows) for j in range(a.batch_size)]
        buffer=collect_buffer(model,ref,tok,engine,[prompts[i] for i in ix],[refs[i] for i in ix],a.generations)
        for epoch in range(a.update_epochs):
            metrics=trainer.update(buffer)
            print(f"step={step} update={epoch+1}/{a.update_epochs} {metrics}")
    Path(a.output).mkdir(parents=True,exist_ok=True)
    model.save_pretrained(a.output); tok.save_pretrained(a.output)
if __name__=="__main__": main()
