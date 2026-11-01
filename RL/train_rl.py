"""Example GRPO entry point. Expects JSONL prompt/reference records."""
import argparse, copy, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rl.rollout_engine import RolloutEngine
from rl.grpo_trainer import GRPOTrainer

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--data",required=True)
    ap.add_argument("--steps",type=int,default=100); ap.add_argument("--batch-size",type=int,default=4)
    ap.add_argument("--generations",type=int,default=4); ap.add_argument("--lr",type=float,default=1e-6)
    ap.add_argument("--no-vllm",action="store_true"); ap.add_argument("--output",default="checkpoints/grpo")
    a=ap.parse_args(); tok=AutoTokenizer.from_pretrained(a.model); tok.pad_token=tok.pad_token or tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype="auto",device_map="auto")
    ref=copy.deepcopy(model).eval()
    for p in ref.parameters(): p.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
    rows=[json.loads(x) for x in open(a.data,encoding="utf-8") if x.strip()]
    prompts=[r["prompt"] for r in rows]; refs=[r.get("reference_answer",r.get("answer")) for r in rows]
    if any(x is None for x in refs): raise ValueError("GRPO records need reference_answer or answer")
    engine=RolloutEngine(a.model,tok,use_vllm=not a.no_vllm,max_tokens=512,temperature=.8,top_p=.95)
    trainer=GRPOTrainer(model,ref,tok,opt,n_generations=a.generations)
    for step in range(a.steps):
        start=(step*a.batch_size)%len(rows); ix=[(start+j)%len(rows) for j in range(a.batch_size)]
        print(step,trainer.step([prompts[i] for i in ix],[refs[i] for i in ix],engine))
    model.save_pretrained(a.output); tok.save_pretrained(a.output)
if __name__=="__main__": main()
