"""Cold-start causal LM SFT with locally injected LoRA adapters."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.dataset_loaders import get_sft_dataloader
from sft.lora import inject_lora, LoRALinear

def save_lora(model, path: str | Path, config: dict) -> None:
    """Save adapter tensors and config separately from the frozen base model."""
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if ".lora_A" in k or ".lora_B" in k}
    torch.save(state, path / "adapter_model.pt")
    import json
    (path / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base Qwen model ID or local path")
    ap.add_argument("--data", required=True, help="Prepared SFT JSONL with prompt/response")
    ap.add_argument("--output", default="checkpoints/sft-lora")
    ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=2048); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=8); ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype="auto", device_map="auto")
    target_modules = tuple(x.strip() for x in a.targets.split(",") if x.strip())
    count = inject_lora(model, rank=a.rank, alpha=a.alpha, dropout=a.dropout, target_modules=target_modules)
    print(f"Injected LoRA into {count} linear layers")
    loader = get_sft_dataloader(a.data, tok, batch_size=a.batch_size, max_length=a.max_length)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=a.lr)
    model.train()
    for epoch in range(a.epochs):
        for step, batch in enumerate(loader):
            batch = {k: v.to(model.get_input_embeddings().weight.device) for k, v in batch.items()
                     if k in {"input_ids", "attention_mask", "labels"}}
            loss = model(**batch).loss
            loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if step % 10 == 0: print(f"epoch={epoch+1} step={step} loss={loss.item():.4f}")
    config = {"base_model": a.model, "rank": a.rank, "alpha": a.alpha,
              "dropout": a.dropout, "target_modules": list(target_modules)}
    save_lora(model, a.output, config); tok.save_pretrained(a.output)
    print(f"Saved LoRA adapter to {a.output}")

if __name__ == "__main__": main()
