"""Minimal LoRA injection for nn.Linear modules (no PEFT dependency)."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """Frozen linear base plus trainable low-rank A/B update."""
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0: raise ValueError("rank must be positive")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None: self.base.bias.requires_grad_(False)
        self.rank, self.scaling = rank, alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device,
                                                dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device,
                                               dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x):
        base_out = self.base(x)
        delta = F.linear(F.linear(self.dropout(x).to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return base_out + delta.to(base_out.dtype) * self.scaling

    @torch.no_grad()
    def merge(self):
        self.base.weight.add_((self.lora_B @ self.lora_A).to(self.base.weight.dtype), alpha=self.scaling)
        return self.base

def inject_lora(model: nn.Module, *, rank=8, alpha=16.0, dropout=0.0,
                target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                                   "gate_proj", "up_proj", "down_proj")) -> int:
    """Replace matching Linear modules by suffix; return number replaced."""
    for p in model.parameters(): p.requires_grad_(False)
    replacements = []
    for full_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(full_name.endswith(name) for name in target_modules):
            parent_name, _, child_name = full_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            replacements.append((parent, child_name, module))
    for parent, name, layer in replacements:
        setattr(parent, name, LoRALinear(layer, rank, alpha, dropout))
    if not replacements:
        raise ValueError(f"No Linear layers matched target_modules={target_modules}")
    return len(replacements)
