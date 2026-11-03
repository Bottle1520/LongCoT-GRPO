"""Shared SFT and GRPO dataset/data-loader utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader


class MathReasoningDataset(Dataset):
    """Reads JSONL records containing prompt and optionally response/answer."""
    def __init__(self, path: str | Path, tokenizer: Any | None = None,
                 max_length: int = 2048, mode: str = "sft"):
        if mode not in {"sft", "grpo"}:
            raise ValueError("mode must be 'sft' or 'grpo'")
        self.tokenizer, self.max_length, self.mode = tokenizer, max_length, mode
        self.rows: list[dict[str, Any]] = []
        with Path(path).open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
                    raise ValueError(f"Invalid prompt at {path}:{line_no}")
                if mode == "sft" and not isinstance(row.get("response"), str):
                    raise ValueError(f"SFT response missing at {path}:{line_no}")
                self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        result = {"prompt": row["prompt"], "source_id": row.get("source_id", str(index))}
        if self.mode == "sft":
            result["response"] = row["response"]
        if self.tokenizer is not None:
            prompt = self.tokenizer(row["prompt"], truncation=True,
                                    max_length=self.max_length, add_special_tokens=True)
            result["prompt_input_ids"] = prompt["input_ids"]
            result["prompt_attention_mask"] = prompt["attention_mask"]
            if self.mode == "sft":
                text = row["prompt"] + row["response"]
                encoded = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                         add_special_tokens=True)
                result["input_ids"] = encoded["input_ids"]
                result["attention_mask"] = encoded["attention_mask"]
                labels = list(encoded["input_ids"])
                prompt_len = min(len(prompt["input_ids"]), len(labels))
                labels[:prompt_len] = [-100] * prompt_len
                result["labels"] = labels
        return result
"""
（与Qwen聊天模板对齐的标签化数据实现）
def __getitem__(self, index: int) -> Dict[str, Any]:
    row = self.rows[index]
    
    # 1. 构造标准对话结构（Messages 格式）
    messages = [
        {"role": "user", "content": row["prompt"]},
    ]
    if self.mode == "sft":
        messages.append({"role": "assistant", "content": row["response"]})

    result = {
        "prompt": row["prompt"], 
        "source_id": row.get("source_id", str(index))
    }
    if self.mode == "sft":
        result["response"] = row["response"]

    if self.tokenizer is not None:
        # 2. 单独获取 Prompt 的 Token 长度（用于后续切分 Loss 掩码）
        # add_generation_prompt=True 会自动在末尾加上如 "<|im_start|>assistant\n" 的引导头
        prompt_text = self.tokenizer.apply_chat_template(
            [messages[0]], 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompt_encoded = self.tokenizer(
            prompt_text, 
            truncation=True, 
            max_length=self.max_length, 
            add_special_tokens=False  # chat template 内部已包含特殊控制 Token
        )
        result["prompt_input_ids"] = prompt_encoded["input_ids"]
        result["prompt_attention_mask"] = prompt_encoded["attention_mask"]

        # 3. 获取完整对话（Prompt + Response）的 Token 编码
        full_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        encoded = self.tokenizer(
            full_text, 
            truncation=True, 
            max_length=self.max_length, 
            add_special_tokens=False
        )
        
        result["input_ids"] = encoded["input_ids"]
        result["attention_mask"] = encoded["attention_mask"]

        # 4. 构造 SFT 的 Labels（将 Prompt 部分设为 -100）
        if self.mode == "sft":
            labels = list(encoded["input_ids"])
            # 利用 prompt 编码的实际长度作为截断和掩码分界线
            prompt_len = min(len(prompt_encoded["input_ids"]), len(labels))
            labels[:prompt_len] = [-100] * prompt_len
            result["labels"] = labels

    return result
"""


def _collate(tokenizer: Any | None):
    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {"prompt": [r["prompt"] for r in rows],
                                 "source_id": [r["source_id"] for r in rows]}
        if "response" in rows[0]:
            batch["response"] = [r["response"] for r in rows]
        for key in ("prompt_input_ids", "prompt_attention_mask", "input_ids", "attention_mask", "labels"):
            if key not in rows[0]:
                continue
            pad = -100 if key == "labels" else (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
            seqs = [torch.tensor(r[key], dtype=torch.long) for r in rows]
            batch[key] = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True,
                                                         padding_value=pad)
        return batch
    return collate


def create_dataloader(path: str | Path, tokenizer: Any | None = None,
                      batch_size: int = 4, max_length: int = 2048,
                      mode: str = "sft", shuffle: bool | None = None,
                      num_workers: int = 0) -> DataLoader:
    dataset = MathReasoningDataset(path, tokenizer, max_length, mode)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=(mode == "sft" if shuffle is None else shuffle),
                      num_workers=num_workers, collate_fn=_collate(tokenizer))


def get_sft_dataloader(path: str | Path, tokenizer: Any, **kwargs: Any) -> DataLoader:
    return create_dataloader(path, tokenizer, mode="sft", **kwargs)


def get_grpo_dataloader(path: str | Path, tokenizer: Any | None = None, **kwargs: Any) -> DataLoader:
    return create_dataloader(path, tokenizer, mode="grpo", **kwargs)
