"""Prepare cold-start SFT examples from a JSON/JSONL math-reasoning source.

Accepted input keys: question/prompt/problem and answer/solution/response.
Output is JSONL with prompt, response, and source_id fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError(f"{path}:{line_no}: expected a JSON object")
                    yield item
    else:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data", data.get("train", [data]))
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of objects or contain a data/train list")
        yield from data


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def format_record(record: dict[str, Any], index: int) -> dict[str, str] | None:
    question = _first(record, ("question", "prompt", "problem"))
    answer = _first(record, ("answer", "solution", "response"))
    if question is None or answer is None:
        return None
    # Preserve authored reasoning when present; provide the expected tags for the verifier.
    if "<answer>" not in answer:
        answer = f"<think>\n{answer}\n</think>\n<answer>{answer}</answer>"
    return {"prompt": question, "response": answer,
            "source_id": str(record.get("id", index))}


def prepare(input_path: Path, output_path: Path) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    with output_path.open("w", encoding="utf-8") as out:
        for i, record in enumerate(_records(input_path)):
            if not isinstance(record, dict):
                skipped += 1
                continue
            item = format_record(record, i)
            if item is None:
                skipped += 1
                continue
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
    return kept, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source .json or .jsonl")
    parser.add_argument("output", type=Path, help="Destination .jsonl")
    args = parser.parse_args()
    kept, skipped = prepare(args.input, args.output)
    print(f"Wrote {kept} records; skipped {skipped} invalid/incomplete records.")


if __name__ == "__main__":
    main()
