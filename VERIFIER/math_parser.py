"""Parse the expected <think> and <answer> response sections."""
import re

def parse_response(text: str) -> dict[str, str | None]:
    def extract(tag: str) -> str | None:
        match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.S | re.I)
        return match.group(1).strip() if match else None
    return {"think": extract("think"), "answer": extract("answer")}
