"""Parse the expected <think> and <answer> response sections."""
import re
def parse_response(text: str) -> dict[str, str | None]:
    def extract(tag: str) -> str | None:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        pattern = open_tag + r"(.*?)" + close_tag
        match_flags = re.DOTALL | re.IGNORECASE
        match = re.search(pattern, text, flags=match_flags)
        if match:
            content = match.group(1)
            return content.strip()
        else:
            return None

    return {
        "think": extract("think"), 
        "answer": extract("answer")
    }
