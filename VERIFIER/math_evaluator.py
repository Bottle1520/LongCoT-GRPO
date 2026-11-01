"""Deterministic rule-based reward; no learned reward model is used."""
from __future__ import annotations
import re
from typing import Any
from .math_parser import parse_response

def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().replace("$", "")).rstrip(".")

def equivalent(prediction: str, reference: str) -> bool:
    if _norm(prediction) == _norm(reference):
        return True
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
        trans = standard_transformations + (implicit_multiplication_application,)
        a = parse_expr(prediction.replace("^", "**"), transformations=trans)
        b = parse_expr(reference.replace("^", "**"), transformations=trans)
        return bool(sympy.simplify(a-b) == 0)
    except Exception:
        return False

def rule_reward(response: str, reference: str, *, format_bonus: float = 0.1,
                think_bonus: float = 0.0) -> tuple[float, dict[str, Any]]:
    parsed = parse_response(response)
    valid = parsed["answer"] is not None
    correct = bool(valid and equivalent(parsed["answer"] or "", reference))
    reward = (1.0 if correct else 0.0) + (format_bonus if valid else 0.0)
    if think_bonus and parsed["think"]:
        reward += think_bonus
    return reward, {"correct": correct, "format_valid": valid}
