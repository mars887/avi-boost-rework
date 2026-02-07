"""Encoder parameter parsing and normalization utilities."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Tuple

from ab_logging import eprint

_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")

def strip_params_tokens(tokens: List[str], *, keys: Sequence[str]) -> List[str]:
    keyset = set(keys)
    out: List[str] = []
    skip_next = False
    for t in tokens:
        if skip_next:
            skip_next = False
            continue
        if t in keyset:
            skip_next = True
            continue
        out.append(t)
    return out

def is_param_key(tok: str) -> bool:
    # Treat only long options as keys to avoid breaking values like -1
    return tok.startswith("--")

def normalize_param_key(name: str) -> str:
    s = str(name).strip()
    if not s:
        raise ValueError("Empty parameter name.")
    if s.startswith("--"):
        return s
    if s.startswith("-"):
        return "--" + s.lstrip("-")
    return "--" + s

def find_last_option(tokens: List[str], key: str) -> Optional[Tuple[int, bool]]:
    # Returns (index_of_key, has_value_after_key)
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] == key:
            has_val = (i + 1 < len(tokens)) and (not is_param_key(tokens[i + 1]))
            return i, has_val
    return None

def coerce_param_value(token: str) -> Any:
    if _NUMBER_RE.fullmatch(token):
        if re.fullmatch(r"-?\d+", token):
            try:
                return int(token)
            except Exception:
                return token
        try:
            return float(token)
        except Exception:
            return token
    return token

def parse_numeric_value(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and _NUMBER_RE.fullmatch(value.strip()):
        return float(value)
    return None

def format_param_value(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)

def normalize_video_params(tokens: List[str]) -> None:
    crf_loc = find_last_option(tokens, "--crf")
    if crf_loc is not None:
        k_idx, has_val = crf_loc
        if has_val:
            cur = parse_numeric_value(tokens[k_idx + 1])
            if cur is None:
                eprint(f"[warn] --crf value is not numeric: {tokens[k_idx + 1]}")
            else:
                cur = max(0.0, min(63.0, float(cur)))
                tokens[k_idx + 1] = f"{cur:.2f}"
        else:
            eprint("[warn] --crf flag present without value.")

    preset_loc = find_last_option(tokens, "--preset")
    if preset_loc is not None:
        k_idx, has_val = preset_loc
        if has_val:
            cur = parse_numeric_value(tokens[k_idx + 1])
            if cur is None:
                eprint(f"[warn] --preset value is not numeric: {tokens[k_idx + 1]}")
            else:
                tokens[k_idx + 1] = str(int(cur))
        else:
            eprint("[warn] --preset flag present without value.")
