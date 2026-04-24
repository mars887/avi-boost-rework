"""Shared encoder parameter token helpers."""

from __future__ import annotations

from typing import List, Optional, Tuple


def is_param_key(token: str) -> bool:
    return str(token or "").strip().startswith("--")


def find_last_option(tokens: List[str], key: str) -> Optional[Tuple[int, bool]]:
    for index in range(len(tokens) - 1, -1, -1):
        if tokens[index] != key:
            continue
        has_value = index + 1 < len(tokens) and not is_param_key(tokens[index + 1])
        return index, has_value
    return None


def apply_override(base_tokens: List[str], override_tokens: List[str]) -> List[str]:
    index = 0
    while index < len(override_tokens):
        token = override_tokens[index]
        if not is_param_key(token):
            index += 1
            continue

        has_value = index + 1 < len(override_tokens) and not is_param_key(override_tokens[index + 1])
        value = override_tokens[index + 1] if has_value else None
        location = find_last_option(base_tokens, token)
        if location is None:
            base_tokens.append(token)
            if value is not None:
                base_tokens.append(value)
        else:
            key_index, base_has_value = location
            if value is None:
                if base_has_value:
                    del base_tokens[key_index + 1]
            elif base_has_value:
                base_tokens[key_index + 1] = value
            else:
                base_tokens.insert(key_index + 1, value)
        index += 2 if has_value else 1
    return base_tokens


def strip_param_tokens(tokens: List[str], keys: List[str]) -> List[str]:
    keys_set = {str(key) for key in keys}
    out: List[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in keys_set:
            has_value = index + 1 < len(tokens) and not is_param_key(tokens[index + 1])
            index += 2 if has_value else 1
            continue
        out.append(token)
        index += 1
    return out
