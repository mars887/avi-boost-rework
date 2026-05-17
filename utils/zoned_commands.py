from __future__ import annotations

import keyword
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ZONED_COMMAND_NAME = "zoned_command.txt"
LEGACY_ZONE_COMMAND_NAME = "zone_edit_command.txt"
LEGACY_CROP_RESIZE_COMMAND_NAME = "crop_resize_command.txt"

_DIVIDERS = ("-", "!", "^")
_EXPLICIT_CROP = "_CROP"
_EXPLICIT_VPY = "_VPY"
_SIZE_SEP_CHARS = "*:/xX"
_RX_SIMPLE_CROP = re.compile(r"^\d.*[*:/xX].*\d$")
_RX_CROP_OP = re.compile(r"^crop=\d.*", re.IGNORECASE)
_RX_SCALE_OP = re.compile(r"^scale=-?\d.*", re.IGNORECASE)


@dataclass(frozen=True)
class ZonedCommandSplit:
    selectors_part: str
    sep: str
    payload_tokens: Tuple[str, ...]
    option_tokens: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ZonedPayload:
    params: Tuple[str, ...] = ()
    crop: str = ""
    vpy_function: str = ""


def zoned_command_path(workdir: Path) -> Path:
    return Path(workdir) / ZONED_COMMAND_NAME


def legacy_zone_command_path(workdir: Path) -> Path:
    return Path(workdir) / LEGACY_ZONE_COMMAND_NAME


def legacy_crop_resize_command_path(workdir: Path) -> Path:
    return Path(workdir) / LEGACY_CROP_RESIZE_COMMAND_NAME


def read_text_guess(path: Path) -> str:
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return normalize_zoned_text(raw.decode(encoding))
        except Exception:
            continue
    return normalize_zoned_text(raw.decode("utf-8", errors="replace"))


def normalize_zoned_text(text: str) -> str:
    return (
        str(text or "")
        .replace("\u00c2\u00a0", " ")
        .replace("\u00a0", " ")
        .replace("\u202f", " ")
        .replace("\u2007", " ")
        .replace("\ufeff", "")
        .replace("\u200b", "")
    )


def merge_legacy_zoned_text(zone_text: str, crop_resize_text: str) -> str:
    parts: List[str] = []
    zone = str(zone_text or "").strip()
    crop = str(crop_resize_text or "").strip()
    if zone:
        parts.append(zone)
    if crop:
        parts.append(crop)
    if not parts:
        return ""
    return "\n\n".join(parts).rstrip() + "\n"


def read_zoned_command_text(workdir: Path, zoned_path: Path | None = None) -> str:
    path = Path(zoned_path) if zoned_path is not None else zoned_command_path(workdir)
    if path.exists():
        return read_text_guess(path)

    legacy_zone = legacy_zone_command_path(workdir)
    legacy_crop = legacy_crop_resize_command_path(workdir)
    zone_text = read_text_guess(legacy_zone) if legacy_zone.exists() else ""
    crop_text = read_text_guess(legacy_crop) if legacy_crop.exists() else ""
    return merge_legacy_zoned_text(zone_text, crop_text)


def ensure_zoned_command_file(workdir: Path, zoned_path: Path | None = None) -> Path:
    path = Path(zoned_path) if zoned_path is not None else zoned_command_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path

    text = read_zoned_command_text(workdir, path)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def combine_zoned_texts(zone_text: str, crop_resize_text: str) -> str:
    zone = str(zone_text or "").strip()
    crop = str(crop_resize_text or "").strip()
    if zone and crop and zone != crop:
        return f"{zone}\n\n{crop}\n"
    if zone:
        return zone + "\n"
    if crop:
        return crop + "\n"
    return ""


def _strip_line(raw: str) -> str:
    line = normalize_zoned_text(str(raw or "")).strip()
    if not line or line.startswith("#") or line.startswith("//"):
        return ""
    return line


def _space_boundary_bars(line: str) -> str:
    out: List[str] = []
    quote = ""
    for ch in line:
        if ch in ("'", '"'):
            if quote == ch:
                quote = ""
            elif not quote:
                quote = ch
            out.append(ch)
            continue
        if ch == "|" and not quote:
            out.append(" | ")
            continue
        out.append(ch)
    return "".join(out)


def split_zoned_command_line(line: str) -> ZonedCommandSplit:
    tokens = shlex.split(_space_boundary_bars(line), posix=False)
    if not tokens:
        raise ValueError("empty command line")

    if "|" in tokens:
        first_bar = tokens.index("|")
        try:
            second_bar = tokens.index("|", first_bar + 1)
        except ValueError as exc:
            raise ValueError("boundary command must contain two standalone '|' separators") from exc
        selectors_part = " ".join(tokens[:first_bar]).strip()
        option_tokens = tuple(tokens[first_bar + 1 : second_bar])
        payload_tokens = tuple(tokens[second_bar + 1 :])
        if not selectors_part:
            raise ValueError("missing selectors part before boundary options")
        if not payload_tokens:
            raise ValueError("missing payload after boundary options")
        return ZonedCommandSplit(selectors_part, "|", payload_tokens, option_tokens)

    sep_idx = None
    sep = ""
    for idx, token in enumerate(tokens):
        if token in _DIVIDERS:
            sep_idx = idx
            sep = token
            break
    if sep_idx is None:
        raise ValueError("command separator not found")

    selectors_part = " ".join(tokens[:sep_idx]).strip()
    payload_tokens = tuple(tokens[sep_idx + 1 :])
    if not selectors_part:
        raise ValueError("missing selectors part before separator")
    if not payload_tokens:
        raise ValueError("missing payload after separator")
    return ZonedCommandSplit(selectors_part, sep, payload_tokens)


def _is_preset_selector(selectors_part: str) -> bool:
    text = str(selectors_part or "").strip()
    return bool(text.startswith("@") and text != "@default" and not any(ch.isspace() for ch in text))


def _collect_preset_tokens(lines: Iterable[str]) -> Dict[str, Tuple[str, ...]]:
    presets: Dict[str, Tuple[str, ...]] = {}
    for raw in lines:
        line = _strip_line(raw)
        if not line:
            continue
        try:
            split = split_zoned_command_line(line)
        except ValueError:
            continue
        if _is_preset_selector(split.selectors_part):
            presets[split.selectors_part] = split.payload_tokens
    return presets


def _expand_preset_tokens(
    tokens: Sequence[str],
    presets: Dict[str, Tuple[str, ...]],
    *,
    _stack: Tuple[str, ...] = (),
) -> Tuple[str, ...]:
    out: List[str] = []
    for token in tokens:
        text = str(token)
        if text.startswith("@") and text in presets:
            if text in _stack:
                cycle = " -> ".join(_stack + (text,))
                raise ValueError(f"preset cycle detected: {cycle}")
            out.extend(_expand_preset_tokens(presets[text], presets, _stack=_stack + (text,)))
        else:
            out.append(text)
    return tuple(out)


def _is_marker(token: str) -> bool:
    return str(token or "").strip().upper() in (_EXPLICIT_CROP, _EXPLICIT_VPY)


def _is_single_crop_token(token: str) -> bool:
    text = str(token or "").strip()
    if not text:
        return False
    if _RX_CROP_OP.match(text) or _RX_SCALE_OP.match(text):
        return True
    return bool(
        text[0].isdigit()
        and text[-1].isdigit()
        and any(ch in text for ch in _SIZE_SEP_CHARS)
        and _RX_SIMPLE_CROP.match(text)
    )


def is_crop_token(token: str) -> bool:
    text = str(token or "").strip()
    if _is_single_crop_token(text):
        return True
    if "->" not in text:
        return False
    left, right = (part.strip() for part in text.split("->", 1))
    return bool(left and right and _is_single_crop_token(left) and _is_single_crop_token(right))


def is_vpy_function_token(token: str) -> bool:
    text = str(token or "").strip()
    return bool(text and text.isidentifier() and not keyword.iskeyword(text))


def _consume_crop_value(tokens: Sequence[str], start: int) -> Tuple[str, int]:
    if start >= len(tokens):
        return "", start
    token = str(tokens[start]).strip()
    if not is_crop_token(token):
        return "", start
    if "->" in token:
        return token, start + 1
    if start + 2 < len(tokens):
        arrow = str(tokens[start + 1]).strip()
        right = str(tokens[start + 2]).strip()
        if arrow == "->" and is_crop_token(right):
            return f"{token} -> {right}", start + 3
    return token, start + 1


def parse_zoned_payload(tokens: Sequence[str]) -> ZonedPayload:
    params: List[str] = []
    crop = ""
    function = ""

    i = 0
    while i < len(tokens):
        token = str(tokens[i]).strip()
        upper = token.upper()
        if upper == _EXPLICIT_CROP:
            crop, i = _consume_crop_value(tokens, i + 1)
            if not crop:
                raise ValueError("_CROP requires a value")
            continue
        if upper == _EXPLICIT_VPY:
            if i + 1 >= len(tokens):
                raise ValueError("_VPY requires a function name")
            candidate = str(tokens[i + 1]).strip()
            if not is_vpy_function_token(candidate):
                raise ValueError(f"invalid _VPY function name: {candidate!r}")
            function = candidate
            i += 2
            continue
        if token.startswith("-"):
            params.append(token)
            if i + 1 < len(tokens) and not _is_marker(str(tokens[i + 1])):
                params.append(str(tokens[i + 1]))
                i += 2
            else:
                i += 1
            continue
        if not crop and is_crop_token(token):
            crop, i = _consume_crop_value(tokens, i)
            continue
        if not function and is_vpy_function_token(token):
            if i + 1 < len(tokens):
                next_token = str(tokens[i + 1]).strip()
                if (
                    next_token
                    and not next_token.startswith("-")
                    and not _is_marker(next_token)
                    and not is_crop_token(next_token)
                ):
                    params.extend([token, next_token])
                    i += 2
                    continue
            function = token
            i += 1
            continue

        params.append(token)
        i += 1

    return ZonedPayload(params=tuple(params), crop=crop, vpy_function=function)


def _format_payload(tokens: Sequence[str]) -> str:
    return " ".join(str(token).strip() for token in tokens if str(token).strip()).strip()


def _format_zone_line(split: ZonedCommandSplit, params: Sequence[str]) -> str:
    payload = _format_payload(params)
    if split.sep == "|":
        return f"{split.selectors_part} | {_format_payload(split.option_tokens)} | {payload}".strip()
    return f"{split.selectors_part} {split.sep} {payload}".strip()


def _format_simple_line(selectors_part: str, payload: str) -> str:
    return f"{selectors_part} - {payload}".strip()


def project_zone_command_lines(lines: Iterable[str]) -> List[str]:
    source_lines = list(lines)
    presets = _collect_preset_tokens(source_lines)
    out: List[str] = []
    for raw in source_lines:
        line = _strip_line(raw)
        if not line:
            continue
        try:
            split = split_zoned_command_line(line)
            tokens = _expand_preset_tokens(split.payload_tokens, presets)
            payload = parse_zoned_payload(tokens)
        except ValueError:
            out.append(line)
            continue
        if payload.params:
            out.append(_format_zone_line(split, payload.params))
    return out


def project_crop_resize_command_lines(lines: Iterable[str]) -> List[str]:
    source_lines = list(lines)
    presets = _collect_preset_tokens(source_lines)
    out: List[str] = []
    for raw in source_lines:
        line = _strip_line(raw)
        if not line:
            continue
        try:
            split = split_zoned_command_line(line)
            if _is_preset_selector(split.selectors_part):
                continue
            tokens = _expand_preset_tokens(split.payload_tokens, presets)
            payload = parse_zoned_payload(tokens)
        except ValueError:
            continue
        if payload.crop:
            out.append(_format_simple_line(split.selectors_part, payload.crop))
    return out


def project_vpy_function_lines(lines: Iterable[str]) -> List[str]:
    source_lines = list(lines)
    presets = _collect_preset_tokens(source_lines)
    out: List[str] = []
    for raw in source_lines:
        line = _strip_line(raw)
        if not line:
            continue
        try:
            split = split_zoned_command_line(line)
            if _is_preset_selector(split.selectors_part):
                continue
            tokens = _expand_preset_tokens(split.payload_tokens, presets)
            payload = parse_zoned_payload(tokens)
        except ValueError:
            continue
        if payload.vpy_function:
            out.append(_format_simple_line(split.selectors_part, payload.vpy_function))
    return out
