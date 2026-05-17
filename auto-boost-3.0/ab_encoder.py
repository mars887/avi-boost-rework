"""Encoder normalization and encoder-specific parameter helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

PARENT_DIR = Path(__file__).resolve().parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from utils.av1an_hdr_metadata_patch_v2 import build_fastpass_hdr10_params, build_x265_hdr10_params


DEFAULT_ENCODER = "svt-av1"
_FAST_PRESET_DEFAULTS = {
    "svt-av1": "7",
    "x265": "fast",
}
_FINAL_PRESET_DEFAULTS = {
    "svt-av1": "2",
    "x265": "slow",
}
def normalize_encoder(value: Optional[str], *, default: str = DEFAULT_ENCODER) -> str:
    raw = str(value or "").strip().lower().replace("_", "-")
    if raw in ("", "default", "auto"):
        raw = default
    if raw == "svt-av1":
        return "svt-av1"
    if raw in ("x265", "libx265"):
        return "x265"
    raise ValueError(f"Unsupported encoder: {value!r}")


def scene_encoder_name(encoder: str) -> str:
    normalized = normalize_encoder(encoder)
    if normalized == "svt-av1":
        return "svt_av1"
    return normalized


def resolve_preset(encoder: str, preset: Optional[str], *, fast: bool) -> str:
    normalized = normalize_encoder(encoder)
    text = str(preset or "").strip()
    if text:
        return text
    defaults = _FAST_PRESET_DEFAULTS if fast else _FINAL_PRESET_DEFAULTS
    return defaults[normalized]


def build_fastpass_params(
    *,
    encoder: str,
    preset: str,
    crf: float,
    lp: int,
    video_params: str,
) -> str:
    normalized = normalize_encoder(encoder)
    parts: List[str] = ["--preset", str(preset).strip(), "--crf", f"{float(crf):.2f}"]
    if normalized == "svt-av1":
        parts.extend(["--lp", str(int(lp))])
    extra = str(video_params or "").strip()
    if extra:
        parts.append(extra)
    return " ".join(part for part in parts if part)
