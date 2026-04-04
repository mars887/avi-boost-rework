"""Encoder normalization and encoder-specific parameter helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


DEFAULT_ENCODER = "svt-av1"
_FAST_PRESET_DEFAULTS = {
    "svt-av1": "7",
    "x265": "fast",
}
_FINAL_PRESET_DEFAULTS = {
    "svt-av1": "2",
    "x265": "slow",
}
_MASTER_DISPLAY_PATTERN = re.compile(
    r"^G\((?P<gx>[0-9.]+),(?P<gy>[0-9.]+)\)"
    r"B\((?P<bx>[0-9.]+),(?P<by>[0-9.]+)\)"
    r"R\((?P<rx>[0-9.]+),(?P<ry>[0-9.]+)\)"
    r"WP\((?P<wpx>[0-9.]+),(?P<wpy>[0-9.]+)\)"
    r"L\((?P<max_l>[0-9.]+),(?P<min_l>[0-9.]+)\)$"
)


def _normalize_x265_value(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def normalize_x265_colorprim(value: Any) -> Optional[str]:
    v = _normalize_x265_value(value)
    if not v:
        return None
    mapping = {
        "bt709": "bt709",
        "bt2020": "bt2020",
        "display-p3": "smpte432",
        "display p3": "smpte432",
        "p3-d65": "smpte432",
        "p3 d65": "smpte432",
        "smpte432": "smpte432",
    }
    return mapping.get(v, v)


def normalize_x265_transfer(value: Any) -> Optional[str]:
    v = _normalize_x265_value(value)
    if not v:
        return None
    mapping = {
        "pq": "smpte2084",
        "smpte2084": "smpte2084",
        "hlg": "arib-std-b67",
        "arib-std-b67": "arib-std-b67",
    }
    return mapping.get(v, v)


def normalize_x265_matrix(value: Any) -> Optional[str]:
    v = _normalize_x265_value(value)
    if not v:
        return None
    mapping = {
        "bt2020-ncl": "bt2020nc",
        "bt2020 non-constant": "bt2020nc",
        "bt2020nc": "bt2020nc",
        "bt2020-cl": "bt2020c",
        "bt2020 constant": "bt2020c",
        "bt2020c": "bt2020c",
    }
    return mapping.get(v, v)


def normalize_x265_range(value: Any) -> Optional[str]:
    v = _normalize_x265_value(value)
    if not v:
        return None
    mapping = {
        "tv": "limited",
        "mpeg": "limited",
        "studio": "limited",
        "video": "limited",
        "limited": "limited",
        "pc": "full",
        "jpeg": "full",
        "data": "full",
        "full": "full",
    }
    return mapping.get(v, v)


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


def build_fastpass_hdr10_params(payload: Dict[str, Any], *, encoder: str) -> List[str]:
    normalized = normalize_encoder(encoder)
    if normalized == "svt-av1":
        raw_params = payload.get("video_params") or []
        return [str(item).strip() for item in raw_params if str(item).strip()]
    return build_x265_hdr10_params(payload)


def build_x265_hdr10_params(payload: Dict[str, Any]) -> List[str]:
    static = payload.get("static") or {}
    if not isinstance(static, dict):
        static = {}

    params: List[str] = []
    if bool(payload.get("has_hdr_signal")):
        params.append("--hdr10")

    value_map = (
        ("color_primaries", "--colorprim", normalize_x265_colorprim),
        ("transfer_characteristics", "--transfer", normalize_x265_transfer),
        ("matrix_coefficients", "--colormatrix", normalize_x265_matrix),
        ("color_range", "--range", normalize_x265_range),
    )
    for source_key, target_flag, normalizer in value_map:
        value = normalizer(static.get(source_key))
        if value:
            params.extend([target_flag, value])

    master_display = convert_mastering_display_to_x265(static.get("mastering_display"))
    if master_display:
        params.extend(["--master-display", master_display])

    content_light = str(static.get("content_light") or "").strip()
    if content_light:
        params.extend(["--max-cll", content_light])

    return params


def convert_mastering_display_to_x265(raw_value: Any) -> Optional[str]:
    text = str(raw_value or "").strip()
    if not text:
        return None
    match = _MASTER_DISPLAY_PATTERN.match(text)
    if not match:
        return None

    def scale_coord(group: str) -> int:
        return int(round(float(match.group(group)) * 50000.0))

    def scale_luminance(group: str) -> int:
        return int(round(float(match.group(group)) * 10000.0))

    return (
        f"G({scale_coord('gx')},{scale_coord('gy')})"
        f"B({scale_coord('bx')},{scale_coord('by')})"
        f"R({scale_coord('rx')},{scale_coord('ry')})"
        f"WP({scale_coord('wpx')},{scale_coord('wpy')})"
        f"L({scale_luminance('max_l')},{scale_luminance('min_l')})"
    )
