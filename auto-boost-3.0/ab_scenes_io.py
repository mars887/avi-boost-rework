"""Scene JSON I/O and normalization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ab_fs import ensure_dir, load_json, save_json
from ab_logging import eprint
from ab_scene_ops import build_crf_adjusted_scenes_obj, build_uniform_scenes_obj


def sanitize_scenes_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure structure:
      {
        "frames": int,
        "scenes": [ {start_frame,end_frame,zone_overrides}, ... ],
        "split_scenes": [same as scenes]
      }

    PSD outputs bare scenes.json; av1an outputs scenes with params. We normalise:
      - Keep only start_frame/end_frame/zone_overrides in scene items
      - Force zone_overrides to None in "base" scenes
      - Duplicate scenes -> split_scenes
    """
    scenes = obj.get("split_scenes") or obj.get("scenes") or []
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Invalid scenes.json: expected non-empty 'scenes' or 'split_scenes' list")

    norm: List[Dict[str, Any]] = []
    last_end: Optional[int] = None
    for i, s in enumerate(scenes):
        if not isinstance(s, dict):
            raise ValueError(f"Invalid scene entry at index {i}: expected object")
        if "start_frame" not in s or "end_frame" not in s:
            raise ValueError(f"Invalid scene entry at index {i}: missing start_frame/end_frame")
        st = int(s["start_frame"])
        en = int(s["end_frame"])
        if en <= st:
            raise ValueError(f"Invalid scene entry at index {i}: end_frame <= start_frame ({st}..{en})")
        if last_end is not None and st != last_end:
            eprint(f"[warn] Non-contiguous scenes at index {i}: prev_end={last_end}, start={st}")
        last_end = en
        norm.append({"start_frame": st, "end_frame": en, "zone_overrides": None})

    frames = obj.get("frames")
    if frames is None:
        frames = last_end if last_end is not None else 0
    frames = int(frames)

    return {"frames": frames, "scenes": norm, "split_scenes": [dict(x) for x in norm]}


def scenes_to_ranges(scenes_obj: Dict[str, Any]) -> List[Tuple[int, int]]:
    scenes = scenes_obj.get("split_scenes") or scenes_obj.get("scenes") or []
    return [(int(s["start_frame"]), int(s["end_frame"])) for s in scenes]


def write_base_scenes_from_av1an(av1an_temp: Path, base_scenes_path: Path) -> None:
    """Normalize av1an scenes.json into base scenes.json."""
    av1an_scenes_path = av1an_temp / "scenes.json"
    if not av1an_scenes_path.exists():
        raise FileNotFoundError(
            f"av1an scenes.json not found: {av1an_scenes_path} "
            "(run stage 2 with --sdm av1an; use --keep if temp is cleaned)."
        )
    raw = load_json(av1an_scenes_path)
    norm = sanitize_scenes_json(raw)
    ensure_dir(base_scenes_path.parent)
    save_json(base_scenes_path, norm)
    print(f"[ok] av1an scenes written: {base_scenes_path}")


def write_uniform_scenes(
    *,
    base_scenes_path: Path,
    out_scenes_path: Path,
    base_crf: float,
    final_preset: int,
    video_params: str,
    final_override: str,
) -> None:
    """Write scenes.json with uniform overrides."""
    base = load_json(base_scenes_path)
    base_norm = sanitize_scenes_json(base)
    out_obj = build_uniform_scenes_obj(
        base_norm=base_norm,
        base_crf=base_crf,
        final_preset=final_preset,
        video_params=video_params,
        final_override=final_override,
    )
    ensure_dir(out_scenes_path.parent)
    save_json(out_scenes_path, out_obj)
    print(f"[ok] uniform scenes.json written: {out_scenes_path}")


def apply_crf_adjustments_to_scenes(
    *,
    base_scenes_path: Path,
    out_scenes_path: Path,
    scene_ranges: List[Tuple[int, int]],
    per_chunk_5: List[float],
    avg_total: float,
    base_crf: float,
    pos_dev_multiplier: float,
    neg_dev_multiplier: float,
    deviation: float,
    max_positive_dev: Optional[float],
    max_negative_dev: Optional[float],
    final_preset: int,
    video_params: str,
    final_override: str,
) -> None:
    """Compute per-scene CRF adjustments and write scenes.json."""
    base = load_json(base_scenes_path)
    base_norm = sanitize_scenes_json(base)
    out_obj = build_crf_adjusted_scenes_obj(
        base_norm=base_norm,
        scene_ranges=scene_ranges,
        per_chunk_5=per_chunk_5,
        avg_total=avg_total,
        base_crf=base_crf,
        pos_dev_multiplier=pos_dev_multiplier,
        neg_dev_multiplier=neg_dev_multiplier,
        deviation=deviation,
        max_positive_dev=max_positive_dev,
        max_negative_dev=max_negative_dev,
        final_preset=final_preset,
        video_params=video_params,
        final_override=final_override,
    )
    ensure_dir(out_scenes_path.parent)
    save_json(out_scenes_path, out_obj)
    print(f"[ok] base scenes.json written: {out_scenes_path}")
