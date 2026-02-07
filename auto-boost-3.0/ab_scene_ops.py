"""Scene computations and zone override builders."""

from __future__ import annotations

import math
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ab_params import find_last_option, is_param_key, strip_params_tokens
from ab_ssimu2 import calc_stats, parse_ssimu2_log, slice_samples_for_scene


def apply_avg_func(avg_total: float, avg_func: str) -> float:
    """Apply avg-func syntax to a metric average."""
    s = str(avg_func or "").strip()
    if not s:
        return avg_total
    if s[0] in "+-" and len(s) > 1:
        try:
            delta = float(s)
        except ValueError as exc:
            raise ValueError(f"Invalid --avg-func value: {avg_func!r}") from exc
        return avg_total + delta
    if s[0] == "!":
        if len(s) == 1:
            raise ValueError(f"Invalid --avg-func value: {avg_func!r}")
        try:
            target = float(s[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid --avg-func value: {avg_func!r}") from exc
        return target
    if "%" in s:
        left, right = s.split("%", 1)
        if not left or not right:
            raise ValueError(f"Invalid --avg-func value: {avg_func!r}")
        try:
            target = float(left)
            pct = float(right)
        except ValueError as exc:
            raise ValueError(f"Invalid --avg-func value: {avg_func!r}") from exc
        return avg_total + (target - avg_total) * (pct / 100.0)
    raise ValueError(f"Invalid --avg-func value: {avg_func!r}")


def compute_chunk_5p_single_metric(
    *,
    scene_ranges: List[Tuple[int, int]],
    ssimu2_path: Path,
) -> Tuple[int, List[float], float]:
    """
    Return:
      skip_used_for_ssimu2,
      per_chunk_percentile_5_list,
      global_average
    """
    skip, ssimu2_scores = parse_ssimu2_log(ssimu2_path)
    per_chunk_5: List[float] = []

    for (st, en) in scene_ranges:
        s_chunk = slice_samples_for_scene(ssimu2_scores, st, en, skip)
        _, p5, _ = calc_stats(s_chunk)
        per_chunk_5.append(p5)

    avg_total, _, _ = calc_stats(ssimu2_scores)
    return skip, per_chunk_5, avg_total


def build_zone_overrides(
    *,
    crf: float,
    preset: int,
    video_params_str: str,
    final_override: str,
) -> Dict[str, Any]:
    """Build zone_overrides payload for a scene."""
    def apply_override(base_tokens: List[str], override_tokens: List[str]) -> List[str]:
        i = 0
        while i < len(override_tokens):
            tok = override_tokens[i]
            if not is_param_key(tok):
                i += 1
                continue

            key = tok
            has_val = (i + 1 < len(override_tokens)) and (not is_param_key(override_tokens[i + 1]))
            val = override_tokens[i + 1] if has_val else None

            loc = find_last_option(base_tokens, key)
            if loc is None:
                base_tokens.append(key)
                if val is not None:
                    base_tokens.append(val)
            else:
                k_idx, base_has_val = loc
                if val is None:
                    if base_has_val:
                        del base_tokens[k_idx + 1]
                else:
                    if base_has_val:
                        base_tokens[k_idx + 1] = val
                    else:
                        base_tokens.insert(k_idx + 1, val)

            i += 2 if has_val else 1

        return base_tokens

    tokens = shlex.split(video_params_str) if video_params_str else []
    tokens = strip_params_tokens(tokens, keys=["--crf", "--preset"])

    video_params: List[str] = ["--crf", f"{crf:.2f}", "--preset", str(int(preset))]
    video_params.extend(tokens)

    override_tokens = shlex.split(final_override) if final_override else []
    if override_tokens:
        video_params = apply_override(video_params, override_tokens)

    return {
        "encoder": "svt_av1",
        "passes": 1,
        "video_params": video_params,
        "photon_noise": None,
        "photon_noise_height": None,
        "photon_noise_width": None,
        "chroma_noise": False,
        "extra_splits_len": 9999,
        "min_scene_len": 9,
    }


def build_uniform_scenes_obj(
    *,
    base_norm: Dict[str, Any],
    base_crf: float,
    final_preset: int,
    video_params: str,
    final_override: str,
) -> Dict[str, Any]:
    """Build a uniform zone_overrides scene list from a normalized base object."""
    frames_count = int(base_norm["frames"])
    scenes = base_norm.get("scenes") or base_norm.get("split_scenes") or []

    updated: List[Dict[str, Any]] = []
    for s in scenes:
        st = int(s["start_frame"])
        en = int(s["end_frame"])
        updated.append({
            "start_frame": st,
            "end_frame": en,
            "zone_overrides": build_zone_overrides(
                crf=float(base_crf),
                preset=int(final_preset),
                video_params_str=str(video_params),
                final_override=str(final_override),
            ),
        })

    return {"frames": frames_count, "scenes": updated, "split_scenes": [dict(x) for x in updated]}


def build_crf_adjusted_scenes_obj(
    *,
    base_norm: Dict[str, Any],
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
) -> Dict[str, Any]:
    """Build per-scene CRF-adjusted overrides from SSIMU2 percentiles."""
    frames_count = int(base_norm["frames"])

    base_dev = float(deviation)
    max_pos = max_positive_dev if max_positive_dev is not None else base_dev
    max_neg = max_negative_dev if max_negative_dev is not None else base_dev

    pos_multiplier = float(pos_dev_multiplier) * 20.0
    neg_multiplier = float(neg_dev_multiplier) * 20.0

    updated: List[Dict[str, Any]] = []
    for i, (st, en) in enumerate(scene_ranges):
        delta = 1.0 - (per_chunk_5[i] / (avg_total + 0.001))
        multiplier = pos_multiplier if delta >= 0 else neg_multiplier
        adj = math.ceil(delta * multiplier * 4.0) / 4.0
        new_crf = float(base_crf) - float(adj)

        if adj < 0:
            if max_pos == 0:
                new_crf = float(base_crf)
            elif abs(adj) > float(max_pos):
                new_crf = float(base_crf) + float(max_pos)
        else:
            if max_neg == 0:
                new_crf = float(base_crf)
            elif abs(adj) > float(max_neg):
                new_crf = float(base_crf) - float(max_neg)

        print(
            f"Enc:  [{st}:{en}] | chunk_p5={per_chunk_5[i]:.4f} | "
            f"avg={avg_total:.4f} | adj={adj:+.2f} | crf={new_crf:.2f}"
        )

        updated.append({
            "start_frame": st,
            "end_frame": en,
            "zone_overrides": build_zone_overrides(
                crf=new_crf,
                preset=final_preset,
                video_params_str=video_params,
                final_override=final_override,
            ),
        })

    return {"frames": frames_count, "scenes": updated, "split_scenes": [dict(x) for x in updated]}
