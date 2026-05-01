"""SSIMULACRA2 computation and parsing helpers."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import tqdm

from ab_metrics_core import has_vapoursynth, load_vs_clip
from ab_runner_events import emit_runner_child_event


def _format_eta(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "unknown"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"

def calculate_ssimu2(
    *,
    src_file: Path,
    enc_file: Path,
    out_path: Path,
    frames_count: int,
    skip: int,
    backend: str,
    vs_source: str,
    vpy_src: Optional[Path] = None,
    vpy_args: Optional[Dict[str, str]] = None,
) -> int:
    """Compute SSIMULACRA2 and write a log file.

    Important: This script intentionally uses ONLY VapourSynth plugins for SSIMULACRA2.
    Any CLI-based backends are intentionally not used here.

    Output format:
        skip: N
        1: score
        2: score
        ...
    """

    # Accept legacy names but force VapourSynth plugin backends.
    selected = (backend or "auto").strip().lower()

    if not has_vapoursynth():
        raise RuntimeError("VapourSynth is not available (python module not found). Install VapourSynth and the required plugins.")

    import vapoursynth as vs  # type: ignore

    core = vs.core

    have_vship_plugin = hasattr(core, "vship") and hasattr(core.vship, "SSIMULACRA2")
    have_vszip_plugin = hasattr(core, "vszip") and hasattr(core.vszip, "SSIMULACRA2")

    if selected == "auto":
        if have_vship_plugin:
            selected = "vship"
        elif have_vszip_plugin:
            selected = "vszip"
        else:
            raise RuntimeError("No SSIMULACRA2 VapourSynth plugin found. Install vship or vszip.")

    if selected == "vship" and not have_vship_plugin:
        raise RuntimeError("Vship VapourSynth plugin is not loaded (core.vship.SSIMULACRA2 not available).")
    if selected == "vszip" and not have_vszip_plugin:
        raise RuntimeError("vszip VapourSynth plugin is not loaded (core.vszip.SSIMULACRA2 not available).")

    # Always build the clips first (fixes UnboundLocalError in vship branch).
    source_clip = load_vs_clip(src_file, vs_source, vpy_src=vpy_src, vpy_args=vpy_args)
    encoded_clip = load_vs_clip(enc_file, vs_source, vpy_src=vpy_src)

    # Match geometry (SSIMU2 needs same dimensions).
    if source_clip.width != encoded_clip.width or source_clip.height != encoded_clip.height:
        source_clip = source_clip.resize.Lanczos(width=encoded_clip.width, height=encoded_clip.height)

    # Subsample frames if requested.
    if skip and skip > 1:
        source_clip = source_clip.std.SelectEvery(cycle=skip, offsets=0)
        encoded_clip = encoded_clip.std.SelectEvery(cycle=skip, offsets=0)

    # Run metric plugin.
    if selected == "vship":
        # numStream trades VRAM for speed (Vship README suggests 4 as a good default).
        result = core.vship.SSIMULACRA2(source_clip, encoded_clip, numStream=8)
        prop_keys = ("_SSIMULACRA2", "SSIMULACRA2")
    else:
        result = core.vszip.SSIMULACRA2(source_clip, encoded_clip)
        prop_keys = ("SSIMULACRA2", "_SSIMULACRA2")

    scores: List[float] = []
    sample_step = max(1, int(skip or 1))
    total_samples = max(1, int(result.num_frames or 0))
    started_at = time.monotonic()
    last_progress_at = 0.0

    def emit_progress(*, force: bool = False) -> None:
        nonlocal last_progress_at
        if not scores:
            return
        now = time.monotonic()
        if not force and now - last_progress_at < 1.0:
            return
        last_progress_at = now
        samples_done = len(scores)
        elapsed = max(0.001, now - started_at)
        sample_rate = samples_done / elapsed
        frame_rate = sample_rate * sample_step
        remaining_samples = max(0, total_samples - samples_done)
        eta_seconds = remaining_samples / sample_rate if sample_rate > 0 else float("inf")
        emit_runner_child_event(
            "SSIMU2 Metrics",
            "progress",
            source=src_file,
            workdir=out_path.parent.parent if out_path.parent.name == "fastpass" else out_path.parent,
            progress=max(0.0, min(100.0, samples_done * 100.0 / total_samples)),
            details={
                "fps": frame_rate,
                "eta": _format_eta(eta_seconds),
                "ssimu2": sum(scores) / samples_done,
                "samples_done": samples_done,
                "samples_total": total_samples,
                "step": sample_step,
            },
        )

    with tqdm.tqdm(total=result.num_frames, desc=f"SSIMULACRA2 ({selected})") as pbar:
        for frame in result.frames():
            v = None
            for k in prop_keys:
                if k in frame.props:
                    try:
                        v = float(frame.props[k])
                    except Exception:
                        v = None
                    break
            if v is None:
                raise RuntimeError(f"{selected} produced frames without SSIMULACRA2 props ({', '.join(prop_keys)}).")

            scores.append(max(v, 0.0))
            pbar.update(1)
            emit_progress()

    emit_progress(force=True)

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"skip: {skip}\n")
        for idx, sc in enumerate(scores, start=1):
            f.write(f"{idx}: {sc}\n")

    print(f"[ok] SSIMU2 via {selected} -> {out_path} (samples={len(scores)}, skip={skip})")
    return skip

def parse_ssimu2_log(path: Path) -> Tuple[int, List[float]]:
    """Parse an SSIMU2 log file and return skip and scores."""
    scores: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
        m = re.search(r"skip:\s*([0-9]+)", first)
        if not m:
            raise ValueError("Skip value not detected in SSIMU2 file.")
        skip = int(m.group(1))
        for line in f:
            m2 = re.search(r"([0-9]+):\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", line.strip())
            if m2:
                scores.append(max(float(m2.group(2)), 0.0))
    if not scores:
        raise ValueError("No SSIMU2 scores parsed.")
    return skip, scores

def calc_stats(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        raise ValueError("Empty metric list.")
    filtered = [v if v >= 0 else 0.0 for v in values]
    sorted_vals = sorted(filtered)
    avg = sum(filtered) / len(filtered)
    p5 = sorted_vals[max(0, len(filtered) // 20)]
    p95 = sorted_vals[min(len(filtered) - 1, int(len(filtered) * 0.95))]
    return avg, p5, p95

def slice_samples_for_scene(scores: List[float], st: int, en: int, skip: int) -> List[float]:
    """
    scores[k] corresponds to frame index k*skip (global sampling).
    Select the subset with st <= k*skip < en.
    """
    if not scores:
        return []
    if en <= st:
        return [scores[0]]
    k0 = (st + skip - 1) // skip
    k1 = (en - 1) // skip  # inclusive
    k0 = max(0, min(k0, len(scores) - 1))
    k1 = max(0, min(k1, len(scores) - 1))
    if k1 < k0:
        return [scores[k0]]
    out = scores[k0: k1 + 1]
    return out if out else [scores[min(k0, len(scores) - 1)]]
