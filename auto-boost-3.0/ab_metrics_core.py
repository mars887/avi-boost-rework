"""VapourSynth loading and low-level metric helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import tqdm


def has_vapoursynth() -> bool:
    try:
        import vapoursynth  # noqa: F401
        return True
    except Exception:
        return False


def _unwrap_vs_output(out: object, vs_mod):
    """Best-effort unwrap of VapourSynth output containers to a VideoNode."""
    try:
        if isinstance(out, vs_mod.VideoNode):
            return out
    except Exception:
        pass

    for attr in ("clip", "node", "video", "output"):
        try:
            v = getattr(out, attr, None)
            if v is not None and isinstance(v, vs_mod.VideoNode):
                return v
        except Exception:
            continue

    if isinstance(out, (tuple, list)):
        for v in out:
            try:
                if isinstance(v, vs_mod.VideoNode):
                    return v
            except Exception:
                continue

    return None


def _load_vs_clip_from_vpy(vpy_path: Path, *, vpy_src: Path):
    """Execute a .vpy in-process and return its first output clip."""
    if not has_vapoursynth():
        raise RuntimeError("VapourSynth is not available (python module not found).")

    import vapoursynth as vs  # type: ignore
    import runpy

    try:
        if hasattr(vs, "clear_outputs"):
            vs.clear_outputs()
        elif hasattr(vs.core, "clear_outputs"):
            vs.core.clear_outputs()  # type: ignore[attr-defined]
    except Exception:
        pass

    old_cwd = os.getcwd()
    old_sys_path = list(sys.path)

    try:
        os.chdir(str(vpy_path.parent))
        if str(vpy_path.parent) not in sys.path:
            sys.path.insert(0, str(vpy_path.parent))

        init_globals = {
            "__file__": str(vpy_path),
            "__name__": "__vapoursynth__",
            "src": str(vpy_src),
        }
        ns = runpy.run_path(str(vpy_path), init_globals=init_globals)
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path

    getters = []
    if hasattr(vs, "get_output"):
        getters.append(getattr(vs, "get_output"))
    if hasattr(vs.core, "get_output"):
        getters.append(getattr(vs.core, "get_output"))

    for get_out in getters:
        try:
            out = get_out(0)
        except Exception:
            continue
        node = _unwrap_vs_output(out, vs)
        if node is not None:
            return node

    for k in ("clip", "out", "output", "src_clip"):
        v = ns.get(k)
        try:
            if isinstance(v, vs.VideoNode):
                return v
        except Exception:
            continue

    for v in ns.values():
        try:
            if isinstance(v, vs.VideoNode):
                return v
        except Exception:
            continue

    raise RuntimeError(
        f"No VapourSynth output found in vpy: {vpy_path}. "
        "Ensure it either calls clip.set_output() (output index 0) or leaves a VideoNode variable (e.g. clip=...)."
    )


def load_vs_clip(path: Path, vs_source: str, *, vpy_src: Optional[Path] = None):
    """Load a VapourSynth clip from a file or .vpy script."""
    import vapoursynth as vs  # type: ignore

    if path.suffix.lower() == ".vpy":
        if vpy_src is None:
            raise ValueError(f"Loading a .vpy requires vpy_src (original input path), but none was provided: {path}")
        return _load_vs_clip_from_vpy(path, vpy_src=vpy_src)

    core = vs.core
    req = (vs_source or "ffms2").lower().strip()
    providers = []

    if req == "auto":
        if hasattr(core, "ffms2"):
            providers.append(core.ffms2.Source)
        if hasattr(core, "bs"):
            providers.append(core.bs.VideoSource)
        if hasattr(core, "lsmas"):
            providers.append(core.lsmas.LWLibavSource)
    elif req == "ffms2":
        if hasattr(core, "ffms2"):
            providers.append(core.ffms2.Source)
    elif req in ("bs", "bestsource"):
        if hasattr(core, "bs"):
            providers.append(core.bs.VideoSource)
    elif req in ("lsmas", "lwlibavsource"):
        if hasattr(core, "lsmas"):
            providers.append(core.lsmas.LWLibavSource)

    if not providers:
        raise RuntimeError(f"VapourSynth source '{vs_source}' is not available (plugin missing).")

    last_err = None
    for loader in providers:
        try:
            return loader(str(path))
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load {path}. Last error: {last_err}")


def compute_luma_samples(src_file: Path, skip: int, vs_source: str, *, vpy_src: Optional[Path] = None):
    """Compute per-frame luma averages with optional sampling."""
    if not has_vapoursynth():
        raise RuntimeError("VapourSynth is not available (python module not found).")
    import vapoursynth as vs  # type: ignore

    core = vs.core
    clip = load_vs_clip(src_file, vs_source, vpy_src=vpy_src)
    if clip.format is None:
        raise RuntimeError("Unknown clip format for luma calculation.")

    if clip.format.color_family != vs.YUV or clip.format.sample_type != vs.INTEGER or clip.format.bits_per_sample != 8:
        clip = core.resize.Bicubic(clip, format=vs.YUV444P8)

    clip = clip.std.ShufflePlanes(planes=0, colorfamily=vs.GRAY)
    if skip and skip > 1:
        clip = clip.std.SelectEvery(cycle=skip, offsets=0)

    stats = clip.std.PlaneStats()
    samples = []
    with tqdm.tqdm(total=stats.num_frames, desc="Luma samples") as pbar:
        for frame in stats.frames():
            avg = frame.props.get("PlaneStatsAverage")
            if avg is None:
                raise RuntimeError("PlaneStatsAverage missing from VapourSynth frame props.")
            samples.append(float(avg))
            pbar.update(1)

    if not samples:
        raise ValueError("No luma samples computed.")
    return samples
