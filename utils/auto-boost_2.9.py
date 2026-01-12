#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto-boost (2.7) — PSD + scenes.json workflow + resume markers + VapourSynth Vship/FFMS2 defaults

Pipeline (default --stage 0, with automatic resume):
  1) Scene detection (PSD) if --sdm psd -> base scenes.json (zone_overrides = null)
  2) Fast-pass: run av1an using PSD scenes (--scenes) or av1an internal detection (--sdm av1an)
  3) Metrics: compute SSIMULACRA2 (prefer VapourSynth vship plugin; fallback to vszip) and/or XPSNR
  4) Fill the same base scenes.json with per-scene zone_overrides (CRF adjustments)
     and write base scenes.json ready for av1an --scenes
  5) Apply rule script (optional) to adjust per-scene video_params and write final scenes.json

Notes:
  - zones.txt output has been removed entirely (scenes.json only).
  - --sdm av1an normalizes av1an temp scenes.json into PSD-like base scenes.
  - Resume is implemented via marker files in <project>/.state.
  - Default SSIMU2 backend is "auto": prefer VapourSynth vship plugin, then fall back to VapourSynth vszip.
  - For VapourSynth fallback, FFMS2 is preferred for decoding (then BestSource, then LSMASH).
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple
import tqdm

IS_WINDOWS = (os.name == "nt")
NULL_DEVICE = "NUL" if IS_WINDOWS else "/dev/null"

_NP = None
_CV2 = None


def _import_numpy():
    global _NP
    if _NP is None:
        try:
            import numpy as np  # type: ignore
        except Exception as exc:
            raise RuntimeError("numpy is required for grad_mad metrics.") from exc
        _NP = np
    return _NP


def _import_cv2():
    global _CV2
    if _CV2 is None:
        try:
            import cv2  # type: ignore
        except Exception as exc:
            raise RuntimeError("OpenCV (cv2) is required for farneback metrics.") from exc
        _CV2 = cv2
    return _CV2


# ---------------------------
# Utilities
# ---------------------------

def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

PROGRESS_RE = re.compile(
    r"^(progress|processed)\s*[:\-]?\s*\d+%\s*$|^frame\s+\d+\s*/.*fps$|^creating lwi index file\s+\d+%\s*$",
    re.IGNORECASE,
)


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def is_progress_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if PROGRESS_RE.match(s):
        return True
    if s.endswith("%") and any(ch.isdigit() for ch in s) and len(s) <= 60:
        return True
    if s.lower().startswith("frame ") and "fps" in s.lower():
        return True
    return False


class TeeStream:
    def __init__(self, stream: TextIO, log_file: TextIO) -> None:
        self._stream = stream
        self._log: Optional[TextIO] = log_file

    def write(self, s: str) -> int:
        try:
            self._stream.write(s)
            self._stream.flush()
        except Exception:
            pass
        if self._log is not None:
            try:
                self._log.write(s)
                self._log.flush()
            except Exception:
                self._log = None
        return len(s)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception:
            pass
        if self._log is not None:
            try:
                self._log.flush()
            except Exception:
                self._log = None

    def close_log(self) -> None:
        if self._log is None:
            return
        try:
            self._log.flush()
        except Exception:
            pass
        try:
            self._log.close()
        except Exception:
            pass
        self._log = None

    def isatty(self) -> bool:
        return bool(getattr(self._stream, "isatty", lambda: False)())

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8")


def setup_logging(log_path: str, workdir: Optional[Path] = None) -> None:
    if not log_path:
        return
    p = Path(log_path)
    if not p.is_absolute() and workdir is not None:
        p = workdir / p
    p.parent.mkdir(parents=True, exist_ok=True)
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    log_fh = p.open("w", encoding=enc, errors="replace")
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    tee_out = TeeStream(orig_stdout, log_fh)
    tee_err = TeeStream(orig_stderr, log_fh)
    sys.stdout = tee_out
    sys.stderr = tee_err

    def _cleanup() -> None:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        tee_out.close_log()
        tee_err.close_log()

    atexit.register(_cleanup)


def run_cmd(cmd: Sequence[str], *, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    cmd_str = " ".join(shlex.quote(str(x)) for x in cmd)
    print(f"[cmd] {cmd_str}")
    p = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    buf = ""
    progress_width = 0
    had_progress = False
    last_was_cr = False
    if p.stdout is not None:
        while True:
            ch = p.stdout.read(1)
            if ch == "":
                break
            if last_was_cr and ch == "\n":
                last_was_cr = False
                continue
            if ch in ("\n", "\r"):
                line = buf
                buf = ""
                clean = strip_ansi(line)
                if ch == "\r" or is_progress_line(clean):
                    s = clean.strip()
                    if len(s) < progress_width:
                        s = s + (" " * (progress_width - len(s)))
                    progress_width = max(progress_width, len(s))
                    sys.stdout.write(s + "\r")
                    sys.stdout.flush()
                    had_progress = True
                    last_was_cr = (ch == "\r")
                    continue

                if had_progress:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    had_progress = False
                    progress_width = 0
                last_was_cr = False

                sys.stdout.write(clean)
                sys.stdout.write("\n")
                sys.stdout.flush()
                continue

            buf += ch
            last_was_cr = False
            if len(buf) >= 2048:
                sys.stdout.write(strip_ansi(buf))
                sys.stdout.flush()
                buf = ""
    if buf:
        clean = strip_ansi(buf)
        if is_progress_line(clean):
            s = clean.strip()
            if len(s) < progress_width:
                s = s + (" " * (progress_width - len(s)))
            sys.stdout.write(s + "\r")
            sys.stdout.flush()
            had_progress = True
        else:
            if had_progress:
                sys.stdout.write("\n")
                sys.stdout.flush()
                had_progress = False
                progress_width = 0
            sys.stdout.write(clean)
            sys.stdout.flush()
    rc = p.wait()
    if had_progress:
        sys.stdout.write("\n")
        sys.stdout.flush()
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, list(cmd))
    return subprocess.CompletedProcess(list(cmd), rc)

def which_or_none(exe: str) -> Optional[str]:
    return shutil.which(exe)


def ensure_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def touch(path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("ok\n")


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


# ---------------------------
# Resume markers
# ---------------------------

def state_dir(project_dir: Path) -> Path:
    d = project_dir / ".state"
    ensure_dir(d)
    return d


def marker_paths(project_dir: Path) -> Dict[str, Path]:
    sd = state_dir(project_dir)
    return {
        "psd": sd / "PSD_FINISHED",
        "fastpass": sd / "FASTPASS_COMPLETED",
        "ssimu2": sd / "SSIMU2_COMPLETED",
        "xpsnr": sd / "XPSNR_COMPLETED",
        "final": sd / "FINAL_SCENES_COMPLETED",
        "rules": sd / "RULES_APPLIED",
    }


def is_valid_base_scenes(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 10:
        return False
    try:
        sanitize_scenes_json(load_json(path))
        return True
    except Exception:
        return False


def is_valid_ssimu2_log(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 10:
        return False
    try:
        _skip, scores = parse_ssimu2_log(path)
        return len(scores) > 0
    except Exception:
        return False


def is_valid_xpsnr_log(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 10:
        return False
    try:
        vals = parse_xpsnr_log(path)
        return len(vals) > 0
    except Exception:
        return False


def is_valid_final_scenes(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 10:
        return False
    try:
        obj = load_json(path)
        scenes = obj.get("scenes") or obj.get("split_scenes") or []
        if not scenes:
            return False
        # ensure at least one zone_overrides is populated
        for s in scenes:
            if isinstance(s, dict) and s.get("zone_overrides"):
                return True
        return False
    except Exception:
        return False


# ---------------------------
# Stage 1: PSD (scene detection)
# ---------------------------

def run_psd(psd_script: Path, psd_python: Optional[Path], input_file: Path, base_scenes_path: Path, extra_args: str) -> None:
    ensure_exists(psd_script, "PSD script")
    ensure_dir(base_scenes_path.parent)

    py = str(psd_python) if psd_python else sys.executable
    cmd = [py, str(psd_script), "-i", str(input_file), "-o", str(base_scenes_path)]
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    run_cmd(cmd, check=True)

    # Normalise to the exact "base scenes" contract we want downstream.
    raw = load_json(base_scenes_path)
    norm = sanitize_scenes_json(raw)
    save_json(base_scenes_path, norm)
    print(f"[ok] PSD scenes written: {base_scenes_path}")


# ---------------------------
# Stage 2: fast-pass encode (av1an)
# ---------------------------

def build_av1an_filter_arg(ffmpeg_arg: str) -> List[str]:
    """
    av1an expects `-f <ffmpeg options>`.
    Support both:
      - If arg starts with '-', treat as raw ffmpeg args
      - Else treat as filtergraph and wrap as: -vf "<filtergraph>"
    """
    if not ffmpeg_arg:
        return []
    s = ffmpeg_arg.strip()
    if not s:
        return []
    if s.startswith("-"):
        return ["-f", s]
    return ["-f", f'-vf "{s}"']


def run_fastpass_av1an(
    *,
    input_file: Path,
    output_file: Path,
    scenes_path: Optional[Path],
    av1an_temp: Path,
    workers: int,
    lp: int,
    fast_preset: int,
    fast_crf: float,
    video_params: str,
    ffmpeg_arg: str,
    verbose: bool,
    keep: bool,
) -> None:
    if scenes_path is not None:
        ensure_exists(scenes_path, "Base scenes.json")
    ensure_dir(av1an_temp)

    cmd: List[str] = [
        "av1an",
        "-i", str(input_file),
        "--temp", str(av1an_temp),
        "-y",
    ]
    if verbose:
        cmd.append("--verbose")
    if keep:
        cmd.append("--keep")

    # Provide scenes file (skip scene detection)
    if scenes_path is not None:
        cmd.extend(["--scenes", str(scenes_path)])

    # Muxing defaults from old scripts
    cmd.extend(["-m", "lsmash", "-c", "mkvmerge"])
    cmd.extend(["--chunk-order", "random"])

    # Encoder & encode settings (fast pass)
    cmd.extend(["-e", "svt-av1", "--force"])
    cmd.extend(["-a","-an -sn"])

    enc_params = f'--preset {int(fast_preset)} --crf {float(fast_crf):.2f} --lp {int(lp)}'
    if video_params:
        enc_params += f" {video_params.strip()}"
    cmd.extend(["-v", enc_params])

    cmd.extend(build_av1an_filter_arg(ffmpeg_arg))
    cmd.extend(["-w", str(int(workers))])
    cmd.extend(["-o", str(output_file)])

    run_cmd(cmd, check=True)
    print(f"[ok] fast-pass output: {output_file}")


# ---------------------------
# Stage 3: metrics
# ---------------------------

def has_vapoursynth() -> bool:
    try:
        import vapoursynth  # noqa: F401
        return True
    except Exception:
        return False


def has_cv2() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


def load_vs_clip(path: Path, vs_source: str):
    import vapoursynth as vs  # type: ignore

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


def frame_to_ndarray(frame) -> "Any":
    if hasattr(frame, "get_read_array"):
        np = _import_numpy()
        return np.array(frame.get_read_array(0), copy=True)
    try:
        np = _import_numpy()
        return np.array(np.asarray(frame[0]), copy=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to access VapourSynth frame data: {exc}") from exc


def conv3x3(img, kernel):
    np = _import_numpy()
    padded = np.pad(img, 1, mode="edge")
    return (
        kernel[0, 0] * padded[:-2, :-2] +
        kernel[0, 1] * padded[:-2, 1:-1] +
        kernel[0, 2] * padded[:-2, 2:] +
        kernel[1, 0] * padded[1:-1, :-2] +
        kernel[1, 1] * padded[1:-1, 1:-1] +
        kernel[1, 2] * padded[1:-1, 2:] +
        kernel[2, 0] * padded[2:, :-2] +
        kernel[2, 1] * padded[2:, 1:-1] +
        kernel[2, 2] * padded[2:, 2:]
    )


def robust_mad_sigma(values) -> float:
    np = _import_numpy()
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad < 1e-12:
        return 0.0
    return float(mad / 0.6745)


_SOBEL_X = (
    (1.0, 0.0, -1.0),
    (2.0, 0.0, -2.0),
    (1.0, 0.0, -1.0),
)
_SOBEL_Y = (
    (1.0, 2.0, 1.0),
    (0.0, 0.0, 0.0),
    (-1.0, -2.0, -1.0),
)


def grad_mad_from_gray(gray) -> float:
    np = _import_numpy()
    img = np.asarray(gray, dtype=np.float32)
    kx = np.asarray(_SOBEL_X, dtype=np.float32)
    ky = np.asarray(_SOBEL_Y, dtype=np.float32)
    gx = conv3x3(img, kx)
    gy = conv3x3(img, ky)
    mag = np.hypot(gx, gy)
    return robust_mad_sigma(mag)


def sample_indices_for_scene(
    st: int,
    en: int,
    sample_count: int,
    max_gap: int,
) -> List[int]:
    length = max(0, int(en) - int(st))
    if length <= 0:
        return []
    count = max(1, int(sample_count))
    step = max(1, (length + count - 1) // count)
    if max_gap and max_gap > 0:
        step = min(step, int(max_gap))
    return list(range(int(st), int(en), step))


_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
_METRIC_CALL_RE = re.compile(r"metric\(\s*['\"]([^'\"]+)['\"]\s*\)")
_LUMA_P_RE = re.compile(r"^luma_p(\d+)$")


def calculate_ssimu2(
    *,
    src_file: Path,
    enc_file: Path,
    out_path: Path,
    frames_count: int,
    skip: int,
    backend: str,
    vs_source: str,
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
    if selected in {"ffvship", "ffvship.exe", "ffvship-cli"}:
        selected = "vship"

    if selected in {"turbo", "cli"}:
        raise RuntimeError(
            "SSIMU2 backend 'turbo' is disabled in this build. "
            "Use --ssimu2-backend vship (preferred) or vszip."
        )

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
    source_clip = load_vs_clip(src_file, vs_source)
    encoded_clip = load_vs_clip(enc_file, vs_source)

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

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"skip: {skip}\n")
        for idx, sc in enumerate(scores, start=1):
            f.write(f"{idx}: {sc}\n")

    print(f"[ok] SSIMU2 via {selected} -> {out_path} (samples={len(scores)}, skip={skip})")
    return skip


def calculate_xpsnr(src_file: Path, enc_file: Path, out_path: Path, src_filtergraph: str = "") -> None:
    """
    Use FFmpeg xpsnr filter, writing per-frame stats to out_path.

    If `src_filtergraph` is provided (e.g. "crop=...,..."), it is applied to the *source* stream
    before comparing against `enc_file` (which already contains those filters from the fast-pass).

    On Windows we run ffmpeg with cwd=out_path.parent and use a relative stats_file name to avoid
    colon parsing issues in filter options.
    """
    ensure_dir(out_path.parent)

    # Where stats_file should be written.
    if IS_WINDOWS:
        cwd = out_path.parent
        stats_name = out_path.name
    else:
        cwd = None
        stats_name = str(out_path)

    src_fg = src_filtergraph.strip()
    if src_fg and src_fg.startswith("-"):
        # Not a pure filtergraph; cannot safely embed into -filter_complex.
        src_fg = ""

    if src_fg:
        # Apply filtergraph to source only, then align to encoded via scale2ref, then compute XPSNR.
        fg = (
            f"[0:v]{src_fg}[srcf];"
            f"[srcf][1:v]scale2ref=flags=bicubic[src][enc];"
            f"[src][enc]xpsnr=stats_file={stats_name}:shortest=1"
        )
    else:
        # No source filtergraph known; just align dimensions and compute XPSNR.
        fg = (
            f"[0:v][1:v]scale2ref=flags=bicubic[src][enc];"
            f"[src][enc]xpsnr=stats_file={stats_name}:shortest=1"
        )

    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin",
        "-i", str(src_file),
        "-i", str(enc_file),
        "-filter_complex", fg,
        "-f", "null", NULL_DEVICE,
    ]
    run_cmd(cmd, cwd=cwd, check=True)

    print(f"[ok] XPSNR -> {out_path}")



def parse_ssimu2_log(path: Path) -> Tuple[int, List[float]]:
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


def parse_xpsnr_log(path: Path) -> List[float]:
    """
    Parse FFmpeg xpsnr stats_file, returning weighted per-frame values:
      W = (4*Y + U + V)/6
    """
    values: List[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(
                r"XPSNR [yY]: ([0-9]+\.[0-9]+|inf).*?: ([0-9]+\.[0-9]+|inf)\s+XPSNR [vV]: ([0-9]+\.[0-9]+|inf)",
                line
            )
            if not m:
                continue
            y = 100.0 if m.group(1) == "inf" else float(m.group(1))
            u = 100.0 if m.group(2) == "inf" else float(m.group(2))
            v = 100.0 if m.group(3) == "inf" else float(m.group(3))
            w = (4.0 * y + u + v) / 6.0
            values.append(w)
    if not values:
        raise ValueError("No XPSNR values parsed.")
    return values


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


# ---------------------------
# Rules and metrics helpers
# ---------------------------

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


def extract_metric_calls(rule_text: str) -> List[str]:
    if not rule_text:
        return []
    return [m.strip().lower() for m in _METRIC_CALL_RE.findall(rule_text)]


def parse_frame_rate(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if "/" in s:
            return float(Fraction(s))
        return float(s)
    raise ValueError(f"Unsupported frame_rate value: {value!r}")


def percentile(values: List[float], pct: float) -> float:
    if not values:
        raise ValueError("Empty metric list.")
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * (pct / 100.0))
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]


def load_av1an_chunks(av1an_temp: Path) -> List[Dict[str, Any]]:
    chunks_path = av1an_temp / "chunks.json"
    ensure_exists(chunks_path, "av1an chunks.json")
    raw = load_json(chunks_path)
    chunks = raw.get("chunks") if isinstance(raw, dict) else raw
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"Invalid chunks.json: {chunks_path}")
    return chunks


def compute_luma_samples(src_file: Path, skip: int, vs_source: str) -> List[float]:
    if not has_vapoursynth():
        raise RuntimeError("VapourSynth is not available (python module not found).")
    import vapoursynth as vs  # type: ignore

    core = vs.core
    clip = load_vs_clip(src_file, vs_source)
    if clip.format is None:
        raise RuntimeError("Unknown clip format for luma calculation.")

    # Normalize to 8-bit YUV to keep luma in 0..255.
    if clip.format.color_family != vs.YUV or clip.format.sample_type != vs.INTEGER or clip.format.bits_per_sample != 8:
        clip = core.resize.Bicubic(clip, format=vs.YUV444P8)

    clip = clip.std.ShufflePlanes(planes=0, colorfamily=vs.GRAY)
    if skip and skip > 1:
        clip = clip.std.SelectEvery(cycle=skip, offsets=0)

    stats = clip.std.PlaneStats()
    samples: List[float] = []
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


# Internal tuning for rule metrics.
GRAD_MAD_SAMPLES_PER_SCENE = 10
GRAD_MAD_MAX_GAP = 16
FARNEBACK_DOWNSCALE_RATIO = 5.0
FARNEBACK_BLUR_RADIUS = 2.0


class RuleMetricsState:
    def __init__(
        self,
        *,
        src_file: Path,
        frames_count: int,
        skip: int,
        av1an_temp: Path,
        fastpass_out: Path,
        vs_source: str,
        verbose: bool,
        scene_ranges: Optional[List[Tuple[int, int]]] = None,
        required_metrics: Optional[Sequence[str]] = None,
    ) -> None:
        self.src_file = src_file
        self.frames_count = int(frames_count)
        self.skip = int(skip)
        self.av1an_temp = av1an_temp
        self.fastpass_out = fastpass_out
        self.vs_source = vs_source
        self.verbose = verbose
        self.scene_ranges = scene_ranges or []
        self.grad_mad_samples_per_scene = int(GRAD_MAD_SAMPLES_PER_SCENE)
        self.grad_mad_max_gap = int(GRAD_MAD_MAX_GAP)
        self.farneback_downscale_ratio = float(FARNEBACK_DOWNSCALE_RATIO)
        self.farneback_blur_radius = float(FARNEBACK_BLUR_RADIUS)
        normalized = [m.strip().lower() for m in (required_metrics or []) if m and str(m).strip()]
        self._shared_need_luma = any(m.startswith("luma_") for m in normalized)
        self._shared_need_grad_mad = any(m.startswith("grad_mad") for m in normalized)
        self._shared_need_farneback = any(m.startswith("farneback") for m in normalized)
        self._shared_metrics_ready = False
        self._luma_samples: Optional[List[float]] = None
        self._luma_avg_all: Optional[float] = None
        self._chunks_by_index: Optional[Dict[int, Dict[str, Any]]] = None
        self._scene_bitrate_cache: Dict[int, float] = {}
        self._video_bitrate_avg: Optional[float] = None
        self._fps: Optional[float] = None
        self._grad_mad_scene_cache: Dict[int, Tuple[float, int]] = {}
        self._grad_mad_global_avg: Optional[float] = None
        self._farneback_scene_cache: Dict[int, Tuple[float, int]] = {}
        self._farneback_global_avg: Optional[float] = None

    def luma_samples(self) -> List[float]:
        if self._luma_samples is None:
            self._ensure_shared_metrics(need_luma=True)
            if self._luma_samples is None:
                raise ValueError("Luma samples are not available.")
        return self._luma_samples

    def luma_avg_all(self) -> float:
        if self._luma_avg_all is None:
            samples = self.luma_samples()
            if not samples:
                raise ValueError("No luma samples for global average.")
            self._luma_avg_all = sum(samples) / len(samples)
        return float(self._luma_avg_all)

    def _iter_scene_ranges(self) -> List[Tuple[int, int]]:
        if self.scene_ranges:
            return self.scene_ranges
        return [(0, self.frames_count)]

    def _ensure_shared_metrics(
        self,
        *,
        need_luma: bool = False,
        need_grad_mad: bool = False,
        need_farneback: bool = False,
    ) -> None:
        if self._shared_metrics_ready:
            missing = []
            if need_luma and not self._shared_need_luma:
                missing.append("luma_*")
            if need_grad_mad and not self._shared_need_grad_mad:
                missing.append("grad_mad_ratio")
            if need_farneback and not self._shared_need_farneback:
                missing.append("farneback_ratio")
            if missing:
                missing_msg = ", ".join(missing)
                raise RuntimeError(
                    f"Shared metrics pass already completed without: {missing_msg}. "
                    "Add them to --rules-required-metrics and re-run."
                )
            return

        if need_luma:
            self._shared_need_luma = True
        if need_grad_mad:
            self._shared_need_grad_mad = True
        if need_farneback:
            self._shared_need_farneback = True

        if not (self._shared_need_luma or self._shared_need_grad_mad or self._shared_need_farneback):
            return

        self._compute_shared_metrics()
        self._shared_metrics_ready = True

    def _compute_shared_metrics(self) -> None:
        if not has_vapoursynth():
            raise RuntimeError("VapourSynth is not available (shared metrics require it).")
        if self._shared_need_farneback and not has_cv2():
            raise RuntimeError("OpenCV is not available (farneback requires it).")
        if self._shared_need_luma or self._shared_need_grad_mad or self._shared_need_farneback:
            _import_numpy()

        import vapoursynth as vs  # type: ignore

        core = vs.core
        clip = load_vs_clip(self.src_file, self.vs_source)
        if clip.format is None or clip.format.color_family != vs.GRAY or clip.format.bits_per_sample != 8:
            clip = core.resize.Bicubic(clip, format=vs.GRAY8)

        total_frames = int(clip.num_frames)
        scene_ranges = self._iter_scene_ranges()
        if not scene_ranges:
            scene_ranges = [(0, total_frames)]
        scene_count = len(scene_ranges)

        luma_samples: Optional[List[float]] = [] if self._shared_need_luma else None
        grad_sums = [0.0] * scene_count if self._shared_need_grad_mad else None
        grad_counts = [0] * scene_count if self._shared_need_grad_mad else None
        far_sums = [0.0] * scene_count if self._shared_need_farneback else None
        far_counts = [0] * scene_count if self._shared_need_farneback else None

        stop_event = threading.Event()
        errors: List[BaseException] = []
        threads: List[threading.Thread] = []
        queues: List[queue.Queue] = []

        def worker_wrap(fn):
            def _run():
                try:
                    fn()
                except BaseException as exc:
                    errors.append(exc)
                    stop_event.set()
            return _run

        def safe_put(q, item) -> bool:
            while True:
                if stop_event.is_set():
                    return False
                try:
                    q.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue

        if self._shared_need_luma:
            q_luma: queue.Queue = queue.Queue(maxsize=4)
            queues.append(q_luma)

            def luma_worker():
                np = _import_numpy()
                skip = max(1, int(self.skip))
                while True:
                    try:
                        item = q_luma.get(timeout=0.1)
                    except queue.Empty:
                        if stop_event.is_set():
                            break
                        continue
                    if stop_event.is_set():
                        break
                    if item is None:
                        break
                    idx, frame = item
                    if idx % skip == 0:
                        luma_samples.append(float(np.mean(frame)))

            t = threading.Thread(target=worker_wrap(luma_worker), name="metrics-luma", daemon=True)
            threads.append(t)
            t.start()

        if self._shared_need_grad_mad:
            q_grad: queue.Queue = queue.Queue(maxsize=4)
            queues.append(q_grad)

            def grad_worker():
                sample_count = max(1, int(self.grad_mad_samples_per_scene))
                max_gap = int(self.grad_mad_max_gap)
                scene_idx = 0
                st, en = scene_ranges[0]

                def scene_step(s, e):
                    length = max(0, int(e) - int(s))
                    if length <= 0:
                        return 1
                    step = max(1, (length + sample_count - 1) // sample_count)
                    if max_gap and max_gap > 0:
                        step = min(step, max_gap)
                    return int(step)

                step = scene_step(st, en)
                next_sample = st
                while True:
                    try:
                        item = q_grad.get(timeout=0.1)
                    except queue.Empty:
                        if stop_event.is_set():
                            break
                        continue
                    if stop_event.is_set():
                        break
                    if item is None:
                        break
                    idx, frame = item
                    while scene_idx < scene_count and idx >= en:
                        scene_idx += 1
                        if scene_idx >= scene_count:
                            break
                        st, en = scene_ranges[scene_idx]
                        step = scene_step(st, en)
                        next_sample = st
                    if scene_idx >= scene_count:
                        continue
                    if idx == next_sample:
                        val = float(grad_mad_from_gray(frame))
                        grad_sums[scene_idx] += val
                        grad_counts[scene_idx] += 1
                        next_sample += step

            t = threading.Thread(target=worker_wrap(grad_worker), name="metrics-grad_mad", daemon=True)
            threads.append(t)
            t.start()

        if self._shared_need_farneback:
            q_far: queue.Queue = queue.Queue(maxsize=4)
            queues.append(q_far)

            def farneback_worker():
                np = _import_numpy()
                cv2 = _import_cv2()
                ratio = float(self.farneback_downscale_ratio)
                blur = float(self.farneback_blur_radius)
                resize_wh: Optional[Tuple[int, int]] = None

                def preprocess(gray):
                    nonlocal resize_wh
                    out = gray
                    if ratio > 1.0:
                        if resize_wh is None:
                            new_w = max(1, int(round(out.shape[1] / ratio)))
                            new_h = max(1, int(round(out.shape[0] / ratio)))
                            resize_wh = (new_w, new_h)
                        out = cv2.resize(out, resize_wh, interpolation=cv2.INTER_AREA)
                    if blur > 0:
                        k = max(3, int(round(blur * 2 + 1)))
                        if k % 2 == 0:
                            k += 1
                        out = cv2.GaussianBlur(out, (k, k), sigmaX=blur)
                    return out

                scene_idx = 0
                st, en = scene_ranges[0]
                prev = None
                prev_scene_idx: Optional[int] = None
                while True:
                    try:
                        item = q_far.get(timeout=0.1)
                    except queue.Empty:
                        if stop_event.is_set():
                            break
                        continue
                    if stop_event.is_set():
                        break
                    if item is None:
                        break
                    idx, frame = item
                    while scene_idx < scene_count and idx >= en:
                        scene_idx += 1
                        if scene_idx >= scene_count:
                            break
                        st, en = scene_ranges[scene_idx]
                    cur_scene_idx = scene_idx if scene_idx < scene_count else None
                    gray = preprocess(frame)
                    if prev is not None and prev_scene_idx is not None and cur_scene_idx == prev_scene_idx:
                        flow = cv2.calcOpticalFlowFarneback(
                            prev,
                            gray,
                            None,
                            0.5,
                            3,
                            15,
                            3,
                            5,
                            1.2,
                            0,
                        )
                        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                        far_sums[cur_scene_idx] += float(np.mean(mag))
                        far_counts[cur_scene_idx] += 1
                    prev = gray
                    prev_scene_idx = cur_scene_idx

            t = threading.Thread(target=worker_wrap(farneback_worker), name="metrics-farneback", daemon=True)
            threads.append(t)
            t.start()

        with tqdm.tqdm(total=total_frames, desc="Metrics decode", unit="frame") as pbar:
            for idx, frame in enumerate(clip.frames()):
                if stop_event.is_set():
                    break
                arr = frame_to_ndarray(frame)
                failed = False
                for q in queues:
                    if not safe_put(q, (idx, arr)):
                        failed = True
                        break
                if failed:
                    break
                pbar.update(1)

        if not stop_event.is_set():
            for q in queues:
                q.put(None)
        else:
            for q in queues:
                try:
                    q.put(None, timeout=0.1)
                except queue.Full:
                    pass

        for t in threads:
            t.join()

        if errors:
            raise errors[0]

        if self._shared_need_luma:
            self._luma_samples = luma_samples or []

        if self._shared_need_grad_mad:
            total_sum = 0.0
            total_count = 0
            for i in range(scene_count):
                count = int(grad_counts[i])
                avg = float(grad_sums[i] / count) if count > 0 else 0.0
                self._grad_mad_scene_cache[i] = (avg, count)
                total_sum += float(grad_sums[i])
                total_count += count
            if total_count <= 0:
                raise ValueError("No grad_mad samples for global average.")
            self._grad_mad_global_avg = total_sum / total_count

        if self._shared_need_farneback:
            total_sum = 0.0
            total_count = 0
            for i in range(scene_count):
                count = int(far_counts[i])
                avg = float(far_sums[i] / count) if count > 0 else 0.0
                self._farneback_scene_cache[i] = (avg, count)
                total_sum += float(far_sums[i])
                total_count += count
            if total_count <= 0:
                raise ValueError("No farneback samples for global average.")
            self._farneback_global_avg = total_sum / total_count

    def grad_mad_scene_stats(self, scene_index: int, st: int, en: int) -> Tuple[float, int]:
        if scene_index not in self._grad_mad_scene_cache:
            self._ensure_shared_metrics(need_grad_mad=True)
        return self._grad_mad_scene_cache.get(scene_index, (0.0, 0))

    def grad_mad_scene_avg(self, scene_index: int, st: int, en: int) -> float:
        avg, _count = self.grad_mad_scene_stats(scene_index, st, en)
        return float(avg)

    def grad_mad_global_avg(self) -> float:
        if self._grad_mad_global_avg is None:
            self._ensure_shared_metrics(need_grad_mad=True)
        if self._grad_mad_global_avg is None:
            raise ValueError("No grad_mad samples for global average.")
        return float(self._grad_mad_global_avg)

    def farneback_scene_stats(self, scene_index: int, st: int, en: int) -> Tuple[float, int]:
        if scene_index not in self._farneback_scene_cache:
            self._ensure_shared_metrics(need_farneback=True)
        return self._farneback_scene_cache.get(scene_index, (0.0, 0))

    def farneback_scene_avg(self, scene_index: int, st: int, en: int) -> float:
        avg, _count = self.farneback_scene_stats(scene_index, st, en)
        return float(avg)

    def farneback_global_avg(self) -> float:
        if self._farneback_global_avg is None:
            self._ensure_shared_metrics(need_farneback=True)
        if self._farneback_global_avg is None:
            raise ValueError("No farneback samples for global average.")
        return float(self._farneback_global_avg)

    def _load_chunks(self) -> Dict[int, Dict[str, Any]]:
        if self._chunks_by_index is None:
            chunks = load_av1an_chunks(self.av1an_temp)
            by_index: Dict[int, Dict[str, Any]] = {}
            for ch in chunks:
                if not isinstance(ch, dict) or "index" not in ch:
                    raise ValueError("Invalid chunks.json entry (missing index).")
                by_index[int(ch["index"])] = ch
            if not by_index:
                raise ValueError("No chunks found in chunks.json.")
            if self._fps is None:
                first = by_index[min(by_index.keys())]
                if "frame_rate" not in first:
                    raise ValueError("chunks.json missing frame_rate field.")
                self._fps = parse_frame_rate(first["frame_rate"])
            self._chunks_by_index = by_index
        return self._chunks_by_index

    def _get_fps(self) -> float:
        if self._fps is None:
            self._load_chunks()
        if not self._fps or self._fps <= 0:
            raise ValueError("Invalid fps derived from chunks.json.")
        return float(self._fps)

    def scene_bitrate(self, scene_index: int) -> float:
        if scene_index in self._scene_bitrate_cache:
            return self._scene_bitrate_cache[scene_index]

        chunks = self._load_chunks()
        if scene_index not in chunks:
            raise KeyError(f"Chunk index {scene_index} not found in chunks.json.")
        ch = chunks[scene_index]

        try:
            st = int(ch["start_frame"])
            en = int(ch["end_frame"])
        except Exception as e:
            raise ValueError(f"Invalid chunk frame range for index {scene_index}: {e}") from e

        fps = parse_frame_rate(ch.get("frame_rate", self._get_fps()))
        duration = (en - st) / fps if fps else 0.0
        if duration <= 0:
            raise ValueError(f"Invalid duration for chunk {scene_index}: {st}..{en}, fps={fps}")

        ivf_path = self.av1an_temp / "encode" / f"{scene_index:05d}.ivf"
        ensure_exists(ivf_path, "av1an encode chunk")
        bytes_size = ivf_path.stat().st_size
        bitrate_kbps = (bytes_size * 8.0) / duration / 1000.0
        self._scene_bitrate_cache[scene_index] = bitrate_kbps
        return bitrate_kbps

    def video_bitrate_avg(self) -> float:
        if self._video_bitrate_avg is None:
            ensure_exists(self.fastpass_out, "Fast-pass output")
            bytes_size = self.fastpass_out.stat().st_size
            fps = self._get_fps()
            duration = self.frames_count / fps if fps else 0.0
            if duration <= 0:
                raise ValueError(f"Invalid video duration: frames={self.frames_count}, fps={fps}")
            self._video_bitrate_avg = (bytes_size * 8.0) / duration / 1000.0
        return float(self._video_bitrate_avg)

    def scene_bitrate_ratio(self, scene_index: int) -> float:
        avg = self.video_bitrate_avg()
        if avg <= 0:
            raise ValueError("Invalid average bitrate for ratio calculation.")
        return self.scene_bitrate(scene_index) / avg


class SceneMetrics:
    def __init__(self, state: RuleMetricsState, scene_index: int, st: int, en: int) -> None:
        self.state = state
        self.scene_index = int(scene_index)
        self.st = int(st)
        self.en = int(en)

    def metric(self, name: str) -> float:
        key = str(name).strip().lower()
        if key.startswith("luma_"):
            samples = slice_samples_for_scene(self.state.luma_samples(), self.st, self.en, self.state.skip)
            if not samples:
                raise ValueError("Empty luma sample set for scene.")
            if key == "luma_avg":
                return sum(samples) / len(samples)
            if key == "luma_min":
                return min(samples)
            if key == "luma_max":
                return max(samples)
            if key == "luma_ratio":
                scene_avg = sum(samples) / len(samples)
                global_avg = self.state.luma_avg_all()
                if global_avg <= 0:
                    raise ValueError("Invalid global luma average for ratio.")
                return scene_avg / global_avg
            m = _LUMA_P_RE.match(key)
            if m:
                pct = int(m.group(1))
                return percentile(samples, pct)
            raise KeyError(f"Unsupported luma metric: {name}")

        if key == "grad_mad_ratio":
            scene_avg = self.state.grad_mad_scene_avg(self.scene_index, self.st, self.en)
            global_avg = self.state.grad_mad_global_avg()
            if global_avg <= 0:
                raise ValueError("Invalid global grad_mad average for ratio.")
            return scene_avg / global_avg

        if key == "farneback_ratio":
            scene_avg = self.state.farneback_scene_avg(self.scene_index, self.st, self.en)
            global_avg = self.state.farneback_global_avg()
            if global_avg <= 0:
                raise ValueError("Invalid global farneback average for ratio.")
            return scene_avg / global_avg

        if key == "scene_bitrate":
            return self.state.scene_bitrate(self.scene_index)
        if key == "scene_bitrate_ratio":
            return self.state.scene_bitrate_ratio(self.scene_index)

        raise KeyError(f"Unknown metric name: {name}")


class VideoParamEditor:
    def __init__(
        self,
        tokens: List[str],
        *,
        scene_index: int,
        start_frame: int,
        end_frame: int,
        rule_name: str,
        verbose: bool,
    ) -> None:
        self.tokens = tokens
        self.scene_index = int(scene_index)
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)
        self.rule_name = rule_name
        self.verbose = verbose

    def _format_log_value(self, value: Any) -> str:
        if value is None:
            return "<missing>"
        if value is True:
            return "<flag>"
        return str(value)

    def _rule_origin(self) -> Optional[str]:
        try:
            frame = sys._getframe()
        except Exception:
            return None
        while frame:
            if frame.f_code.co_filename == self.rule_name:
                return f"{self.rule_name}:{frame.f_lineno}"
            frame = frame.f_back
        return None

    def _log_change(self, action: str, key: str, old: Any, new: Any) -> None:
        if not self.verbose:
            return
        origin = self._rule_origin()
        origin_msg = f" ({origin})" if origin else ""
        print(
            f"[rules] scene {self.scene_index} [{self.start_frame}:{self.end_frame}] "
            f"{action} {key} {self._format_log_value(old)} -> {self._format_log_value(new)}{origin_msg}"
        )

    def param(self, name: str) -> Any:
        key = normalize_param_key(name)
        loc = find_last_option(self.tokens, key)
        if loc is None:
            return None
        k_idx, has_val = loc
        if not has_val:
            return True
        return coerce_param_value(self.tokens[k_idx + 1])

    def sparam(self, name: str, value: Any = None) -> None:
        key = normalize_param_key(name)
        old_val = self.param(key)
        val_token: Optional[str]
        if value is None or (isinstance(value, bool) and value is True):
            val_token = None
        else:
            val_token = format_param_value(value)

        loc = find_last_option(self.tokens, key)
        if loc is None:
            self.tokens.append(key)
            if val_token is not None:
                self.tokens.append(val_token)
        else:
            k_idx, has_val = loc
            if val_token is None:
                if has_val:
                    del self.tokens[k_idx + 1]
            else:
                if has_val:
                    self.tokens[k_idx + 1] = val_token
                else:
                    self.tokens.insert(k_idx + 1, val_token)

        new_val = self.param(key)
        self._log_change("set", key, old_val, new_val)

    def cparam(self, name: str, delta: Any) -> None:
        key = normalize_param_key(name)
        old_val = self.param(key)
        cur_num = parse_numeric_value(old_val)
        delta_num = parse_numeric_value(delta)
        if cur_num is None or delta_num is None:
            eprint(f"[warn] cparam skipped for {key}: non-numeric or missing value ({old_val}).")
            return

        new_val = cur_num + delta_num
        val_token = format_param_value(new_val)
        loc = find_last_option(self.tokens, key)
        if loc is None:
            self.tokens.append(key)
            self.tokens.append(val_token)
        else:
            k_idx, has_val = loc
            if has_val:
                self.tokens[k_idx + 1] = val_token
            else:
                self.tokens.insert(k_idx + 1, val_token)

        self._log_change("change", key, old_val, new_val)


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


def validate_scenes_with_overrides(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    scenes = obj.get("scenes") or obj.get("split_scenes") or []
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Invalid scenes.json: expected non-empty 'scenes' or 'split_scenes' list")

    for i, s in enumerate(scenes):
        if not isinstance(s, dict):
            raise ValueError(f"Invalid scene entry at index {i}: expected object")
        if "start_frame" not in s or "end_frame" not in s:
            raise ValueError(f"Invalid scene entry at index {i}: missing start_frame/end_frame")
        zo = s.get("zone_overrides")
        if not isinstance(zo, dict) or "video_params" not in zo:
            raise ValueError(f"Scene {i} missing zone_overrides/video_params (rules require Stage 4 output).")
        if not isinstance(zo["video_params"], list):
            raise ValueError(f"Scene {i} zone_overrides.video_params must be a list.")

    return scenes


def check_rules_prereqs(required_metrics: Sequence[str], av1an_temp: Path, fastpass_out: Path) -> None:
    if not required_metrics:
        return
    normalized = [m.strip().lower().rstrip("*") for m in required_metrics if m and m.strip()]
    if any(m.startswith("luma_") for m in normalized):
        if not has_vapoursynth():
            raise RuntimeError("Rules require luma_* metrics, but VapourSynth is not available.")
        _import_numpy()

    if any(m.startswith("grad_mad") for m in normalized):
        if not has_vapoursynth():
            raise RuntimeError("Rules require grad_mad metrics, but VapourSynth is not available.")
        _import_numpy()

    if any(m.startswith("farneback") for m in normalized):
        if not has_vapoursynth():
            raise RuntimeError("Rules require farneback metrics, but VapourSynth is not available.")
        _import_numpy()
        _import_cv2()

    if any(m.startswith("scene_bitrate") for m in normalized):
        chunks_path = av1an_temp / "chunks.json"
        encode_dir = av1an_temp / "encode"
        if not chunks_path.exists() or not encode_dir.exists():
            raise RuntimeError(
                "Rules require scene_bitrate*, but av1an temp artifacts are missing. "
                "Run fast-pass with --keep and keep access to av1an_temp."
            )
        ensure_exists(fastpass_out, "Fast-pass output")


# ---------------------------
# Stage 4: base scenes (zone_overrides)
# ---------------------------

def compute_chunk_5p_single_metric(
    *,
    scene_ranges: List[Tuple[int, int]],
    metrics_mode: int,
    ssimu2_path: Optional[Path],
    xpsnr_path: Optional[Path],
) -> Tuple[int, List[float], float]:
    """
    Return:
      skip_used_for_ssimu2,
      per_chunk_percentile_5_list,
      global_average
    metrics_mode:
      1 = SSIMU2 (sampled at skip)
      2 = XPSNR (per-frame)
    """
    if metrics_mode == 1:
        assert ssimu2_path is not None
        skip, ssimu2_scores = parse_ssimu2_log(ssimu2_path)

        per_chunk_5: List[float] = []
        for (st, en) in scene_ranges:
            chunk = slice_samples_for_scene(ssimu2_scores, st, en, skip)
            _, p5, _ = calc_stats(chunk)
            per_chunk_5.append(p5)

        avg_total, _, _ = calc_stats(ssimu2_scores)
        return skip, per_chunk_5, avg_total

    if metrics_mode == 2:
        assert xpsnr_path is not None
        xpsnr_scores = parse_xpsnr_log(xpsnr_path)

        per_chunk_5 = []
        for (st, en) in scene_ranges:
            chunk = xpsnr_scores[st:en] if en > st else []
            if not chunk:
                chunk = [xpsnr_scores[min(st, len(xpsnr_scores) - 1)]]
            _, p5, _ = calc_stats(chunk)
            per_chunk_5.append(p5)

        avg_total, _, _ = calc_stats(xpsnr_scores)
        return 1, per_chunk_5, avg_total

    raise ValueError("compute_chunk_5p expects metrics_mode 1 or 2.")


def combine_metrics_per_chunk(
    *,
    scene_ranges: List[Tuple[int, int]],
    ssimu2_path: Path,
    xpsnr_path: Path,
    combine_mode: int,
) -> Tuple[int, List[float], float]:
    """
    combine_mode:
      1 = Division
      2 = Addition
      3 = Multiplication
      4 = Lowest result (min)

    Align XPSNR to SSIMU2 by downsampling XPSNR into groups of `skip` frames:
      xpsnr_ds[k] = mean(xpsnr[k*skip:(k+1)*skip])

    Then for each scene we take the same sampled indices (k*skip within the scene),
    and combine per-sample.

    Returns (skip, per_chunk_5, global_avg).
    """
    skip, ssimu2_scores = parse_ssimu2_log(ssimu2_path)
    xpsnr_scores = parse_xpsnr_log(xpsnr_path)

    # Downsample XPSNR globally to match SSIMU2 sample indices.
    xpsnr_ds: List[float] = []
    for k in range(len(ssimu2_scores)):
        g = xpsnr_scores[k * skip: (k + 1) * skip]
        if not g:
            g = [xpsnr_scores[-1]]
        xpsnr_ds.append(sum(g) / len(g))

    total_combined: List[float] = []
    per_chunk_5: List[float] = []

    for (st, en) in scene_ranges:
        s_chunk = slice_samples_for_scene(ssimu2_scores, st, en, skip)

        # Determine which sample indices we used, to slice XPSNR DS identically.
        k0 = (st + skip - 1) // skip
        k1 = (en - 1) // skip
        k0 = max(0, min(k0, len(xpsnr_ds) - 1))
        k1 = max(0, min(k1, len(xpsnr_ds) - 1))
        x_chunk = xpsnr_ds[k0: k1 + 1] if k1 >= k0 else [xpsnr_ds[k0]]

        # Safety: lengths should match; if not, clamp.
        n = min(len(s_chunk), len(x_chunk))
        if n <= 0:
            s_chunk = [ssimu2_scores[min(k0, len(ssimu2_scores) - 1)]]
            x_chunk = [xpsnr_ds[min(k0, len(xpsnr_ds) - 1)]]
            n = 1

        combined_chunk: List[float] = []
        for s_val, x_val in zip(s_chunk[:n], x_chunk[:n]):
            if combine_mode == 1:
                combined = x_val / s_val if s_val != 0 else x_val
            elif combine_mode == 2:
                combined = x_val + s_val
            elif combine_mode == 3:
                combined = x_val * s_val
            elif combine_mode == 4:
                combined = min(x_val, s_val)
            else:
                raise ValueError("Invalid combine_mode (expected 1..4).")
            combined_chunk.append(combined)

        total_combined.extend(combined_chunk)
        _, p5, _ = calc_stats(combined_chunk)
        per_chunk_5.append(p5)

    avg_total, _, _ = calc_stats(total_combined)
    return skip, per_chunk_5, avg_total


def build_zone_overrides(
    *,
    crf: float,
    preset: int,
    video_params_str: str,
    final_override: str,
) -> Dict[str, Any]:
    def apply_override(base_tokens: List[str], override_tokens: List[str]) -> List[str]:
        i = 0
        while i < len(override_tokens):
            tok = override_tokens[i]
            if not is_param_key(tok):
                # Ignore stray/positional tokens in override
                i += 1
                continue

            key = tok
            # One-value override (common for SVT options). If next token is not a key, treat as value.
            has_val = (i + 1 < len(override_tokens)) and (not is_param_key(override_tokens[i + 1]))
            val = override_tokens[i + 1] if has_val else None

            loc = find_last_option(base_tokens, key)
            if loc is None:
                # Not present -> append
                base_tokens.append(key)
                if val is not None:
                    base_tokens.append(val)
            else:
                k_idx, base_has_val = loc
                if val is None:
                    # Override sets flag form: remove existing value if any
                    if base_has_val:
                        del base_tokens[k_idx + 1]
                else:
                    # Override sets/changes value
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



def apply_crf_adjustments_to_scenes(
    *,
    base_scenes_path: Path,
    out_scenes_path: Path,
    scene_ranges: List[Tuple[int, int]],
    per_chunk_5: List[float],
    avg_total: float,
    base_crf: float,
    aggressive: float,
    deviation: float,
    max_positive_dev: Optional[float],
    max_negative_dev: Optional[float],
    final_preset: int,
    video_params: str,
    final_override: str
) -> None:
    base = load_json(base_scenes_path)
    base_norm = sanitize_scenes_json(base)
    frames_count = int(base_norm["frames"])

    # Determine deviation limits (match 2.5 semantics)
    base_dev = float(deviation)
    max_pos = max_positive_dev if max_positive_dev is not None else base_dev
    max_neg = max_negative_dev if max_negative_dev is not None else base_dev

    aggressive = aggressive * 20


    updated: List[Dict[str, Any]] = []
    for i, (st, en) in enumerate(scene_ranges):
        adj = math.ceil((1.0 - (per_chunk_5[i] / avg_total)) * aggressive * 4.0) / 4.0
        new_crf = float(base_crf) - float(adj)

        # Clamp deviations
        if adj < 0:  # increasing CRF
            if max_pos == 0:
                new_crf = float(base_crf)
            elif abs(adj) > float(max_pos):
                new_crf = float(base_crf) + float(max_pos)
        else:  # decreasing CRF
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
            "zone_overrides": build_zone_overrides(crf=new_crf, preset=final_preset, video_params_str=video_params, final_override=final_override),
        })

    out_obj = {"frames": frames_count, "scenes": updated, "split_scenes": [dict(x) for x in updated]}
    ensure_dir(out_scenes_path.parent)
    save_json(out_scenes_path, out_obj)
    print(f"[ok] base scenes.json written: {out_scenes_path}")


def apply_rules_to_scenes(
    *,
    base_obj: Dict[str, Any],
    compiled_rules: Any,
    rule_name: str,
    metrics_state: RuleMetricsState,
    verbose: bool,
) -> Dict[str, Any]:
    scenes = validate_scenes_with_overrides(base_obj)
    frames = int(base_obj.get("frames", scenes[-1]["end_frame"] if scenes else 0))

    updated: List[Dict[str, Any]] = []
    for i, s in enumerate(scenes):
        st = int(s["start_frame"])
        en = int(s["end_frame"])
        zone = dict(s["zone_overrides"])
        video_params = list(zone["video_params"])

        def rule_log(*args: Any) -> None:
            if not verbose:
                return
            try:
                frame = sys._getframe()
            except Exception:
                frame = None
            origin = None
            while frame:
                if frame.f_code.co_filename == rule_name:
                    origin = f"{rule_name}:{frame.f_lineno}"
                    break
                frame = frame.f_back
            origin_msg = f" ({origin})" if origin else ""
            msg = " ".join(str(a) for a in args)
            print(f"[rules] scene {i} [{st}:{en}] {msg}{origin_msg}")

        editor = VideoParamEditor(
            video_params,
            scene_index=i,
            start_frame=st,
            end_frame=en,
            rule_name=rule_name,
            verbose=verbose,
        )
        metrics = SceneMetrics(metrics_state, i, st, en)

        ctx_globals = {
            "__builtins__": __builtins__,
            "metric": metrics.metric,
            "param": editor.param,
            "sparam": editor.sparam,
            "cparam": editor.cparam,
            "log": rule_log,
        }
        ctx_locals: Dict[str, Any] = {}

        try:
            exec(compiled_rules, ctx_globals, ctx_locals)
        except Exception as exc:
            eprint(f"[error] rules failed for scene {i} [{st}:{en}]: {exc}")
            raise

        normalize_video_params(video_params)
        zone["video_params"] = video_params
        updated.append({
            "start_frame": st,
            "end_frame": en,
            "zone_overrides": zone,
        })

    return {"frames": frames, "scenes": updated, "split_scenes": [dict(x) for x in updated]}


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="auto-boost 2.8.json + av1an fastpass + per-scene CRF zones + resume.")
    parser.add_argument("-s", "--stage", type=int, default=0,
                        help="Stage: 0=all, 1=PSD scenes (sdm=psd), 2=fastpass, 3=metrics, 4=write base scenes.json, 5=apply rules.")
    parser.add_argument("-i", "--input", required=True, help="Input video file (source).")
    parser.add_argument("-t", "--temp", default=None,
                        help="Project directory. Default: <input stem>_autoboost next to input.")
    parser.add_argument("--log", default="",
                        help="Optional log file path (relative to --temp if not absolute).")
    parser.add_argument("--force", action="store_true",
                        help="Start from scratch: clear resume markers and remove generated artifacts in the project dir.")
    parser.add_argument("--sdm", choices=["psd", "av1an"], default="psd",
                        help="Scene detection mode: psd (default, run PSD and pass --scenes) or av1an (use av1an internal detection).")

    # PSD
    parser.add_argument("--psd-script", default="Progressive-Scene-Detection.py",
                        help="Path to Progressive-Scene-Detection.py (PSD). Default: Progressive-Scene-Detection.py (cwd).")
    parser.add_argument("--psd-python", default=None,
                        help="Python executable for PSD. Default: current Python.")
    parser.add_argument("--psd-args", default="",
                        help="Extra arguments appended to PSD invocation (string, shlex-split).")
    parser.add_argument("--base-scenes", default=None,
                        help="Use an existing PSD scenes.json instead of running PSD (zone_overrides must be null; --sdm psd only).")
    parser.add_argument("--out-scenes", default=None,
                        help="Final scenes.json output path. Default: <project>/scenes.final.json")

    # av1an fastpass
    parser.add_argument("--fastpass-out", default=None,
                        help="Fast-pass output file. Default: <project>/<stem>.fastpass.mkv")
    parser.add_argument("--workers", type=int, default=8,
                        help="av1an workers for fast-pass.")
    parser.add_argument("--lp", type=int, default=3, help="--lp for fast-pass encoding (svt-av1).")
    parser.add_argument("-q", "--quality", type=float, default=30.0, help="Base CRF for targeting (also used for fast-pass).")
    parser.add_argument("--fast-preset", type=int, default=7, help="Fast-pass SVT preset (speed).")
    parser.add_argument("-p", "--preset", type=int, default=2, help="Final SVT preset embedded into zone_overrides.")
    parser.add_argument("-v", "--video-params", default="",
                        help="Extra SVT encoder params to embed into zone_overrides (string). Do NOT include --crf/--preset.")
    parser.add_argument("--final-override", default="",
                        help="Overriding params from video-params for final encoding scenes.")
    parser.add_argument("-f", "--ffmpeg", default="",
                        help="FFmpeg options for av1an. If it starts with '-', passed as-is to av1an -f. Otherwise treated as a filtergraph and wrapped as -vf \"...\".")
    parser.add_argument("--verbose", action="store_true", help="Verbose output (fast-pass + rules logging).")
    parser.add_argument("--keep", action="store_true", help="Pass --keep to av1an (keep temp files).")

    # Rules
    rules_group = parser.add_mutually_exclusive_group()
    rules_group.add_argument("--rules", default=None, help="Path to python rule script.")
    rules_group.add_argument("--rules-inline", default=None, help="Inline python rules string.")
    parser.add_argument("--rules-required-metrics", nargs="*", default=["luma","grad_mad","farneback"],
                        help="Space-separated list of metric names required by rules (fail-fast).")
    parser.add_argument("--rule-test", action="store_true",
                        help="Execute rules without saving results; implies --verbose.")

    # Metrics
    parser.add_argument("-m", "--metrics", type=int, default=1, help="Metrics: 1=SSIMU2, 2=XPSNR, 3=Both.")
    parser.add_argument("-S", "--skip", type=int, default=2, help="SSIMU2 sampling step (VapourSynth SelectEvery).")

    # SSIMU2 backend selection
    parser.add_argument("--ssimu2-backend", default="auto",
                        help="SSIMU2 backend (VapourSynth only): auto (default), vship (preferred), vszip.")
    parser.add_argument("--vs-source", default="ffms2",
                        help="VapourSynth decode provider for SSIMU2 clips: ffms2 (default), bestsource, lsmas, auto.")

    # Zone computation
    parser.add_argument("-z", "--zones", type=int, default=1,
                        help="When --metrics 3: combine mode: 1=Division, 2=Addition, 3=Multiplication, 4=Lowest result.")
    parser.add_argument("-d", "--deviation", type=float, default=10.0,
                        help="Max CRF deviation (used for both directions if --max-* not set).")
    parser.add_argument("--max-positive-dev", type=float, default=None,
                        help="Max allowed CRF increase above base (worse quality).")
    parser.add_argument("--max-negative-dev", type=float, default=None,
                        help="Max allowed CRF decrease below base (better quality).")
    parser.add_argument("-a", "--aggressive", type=float, default=1.0, help="boosting multiplier.")
    parser.add_argument("--avg-shift", type=float, default=0.0, help="shift metric avg")

    args = parser.parse_args()

    if args.rule_test and not args.verbose:
        args.verbose = True
    if args.sdm == "av1an" and args.base_scenes:
        raise RuntimeError("--base-scenes cannot be used with --sdm av1an (av1an generates scenes during stage 2).")

    input_file = Path(args.input).expanduser().resolve()
    ensure_exists(input_file, "Input file")

    project_dir = Path(args.temp).expanduser().resolve() if args.temp else (input_file.parent / f"{input_file.stem}")
    ensure_dir(project_dir)
    setup_logging(args.log, project_dir)

    av1an_temp = project_dir / "av1an_temp"
    ensure_dir(av1an_temp)

    base_scenes_path = Path(args.base_scenes).expanduser().resolve() if args.base_scenes else (project_dir / "scenes.psd.json")
    out_scenes_path = Path(args.out_scenes).expanduser().resolve() if args.out_scenes else (project_dir / "scenes.final.json")
    fastpass_out = Path(args.fastpass_out).expanduser().resolve() if args.fastpass_out else (project_dir / f"{input_file.stem}.fastpass.mkv")

    ssimu2_log = project_dir / f"{input_file.stem}_ssimu2.log"
    xpsnr_log = project_dir / f"{input_file.stem}_xpsnr.log"

    marks = marker_paths(project_dir)

    rule_name: Optional[str] = None
    compiled_rules: Optional[Any] = None
    required_metrics: List[str] = []
    if args.stage in (0, 5):
        if args.rules or args.rules_inline:
            if args.rules:
                rules_path = Path(args.rules).expanduser().resolve()
                ensure_exists(rules_path, "Rules file")
                rules_text = rules_path.read_text(encoding="utf-8")
                rule_name = str(rules_path)
            else:
                rules_text = str(args.rules_inline)
                rule_name = "<rules-inline>"

            compiled_rules = compile(rules_text, rule_name, "exec")
            required_metrics.extend([m.strip() for m in args.rules_required_metrics if m and m.strip()])
            for m in extract_metric_calls(rules_text):
                if m not in required_metrics:
                    required_metrics.append(m)
        elif args.rules_required_metrics:
            eprint("[warn] --rules-required-metrics specified without --rules/--rules-inline; ignoring.")

    if args.force:
        print("[force] clearing state markers and generated outputs...")
        for mp in marks.values():
            safe_unlink(mp)

        # Remove generated artifacts (do not remove user-provided --base-scenes).
        if not args.base_scenes:
            safe_unlink(base_scenes_path)
        safe_unlink(out_scenes_path)
        safe_unlink(ssimu2_log)
        safe_unlink(xpsnr_log)
        safe_unlink(fastpass_out)

        # Wipe av1an temp directory unless user wants to keep it explicitly.
        if av1an_temp.exists():
            try:
                shutil.rmtree(av1an_temp)
            except Exception as ex:
                eprint(f"[warn] failed to remove av1an temp dir: {av1an_temp} ({ex})")
        ensure_dir(av1an_temp)

    # Tool sanity checks (warnings only)
    if not which_or_none("av1an"):
        eprint("[warn] 'av1an' not found in PATH. Stages 2+ will fail unless you add it to PATH.")
    if not which_or_none("ffmpeg") and args.metrics in (2, 3):
        raise RuntimeError("'ffmpeg' not found in PATH, but XPSNR was requested.")

    # -----------------
    # Stage 1: PSD (scene detection)
    # -----------------
    if args.stage in (0, 1):
        if args.sdm == "av1an":
            if args.stage == 1:
                print("[skip] --sdm av1an uses av1an scene detection during stage 2.")
        else:
            if args.base_scenes:
                print(f"[skip] using existing base scenes: {base_scenes_path}")
                if not is_valid_base_scenes(base_scenes_path):
                    raise RuntimeError("--base-scenes provided but is not a valid scenes.json (or cannot be sanitized).")
                touch(marks["psd"])
            else:
                if marks["psd"].exists() and is_valid_base_scenes(base_scenes_path):
                    print(f"[resume] PSD already completed: {base_scenes_path}")
                else:
                    psd_script = Path(args.psd_script).expanduser()
                    if not psd_script.exists():
                        cand = project_dir / psd_script
                        if cand.exists():
                            psd_script = cand
                    psd_python = Path(args.psd_python).expanduser() if args.psd_python else None
                    run_psd(psd_script=psd_script, psd_python=psd_python, input_file=input_file,
                            base_scenes_path=base_scenes_path, extra_args=args.psd_args)
                    touch(marks["psd"])

    # -----------------
    # Stage 2: fast-pass
    # -----------------
    if args.stage in (0, 2):
        if marks["fastpass"].exists() and fastpass_out.exists() and fastpass_out.stat().st_size > 0:
            print(f"[resume] fast-pass already completed: {fastpass_out}")
        else:
            run_fastpass_av1an(
                input_file=input_file,
                output_file=fastpass_out,
                scenes_path=base_scenes_path if args.sdm == "psd" else None,
                av1an_temp=av1an_temp,
                workers=int(args.workers),
                lp=int(args.lp),
                fast_preset=int(args.fast_preset),
                fast_crf=float(args.quality),
                video_params=str(args.video_params),
                ffmpeg_arg=str(args.ffmpeg),
                verbose=bool(args.verbose),
                keep=bool(args.keep),
            )
            touch(marks["fastpass"])

    frames_count = 0
    scene_ranges: List[Tuple[int, int]] = []
    if args.stage in (0, 3, 4, 5):
        if args.sdm == "av1an":
            try:
                write_base_scenes_from_av1an(av1an_temp, base_scenes_path)
            except FileNotFoundError as exc:
                if not is_valid_base_scenes(base_scenes_path):
                    raise
                eprint(f"[warn] {exc}. Using existing base scenes: {base_scenes_path}")
        ensure_exists(base_scenes_path, "Base scenes.json")
        base_scenes_obj = sanitize_scenes_json(load_json(base_scenes_path))
        frames_count = int(base_scenes_obj["frames"])
        scene_ranges = scenes_to_ranges(base_scenes_obj)

    # -----------------
    # Stage 3: metrics
    # -----------------
    if args.stage in (0, 3):
        ensure_exists(fastpass_out, "Fast-pass output")

        if args.metrics in (1, 3):
            if marks["ssimu2"].exists() and is_valid_ssimu2_log(ssimu2_log):
                print(f"[resume] SSIMU2 already completed: {ssimu2_log}")
            else:
                calculate_ssimu2(
                    src_file=input_file,
                    enc_file=fastpass_out,
                    out_path=ssimu2_log,
                    frames_count=frames_count,
                    skip=int(args.skip),
                    backend=str(args.ssimu2_backend),
                    vs_source=str(args.vs_source),
                )
                touch(marks["ssimu2"])

        if args.metrics in (2, 3):
            if marks["xpsnr"].exists() and is_valid_xpsnr_log(xpsnr_log):
                print(f"[resume] XPSNR already completed: {xpsnr_log}")
            else:
                # If --ffmpeg was provided as a pure filtergraph (no leading '-'), apply it to the source for XPSNR.
                src_fg = str(args.ffmpeg).strip()
                if src_fg.startswith('-'):
                    src_fg = ''
                calculate_xpsnr(input_file, fastpass_out, xpsnr_log, src_filtergraph=src_fg)
                touch(marks["xpsnr"])

    # -----------------
    # Stage 4: base scenes
    # -----------------
    if args.stage in (0, 4):
        if marks["final"].exists() and is_valid_final_scenes(out_scenes_path):
            print(f"[resume] base scenes already written: {out_scenes_path}")
        else:
            if args.metrics == 1:
                _, per_chunk_5, avg_total = compute_chunk_5p_single_metric(
                    scene_ranges=scene_ranges,
                    metrics_mode=1,
                    ssimu2_path=ssimu2_log,
                    xpsnr_path=None,
                )
            elif args.metrics == 2:
                _, per_chunk_5, avg_total = compute_chunk_5p_single_metric(
                    scene_ranges=scene_ranges,
                    metrics_mode=2,
                    ssimu2_path=None,
                    xpsnr_path=xpsnr_log,
                )
            elif args.metrics == 3:
                _, per_chunk_5, avg_total = combine_metrics_per_chunk(
                    scene_ranges=scene_ranges,
                    ssimu2_path=ssimu2_log,
                    xpsnr_path=xpsnr_log,
                    combine_mode=int(args.zones),
                )
            else:
                raise ValueError("Invalid --metrics (expected 1..3).")
            
            if args.avg_shift:
                print(f"[avg-shift] {avg_total} -> {avg_total + args.avg_shift}")
                
            apply_crf_adjustments_to_scenes(
                base_scenes_path=base_scenes_path,
                out_scenes_path=out_scenes_path,
                scene_ranges=scene_ranges,
                per_chunk_5=per_chunk_5,
                avg_total=avg_total + args.avg_shift,
                base_crf=float(args.quality),
                aggressive=float(args.aggressive),
                deviation=float(args.deviation),
                max_positive_dev=args.max_positive_dev,
                max_negative_dev=args.max_negative_dev,
                final_preset=int(args.preset),
                video_params=str(args.video_params),
                final_override=str(args.final_override)
            )
            touch(marks["final"])

    # -----------------
    # Stage 5: apply rules
    # -----------------
    if args.stage in (0, 5):
        if compiled_rules is None:
            if args.stage == 5:
                print("[skip] no rules provided for stage 5.")
        else:
            if marks["rules"].exists() and is_valid_final_scenes(out_scenes_path):
                print(f"[resume] rules already applied: {out_scenes_path}")
            else:
                ensure_exists(out_scenes_path, "Base scenes.json (Stage 4 output)")
                check_rules_prereqs(required_metrics, av1an_temp, fastpass_out)

                base_obj = load_json(out_scenes_path)
                rule_scene_ranges = scenes_to_ranges(base_obj)
                metrics_state = RuleMetricsState(
                    src_file=input_file,
                    frames_count=frames_count,
                    skip=int(args.skip),
                    av1an_temp=av1an_temp,
                    fastpass_out=fastpass_out,
                    vs_source=str(args.vs_source),
                    verbose=bool(args.verbose),
                    scene_ranges=rule_scene_ranges,
                    required_metrics=required_metrics,
                )
                updated_obj = apply_rules_to_scenes(
                    base_obj=base_obj,
                    compiled_rules=compiled_rules,
                    rule_name=rule_name or "<rules>",
                    metrics_state=metrics_state,
                    verbose=bool(args.verbose),
                )
                if args.rule_test:
                    print("[ok] rules executed (test mode): no output written.")
                    return 0

                save_json(out_scenes_path, updated_obj)
                touch(marks["rules"])
                print(f"[ok] rules applied: {out_scenes_path}")

    print("\nDone.")
    print(f"Base scenes : {base_scenes_path}")
    print(f"Fast-pass   : {fastpass_out}")
    print(f"SSIMU2 log  : {ssimu2_log if args.metrics in (1,3) else '(disabled)'}")
    print(f"XPSNR log   : {xpsnr_log if args.metrics in (2,3) else '(disabled)'}")
    print(f"Final scenes: {out_scenes_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        eprint(f"[error] command failed with exit code {e.returncode}")
        raise
