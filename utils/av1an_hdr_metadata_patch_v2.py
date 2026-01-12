#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
av1an_hdr_metadata_patch.py

Patch an av1an scenes.json to carry HDR metadata per-scene:

- HDR10 static metadata (applied to every scene):
  --enable-hdr 1
  --color-primaries
  --transfer-characteristics
  --matrix-coefficients
  --color-range
  --chroma-sample-position
  --mastering-display
  --content-light

- HDR10+ dynamic metadata (per scene):
  --hdr10plus-json <chunk.json>

- Dolby Vision dynamic metadata (per scene):
  --dolby-vision-rpu <chunk.rpu>

The script expects scenes.json with this structure (as produced by av1an):
{
  "frames": <total_frames>,
  "scenes": [ { "start_frame": 0, "end_frame": 70, "zone_overrides": { "video_params": [...] } }, ... ],
  "split_scenes": [ ... ]  # duplicated in your case
}

Notes:
- start_frame inclusive; end_frame exclusive.
- For HDR10+ we use hdr10plus_tool editor (remove everything outside the scene range),
  so the chunk JSON becomes chunk-local (starts at frame 0).
- For DV we reuse the AnnexB-NAL splitting logic (one RPU NAL per frame),
  producing chunk-local .rpu files.

Dependencies (must be in PATH, depending on enabled features):
  ffprobe, ffmpeg, hdr10plus_tool, dovi_tool
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


LOG = logging.getLogger("av1an_hdrmeta_patch")
STATE_DIR_NAME = ".state"
HDR_PATCH_MARKER = "HDR_PATCH_DONE"


# --------------------------
# Utilities
# --------------------------

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    LOG.handlers.clear()
    LOG.addHandler(handler)
    LOG.setLevel(level)


def state_root_from_workdir(workdir: Path) -> Path:
    if workdir.name.lower() == "hdr_tmp" and workdir.parent.name.lower() == "video":
        return workdir.parent.parent
    return workdir


def marker_path(workdir: Path) -> Path:
    root = state_root_from_workdir(workdir)
    return root / STATE_DIR_NAME / HDR_PATCH_MARKER


def write_marker(workdir: Path) -> None:
    p = marker_path(workdir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("ok\n", encoding="utf-8")


def require_tool(name: str) -> None:
    if not shutil.which(name):
        raise RuntimeError(f"Required tool not found in PATH: {name}")


def run_checked(cmd: List[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    LOG.debug("CMD: %s", " ".join(cmd))
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        stdout = (e.stdout or b"").decode("utf-8", errors="replace").strip()
        stderr = (e.stderr or b"").decode("utf-8", errors="replace").strip()
        if stdout:
            LOG.debug("STDOUT: %s", stdout)
        if stderr:
            LOG.error("STDERR: %s", stderr)
        raise


def pipe_checked(producer_cmd: List[str], consumer_cmd: List[str]) -> None:
    """
    producer stdout -> consumer stdin
    """
    LOG.debug("PIPE: %s | %s", " ".join(producer_cmd), " ".join(consumer_cmd))
    p1 = subprocess.Popen(producer_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        p2 = subprocess.Popen(consumer_cmd, stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert p1.stdout is not None
        p1.stdout.close()

        out2, err2 = p2.communicate()
        err1 = p1.stderr.read() if p1.stderr else b""
        rc1 = p1.wait()
        rc2 = p2.returncode

        if rc1 != 0:
            raise RuntimeError(f"Producer failed (rc={rc1}): {err1.decode('utf-8', errors='replace').strip()}")
        if rc2 != 0:
            raise RuntimeError(f"Consumer failed (rc={rc2}): {err2.decode('utf-8', errors='replace').strip()}")
        # keep out2 in case you want it in future
    finally:
        try:
            if p1.poll() is None:
                p1.kill()
        except Exception:
            pass


def parse_fraction(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            return float(a) / float(b)
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    if "/" in s:
        f = parse_fraction(s)
        return int(round(f)) if f is not None else None
    try:
        return int(s)
    except Exception:
        return None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=4, ensure_ascii=False), encoding="utf-8")


def upsert_flag_value(params: List[str], flag: str, value: str) -> List[str]:
    """
    Idempotent:
      - if flag exists (possibly multiple times) -> keep only one, set value
      - if flag absent -> append [flag, value]
    """
    out: List[str] = []
    i = 0
    found = False

    while i < len(params):
        if params[i] == flag:
            if not found:
                out.append(flag)
                out.append(value)
                found = True
            # skip old value if present
            i += 2 if (i + 1) < len(params) else 1
            # skip duplicates
            while i < len(params) and params[i] == flag:
                i += 2 if (i + 1) < len(params) else 1
            continue

        out.append(params[i])
        i += 1

    if not found:
        out.extend([flag, value])

    return out


# --------------------------
# ffprobe HDR10 static metadata
# --------------------------

@dataclass
class Hdr10Static:
    color_primaries: Optional[str] = None
    transfer_characteristics: Optional[str] = None
    matrix_coefficients: Optional[str] = None
    color_range: Optional[str] = None
    chroma_sample_position: Optional[str] = None
    mastering_display: Optional[str] = None
    content_light: Optional[str] = None


def ffprobe_json(
    video_path: Path,
    show_entries: str,
    *,
    read_intervals: Optional[str] = None,
    show_frames: bool = False,
    show_streams: bool = False,
    show_format: bool = False,
) -> Dict[str, Any]:
    """Small ffprobe JSON helper.

    Note: for frame-level metadata (e.g. HDR side_data_list) you typically need -show_frames.
    """
    require_tool("ffprobe")
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-v", "error",
        "-select_streams", "v:0",
        "-of", "json",
    ]
    if show_frames:
        cmd.append("-show_frames")
    if show_streams:
        cmd.append("-show_streams")
    if show_format:
        cmd.append("-show_format")
    if read_intervals:
        cmd += ["-read_intervals", read_intervals]
    cmd += ["-show_entries", show_entries, str(video_path)]
    cp = run_checked(cmd)
    return json.loads(cp.stdout.decode("utf-8", errors="replace"))


def normalize_matrix(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    v = val.strip().lower()
    # ffprobe sometimes outputs bt2020nc
    if v == "bt2020nc":
        return "bt2020-ncl"
    if v == "bt2020c":
        return "bt2020-cl"
    return v


def normalize_color_range(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    v = val.strip().lower()
    # ffprobe typically: "tv" or "pc"
    if v == "tv":
        return "studio"
    if v == "pc":
        return "full"
    # sometimes "limited"/"full"
    if v == "limited":
        return "studio"
    return v


def normalize_chroma_location(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    v = val.strip().lower()
    # svt-av1 uses unknown / left|vertical / topleft|colocated
    if v in ("topleft", "colocated"):
        return "topleft"
    if v in ("left", "vertical"):
        return "left"
    if v in ("center", "centre", "unspecified", "unknown"):
        return "unknown"
    return v



def _pick_hdr_side_data(side_list: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    '''
    Returns (mastering_display, content_light) from a ffprobe side_data_list, if present.

    mastering_display formatted for SVT-AV1:
      G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)

    content_light formatted for SVT-AV1:
      max_cll,max_fall
    '''
    mastering_display = None
    content_light = None

    for sd in side_list:
        sdt = (sd.get("side_data_type") or "").lower()

        if (mastering_display is None) and ("mastering display metadata" in sdt):
            rx = parse_fraction(sd.get("red_x"))
            ry = parse_fraction(sd.get("red_y"))
            gx = parse_fraction(sd.get("green_x"))
            gy = parse_fraction(sd.get("green_y"))
            bx = parse_fraction(sd.get("blue_x"))
            by = parse_fraction(sd.get("blue_y"))
            wpx = parse_fraction(sd.get("white_point_x"))
            wpy = parse_fraction(sd.get("white_point_y"))
            max_l = parse_fraction(sd.get("max_luminance"))
            min_l = parse_fraction(sd.get("min_luminance"))

            if None not in (rx, ry, gx, gy, bx, by, wpx, wpy, max_l, min_l):
                mastering_display = (
                    f"G({gx:.4f},{gy:.4f})"
                    f"B({bx:.4f},{by:.4f})"
                    f"R({rx:.4f},{ry:.4f})"
                    f"WP({wpx:.4f},{wpy:.4f})"
                    f"L({max_l:.4f},{min_l:.6f})"
                )

        if (content_light is None) and ("content light level metadata" in sdt):
            max_content = to_int(sd.get("max_content"))
            max_average = to_int(sd.get("max_average"))
            if max_content is not None and max_average is not None:
                content_light = f"{max_content},{max_average}"

        if mastering_display is not None and content_light is not None:
            break

    return mastering_display, content_light


def _extract_hdr10_side_data(video_path: Path, scan_frames: int) -> Tuple[Optional[str], Optional[str]]:
    '''
    Try to locate HDR10 static side data (SMPTE ST 2086 + MaxCLL/MaxFALL) using ffprobe.

    These are conveyed via SEI messages in HEVC bitstreams, and may not appear on the very first frame.
    We scan the first N frames and also fall back to stream-level side_data_list.
    '''
    mastering_display = None
    content_light = None

    # Frame-level scan (first N frames)
    frames_obj = ffprobe_json(
        video_path,
        "frame=side_data_list",
        read_intervals=f"%+#{max(1, int(scan_frames))}",
        show_frames=True,
    )
    for fr in (frames_obj.get("frames") or []):
        side_list = fr.get("side_data_list") or []
        md, cll = _pick_hdr_side_data(side_list)
        mastering_display = mastering_display or md
        content_light = content_light or cll
        if mastering_display and content_light:
            return mastering_display, content_light

    # Stream-level fallback (some demuxers expose this here)
    streams_obj = ffprobe_json(
        video_path,
        "stream=side_data_list",
        show_streams=True,
    )
    for st in (streams_obj.get("streams") or []):
        side_list = st.get("side_data_list") or []
        md, cll = _pick_hdr_side_data(side_list)
        mastering_display = mastering_display or md
        content_light = content_light or cll
        if mastering_display and content_light:
            break

    return mastering_display, content_light

def extract_hdr10_static(video_path: Path, *, scan_frames: int = 120) -> Hdr10Static:
    # Stream-level color description
    streams = ffprobe_json(
        video_path,
        "stream=color_primaries,color_transfer,color_space,color_range,chroma_location",
        show_streams=True,
    ).get("streams") or []

    s0 = streams[0] if streams else {}
    color_primaries = (s0.get("color_primaries") or None)
    transfer = (s0.get("color_transfer") or None)
    matrix = normalize_matrix(s0.get("color_space") or None)
    color_range = normalize_color_range(s0.get("color_range") or None)
    chroma_pos = normalize_chroma_location(s0.get("chroma_location") or None)

    mastering_display, content_light = _extract_hdr10_side_data(video_path, scan_frames=int(scan_frames))

    return Hdr10Static(
        color_primaries=(str(color_primaries) if color_primaries else None),
        transfer_characteristics=(str(transfer) if transfer else None),
        matrix_coefficients=(str(matrix) if matrix else None),
        color_range=(str(color_range) if color_range else None),
        chroma_sample_position=(str(chroma_pos) if chroma_pos else None),
        mastering_display=mastering_display,
        content_light=content_light,
    )


# --------------------------
# HDR10+ extraction and chunking via hdr10plus_tool
# --------------------------

def extract_hdr10plus_json(source: Path, out_json: Path, *, video_stream: str, skip_reorder: bool) -> Path:
    require_tool("hdr10plus_tool")
    require_tool("ffmpeg")
    codec = ffprobe_json(source, "stream=codec_name").get("streams", [{}])[0].get("codec_name", "")
    if str(codec).lower() != "hevc":
        raise RuntimeError(f"HDR10+ extraction expects HEVC source video. codec={codec!r}")

    out_json.parent.mkdir(parents=True, exist_ok=True)

    # hdr10plus_tool supports MKV or elementary stream directly.
    suffix = source.suffix.lower()
    if suffix in {".mkv", ".hevc", ".h265", ".265"}:
        cmd = ["hdr10plus_tool"]
        if skip_reorder:
            cmd += ["--skip-reorder"]
        cmd += ["extract", str(source), "-o", str(out_json)]
        run_checked(cmd)
        return out_json

    # Otherwise: ffmpeg pipe -> hdr10plus_tool extract -
    producer = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(source),
        "-map", video_stream,
        "-an", "-sn", "-dn",
        "-c:v", "copy",
        "-bsf:v", "hevc_mp4toannexb",
        "-f", "hevc",
        "-",
    ]
    consumer = ["hdr10plus_tool"]
    if skip_reorder:
        consumer += ["--skip-reorder"]
    consumer += ["extract", "-o", str(out_json), "-"]
    pipe_checked(producer, consumer)
    if not out_json.exists() or out_json.stat().st_size == 0:
        raise RuntimeError(f"hdr10plus_tool produced empty output: {out_json}")
    return out_json


def make_hdr10plus_chunk(
    full_json: Path,
    start: int,
    end: int,
    total_frames: int,
    chunk_path: Path,
    edits_path: Path,
) -> bool:
    """
    Create a chunk-local HDR10+ JSON with hdr10plus_tool editor by removing frames outside [start, end).
    """
    if end <= start:
        return False
    total_frames = max(total_frames, 0)

    remove_ranges: List[str] = []
    if start > 0:
        remove_ranges.append(f"0-{start-1}")
    if total_frames > 0 and end < total_frames:
        remove_ranges.append(f"{end}-{total_frames-1}")

    edits = {"remove": remove_ranges, "duplicate": []}
    edits_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(edits_path, edits)

    cmd = ["hdr10plus_tool", "editor", str(full_json), "-j", str(edits_path), "-o", str(chunk_path)]
    run_checked(cmd)
    return chunk_path.exists() and chunk_path.stat().st_size > 0


# --------------------------
# Dolby Vision RPU extraction and chunking (AnnexB split)
# --------------------------

def split_annexb_nals(data: bytes) -> List[bytes]:
    """
    Split AnnexB stream into NAL units. Supports 3 and 4 byte start codes:
      00 00 01
      00 00 00 01
    Returns a list of NAL units, each starting with the start code.
    """
    if not data:
        return []

    starts: List[int] = []
    i = 0
    while True:
        j = data.find(b"\x00\x00\x01", i)
        if j == -1:
            break
        start = j - 1 if j > 0 and data[j - 1] == 0x00 else j
        if not starts or start != starts[-1]:
            starts.append(start)
        i = j + 3

    if not starts:
        return []

    nals: List[bytes] = []
    for idx, s in enumerate(starts):
        e = starts[idx + 1] if idx + 1 < len(starts) else len(data)
        chunk = data[s:e]
        if chunk:
            nals.append(chunk)
    return nals


def extract_dovi_rpu(source: Path, out_rpu: Path, *, video_stream: str) -> Path:
    require_tool("dovi_tool")
    require_tool("ffmpeg")
    codec = ffprobe_json(source, "stream=codec_name").get("streams", [{}])[0].get("codec_name", "")
    if str(codec).lower() != "hevc":
        raise RuntimeError(f"Dolby Vision RPU extraction expects HEVC source video. codec={codec!r}")

    out_rpu.parent.mkdir(parents=True, exist_ok=True)

    # Try direct first (works for MKV / HEVC elementary streams; dovi_tool also supports MKV directly)
    suffix = source.suffix.lower()
    if suffix in {".mkv", ".hevc", ".h265", ".265"}:
        try:
            run_checked(["dovi_tool", "extract-rpu", str(source), "-o", str(out_rpu)])
            if out_rpu.exists() and out_rpu.stat().st_size > 0:
                return out_rpu
        except Exception:
            LOG.warning("Direct dovi_tool extract-rpu failed; falling back to ffmpeg pipe...")

    # ffmpeg -> dovi_tool (raw HEVC AnnexB)
    producer = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(source),
        "-map", video_stream,
        "-an", "-sn", "-dn",
        "-c:v", "copy",
        "-bsf:v", "hevc_mp4toannexb",
        "-f", "hevc",
        "-",
    ]
    consumer = ["dovi_tool", "extract-rpu", "-", "-o", str(out_rpu)]
    pipe_checked(producer, consumer)
    if not out_rpu.exists() or out_rpu.stat().st_size == 0:
        raise RuntimeError(f"dovi_tool produced empty output: {out_rpu}")
    return out_rpu


def parse_rpu_frames(rpu_path: Path) -> List[bytes]:
    data = rpu_path.read_bytes()
    frames = split_annexb_nals(data)
    if not frames:
        raise RuntimeError("Could not parse RPU AnnexB stream (no start codes found).")
    return frames


def make_rpu_chunk(frames: List[bytes], start: int, end: int) -> bytes:
    """
    start/end follow scenes.json: end is EXCLUSIVE.
    """
    if start < 0 or end < 0 or end <= start:
        return b""
    if start >= len(frames):
        return b""
    actual_end = min(end, len(frames))
    if actual_end <= start:
        return b""
    return b"".join(frames[start:actual_end])


# --------------------------
# Scene patching
# --------------------------

def iter_scene_sections(scenes_data: Dict[str, Any]) -> List[str]:
    # av1an uses "scenes" and also "split_scenes". In your case they are duplicates.
    out = []
    for key in ("scenes", "split_scenes"):
        if isinstance(scenes_data.get(key), list):
            out.append(key)
    return out


def patch_hdr10_static_inplace(
    scenes_data: Dict[str, Any],
    static: Hdr10Static,
    *,
    add_enable_hdr: bool,
) -> int:
    keys = iter_scene_sections(scenes_data)
    patched = 0

    for key in keys:
        for scene in scenes_data.get(key, []):
            if not isinstance(scene, dict):
                continue
            zo = scene.get("zone_overrides")
            if not isinstance(zo, dict):
                continue
            params = zo.get("video_params")
            if not isinstance(params, list):
                continue

            new_params = params[:]
            # Enable HDR metadata writing if requested
            if add_enable_hdr:
                new_params = upsert_flag_value(new_params, "--enable-hdr", "1")

            if static.color_primaries:
                new_params = upsert_flag_value(new_params, "--color-primaries", static.color_primaries)
            if static.transfer_characteristics:
                new_params = upsert_flag_value(new_params, "--transfer-characteristics", static.transfer_characteristics)
            if static.matrix_coefficients:
                new_params = upsert_flag_value(new_params, "--matrix-coefficients", static.matrix_coefficients)
            if static.color_range:
                new_params = upsert_flag_value(new_params, "--color-range", static.color_range)
            if static.chroma_sample_position:
                new_params = upsert_flag_value(new_params, "--chroma-sample-position", static.chroma_sample_position)
            if static.mastering_display:
                new_params = upsert_flag_value(new_params, "--mastering-display", static.mastering_display)
            if static.content_light:
                new_params = upsert_flag_value(new_params, "--content-light", static.content_light)

            if new_params != params:
                zo["video_params"] = new_params
                patched += 1

    return patched


def patch_per_scene_file_flag(
    scenes_data: Dict[str, Any],
    flag: str,
    path_map: Dict[Tuple[int, int], Path],
) -> int:
    """
    Patch scenes_data[*].zone_overrides.video_params to set:
      flag <path_map[(start,end)]>
    where start/end are (start_frame,end_frame) from the scene item.
    """
    keys = iter_scene_sections(scenes_data)
    patched = 0

    for key in keys:
        for scene in scenes_data.get(key, []):
            if not isinstance(scene, dict):
                continue
            start = int(scene.get("start_frame", -1))
            end = int(scene.get("end_frame", -1))
            if start < 0 or end < 0:
                continue

            zo = scene.get("zone_overrides")
            if not isinstance(zo, dict):
                continue
            params = zo.get("video_params")
            if not isinstance(params, list):
                continue

            p = path_map.get((start, end))
            if not p:
                continue

            new_params = upsert_flag_value(params, flag, str(p))
            if new_params != params:
                zo["video_params"] = new_params
                patched += 1

    return patched


def unique_scene_ranges(scenes_data: Dict[str, Any]) -> List[Tuple[int, int]]:
    seen: set[Tuple[int, int]] = set()
    ranges: List[Tuple[int, int]] = []
    for key in iter_scene_sections(scenes_data):
        for scene in scenes_data.get(key, []):
            if not isinstance(scene, dict):
                continue
            start = int(scene.get("start_frame", -1))
            end = int(scene.get("end_frame", -1))
            if start < 0 or end < 0 or end <= start:
                continue
            tup = (start, end)
            if tup not in seen:
                seen.add(tup)
                ranges.append(tup)
    ranges.sort()
    return ranges


# --------------------------
# Main
# --------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch av1an scenes.json to pass HDR10/HDR10+/Dolby Vision metadata per-scene (svt-av1 flags)."
    )

    parser.add_argument("--source", required=True, help="Source file (used to extract all metadata).")
    parser.add_argument("--scenes", required=True, help="Input av1an scenes.json (generated by av1an --sc-only).")

    parser.add_argument("--workdir", default="hdrmeta_work", help="Work directory for extracted/chunk metadata.")
    parser.add_argument("--output", default="scenes_patched.json", help="Output scenes.json filename (inside workdir).")

    parser.add_argument("--video-stream", default="0:v:0", help="ffmpeg -map selector for the video stream (default: 0:v:0).")

    parser.add_argument("--no-hdr10", action="store_true", help="Do not inject HDR10 static metadata.")
    parser.add_argument("--no-hdr10plus", action="store_true", help="Do not process HDR10+ dynamic metadata.")
    parser.add_argument("--no-dv", action="store_true", help="Do not process Dolby Vision dynamic metadata.")

    parser.add_argument("--hdr10plus-skip-reorder", action="store_true", help="Pass --skip-reorder to hdr10plus_tool extract.")
    parser.add_argument("--dv-frame-offset", type=int, default=0, help="Shift DV RPU frame indexing by N (can be negative).")

    parser.add_argument("--hdr10-add-enable-hdr", action="store_true", help="Also add --enable-hdr 1 to every scene (recommended).")
    parser.add_argument("--hdr10-scan-frames", type=int, default=120, help="How many initial frames to scan with ffprobe to find Mastering Display / Content Light Level side data (default: 120).")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")

    args = parser.parse_args()
    setup_logging(args.verbose)

    source = Path(args.source).resolve()
    scenes_in = Path(args.scenes).resolve()
    workdir = Path(args.workdir).resolve()
    marker = marker_path(workdir)
    if marker.exists():
        LOG.info("Skip: marker exists: %s", marker)
        return
    workdir.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(source)
    if not scenes_in.exists():
        raise FileNotFoundError(scenes_in)

    scenes_data = load_json(scenes_in)
    if not isinstance(scenes_data, dict):
        raise RuntimeError("Invalid scenes.json: root must be an object/dict.")

    total_frames = int(scenes_data.get("frames", 0) or 0)
    LOG.info("Source: %s", source)
    LOG.info("Scenes: %s", scenes_in)
    LOG.info("Total frames (from scenes.json): %d", total_frames)

    # Determine scene ranges once (for chunk caches)
    ranges = unique_scene_ranges(scenes_data)
    LOG.info("Unique scene ranges: %d", len(ranges))

    # HDR10 static
    if not args.no_hdr10:
        LOG.info("Extracting HDR10 static metadata with ffprobe...")
        static = extract_hdr10_static(source, scan_frames=int(args.hdr10_scan_frames))
        LOG.info("HDR10 static: primaries=%s transfer=%s matrix=%s range=%s chroma_pos=%s mastering=%s cll=%s",
                 static.color_primaries, static.transfer_characteristics, static.matrix_coefficients,
                 static.color_range, static.chroma_sample_position, bool(static.mastering_display), bool(static.content_light))
        patched = patch_hdr10_static_inplace(
            scenes_data,
            static,
            add_enable_hdr=bool(args.hdr10_add_enable_hdr),
        )
        LOG.info("HDR10 static patched scenes: %d", patched)
    else:
        LOG.info("HDR10 static injection: skipped (--no-hdr10).")

    # HDR10+ dynamic
    hdr10plus_map: Dict[Tuple[int, int], Path] = {}
    if not args.no_hdr10plus:
        try:
            full_hdr10plus = extract_hdr10plus_json(
                source,
                workdir / "hdr10plus_full.json",
                video_stream=args.video_stream,
                skip_reorder=bool(args.hdr10plus_skip_reorder),
            )
            LOG.info("HDR10+ metadata extracted: %s", full_hdr10plus)

            frag_dir = workdir / "hdr10plus_fragments"
            edits_dir = workdir / "hdr10plus_edits"
            frag_dir.mkdir(parents=True, exist_ok=True)
            edits_dir.mkdir(parents=True, exist_ok=True)

            for (start, end) in ranges:
                chunk_path = frag_dir / f"chunk_{start:06d}_{end:06d}.json"
                edits_path = edits_dir / f"edits_{start:06d}_{end:06d}.json"

                ok = make_hdr10plus_chunk(
                    full_json=full_hdr10plus,
                    start=start,
                    end=end,
                    total_frames=total_frames,
                    chunk_path=chunk_path,
                    edits_path=edits_path,
                )
                if ok:
                    hdr10plus_map[(start, end)] = chunk_path
                else:
                    LOG.warning("HDR10+ chunk skipped for %d-%d", start, end)

            patched = patch_per_scene_file_flag(scenes_data, "--hdr10plus-json", hdr10plus_map)
            LOG.info("HDR10+ patched scenes: %d", patched)

        except Exception as e:
            LOG.warning("HDR10+ processing failed; continuing without HDR10+: %s", e)
    else:
        LOG.info("HDR10+ processing: skipped (--no-hdr10plus).")

    # Dolby Vision dynamic
    dv_map: Dict[Tuple[int, int], Path] = {}
    if not args.no_dv:
        try:
            full_rpu = extract_dovi_rpu(
                source,
                workdir / "dv_full.rpu",
                video_stream=args.video_stream,
            )
            LOG.info("DV RPU extracted: %s", full_rpu)

            rpu_frames = parse_rpu_frames(full_rpu)
            LOG.info("DV RPU NAL units: %d", len(rpu_frames))
            if total_frames and len(rpu_frames) != total_frames:
                LOG.warning(
                    "DV WARNING: RPU NAL count (%d) != scenes.json frames (%d). "
                    "Slicing uses min(end, rpu_len) after offset.",
                    len(rpu_frames), total_frames,
                )

            frag_dir = workdir / "rpu_fragments"
            frag_dir.mkdir(parents=True, exist_ok=True)

            offset = int(args.dv_frame_offset or 0)

            for (start, end) in ranges:
                start2 = start + offset
                end2 = end + offset
                chunk_bytes = make_rpu_chunk(rpu_frames, start2, end2)
                if not chunk_bytes:
                    LOG.warning("No DV RPU data for %d-%d (after offset=%d); skipping.", start, end, offset)
                    continue
                chunk_path = frag_dir / f"chunk_{start:06d}_{end:06d}.rpu"
                chunk_path.write_bytes(chunk_bytes)
                dv_map[(start, end)] = chunk_path

            patched = patch_per_scene_file_flag(scenes_data, "--dolby-vision-rpu", dv_map)
            LOG.info("DV patched scenes: %d", patched)

        except Exception as e:
            LOG.warning("Dolby Vision processing failed; continuing without DV: %s", e)
    else:
        LOG.info("Dolby Vision processing: skipped (--no-dv).")

    # Save patched scenes
    out_path = (workdir / args.output).resolve()
    dump_json(out_path, scenes_data)
    write_marker(workdir)

    LOG.info("---------------------------------------------------")
    LOG.info("Done.")
    LOG.info("Patched scenes.json: %s", out_path)
    LOG.info("Workdir: %s", workdir)
    if hdr10plus_map:
        LOG.info("HDR10+ fragments: %s", workdir / "hdr10plus_fragments")
    if dv_map:
        LOG.info("DV fragments: %s", workdir / "rpu_fragments")
    LOG.info("---------------------------------------------------")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        LOG.error("Fatal: %s", exc)
        sys.exit(2)
