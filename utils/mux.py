#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mux.py - final mux stage for file .plan pipeline.

Inputs:
  --plan   file .plan

Uses (if present):
  workdir/video/video-final.mkv
  workdir/00_meta/audio_manifest.json
  workdir/00_meta/demux_manifest.json
  workdir/chapters/chapters.xml

Output:
  <source_dir>/<basename>-av1.mkv

Requires:
  mkvmerge in PATH (MKVToolNix)
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plan_model import resolve_file_plan


# ---------------------------
# utils
# ---------------------------

STATE_DIR_NAME = ".state"
MUX_MARKER = "MUX_DONE"
RUNNER_MANAGED_STATE_ENV = "PBBATCH_RUNNER_MANAGED_STATE"
CRF_METADATA_STEP = 1.0
SOURCE_BITRATE_TIMEOUT_SEC = 30.0
SOURCE_BITRATE_TIMEOUT_CHECK_EVERY = 2048

class TeeStream:
    def __init__(self, stream, log_file) -> None:
        self._stream = stream
        self._log = log_file

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


def setup_logging(log_path: str, workdir: Optional[Path] = None) -> None:
    if not log_path:
        return
    p = Path(log_path)
    if not p.is_absolute() and workdir is not None:
        p = workdir / p
    p.parent.mkdir(parents=True, exist_ok=True)
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    log_fh = p.open("a", encoding=enc, errors="replace")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        log_fh.write(f"=== START mux {ts} ===\n")
        log_fh.flush()
    except Exception:
        pass
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    tee_out = TeeStream(orig_stdout, log_fh)
    tee_err = TeeStream(orig_stderr, log_fh)
    sys.stdout = tee_out
    sys.stderr = tee_err

    def _cleanup() -> None:
        ts_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            log_fh.write(f"=== END mux {ts_end} ===\n")
            log_fh.flush()
        except Exception:
            pass
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        tee_out.close_log()
        tee_err.close_log()

    atexit.register(_cleanup)

def eprint(*a: Any) -> None:
    print(*a, file=sys.stderr)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def marker_path(workdir: Path) -> Path:
    return workdir / STATE_DIR_NAME / MUX_MARKER

def write_marker(workdir: Path) -> None:
    p = marker_path(workdir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("ok\n", encoding="utf-8")

def runner_managed_state() -> bool:
    return os.environ.get(RUNNER_MANAGED_STATE_ENV, "").strip().lower() in ("1", "true", "yes", "on")

def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))

def write_json(p: Path, obj: Any) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def which_or(name: str) -> str:
    return shutil.which(name) or name

def norm_type(t: str) -> str:
    v = (t or "").strip().lower()
    if v.startswith("vid") or v == "video":
        return "video"
    if v.startswith("aud") or v == "audio":
        return "audio"
    if v.startswith("sub") or v == "subtitle":
        return "sub"
    return v

def is_skip(status: str) -> bool:
    return (status or "").strip().upper() == "SKIP"

def is_copy(status: str) -> bool:
    return (status or "").strip().upper() == "COPY"

def is_edit(status: str) -> bool:
    return (status or "").strip().upper() == "EDIT"

def sanitize_for_cmd_log(s: str) -> str:
    # only for printing command; doesn't affect execution
    return s.replace("\n", " ").replace("\r", " ")

def run_cmd(cmd: List[str]) -> None:
    print("[cmd]", " ".join(sanitize_for_cmd_log(x) for x in cmd))
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if p.stdout:
        print(p.stdout, end="" if p.stdout.endswith("\n") else "\n")
    if p.returncode != 0:
        raise RuntimeError(f"mkvmerge_failed_rc_{p.returncode}")

def parse_frame_rate(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if v > 0 else None
    s = str(value).strip()
    if not s:
        return None
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            num = float(a)
            den = float(b)
            if den == 0:
                return None
            v = num / den
            return v if v > 0 else None
        except Exception:
            return None
    try:
        v = float(s)
        return v if v > 0 else None
    except Exception:
        return None

def get_stream_fps(source: Path) -> Optional[float]:
    if not shutil.which("ffprobe"):
        return None
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate",
        "-of", "csv=p=0",
        str(source),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        return None
    for line in p.stdout.splitlines():
        parts = [x.strip() for x in line.split(",") if x.strip()]
        for part in parts:
            fps = parse_frame_rate(part)
            if fps:
                return fps
    return None

def iter_frame_sizes(source: Path, *, timeout_sec: Optional[float] = None) -> Iterator[Tuple[int, Optional[float]]]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_frames",
        "-show_entries", "frame=pkt_size,pkt_duration_time",
        "-of", "csv=p=0",
        str(source),
    ]
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert p.stdout is not None
    started = time.monotonic()
    line_count = 0
    try:
        for raw in p.stdout:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",")
            try:
                size = int(parts[0]) if parts[0] else 0
            except Exception:
                continue
            dur: Optional[float] = None
            if len(parts) > 1 and parts[1]:
                try:
                    dur = float(parts[1])
                except Exception:
                    dur = None
            line_count += 1
            if (
                timeout_sec is not None
                and timeout_sec > 0
                and line_count % SOURCE_BITRATE_TIMEOUT_CHECK_EVERY == 0
                and (time.monotonic() - started) > timeout_sec
            ):
                p.kill()
                raise TimeoutError(
                    f"source bitrate ffprobe scan timed out after {time.monotonic() - started:.1f}s"
                )
            yield size, dur
    finally:
        try:
            p.stdout.close()
        except Exception:
            pass
        try:
            p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()

    if p.returncode != 0:
        raise RuntimeError(f"source bitrate ffprobe scan failed rc={p.returncode}")

def compute_source_bitrates(
    scenes: List[Dict[str, Any]],
    source: Path,
    *,
    timeout_sec: Optional[float] = SOURCE_BITRATE_TIMEOUT_SEC,
) -> List[Optional[float]]:
    count = len(scenes)
    if count == 0:
        return []
    if not shutil.which("ffprobe"):
        print("[warn] ffprobe not found; source bitrate skipped.")
        return [None] * count

    ranges: List[Tuple[int, int]] = []
    for sc in scenes:
        try:
            st = int(sc.get("start_frame"))
            en = int(sc.get("end_frame"))
        except Exception:
            st, en = 0, 0
        ranges.append((st, en))

    sizes = [0] * count
    durations = [0.0] * count
    frames = [0] * count

    scene_idx = 0
    frame_idx = 0
    started = time.monotonic()
    if timeout_sec is None or timeout_sec <= 0:
        print(f"[info] source bitrate scan: {source.name} (timeout=disabled)")
    else:
        print(f"[info] source bitrate scan: {source.name} (timeout={timeout_sec:.1f}s)")
    try:
        for size, dur in iter_frame_sizes(source, timeout_sec=timeout_sec):
            while scene_idx < count and frame_idx >= ranges[scene_idx][1]:
                scene_idx += 1
            if scene_idx >= count:
                break
            if frame_idx >= ranges[scene_idx][0]:
                sizes[scene_idx] += int(size)
                if dur is not None and dur > 0:
                    durations[scene_idx] += float(dur)
                frames[scene_idx] += 1
            frame_idx += 1
    except TimeoutError as ex:
        print(f"[warn] {ex}; source_kbps skipped.")
        return [None] * count
    except Exception as ex:
        print(f"[warn] source bitrate scan failed: {ex}; source_kbps skipped.")
        return [None] * count
    print(f"[info] source bitrate scan done: frames={frame_idx} elapsed={time.monotonic() - started:.1f}s")

    fps: Optional[float] = None
    if any(d <= 0 and f > 0 for d, f in zip(durations, frames)):
        fps = get_stream_fps(source)

    out: List[Optional[float]] = []
    for i in range(count):
        if sizes[i] <= 0:
            out.append(None)
            continue
        dur = durations[i]
        if dur <= 0:
            if fps and frames[i] > 0:
                dur = frames[i] / fps
            else:
                out.append(None)
                continue
        if dur <= 0:
            out.append(None)
            continue
        kbps = (sizes[i] * 8.0) / dur / 1000.0
        out.append(kbps)
    return out

def compute_chunk_bitrates(av1an_temp: Path, scene_count: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * scene_count
    chunks_path = av1an_temp / "chunks.json"
    encode_dir = av1an_temp / "encode"
    if not chunks_path.exists() or not encode_dir.exists():
        return out
    try:
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    except Exception:
        return out
    if not isinstance(chunks, list):
        return out
    fps_default: Optional[float] = None
    for ch in chunks:
        fps_default = parse_frame_rate(ch.get("frame_rate"))
        if fps_default:
            break
    for ch in chunks:
        if not isinstance(ch, dict) or "index" not in ch:
            continue
        try:
            idx = int(ch.get("index"))
        except Exception:
            continue
        if idx < 0 or idx >= scene_count:
            continue
        try:
            st = int(ch.get("start_frame"))
            en = int(ch.get("end_frame"))
        except Exception:
            continue
        fps = parse_frame_rate(ch.get("frame_rate")) or fps_default
        if not fps or fps <= 0:
            continue
        duration = (en - st) / fps
        if duration <= 0:
            continue
        ivf_path = encode_dir / f"{idx:05d}.ivf"
        if not ivf_path.exists():
            continue
        size = ivf_path.stat().st_size
        out[idx] = (size * 8.0) / duration / 1000.0
    return out

def apply_bitrates(scene_list: List[Dict[str, Any]], values: List[Optional[float]], key: str) -> bool:
    changed = False
    for i, sc in enumerate(scene_list):
        if i >= len(values):
            break
        val = values[i]
        if val is None:
            continue
        meta = sc.get("pb_meta")
        if not isinstance(meta, dict):
            meta = {}
            sc["pb_meta"] = meta
        bitrate = meta.get("bitrate")
        if not isinstance(bitrate, dict):
            bitrate = {}
            meta["bitrate"] = bitrate
        new_val = round(float(val), 3)
        if bitrate.get(key) != new_val:
            bitrate[key] = new_val
            changed = True
    return changed

def scene_bitrate_complete(scene_list: List[Dict[str, Any]], key: str) -> bool:
    if not scene_list:
        return False
    for scene in scene_list:
        if not isinstance(scene, dict):
            return False
        meta = scene.get("pb_meta")
        if not isinstance(meta, dict):
            return False
        bitrate = meta.get("bitrate")
        if not isinstance(bitrate, dict):
            return False
        value = bitrate.get(key)
        if value in (None, ""):
            return False
        try:
            float(value)
        except Exception:
            return False
    return True

def update_scene_bitrates(
    workdir: Path,
    source: Path,
    *,
    include_source: bool = True,
    source_timeout_sec: Optional[float] = SOURCE_BITRATE_TIMEOUT_SEC,
) -> None:
    video_dir = workdir / "video"
    scenes_path = None
    for name in ("scenes-final.json", "scenes-hdr.json", "scenes.json"):
        p = video_dir / name
        if p.exists():
            scenes_path = p
            break
    if scenes_path is None:
        print("[warn] scenes json not found; bitrate update skipped.")
        return

    try:
        data = read_json(scenes_path)
    except Exception as e:
        print(f"[warn] failed to read scenes json: {e}")
        return

    scenes = data.get("scenes") if isinstance(data.get("scenes"), list) else None
    split_scenes = data.get("split_scenes") if isinstance(data.get("split_scenes"), list) else None
    base_list = scenes or split_scenes or []
    if not base_list:
        print("[warn] scenes list missing; bitrate update skipped.")
        return

    count = len(base_list)
    fast_list = compute_chunk_bitrates(video_dir / "fastpass", count)
    main_list = compute_chunk_bitrates(video_dir / "mainpass", count)
    need_source_scan = bool(include_source)
    if need_source_scan:
        source_already_present = True
        if scenes is not None and len(scenes) == count:
            source_already_present = source_already_present and scene_bitrate_complete(scenes, "source_kbps")
        if split_scenes is not None and len(split_scenes) == count:
            source_already_present = source_already_present and scene_bitrate_complete(split_scenes, "source_kbps")
        if source_already_present:
            print("[info] source bitrate already present in scenes json; source scan skipped.")
            need_source_scan = False
    source_list = (
        compute_source_bitrates(base_list, source, timeout_sec=source_timeout_sec)
        if need_source_scan
        else [None] * count
    )

    changed = False
    if scenes is not None and len(scenes) == count:
        changed |= apply_bitrates(scenes, source_list, "source_kbps")
        changed |= apply_bitrates(scenes, fast_list, "fastpass_kbps")
        changed |= apply_bitrates(scenes, main_list, "mainpass_kbps")
    if split_scenes is not None and len(split_scenes) == count:
        changed |= apply_bitrates(split_scenes, source_list, "source_kbps")
        changed |= apply_bitrates(split_scenes, fast_list, "fastpass_kbps")
        changed |= apply_bitrates(split_scenes, main_list, "mainpass_kbps")

    if changed:
        write_json(scenes_path, data)
        print(f"[ok] bitrate updated: {scenes_path}")
    else:
        print("[info] bitrate update: no changes.")

def mkvmerge_json(mkvmerge: str, source: Path) -> Dict[str, Any]:
    cmd = [mkvmerge, "-J", str(source)]
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(f"mkvmerge_J_failed_rc_{p.returncode}")
    try:
        return json.loads(p.stdout)
    except Exception as ex:
        raise RuntimeError(f"mkvmerge_J_json_parse_failed_{ex}")

def get_src_track_codec(mkvj: Dict[str, Any], track_id: int) -> str:
    for t in mkvj.get("tracks") or []:
        if int(t.get("id", -1)) == int(track_id):
            props = t.get("properties") or {}
            return str(props.get("codec_id") or "")
    return ""

def ext_from_sub_codec(codec_id: str) -> str:
    c = (codec_id or "").upper()
    if "S_TEXT/ASS" in c:
        return ".ass"
    if "S_TEXT/SSA" in c:
        return ".ssa"
    if "S_TEXT/UTF8" in c:
        return ".srt"
    if "S_TEXT/WEBVTT" in c:
        return ".vtt"
    if "S_HDMV/PGS" in c:
        return ".sup"
    if "S_VOBSUB" in c:
        return ".sub"
    return ".sub"

def mime_from_ext(path: Path) -> str:
    ext = path.suffix.lower()
    # common attachments in anime: ttf/otf, images
    if ext == ".ttf":
        return "application/x-truetype-font"
    if ext == ".otf":
        return "application/vnd.ms-opentype"
    if ext == ".jpg" or ext == ".jpeg":
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    # leave empty => mkvmerge will try to guess
    return ""

def bool_yesno(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return "yes"
    if s in ("0", "false", "no", "n", "off"):
        return "no"
    return None

def parse_bool_flag(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def strip_outer_quotes(value: str) -> str:
    s = str(value or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s

def split_cmd_tokens(value: Any) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        tokens = shlex.split(text, posix=False)
    except ValueError:
        tokens = text.split()
    return [strip_outer_quotes(tok) for tok in tokens if str(tok).strip()]

def is_param_key(tok: str) -> bool:
    t = str(tok or "").strip()
    return t.startswith("--") or t.startswith("-")

def find_last_option(tokens: List[str], key: str) -> Optional[Tuple[int, bool]]:
    for idx in range(len(tokens) - 1, -1, -1):
        if tokens[idx] != key:
            continue
        has_value = (idx + 1 < len(tokens)) and (not is_param_key(tokens[idx + 1]))
        return idx, has_value
    return None

def apply_override(base_tokens: List[str], override_tokens: List[str]) -> List[str]:
    i = 0
    while i < len(override_tokens):
        tok = override_tokens[i]
        if not is_param_key(tok):
            i += 1
            continue

        key = tok
        has_value = (i + 1 < len(override_tokens)) and (not is_param_key(override_tokens[i + 1]))
        val = override_tokens[i + 1] if has_value else None

        loc = find_last_option(base_tokens, key)
        if loc is None:
            base_tokens.append(key)
            if val is not None:
                base_tokens.append(val)
        else:
            key_idx, base_has_value = loc
            if val is None:
                if base_has_value:
                    del base_tokens[key_idx + 1]
            else:
                if base_has_value:
                    base_tokens[key_idx + 1] = val
                else:
                    base_tokens.insert(key_idx + 1, val)

        i += 2 if has_value else 1

    return base_tokens

def strip_param_tokens(tokens: List[str], keys: List[str]) -> List[str]:
    keys_set = {str(key) for key in keys}
    out: List[str] = []
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if tok in keys_set:
            has_value = (idx + 1 < len(tokens)) and (not is_param_key(tokens[idx + 1]))
            idx += 2 if has_value else 1
            continue
        out.append(tok)
        idx += 1
    return out

def parse_decimal_value(value: Any) -> Optional[Decimal]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return Decimal(text)
    except Exception:
        return None

def format_number(value: Any) -> str:
    dec = parse_decimal_value(value)
    if dec is None:
        return str(value)
    normalized = dec.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"

def quantize_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    units = (value / step).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return units * step

def extract_scene_crf(scene: Dict[str, Any]) -> Optional[Decimal]:
    zo = scene.get("zone_overrides")
    if not isinstance(zo, dict):
        return None
    vp = zo.get("video_params")
    if isinstance(vp, list):
        tokens = [strip_outer_quotes(str(item)) for item in vp]
    elif isinstance(vp, str):
        tokens = split_cmd_tokens(vp)
    else:
        return None
    loc = find_last_option(tokens, "--crf")
    if loc is None:
        return None
    idx, has_value = loc
    if not has_value or idx + 1 >= len(tokens):
        return None
    return parse_decimal_value(tokens[idx + 1])

def build_crf_deviation_text(workdir: Path) -> Optional[str]:
    scenes_path = workdir / "video" / "scenes-final.json"
    if not scenes_path.exists():
        print(f"[warn] scenes-final.json not found; CRF Deviation skipped: {scenes_path}")
        return None

    try:
        data = read_json(scenes_path)
    except Exception as exc:
        print(f"[warn] failed to read scenes-final.json: {exc}")
        return None

    scenes = data.get("scenes") if isinstance(data.get("scenes"), list) else None
    if scenes is None:
        scenes = data.get("split_scenes") if isinstance(data.get("split_scenes"), list) else None
    if not scenes:
        return None

    step = parse_decimal_value(CRF_METADATA_STEP) or Decimal("1.0")
    buckets: Dict[Decimal, int] = {}
    total_frames = 0
    skipped_frames = 0

    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        try:
            start_frame = int(scene.get("start_frame"))
            end_frame = int(scene.get("end_frame"))
        except Exception:
            continue
        frames = max(0, end_frame - start_frame)
        if frames <= 0:
            continue
        crf = extract_scene_crf(scene)
        if crf is None:
            skipped_frames += frames
            continue
        bucket = quantize_step(crf, step)
        buckets[bucket] = buckets.get(bucket, 0) + frames
        total_frames += frames

    if skipped_frames:
        print(f"[warn] CRF Deviation skipped {skipped_frames} frame(s) without scene CRF.")
    if total_frames <= 0 or not buckets:
        return None

    parts: List[str] = []
    for crf in sorted(buckets.keys()):
        pct = (Decimal(buckets[crf]) * Decimal("100")) / Decimal(total_frames)
        pct_text = format_number(pct.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
        parts.append(f"{format_number(crf)}:{pct_text}%")
    return " | ".join(parts)

def query_encoder_metadata(preferred_encoder: str) -> Optional[str]:
    normalized = str(preferred_encoder or "").strip().lower().replace("_", "-")
    if normalized and normalized not in ("svt-av1", "svt-av1-psy", "svt"):
        return preferred_encoder.strip()

    for exe in ("SvtAv1EncApp", "SvtAv1EncApp.exe"):
        try:
            p = subprocess.run(
                [exe, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except Exception:
            continue
        for raw in (p.stdout or "").splitlines():
            line = raw.strip()
            if line.startswith("Svt[info]:"):
                value = line[len("Svt[info]:"):].strip()
                if value:
                    return value

    fallback = preferred_encoder.strip()
    return fallback or None

def write_global_tags_xml(path: Path, tags: List[Tuple[str, str]]) -> None:
    ensure_dir(path.parent)
    root = ET.Element("Tags")
    tag_el = ET.SubElement(root, "Tag")
    for name, value in tags:
        if not value:
            continue
        simple = ET.SubElement(tag_el, "Simple")
        ET.SubElement(simple, "Name").text = name
        ET.SubElement(simple, "String").text = value
    try:
        ET.indent(root, space="  ")
    except Exception:
        pass
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    path.write_bytes(xml_bytes)

def pick_video_track_mux(tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
    for track in tracks:
        if norm_type(str(track.get("type") or "")) != "video":
            continue
        tmux = track.get("trackMux")
        if isinstance(tmux, dict):
            return tmux
    return {}

def prepare_global_tags(
    workdir: Path,
    source: Path,
    tracks: List[Dict[str, Any]],
    *,
    encode_params_text: Optional[str] = None,
) -> Tuple[Optional[Path], List[Dict[str, str]]]:
    video_track_mux = pick_video_track_mux(tracks)
    if not parse_bool_flag(video_track_mux.get("attachEncodeInfo")):
        return None, []

    tags: List[Tuple[str, str]] = []

    encode_params = encode_params_text
    if encode_params:
        tags.append(("Encode Params", encode_params))

    encoder_info = query_encoder_metadata(str(video_track_mux.get("encoder") or ""))
    if encoder_info:
        tags.append(("Encoder", encoder_info))

    crf_deviation = build_crf_deviation_text(workdir)
    if crf_deviation:
        tags.append(("CRF Deviation", crf_deviation))

    tags.append(("Source Name", source.name))

    note = str(video_track_mux.get("note") or "").strip()
    if note:
        tags.append(("Note", note))

    if not tags:
        return None, []

    tags_path = workdir / "00_meta" / "mux_global_tags.xml"
    write_global_tags_xml(tags_path, tags)
    print(f"[ok] global tags prepared: {tags_path}")
    return tags_path, [{"name": name, "value": value} for name, value in tags]


# ---------------------------
# plan loading
# ---------------------------

def load_audio_manifest(workdir: Path) -> Optional[Dict[str, Any]]:
    p = workdir / "00_meta" / "audio_manifest.json"
    return read_json(p) if p.exists() else None

def load_demux_manifest(workdir: Path) -> Optional[Dict[str, Any]]:
    p = workdir / "00_meta" / "demux_manifest.json"
    return read_json(p) if p.exists() else None

def pick_chapters_path(workdir: Path, demux_manifest: Optional[Dict[str, Any]]) -> Optional[Path]:
    # prefer explicit chapters.xml in workdir/chapters
    p1 = workdir / "chapters" / "chapters.xml"
    if p1.exists() and p1.stat().st_size > 0:
        return p1
    if demux_manifest and isinstance(demux_manifest.get("chapters"), dict):
        ch = demux_manifest["chapters"].get("path")
        if ch:
            p = Path(ch)
            if not p.is_absolute():
                p = workdir / p
            if p.exists() and p.stat().st_size > 0:
                return p
    return None

def pick_attachments(workdir: Path, demux_manifest: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not demux_manifest:
        return out
    atts = demux_manifest.get("attachments")
    if not isinstance(atts, list):
        return out
    for a in atts:
        if not isinstance(a, dict):
            continue
        p = a.get("path")
        if not p:
            continue
        pp = Path(p)
        if not pp.is_absolute():
            pp = workdir / pp
        if not pp.exists() or pp.stat().st_size == 0:
            continue
        out.append({
            "path": str(pp),
            "file_name": a.get("file_name") or pp.name,
            "content_type": a.get("content_type") or "",
        })
    return out

def pick_extracted_subs(workdir: Path, demux_manifest: Optional[Dict[str, Any]]) -> Dict[int, Path]:
    """
    Returns map: trackId -> extracted subtitle file path
    """
    res: Dict[int, Path] = {}
    if not demux_manifest:
        return res
    subs = demux_manifest.get("subs")
    if not isinstance(subs, list):
        return res
    for s in subs:
        if not isinstance(s, dict):
            continue
        tid = s.get("trackId")
        p = s.get("path")
        if tid is None or not p:
            continue
        try:
            tid_i = int(tid)
        except Exception:
            continue
        pp = Path(p)
        if not pp.is_absolute():
            pp = workdir / pp
        if pp.exists() and pp.stat().st_size >= 1:
            res[tid_i] = pp
    return res

def pick_audio_outputs(workdir: Path, audio_manifest: Optional[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Returns map: trackId -> {"path": Path, "mux_delay_ms": int}
    """
    res: Dict[int, Dict[str, Any]] = {}
    if not audio_manifest:
        return res
    outs = audio_manifest.get("outputs")
    if not isinstance(outs, list):
        return res
    for o in outs:
        if not isinstance(o, dict):
            continue
        if str(o.get("role") or "") != "primary":
            continue
        try:
            tid = int(o.get("srcTrackId"))
        except Exception:
            continue
        op = o.get("outPath")
        if not op:
            continue
        p = Path(str(op))
        if not p.is_absolute():
            p = workdir / p
        if p.exists() and p.stat().st_size > 0:
            try:
                mux_delay_ms = int(o.get("mux_delay_ms") or 0)
            except Exception:
                mux_delay_ms = 0
            res[tid] = {"path": p, "mux_delay_ms": mux_delay_ms}
    return res


# ---------------------------
# mkvmerge args building
# ---------------------------

def add_track_meta_args(args: List[str], track_mux: Dict[str, Any], track_index_in_file: int = 0) -> None:
    """
    Adds mkvmerge metadata options for a specific track index in the *next input file*.
    For external single-track files, index is 0.
    """
    # language
    lang = track_mux.get("lang")
    if lang:
        args += ["--language", f"{track_index_in_file}:{lang}"]

    # track name
    name = track_mux.get("name")
    if name:
        args += ["--track-name", f"{track_index_in_file}:{name}"]

    # default / forced (strings "true"/"false" from GUI)
    d = bool_yesno(track_mux.get("default"))
    if d:
        args += ["--default-track", f"{track_index_in_file}:{d}"]

    f = bool_yesno(track_mux.get("forced"))
    if f:
        args += ["--forced-track", f"{track_index_in_file}:{f}"]


def build_mux_command(
    mkvmerge: str,
    source: Path,
    workdir: Path,
    tracks: List[Dict[str, Any]],
    global_tags_path: Optional[Path] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    mkvj = mkvmerge_json(mkvmerge, source)

    base = source.stem
    out_path = source.parent / f"{base}-av1.mkv"

    demux_manifest = load_demux_manifest(workdir)
    audio_manifest = load_audio_manifest(workdir)

    extracted_subs = pick_extracted_subs(workdir, demux_manifest)
    audio_out = pick_audio_outputs(workdir, audio_manifest)
    chapters = pick_chapters_path(workdir, demux_manifest)
    attachments = pick_attachments(workdir, demux_manifest)

    # Decide video input
    video_final = workdir / "video" / "video-final.mkv"
    have_video_final = video_final.exists() and video_final.stat().st_size > 0

    # Track lists from plan
    video_tracks = [t for t in tracks if norm_type(str(t.get("type") or "")) == "video" and not is_skip(str(t.get("trackStatus") or ""))]
    audio_tracks = [t for t in tracks if norm_type(str(t.get("type") or "")) == "audio" and not is_skip(str(t.get("trackStatus") or ""))]
    sub_tracks   = [t for t in tracks if norm_type(str(t.get("type") or "")) == "sub"   and not is_skip(str(t.get("trackStatus") or ""))]

    # mkvmerge base args
    args: List[str] = [mkvmerge, "-o", str(out_path), "--ui-language", "en"]

    # chapters
    if chapters:
        args += ["--chapters", str(chapters)]
    if global_tags_path is not None:
        args += ["--global-tags", str(global_tags_path)]

    # Start building inputs.
    plan: Dict[str, Any] = {
        "output": str(out_path),
        "video_source": None,
        "audio_sources": [],
        "sub_sources": [],
        "chapters": str(chapters) if chapters else None,
        "global_tags": str(global_tags_path) if global_tags_path is not None else None,
        "attachments": attachments,
        "fallbacks": [],
    }

    # 1) Video
    if have_video_final:
        # external video file (single video track expected)
        args += ["--no-audio", "--no-subtitles"]
        # note: track meta for video usually not needed; language/name usually irrelevant for video,
        # but you can extend here if you store it.
        args += [str(video_final)]
        plan["video_source"] = str(video_final)
    else:
        # fallback: take video from source using COPY-selected video track ids
        # If multiple, include all.
        copy_vid_ids = [int(t.get("trackId")) for t in video_tracks if is_copy(str(t.get("trackStatus") or ""))]
        if not copy_vid_ids:
            raise RuntimeError("no_video_final_and_no_copy_video_tracks")
        args += ["--no-audio", "--no-subtitles", "--video-tracks", ",".join(str(x) for x in copy_vid_ids)]
        args += [str(source)]
        plan["video_source"] = f"source:{copy_vid_ids}"
        plan["fallbacks"].append("video_from_source")

    # 2) Audio
    # Primary approach: use audio_manifest outputs for each planned audio track
    # If missing for some track, fallback to source COPY.
    for t in sorted(audio_tracks, key=lambda x: int(x.get("trackId", 0))):
        tid = int(t.get("trackId"))
        status = str(t.get("trackStatus") or "")
        tmux = t.get("trackMux") or {}
        if not isinstance(tmux, dict):
            tmux = {}

        if tid in audio_out:
            ao = audio_out[tid]
            ap = Path(str(ao["path"]))
            try:
                mux_delay_ms = int(ao.get("mux_delay_ms") or 0)
            except Exception:
                mux_delay_ms = 0
            # metadata for single-track external file index=0
            add_track_meta_args(args, tmux, 0)
            if mux_delay_ms != 0:
                args += ["--sync", f"0:{mux_delay_ms}"]
            args += [str(ap)]
            plan["audio_sources"].append({
                "trackId": tid,
                "path": str(ap),
                "from": "audio_manifest",
                "mux_delay_ms": mux_delay_ms,
            })
        else:
            # fallback: allow COPY from source
            if is_copy(status):
                add_track_meta_args(args, tmux, 0)
                args += ["--audio-tracks", str(tid), "--no-video", "--no-subtitles", str(source)]
                plan["audio_sources"].append({"trackId": tid, "path": str(source), "from": "source_copy"})
                plan["fallbacks"].append(f"audio_{tid}_from_source")
            else:
                raise RuntimeError(f"missing_audio_output_track_{tid}")

    # 3) Subtitles
    # Prefer extracted subtitle files (demux_manifest). If absent, fallback to source COPY.
    for t in sorted(sub_tracks, key=lambda x: int(x.get("trackId", 0))):
        tid = int(t.get("trackId"))
        status = str(t.get("trackStatus") or "")
        tmux = t.get("trackMux") or {}
        if not isinstance(tmux, dict):
            tmux = {}

        if tid in extracted_subs:
            sp = extracted_subs[tid]
            add_track_meta_args(args, tmux, 0)
            args += [str(sp)]
            plan["sub_sources"].append({"trackId": tid, "path": str(sp), "from": "demux_manifest"})
        else:
            if is_copy(status):
                add_track_meta_args(args, tmux, 0)
                args += ["--subtitle-tracks", str(tid), "--no-video", "--no-audio", str(source)]
                plan["sub_sources"].append({"trackId": tid, "path": str(source), "from": "source_copy"})
                plan["fallbacks"].append(f"sub_{tid}_from_source")
            else:
                raise RuntimeError(f"missing_sub_file_track_{tid}")

    # 4) Attachments (from extracted files)
    # Add at the end as global options; mkvmerge applies them to output
    for a in attachments:
        p = Path(a["path"])
        name = a.get("file_name") or p.name
        ctype = a.get("content_type") or ""
        args += ["--attachment-name", str(name)]
        if ctype:
            args += ["--attachment-mime-type", str(ctype)]
        else:
            # optional: infer from ext if missing
            m = mime_from_ext(p)
            if m:
                args += ["--attachment-mime-type", m]
        args += ["--attach-file", str(p)]

    return args, plan


# ---------------------------
# main
# ---------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="mux")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--mkvmerge", default="mkvmerge", help="Path to mkvmerge (MKVToolNix)")
    ap.add_argument("--log", default="", help="Optional log file path (relative to --workdir if not absolute)")
    ap.add_argument("--no-source-bitrate", action="store_true", help="Skip per-scene source bitrate calculation.")
    ap.add_argument(
        "--source-bitrate-timeout",
        type=float,
        default=SOURCE_BITRATE_TIMEOUT_SEC,
        help="Abort per-scene source bitrate scan after N seconds (0 disables timeout).",
    )
    args = ap.parse_args(argv)

    resolved_plan = resolve_file_plan(Path(args.plan))
    source = resolved_plan.paths.source
    workdir = resolved_plan.paths.workdir

    mkvmerge = which_or(args.mkvmerge)
    setup_logging(args.log, workdir)
    marker = marker_path(workdir)
    if marker.exists() and not runner_managed_state():
        print(f"[mux] skip: marker exists: {marker}")
        return 0

    try:
        if not source.exists():
            raise RuntimeError("missing_source")
#         if source.suffix.lower() != ".mkv":
#             raise RuntimeError("source_not_mkv")
        if not workdir.exists():
            raise RuntimeError("missing_workdir")
        if not (Path(mkvmerge).exists() or shutil.which(mkvmerge)):
            raise RuntimeError("missing_mkvmerge")

        source_timeout_sec = args.source_bitrate_timeout
        if source_timeout_sec is not None and source_timeout_sec <= 0:
            source_timeout_sec = None

        update_scene_bitrates(
            workdir,
            source,
            include_source=(not args.no_source_bitrate),
            source_timeout_sec=source_timeout_sec,
        )

        tracks = resolved_plan.runtime_tracks()
        global_tags_path, global_tags = prepare_global_tags(
            workdir,
            source,
            tracks,
            encode_params_text=resolved_plan.build_encode_params_text(),
        )

        cmd, plan = build_mux_command(mkvmerge, source, workdir, tracks, global_tags_path=global_tags_path)

        # Save manifest for debugging/reproducibility
        write_json(workdir / "00_meta" / "mux_manifest.json", {
            "source": str(source),
            "workdir": str(workdir),
            "mkvmerge": str(mkvmerge),
            "plan": plan,
            "global_tags_written": global_tags,
            "command": cmd,
        })

        run_cmd(cmd)

        print("[mux] OK")
        if not runner_managed_state():
            write_marker(workdir)
        return 0

    except Exception as ex:
        eprint(f"[mux] ERROR: {ex}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
