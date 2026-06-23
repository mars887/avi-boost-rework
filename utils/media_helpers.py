"""Shared media track and filesystem-name helpers."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


WIN_BAD = r'<>:"/\|?*'
WIN_BAD_RE = re.compile(rf"[{re.escape(WIN_BAD)}]")


def run_ffprobe_json(path: Path, *, ffprobe: Optional[str] = None, timeout: float = 30.0) -> Dict[str, Any]:
    """Run ``ffprobe -show_format -show_streams`` and return the parsed JSON.

    Single source for the full-probe invocation used across the pipeline. Returns
    ``{}`` when ffprobe is missing, errors, or emits unparseable output (callers
    decide how to treat the empty result).
    """
    exe = ffprobe or shutil.which("ffprobe")
    if not exe:
        return {}
    cmd = [
        exe,
        "-hide_banner",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            return {}
        parsed = json.loads(proc.stdout or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def media_duration_seconds(payload: Dict[str, Any]) -> float:
    """Best-effort media duration from an ffprobe payload.

    Uses ``format.duration`` and falls back to the longest per-stream duration
    (some containers, e.g. raw streams, only carry the latter). Returns ``0.0``
    when nothing usable is found. Single source so the runner, verify and web
    exporters agree on how a file's length is read.
    """
    candidates = []
    fmt = payload.get("format")
    if isinstance(fmt, dict) and fmt.get("duration") is not None:
        candidates.append(fmt.get("duration"))
    for stream in payload.get("streams") or []:
        if isinstance(stream, dict) and stream.get("duration") is not None:
            candidates.append(stream.get("duration"))
    best = 0.0
    for value in candidates:
        try:
            best = max(best, float(value))
        except (TypeError, ValueError):
            continue
    return max(0.0, best)


def normalize_track_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sub") or text == "subtitle":
        return "sub"
    if text.startswith("aud") or text == "audio":
        return "audio"
    if text.startswith("vid") or text == "video":
        return "video"
    return text


def sanitize_component(name: str, *, default: str = "untitled", max_len: int = 80) -> str:
    text = str(name or "").strip()
    text = WIN_BAD_RE.sub("_", text)
    text = re.sub(r"\s+", " ", text).strip().rstrip(". ")
    if not text:
        text = default
    if len(text) > max_len:
        text = text[:max_len].rstrip(". ")
    return text or default


def find_track_info(mkv_json: Dict[str, Any], track_id: int) -> Optional[Dict[str, Any]]:
    for track in mkv_json.get("tracks", []) or []:
        try:
            if int(track.get("id", -1)) == int(track_id):
                return track
        except Exception:
            continue
    return None


def subtitle_extension_from_codec(codec_id: str, *, default: str = ".sub") -> str:
    codec = str(codec_id or "").upper()
    if "S_TEXT/ASS" in codec:
        return ".ass"
    if "S_TEXT/SSA" in codec:
        return ".ssa"
    if "S_TEXT/UTF8" in codec:
        return ".srt"
    if "S_TEXT/WEBVTT" in codec:
        return ".vtt"
    if "S_TEXT/USF" in codec:
        return ".usf"
    if "S_TEXT/TIMEDTEXT" in codec or "S_TEXT/TTML" in codec:
        return ".ttml"
    if "S_HDMV/PGS" in codec:
        return ".sup"
    if "S_VOBSUB" in codec or "S_DVBSUB" in codec:
        return ".sub"
    return default
