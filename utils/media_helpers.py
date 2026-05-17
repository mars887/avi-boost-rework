"""Shared media track and filesystem-name helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


WIN_BAD = r'<>:"/\|?*'
WIN_BAD_RE = re.compile(rf"[{re.escape(WIN_BAD)}]")


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
