#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_mkvmerge(path: Path) -> Optional[Dict[str, Any]]:
    """Run mkvmerge and return parsed JSON, or None on error."""
    cmd = [
        "mkvmerge",
        "-J",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        print(
            "Error: mkvmerge not found. Make sure MKVToolNix/mkvmerge is in PATH.",
            file=sys.stderr,
        )
        return None

    if result.returncode != 0:
        print(f"Error probing {path}:\n{result.stderr}", file=sys.stderr)
        return None

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"Error parsing mkvmerge JSON for {path}: {e}", file=sys.stderr)
        return None


def safe_get(d: Dict[str, Any], *keys, default=None):
    """Safely get nested dict value."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def normalize_language(lang: Optional[str]) -> str:
    if not lang or str(lang).upper() == "N/A":
        return "und"
    return str(lang)


def normalize_title(title: Optional[str]) -> str:
    if not title or str(title).upper() == "N/A":
        return "-"
    return str(title)


def pick_bitrate(props: Dict[str, Any]) -> str:
    """
    Try to get bitrate from mkvmerge properties.
    Return human-readable kb/s if possible, otherwise 'N/A'.
    """
    bit_rate = props.get("bit_rate") or props.get("audio_bit_rate")
    if not bit_rate or str(bit_rate).upper() == "N/A":
        return "N/A"

    try:
        br_int = int(float(bit_rate))
        kbps = br_int / 1000.0
        return f"{kbps:.0f} kb/s"
    except (TypeError, ValueError):
        return str(bit_rate)


def track_type_from_mkv(track_type: Optional[str]) -> str:
    if not track_type:
        return "Unknown"
    lowered = track_type.strip().lower()
    if lowered == "video":
        return "Video"
    if lowered == "audio":
        return "Audio"
    if lowered in ("subtitle", "subtitles"):
        return "Subtitle"
    return track_type


def build_extra_info(track_type: str, props: Dict[str, Any]) -> str:
    lowered = (track_type or "").strip().lower()

    if lowered == "video":
        dims = props.get("display_dimensions")
        if not dims:
            width = props.get("pixel_width") or props.get("display_width")
            height = props.get("pixel_height") or props.get("display_height")
            if width and height:
                dims = f"{width}x{height}"

        fps = None
        default_duration = props.get("default_duration")
        if default_duration:
            try:
                fps_val = 1_000_000_000 / float(default_duration)
                fps = f"@ {fps_val:.3f}fps"
            except (TypeError, ValueError, ZeroDivisionError):
                fps = None

        parts: List[str] = []
        if dims:
            parts.append(str(dims))
        if fps:
            parts.append(fps)
        return " ".join(parts) if parts else ""

    if lowered == "audio":
        sample_rate = props.get("audio_sampling_frequency")
        channels = props.get("audio_channels")
        ch_layout = props.get("audio_channel_layout")

        parts = []
        if sample_rate:
            try:
                parts.append(f"{int(float(sample_rate))} Hz")
            except (TypeError, ValueError):
                parts.append(f"{sample_rate} Hz")
        if channels:
            parts.append(f"{channels} ch")
        if ch_layout:
            parts.append(f"({ch_layout})")
        return ", ".join(parts) if parts else ""

    if lowered in ("subtitle", "subtitles"):
        return "subtitle"

    return ""


def print_file_info(path: Path):
    data = run_mkvmerge(path)
    if data is None:
        return

    print(f"=== {path} ===")

    tracks = data.get("tracks") or []
    if not tracks:
        print("No tracks found")
        return

    for track in tracks:
        raw_type = track.get("type") or ""
        track_type = track_type_from_mkv(raw_type)
        if raw_type.strip().lower() == "attachment":
            continue

        idx = track.get("id", "-")
        props = track.get("properties") or {}

        title = normalize_title(props.get("track_name") or props.get("name"))
        lang = normalize_language(props.get("language") or props.get("language_ietf"))

        bitrate = pick_bitrate(props)
        is_default = 1 if props.get("default_track") else 0

        codec_name = track.get("codec") or props.get("codec_name") or props.get("codec_id") or "-"
        extra = build_extra_info(raw_type, props)

        # Final line:
        # {trackType} | {id} | {Title} | {language} | {bitrate} | {isDefault} | {codec} | {extraInfo}
        print(
            f"{track_type} | {idx} | {title} | {lang} | {bitrate} | "
            f"{is_default} | {codec_name} | {extra}"
        )

    print()  # blank line between files


def main():
    parser = argparse.ArgumentParser(
        description="Print track info for media files using mkvmerge."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Input media files (mkv, mp4, etc.)",
    )
    args = parser.parse_args()

    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            continue
        print_file_info(path)


if __name__ == "__main__":
    main()
