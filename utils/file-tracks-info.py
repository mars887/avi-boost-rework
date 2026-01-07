#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_ffprobe(path: Path) -> Optional[Dict[str, Any]]:
    """Run ffprobe and return parsed JSON, or None on error."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("Error: ffprobe not found. Make sure FFmpeg/ffprobe is in PATH.", file=sys.stderr)
        return None

    if result.returncode != 0:
        print(f"Error probing {path}:\n{result.stderr}", file=sys.stderr)
        return None

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"Error parsing ffprobe JSON for {path}: {e}", file=sys.stderr)
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
    if not lang or lang.upper() == "N/A":
        return "und"
    return lang


def normalize_title(title: Optional[str]) -> str:
    if not title or title.upper() == "N/A":
        return "-"
    return title


def pick_bitrate(stream: Dict[str, Any]) -> str:
    """
    Try to get bitrate:
    - stream["bit_rate"]
    - stream["tags"]["BPS"]
    Return human-readable kb/s if possible, otherwise 'N/A'.
    """
    bit_rate = stream.get("bit_rate")
    if not bit_rate or str(bit_rate).upper() == "N/A":
        bit_rate = safe_get(stream, "tags", "BPS")

    if not bit_rate or str(bit_rate).upper() == "N/A":
        return "N/A"

    # Convert to kb/s if possible
    try:
        br_int = int(bit_rate)
        kbps = br_int / 1000.0
        # Format like "123 kb/s"
        return f"{kbps:.0f} kb/s"
    except (TypeError, ValueError):
        # Just return raw value
        return str(bit_rate)


def track_type_from_codec(codec_type: Optional[str]) -> str:
    if codec_type == "video":
        return "Video"
    if codec_type == "audio":
        return "Audio"
    if codec_type == "subtitle":
        return "Subtitle"
    if codec_type:
        return codec_type
    return "Unknown"


def build_extra_info(stream: Dict[str, Any]) -> str:
    codec_type = stream.get("codec_type")

    if codec_type == "video":
        width = stream.get("width")
        height = stream.get("height")
        # r_frame_rate may be "24000/1001"
        fps_str = stream.get("r_frame_rate")
        fps_human = None
        if fps_str and fps_str != "0/0":
            try:
                num, den = fps_str.split("/")
                fps_val = float(num) / float(den)
                fps_human = f"{fps_val:.3f}fps"
            except Exception:
                fps_human = fps_str

        parts: List[str] = []
        if width and height:
            parts.append(f"{width}x{height}")
        if fps_human:
            parts.append(f"@ {fps_human}")
        return " ".join(parts) if parts else ""

    if codec_type == "audio":
        sample_rate = stream.get("sample_rate")
        channels = stream.get("channels")
        ch_layout = stream.get("channel_layout")

        parts: List[str] = []
        if sample_rate:
            try:
                parts.append(f"{int(sample_rate)} Hz")
            except ValueError:
                parts.append(f"{sample_rate} Hz")
        if channels:
            parts.append(f"{channels} ch")
        if ch_layout:
            parts.append(f"({ch_layout})")
        return ", ".join(parts) if parts else ""

    if codec_type == "subtitle":
        return "subtitle"

    return ""


def print_file_info(path: Path):
    data = run_ffprobe(path)
    if data is None:
        return

    print(f"=== {path} ===")

    streams = data.get("streams", [])
    if not streams:
        print("No streams found")
        return

    for stream in streams:
        codec_type = stream.get("codec_type")
        track_type = track_type_from_codec(codec_type)
        idx = stream.get("index", "-")

        tags = stream.get("tags", {}) or {}
        title = normalize_title(tags.get("title"))
        lang = normalize_language(tags.get("language"))

        bitrate = pick_bitrate(stream)
        is_default = safe_get(stream, "disposition", "default", default=0)

        codec_name = stream.get("codec_name", "-")
        extra = build_extra_info(stream)

        # Final line:
        # {trackType} | {id} | {Title} | {language} | {bitrate} | {isDefault} | {codec} | {extraInfo}
        print(
            f"{track_type} | {idx} | {title} | {lang} | {bitrate} | "
            f"{is_default} | {codec_name} | {extra}"
        )

    print()  # blank line between files


def main():
    parser = argparse.ArgumentParser(
        description="Print stream info for media files using ffprobe."
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
