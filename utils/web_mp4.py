#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TEXT_SUBTITLE_CODECS = {
    "ass",
    "ssa",
    "subrip",
    "srt",
    "text",
    "webvtt",
    "mov_text",
}

VIDEO_INPUT_EXTS = {
    ".3g2",
    ".3gp",
    ".asf",
    ".avi",
    ".av1",
    ".divx",
    ".flv",
    ".h264",
    ".h265",
    ".hevc",
    ".ivf",
    ".m2t",
    ".m2ts",
    ".m2v",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mts",
    ".mxf",
    ".ogm",
    ".ogv",
    ".rm",
    ".rmvb",
    ".ts",
    ".vob",
    ".webm",
    ".wmv",
    ".y4m",
}


@dataclass(frozen=True)
class StreamInfo:
    index: int
    codec_type: str
    codec_name: str
    title: str = ""
    language: str = ""


@dataclass(frozen=True)
class WebMp4Command:
    command: List[str]
    skipped_subtitles: List[StreamInfo]


def collect_input_files(paths: Iterable[str], *, recursive: bool) -> Tuple[List[Path], List[str]]:
    seen: set[str] = set()
    files: List[Path] = []
    unknown: List[str] = []

    def add_file(path: Path) -> None:
        resolved = path.resolve()
        key = str(resolved).lower()
        if key in seen:
            return
        seen.add(key)
        files.append(resolved)

    def is_video_file(path: Path) -> bool:
        return path.suffix.lower() in VIDEO_INPUT_EXTS

    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            candidates = path.rglob("*") if recursive else path.iterdir()
            found = False
            for candidate in sorted(candidates):
                if candidate.is_file() and is_video_file(candidate):
                    add_file(candidate)
                    found = True
            if not found:
                unknown.append(raw)
            continue

        if path.is_file() and is_video_file(path):
            add_file(path)
            continue

        unknown.append(raw)

    return files, unknown


collect_mkv_files = collect_input_files


def probe_streams(input_path: Path) -> List[StreamInfo]:
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-v", "error",
        "-show_entries", "stream=index,codec_type,codec_name:stream_tags=language,title",
        "-of", "json",
        str(input_path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"ffprobe failed for {input_path}: {details}")

    try:
        payload: Dict[str, Any] = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON for {input_path}: {exc}") from exc

    streams: List[StreamInfo] = []
    for raw in payload.get("streams", []) or []:
        tags = raw.get("tags", {}) or {}
        try:
            index = int(raw.get("index"))
        except Exception:
            continue
        streams.append(
            StreamInfo(
                index=index,
                codec_type=str(raw.get("codec_type") or ""),
                codec_name=str(raw.get("codec_name") or "").lower(),
                title=str(tags.get("title") or ""),
                language=str(tags.get("language") or ""),
            )
        )
    return streams


def build_web_mp4_command(input_path: Path, output_path: Path) -> WebMp4Command:
    streams = probe_streams(input_path)
    subtitles = [stream for stream in streams if stream.codec_type == "subtitle"]
    text_subtitles = [
        stream for stream in subtitles if stream.codec_name in TEXT_SUBTITLE_CODECS
    ]
    skipped_subtitles = [
        stream for stream in subtitles if stream.codec_name not in TEXT_SUBTITLE_CODECS
    ]

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-map", "0:v?",
        "-map", "0:a?",
    ]
    for stream in text_subtitles:
        cmd.extend(["-map", f"0:{stream.index}"])

    cmd.extend(
        [
            "-c:v", "copy",
            "-c:a", "copy",
            "-c:s", "mov_text",
            "-movflags", "+faststart",
            str(output_path),
        ]
    )
    return WebMp4Command(command=cmd, skipped_subtitles=skipped_subtitles)


def format_command(command: List[str]) -> str:
    return subprocess.list2cmdline(command)


def make_web_mp4(
    input_path: Path,
    output_path: Optional[Path] = None,
    *,
    dry_run: bool = False,
) -> bool:
    input_path = input_path.resolve()
    if output_path is None:
        output_path = input_path.with_suffix(".mp4")
    else:
        output_path = output_path.resolve()

    if not input_path.exists():
        print(f"[skip] missing input: {input_path}")
        return False
    if output_path.exists():
        print(f"[skip] mp4 exists: {output_path}")
        return True

    try:
        planned = build_web_mp4_command(input_path, output_path)
    except RuntimeError as exc:
        print(f"[err] {exc}")
        return False

    for stream in planned.skipped_subtitles:
        label = f"#{stream.index} {stream.codec_name}"
        if stream.language:
            label += f" {stream.language}"
        if stream.title:
            label += f" ({stream.title})"
        print(f"[skip] subtitle not supported by MP4: {label}")

    print("[cmd]", format_command(planned.command))
    if dry_run:
        return True

    completed = subprocess.run(planned.command)
    if completed.returncode != 0:
        print(f"[err] ffmpeg failed for {input_path} (code={completed.returncode})")
        return False
    return True


def print_help(program: str) -> None:
    print(
        "Usage:\n"
        f"  {program} [--recursive] [--dry-run] <folder-or-file> [...]\n"
        "\n"
        "Creates a sibling .mp4 for every input video file that does not already have one.\n"
        "Video and audio streams are copied. Text subtitles are converted to mov_text."
    )


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("-r", "--recursive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None, *, program: Optional[str] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if program is None:
        program = Path(sys.argv[0]).as_posix()

    args = parse_args(argv)
    if args.help:
        print_help(program)
        return 0

    paths = args.paths or ["."]
    files, unknown = collect_input_files(paths, recursive=args.recursive)
    if unknown:
        print("Unknown or empty inputs:")
        for item in unknown:
            print(f"  {item}")

    if not files:
        print("No video input files found.")
        return 1

    ok = True
    for input_path in files:
        print()
        print(f"[Make Web MP4] {input_path}")
        if not make_web_mp4(input_path, dry_run=args.dry_run):
            ok = False

    return 0 if ok else 1
