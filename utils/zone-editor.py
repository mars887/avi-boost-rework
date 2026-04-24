#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
edit_av1an_scenes.py

Пример:
  python edit_av1an_scenes.py --scenes scenes.json --out out_scenes.json --source video.mkv ^
    --command "100f700 - --preset 1?70t2:40 - --crf -10%?4s7, 200f600 - --crf +5 --scm 0"

Или:
  python edit_av1an_scenes.py --scenes scenes.json --out out_scenes.json --source video.mkv --command commands.txt

Где commands.txt: одна команда на строку.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shlex
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.pipeline_runtime import setup_stage_logging
from utils.zoned_commands import project_zone_command_lines


def setup_logging(log_path: str, base_dir: Optional[Path] = None) -> None:
    setup_stage_logging(log_path, stage_name="zone-editor", base_dir=base_dir)


# Можно заполнить позже: {"--crf": Decimal("0.25"), "--someparam": Decimal("2")}
# По ТЗ — список оставить пустым.
DEFAULT_PARAM_STEP: Dict[str, Decimal] = {"--crf": Decimal("0.25")}

# Полезные дефолты форматирования (можно расширять при желании)
FORCE_2DP: set[str] = {"--crf"}  # CRF обычно хранится с 2 знаками
META_KEY = "pb_meta"


# ------------------------- ffprobe helpers -------------------------

def _run_ffprobe_json(args: List[str]) -> Dict[str, Any]:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json"] + args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found in PATH. Install FFmpeg and ensure ffprobe is available.")

    if r.returncode != 0:
        msg = (r.stderr or "").strip() or (r.stdout or "").strip()
        raise RuntimeError(f"ffprobe failed (code={r.returncode}): {msg}")

    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe returned invalid JSON: {e}")


@dataclass(frozen=True)
class Chapter:
    title: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class VideoInfo:
    fps: Fraction
    duration_sec: float
    chapters: List[Chapter]
    is_vfr_suspected: bool


def _parse_fraction(s: str) -> Fraction:
    # ffprobe обычно отдаёт "24000/1001"
    try:
        return Fraction(s)
    except Exception:
        # fallback: float
        return Fraction(Decimal(str(float(s))))


def _normalize_chapter_title(value: Any) -> str:
    text = str(value or "")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE)
    return text.strip()


def _extract_chapter_title(chapter: Dict[str, Any]) -> str:
    tags = chapter.get("tags") or {}
    if not isinstance(tags, dict):
        tags = {}

    preferred_keys = ("title", "TITLE", "ChapterString", "CHAPTERSTRING")
    for key in preferred_keys:
        title = _normalize_chapter_title(tags.get(key))
        if title:
            return title

    for key, value in tags.items():
        key_norm = str(key or "").strip().casefold()
        if key_norm == "title" or key_norm.startswith("title-") or key_norm == "chapterstring" or key_norm.startswith("chapterstring-"):
            title = _normalize_chapter_title(value)
            if title:
                return title

    return ""


def read_video_info(source_path: str) -> VideoInfo:
    data = _run_ffprobe_json([
        "-show_streams",
        "-show_format",
        "-show_chapters",
        "-select_streams", "v:0",
        source_path,
    ])

    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError("No video stream found (ffprobe -select_streams v:0 returned nothing).")

    v0 = streams[0]
    avg_fr = v0.get("avg_frame_rate") or "0/0"
    r_fr = v0.get("r_frame_rate") or "0/0"

    fps = _parse_fraction(avg_fr)
    if fps.numerator == 0:
        # fallback to r_frame_rate
        fps = _parse_fraction(r_fr)

    if fps.numerator == 0:
        raise RuntimeError("Could not determine FPS from ffprobe (avg_frame_rate/r_frame_rate are 0).")

    fmt = data.get("format") or {}
    duration_sec = float(fmt.get("duration") or 0.0)

    # chapters
    raw_ch = data.get("chapters") or []
    chapters: List[Chapter] = []
    for ch in raw_ch:
        title = _extract_chapter_title(ch)
        start_time = float(ch.get("start_time") or 0.0)
        end_time = ch.get("end_time")
        end_time_f = float(end_time) if end_time is not None else 0.0
        chapters.append(Chapter(title=title, start_sec=start_time, end_sec=end_time_f))

    # If end_sec missing/0, derive from next start or duration
    if chapters:
        fixed: List[Chapter] = []
        for i, ch in enumerate(chapters):
            end_sec = ch.end_sec
            if end_sec <= ch.start_sec:
                if i + 1 < len(chapters):
                    end_sec = chapters[i + 1].start_sec
                elif duration_sec > 0:
                    end_sec = duration_sec
                else:
                    end_sec = ch.start_sec
            fixed.append(Chapter(title=ch.title, start_sec=ch.start_sec, end_sec=end_sec))
        chapters = fixed

    # Heuristic VFR suspicion: avg != r (often happens with VFR)
    is_vfr_suspected = False
    try:
        is_vfr_suspected = (Fraction(avg_fr) != Fraction(r_fr)) and (avg_fr != "0/0") and (r_fr != "0/0")
    except Exception:
        pass

    return VideoInfo(fps=fps, duration_sec=duration_sec, chapters=chapters, is_vfr_suspected=is_vfr_suspected)


def sec_to_frame_floor(sec: float, fps: Fraction) -> int:
    # start boundary: floor
    return int(math.floor(sec * float(fps) + 1e-9))


def sec_to_frame_ceil(sec: float, fps: Fraction) -> int:
    # end boundary: ceil
    return int(math.ceil(sec * float(fps) - 1e-9))


def frame_to_sec(frame: int, fps: Fraction) -> float:
    return float(frame) / float(fps)


def format_timecode(frame: int, fps: Fraction) -> str:
    total = frame_to_sec(frame, fps)
    if total < 0:
        total = 0.0
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    seconds = total % 60.0
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def ensure_scene_meta(scene: Dict[str, Any], scene_idx: int, s0: int, s1: int, fps: Fraction) -> Dict[str, Any]:
    meta = scene.get(META_KEY)
    if not isinstance(meta, dict):
        meta = {}
        scene[META_KEY] = meta

    meta["index"] = scene_idx

    frames_meta = meta.get("frames")
    if not isinstance(frames_meta, dict):
        frames_meta = {}
        meta["frames"] = frames_meta
    frames_meta["start"] = s0
    frames_meta["end"] = s1
    frames_meta["len"] = max(0, s1 - s0)

    t0 = frame_to_sec(s0, fps)
    t1 = frame_to_sec(s1, fps)
    time_meta = meta.get("time")
    if not isinstance(time_meta, dict):
        time_meta = {}
        meta["time"] = time_meta
    time_meta["start_sec"] = round(t0, 3)
    time_meta["end_sec"] = round(t1, 3)
    time_meta["duration_sec"] = round(max(0.0, t1 - t0), 3)
    time_meta["start_tc"] = format_timecode(s0, fps)
    time_meta["end_tc"] = format_timecode(s1, fps)

    bitrate_meta = meta.get("bitrate")
    if not isinstance(bitrate_meta, dict):
        bitrate_meta = {}
        meta["bitrate"] = bitrate_meta
    bitrate_meta.setdefault("source_kbps", None)
    bitrate_meta.setdefault("fastpass_kbps", None)
    bitrate_meta.setdefault("mainpass_kbps", None)

    return meta


# ------------------------- command model -------------------------

@dataclass(frozen=True)
class FrameRange:
    start: int
    end: int  # half-open [start, end)


@dataclass(frozen=True)
class Selector:
    kind: str  # "frames" | "scene_idx" | "time" | "chapter"
    frame_ranges: Tuple[FrameRange, ...] = ()
    scene_ranges: Tuple[Tuple[int, int], ...] = ()  # inclusive [a,b]
    raw: str = ""


@dataclass(frozen=True)
class Action:
    key: str  # normalized: "--param"
    raw_value: str


@dataclass(frozen=True)
class BoundaryOptions:
    min_len: int = 0
    min2_len: int = 0
    magnet: int = 0
    threshold: int = 0


@dataclass(frozen=True)
class Command:
    selectors: Tuple[Selector, ...]
    sep: str  # '-', '!', '^', '|'
    actions: Tuple[Action, ...]
    raw_line: str
    boundary_options: Optional[BoundaryOptions] = None


# ------------------------- parsing helpers -------------------------

_RX_FRAME = re.compile(r"^\s*(\d+)\s*[fF]\s*(\d+)\s*$")
_RX_SCENE = re.compile(r"^\s*(\d+)\s*[sS]\s*(\d+)\s*$")
_RX_TIME = re.compile(r"^\s*(.+?)\s*[tT]\s*(.+?)\s*$")


def parse_time_token(tok: str) -> float:
    """
    tok can be:
      - "70" -> seconds
      - "0:30" -> mm:ss
      - "1:02:03" -> hh:mm:ss
      - allow decimals: "12.5"
    """
    tok = tok.strip()
    if ":" not in tok:
        return float(tok)

    parts = tok.split(":")
    if len(parts) == 2:
        mm, ss = parts
        return float(mm) * 60.0 + float(ss)
    if len(parts) == 3:
        hh, mm, ss = parts
        return float(hh) * 3600.0 + float(mm) * 60.0 + float(ss)
    raise ValueError(f"Invalid time token: {tok!r}")


def normalize_key(k: str) -> str:
    k = k.strip()
    k = k.lstrip("-")
    return "--" + k


def space_boundary_bars(line: str) -> str:
    out: List[str] = []
    quote = ""
    for ch in line:
        if ch in ("'", '"'):
            if quote == ch:
                quote = ""
            elif not quote:
                quote = ch
            out.append(ch)
            continue
        if ch == "|" and not quote:
            out.append(" | ")
            continue
        out.append(ch)
    return "".join(out)


def split_command_tokens(line: str) -> Tuple[str, str, List[str], List[str]]:
    """
    Returns: (selectors_part, sep, action_tokens, boundary_option_tokens)
    Parsing strategy:
      - boundary edit mode: selectors | option_tokens | action_tokens
      - legacy mode: find first token that is exactly '-', '!', '^'
    """
    tokens = shlex.split(space_boundary_bars(line), posix=False)
    if not tokens:
        raise ValueError("Empty command line")

    if "|" in tokens:
        first_bar = tokens.index("|")
        try:
            second_bar = tokens.index("|", first_bar + 1)
        except ValueError:
            raise ValueError("Boundary command must contain two standalone '|' separators")

        selectors_part = " ".join(tokens[:first_bar]).strip()
        option_tokens = tokens[first_bar + 1 : second_bar]
        action_tokens = tokens[second_bar + 1 :]
        if not selectors_part:
            raise ValueError("Missing selectors part before boundary options")
        if not action_tokens:
            raise ValueError("Missing actions part after boundary options")
        return selectors_part, "|", action_tokens, option_tokens

    sep_idx = None
    sep = ""
    for i, t in enumerate(tokens):
        if t in ("-", "!", "^"):
            sep_idx = i
            sep = t
            break

    if sep_idx is None:
        raise ValueError("Command separator not found (expected one of: -, !, ^ as a standalone token)")

    selectors_part = " ".join(tokens[:sep_idx]).strip()
    action_tokens = tokens[sep_idx + 1 :]
    if not selectors_part:
        raise ValueError("Missing selectors part before separator")
    if not action_tokens:
        raise ValueError("Missing actions part after separator")

    return selectors_part, sep, action_tokens, []


def parse_boundary_options(option_tokens: List[str]) -> BoundaryOptions:
    key_aliases = {
        "min": "min_len",
        "min-len": "min_len",
        "min_len": "min_len",
        "min-scene-len": "min_len",
        "min_scene_len": "min_len",
        "min2": "min2_len",
        "min2-len": "min2_len",
        "min2_len": "min2_len",
        "special-min": "min2_len",
        "special_min": "min2_len",
        "magnet": "magnet",
        "snap": "magnet",
        "snap-to": "magnet",
        "snap_to": "magnet",
        "thr": "threshold",
        "threshold": "threshold",
        "tolerance": "threshold",
        "ignore": "threshold",
    }

    values: Dict[str, Optional[int]] = {
        "min_len": None,
        "min2_len": None,
        "magnet": None,
        "threshold": None,
    }

    i = 0
    while i < len(option_tokens):
        tok = str(option_tokens[i]).strip()
        if not tok:
            i += 1
            continue

        key_text: str
        value_text: str
        if "=" in tok:
            key_text, value_text = tok.split("=", 1)
            i += 1
        elif ":" in tok:
            key_text, value_text = tok.split(":", 1)
            i += 1
        else:
            if i + 1 >= len(option_tokens):
                raise ValueError(f"Boundary option {tok!r} has no value")
            key_text = tok
            value_text = str(option_tokens[i + 1])
            i += 2

        key_norm = key_text.strip().lstrip("-").casefold()
        key_norm = key_aliases.get(key_norm, "")
        if not key_norm:
            raise ValueError(f"Unknown boundary option: {key_text!r}")

        try:
            value_dec = Decimal(str(value_text))
        except Exception:
            raise ValueError(f"Invalid integer value for boundary option {key_text!r}: {value_text!r}")
        if value_dec != value_dec.to_integral_value():
            raise ValueError(f"Boundary option {key_text!r} must be an integer frame count")
        value = int(value_dec)
        if value < 0:
            raise ValueError(f"Boundary option {key_text!r} must be >= 0")
        values[key_norm] = value

    min_len = values["min_len"] if values["min_len"] is not None else 0
    min2_len = values["min2_len"] if values["min2_len"] is not None else min_len
    magnet = values["magnet"] if values["magnet"] is not None else 0
    threshold = values["threshold"] if values["threshold"] is not None else 0
    return BoundaryOptions(
        min_len=int(min_len),
        min2_len=int(min2_len),
        magnet=int(magnet),
        threshold=int(threshold),
    )


def parse_preset_definition(line: str) -> Tuple[str, List[str]]:
    selectors_part, sep, action_tokens, _option_tokens = split_command_tokens(line)
    if sep == "|":
        raise ValueError("Preset definitions cannot use boundary edit mode")
    name = selectors_part.strip()
    if not name.startswith("@"):
        raise ValueError("Preset line must start with '@'")
    name = name[1:].strip()
    if not name:
        raise ValueError("Preset name is empty")
    if any(ch.isspace() for ch in name):
        raise ValueError(f"Preset name must be a single token: {name!r}")
    if not action_tokens:
        raise ValueError(f"Preset {name!r} has no actions")
    return name, action_tokens


def expand_preset_tokens(
    tokens: List[str],
    presets: Dict[str, List[str]],
    *,
    _stack: Optional[List[str]] = None,
) -> List[str]:
    if _stack is None:
        _stack = []
    out: List[str] = []
    for tok in tokens:
        if tok.startswith("@"):
            name = tok[1:]
            if not name:
                raise ValueError("Invalid preset reference '@' with no name")
            if name not in presets:
                raise ValueError(f"Unknown preset @{name}")
            if name in _stack:
                cycle = " -> ".join(_stack + [name])
                raise ValueError(f"Preset cycle detected: {cycle}")
            expanded = expand_preset_tokens(presets[name], presets, _stack=_stack + [name])
            out.extend(expanded)
        else:
            out.append(tok)
    return out


def find_chapter_ranges_by_title(chapters: List[Chapter], title_query: str) -> List[Tuple[float, float, str]]:
    q = _normalize_chapter_title(title_query).casefold()
    if not q:
        return []

    # 1) exact match (case-insensitive)
    exact = [c for c in chapters if _normalize_chapter_title(c.title).casefold() == q]
    if exact:
        return [(c.start_sec, c.end_sec, c.title) for c in exact]

    # 2) substring match
    sub = [c for c in chapters if q in _normalize_chapter_title(c.title).casefold()]
    return [(c.start_sec, c.end_sec, c.title) for c in sub]


def parse_selectors(
    selectors_part: str,
    *,
    video: VideoInfo,
    total_frames: Optional[int],
) -> List[Selector]:
    out: List[Selector] = []
    chunks = [s.strip() for s in selectors_part.split(",") if s.strip()]
    if not chunks:
        raise ValueError("No selectors found")

    for raw in chunks:
        m = _RX_FRAME.match(raw)
        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            if b < a:
                a, b = b, a
            # Frame selectors are inclusive in the user-facing DSL: 100F300
            # means frames [100, 300]. Internally scenes use end-exclusive
            # ranges, so store it as [100, 301).
            fr = FrameRange(start=a, end=b + 1)
            out.append(Selector(kind="frames", frame_ranges=(fr,), raw=raw))
            continue

        m = _RX_SCENE.match(raw)
        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            if b < a:
                a, b = b, a
            out.append(Selector(kind="scene_idx", scene_ranges=((a, b),), raw=raw))
            continue

        m = _RX_TIME.match(raw)
        if m:
            try:
                t0 = parse_time_token(m.group(1))
                t1 = parse_time_token(m.group(2))
            except ValueError:
                pass
            else:
                if t1 < t0:
                    t0, t1 = t1, t0

                f0 = sec_to_frame_floor(t0, video.fps)
                f1 = sec_to_frame_ceil(t1, video.fps)

                if total_frames is not None:
                    f0 = max(0, min(f0, total_frames))
                    f1 = max(0, min(f1, total_frames))

                fr = FrameRange(start=f0, end=f1)
                out.append(Selector(kind="time", frame_ranges=(fr,), raw=raw))
                continue

        # Otherwise: chapter title
        if not video.chapters:
            print(
                f"[warn] Selector {raw!r} treated as chapter title, but video has no chapters. Skipping.",
                file=sys.stderr,
            )
            continue

        matches = find_chapter_ranges_by_title(video.chapters, raw)
        if not matches:
            print(
                f"[warn] Chapter title {raw!r} not found in source video chapters. Skipping.",
                file=sys.stderr,
            )
            continue

        frs: List[FrameRange] = []
        for (cs, ce, _title) in matches:
            f0 = sec_to_frame_floor(cs, video.fps)
            f1 = sec_to_frame_ceil(ce, video.fps)
            if total_frames is not None:
                f0 = max(0, min(f0, total_frames))
                f1 = max(0, min(f1, total_frames))
            frs.append(FrameRange(start=f0, end=f1))

        out.append(Selector(kind="chapter", frame_ranges=tuple(frs), raw=raw))

    return out


def parse_actions(action_tokens: List[str]) -> List[Action]:
    if len(action_tokens) % 2 != 0:
        raise ValueError("Actions must be pairs: param value param value ... (odd number of tokens found)")

    acts: List[Action] = []
    for i in range(0, len(action_tokens), 2):
        k = normalize_key(action_tokens[i])
        v = action_tokens[i + 1]
        acts.append(Action(key=k, raw_value=v))
    return acts


def load_commands(command_arg: str) -> List[str]:
    # If it's a real file path -> read lines
    if os.path.exists(command_arg) and os.path.isfile(command_arg):
        with open(command_arg, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        return project_zone_command_lines(lines)

    # Otherwise treat as inline:
    # delimiter is "?" (per spec). Also allow newlines if user passed them.
    if "\n" in command_arg:
        return project_zone_command_lines(command_arg.splitlines())

    return project_zone_command_lines([x.strip() for x in command_arg.split("?")])


def parse_commands(
    command_lines: List[str],
    *,
    video: VideoInfo,
    total_frames: Optional[int],
) -> List[Command]:
    cmds: List[Command] = []
    presets: Dict[str, List[str]] = {}
    for raw_line in command_lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("@"):
            name, action_tokens = parse_preset_definition(line)
            if name in presets:
                raise ValueError(f"Duplicate preset @{name}")
            presets[name] = action_tokens
            continue

        selectors_part, sep, action_tokens, option_tokens = split_command_tokens(line)
        selectors = parse_selectors(selectors_part, video=video, total_frames=total_frames)
        expanded_tokens = expand_preset_tokens(action_tokens, presets)
        actions = parse_actions(expanded_tokens)
        boundary_options = parse_boundary_options(option_tokens) if sep == "|" else None

        cmds.append(Command(
            selectors=tuple(selectors),
            sep=sep,
            actions=tuple(actions),
            raw_line=raw_line.strip(),
            boundary_options=boundary_options,
        ))
    return cmds


# ------------------------- video_params editing -------------------------

def parse_video_params(tokens: List[str]) -> List[Tuple[str, Optional[str]]]:
    """
    Parse token list like:
      ["--crf","31.50","--preset","2","--flag"]
    into:
      [("--crf","31.50"),("--preset","2"),("--flag",None)]
    """
    out: List[Tuple[str, Optional[str]]] = []
    i = 0
    while i < len(tokens):
        k = tokens[i]
        if not isinstance(k, str):
            k = str(k)
        if k.startswith("--"):
            v: Optional[str] = None
            if i + 1 < len(tokens) and not str(tokens[i + 1]).startswith("--"):
                v = str(tokens[i + 1])
                i += 2
            else:
                i += 1
            out.append((k, v))
        else:
            # stray token; keep as-is as a pseudo-flag to avoid data loss
            out.append((str(k), None))
            i += 1
    return out


def build_video_params(pairs: List[Tuple[str, Optional[str]]]) -> List[str]:
    out: List[str] = []
    for k, v in pairs:
        out.append(k)
        if v is not None:
            out.append(v)
    return out


def _is_number(s: str) -> bool:
    try:
        Decimal(str(s))
        return True
    except Exception:
        return False


def _quantize_to_step(val: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return val
    # nearest multiple of step
    q = (val / step).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return q * step


def _format_value(key: str, new_val: Decimal, old_str: Optional[str]) -> str:
    if key in FORCE_2DP:
        return f"{float(new_val):.2f}"

    if old_str is not None and "." in old_str:
        # preserve decimals count
        decimals = len(old_str.split(".", 1)[1])
        fmt = "{:." + str(decimals) + "f}"
        return fmt.format(float(new_val))

    # if integer-ish -> int
    if new_val == new_val.to_integral_value(rounding=ROUND_HALF_UP):
        return str(int(new_val))

    # fallback
    return str(new_val.normalize())


def apply_action_to_pairs(
    pairs: List[Tuple[str, Optional[str]]],
    action: Action,
    *,
    param_step: Dict[str, Decimal],
) -> Tuple[List[Tuple[str, Optional[str]]], Optional[str], Optional[str]]:
    """
    Returns updated pairs and (before, after) for this key (if changed).
    before/after are string representations.
    """
    key = action.key
    raw = action.raw_value.strip()

    # gather all occurrences
    idxs = [i for i, (k, _v) in enumerate(pairs) if k == key]

    def current_value_str() -> Optional[str]:
        if not idxs:
            return None
        return pairs[idxs[-1]][1]

    before = current_value_str()

    # deletion
    if raw == "-":
        if not idxs:
            return pairs, None, None
        new_pairs = [p for p in pairs if p[0] != key]
        return new_pairs, before, None

    if key != "--crf":
        if idxs:
            new_pairs = list(pairs)
            last_i = idxs[-1]
            new_pairs[last_i] = (key, raw)
            return new_pairs, before, raw
        new_pairs = list(pairs) + [(key, raw)]
        return new_pairs, None, raw

    # decide operation kind
    # percent forms: "+10%" "-10%" "+10%0.25" "-10%0.25"
    m_pct = re.match(r"^([+-])(\d+(?:\.\d+)?)%(\d+(?:\.\d+)?)?$", raw)
    m_add = re.match(r"^([+-])(\d+(?:\.\d+)?)$", raw)

    if m_pct or m_add:
        if before is None or before == "":
            # can't do relative change if missing
            return pairs, None, None
        if not _is_number(before):
            return pairs, None, None

        old = Decimal(before)
        new_val = old

        step: Optional[Decimal] = None

        if m_pct:
            sign = m_pct.group(1)
            pct = Decimal(m_pct.group(2)) / Decimal("100")
            if sign == "-":
                pct = -pct
            new_val = old * (Decimal("1") + pct)

            if m_pct.group(3) is not None:
                step = Decimal(m_pct.group(3))
            else:
                step = param_step.get(key)

        else:
            sign = m_add.group(1)
            delta = Decimal(m_add.group(2))
            if sign == "-":
                delta = -delta
            new_val = old + delta
            step = param_step.get(key)

        if step is not None:
            new_val = _quantize_to_step(new_val, step)

        after = _format_value(key, new_val, before)

        # update last occurrence
        new_pairs = list(pairs)
        last_i = idxs[-1]
        new_pairs[last_i] = (key, after)
        return new_pairs, before, after

    # absolute set
    # numeric -> keep numeric formatting rules; non-numeric -> set raw
    if idxs:
        new_pairs = list(pairs)
        last_i = idxs[-1]
        if _is_number(raw) and before is not None and _is_number(before):
            after = _format_value(key, Decimal(raw), before)
        else:
            after = raw
        new_pairs[last_i] = (key, after)
        return new_pairs, before, after

    # key not present -> append
    if _is_number(raw):
        after = _format_value(key, Decimal(raw), None)
    else:
        after = raw

    new_pairs = list(pairs) + [(key, after)]
    return new_pairs, None, after


# ------------------------- boundary editing -------------------------

@dataclass
class SceneSegment:
    start: int
    end: int
    scene: Dict[str, Any]
    selected: bool


def normalize_frame_ranges(ranges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    cleaned = sorted((int(a), int(b)) for a, b in ranges if int(b) > int(a))
    out: List[Tuple[int, int]] = []
    for start, end in cleaned:
        if not out:
            out.append((start, end))
            continue
        prev_start, prev_end = out[-1]
        if start <= prev_end:
            out[-1] = (prev_start, max(prev_end, end))
        else:
            out.append((start, end))
    return out


def command_frame_ranges(cmd: Command) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for sel in cmd.selectors:
        if sel.kind in ("frames", "time", "chapter"):
            for fr in sel.frame_ranges:
                ranges.append((fr.start, fr.end))
    return normalize_frame_ranges(ranges)


def scene_timeline(scenes: Sequence[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    starts: List[int] = []
    ends: List[int] = []
    for sc in scenes:
        try:
            starts.append(int(sc.get("start_frame")))
            ends.append(int(sc.get("end_frame")))
        except Exception:
            continue
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def scene_boundaries(scenes: Sequence[Dict[str, Any]]) -> List[int]:
    bounds: set[int] = set()
    for sc in scenes:
        try:
            bounds.add(int(sc.get("start_frame")))
            bounds.add(int(sc.get("end_frame")))
        except Exception:
            continue
    return sorted(bounds)


def clip_ranges_to_timeline(
    ranges: Sequence[Tuple[int, int]],
    timeline_start: int,
    timeline_end: int,
) -> List[Tuple[int, int]]:
    clipped: List[Tuple[int, int]] = []
    for start, end in ranges:
        s0 = max(timeline_start, min(int(start), timeline_end))
        s1 = max(timeline_start, min(int(end), timeline_end))
        if s1 > s0:
            clipped.append((s0, s1))
    return normalize_frame_ranges(clipped)


def nearest_boundary(pos: int, bounds: Sequence[int]) -> Tuple[Optional[int], int]:
    if not bounds:
        return None, 0
    best = min((int(b) for b in bounds), key=lambda b: (abs(b - pos), b))
    return best, abs(best - pos)


def snap_boundary_to_scene(pos: int, bounds: Sequence[int], opts: BoundaryOptions) -> int:
    nearest, dist = nearest_boundary(pos, bounds)
    if nearest is None or dist == 0:
        return pos
    if opts.threshold > 0 and dist <= opts.threshold:
        return nearest
    if opts.magnet > 0 and dist <= opts.magnet:
        return nearest
    return pos


def snap_ranges_to_scene_boundaries(
    ranges: Sequence[Tuple[int, int]],
    bounds: Sequence[int],
    opts: BoundaryOptions,
) -> List[Tuple[int, int]]:
    snapped: List[Tuple[int, int]] = []
    for start, end in ranges:
        s0 = snap_boundary_to_scene(int(start), bounds, opts)
        s1 = snap_boundary_to_scene(int(end), bounds, opts)
        if s1 > s0:
            snapped.append((s0, s1))
    return normalize_frame_ranges(snapped)


def merge_close_selector_ranges(
    ranges: Sequence[Tuple[int, int]],
    opts: BoundaryOptions,
) -> List[Tuple[int, int]]:
    if opts.min2_len <= 0:
        return normalize_frame_ranges(ranges)

    out: List[Tuple[int, int]] = []
    for start, end in normalize_frame_ranges(ranges):
        if not out:
            out.append((start, end))
            continue
        prev_start, prev_end = out[-1]
        gap = start - prev_end
        if 0 < gap < opts.min2_len:
            out[-1] = (prev_start, end)
        else:
            out.append((start, end))
    return out


def interval_inside_ranges(start: int, end: int, ranges: Sequence[Tuple[int, int]]) -> bool:
    for r0, r1 in ranges:
        if start >= r0 and end <= r1:
            return True
    return False


def split_scenes_for_boundary_ranges(
    scenes: Sequence[Dict[str, Any]],
    ranges: Sequence[Tuple[int, int]],
) -> List[SceneSegment]:
    cut_points = sorted({p for r in ranges for p in r})
    segments: List[SceneSegment] = []

    for sc in scenes:
        try:
            scene_start = int(sc.get("start_frame"))
            scene_end = int(sc.get("end_frame"))
        except Exception:
            continue
        if scene_end <= scene_start:
            continue

        points = [scene_start]
        points.extend(p for p in cut_points if scene_start < p < scene_end)
        points.append(scene_end)
        points = sorted(set(points))

        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            if end <= start:
                continue
            new_scene = copy.deepcopy(sc)
            new_scene["start_frame"] = start
            new_scene["end_frame"] = end
            segments.append(SceneSegment(
                start=start,
                end=end,
                scene=new_scene,
                selected=interval_inside_ranges(start, end, ranges),
            ))

    return segments


def short_segment_threshold(segments: Sequence[SceneSegment], idx: int, opts: BoundaryOptions) -> int:
    seg = segments[idx]
    special = idx == 0 or idx == len(segments) - 1
    if 0 < idx < len(segments) - 1:
        left = segments[idx - 1]
        right = segments[idx + 1]
        if left.selected == right.selected and seg.selected != left.selected:
            special = True
    if special:
        return opts.min2_len
    return opts.min_len


def choose_merge_direction(segments: Sequence[SceneSegment], idx: int) -> Optional[str]:
    seg = segments[idx]
    candidates: List[Tuple[int, str]] = []

    if idx > 0 and segments[idx - 1].selected == seg.selected:
        candidates.append((segments[idx - 1].end - segments[idx - 1].start, "left"))
    if idx + 1 < len(segments) and segments[idx + 1].selected == seg.selected:
        candidates.append((segments[idx + 1].end - segments[idx + 1].start, "right"))

    if not candidates:
        if idx > 0:
            candidates.append((segments[idx - 1].end - segments[idx - 1].start, "left"))
        if idx + 1 < len(segments):
            candidates.append((segments[idx + 1].end - segments[idx + 1].start, "right"))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], 1 if item[1] == "left" else 0), reverse=True)
    return candidates[0][1]


def merge_segment(segments: List[SceneSegment], idx: int, direction: str) -> None:
    seg = segments[idx]
    if direction == "left" and idx > 0:
        left = segments[idx - 1]
        left.end = seg.end
        del segments[idx]
        return
    if direction == "right" and idx + 1 < len(segments):
        right = segments[idx + 1]
        right.start = seg.start
        del segments[idx]


def merge_short_segments(segments: List[SceneSegment], opts: BoundaryOptions) -> List[SceneSegment]:
    if not segments:
        return segments
    if opts.min_len <= 0 and opts.min2_len <= 0:
        return segments

    changed = True
    while changed:
        changed = False
        for idx, seg in enumerate(list(segments)):
            if idx >= len(segments):
                break
            length = seg.end - seg.start
            threshold = short_segment_threshold(segments, idx, opts)
            if threshold <= 0 or length >= threshold:
                continue
            direction = choose_merge_direction(segments, idx)
            if direction is None:
                continue
            merge_segment(segments, idx, direction)
            changed = True
            break

    return segments


def materialize_segments(segments: Sequence[SceneSegment]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seg in segments:
        if seg.end <= seg.start:
            continue
        sc = seg.scene
        sc["start_frame"] = int(seg.start)
        sc["end_frame"] = int(seg.end)
        out.append(sc)
    return out


def align_scene_boundaries_for_command(
    scenes: List[Dict[str, Any]],
    cmd: Command,
    cmd_index: int,
) -> set[int]:
    opts = cmd.boundary_options
    if opts is None:
        return set()

    timeline = scene_timeline(scenes)
    if timeline is None:
        return set()
    timeline_start, timeline_end = timeline

    ranges = command_frame_ranges(cmd)
    ranges = clip_ranges_to_timeline(ranges, timeline_start, timeline_end)
    if not ranges:
        return set()

    before_count = len(scenes)
    bounds = scene_boundaries(scenes)
    ranges = snap_ranges_to_scene_boundaries(ranges, bounds, opts)
    ranges = merge_close_selector_ranges(ranges, opts)
    if not ranges:
        return set()

    segments = split_scenes_for_boundary_ranges(scenes, ranges)
    segments = merge_short_segments(segments, opts)

    selected_indexes = {idx for idx, seg in enumerate(segments) if seg.selected}
    scenes[:] = materialize_segments(segments)

    after_count = len(scenes)
    ranges_text = format_frame_group(ranges)
    print(
        f"[boundaries] cmd#{cmd_index} {ranges_text} "
        f"scenes {before_count}->{after_count}, selected={format_scene_group(sorted(selected_indexes))}"
    )
    return selected_indexes


# ------------------------- scene matching -------------------------

def overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def match_scene_by_range(scene_start: int, scene_end: int, fr: FrameRange, sep: str) -> bool:
    ol = overlap_len(scene_start, scene_end, fr.start, fr.end)
    if ol <= 0:
        return False
    scene_len = max(1, scene_end - scene_start)

    if sep == "!":
        return ol > 0
    if sep == "^":
        return scene_start >= fr.start and scene_end <= fr.end
    # default '-': at least half of the scene in range
    return (ol * 2) >= scene_len


def scene_selected(scene_idx: int, scene_start: int, scene_end: int, cmd: Command) -> bool:
    # Index selectors ignore sep (per spec)
    for sel in cmd.selectors:
        if sel.kind == "scene_idx":
            for (a, b) in sel.scene_ranges:
                if a <= scene_idx <= b:
                    return True

    # Other selectors use sep logic (union across selectors and ranges)
    for sel in cmd.selectors:
        if sel.kind in ("frames", "time", "chapter"):
            for fr in sel.frame_ranges:
                if match_scene_by_range(scene_start, scene_end, fr, cmd.sep):
                    return True

    return False


def scene_index_selected(scene_idx: int, cmd: Command) -> bool:
    for sel in cmd.selectors:
        if sel.kind != "scene_idx":
            continue
        for (a, b) in sel.scene_ranges:
            if a <= scene_idx <= b:
                return True
    return False


def selector_matches(scene_idx: int, scene_start: int, scene_end: int, cmd: Command) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    scene_len = max(1, scene_end - scene_start)

    for sel in cmd.selectors:
        if sel.kind == "scene_idx":
            for (a, b) in sel.scene_ranges:
                if a <= scene_idx <= b:
                    matches.append({
                        "kind": "scene_idx",
                        "range": [a, b],
                        "raw": sel.raw,
                    })
        else:
            for fr in sel.frame_ranges:
                ol = overlap_len(scene_start, scene_end, fr.start, fr.end)
                if ol <= 0:
                    continue
                matches.append({
                    "kind": sel.kind,
                    "range": [fr.start, fr.end],
                    "raw": sel.raw,
                    "rule": cmd.sep,
                    "overlap_frames": ol,
                    "overlap_ratio": round(float(ol) / float(scene_len), 4),
                })

    return matches


def format_scene_group(values: Sequence[int]) -> str:
    ordered: List[int] = []
    for value in values:
        ivalue = int(value)
        if ordered and ordered[-1] == ivalue:
            continue
        ordered.append(ivalue)

    if not ordered:
        return "-"

    parts: List[str] = []
    start = end = ordered[0]
    for value in ordered[1:]:
        if value == end + 1:
            end = value
            continue
        parts.append(str(start) if start == end else f"{start}..{end}")
        start = end = value
    parts.append(str(start) if start == end else f"{start}..{end}")
    return ",".join(parts)


def format_frame_group(ranges: Sequence[Tuple[int, int]]) -> str:
    normalized: List[Tuple[int, int]] = []
    for start, end in ranges:
        s0 = int(start)
        s1 = int(end)
        if not normalized:
            normalized.append((s0, s1))
            continue
        prev_start, prev_end = normalized[-1]
        if s0 <= prev_end:
            normalized[-1] = (prev_start, max(prev_end, s1))
            continue
        normalized.append((s0, s1))
    return ",".join(f"{start}-{end}" for start, end in normalized) if normalized else "-"


def render_command_actions(cmd: Command) -> str:
    try:
        _selectors_part, _sep, action_tokens, _option_tokens = split_command_tokens(cmd.raw_line)
    except ValueError:
        parts: List[str] = []
        for action in cmd.actions:
            parts.append(action.key)
            parts.append(action.raw_value)
        return " ".join(parts).strip()
    return " ".join(action_tokens).strip()


def render_boundary_options(opts: BoundaryOptions) -> str:
    return (
        f"min={opts.min_len} "
        f"min2={opts.min2_len} "
        f"magnet={opts.magnet} "
        f"thr={opts.threshold}"
    )


def render_command_context(cmd: Command, matches: Sequence[Dict[str, Any]]) -> str:
    matched_raws = {str(item.get("raw") or "") for item in matches if item.get("raw")}
    selector_parts: List[str] = []
    for selector in cmd.selectors:
        raw = selector.raw
        if raw in matched_raws:
            selector_parts.append(f"[ {raw} ]")
        else:
            selector_parts.append(raw)
    selectors_text = ",".join(selector_parts)
    actions_text = render_command_actions(cmd)
    if cmd.sep == "|" and cmd.boundary_options is not None:
        options_text = render_boundary_options(cmd.boundary_options)
        if actions_text:
            return f"{selectors_text} | {options_text} | {actions_text}"
        return f"{selectors_text} | {options_text} |"
    if actions_text:
        return f"{selectors_text} {cmd.sep} {actions_text}"
    return selectors_text


# ------------------------- main apply -------------------------

def apply_commands_to_scenes(
    scenes_data: Dict[str, Any],
    commands: List[Command],
    *,
    param_step: Dict[str, Decimal],
    video: VideoInfo,
) -> None:
    scenes = scenes_data.get("scenes") or []
    if not isinstance(scenes, list):
        raise RuntimeError("Invalid scenes.json: top-level 'scenes' must be a list.")

    log_groups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    for idx, sc in enumerate(scenes):
        try:
            s0 = int(sc.get("start_frame"))
            s1 = int(sc.get("end_frame"))
        except Exception:
            continue
        ensure_scene_meta(sc, idx, s0, s1, video.fps)

    for cmd_i, cmd in enumerate(commands, start=1):
        boundary_selected: Optional[set[int]] = None
        if cmd.boundary_options is not None:
            boundary_selected = align_scene_boundaries_for_command(scenes, cmd, cmd_i)
            for idx, sc in enumerate(scenes):
                try:
                    s0 = int(sc.get("start_frame"))
                    s1 = int(sc.get("end_frame"))
                except Exception:
                    continue
                ensure_scene_meta(sc, idx, s0, s1, video.fps)

        for idx, sc in enumerate(scenes):
            try:
                s0 = int(sc.get("start_frame"))
                s1 = int(sc.get("end_frame"))
            except Exception:
                continue

            if boundary_selected is not None:
                if idx not in boundary_selected and not scene_index_selected(idx, cmd):
                    continue
            elif not scene_selected(idx, s0, s1, cmd):
                continue

            zo = sc.get("zone_overrides")
            if zo is None or not isinstance(zo, dict):
                zo = {}
                sc["zone_overrides"] = zo

            vp = zo.get("video_params")
            if vp is None:
                vp = []
                zo["video_params"] = vp
            if not isinstance(vp, list):
                # if someone stored as string, try split
                if isinstance(vp, str):
                    vp = shlex.split(vp, posix=False)
                    zo["video_params"] = vp
                else:
                    vp = []
                    zo["video_params"] = vp

            pairs = parse_video_params([str(x) for x in vp])

            changes: List[str] = []
            changes_detail: List[Dict[str, Optional[str]]] = []
            for act in cmd.actions:
                new_pairs, before, after = apply_action_to_pairs(pairs, act, param_step=param_step)
                if new_pairs is not pairs:
                    # detect actual change for this key
                    if before != after:
                        if before is None:
                            changes.append(f"{act.key}: (add) -> {after}")
                        elif after is None:
                            changes.append(f"{act.key}: {before} -> (removed)")
                        else:
                            changes.append(f"{act.key}: {before} -> {after}")
                        changes_detail.append({
                            "key": act.key,
                            "before": before,
                            "after": after,
                        })
                pairs = new_pairs

            if changes:
                zo["video_params"] = build_video_params(pairs)
                meta = ensure_scene_meta(sc, idx, s0, s1, video.fps)
                ze = meta.get("zone_editor")
                if not isinstance(ze, dict):
                    ze = {}
                    meta["zone_editor"] = ze
                applied = ze.get("applied_commands")
                if not isinstance(applied, list):
                    applied = []
                    ze["applied_commands"] = applied
                match_info = selector_matches(idx, s0, s1, cmd)
                applied.append({
                    "cmd_index": cmd_i,
                    "cmd_line": cmd.raw_line,
                    "selectors": [sel.raw for sel in cmd.selectors],
                    "sep": cmd.sep,
                    "matches": match_info,
                    "changes": changes_detail,
                })
                ze["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                command_context = render_command_context(cmd, match_info)
                group_key = (cmd_i, command_context, tuple(changes))
                group = log_groups.get(group_key)
                if group is None:
                    group = {
                        "cmd_index": cmd_i,
                        "command_context": command_context,
                        "changes": tuple(changes),
                        "scene_indexes": [],
                        "frame_ranges": [],
                    }
                    log_groups[group_key] = group
                group["scene_indexes"].append(idx)
                group["frame_ranges"].append((s0, s1))

    for idx, sc in enumerate(scenes):
        try:
            s0 = int(sc.get("start_frame"))
            s1 = int(sc.get("end_frame"))
        except Exception:
            continue
        ensure_scene_meta(sc, idx, s0, s1, video.fps)

    for group in log_groups.values():
        scene_text = format_scene_group(group["scene_indexes"])
        frame_text = format_frame_group(group["frame_ranges"])
        print(f"[scene {scene_text:>8}  {frame_text}]  - cmd#{group['cmd_index']} - {group['command_context']}")
        for change in group["changes"]:
            print(f"  {change}")

    scenes_data["split_scenes"] = copy.deepcopy(scenes)


# ------------------------- CLI -------------------------

def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description="Edit av1an scenes.json using a simple command DSL.")
    ap.add_argument("--scenes", required=False, help="Input scenes.json (av1an scenes file)")
    ap.add_argument("--out", required=False, help="Output scenes.json path")
    ap.add_argument("--source", required=False, help="Source video file (for chapters/time->frames mapping)")
    ap.add_argument(
        "--command",
        required=False,
        help=(
            "Commands: file path (newline-separated, supports # comments and @presets) "
            "OR inline string (separator '?'). Boundary mode: "
            "selectors | min=9 min2=4 magnet=2 thr=1 | actions"
        ),
    )
    ap.add_argument("--no-vfr-warn", action="store_true", help="Do not warn when avg_frame_rate differs from r_frame_rate")
    ap.add_argument("--parse-check", action="store_true", help="Parse-check command(s) and exit without writing output.")
    ap.add_argument("--log", default="", help="Optional log file path (relative to --out dir if not absolute)")
    args = ap.parse_args(list(argv))

    if args.parse_check:
        if not args.source:
            raise SystemExit("--source is required for --parse-check")
        if not args.command:
            raise SystemExit("--command is required for --parse-check")
        base_dir = Path.cwd()
        if args.out:
            base_dir = Path(args.out).resolve().parent
        setup_logging(args.log, base_dir)
        video = read_video_info(args.source)
        raw_lines = load_commands(args.command)
        commands = parse_commands(raw_lines, video=video, total_frames=None)
        print(f"[ok] parse-check: {len(commands)} commands")
        return 0

    if not args.scenes:
        raise SystemExit("--scenes is required")
    if not args.out:
        raise SystemExit("--out is required")
    if not args.source:
        raise SystemExit("--source is required")
    if not args.command:
        raise SystemExit("--command is required")

    out_path = Path(args.out).resolve()
    setup_logging(args.log, out_path.parent)

    # Load scenes json
    with open(args.scenes, "r", encoding="utf-8") as f:
        scenes_data = json.load(f)

    total_frames = scenes_data.get("frames")
    if total_frames is not None:
        try:
            total_frames = int(total_frames)
        except Exception:
            total_frames = None

    # Video info
    video = read_video_info(args.source)

    if video.is_vfr_suspected and not args.no_vfr_warn:
        print("[warn] Source looks like it may be VFR (avg_frame_rate != r_frame_rate). "
              "Time/chapter -> frame conversion uses avg_frame_rate and may be approximate.",
              file=sys.stderr)

    # Commands
    raw_lines = load_commands(args.command)
    commands = parse_commands(raw_lines, video=video, total_frames=total_frames)

    # Apply
    apply_commands_to_scenes(scenes_data, commands, param_step=DEFAULT_PARAM_STEP, video=video)

    # Save
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scenes_data, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
