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
import atexit
import copy
import json
import math
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ------------------------- state markers -------------------------

STATE_DIR_NAME = ".state"
ZONE_MARKER = "ZONE_EDIT_DONE"


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


def setup_logging(log_path: str, base_dir: Optional[Path] = None) -> None:
    if not log_path:
        return
    p = Path(log_path)
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    p.parent.mkdir(parents=True, exist_ok=True)
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    log_fh = p.open("a", encoding=enc, errors="replace")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        log_fh.write(f"=== START zone-editor {ts} ===\n")
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
            log_fh.write(f"=== END zone-editor {ts_end} ===\n")
            log_fh.flush()
        except Exception:
            pass
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        tee_out.close_log()
        tee_err.close_log()

    atexit.register(_cleanup)


def state_root_from_out(out_path: Path) -> Path:
    p = out_path.resolve()
    if p.parent.name.lower() == "video" and p.parent.parent.exists():
        return p.parent.parent
    return p.parent


def marker_path(out_path: Path) -> Path:
    root = state_root_from_out(out_path)
    return root / STATE_DIR_NAME / ZONE_MARKER


def write_marker(out_path: Path) -> None:
    p = marker_path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("ok\n", encoding="utf-8")


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
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
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
        tags = ch.get("tags") or {}
        title = str(tags.get("title") or "").strip()
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
class Command:
    selectors: Tuple[Selector, ...]
    sep: str  # '-', '!', '^'
    actions: Tuple[Action, ...]
    raw_line: str


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


def split_command_tokens(line: str) -> Tuple[str, str, List[str]]:
    """
    Returns: (selectors_part, sep, action_tokens)
    Parsing strategy: shlex.split -> find first token that is exactly '-', '!', '^'
    """
    tokens = shlex.split(line, posix=False)
    if not tokens:
        raise ValueError("Empty command line")

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

    return selectors_part, sep, action_tokens


def parse_preset_definition(line: str) -> Tuple[str, List[str]]:
    selectors_part, _sep, action_tokens = split_command_tokens(line)
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
    q = title_query.strip().casefold()
    if not q:
        return []

    # 1) exact match (case-insensitive)
    exact = [c for c in chapters if c.title.strip().casefold() == q]
    if exact:
        return [(c.start_sec, c.end_sec, c.title) for c in exact]

    # 2) substring match
    sub = [c for c in chapters if q in c.title.strip().casefold()]
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
            fr = FrameRange(start=a, end=b)
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
            t0 = parse_time_token(m.group(1))
            t1 = parse_time_token(m.group(2))
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
            raise ValueError(f"Selector {raw!r} treated as chapter title, but video has no chapters.")

        matches = find_chapter_ranges_by_title(video.chapters, raw)
        if not matches:
            raise ValueError(f"Chapter title {raw!r} not found in source video chapters.")

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
        return lines

    # Otherwise treat as inline:
    # delimiter is "?" (per spec). Also allow newlines if user passed them.
    if "\n" in command_arg:
        return command_arg.splitlines()

    return [x.strip() for x in command_arg.split("?")]


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

        selectors_part, sep, action_tokens = split_command_tokens(line)
        selectors = parse_selectors(selectors_part, video=video, total_frames=total_frames)
        expanded_tokens = expand_preset_tokens(action_tokens, presets)
        actions = parse_actions(expanded_tokens)

        cmds.append(Command(
            selectors=tuple(selectors),
            sep=sep,
            actions=tuple(actions),
            raw_line=raw_line.strip(),
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

    for idx, sc in enumerate(scenes):
        try:
            s0 = int(sc.get("start_frame"))
            s1 = int(sc.get("end_frame"))
        except Exception:
            continue
        ensure_scene_meta(sc, idx, s0, s1, video.fps)

    for cmd_i, cmd in enumerate(commands, start=1):
        for idx, sc in enumerate(scenes):
            try:
                s0 = int(sc.get("start_frame"))
                s1 = int(sc.get("end_frame"))
            except Exception:
                continue

            if not scene_selected(idx, s0, s1, cmd):
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
                print(f"[scene {idx:4d}  {s0}-{s1})  cmd#{cmd_i}: {cmd.raw_line}")
                for c in changes:
                    print(f"  - {c}")
                meta = ensure_scene_meta(sc, idx, s0, s1, video.fps)
                ze = meta.get("zone_editor")
                if not isinstance(ze, dict):
                    ze = {}
                    meta["zone_editor"] = ze
                applied = ze.get("applied_commands")
                if not isinstance(applied, list):
                    applied = []
                    ze["applied_commands"] = applied
                applied.append({
                    "cmd_index": cmd_i,
                    "cmd_line": cmd.raw_line,
                    "selectors": [sel.raw for sel in cmd.selectors],
                    "sep": cmd.sep,
                    "matches": selector_matches(idx, s0, s1, cmd),
                    "changes": changes_detail,
                })
                ze["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    scenes_data["split_scenes"] = copy.deepcopy(scenes)


# ------------------------- CLI -------------------------

def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description="Edit av1an scenes.json using a simple command DSL.")
    ap.add_argument("--scenes", required=False, help="Input scenes.json (av1an scenes file)")
    ap.add_argument("--out", required=False, help="Output scenes.json path")
    ap.add_argument("--source", required=False, help="Source video file (for chapters/time->frames mapping)")
    ap.add_argument("--command", required=False, help="Commands: file path (newline-separated, supports # comments and @presets) OR inline string (separator '?')")
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
    marker = marker_path(out_path)
    if marker.exists():
        print(f"[skip] marker exists: {marker}")
        return 0

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
    write_marker(out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
