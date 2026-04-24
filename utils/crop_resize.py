from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.zoned_commands import project_crop_resize_command_lines


_RX_FRAME = re.compile(r"^(\d+)f(\d+)$", re.IGNORECASE)
_RX_TIME = re.compile(r"^([0-9:.,]+)t([0-9:.,]+)$", re.IGNORECASE)
_SIZE_SEP = r"[:xX*/]"
_RX_SIZE = re.compile(rf"^(\d+)\s*{_SIZE_SEP}\s*(\d+)$", re.IGNORECASE)
_RX_CROP = re.compile(rf"^crop=(\d+)\s*{_SIZE_SEP}\s*(\d+)(?::(-?\d+):(-?\d+))?$", re.IGNORECASE)
_RX_SCALE = re.compile(rf"^scale=(-?\d+)\s*{_SIZE_SEP}\s*(-?\d+)$", re.IGNORECASE)
_RX_FUNCTION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class CropResizeOp:
    kind: str
    width: int
    height: int
    x: Optional[int] = None
    y: Optional[int] = None


@dataclass(frozen=True)
class FrameRange:
    start: int
    end: int


@dataclass(frozen=True)
class CropResizeRule:
    selectors: Tuple[str, ...]
    ops: Tuple[CropResizeOp, ...]
    raw_line: str
    end_ops: Optional[Tuple[CropResizeOp, ...]] = None

    @property
    def is_transition(self) -> bool:
        return self.end_ops is not None


@dataclass(frozen=True)
class CropResizePlan:
    default_ops: Tuple[CropResizeOp, ...]
    rules: Tuple[CropResizeRule, ...]
    target_width: int
    target_height: int

    @property
    def enabled(self) -> bool:
        return bool(self.default_ops)


@dataclass(frozen=True)
class Chapter:
    title: str
    start_sec: float
    end_sec: float


EMPTY_PLAN = CropResizePlan(default_ops=(), rules=(), target_width=0, target_height=0)


def parse_time_token(raw: str) -> float:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty time token")
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"invalid time token: {raw!r}")
    try:
        values = [float(part.replace(",", ".")) for part in parts]
    except ValueError as exc:
        raise ValueError(f"invalid time token: {raw!r}") from exc
    seconds = 0.0
    for value in values:
        seconds = seconds * 60.0 + value
    return seconds


def sec_to_frame_floor(sec: float, fps: Fraction) -> int:
    return int(Fraction(sec).limit_denominator(1000000) * fps)


def sec_to_frame_ceil(sec: float, fps: Fraction) -> int:
    value = Fraction(sec).limit_denominator(1000000) * fps
    return int((value.numerator + value.denominator - 1) // value.denominator)


def _strip_line(raw: str) -> str:
    line = str(raw or "").strip()
    if not line or line.startswith("#") or line.startswith("//"):
        return ""
    return line


def load_crop_resize_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _split_rule_line(line: str) -> Tuple[str, str]:
    if " - " in line:
        left, right = line.split(" - ", 1)
    else:
        parts = line.split(None, 2)
        if len(parts) < 3 or parts[1] != "-":
            raise ValueError("expected '<selector> - <crop/resize chain>'")
        left, right = parts[0], parts[2]
    left = left.strip()
    right = right.strip()
    if not left:
        raise ValueError("missing selector")
    if not right:
        raise ValueError("missing crop/resize chain")
    return left, right


def _parse_op(token: str) -> CropResizeOp:
    text = token.strip()
    m = _RX_CROP.match(text)
    if m:
        width = int(m.group(1))
        height = int(m.group(2))
        x = int(m.group(3)) if m.group(3) is not None else None
        y = int(m.group(4)) if m.group(4) is not None else None
        return CropResizeOp("crop", width, height, x, y)
    m = _RX_SCALE.match(text)
    if m:
        width = int(m.group(1))
        height = int(m.group(2))
        if width == 0 or height == 0:
            raise ValueError(f"scale dimensions cannot be 0: {text}")
        if width < 0 and height < 0:
            raise ValueError(f"scale cannot use -1 for both dimensions: {text}")
        return CropResizeOp("scale", width, height)
    raise ValueError(f"unsupported crop/resize operation: {text!r}")


def _parse_chain(chain: str) -> Tuple[CropResizeOp, ...]:
    short_size = _RX_SIZE.match(str(chain or "").strip())
    if short_size:
        return (CropResizeOp("crop", int(short_size.group(1)), int(short_size.group(2))),)
    parts = [part.strip() for part in str(chain or "").split(",") if part.strip()]
    if not parts:
        raise ValueError("empty crop/resize chain")
    return tuple(_parse_op(part) for part in parts)


def _target_from_ops(ops: Sequence[CropResizeOp]) -> Tuple[int, int]:
    for op in reversed(ops):
        if op.width > 0 and op.height > 0:
            return op.width, op.height
    raise ValueError("@default must define a fixed target size, for example: @default - crop=3840:1608")


def _expand_short_rule(size_text: str, target_width: int, target_height: int) -> Tuple[CropResizeOp, ...]:
    m = _RX_SIZE.match(size_text.strip())
    if not m:
        raise ValueError(f"invalid short crop size: {size_text!r}")
    source_width = int(m.group(1))
    source_height = int(m.group(2))
    return (
        CropResizeOp("crop", source_width, source_height),
        CropResizeOp("scale", -1, target_height),
        CropResizeOp("crop", target_width, target_height),
    )


def _parse_rule_chain(chain_text: str, target_width: int, target_height: int) -> Tuple[CropResizeOp, ...]:
    text = str(chain_text or "").strip()
    if _RX_SIZE.match(text):
        return _expand_short_rule(text, target_width, target_height)
    return _parse_chain(text)


def _validate_transition_ops(
    start_ops: Sequence[CropResizeOp],
    end_ops: Sequence[CropResizeOp],
    *,
    raw_line: str,
) -> None:
    if len(start_ops) != len(end_ops):
        raise ValueError(
            f"transition requires matching crop/resize chains on both sides of '->': {raw_line}"
        )
    for index, (start_op, end_op) in enumerate(zip(start_ops, end_ops), start=1):
        if start_op.kind != end_op.kind:
            raise ValueError(
                f"transition requires matching operation kinds at step {index}: {raw_line}"
            )
        if start_op.kind != "crop" and start_op != end_op:
            raise ValueError(
                f"transition currently supports animating crop sizes only; step {index} must stay unchanged: {raw_line}"
            )
        if start_op.kind == "crop" and (start_op.x != end_op.x or start_op.y != end_op.y):
            raise ValueError(
                f"transition currently supports crop size changes only; x/y offsets must match: {raw_line}"
            )


def _parse_rule_ops(
    chain_text: str,
    *,
    target_width: int,
    target_height: int,
    raw_line: str,
) -> Tuple[Tuple[CropResizeOp, ...], Optional[Tuple[CropResizeOp, ...]]]:
    text = str(chain_text or "").strip()
    if "->" not in text:
        return _parse_rule_chain(text, target_width, target_height), None
    left, right = (part.strip() for part in text.split("->", 1))
    if not left or not right:
        raise ValueError(f"invalid transition syntax: {raw_line}")
    start_ops = _parse_rule_chain(left, target_width, target_height)
    end_ops = _parse_rule_chain(right, target_width, target_height)
    _validate_transition_ops(start_ops, end_ops, raw_line=raw_line)
    return start_ops, end_ops


def parse_crop_resize_lines(lines: Iterable[str]) -> CropResizePlan:
    default_ops: Optional[Tuple[CropResizeOp, ...]] = None
    raw_rules: List[Tuple[Tuple[str, ...], str, str]] = []

    for raw in project_crop_resize_command_lines(lines):
        line = _strip_line(raw)
        if not line:
            continue
        selector_text, chain_text = _split_rule_line(line)
        if _RX_FUNCTION_NAME.match(chain_text):
            continue
        if selector_text == "@default":
            default_ops = _parse_chain(chain_text)
            continue
        selectors = tuple(part.strip() for part in selector_text.split(",") if part.strip())
        if not selectors:
            raise ValueError(f"no selectors in line: {line}")
        raw_rules.append((selectors, chain_text, line))

    if default_ops is None:
        if raw_rules:
            raise ValueError("crop/resize rules require '@default - crop=<w>:<h>'")
        return EMPTY_PLAN

    target_width, target_height = _target_from_ops(default_ops)
    rules: List[CropResizeRule] = []
    for selectors, chain_text, raw_line in raw_rules:
        ops, end_ops = _parse_rule_ops(
            chain_text,
            target_width=target_width,
            target_height=target_height,
            raw_line=raw_line,
        )
        rules.append(CropResizeRule(selectors=selectors, ops=ops, raw_line=raw_line, end_ops=end_ops))

    return CropResizePlan(
        default_ops=default_ops,
        rules=tuple(rules),
        target_width=target_width,
        target_height=target_height,
    )


def load_crop_resize_plan(path: Path) -> CropResizePlan:
    return parse_crop_resize_lines(load_crop_resize_lines(path))


def _run_ffprobe_json(args: List[str]) -> Dict[str, Any]:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json"] + args
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() or f"exit code {proc.returncode}"
        raise RuntimeError(f"ffprobe failed: {detail}")
    return json.loads(proc.stdout or "{}")


def load_chapters(source: Path) -> List[Chapter]:
    data = _run_ffprobe_json(["-show_chapters", str(source)])
    chapters: List[Chapter] = []
    for item in data.get("chapters") or []:
        tags = item.get("tags") if isinstance(item, dict) else {}
        title = str((tags or {}).get("title") or "").strip()
        if not title:
            continue
        start = float(item.get("start_time") or 0.0)
        end = float(item.get("end_time") or start)
        if end <= start:
            continue
        chapters.append(Chapter(title=title, start_sec=start, end_sec=end))
    return chapters


def _normalize_title(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _chapter_ranges(selector: str, chapters: Sequence[Chapter], fps: Fraction, total_frames: int) -> List[FrameRange]:
    query = _normalize_title(selector)
    exact = [chapter for chapter in chapters if _normalize_title(chapter.title) == query]
    matches = exact or [chapter for chapter in chapters if query and query in _normalize_title(chapter.title)]
    if not matches:
        raise ValueError(f"chapter selector not found: {selector!r}")
    out: List[FrameRange] = []
    for chapter in matches:
        start = max(0, min(sec_to_frame_floor(chapter.start_sec, fps), total_frames))
        end = max(0, min(sec_to_frame_ceil(chapter.end_sec, fps), total_frames))
        if end > start:
            out.append(FrameRange(start, end))
    return out


def resolve_selector(selector: str, *, fps: Fraction, total_frames: int, chapters: Sequence[Chapter]) -> List[FrameRange]:
    raw = str(selector or "").strip()
    m = _RX_FRAME.match(raw)
    if m:
        start = int(m.group(1))
        end = int(m.group(2))
        if end < start:
            start, end = end, start
        start = max(0, min(start, total_frames))
        end = max(0, min(end + 1, total_frames))
        return [FrameRange(start, end)] if end > start else []

    m = _RX_TIME.match(raw)
    if m:
        start_sec = parse_time_token(m.group(1))
        end_sec = parse_time_token(m.group(2))
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
        start = max(0, min(sec_to_frame_floor(start_sec, fps), total_frames))
        end = max(0, min(sec_to_frame_ceil(end_sec, fps), total_frames))
        return [FrameRange(start, end)] if end > start else []

    return _chapter_ranges(raw, chapters, fps, total_frames)


def resolve_rule_ranges(
    plan: CropResizePlan,
    *,
    fps: Fraction,
    total_frames: int,
    source: Optional[Path] = None,
) -> List[List[FrameRange]]:
    chapters: List[Chapter] = []
    needs_chapters = any(
        not _RX_FRAME.match(selector) and not _RX_TIME.match(selector)
        for rule in plan.rules
        for selector in rule.selectors
    )
    if needs_chapters:
        if source is None:
            raise ValueError("chapter selectors require source path")
        chapters = load_chapters(source)

    resolved: List[List[FrameRange]] = []
    for rule in plan.rules:
        ranges: List[FrameRange] = []
        for selector in rule.selectors:
            ranges.extend(resolve_selector(selector, fps=fps, total_frames=total_frames, chapters=chapters))
        resolved.append(ranges)
    return resolved


def validate_crop_resize_plan(plan: CropResizePlan, *, source: Optional[Path] = None, fps: Fraction = Fraction(24000, 1001), total_frames: int = 1_000_000) -> None:
    if not plan.enabled:
        return
    resolve_rule_ranges(plan, fps=fps, total_frames=total_frames, source=source)


def _even(value: int) -> int:
    return max(2, int(value) - (int(value) % 2))


def _split_even_padding(delta: int) -> Tuple[int, int]:
    left = (int(delta) // 2) & ~1
    right = int(delta) - left
    if right % 2:
        right += 1
    return left, right


def _resize_dimensions(width: int, height: int, current_width: int, current_height: int) -> Tuple[int, int]:
    if width < 0:
        width = round(current_width * (height / current_height))
    if height < 0:
        height = round(current_height * (width / current_width))
    return _even(width), _even(height)


def _pad_to_at_least(clip: Any, width: int, height: int) -> Any:
    target_width = int(width)
    target_height = int(height)
    resize_width = int(clip.width)
    resize_height = int(clip.height)
    if target_width > resize_width and (target_width - resize_width) % 2 and resize_width > 2:
        resize_width -= 1
    if target_height > resize_height and (target_height - resize_height) % 2 and resize_height > 2:
        resize_height -= 1
    if resize_width != int(clip.width) or resize_height != int(clip.height):
        clip = clip.resize.Lanczos(width=resize_width, height=resize_height)

    add_w = max(0, int(width) - int(clip.width))
    add_h = max(0, int(height) - int(clip.height))
    if add_w <= 0 and add_h <= 0:
        return clip
    left, right = _split_even_padding(add_w)
    top, bottom = _split_even_padding(add_h)
    return clip.std.AddBorders(left=left, right=right, top=top, bottom=bottom)


def _crop_clip(clip: Any, op: CropResizeOp) -> Any:
    width = int(op.width)
    height = int(op.height)
    clip = _pad_to_at_least(clip, width, height)
    x = int(op.x) if op.x is not None else max(0, (int(clip.width) - width) // 2)
    y = int(op.y) if op.y is not None else max(0, (int(clip.height) - height) // 2)
    x = max(0, min(x, int(clip.width) - width))
    y = max(0, min(y, int(clip.height) - height))
    right = int(clip.width) - x - width
    bottom = int(clip.height) - y - height
    return clip.std.Crop(left=x, right=right, top=y, bottom=bottom)


def normalize_canvas(clip: Any, width: int, height: int) -> Any:
    clip = _pad_to_at_least(clip, width, height)
    if int(clip.width) != int(width) or int(clip.height) != int(height):
        clip = _crop_clip(clip, CropResizeOp("crop", int(width), int(height)))
    return clip


def _lerp_even(start: int, end: int, t: float) -> int:
    value = float(start) + (float(end) - float(start)) * float(t)
    return max(2, int(round(value / 2.0)) * 2)


def _interpolate_ops(
    start_ops: Sequence[CropResizeOp],
    end_ops: Sequence[CropResizeOp],
    *,
    t: float,
) -> Tuple[CropResizeOp, ...]:
    out: List[CropResizeOp] = []
    for start_op, end_op in zip(start_ops, end_ops):
        if start_op == end_op:
            out.append(start_op)
            continue
        out.append(
            CropResizeOp(
                start_op.kind,
                _lerp_even(start_op.width, end_op.width, t),
                _lerp_even(start_op.height, end_op.height, t),
                start_op.x,
                start_op.y,
            )
        )
    return tuple(out)


def _transition_progress(frame_no: int, frame_range: FrameRange) -> float:
    span = int(frame_range.end) - int(frame_range.start)
    if span <= 1:
        return 0.0
    pos = max(0, min(int(frame_no) - int(frame_range.start), span - 1))
    return float(pos) / float(span - 1)


def apply_ops_to_clip(clip: Any, ops: Sequence[CropResizeOp], *, target_width: int, target_height: int) -> Any:
    out = clip
    for op in ops:
        if op.kind == "crop":
            out = _crop_clip(out, op)
        elif op.kind == "scale":
            width, height = _resize_dimensions(op.width, op.height, int(out.width), int(out.height))
            out = out.resize.Lanczos(width=width, height=height)
        else:
            raise ValueError(f"unsupported crop/resize op: {op.kind}")
    return normalize_canvas(out, target_width, target_height)


def _fps_from_clip(clip: Any) -> Fraction:
    num = int(getattr(clip, "fps_num", 0) or 0)
    den = int(getattr(clip, "fps_den", 0) or 0)
    if num > 0 and den > 0:
        return Fraction(num, den)
    return Fraction(24000, 1001)


def apply_crop_resize_to_clip(clip: Any, plan: CropResizePlan, *, source: Optional[Path] = None) -> Any:
    if not plan.enabled:
        return clip

    import vapoursynth as vs  # type: ignore

    total_frames = int(clip.num_frames)
    fps = _fps_from_clip(clip)
    rule_ranges = resolve_rule_ranges(plan, fps=fps, total_frames=total_frames, source=source)

    variants: List[Any] = [
        apply_ops_to_clip(clip, plan.default_ops, target_width=plan.target_width, target_height=plan.target_height)
    ]
    frame_actions: List[Any] = [0] * total_frames
    variant_by_ops: Dict[Tuple[CropResizeOp, ...], int] = {plan.default_ops: 0}
    transition_entries: List[Tuple[CropResizeRule, FrameRange]] = []

    for rule, ranges in zip(plan.rules, rule_ranges):
        if rule.is_transition:
            for frame_range in ranges:
                transition_index = len(transition_entries)
                transition_entries.append((rule, frame_range))
                for frame_no in range(frame_range.start, frame_range.end):
                    frame_actions[frame_no] = (transition_index,)
            continue
        if rule.ops not in variant_by_ops:
            variant_by_ops[rule.ops] = len(variants)
            variants.append(apply_ops_to_clip(clip, rule.ops, target_width=plan.target_width, target_height=plan.target_height))
        variant_index = variant_by_ops[rule.ops]
        for frame_range in ranges:
            for frame_no in range(frame_range.start, frame_range.end):
                frame_actions[frame_no] = variant_index

    if len(variants) == 1 and not transition_entries:
        return variants[0]

    interpolated_variants: Dict[Tuple[CropResizeOp, ...], Any] = {}

    def select_frame(n: int) -> Any:
        action = frame_actions[n]
        if isinstance(action, int):
            return variants[action]
        transition_index = int(action[0])
        rule, frame_range = transition_entries[transition_index]
        end_ops = rule.end_ops or rule.ops
        ops = _interpolate_ops(rule.ops, end_ops, t=_transition_progress(n, frame_range))
        cached = interpolated_variants.get(ops)
        if cached is None:
            cached = apply_ops_to_clip(clip, ops, target_width=plan.target_width, target_height=plan.target_height)
            interpolated_variants[ops] = cached
        return cached

    return vs.core.std.FrameEval(variants[0], select_frame)


__all__ = [
    "CropResizeOp",
    "FrameRange",
    "CropResizeRule",
    "CropResizePlan",
    "EMPTY_PLAN",
    "parse_crop_resize_lines",
    "load_crop_resize_lines",
    "load_crop_resize_plan",
    "validate_crop_resize_plan",
    "resolve_rule_ranges",
    "apply_crop_resize_to_clip",
    "apply_ops_to_clip",
    "normalize_canvas",
]
