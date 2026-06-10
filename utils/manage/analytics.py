from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

from utils.manage import av1an_state
from utils.manage.av1an_state import ChunkState, PassName
from utils.manage.context import WorkdirContext
from utils.manage.store import COLLECTION_PASS_CACHE, WorkdirStore


@dataclass
class PassChunkRow:
    index: int
    queue_position: int
    done: bool
    start_frame: int
    end_frame: int
    frames: int
    duration_seconds: float
    output_path: str
    output_exists: bool
    output_size: int
    bitrate_kbps: Optional[float]
    crf: Optional[float]
    passes: int
    encoder: str
    video_params: str
    ssimu2_avg: Optional[float] = None
    ssimu2_p5: Optional[float] = None
    chapter: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class SortSpec:
    field: str = "index"
    descending: bool = False


@dataclass
class AnalyticsResult:
    pass_name: str
    rows: List[PassChunkRow] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    csv_path: Optional[Path] = None


def parse_crf_from_video_params(tokens: Sequence[str]) -> Optional[float]:
    crf_index = -1
    for position, token in enumerate(tokens):
        if str(token) == "--crf":
            crf_index = position
    if crf_index < 0 or crf_index + 1 >= len(tokens):
        return None
    try:
        return float(str(tokens[crf_index + 1]).strip())
    except (TypeError, ValueError):
        return None


def parse_ssimu2_log_file(path: Path) -> Tuple[int, List[float]]:
    """Returns (skip, scores); scores[k] is the score of frame ``k*skip``."""
    scores: List[float] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
        match = re.search(r"skip:\s*([0-9]+)", first)
        if not match:
            raise RuntimeError(f"skip value not detected in SSIMU2 log: {path}")
        skip = int(match.group(1))
        for line in fh:
            value_match = re.search(r"([0-9]+):\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", line.strip())
            if not value_match:
                continue
            try:
                scores.append(max(float(value_match.group(2)), 0.0))
            except ValueError:
                continue
    if not scores:
        raise RuntimeError(f"no SSIMU2 values parsed: {path}")
    return skip, scores


def slice_metric_samples(scores: List[float], start: int, end: int, skip: int) -> List[float]:
    if not scores:
        return []
    if skip <= 0:
        skip = 1
    if end <= start:
        clamp = max(0, min((start + skip - 1) // skip, len(scores) - 1))
        return [scores[clamp]]
    first = max(0, min((start + skip - 1) // skip, len(scores) - 1))
    last = max(0, min((end - 1) // skip, len(scores) - 1))
    if last < first:
        return [scores[first]]
    return scores[first : last + 1] or [scores[first]]


def calc_avg_p5(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        raise RuntimeError("cannot compute stats for empty metric list")
    filtered = [max(float(value), 0.0) for value in values]
    ordered = sorted(filtered)
    avg = sum(filtered) / len(filtered)
    p5 = ordered[max(0, len(filtered) // 20)]
    return float(avg), float(p5)


def _parse_chapter_time(raw: str) -> Optional[float]:
    text = str(raw).strip().replace(",", ".")
    match = re.match(r"^(\d+):(\d+):(\d+(?:\.\d+)?)$", text)
    if not match:
        return None
    return int(match.group(1)) * 3600.0 + int(match.group(2)) * 60.0 + float(match.group(3))


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[1] if "}" in tag else tag


def load_chapters(ctx: WorkdirContext) -> List[Dict[str, Any]]:
    path = ctx.workdir / "chapters" / "chapters.xml"
    if not path.exists():
        return []
    try:
        root = ET.parse(str(path)).getroot()
    except ET.ParseError:
        return []
    chapters: List[Dict[str, Any]] = []
    for atom in root.iter():
        if _local_name(atom.tag) != "ChapterAtom":
            continue
        title = ""
        start: Optional[float] = None
        for child in atom.iter():
            name = _local_name(child.tag)
            if name == "ChapterTimeStart" and child.text:
                start = _parse_chapter_time(child.text)
            elif name == "ChapterString" and child.text and not title:
                title = child.text.strip()
        if start is not None:
            chapters.append({"start_seconds": start, "title": title})
    chapters.sort(key=lambda chapter: chapter["start_seconds"])
    return chapters


def _chapter_for_time(chapters: List[Dict[str, Any]], seconds: float) -> str:
    current = ""
    for chapter in chapters:
        if chapter["start_seconds"] <= seconds:
            current = chapter["title"] or current
        else:
            break
    return current


def collect_pass_rows(ctx: WorkdirContext, pass_name: PassName) -> AnalyticsResult:
    result = AnalyticsResult(pass_name=pass_name)
    pdir = av1an_state.pass_dir(ctx, pass_name)
    try:
        chunks = av1an_state.load_chunks(pdir)
    except av1an_state.Av1anStateError as exc:
        result.warnings.append(str(exc))
        return result
    done = av1an_state.load_done(pdir)
    chapters = load_chapters(ctx)

    ssimu2_skip = 0
    ssimu2_scores: List[float] = []
    if pass_name == "fastpass" and ctx.source is not None:
        log_path = pdir / f"{ctx.source.stem}_ssimu2.log"
        if log_path.exists():
            try:
                ssimu2_skip, ssimu2_scores = parse_ssimu2_log_file(log_path)
            except RuntimeError as exc:
                result.warnings.append(str(exc))

    for queue_position, chunk in enumerate(chunks):
        output = av1an_state.chunk_output_path(chunk, pdir)
        output_exists = output.is_file()
        size = 0
        if output_exists:
            try:
                size = output.stat().st_size
            except OSError:
                size = 0
        duration = chunk.duration_seconds
        bitrate = (size * 8 / 1000.0 / duration) if (size > 0 and duration > 0) else None
        row = PassChunkRow(
            index=chunk.index,
            queue_position=queue_position,
            done=done.is_done(chunk.index),
            start_frame=chunk.start_frame,
            end_frame=chunk.end_frame,
            frames=chunk.frames,
            duration_seconds=duration,
            output_path=str(output),
            output_exists=output_exists,
            output_size=size,
            bitrate_kbps=round(bitrate, 1) if bitrate is not None else None,
            crf=parse_crf_from_video_params(chunk.video_params),
            passes=int(chunk.data.get("passes") or 1),
            encoder=str(chunk.data.get("encoder") or ""),
            video_params=" ".join(chunk.video_params),
        )
        if ssimu2_scores:
            samples = slice_metric_samples(ssimu2_scores, chunk.start_frame, chunk.end_frame, ssimu2_skip)
            if samples:
                avg, p5 = calc_avg_p5(samples)
                row.ssimu2_avg = round(avg, 3)
                row.ssimu2_p5 = round(p5, 3)
        if chapters and chunk.frame_rate > 0:
            row.chapter = _chapter_for_time(chapters, chunk.start_frame / chunk.frame_rate)
        result.rows.append(row)

    warnings = av1an_state.validate_done(done, chunks)
    result.warnings.extend(warnings)
    return result


def sort_pass_rows(rows: List[PassChunkRow], sort: SortSpec) -> List[PassChunkRow]:
    field_name = str(sort.field or "index")
    if rows and not hasattr(rows[0], field_name):
        raise RuntimeError(f"unknown sort field: {field_name}")

    def key(row: PassChunkRow):
        value = getattr(row, field_name)
        if value is None:
            return (1, 0.0)
        if isinstance(value, bool):
            return (0, 1.0 if value else 0.0)
        if isinstance(value, (int, float)):
            return (0, float(value)) if math.isfinite(float(value)) else (1, 0.0)
        return (0, str(value).lower())

    return sorted(rows, key=key, reverse=bool(sort.descending))


def fastpass_stats_for_mainpass(
    ctx: WorkdirContext,
    mainpass_chunks: Sequence[ChunkState],
) -> Dict[int, Dict[str, float]]:
    """Map mainpass chunk index -> fastpass bitrate/SSIMU2 at matching ranges."""
    fast = collect_pass_rows(ctx, "fastpass")
    by_range = {(row.start_frame, row.end_frame): row for row in fast.rows}
    out: Dict[int, Dict[str, float]] = {}
    for chunk in mainpass_chunks:
        row = by_range.get((chunk.start_frame, chunk.end_frame))
        if row is None:
            continue
        values: Dict[str, float] = {}
        if row.bitrate_kbps is not None:
            values["bitrate_kbps"] = row.bitrate_kbps
        if row.ssimu2_avg is not None:
            values["ssimu2_avg"] = row.ssimu2_avg
        if row.ssimu2_p5 is not None:
            values["ssimu2_p5"] = row.ssimu2_p5
        if values:
            out[chunk.index] = values
    return out


def open_chunk_external(row: PassChunkRow) -> None:
    path = Path(row.output_path)
    if not path.exists():
        raise RuntimeError(f"chunk output does not exist: {path}")
    open_path_external(path)


def open_path_external(path: Path) -> None:
    path = Path(path)
    if os.name == "nt":
        os.startfile(str(path))  # noqa: S606 - intended shell open
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, str(path)])


def cache_pass_rows(store: WorkdirStore, pass_name: PassName, result: AnalyticsResult) -> None:
    store.put_doc(
        COLLECTION_PASS_CACHE,
        pass_name,
        {
            "generated_at": time.time(),
            "warnings": list(result.warnings),
            "rows": [row.as_dict() for row in result.rows],
        },
    )


def cached_pass_rows(store: WorkdirStore, pass_name: PassName) -> Optional[AnalyticsResult]:
    payload = store.get_doc(COLLECTION_PASS_CACHE, pass_name)
    if not isinstance(payload, dict):
        return None
    rows = []
    for raw in payload.get("rows") or []:
        if isinstance(raw, dict):
            try:
                rows.append(PassChunkRow(**raw))
            except TypeError:
                return None
    return AnalyticsResult(pass_name=pass_name, rows=rows, warnings=list(payload.get("warnings") or []))


def refresh_pass_analytics_files(ctx: WorkdirContext, pass_name: PassName) -> AnalyticsResult:
    """Recollect rows and dump a CSV summary next to legacy analytics output."""
    result = collect_pass_rows(ctx, pass_name)
    out_dir = ctx.workdir / f"{pass_name}-analytics"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "manage_chunks.csv"
    headers = [
        "index",
        "queue_position",
        "done",
        "start_frame",
        "end_frame",
        "frames",
        "duration_seconds",
        "output_size",
        "bitrate_kbps",
        "crf",
        "ssimu2_avg",
        "ssimu2_p5",
        "chapter",
        "output_path",
    ]
    lines = [";".join(headers)]
    for row in result.rows:
        payload = row.as_dict()
        lines.append(";".join("" if payload.get(name) is None else str(payload.get(name)) for name in headers))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    result.csv_path = csv_path
    return result


__all__ = [
    "AnalyticsResult",
    "PassChunkRow",
    "SortSpec",
    "cache_pass_rows",
    "cached_pass_rows",
    "calc_avg_p5",
    "collect_pass_rows",
    "fastpass_stats_for_mainpass",
    "load_chapters",
    "open_chunk_external",
    "open_path_external",
    "parse_crf_from_video_params",
    "parse_ssimu2_log_file",
    "refresh_pass_analytics_files",
    "slice_metric_samples",
    "sort_pass_rows",
]
