from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from utils.plan.types import SourceTrack, normalize_track_type


def probe_source_tracks(source: Path) -> List[SourceTrack]:
    mkvmerge = shutil.which("mkvmerge") or "mkvmerge"
    if shutil.which(mkvmerge) is None and not Path(mkvmerge).exists():
        raise RuntimeError("mkvmerge not found in PATH")

    proc = subprocess.run(
        [mkvmerge, "-J", str(source)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"mkvmerge -J failed for {source}:\n{proc.stderr or proc.stdout}")

    data = json.loads(proc.stdout)
    out: List[SourceTrack] = []
    for item in data.get("tracks") or []:
        raw_type = normalize_track_type(item.get("type"))
        if raw_type not in ("video", "audio", "sub"):
            continue
        props = item.get("properties") or {}
        name = str(props.get("track_name") or props.get("name") or "").strip()
        lang = str(props.get("language") or props.get("language_ietf") or "und").strip() or "und"
        out.append(
            SourceTrack(
                track_id=int(item.get("id")),
                track_type=raw_type,
                name=name,
                lang=lang,
                default=bool(props.get("default_track")),
                forced=bool(props.get("forced_track")),
                codec=str(item.get("codec") or props.get("codec_id") or ""),
            )
        )
    return out


def build_summary_rows(files: Sequence[Path], tracks_by_file: Dict[int, List[SourceTrack]]) -> List[Dict[str, Any]]:
    def ident_for(track: SourceTrack) -> str:
        return track.lang if track.track_type == "video" else track.name

    class _TrackKey:
        __slots__ = ("track_type", "ident", "lang")

        def __init__(self, track_type: str, ident: str, lang: str) -> None:
            self.track_type = track_type
            self.ident = ident
            self.lang = lang

        def __hash__(self) -> int:
            return hash((self.track_type, self.ident, self.lang))

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, _TrackKey):
                return False
            return (self.track_type, self.ident, self.lang) == (other.track_type, other.ident, other.lang)

    def format_file_set(file_indexes: Iterable[int]) -> str:
        sorted_indexes = sorted(set(int(x) for x in file_indexes))
        if not sorted_indexes:
            return ""
        ranges: List[range] = []
        start = sorted_indexes[0]
        end = sorted_indexes[0]
        for idx in sorted_indexes[1:]:
            if idx == end + 1:
                end = idx
            else:
                ranges.append(range(start, end + 1))
                start = idx
                end = idx
        ranges.append(range(start, end + 1))
        parts: List[str] = []
        for current in ranges:
            first = current.start
            last = current.stop - 1
            parts.append(str(first) if first == last else f"{first}-{last}")
        return ", ".join(parts)

    files_count = len(files)
    raw_tracks: List[tuple[int, SourceTrack]] = []
    for file_index, tracks in tracks_by_file.items():
        for track in tracks:
            raw_tracks.append((file_index, track))
    if not raw_tracks:
        return []

    track_map: Dict[_TrackKey, Dict[int, int]] = {}
    for file_index, track in raw_tracks:
        key = _TrackKey(track.track_type, ident_for(track), track.lang)
        track_map.setdefault(key, {})[file_index] = track.track_id

    anchored: Dict[_TrackKey, int] = {}
    shifting: Dict[_TrackKey, set[int]] = {}
    for key, by_file in track_map.items():
        index_set = set(by_file.values())
        if len(by_file) == files_count and len(index_set) == 1:
            anchored[key] = next(iter(index_set))
        elif len(by_file) >= 2 and len(index_set) > 1:
            shifting[key] = set(index_set)

    by_track_id: Dict[int, List[tuple[int, SourceTrack]]] = {}
    for file_index, track in raw_tracks:
        by_track_id.setdefault(track.track_id, []).append((file_index, track))

    star_by_index: Dict[int, bool] = {}
    for track_id, members in by_track_id.items():
        file_indexes = {file_index for file_index, _ in members}
        track_types = {track.track_type for _, track in members}
        has_anchor = any(index == track_id for index in anchored.values())
        has_shifted_here = any(track_id in values for values in shifting.values())
        safe = has_anchor or (len(file_indexes) == files_count and len(track_types) == 1 and not has_shifted_here)
        star_by_index[track_id] = not safe

    rows: List[Dict[str, Any]] = []
    for track_id, members in sorted(by_track_id.items()):
        grouped: Dict[_TrackKey, set[int]] = {}
        for file_index, track in members:
            key = _TrackKey(track.track_type, ident_for(track), track.lang)
            grouped.setdefault(key, set()).add(file_index)
        for key, present in grouped.items():
            rows.append(
                {
                    "index": track_id,
                    "star": star_by_index.get(track_id, True),
                    "type": key.track_type,
                    "displayName": key.ident,
                    "lang": key.lang,
                    "presentIn": sorted(present),
                    "reason": "",
                }
            )

    rows_by_index: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        rows_by_index.setdefault(int(row["index"]), []).append(row)

    all_file_indexes = set(range(1, files_count + 1))
    majority_threshold = 0.6
    for idx_rows in rows_by_index.values():
        starred = bool(idx_rows[0]["star"])
        if starred:
            for row in idx_rows:
                present = set(int(x) for x in row["presentIn"])
                if len(present) < files_count:
                    row["reason"] = f"-> (only {format_file_set(present)})"
            continue
        if len(idx_rows) == 1:
            row = idx_rows[0]
            present = set(int(x) for x in row["presentIn"])
            if len(present) < files_count:
                row["reason"] = f"-> (only {format_file_set(present)})"
            continue

        major_rows = []
        for row in idx_rows:
            present = set(int(x) for x in row["presentIn"])
            fraction = len(present) / float(files_count)
            if len(present) >= 2 and fraction >= majority_threshold:
                major_rows.append(row)
        if not major_rows:
            continue
        for row in major_rows:
            present = set(int(x) for x in row["presentIn"])
            missing = all_file_indexes - present
            if missing:
                row["reason"] = f"-> (except {format_file_set(missing)})"
        for row in idx_rows:
            if row in major_rows:
                continue
            row["reason"] = f"-> (only {format_file_set(row['presentIn'])})"

    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["index"]),
            {"video": 0, "audio": 1, "sub": 2}.get(normalize_track_type(row["type"]), 9),
            str(row["displayName"]).lower(),
            str(row["lang"]).lower(),
        ),
    )
    for row in ordered:
        prefix = "*" if row["star"] else " "
        display_name = row["displayName"] if row["displayName"] else "-"
        lang = row["lang"] if row["lang"] else "-"
        reason = f" {row['reason']}" if row["reason"] else ""
        row["line"] = f"{prefix} {row['index']} | {row['type']} | {display_name} | {lang}{reason}"
    return ordered


__all__ = ["probe_source_tracks", "build_summary_rows"]
