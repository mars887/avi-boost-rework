from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from utils.plan_model import SourceTrack, build_summary_rows, probe_source_tracks


def collect_tracks_by_file(files: Sequence[Path]) -> Dict[int, List[SourceTrack]]:
    tracks_by_file: Dict[int, List[SourceTrack]] = {}
    for index, path in enumerate(files, start=1):
        tracks_by_file[index] = list(probe_source_tracks(Path(path)))
    return tracks_by_file


def summarize_files(files: Sequence[Path]) -> List[Dict[str, object]]:
    return build_summary_rows(list(files), collect_tracks_by_file(files))


def summary_lines(files: Sequence[Path]) -> List[str]:
    return [str(row.get("line") or "") for row in summarize_files(files)]

