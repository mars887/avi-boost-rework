"""NVOF/noise/luma metrics integration helpers (schedule + CSV parsing)."""

from __future__ import annotations

import csv
import json
import math
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ab_cmd import run_cmd


@dataclass(frozen=True)
class MetricSpec:
    metric_type: str  # nvof | luma | noise
    base: str         # base CSV column prefix
    scope: str        # scene | global
    requires_global: bool = False


_ALIASES: Dict[str, str] = {
    "luma_avg": "scene_luma_avg",
    "luma_p25": "scene_luma_p25",
    "luma_p75": "scene_luma_p75",
    "luma_mean": "scene_luma_avg",
    "nvof_mean": "scene_nvof_avg",
    "noise_sigma": "scene_noise_sigma",
    "grain_ratio": "scene_grain_ratio",
}

_METRIC_SPECS: Dict[str, MetricSpec] = {
    "scene_nvof_avg": MetricSpec("nvof", "nvof_mean", "scene"),
    "global_nvof_avg": MetricSpec("nvof", "nvof_mean", "global"),
    "scene_luma_avg": MetricSpec("luma", "luma_mean", "scene"),
    "global_luma_avg": MetricSpec("luma", "luma_mean", "global"),
    "scene_luma_p25": MetricSpec("luma", "luma_p25", "scene"),
    "global_luma_p25": MetricSpec("luma", "luma_p25", "global"),
    "scene_luma_p75": MetricSpec("luma", "luma_p75", "scene"),
    "global_luma_p75": MetricSpec("luma", "luma_p75", "global"),
    "scene_noise_sigma": MetricSpec("noise", "noise_sigma", "scene"),
    "global_noise_sigma": MetricSpec("noise", "noise_sigma", "global"),
    "scene_grain_ratio": MetricSpec("noise", "grain_ratio", "scene"),
    "global_grain_ratio": MetricSpec("noise", "grain_ratio", "global"),
    "luma_ratio": MetricSpec("luma", "luma_mean", "scene", requires_global=True),
}

_COLUMN_BASES = {
    "nvof_mean",
    "luma_mean",
    "luma_p25",
    "luma_p75",
    "noise_sigma",
    "grain_ratio",
}


def normalize_metric_name(name: str) -> str:
    key = str(name).strip().lower()
    return _ALIASES.get(key, key)


def metric_spec(name: str) -> Optional[MetricSpec]:
    return _METRIC_SPECS.get(normalize_metric_name(name))


def is_nvof_metric(name: str) -> bool:
    return metric_spec(name) is not None


def metric_type(name: str) -> Optional[str]:
    spec = metric_spec(name)
    return spec.metric_type if spec else None


def metric_base(name: str) -> Optional[str]:
    spec = metric_spec(name)
    return spec.base if spec else None


def requires_global(name: str) -> bool:
    spec = metric_spec(name)
    if not spec:
        return False
    return spec.scope == "global" or bool(spec.requires_global)


def _merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted((int(a), int(b)) for a, b in ranges)
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def scene_indices_to_ranges(indices: Iterable[int], scene_ranges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for idx in sorted(set(int(i) for i in indices)):
        if idx < 0 or idx >= len(scene_ranges):
            continue
        st, en = scene_ranges[idx]
        st = int(st)
        en = int(en)
        if en <= st:
            continue
        ranges.append((st, en - 1))
    return _merge_ranges(ranges)


def build_nvof_schedule(
    *,
    scene_requirements: Dict[str, Iterable[int]],
    global_requirements: Iterable[str],
    scene_ranges: Sequence[Tuple[int, int]],
) -> Optional[dict]:
    types_needed: set[str] = set()
    global_types: set[str] = set()
    type_scene_indices: Dict[str, set[int]] = {"nvof": set(), "luma": set(), "noise": set()}

    for name in global_requirements:
        spec = metric_spec(name)
        if not spec:
            continue
        types_needed.add(spec.metric_type)
        global_types.add(spec.metric_type)

    for name, indices in scene_requirements.items():
        spec = metric_spec(name)
        if not spec:
            continue
        types_needed.add(spec.metric_type)
        if requires_global(name):
            global_types.add(spec.metric_type)
            continue
        type_scene_indices[spec.metric_type].update(int(i) for i in indices)

    if not types_needed:
        return None

    schedule: Dict[str, object] = {}

    if "nvof" in types_needed:
        ranges = None if "nvof" in global_types else scene_indices_to_ranges(type_scene_indices["nvof"], scene_ranges)
        if ranges is None or ranges:
            schedule["nvof"] = [{
                "id": "nvof",
                "ranges": ranges,
            }]

    if "luma" in types_needed:
        ranges = None if "luma" in global_types else scene_indices_to_ranges(type_scene_indices["luma"], scene_ranges)
        if ranges is None or ranges:
            schedule["luma"] = {
                "id": "luma",
                "ranges": ranges,
            }

    if "noise" in types_needed:
        ranges = None if "noise" in global_types else scene_indices_to_ranges(type_scene_indices["noise"], scene_ranges)
        if ranges is None or ranges:
            schedule["noise"] = {
                "id": "noise",
                "ranges": ranges,
            }

    return schedule or None


def run_nvof_script(
    *,
    input_file: Path,
    script_path: Path,
    schedule: dict,
    schedule_path: Path,
    out_csv: Path,
    extra_args: str,
    workdir: Optional[Path],
) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"NVOF script not found: {script_path}")
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    schedule_json = json.dumps(schedule, ensure_ascii=False, indent=2, sort_keys=True)
    if out_csv.exists() and schedule_path.exists():
        try:
            if schedule_path.read_text(encoding="utf-8") == schedule_json:
                return
        except Exception:
            pass
    schedule_path.write_text(schedule_json, encoding="utf-8")

    cmd = [
        sys.executable,
        str(script_path),
        "-i", str(input_file),
        "--schedule", str(schedule_path),
        "--out-csv", str(out_csv),
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    run_cmd(cmd, cwd=workdir, check=True, inherit_output=True)


class NvofMetricsCache:
    """Parsed CSV metrics with per-scene/global helpers."""

    def __init__(self, columns: Dict[str, Dict[int, float]]) -> None:
        self._columns = columns
        self._global_cache: Dict[str, float] = {}

    def has(self, base: str) -> bool:
        return base in self._columns

    def scene_avg(self, base: str, st: int, en: int) -> float:
        if base not in self._columns:
            raise KeyError(f"NVOF metric column not found: {base}")
        values = [
            v for f, v in self._columns[base].items()
            if int(st) <= int(f) < int(en) and not math.isnan(v)
        ]
        if not values:
            raise ValueError(f"No NVOF samples for {base} in range {st}:{en}.")
        return sum(values) / len(values)

    def global_avg(self, base: str) -> float:
        if base in self._global_cache:
            return self._global_cache[base]
        if base not in self._columns:
            raise KeyError(f"NVOF metric column not found: {base}")
        values = [v for v in self._columns[base].values() if not math.isnan(v)]
        if not values:
            raise ValueError(f"No NVOF samples for global average: {base}")
        avg = sum(values) / len(values)
        self._global_cache[base] = avg
        return avg


def _select_csv_columns(header: List[str], needed_bases: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for base in needed_bases:
        if base not in _COLUMN_BASES:
            continue
        matches = [h for h in header if h.startswith(base)]
        if not matches:
            raise ValueError(f"CSV column missing for base '{base}'. Header: {header}")
        if len(matches) > 1:
            raise ValueError(f"Multiple CSV columns match base '{base}': {matches}")
        mapping[base] = matches[0]
    return mapping


def load_nvof_csv(path: Path, *, needed_bases: Iterable[str]) -> NvofMetricsCache:
    if not path.exists():
        raise FileNotFoundError(f"NVOF CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Empty CSV: {path}")

        if header[0].lstrip("#").strip() != "frame":
            raise ValueError(f"Unexpected CSV header: {header}")

        column_map = _select_csv_columns(header, needed_bases)
        idx_map = {name: header.index(col) for name, col in column_map.items()}

        columns: Dict[str, Dict[int, float]] = {name: {} for name in column_map}
        for row in reader:
            if not row:
                continue
            try:
                frame_idx = int(float(row[0]))
            except Exception:
                continue

            for base, col_idx in idx_map.items():
                if col_idx >= len(row):
                    continue
                raw = row[col_idx].strip()
                if not raw:
                    continue
                try:
                    val = float(raw)
                except Exception:
                    continue
                if math.isnan(val):
                    continue
                columns[base][frame_idx] = val

    return NvofMetricsCache(columns)


def needed_bases_for_metrics(names: Iterable[str]) -> List[str]:
    bases: List[str] = []
    for name in names:
        base = metric_base(name)
        if base and base not in bases:
            bases.append(base)
    return bases
