"""Resume marker and artifact validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ab_fs import load_json
from ab_scenes_io import sanitize_scenes_json
from ab_ssimu2 import parse_ssimu2_log

def is_valid_base_scenes(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 10:
        return False
    try:
        sanitize_scenes_json(load_json(path))
        return True
    except Exception:
        return False

def is_valid_ssimu2_log(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 10:
        return False
    try:
        _skip, scores = parse_ssimu2_log(path)
        return len(scores) > 0
    except Exception:
        return False

def is_valid_final_scenes(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 10:
        return False
    try:
        obj = load_json(path)
        scenes = obj.get("scenes") or obj.get("split_scenes") or []
        if not scenes:
            return False
        # ensure at least one zone_overrides is populated
        for s in scenes:
            if isinstance(s, dict) and s.get("zone_overrides"):
                return True
        return False
    except Exception:
        return False
