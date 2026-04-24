from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

from utils.pipeline_runtime import ensure_dir, final_output_path_for_source
from utils.runner_state import (
    autoboost_fastpass_output,
    clear_stage_marker,
    display_stage_plan,
    stage_resume_marker_exists,
)

_SOURCE_INFO_CACHE: Dict[str, Dict[str, Any]] = {}
_STALE_INPUT_KEYS: set[str] = set()


def item_identity_key(item: Any) -> str:
    return f"{item.plan_path.resolve()}|{item.source.resolve()}|{item.workdir.resolve()}".lower()


def source_info_path(item: Any) -> Path:
    return item.workdir / "00_meta" / "source_info.json"


def output_path_for_item(item: Any) -> Path:
    return final_output_path_for_source(item.source)


def fastpass_output_path_for_item(item: Any) -> Path:
    return autoboost_fastpass_output(item)


def safe_file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size) if path.exists() and path.is_file() else 0
    except Exception:
        return 0


def file_sha256(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def file_sample_sha256(path: Path, sample_size: int = 1024 * 1024) -> str:
    try:
        size = path.stat().st_size
        h = hashlib.sha256()
        with path.open("rb") as fh:
            h.update(fh.read(sample_size))
            if size > sample_size:
                fh.seek(max(0, size - sample_size))
                h.update(fh.read(sample_size))
        return h.hexdigest()
    except Exception:
        return ""


def source_signature(source: Path) -> Dict[str, Any]:
    try:
        stat = source.stat()
    except Exception:
        return {
            "path": str(source),
            "exists": False,
            "suffix": source.suffix.lower(),
        }
    return {
        "path": str(source),
        "exists": True,
        "name": source.name,
        "suffix": source.suffix.lower(),
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
        "sample_sha256": file_sample_sha256(source),
    }


def _duration_from_ffprobe(payload: Dict[str, Any]) -> float:
    try:
        return max(0.0, float(dict(payload.get("format") or {}).get("duration") or 0.0))
    except Exception:
        return 0.0


def build_source_info(item: Any) -> Dict[str, Any]:
    signature = source_signature(item.source)
    # TODO: Replace this coarse disabled hash with per-stage plan signatures.
    # The runner should invalidate only the affected stage groups, e.g. audio
    # bitrate changes must not force video/autoboost stages to rerun.
    # plan_hash = file_sha256(item.plan_path)   temporarily disabled
    plan_hash = "0"
    ffprobe_payload: Dict[str, Any] = {}
    ffprobe_error = ""
    ffprobe = shutil.which("ffprobe")
    if ffprobe and signature.get("exists"):
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(item.source),
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
            if proc.returncode == 0:
                parsed = json.loads(str(proc.stdout or "{}"))
                ffprobe_payload = parsed if isinstance(parsed, dict) else {}
            else:
                ffprobe_error = str(proc.stderr or proc.stdout or "").strip()[:4000]
        except Exception as exc:
            ffprobe_error = str(exc)
    duration = _duration_from_ffprobe(ffprobe_payload)
    return {
        "schema": 1,
        "cache_state": "clean",
        "generated_at": time.time(),
        "source": str(item.source),
        "source_signature": signature,
        "plan": str(item.plan_path),
        "plan_sha256": plan_hash,
        "duration_seconds": duration,
        "ffprobe": ffprobe_payload,
        "ffprobe_error": ffprobe_error,
    }


def read_source_info(item: Any) -> Dict[str, Any]:
    path = source_info_path(item)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def source_info_current(item: Any, info: Dict[str, Any]) -> bool:
    if dict(info.get("source_signature") or {}) != source_signature(item.source):
        return False
    stored_plan_hash = str(info.get("plan_sha256") or "")
    return stored_plan_hash == "0" or stored_plan_hash == file_sha256(item.plan_path)


def item_has_resume_state(item: Any) -> bool:
    if output_path_for_item(item).exists():
        return True
    return any(stage_resume_marker_exists(item, stage) for stage in display_stage_plan(item))


def clear_item_stage_markers(item: Any) -> None:
    for stage in display_stage_plan(item):
        clear_stage_marker(item, stage)


def prepare_source_info(item: Any) -> None:
    existing = read_source_info(item)
    stale_reason = ""
    if existing:
        if str(existing.get("cache_state") or "").strip().lower() == "invalidated":
            stale_reason = str(existing.get("invalidate_reason") or "previous_invalidated_run_incomplete")
        elif not source_info_current(item, existing):
            stale_reason = "source_or_plan_changed"
    elif item_has_resume_state(item):
        stale_reason = "missing_source_info_for_existing_state"

    if stale_reason:
        _STALE_INPUT_KEYS.add(item_identity_key(item))
        clear_item_stage_markers(item)
        print(f"[runner] {item.name} | source info changed | invalidated cached stages ({stale_reason})", flush=True)

    info = build_source_info(item) if stale_reason or not existing else existing
    if stale_reason:
        info["cache_state"] = "invalidated"
        info["invalidate_reason"] = stale_reason
        info["invalidated_at"] = time.time()
    path = source_info_path(item)
    ensure_dir(path.parent)
    if stale_reason or not existing:
        path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    _SOURCE_INFO_CACHE[item_identity_key(item)] = info


def mark_source_info_clean(item: Any) -> None:
    info = item_source_info(item)
    if not info:
        return
    info = dict(info)
    info["cache_state"] = "clean"
    info.pop("invalidate_reason", None)
    info["completed_at"] = time.time()
    path = source_info_path(item)
    ensure_dir(path.parent)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    key = item_identity_key(item)
    _SOURCE_INFO_CACHE[key] = info
    _STALE_INPUT_KEYS.discard(key)


def item_source_info(item: Any) -> Dict[str, Any]:
    key = item_identity_key(item)
    info = _SOURCE_INFO_CACHE.get(key)
    if info is None:
        info = read_source_info(item)
        if info:
            _SOURCE_INFO_CACHE[key] = info
    return info or {"source_signature": source_signature(item.source), "duration_seconds": 0.0}


def item_inputs_changed(item: Any) -> bool:
    return item_identity_key(item) in _STALE_INPUT_KEYS


def item_source_size(item: Any) -> int:
    signature = dict(item_source_info(item).get("source_signature") or {})
    try:
        return int(signature.get("size") or 0)
    except Exception:
        return 0


def item_source_duration(item: Any) -> float:
    try:
        return max(0.0, float(item_source_info(item).get("duration_seconds") or 0.0))
    except Exception:
        return 0.0
