from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


STAGE_ITEM = "Item"
STAGE_DEMUX = "Demux"
STAGE_ATTACHMENTS = "Attachments cleanup"
STAGE_AUTOBOOST_SCENE = "Auto-Boost: Scene Detection"
STAGE_AUTOBOOST_PSD_SCENE = "Auto-Boost: PSD Scene Detection"
STAGE_FASTPASS = "Fastpass"
STAGE_SSIMU2 = "SSIMU2 Metrics"
STAGE_HDR_PATCH = "HDR Patch"
STAGE_ZONE_EDIT = "Zone Edit"
STAGE_MAINPASS = "Mainpass"
STAGE_AUDIO = "Audio Tool"
STAGE_VERIFY = "Verify"
STAGE_MUX = "Mux"
CACHED_STAGE_MESSAGE = "cached"
RUNNER_MANAGED_STATE_ENV = "PBBATCH_RUNNER_MANAGED_STATE"
MIN_VALID_MEDIA_BYTES = 1024


@dataclass(frozen=True)
class StageResumeInfo:
    marker: Optional[Path] = None
    marker_exists: bool = False
    marker_valid: bool = True
    completed: bool = False
    reason: str = ""


def autoboost_scene_stage(item: Any) -> str:
    scene_detection = str(item.resolved.plan.video.primary.scene_detection or "").strip().lower()
    return STAGE_AUTOBOOST_PSD_SCENE if scene_detection == "psd" else STAGE_AUTOBOOST_SCENE


def display_stage_plan(item: Any) -> List[str]:
    stages = [STAGE_DEMUX, STAGE_ATTACHMENTS]
    if item.resolved.has_video_edit():
        stages.append(autoboost_scene_stage(item))
        if not item.resolved.plan.video.primary.no_fastpass:
            stages.extend([STAGE_FASTPASS, STAGE_SSIMU2])
        if item.mode == "full":
            stages.extend([STAGE_ZONE_EDIT, STAGE_HDR_PATCH, STAGE_MAINPASS])
    if item.mode == "full":
        stages.extend([STAGE_AUDIO, STAGE_VERIFY, STAGE_MUX])
    return stages


def is_cached_stage_message(message: str) -> bool:
    normalized = str(message or "").strip().lower()
    return normalized in (CACHED_STAGE_MESSAGE, "resume", "using_existing_base_scenes")


def file_has_bytes(path: Path, min_bytes: int = 1) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size >= min_bytes
    except Exception:
        return False


def load_json_object(path: Path) -> Optional[Dict[str, Any]]:
    if not file_has_bytes(path, 2):
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def valid_json_file(path: Path) -> bool:
    return load_json_object(path) is not None


def valid_scenes_json(path: Path, *, require_zone_overrides: bool = False) -> bool:
    obj = load_json_object(path)
    if obj is None:
        return False
    scenes = obj.get("scenes") or obj.get("split_scenes") or []
    if not isinstance(scenes, list) or not scenes:
        return False
    if require_zone_overrides:
        return any(isinstance(scene, dict) and bool(scene.get("zone_overrides")) for scene in scenes)
    return True


def valid_ssimu2_log(path: Path) -> bool:
    if not file_has_bytes(path, 10):
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    return bool(re.search(r"\d+(?:\.\d+)?", text))


def valid_audio_manifest(item: Any) -> bool:
    obj = load_json_object(item.workdir / "00_meta" / "audio_manifest.json")
    if obj is None:
        return False
    outputs = obj.get("outputs")
    if not isinstance(outputs, list):
        return False
    for output in outputs:
        if not isinstance(output, dict):
            return False
        out_path = str(output.get("outPath") or "").strip()
        if not out_path:
            continue
        path = Path(out_path)
        if not path.is_absolute():
            path = item.workdir / path
        if not file_has_bytes(path, MIN_VALID_MEDIA_BYTES):
            return False
    return True


def autoboost_project_dir(item: Any) -> Path:
    return item.workdir / "video"


def autoboost_state_dir(item: Any) -> Path:
    return autoboost_project_dir(item) / ".state"


def autoboost_base_scenes(item: Any) -> Path:
    return autoboost_project_dir(item) / "psd" / "scenes.psd.json"


def autoboost_av1an_scenes(item: Any) -> Path:
    return autoboost_project_dir(item) / "fastpass" / "scenes.json"


def autoboost_fastpass_output(item: Any) -> Path:
    return autoboost_project_dir(item) / "fastpass" / f"{item.source.stem}.fastpass.mkv"


def autoboost_ssimu2_log(item: Any) -> Path:
    return autoboost_project_dir(item) / "fastpass" / f"{item.source.stem}_ssimu2.log"


def autoboost_stage4_scenes(item: Any) -> Path:
    name = "scenes-preview.json" if item.mode == "fastpass" else "scenes.json"
    return autoboost_project_dir(item) / name


def zone_edit_scenes(item: Any) -> Path:
    return autoboost_project_dir(item) / "scenes-zoned.json"


def final_scenes(item: Any) -> Path:
    return autoboost_project_dir(item) / "scenes-final.json"


def file_not_older_than(path: Path, dependency: Path) -> bool:
    try:
        return path.stat().st_mtime >= dependency.stat().st_mtime
    except Exception:
        return False


def stage_marker_path(item: Any, stage: str) -> Optional[Path]:
    workdir = item.workdir
    state_dir = workdir / ".state"
    if stage == STAGE_DEMUX:
        return state_dir / "DEMUX_DONE"
    if stage == STAGE_ATTACHMENTS:
        return state_dir / "ATTACHMENTS_CLEAN_DONE"
    if stage == STAGE_AUTOBOOST_PSD_SCENE:
        return autoboost_state_dir(item) / "PSD_FINISHED"
    if stage == STAGE_AUTOBOOST_SCENE:
        return autoboost_state_dir(item) / "FASTPASS_COMPLETED"
    if stage == STAGE_FASTPASS:
        return autoboost_state_dir(item) / "FASTPASS_COMPLETED"
    if stage == STAGE_SSIMU2:
        return autoboost_state_dir(item) / "SSIMU2_COMPLETED"
    if stage == STAGE_HDR_PATCH:
        return state_dir / "HDR_PATCH_DONE"
    if stage == STAGE_ZONE_EDIT:
        return state_dir / "ZONE_EDIT_DONE"
    if stage == STAGE_AUDIO:
        return state_dir / "AUDIO_DONE"
    if stage == STAGE_VERIFY:
        return state_dir / "VERIFY_DONE"
    if stage == STAGE_MUX:
        return state_dir / "MUX_DONE"
    return None


def stage_marker_artifacts_valid(item: Any, stage: str) -> bool:
    if stage == STAGE_DEMUX:
        return valid_json_file(item.workdir / "00_meta" / "demux_manifest.json")
    if stage == STAGE_ATTACHMENTS:
        return valid_json_file(item.workdir / "attachments" / "attachments_cleaner_report.json")
    if stage == STAGE_AUTOBOOST_PSD_SCENE:
        return valid_scenes_json(autoboost_base_scenes(item))
    if stage == STAGE_AUTOBOOST_SCENE:
        return valid_scenes_json(autoboost_base_scenes(item)) or valid_scenes_json(autoboost_av1an_scenes(item))
    if stage == STAGE_FASTPASS:
        return file_has_bytes(autoboost_fastpass_output(item), MIN_VALID_MEDIA_BYTES)
    if stage == STAGE_SSIMU2:
        return valid_ssimu2_log(autoboost_ssimu2_log(item))
    if stage == STAGE_HDR_PATCH:
        zoned = zone_edit_scenes(item)
        final = final_scenes(item)
        return (
            valid_scenes_json(zoned, require_zone_overrides=True)
            and valid_scenes_json(final, require_zone_overrides=True)
            and file_not_older_than(final, zoned)
        )
    if stage == STAGE_ZONE_EDIT:
        return valid_scenes_json(zone_edit_scenes(item), require_zone_overrides=True)
    if stage == STAGE_AUDIO:
        return valid_audio_manifest(item)
    if stage == STAGE_VERIFY:
        if (item.workdir / "00_logs" / "verify_error.txt").exists():
            return False
        if not valid_json_file(item.workdir / "00_meta" / "demux_manifest.json"):
            return False
        if not valid_audio_manifest(item):
            return False
        if item.resolved.has_video_edit() and not file_has_bytes(
            item.workdir / "video" / "video-final.mkv",
            MIN_VALID_MEDIA_BYTES,
        ):
            return False
        return True
    if stage == STAGE_MUX:
        return file_has_bytes(item.source.parent / f"{item.source.stem}-av1.mkv", MIN_VALID_MEDIA_BYTES)
    return True


def stage_completion_artifacts_valid(item: Any, stage: str) -> bool:
    primary = item.resolved.plan.video.primary
    if stage in (STAGE_AUTOBOOST_SCENE, STAGE_AUTOBOOST_PSD_SCENE) and primary.no_fastpass:
        return stage_marker_artifacts_valid(item, stage) and valid_scenes_json(
            autoboost_stage4_scenes(item),
            require_zone_overrides=True,
        )
    if stage == STAGE_SSIMU2:
        return stage_marker_artifacts_valid(item, stage) and valid_scenes_json(
            autoboost_stage4_scenes(item),
            require_zone_overrides=True,
        )
    if stage == STAGE_MAINPASS:
        return file_has_bytes(item.workdir / "video" / "video-final.mkv", MIN_VALID_MEDIA_BYTES)
    return stage_marker_artifacts_valid(item, stage)


def stage_resume_info(item: Any, stage: str) -> StageResumeInfo:
    marker = stage_marker_path(item, stage)
    if marker is None:
        completed = stage_completion_artifacts_valid(item, stage)
        return StageResumeInfo(marker=None, completed=completed, reason="" if completed else "artifact_missing")
    marker_exists = marker.exists()
    marker_valid = True
    completed = False
    reason = "marker_missing"
    if marker_exists:
        marker_valid = stage_marker_artifacts_valid(item, stage)
        completed = marker_valid and stage_completion_artifacts_valid(item, stage)
        if completed:
            reason = ""
        elif marker_valid:
            reason = "dependent_artifact_missing"
        else:
            reason = "stale_marker"
    return StageResumeInfo(
        marker=marker,
        marker_exists=marker_exists,
        marker_valid=marker_valid,
        completed=completed,
        reason=reason,
    )


def stage_resume_marker_exists(item: Any, stage: str) -> bool:
    return stage_resume_info(item, stage).completed


def write_stage_marker(item: Any, stage: str) -> None:
    marker = stage_marker_path(item, stage)
    if marker is None:
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n", encoding="utf-8")


def clear_stage_marker(item: Any, stage: str) -> None:
    marker = stage_marker_path(item, stage)
    if marker is None or not marker.exists():
        return
    try:
        marker.unlink()
    except Exception:
        pass
