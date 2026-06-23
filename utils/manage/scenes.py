from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from utils import workdir_layout as layout
from utils.manage.av1an_state import ChunkState
from utils.manage.backup import ManageTransaction, write_json_atomic
from utils.manage.context import WorkdirContext

_V = layout.VIDEO_DIRNAME

# scene files known to the pipeline, relative to the workdir, in pipeline order.
# Names are sourced from utils.workdir_layout (the on-disk-layout SSOT) so they
# cannot drift from the paths the runner actually produces.
KNOWN_SCENE_FILES = (
    (f"{_V}/{layout.PSD_DIRNAME}/{layout.PSD_BASE_SCENES_NAME}", "PSD base scenes"),
    (f"{_V}/{layout.FASTPASS_DIRNAME}/{layout.AV1AN_SCENES_NAME}", "av1an fastpass scenes"),
    (f"{_V}/{layout.STAGE4_SCENES_FULL_NAME}", "auto-boost stage4 scenes"),
    (f"{_V}/{layout.STAGE4_SCENES_PREVIEW_NAME}", "auto-boost stage4 scenes (fastpass mode)"),
    (f"{_V}/{layout.BOUNDARY_SCENES_NAME}", "zone boundaries"),
    (f"{_V}/{layout.RECALC_SCENES_NAME}", "zone CRF recalc"),
    (f"{_V}/{layout.ZONED_SCENES_NAME}", "zone edit"),
    (f"{_V}/{layout.FINAL_SCENES_NAME}", "final scenes (HDR patched)"),
    (f"{_V}/{layout.MAINPASS_DIRNAME}/{layout.AV1AN_SCENES_NAME}", "av1an mainpass temp scenes"),
)


class SceneFileError(RuntimeError):
    pass


@dataclass(frozen=True)
class SceneFileRef:
    path: Path
    relative: str
    description: str
    exists: bool
    size_bytes: int


@dataclass
class ValidationIssue:
    level: str  # "error" | "warning"
    message: str

    @property
    def is_error(self) -> bool:
        return self.level == "error"


@dataclass
class SceneFile:
    """Wrapper over an av1an/auto-boost scenes JSON; unknown fields preserved."""

    data: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None

    @property
    def frames(self) -> int:
        try:
            return int(self.data.get("frames") or 0)
        except (TypeError, ValueError):
            return 0

    @property
    def scenes(self) -> List[Dict[str, Any]]:
        scenes = self.data.get("scenes")
        return scenes if isinstance(scenes, list) else []

    @property
    def split_scenes(self) -> List[Dict[str, Any]]:
        scenes = self.data.get("split_scenes")
        return scenes if isinstance(scenes, list) else []

    @property
    def effective_scenes(self) -> List[Dict[str, Any]]:
        """What av1an would actually use: split_scenes, falling back to scenes."""
        return self.split_scenes or self.scenes


def list_scene_files(ctx: WorkdirContext) -> List[SceneFileRef]:
    out: List[SceneFileRef] = []
    for relative, description in KNOWN_SCENE_FILES:
        path = ctx.workdir / Path(relative)
        exists = path.is_file()
        size = 0
        if exists:
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
        out.append(
            SceneFileRef(path=path, relative=relative, description=description, exists=exists, size_bytes=size)
        )
    return out


def load_scene_file(path: Path) -> SceneFile:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SceneFileError(f"scene file not found: {path}") from None
    except Exception as exc:
        raise SceneFileError(f"failed to parse {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SceneFileError(f"scene file must be a JSON object: {path}")
    return SceneFile(data=raw, path=path)


def save_scene_file(path: Path, scene_file: SceneFile, *, tx: Optional[ManageTransaction] = None) -> None:
    issues = validate_scene_file(scene_file)
    errors = [issue for issue in issues if issue.is_error]
    if errors:
        raise SceneFileError("; ".join(issue.message for issue in errors))
    if tx is not None:
        tx.write_json(path, scene_file.data)
    else:
        write_json_atomic(path, scene_file.data)


def _validate_scene_list(
    label: str,
    scenes: List[Dict[str, Any]],
    frames: int,
    issues: List[ValidationIssue],
) -> None:
    previous_end: Optional[int] = None
    sorted_ok = True
    for position, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            issues.append(ValidationIssue("error", f"{label}[{position}] is not an object"))
            continue
        try:
            start = int(scene.get("start_frame"))
            end = int(scene.get("end_frame"))
        except (TypeError, ValueError):
            issues.append(ValidationIssue("error", f"{label}[{position}] has invalid start/end frame"))
            continue
        if start < 0:
            issues.append(ValidationIssue("error", f"{label}[{position}] start_frame is negative"))
        if end <= start:
            issues.append(ValidationIssue("error", f"{label}[{position}] end_frame must be > start_frame"))
        if frames > 0 and end > frames:
            issues.append(ValidationIssue("error", f"{label}[{position}] end_frame {end} exceeds frames {frames}"))
        if previous_end is not None:
            if start < previous_end:
                issues.append(
                    ValidationIssue("warning", f"{label}[{position}] overlaps previous scene")
                )
            if start != previous_end:
                sorted_ok = sorted_ok and start > previous_end
        previous_end = end
        overrides = scene.get("zone_overrides")
        if overrides is not None and not isinstance(overrides, dict):
            issues.append(ValidationIssue("error", f"{label}[{position}] zone_overrides must be object or null"))
        if isinstance(overrides, dict):
            params = overrides.get("video_params")
            if params is not None and not isinstance(params, list):
                issues.append(
                    ValidationIssue("error", f"{label}[{position}] zone_overrides.video_params must be an array")
                )
    starts = [int(scene.get("start_frame") or 0) for scene in scenes if isinstance(scene, dict)]
    if starts != sorted(starts):
        issues.append(ValidationIssue("warning", f"{label} is not sorted by start_frame"))


def validate_scene_file(scene_file: SceneFile) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    frames = scene_file.frames
    if frames <= 0:
        issues.append(ValidationIssue("warning", "frames is not a positive integer"))
    scenes = scene_file.scenes
    split_scenes = scene_file.split_scenes
    if not scenes and not split_scenes:
        issues.append(ValidationIssue("error", "neither scenes nor split_scenes is present/non-empty"))
        return issues
    if scenes:
        _validate_scene_list("scenes", scenes, frames, issues)
    if split_scenes:
        _validate_scene_list("split_scenes", split_scenes, frames, issues)
    return issues


@dataclass(frozen=True)
class SceneSelector:
    """Select scenes by list position or by frame range overlap."""

    positions: Sequence[int] = ()
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    in_split_scenes: bool = True

    def matches(self, position: int, scene: Dict[str, Any]) -> bool:
        if self.positions and position not in self.positions:
            return False
        if self.start_frame is not None or self.end_frame is not None:
            start = int(scene.get("start_frame") or 0)
            end = int(scene.get("end_frame") or 0)
            sel_start = self.start_frame if self.start_frame is not None else start
            sel_end = self.end_frame if self.end_frame is not None else end
            if end <= sel_start or start >= sel_end:
                return False
        return True


@dataclass(frozen=True)
class ZonePatch:
    """Partial update of ``zone_overrides`` for selected scenes."""

    video_params: Optional[List[str]] = None
    min_scene_len: Optional[int] = None
    extra_splits_len: Optional[int] = None
    photon_noise: Optional[int] = None
    chroma_noise: Optional[bool] = None
    raw_overrides: Optional[Dict[str, Any]] = None  # advanced: replace whole object


@dataclass
class ScenePatchResult:
    patched_positions: List[int] = field(default_factory=list)
    issues: List[ValidationIssue] = field(default_factory=list)


def patch_scene_region(
    path: Path,
    selector: SceneSelector,
    patch: ZonePatch,
    *,
    tx: Optional[ManageTransaction] = None,
) -> ScenePatchResult:
    scene_file = load_scene_file(path)
    if selector.in_split_scenes:
        scenes = scene_file.split_scenes or scene_file.scenes
    else:
        scenes = scene_file.scenes or scene_file.split_scenes

    result = ScenePatchResult()
    for position, scene in enumerate(scenes):
        if not isinstance(scene, dict) or not selector.matches(position, scene):
            continue
        _apply_zone_patch(scene, patch)
        result.patched_positions.append(position)
    if not result.patched_positions:
        result.issues.append(ValidationIssue("warning", "selector matched no scenes"))
        return result
    result.issues.extend(validate_scene_file(scene_file))
    if any(issue.is_error for issue in result.issues):
        raise SceneFileError(
            "zone patch produced invalid scenes: "
            + "; ".join(issue.message for issue in result.issues if issue.is_error)
        )
    save_scene_file(path, scene_file, tx=tx)
    return result


def _apply_zone_patch(scene: Dict[str, Any], patch: ZonePatch) -> None:
    if patch.raw_overrides is not None:
        scene["zone_overrides"] = dict(patch.raw_overrides)
        return
    overrides = scene.get("zone_overrides")
    if not isinstance(overrides, dict):
        overrides = {}
        scene["zone_overrides"] = overrides
    if patch.video_params is not None:
        overrides["video_params"] = [str(token) for token in patch.video_params]
    if patch.min_scene_len is not None:
        overrides["min_scene_len"] = int(patch.min_scene_len)
    if patch.extra_splits_len is not None:
        overrides["extra_splits_len"] = int(patch.extra_splits_len)
    if patch.photon_noise is not None:
        overrides["photon_noise"] = int(patch.photon_noise)
    if patch.chroma_noise is not None:
        overrides["chroma_noise"] = bool(patch.chroma_noise)


def rebuild_split_scenes_for_chunks(
    scene_path: Path,
    chunks: List[ChunkState],
    *,
    tx: Optional[ManageTransaction] = None,
) -> None:
    """Sync ``split_scenes`` (and ``scenes``) with chunk ranges after edits.

    For each chunk range the override payload comes from the old split scene
    (or scene) that contains the range start, so zone settings survive edits.
    ``split_scenes`` is what av1an chunks from, so it always tracks chunk
    geometry 1:1. ``scenes`` (the pre-extra-split list) is rebuilt too whenever
    it mirrored ``split_scenes`` or was absent — in this pipeline the two are
    identical — so the file stays consistent "as if it had always been this
    way"; a genuinely coarser ``scenes`` list is left untouched.
    """
    scene_file = load_scene_file(scene_path)
    old_scenes = scene_file.scenes
    old_split = scene_file.split_scenes
    base = old_split or old_scenes
    if not base:
        raise SceneFileError(f"scene file has no scenes to rebuild from: {scene_path}")

    def covering_scene(frame: int) -> Optional[Dict[str, Any]]:
        for scene in base:
            if isinstance(scene, dict):
                if int(scene.get("start_frame") or 0) <= frame < int(scene.get("end_frame") or 0):
                    return scene
        return None

    new_split: List[Dict[str, Any]] = []
    for chunk in sorted(chunks, key=lambda item: item.start_frame):
        template = covering_scene(chunk.start_frame) or {}
        scene: Dict[str, Any] = {
            key: value
            for key, value in template.items()
            if key not in ("start_frame", "end_frame")
        }
        scene["start_frame"] = chunk.start_frame
        scene["end_frame"] = chunk.end_frame
        new_split.append(scene)

    scene_file.data["split_scenes"] = new_split
    if not old_scenes or _same_scene_boundaries(old_scenes, old_split):
        scene_file.data["scenes"] = [dict(scene) for scene in new_split]
    save_scene_file(scene_path, scene_file, tx=tx)


def _same_scene_boundaries(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> bool:
    """True if both scene lists describe the same (start, end) sequence."""
    def bounds(scenes: List[Dict[str, Any]]) -> List[tuple]:
        out: List[tuple] = []
        for scene in scenes:
            if isinstance(scene, dict):
                out.append((int(scene.get("start_frame") or 0), int(scene.get("end_frame") or 0)))
        return out

    return bool(left) and bool(right) and bounds(left) == bounds(right)


__all__ = [
    "KNOWN_SCENE_FILES",
    "SceneFile",
    "SceneFileError",
    "SceneFileRef",
    "ScenePatchResult",
    "SceneSelector",
    "ValidationIssue",
    "ZonePatch",
    "list_scene_files",
    "load_scene_file",
    "patch_scene_region",
    "rebuild_split_scenes_for_chunks",
    "save_scene_file",
    "validate_scene_file",
]
