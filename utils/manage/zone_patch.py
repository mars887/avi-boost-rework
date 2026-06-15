"""Bulk video_params patching for a pass, driven by the zone-edit command DSL.

This is the multi-chunk counterpart of the single-chunk "Edit params" action.
A command selects many chunks at once (the same selector/action grammar as the
``zone-editor`` stage), and the matched chunks get their ``video_params`` edited
in ``chunks.json`` *and* mirrored into the pass's source scenes file, so the
change survives a pass-level reset but is wiped when the scenes file itself is
regenerated upstream (e.g. the HDR-patch stage rebuilds ``scenes-final.json``).

The DSL parsing/apply logic is reused verbatim from ``utils/zone-editor.py`` to
guarantee identical behaviour; only the boundary-edit mode (``| ... |``) is
rejected here because moving chunk boundaries is the job of Split/Merge/Reshape.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.pipeline_runtime import ROOT_DIR

from utils.manage import av1an_state
from utils.manage.av1an_state import EncodeOutputPolicy, PassName
from utils.manage.backup import ManageTransaction
from utils.manage.context import WorkdirContext
from utils.manage.scenes import load_scene_file, save_scene_file

# ``zone-editor.py`` is a hyphenated CLI script, so it cannot be imported with a
# normal ``import``; load it by path once and reuse its pure DSL functions.
_ZONE_EDITOR_PATH = ROOT_DIR / "utils" / "zone-editor.py"
_zone_editor_module: Optional[Any] = None


class ZonePatchError(RuntimeError):
    pass


def _zone_editor() -> Any:
    global _zone_editor_module
    if _zone_editor_module is None:
        if not _ZONE_EDITOR_PATH.exists():
            raise ZonePatchError(f"zone-editor.py not found: {_ZONE_EDITOR_PATH}")
        spec = importlib.util.spec_from_file_location("pb_manage_zone_editor", _ZONE_EDITOR_PATH)
        if spec is None or spec.loader is None:
            raise ZonePatchError(f"cannot load zone-editor.py: {_ZONE_EDITOR_PATH}")
        module = importlib.util.module_from_spec(spec)
        # register before exec so dataclasses can resolve the module's __dict__
        # for its string annotations (the script uses ``from __future__ import
        # annotations``)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _zone_editor_module = module
    return _zone_editor_module


def pass_source_scene_file(ctx: WorkdirContext, pass_name: PassName) -> Path:
    """The scenes file each pass's av1an reads (and we mirror edits into)."""
    if pass_name == "mainpass":
        return ctx.workdir / "video" / "scenes-final.json"
    return ctx.workdir / "video" / "scenes.json"


def build_video_info(source: Optional[Path], fallback_fps: float) -> Any:
    """ffprobe the source for fps + chapters; fall back to fps-only metadata.

    Frame/scene-index/time selectors only need fps; chapter-title selectors need
    the source video, so they are silently skipped when no source is available.
    """
    ze = _zone_editor()
    if source is not None and Path(source).exists():
        try:
            return ze.read_video_info(str(source))
        except Exception:
            pass  # fall through to fps-only info
    if fallback_fps and fallback_fps > 0:
        fps = Fraction(fallback_fps).limit_denominator(1_000_000)
    else:
        fps = Fraction(24000, 1001)
    return ze.VideoInfo(fps=fps, duration_sec=0.0, chapters=[], is_vfr_suspected=False)


def parse_patch_commands(text: str, *, video: Any, total_frames: Optional[int]) -> List[Any]:
    """Parse zone-edit command text into commands; reject boundary-edit mode."""
    ze = _zone_editor()
    raw_lines = ze.load_commands(str(text))
    commands = ze.parse_commands(raw_lines, video=video, total_frames=total_frames)
    if not commands:
        raise ZonePatchError("no commands parsed (only comments/presets?)")
    for cmd in commands:
        if cmd.sep == "|" or cmd.boundary_options is not None:
            raise ZonePatchError(
                "boundary-edit mode (selectors | ... | actions) is not supported here; "
                "use Split / Merge / Reshape to change chunk boundaries"
            )
    return commands


@dataclass
class PassParamPatchResult:
    changed_indices: List[int] = field(default_factory=list)
    reset_indices: List[int] = field(default_factory=list)  # changed chunks that were done
    range_params: Dict[Tuple[int, int], List[str]] = field(default_factory=dict)


def apply_param_commands_to_chunks(
    pass_dir: Path,
    commands: List[Any],
    *,
    policy: EncodeOutputPolicy = "delete",
    tx: Optional[ManageTransaction] = None,
) -> PassParamPatchResult:
    """Apply DSL commands to chunk ``video_params``; reset any done chunk changed.

    ``s`` selectors (``5s10``) match on the chunk ``index`` shown in the table;
    frame/time/chapter selectors match on the chunk frame range with the usual
    ``-``/``!``/``^`` overlap rules.
    """
    ze = _zone_editor()
    pass_dir = Path(pass_dir)
    chunks = av1an_state.load_chunks(pass_dir)
    done = av1an_state.load_done(pass_dir)
    result = PassParamPatchResult()

    for chunk in chunks:
        index, start, end = chunk.index, chunk.start_frame, chunk.end_frame
        pairs = ze.parse_video_params([str(token) for token in chunk.video_params])
        changed = False
        for cmd in commands:
            if not ze.scene_selected(index, start, end, cmd):
                continue
            for action in cmd.actions:
                pairs, before, after = ze.apply_action_to_pairs(pairs, action, param_step=ze.DEFAULT_PARAM_STEP)
                if before != after:
                    changed = True
        if not changed:
            continue
        new_params = [str(token) for token in ze.build_video_params(pairs)]
        chunk.data["video_params"] = new_params
        result.changed_indices.append(index)
        result.range_params[(start, end)] = new_params
        if done.is_done(index):
            result.reset_indices.append(index)

    if not result.changed_indices:
        return result

    result.changed_indices.sort()
    result.reset_indices.sort()
    av1an_state.save_chunks(pass_dir, chunks, tx=tx)
    for index in result.reset_indices:
        av1an_state.mark_chunk_not_done(pass_dir, index, policy=policy, tx=tx)
    return result


def mirror_params_to_scene_file(
    scene_path: Path,
    range_params: Dict[Tuple[int, int], List[str]],
    *,
    tx: Optional[ManageTransaction] = None,
) -> int:
    """Mirror per-range ``video_params`` into a scenes file's ``zone_overrides``.

    Both ``scenes`` and ``split_scenes`` entries whose ``[start, end)`` matches a
    patched chunk get their ``zone_overrides.video_params`` set, so chunks.json
    and the scenes file stay in lockstep. Returns the number of entries updated.
    """
    scene_path = Path(scene_path)
    if not range_params or not scene_path.exists():
        return 0
    scene_file = load_scene_file(scene_path)
    updated = 0
    for key in ("scenes", "split_scenes"):
        scenes = scene_file.data.get(key)
        if not isinstance(scenes, list):
            continue
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            rng = (int(scene.get("start_frame") or 0), int(scene.get("end_frame") or 0))
            params = range_params.get(rng)
            if params is None:
                continue
            overrides = scene.get("zone_overrides")
            if not isinstance(overrides, dict):
                overrides = {}
                scene["zone_overrides"] = overrides
            overrides["video_params"] = [str(token) for token in params]
            updated += 1
    if updated:
        save_scene_file(scene_path, scene_file, tx=tx)
    return updated


__all__ = [
    "PassParamPatchResult",
    "ZonePatchError",
    "apply_param_commands_to_chunks",
    "build_video_info",
    "mirror_params_to_scene_file",
    "parse_patch_commands",
    "pass_source_scene_file",
]
