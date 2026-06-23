from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence

from utils import workdir_layout as layout
from utils.pipeline_runtime import final_output_path_for_source
from utils.runner.stage_bank import downstream_stage_names
from utils.runner_state import (
    STAGE_ATTACHMENTS,
    STAGE_AUDIO,
    STAGE_AUTOBOOST_PSD_SCENE,
    STAGE_AUTOBOOST_SCENE,
    STAGE_DEMUX,
    STAGE_FASTPASS,
    STAGE_HDR_PATCH,
    STAGE_MAINPASS,
    STAGE_MUX,
    STAGE_SSIMU2,
    STAGE_VERIFY,
    STAGE_ZONE_BOUNDARIES,
    STAGE_ZONE_EDIT,
    STAGE_ZONE_RECALC,
    autoboost_base_scenes,
    autoboost_fastpass_output,
    autoboost_ssimu2_log,
    autoboost_stage4_scenes,
    stage_marker_path,
)

from utils.manage import av1an_state
from utils.manage.av1an_state import DEFAULT_POLICY_BY_PASS, EncodeOutputPolicy, PassName
from utils.manage.backup import ManageTransaction
from utils.manage.context import WorkdirContext, make_runner_item
from utils.manage.status import is_runner_active

ActionKind = Literal["delete", "delete_tree", "unlink_marker"]


class ResetBlockedError(RuntimeError):
    pass


@dataclass(frozen=True)
class FileAction:
    action: ActionKind
    path: Path
    reason: str
    required: bool = True


@dataclass
class ResetPlan:
    workdir: Path
    operation: str
    stages: List[str] = field(default_factory=list)
    actions: List[FileAction] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def existing_actions(self) -> List[FileAction]:
        return [action for action in self.actions if action.path.exists()]


@dataclass
class ResetResult:
    plan: ResetPlan
    changed_paths: List[Path] = field(default_factory=list)
    backup_manifest: Optional[Path] = None


def _dedupe(actions: Sequence[FileAction]) -> List[FileAction]:
    out: List[FileAction] = []
    seen: set[str] = set()
    for action in actions:
        key = f"{action.action}|{str(action.path).lower()}"
        if key in seen:
            continue
        seen.add(key)
        out.append(action)
    return out


def stage_reset_actions(ctx: WorkdirContext, stage: str) -> List[FileAction]:
    """Registry of per-stage reset actions (markers + stage-owned artifacts).

    Single-stage reset removes only this stage's marker/artifacts; downstream
    invalidation is the job of ``chain=True`` which composes registries of the
    downstream stages.
    """
    item = make_runner_item(ctx)
    workdir = ctx.workdir
    actions: List[FileAction] = []

    def marker(stage_name: str) -> None:
        path = stage_marker_path(item, stage_name)
        if path is not None:
            actions.append(FileAction("unlink_marker", path, f"{stage_name} marker"))

    if stage == STAGE_DEMUX:
        marker(stage)
        actions.append(FileAction("delete", layout.demux_manifest_path(workdir), "demux manifest"))
    elif stage == STAGE_ATTACHMENTS:
        marker(stage)
        actions.append(
            FileAction("delete", layout.attachments_report_path(workdir), "cleaner report")
        )
    elif stage == STAGE_AUTOBOOST_PSD_SCENE:
        marker(stage)
        actions.append(FileAction("delete_tree", layout.psd_dir(workdir), "PSD scene detection output"))
    elif stage == STAGE_AUTOBOOST_SCENE:
        # Scene detection owns the base scenes: av1an `--sc-only` writes them (or
        # the single-scene "none" mode). Delete them so detection actually re-runs
        # instead of resuming. av1an detection uses video/fastpass as its av1an
        # temp, so clear that too. Fastpass depends on SCD, so the reset chain
        # re-runs Fastpass after this.
        marker(stage)
        actions.append(FileAction("delete", autoboost_base_scenes(item), "base scenes (scene detection output)"))
        actions.append(FileAction("delete_tree", layout.fastpass_dir(workdir), "av1an detection + fastpass temp", required=False))
    elif stage == STAGE_FASTPASS:
        # Resetting Fastpass must NOT reset scene detection: drop only the encode
        # output, keeping the base scenes (the SCD result) so SCD is not re-run.
        marker(stage)
        actions.append(FileAction("delete", autoboost_fastpass_output(item), "fastpass encode output"))
    elif stage == STAGE_SSIMU2:
        marker(stage)
        actions.append(FileAction("delete", autoboost_ssimu2_log(item), "SSIMU2 metrics log"))
        actions.append(FileAction("delete", autoboost_stage4_scenes(item), "auto-boost derived scenes"))
        actions.append(FileAction("delete", layout.legacy_hdr_scenes(workdir), "legacy derived scenes", required=False))
    elif stage == STAGE_ZONE_BOUNDARIES:
        marker(stage)
        actions.append(FileAction("delete", layout.boundary_scenes(workdir), "zone boundaries scenes"))
    elif stage == STAGE_ZONE_RECALC:
        marker(stage)
        actions.append(FileAction("delete", layout.recalc_scenes(workdir), "zone recalc scenes"))
    elif stage == STAGE_ZONE_EDIT:
        marker(stage)
        actions.append(FileAction("delete", layout.zoned_scenes(workdir), "zone edited scenes"))
    elif stage == STAGE_HDR_PATCH:
        marker(stage)
        actions.append(FileAction("delete", layout.final_scenes(workdir), "final scenes"))
        actions.append(FileAction("delete_tree", layout.hdr_tmp_dir(workdir), "HDR patch temp", required=False))
        actions.append(
            FileAction(
                "unlink_marker",
                layout.video_state_dir(workdir) / "FINAL_SCENES_COMPLETED",
                "legacy final scenes marker",
                required=False,
            )
        )
    elif stage == STAGE_MAINPASS:
        actions.append(FileAction("delete_tree", layout.mainpass_dir(workdir), "mainpass temp (chunks, encode)"))
        actions.append(FileAction("delete", layout.video_final_output(workdir), "mainpass output"))
    elif stage == STAGE_AUDIO:
        marker(stage)
        actions.append(FileAction("delete", layout.audio_manifest_path(workdir), "audio manifest"))
        actions.append(FileAction("delete_tree", layout.audio_dir(workdir), "audio outputs"))
    elif stage == STAGE_VERIFY:
        marker(stage)
        actions.append(FileAction("delete", layout.logs_dir(workdir) / "verify_error.txt", "verify error log", required=False))
    elif stage == STAGE_MUX:
        marker(stage)
        actions.append(FileAction("delete", layout.mux_manifest_path(workdir), "mux manifest", required=False))
        # the muxed file next to the source is kept on purpose: it is the
        # user-facing result and a re-run overwrites it anyway
    else:
        raise RuntimeError(f"no reset registry entry for stage: {stage}")
    return actions


def _chain_stages(ctx: WorkdirContext, stage: str, chain: bool) -> List[str]:
    if not chain:
        return [stage]
    return [stage, *downstream_stage_names(stage, ctx.stage_names, ctx.stage_bank)]


def preview_stage_reset(ctx: WorkdirContext, stage: str, chain: bool = False) -> ResetPlan:
    stages = _chain_stages(ctx, stage, chain)
    actions: List[FileAction] = []
    for stage_name in stages:
        actions.extend(stage_reset_actions(ctx, stage_name))
    plan = ResetPlan(
        workdir=ctx.workdir,
        operation="reset_chain" if chain else "reset_stage",
        stages=stages,
        actions=_dedupe(actions),
    )
    if stage == STAGE_MUX or STAGE_MUX in stages:
        final_output = final_output_path_for_source(make_runner_item(ctx).source)
        if final_output.exists():
            plan.notes.append(f"final output is kept and will be overwritten on re-run: {final_output}")
    return plan


def preview_file_changes(actions: List[FileAction]) -> ResetPlan:
    return ResetPlan(workdir=Path("."), operation="custom", actions=_dedupe(actions))


def ensure_not_running(ctx: WorkdirContext) -> None:
    if is_runner_active(ctx):
        raise ResetBlockedError(
            f"runner is active for {ctx.workdir}; destructive operations are blocked"
        )


def apply_actions(
    ctx: WorkdirContext,
    plan: ResetPlan,
    *,
    message: str = "",
    extra: Optional[Dict[str, object]] = None,
) -> ResetResult:
    ensure_not_running(ctx)
    tx = ManageTransaction(
        workdir=ctx.workdir,
        operation=plan.operation,
        source=ctx.source,
        plan=ctx.plan_path,
        notes=list(plan.notes),
    )
    changed: List[Path] = []
    for action in plan.actions:
        path = Path(action.path)
        if action.action in ("delete", "unlink_marker"):
            if tx.delete_file(path):
                changed.append(path)
        elif action.action == "delete_tree":
            if tx.delete_tree(path):
                changed.append(path)
        else:
            raise RuntimeError(f"unknown file action: {action.action}")
    manifest = tx.commit(message=message, extra={"stages": plan.stages, **(extra or {})})
    return ResetResult(plan=plan, changed_paths=changed, backup_manifest=manifest)


def reset_stage(ctx: WorkdirContext, stage: str, chain: bool = False) -> ResetResult:
    plan = preview_stage_reset(ctx, stage, chain=chain)
    return apply_actions(ctx, plan, message=f"reset {'chain' if chain else 'stage'}: {stage}")


def chunk_downstream_reset(ctx: WorkdirContext, pass_name: PassName) -> ResetPlan:
    """Artifacts/markers to invalidate after editing a chunk in ``pass_name``.

    Single source of truth for the fastpass/mainpass downstream-invalidation set
    shared by :func:`preview_chunk_reset`, :func:`reset_chunk` and the manage
    GUI's post-geometry cleanup, so the three never drift. Does NOT touch the
    chunk's own encode output / done.json — that is the caller's job (the preview
    lists it; ``reset_chunk`` delegates to ``mark_chunk_not_done``).
    """
    item = make_runner_item(ctx)
    actions: List[FileAction] = []
    if pass_name == "fastpass":
        actions.append(FileAction("delete", autoboost_fastpass_output(item), "fastpass output container"))
        fastpass_marker = stage_marker_path(item, STAGE_FASTPASS)
        if fastpass_marker is not None:
            actions.append(FileAction("unlink_marker", fastpass_marker, "Fastpass marker"))
        downstream_plan = preview_stage_reset(ctx, STAGE_SSIMU2, chain=True)
        stages = [STAGE_FASTPASS, *downstream_plan.stages]
        actions.extend(downstream_plan.actions)
    else:
        actions.append(FileAction("delete", layout.video_final_output(ctx.workdir), "mainpass output"))
        stages = [STAGE_MAINPASS, STAGE_VERIFY, STAGE_MUX]
        for stage_name in (STAGE_VERIFY, STAGE_MUX):
            stage_marker = stage_marker_path(item, stage_name)
            if stage_marker is not None:
                actions.append(FileAction("unlink_marker", stage_marker, f"{stage_name} marker"))
    return ResetPlan(
        workdir=ctx.workdir, operation="chunk_downstream", stages=stages, actions=_dedupe(actions)
    )


def preview_chunk_reset(
    ctx: WorkdirContext,
    pass_name: PassName,
    chunk_index: int,
    *,
    policy: Optional[EncodeOutputPolicy] = None,
) -> ResetPlan:
    """Preview of reset_chunk: chunk state plus downstream stage artifacts."""
    effective_policy: EncodeOutputPolicy = policy or DEFAULT_POLICY_BY_PASS[pass_name]
    pdir = av1an_state.pass_dir(ctx, pass_name)
    chunks = av1an_state.load_chunks(pdir)
    chunk = next((item for item in chunks if item.index == int(chunk_index)), None)
    if chunk is None:
        raise RuntimeError(f"chunk index {chunk_index} not found in {pass_name}")

    actions: List[FileAction] = []
    output = av1an_state.chunk_output_path(chunk, pdir)
    actions.append(
        FileAction(
            "delete",
            output,
            "encode output (renamed to quarantine)" if effective_policy == "quarantine" else "encode output",
        )
    )
    for sidecar in av1an_state.sidecar_paths(pdir, chunk.index):
        actions.append(FileAction("delete", sidecar, "chunk sidecar/probe file"))

    downstream = chunk_downstream_reset(ctx, pass_name)
    actions.extend(downstream.actions)

    plan = ResetPlan(workdir=ctx.workdir, operation="reset_chunk", stages=downstream.stages, actions=[])
    plan.notes.append(f"done.json entry {chunk.name} will be removed")
    plan.actions = _dedupe(actions)
    return plan


def reset_chunk(
    ctx: WorkdirContext,
    pass_name: PassName,
    chunk_index: int,
    *,
    policy: Optional[EncodeOutputPolicy] = None,
) -> ResetResult:
    ensure_not_running(ctx)
    effective_policy: EncodeOutputPolicy = policy or DEFAULT_POLICY_BY_PASS[pass_name]
    plan = preview_chunk_reset(ctx, pass_name, chunk_index, policy=effective_policy)
    pdir = av1an_state.pass_dir(ctx, pass_name)

    tx = ManageTransaction(
        workdir=ctx.workdir,
        operation="reset_chunk",
        source=ctx.source,
        plan=ctx.plan_path,
        notes=list(plan.notes),
    )
    changed = av1an_state.mark_chunk_not_done(pdir, chunk_index, policy=effective_policy, tx=tx)

    for action in chunk_downstream_reset(ctx, pass_name).actions:
        if action.action == "delete_tree":
            if tx.delete_tree(action.path):
                changed.append(action.path)
        elif tx.delete_file(action.path):
            changed.append(action.path)

    manifest = tx.commit(
        message=f"reset chunk {chunk_index} ({pass_name}, policy={effective_policy})",
        extra={"pass": pass_name, "chunk_index": int(chunk_index), "stages": plan.stages},
    )
    return ResetResult(plan=plan, changed_paths=changed, backup_manifest=manifest)


def reset_crf_recalc(ctx: WorkdirContext) -> ResetResult:
    """Legacy batch-manager «crf calc» reset: drop derived scenes + zoning chain.

    Keeps the SSIMU2 log and marker — the stage re-runs via stale-marker
    detection and reuses the metrics to rebuild base scenes.
    """
    zone_plan = preview_stage_reset(ctx, STAGE_ZONE_BOUNDARIES, chain=True)
    item = make_runner_item(ctx)
    actions = [
        FileAction("delete", autoboost_stage4_scenes(item), "auto-boost derived scenes"),
        FileAction("delete", layout.legacy_hdr_scenes(ctx.workdir), "legacy derived scenes", required=False),
        *zone_plan.actions,
    ]
    plan = ResetPlan(
        workdir=ctx.workdir,
        operation="reset_crf_recalc",
        stages=[STAGE_SSIMU2, *zone_plan.stages],
        actions=_dedupe(actions),
        notes=["SSIMU2 log/marker kept; stage becomes stale and re-runs CRF calc from metrics"],
    )
    return apply_actions(ctx, plan, message="reset crf recalc (derived scenes + zoning chain)")


__all__ = [
    "FileAction",
    "ResetBlockedError",
    "ResetPlan",
    "ResetResult",
    "apply_actions",
    "chunk_downstream_reset",
    "ensure_not_running",
    "preview_chunk_reset",
    "preview_file_changes",
    "preview_stage_reset",
    "reset_chunk",
    "reset_crf_recalc",
    "reset_stage",
    "stage_reset_actions",
]
