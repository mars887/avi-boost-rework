from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.runner.stage_bank import downstream_stage_names, effective_stage_dependencies
from utils.runner_lock import SourceDirLock
from utils.runner_state import (
    STAGE_ITEM,
    is_zone_edit_pipeline_stage,
    public_stage_name,
    stage_resume_info,
)

from utils.manage.context import RunnerItemAdapter, WorkdirContext, load_json_file, make_runner_item

STATE_NOT_STARTED = "not_started"
STATE_RUNNING = "running"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"
STATE_SKIPPED = "skipped"
STATE_STALE_MARKER = "stale_marker"
STATE_ARTIFACT_MISSING = "artifact_missing"
STATE_UNKNOWN = "unknown"

RUNNER_LOCK_NAME = ".pbbatch_runner.lock"
RECENT_SNAPSHOT_SECONDS = 180.0
MAX_CHILD_EVENT_FILES = 32


@dataclass
class StageStatus:
    name: str
    public_name: str
    internal: bool
    state: str
    progress: Optional[float]
    started_at: Optional[float]
    ended_at: Optional[float]
    elapsed_seconds: Optional[float]
    marker_path: Optional[Path]
    marker_exists: bool
    artifact_valid: bool
    reason: str
    last_message: str
    dependencies: List[str] = field(default_factory=list)
    downstream: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunnerHistory:
    events_by_stage: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_item_event: Dict[str, Any] = field(default_factory=dict)
    last_event_at: float = 0.0
    snapshot: Dict[str, Any] = field(default_factory=dict)
    snapshot_plan: Dict[str, Any] = field(default_factory=dict)
    child_progress: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    session_id: str = ""
    plan_run_id: str = ""


@dataclass
class RunnerActivity:
    active: bool
    pid: int = 0
    session_id: str = ""
    lock_path: Optional[Path] = None
    warning: str = ""


@dataclass
class WorkdirSummary:
    name: str
    workdir: Path
    source: Optional[Path]
    plan_path: Optional[Path]
    mode: str
    runner: RunnerActivity
    stage_counts: Dict[str, int]
    current_stage: str
    warnings: List[str]

    @property
    def total_stages(self) -> int:
        return sum(self.stage_counts.values())

    @property
    def completed_stages(self) -> int:
        return self.stage_counts.get(STATE_COMPLETED, 0) + self.stage_counts.get(STATE_SKIPPED, 0)


def _norm(path: Any) -> str:
    return str(path or "").replace("/", "\\").strip().lower().rstrip("\\")


def _iter_jsonl(path: Path):
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    yield payload
    except OSError:
        return


def load_runner_history(ctx: WorkdirContext) -> RunnerHistory:
    history = RunnerHistory()
    meta_dir = ctx.meta_dir
    workdir_key = _norm(ctx.workdir)

    for event in _iter_jsonl(meta_dir / "runner_events.jsonl"):
        if _norm(event.get("workdir")) != workdir_key:
            continue
        stage = str(event.get("stage") or "")
        timestamp = float(event.get("timestamp") or 0.0)
        history.last_event_at = max(history.last_event_at, timestamp)
        history.session_id = str(event.get("session_id") or history.session_id)
        history.plan_run_id = str(event.get("plan_run_id") or history.plan_run_id)
        if stage == STAGE_ITEM:
            history.last_item_event = event
            continue
        if not stage:
            continue
        previous = history.events_by_stage.get(stage)
        if previous is not None and str(event.get("status") or "") == "progress":
            # keep richer terminal info but refresh progress/message/timestamps
            merged = dict(previous)
            merged.update(
                status="progress",
                message=str(event.get("message") or previous.get("message") or ""),
                progress=event.get("progress"),
                timestamp=timestamp,
                started_at=event.get("started_at") or previous.get("started_at"),
            )
            history.events_by_stage[stage] = merged
            continue
        history.events_by_stage[stage] = event

    snapshot = load_json_file(meta_dir / "runner_state.json")
    if isinstance(snapshot, dict):
        history.snapshot = snapshot
        for entry in snapshot.get("active") or []:
            if isinstance(entry, dict) and _norm(entry.get("workdir")) == workdir_key:
                history.snapshot_plan = entry
                break

    history.child_progress = _load_child_progress(meta_dir, plan_run_id=history.plan_run_id)
    return history


def _load_child_progress(meta_dir: Path, *, plan_run_id: str = "") -> Dict[str, Dict[str, Any]]:
    try:
        files = sorted(
            meta_dir.glob("runner_child_*.jsonl"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return {}
    if plan_run_id:
        preferred = [path for path in files if plan_run_id in path.name]
        files = preferred or files
    out: Dict[str, Dict[str, Any]] = {}
    for path in files[:MAX_CHILD_EVENT_FILES]:
        for event in _iter_jsonl(path):
            stage = str(event.get("stage") or "")
            if not stage:
                continue
            current = out.get(stage)
            if current is None or float(event.get("timestamp") or 0.0) >= float(current.get("timestamp") or 0.0):
                out[stage] = event
    return out


def read_runner_lock(ctx: WorkdirContext) -> Optional[Dict[str, Any]]:
    if ctx.source is None:
        return None
    lock_path = ctx.source.parent / RUNNER_LOCK_NAME
    payload = load_json_file(lock_path)
    if isinstance(payload, dict):
        payload["_lock_path"] = str(lock_path)
        return payload
    return None


def runner_activity(ctx: WorkdirContext) -> RunnerActivity:
    lock = read_runner_lock(ctx)
    if lock is not None:
        pid = int(lock.get("pid") or 0)
        lock_path = Path(str(lock.get("_lock_path")))
        if pid and SourceDirLock._pid_alive(pid):
            return RunnerActivity(
                active=True,
                pid=pid,
                session_id=str(lock.get("session_id") or ""),
                lock_path=lock_path,
            )
        return RunnerActivity(
            active=False,
            pid=pid,
            session_id=str(lock.get("session_id") or ""),
            lock_path=lock_path,
            warning="runner lock file exists but its process is dead; runner may have crashed",
        )

    snapshot_path = ctx.meta_dir / "runner_state.json"
    try:
        age = time.time() - snapshot_path.stat().st_mtime
    except OSError:
        age = None
    if age is not None and 0 <= age <= RECENT_SNAPSHOT_SECONDS:
        return RunnerActivity(
            active=False,
            warning=(
                "runner_state.json was updated recently but no runner lock found; "
                "runner may have exited uncleanly"
            ),
        )
    return RunnerActivity(active=False)


def is_runner_active(ctx: WorkdirContext) -> bool:
    return runner_activity(ctx).active


def get_stage_statuses(
    ctx: WorkdirContext,
    include_internal: bool = True,
    *,
    history: Optional[RunnerHistory] = None,
    activity: Optional[RunnerActivity] = None,
) -> List[StageStatus]:
    item = make_runner_item(ctx)
    history = history or load_runner_history(ctx)
    activity = activity or runner_activity(ctx)
    statuses: List[StageStatus] = []
    for stage in ctx.stage_names:
        internal = is_zone_edit_pipeline_stage(stage) and public_stage_name(stage) != stage
        if internal and not include_internal:
            continue
        statuses.append(_build_stage_status(ctx, item, stage, history, runner_active=activity.active))
    return statuses


def get_stage_status(ctx: WorkdirContext, stage: str) -> StageStatus:
    item = make_runner_item(ctx)
    history = load_runner_history(ctx)
    activity = runner_activity(ctx)
    return _build_stage_status(ctx, item, stage, history, runner_active=activity.active)


def _build_stage_status(
    ctx: WorkdirContext,
    item: RunnerItemAdapter,
    stage: str,
    history: RunnerHistory,
    *,
    runner_active: bool,
) -> StageStatus:
    resume = stage_resume_info(item, stage)
    event = history.events_by_stage.get(stage) or {}
    event_status = str(event.get("status") or "").lower()
    child = history.child_progress.get(stage) or {}

    progress: Optional[float] = None
    for candidate in (child.get("progress"), event.get("progress")):
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            progress = value
            break

    started_at = float(event.get("started_at") or 0.0) or None
    ended_at = float(event.get("ended_at") or 0.0) or None
    elapsed = float(event.get("elapsed_seconds") or 0.0) or None
    last_message = str(event.get("message") or "")
    details = dict(child.get("details") or {})

    reason = resume.reason
    if resume.completed:
        state = STATE_COMPLETED
    elif runner_active and event_status in ("started", "progress"):
        state = STATE_RUNNING
    elif resume.marker_exists and not resume.marker_valid:
        state = STATE_STALE_MARKER
    elif event_status == "failed":
        state = STATE_FAILED
    elif event_status == "skipped":
        state = STATE_SKIPPED
    elif resume.marker_exists and resume.marker_valid:
        # marker fine but a dependent completion artifact is missing
        state = STATE_ARTIFACT_MISSING
    elif event_status == "completed":
        state = STATE_ARTIFACT_MISSING
        reason = reason or "history says completed but artifacts are missing"
    elif event_status in ("started", "progress"):
        # runner is gone; an interrupted stage resumes from scratch/markers
        state = STATE_NOT_STARTED
        reason = reason or "interrupted (runner not active)"
    else:
        state = STATE_NOT_STARTED

    if state in (STATE_COMPLETED, STATE_SKIPPED):
        progress = None

    return StageStatus(
        name=stage,
        public_name=public_stage_name(stage),
        internal=is_zone_edit_pipeline_stage(stage) and public_stage_name(stage) != stage,
        state=state,
        progress=progress,
        started_at=started_at,
        ended_at=ended_at,
        elapsed_seconds=elapsed,
        marker_path=resume.marker,
        marker_exists=resume.marker_exists,
        artifact_valid=resume.marker_valid,
        reason=reason,
        last_message=last_message,
        dependencies=effective_stage_dependencies(stage, ctx.stage_names, ctx.stage_bank),
        downstream=downstream_stage_names(stage, ctx.stage_names, ctx.stage_bank),
        details=details,
    )


def summarize_workdir(ctx: WorkdirContext) -> WorkdirSummary:
    history = load_runner_history(ctx)
    activity = runner_activity(ctx)
    statuses = get_stage_statuses(ctx, include_internal=True, history=history, activity=activity)
    counts: Dict[str, int] = {}
    current_stage = ""
    for status in statuses:
        counts[status.state] = counts.get(status.state, 0) + 1
        if status.state == STATE_RUNNING and not current_stage:
            current_stage = status.name
    if not current_stage:
        for status in statuses:
            if status.state not in (STATE_COMPLETED, STATE_SKIPPED):
                current_stage = status.name
                break
    warnings = list(ctx.warnings)
    if activity.warning:
        warnings.append(activity.warning)
    return WorkdirSummary(
        name=ctx.name,
        workdir=ctx.workdir,
        source=ctx.source,
        plan_path=ctx.plan_path,
        mode=ctx.mode,
        runner=activity,
        stage_counts=counts,
        current_stage=current_stage,
        warnings=warnings,
    )


__all__ = [
    "RECENT_SNAPSHOT_SECONDS",
    "RUNNER_LOCK_NAME",
    "RunnerActivity",
    "RunnerHistory",
    "StageStatus",
    "STATE_ARTIFACT_MISSING",
    "STATE_COMPLETED",
    "STATE_FAILED",
    "STATE_NOT_STARTED",
    "STATE_RUNNING",
    "STATE_SKIPPED",
    "STATE_STALE_MARKER",
    "STATE_UNKNOWN",
    "WorkdirSummary",
    "get_stage_status",
    "get_stage_statuses",
    "is_runner_active",
    "load_runner_history",
    "read_runner_lock",
    "runner_activity",
    "summarize_workdir",
]
