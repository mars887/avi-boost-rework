from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.plan_model import ResolvedFilePlan
from utils.runner_source_info import (
    fastpass_output_path_for_item,
    item_source_duration,
    item_source_size,
    output_path_for_item,
    safe_file_size,
)
from utils.runner_state import (
    CACHED_STAGE_MESSAGE,
    is_zone_edit_pipeline_stage,
    public_stage_name,
)


@dataclass
class StageState:
    name: str
    status: str = "pending"
    started_at: float = 0.0
    ended_at: float = 0.0
    message: str = ""
    progress: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        elapsed = 0.0
        if self.started_at:
            elapsed = (self.ended_at or time.time()) - self.started_at
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "details": dict(self.details),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_seconds": round(elapsed, 3),
        }


def _first_nonzero(values: List[float]) -> float:
    nonzero = [value for value in values if value > 0]
    return min(nonzero) if nonzero else 0.0


def _last_nonzero(values: List[float]) -> float:
    nonzero = [value for value in values if value > 0]
    return max(nonzero) if nonzero else 0.0


def _visible_zone_edit_snapshot(stages: List[StageState]) -> Dict[str, Any]:
    statuses = [str(stage.status or "pending").lower() for stage in stages]
    failed = next((stage for stage in stages if str(stage.status or "").lower() == "failed"), None)
    running = [stage for stage in stages if str(stage.status or "").lower() == "started"]
    successful = {"completed", "skipped"}

    if failed is not None:
        status = "failed"
        message = failed.message
        progress = failed.progress
        details = dict(failed.details)
    elif running:
        current = running[-1]
        status = "started"
        message = ""
        progress = current.progress
        details = dict(current.details)
    elif statuses and all(status in successful for status in statuses):
        status = "completed"
        message = CACHED_STAGE_MESSAGE if all(stage.message == CACHED_STAGE_MESSAGE for stage in stages) else ""
        progress = None
        details = {}
    elif any(status in successful for status in statuses):
        status = "started"
        message = ""
        progress = None
        details = {}
    else:
        status = "pending"
        message = ""
        progress = None
        details = {}

    started_at = _first_nonzero([float(stage.started_at or 0.0) for stage in stages])
    ended_at = _last_nonzero([float(stage.ended_at or 0.0) for stage in stages]) if status in ("completed", "failed", "skipped") else 0.0
    elapsed = 0.0
    if started_at:
        elapsed = (ended_at or time.time()) - started_at

    return {
        "name": public_stage_name(stages[0].name if stages else ""),
        "status": status,
        "message": message,
        "progress": progress,
        "details": details,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": round(elapsed, 3),
    }


def visible_stage_snapshots(stages: List[StageState]) -> List[Dict[str, Any]]:
    zone_stages = [stage for stage in stages if is_zone_edit_pipeline_stage(stage.name)]
    emitted_zone = False
    snapshots: List[Dict[str, Any]] = []
    for stage in stages:
        if is_zone_edit_pipeline_stage(stage.name):
            if not emitted_zone:
                snapshots.append(_visible_zone_edit_snapshot(zone_stages))
                emitted_zone = True
            continue
        snapshots.append(stage.snapshot())
    return snapshots


@dataclass
class ActivePlanState:
    plan_run_id: str
    item: "QueueItem"
    status: str = "running"
    started_at: float = 0.0
    ended_at: float = 0.0
    message: str = ""
    stages: List[StageState] = field(default_factory=list)

    def stage(self, name: str) -> StageState:
        for state in self.stages:
            if state.name == name:
                return state
        state = StageState(name=name)
        self.stages.append(state)
        return state

    def snapshot(self) -> Dict[str, Any]:
        source_size = item_source_size(self.item)
        output_path = output_path_for_item(self.item)
        output_size = safe_file_size(output_path)
        fastpass_output_path = fastpass_output_path_for_item(self.item)
        fastpass_output_size = safe_file_size(fastpass_output_path)
        elapsed = 0.0
        if self.started_at:
            elapsed = (self.ended_at or time.time()) - self.started_at
        return {
            "plan_run_id": self.plan_run_id,
            "status": self.status,
            "message": self.message,
            "mode": self.item.mode,
            "name": self.item.name,
            "plan": str(self.item.plan_path),
            "source": str(self.item.source),
            "source_size": source_size,
            "duration_seconds": item_source_duration(self.item),
            "output": str(output_path),
            "output_size": output_size,
            "fastpass_output": str(fastpass_output_path),
            "fastpass_output_size": fastpass_output_size,
            "workdir": str(self.item.workdir),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_seconds": round(elapsed, 3),
            "stages": visible_stage_snapshots(self.stages),
        }


@dataclass(frozen=True)
class FinishedPlanState:
    plan_run_id: str
    item: "QueueItem"
    status: str
    started_at: float
    ended_at: float
    stage: str = ""
    message: str = ""

    def snapshot(self) -> Dict[str, Any]:
        output_path = output_path_for_item(self.item)
        output_size = safe_file_size(output_path)
        fastpass_output_path = fastpass_output_path_for_item(self.item)
        fastpass_output_size = safe_file_size(fastpass_output_path)
        source_size = item_source_size(self.item)
        return {
            "plan_run_id": self.plan_run_id,
            "status": self.status,
            "mode": self.item.mode,
            "name": self.item.name,
            "plan": str(self.item.plan_path),
            "source": str(self.item.source),
            "source_size": source_size,
            "duration_seconds": item_source_duration(self.item),
            "output": str(output_path),
            "output_size": output_size,
            "fastpass_output": str(fastpass_output_path),
            "fastpass_output_size": fastpass_output_size,
            "workdir": str(self.item.workdir),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_seconds": round(max(0.0, self.ended_at - self.started_at), 3),
            "stage": public_stage_name(self.stage),
            "message": self.message,
        }


@dataclass(frozen=True)
class QueueItem:
    resolved: ResolvedFilePlan
    mode: str

    @property
    def plan_path(self) -> Path:
        return self.resolved.paths.plan_path

    @property
    def source(self) -> Path:
        return self.resolved.paths.source

    @property
    def source_dir(self) -> str:
        return str(self.source.parent.resolve())

    @property
    def workdir(self) -> Path:
        return self.resolved.paths.workdir

    @property
    def name(self) -> str:
        return self.resolved.plan.meta.name or self.source.stem


@dataclass
class PlanExecution:
    plan_run_id: str
    item: QueueItem
    commands: List[Tuple[str, List[str]]]
    failed_stages: Dict[str, str] = field(default_factory=dict)
    blocked_stages: Dict[str, str] = field(default_factory=dict)

    @property
    def stage_names(self) -> List[str]:
        return [stage for stage, _cmd in self.commands]

    def command_for_stage(self, stage: str) -> List[str]:
        for name, cmd in self.commands:
            if name == stage:
                return cmd
        raise KeyError(stage)


@dataclass
class RunningStageTask:
    plan_run_id: str
    item: QueueItem
    stage: str
    cmd: List[str]
    cost: int
    thread: Optional[threading.Thread] = None
    done: threading.Event = field(default_factory=threading.Event)
    error: Optional[BaseException] = None
