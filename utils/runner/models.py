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
            "stages": [stage.snapshot() for stage in self.stages],
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
            "stage": self.stage,
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
