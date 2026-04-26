from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import tomllib
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from utils.discord_config import discord_config_value
from utils.discord_bridge import DiscordBridge
from utils.pipeline_runtime import (
    AUTOBOOST_DIR,
    ROOT_DIR,
    UTILS_DIR,
    ensure_dir,
    is_mars_av1an_fork,
    load_toolchain,
    read_command_output,
)
from utils.plan_model import FilePlan, ResolvedFilePlan, RunnerEvent, load_plan, resolve_file_plan
from utils.runner_state import (
    CACHED_STAGE_MESSAGE,
    RUNNER_MANAGED_STATE_ENV,
    STAGE_ATTACHMENTS,
    STAGE_AUDIO,
    STAGE_AUTOBOOST_PSD_SCENE,
    STAGE_AUTOBOOST_SCENE,
    STAGE_DEMUX,
    STAGE_FASTPASS,
    STAGE_HDR_PATCH,
    STAGE_ITEM,
    STAGE_MAINPASS,
    STAGE_MUX,
    STAGE_SSIMU2,
    STAGE_VERIFY,
    STAGE_ZONE_EDIT,
    clear_stage_marker,
    display_stage_plan,
    is_cached_stage_message,
    file_not_older_than,
    stage_completion_artifacts_valid,
    stage_resume_info,
    stage_resume_marker_exists,
    write_stage_marker,
)
from utils.runner_lock import SourceDirLock
from utils.runner_source_info import (
    fastpass_output_path_for_item,
    item_inputs_changed,
    item_source_duration,
    item_source_size,
    mark_source_info_clean,
    output_path_for_item,
    prepare_source_info,
    safe_file_size,
)
from utils.zoned_commands import ensure_zoned_command_file

FAST_INTERRUPT = False
WRAPPER_VPY = ROOT_DIR / "wrapper.vpy"
MIN_REUSABLE_OUTPUT_BYTES = 1024

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
PERCENT_RE = re.compile(r"(?<!\d)(100(?:\.0+)?|[0-9]{1,2}(?:\.[0-9]+)?)\s*%")
AV1AN_PROGRESS_RE = re.compile(
    r"(?P<percent>100(?:\.0+)?|[0-9]{1,2}(?:\.[0-9]+)?)\s*%\s+"
    r"(?P<pos>\d+)\s*/\s*(?P<total>\d+)\s*"
    r"\(\s*(?:(?P<fps>[0-9]+(?:\.[0-9]+)?)\s*fps|(?P<spf>[0-9]+(?:\.[0-9]+)?)\s*s/fr|0\s*fps)"
    r"\s*,\s*eta\s+(?P<eta>[^,\)]+)"
    r"(?:,\s*(?P<kbps>[0-9]+(?:\.[0-9]+)?)\s*Kbps)?",
    re.IGNORECASE,
)
AV1AN_CHUNK_RE = re.compile(r"\[(?P<done>\d+)\s*/\s*(?P<total>\d+)\s+Chunks\]", re.IGNORECASE)
_AV1AN_PROGRESS_JSONL_SUPPORT: Dict[str, bool] = {}

CHILD_EVENT_ENV = "PBBATCH_RUNNER_CHILD_EVENTS_JSONL"
SESSION_ID_ENV = "PBBATCH_RUNNER_SESSION_ID"
PLAN_RUN_ID_ENV = "PBBATCH_RUNNER_PLAN_RUN_ID"
STAGE_BANK_CONFIG_FILE = ROOT_DIR / "StagesBankTree.toml"
TERMINAL_STAGE_STATUSES = {"completed", "failed", "skipped"}


@dataclass(frozen=True)
class StageBankStage:
    cost: int = 1
    priority: int = 2
    requires: Tuple[str, ...] = ()


@dataclass(frozen=True)
class StageBankConfig:
    capacity: int
    max_active_plans: int
    max_running_stages: int
    stages: Dict[str, StageBankStage]

    def stage_cost(self, stage: str) -> int:
        return max(1, int(self.stages.get(stage, StageBankStage()).cost))

    def stage_priority(self, stage: str) -> int:
        return max(1, int(self.stages.get(stage, StageBankStage()).priority))

    def stage_requires(self, stage: str) -> Tuple[str, ...]:
        return tuple(self.stages.get(stage, StageBankStage()).requires)


KNOWN_STAGE_NAMES = {
    STAGE_DEMUX,
    STAGE_ATTACHMENTS,
    STAGE_AUTOBOOST_SCENE,
    STAGE_AUTOBOOST_PSD_SCENE,
    STAGE_FASTPASS,
    STAGE_SSIMU2,
    STAGE_ZONE_EDIT,
    STAGE_HDR_PATCH,
    STAGE_MAINPASS,
    STAGE_AUDIO,
    STAGE_VERIFY,
    STAGE_MUX,
}


DEFAULT_STAGE_BANK = StageBankConfig(
    capacity=10,
    max_active_plans=3,
    max_running_stages=5,
    stages={
        STAGE_DEMUX: StageBankStage(cost=2),
        STAGE_ATTACHMENTS: StageBankStage(cost=1, requires=(STAGE_DEMUX,)),
        STAGE_AUTOBOOST_SCENE: StageBankStage(cost=3, requires=(STAGE_DEMUX,)),
        STAGE_AUTOBOOST_PSD_SCENE: StageBankStage(cost=4, requires=(STAGE_DEMUX,)),
        STAGE_FASTPASS: StageBankStage(
            cost=10,
            priority=1,
            requires=(STAGE_AUTOBOOST_SCENE, STAGE_AUTOBOOST_PSD_SCENE),
        ),
        STAGE_SSIMU2: StageBankStage(cost=5, requires=(STAGE_FASTPASS,)),
        STAGE_ZONE_EDIT: StageBankStage(
            cost=2,
            requires=(STAGE_SSIMU2, STAGE_AUTOBOOST_SCENE, STAGE_AUTOBOOST_PSD_SCENE),
        ),
        STAGE_HDR_PATCH: StageBankStage(cost=2, requires=(STAGE_ZONE_EDIT,)),
        STAGE_MAINPASS: StageBankStage(cost=10, priority=1, requires=(STAGE_HDR_PATCH,)),
        STAGE_AUDIO: StageBankStage(cost=2, requires=(STAGE_DEMUX,)),
        STAGE_VERIFY: StageBankStage(
            cost=1,
            priority=3,
            requires=(STAGE_ATTACHMENTS, STAGE_AUDIO, STAGE_MAINPASS),
        ),
        STAGE_MUX: StageBankStage(cost=3, priority=3, requires=(STAGE_VERIFY,)),
    },
)


def _validated_stage_bank_config(
    *,
    capacity: int,
    max_active_plans: int,
    max_running_stages: int,
    stages: Dict[str, StageBankStage],
) -> StageBankConfig:
    for stage_name, stage in stages.items():
        if int(stage.cost) > capacity:
            raise RuntimeError(f"stage cost exceeds bank capacity: {stage_name}")
    _validate_stage_bank_acyclic(stages)
    return StageBankConfig(
        capacity=capacity,
        max_active_plans=max_active_plans,
        max_running_stages=max_running_stages,
        stages=stages,
    )


def _validate_stage_bank_acyclic(stages: Dict[str, StageBankStage]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(stage_name: str, path: List[str]) -> None:
        if stage_name in visited:
            return
        if stage_name in visiting:
            try:
                start = path.index(stage_name)
            except ValueError:
                start = 0
            cycle = path[start:] + [stage_name]
            raise RuntimeError(f"stage bank dependency cycle: {' -> '.join(cycle)}")
        visiting.add(stage_name)
        path.append(stage_name)
        for dependency in stages.get(stage_name, StageBankStage()).requires:
            if dependency in stages:
                visit(dependency, path)
        path.pop()
        visiting.remove(stage_name)
        visited.add(stage_name)

    for stage_name in stages:
        visit(stage_name, [])


def load_stage_bank_config(path: Path = STAGE_BANK_CONFIG_FILE) -> StageBankConfig:
    if not path.exists():
        return DEFAULT_STAGE_BANK
    try:
        with path.open("rb") as fh:
            payload = tomllib.load(fh)
    except Exception as exc:
        raise RuntimeError(f"failed to read stage bank config {path}: {exc}") from exc

    bank = dict(payload.get("bank") or {})
    try:
        capacity = int(bank.get("capacity", DEFAULT_STAGE_BANK.capacity))
    except Exception as exc:
        raise RuntimeError("stage bank capacity must be an integer") from exc
    if capacity <= 0:
        raise RuntimeError("stage bank capacity must be greater than zero")
    try:
        max_active_plans = int(bank.get("max_active_plans", DEFAULT_STAGE_BANK.max_active_plans))
    except Exception as exc:
        raise RuntimeError("stage bank max_active_plans must be an integer") from exc
    if max_active_plans <= 0:
        raise RuntimeError("stage bank max_active_plans must be greater than zero")
    try:
        max_running_stages = int(bank.get("max_running_stages", DEFAULT_STAGE_BANK.max_running_stages))
    except Exception as exc:
        raise RuntimeError("stage bank max_running_stages must be an integer") from exc
    if max_running_stages <= 0:
        raise RuntimeError("stage bank max_running_stages must be greater than zero")

    raw_stages = payload.get("stages")
    if raw_stages is None:
        return _validated_stage_bank_config(
            capacity=capacity,
            max_active_plans=max_active_plans,
            max_running_stages=max_running_stages,
            stages=dict(DEFAULT_STAGE_BANK.stages),
        )
    if not isinstance(raw_stages, dict):
        raise RuntimeError("stage bank [stages] section must be a table")

    stages: Dict[str, StageBankStage] = {}
    for name, raw_stage in raw_stages.items():
        stage_name = str(name or "").strip()
        if stage_name not in KNOWN_STAGE_NAMES:
            raise RuntimeError(f"unknown stage in stage bank config: {stage_name}")
        if not isinstance(raw_stage, dict):
            raise RuntimeError(f"stage config must be a table: {stage_name}")
        try:
            cost = int(raw_stage.get("cost", DEFAULT_STAGE_BANK.stage_cost(stage_name)))
        except Exception as exc:
            raise RuntimeError(f"stage cost must be an integer: {stage_name}") from exc
        if cost <= 0:
            raise RuntimeError(f"stage cost must be greater than zero: {stage_name}")
        if cost > capacity:
            raise RuntimeError(f"stage cost exceeds bank capacity: {stage_name}")
        try:
            priority = int(raw_stage.get("priority", DEFAULT_STAGE_BANK.stage_priority(stage_name)))
        except Exception as exc:
            raise RuntimeError(f"stage priority must be an integer: {stage_name}") from exc
        if priority <= 0:
            raise RuntimeError(f"stage priority must be greater than zero: {stage_name}")
        raw_requires = raw_stage.get("requires", DEFAULT_STAGE_BANK.stage_requires(stage_name))
        if not isinstance(raw_requires, (list, tuple)):
            raise RuntimeError(f"stage requires must be an array: {stage_name}")
        requires = tuple(str(item or "").strip() for item in raw_requires if str(item or "").strip())
        unknown_requires = [item for item in requires if item not in KNOWN_STAGE_NAMES]
        if unknown_requires:
            raise RuntimeError(f"unknown dependency for {stage_name}: {', '.join(unknown_requires)}")
        stages[stage_name] = StageBankStage(cost=cost, priority=priority, requires=requires)

    for stage_name, default_stage in DEFAULT_STAGE_BANK.stages.items():
        stages.setdefault(stage_name, default_stage)
    return _validated_stage_bank_config(
        capacity=capacity,
        max_active_plans=max_active_plans,
        max_running_stages=max_running_stages,
        stages=stages,
    )


def effective_stage_dependencies(stage: str, stage_names: List[str], config: StageBankConfig) -> List[str]:
    available = set(stage_names)
    return [dependency for dependency in config.stage_requires(stage) if dependency in available]


def downstream_stage_names(stage: str, stage_names: List[str], config: StageBankConfig) -> List[str]:
    available = list(stage_names)
    downstream: List[str] = []
    queue = [stage]
    seen = {stage}
    while queue:
        dependency = queue.pop(0)
        for candidate in available:
            if candidate in seen:
                continue
            if dependency not in effective_stage_dependencies(candidate, available, config):
                continue
            seen.add(candidate)
            downstream.append(candidate)
            queue.append(candidate)
    return downstream


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


def normalize_mode(value: str) -> str:
    return "fastpass" if str(value or "").strip().lower() == "fastpass" else "full"


def av1an_encoder_name(value: str) -> str:
    return "x265" if str(value or "").strip().lower() in ("libx265", "x265") else "svt-av1"


def resolve_optional_path(raw_value: str, plan_path: Path) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (plan_path.parent / path).resolve()
    return str(path)


def bool_arg(value: bool) -> str:
    return "1" if bool(value) else "0"


def av1an_supports_progress_jsonl(av1an_exe: str) -> bool:
    key = str(av1an_exe or "").strip().lower()
    if key in _AV1AN_PROGRESS_JSONL_SUPPORT:
        return _AV1AN_PROGRESS_JSONL_SUPPORT[key]
    help_text = read_command_output([str(av1an_exe or "av1an"), "--help"], timeout=5.0)
    supported = "--progress-jsonl" in help_text
    _AV1AN_PROGRESS_JSONL_SUPPORT[key] = supported
    return supported


def item_short_snapshot(item: QueueItem) -> Dict[str, Any]:
    return {
        "mode": item.mode,
        "name": item.name,
        "plan": str(item.plan_path),
        "source": str(item.source),
        "source_size": item_source_size(item),
        "duration_seconds": item_source_duration(item),
        "workdir": str(item.workdir),
    }


def initial_stage_states(item: QueueItem) -> List[StageState]:
    states: List[StageState] = []
    for name in display_stage_plan(item):
        if not item_inputs_changed(item) and stage_resume_marker_exists(item, name):
            states.append(StageState(name=name, status="completed", message=CACHED_STAGE_MESSAGE))
        else:
            states.append(StageState(name=name))
    return states


def build_wrapper_vspipe_args(
    item: "QueueItem",
    *,
    user_vpy: str,
    pass_name: str,
) -> List[str]:
    plan = item.resolved.plan
    experimental = plan.video.experimental
    return [
        f"src={item.source}",
        f"vpy={user_vpy or ''}",
        f"rules={item.resolved.paths.crop_resize_file}",
        f"pass_name={pass_name}",
        f"source_loader={experimental.source_loader or 'ffms2'}",
        f"crop_enabled={bool_arg(experimental.crop_resize_enabled)}",
        f"plan_path={item.plan_path}",
        f"workdir={item.workdir}",
    ]


def build_queue(plan_args: List[str], cli_mode: str) -> List[QueueItem]:
    queue: List[QueueItem] = []
    seen: set[str] = set()
    visiting: set[str] = set()

    def visit(path: Path, inherited_mode: str) -> None:
        plan_path = path.expanduser().resolve()
        key = str(plan_path).lower()
        if key in visiting:
            raise RuntimeError(f"batch plan cycle detected at {plan_path}")
        plan = load_plan(plan_path)
        if isinstance(plan, FilePlan):
            if key in seen:
                return
            seen.add(key)
            mode = normalize_mode(cli_mode or inherited_mode or plan.meta.mode or "full")
            queue.append(QueueItem(resolved=resolve_file_plan(plan_path), mode=mode))
            return

        visiting.add(key)
        try:
            batch_mode = normalize_mode(cli_mode or plan.meta.mode or inherited_mode)
            for item in plan.items:
                nested = Path(item.plan).expanduser()
                if not nested.is_absolute():
                    nested = (plan_path.parent / nested).resolve()
                visit(nested, batch_mode)
        finally:
            visiting.remove(key)

    for raw in plan_args:
        visit(Path(raw), cli_mode or "")
    return queue


class SessionController:
    def __init__(
        self,
        *,
        items: List[QueueItem],
        events_jsonl: str,
        add_source_bitrate: bool,
        exit_when_idle: bool,
        session_id: str = "",
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex
        self.toolchain = load_toolchain()
        self.av1an_fork_enabled = is_mars_av1an_fork(self.toolchain.av1an_exe)
        self.av1an_progress_jsonl_enabled = av1an_supports_progress_jsonl(self.toolchain.av1an_exe)
        self.stage_bank = load_stage_bank_config()
        self.queue: Deque[QueueItem] = deque(items)
        self.failed: List[QueueItem] = []
        self.completed: List[QueueItem] = []
        self.current: Optional[QueueItem] = None
        self.current_stage = ""
        self.current_plan_run_id = ""
        self.active: Dict[str, ActivePlanState] = {}
        self.executions: Dict[str, PlanExecution] = {}
        self.running_stage_tasks: Dict[Tuple[str, str], RunningStageTask] = {}
        self.running_stage_processes: Dict[Tuple[str, str], Any] = {}
        self.finished_runs: List[FinishedPlanState] = []
        self.last_item: Optional[QueueItem] = None
        self.events_jsonl = Path(events_jsonl).expanduser().resolve() if events_jsonl else None
        self.add_source_bitrate = bool(add_source_bitrate)
        self.source_dirs = sorted({item.source_dir for item in items}, key=str.lower)
        self.pause_after_current_by_source: Dict[str, bool] = {source_dir: False for source_dir in self.source_dirs}
        self.paused_by_source: Dict[str, bool] = {source_dir: False for source_dir in self.source_dirs}
        self.exit_when_idle = bool(exit_when_idle)
        self.rerun_after_current = False
        self.rerun_after_current_plan_run_id = ""
        self.stop_requested = False
        self.worker_done = False
        self.lock = threading.Lock()
        self.event_io_lock = threading.RLock()
        self.wake_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_main, name="runner-worker", daemon=True)
        self.source_dir = self._resolve_source_dir(items)
        self.event_sinks: List[Any] = []

    @staticmethod
    def _resolve_source_dir(items: List[QueueItem]) -> str:
        dirs = sorted({str(item.source.parent.resolve()) for item in items}, key=str.lower)
        return dirs[0] if len(dirs) == 1 else ""

    def _source_dir_for_command(self, source_dir: str = "") -> str:
        value = str(source_dir or "").strip()
        if value:
            return value
        if self.source_dir:
            return self.source_dir
        if self.current is not None:
            return self.current.source_dir
        if self.active:
            return next(iter(self.active.values())).item.source_dir
        return self.source_dirs[0] if self.source_dirs else ""

    def _items_for_source(self, items: List[Any], source_dir: str) -> List[Any]:
        if not source_dir:
            return list(items)
        out: List[Any] = []
        for item in items:
            queue_item = getattr(item, "item", item)
            if getattr(queue_item, "source_dir", "") == source_dir:
                out.append(item)
        return out

    def _folder_paused(self, source_dir: str) -> bool:
        return bool(self.paused_by_source.get(source_dir, False))

    def _folder_pause_after_current(self, source_dir: str) -> bool:
        return bool(self.pause_after_current_by_source.get(source_dir, False))

    def start(self) -> None:
        self.worker.start()
        self.wake_event.set()

    def join(self) -> None:
        self.worker.join()

    def add_event_sink(self, sink: Any) -> None:
        self.event_sinks.append(sink)

    def is_idle(self) -> bool:
        with self.lock:
            return not self.active and not self.queue

    def is_busy(self) -> bool:
        with self.lock:
            return bool(self.active)

    def is_finished(self) -> bool:
        with self.lock:
            return self.worker_done

    def status_text(self) -> str:
        with self.lock:
            active = list(self.active.values())
            queued = len(self.queue)
            failed = len(self.failed)
            completed = len(self.completed)
            paused = ", ".join(sorted(k for k, v in self.paused_by_source.items() if v)) or "no"
            pause_after = ", ".join(sorted(k for k, v in self.pause_after_current_by_source.items() if v)) or "no"
            exit_idle = "yes" if self.exit_when_idle else "no"
        if active:
            current = "\n".join(
                f"- {state.item.name} [{self._current_stage_names(state) or 'pending'}]"
                for state in active
            )
        else:
            current = "-"
        return (
            f"active:\n{current}\n"
            f"queued: {queued}\n"
            f"completed: {completed}\n"
            f"failed: {failed}\n"
            f"paused: {paused}\n"
            f"pause_after_current: {pause_after}\n"
            f"exit_when_idle: {exit_idle}"
        )

    def snapshot(self, source_dir: str = "", session_id: str = "") -> Dict[str, Any]:
        with self.lock:
            selected_source = source_dir or self.source_dir
            active = self._items_for_source(list(self.active.values()), selected_source)
            queue = self._items_for_source(list(self.queue), selected_source)
            finished = self._items_for_source(list(self.finished_runs), selected_source)
            completed_runs = [run for run in finished if run.status in ("completed", "skipped")]
            failed_runs = [run for run in finished if run.status == "failed"]
            paused = self._folder_paused(selected_source) if selected_source else any(self.paused_by_source.values())
            pause_after_current = (
                self._folder_pause_after_current(selected_source)
                if selected_source
                else any(self.pause_after_current_by_source.values())
            )
            state = (
                "finished"
                if (self.worker_done or (bool(selected_source) and (completed_runs or failed_runs))) and not active and not queue
                else "paused"
                if paused
                else "running"
                if active
                else "idle"
            )
            return {
                "session_id": session_id or self.session_id,
                "runner_session_id": self.session_id,
                "source_dir": selected_source,
                "snapshot_at": time.time(),
                "state": state,
                "paused": paused,
                "pause_after_current": pause_after_current,
                "exit_when_idle": self.exit_when_idle,
                "stop_requested": self.stop_requested,
                "active": [state.snapshot() for state in active],
                "queue": [item_short_snapshot(item) for item in queue],
                "completed": [run.snapshot() for run in completed_runs],
                "failed": [run.snapshot() for run in failed_runs],
                "counts": {
                    "active": len(active),
                    "queued": len(queue),
                    "completed": len(completed_runs),
                    "failed": len(failed_runs),
                },
            }

    @staticmethod
    def _current_stage_name(state: ActivePlanState) -> str:
        running = [stage for stage in state.stages if stage.status == "started"]
        if running:
            return running[-1].name
        completed = [stage for stage in state.stages if stage.status == "completed"]
        if completed:
            return completed[-1].name
        return ""

    @staticmethod
    def _current_stage_names(state: ActivePlanState) -> str:
        running = [stage.name for stage in state.stages if stage.status == "started"]
        if running:
            return ", ".join(running)
        return SessionController._current_stage_name(state)

    def request_pause_after_current(self, source_dir: str = "") -> None:
        with self.lock:
            selected_source = self._source_dir_for_command(source_dir)
            has_active = any(state.item.source_dir == selected_source for state in self.active.values())
            if has_active:
                self.pause_after_current_by_source[selected_source] = True
            else:
                self.paused_by_source[selected_source] = True
                self.pause_after_current_by_source[selected_source] = False

    def resume(self, source_dir: str = "") -> None:
        with self.lock:
            selected_source = self._source_dir_for_command(source_dir)
            self.paused_by_source[selected_source] = False
            self.pause_after_current_by_source[selected_source] = False
        self.wake_event.set()

    def retry_failed(self) -> None:
        with self.lock:
            if not self.failed:
                return
            failed_items = list(self.failed)
            failed_keys = {(str(item.plan_path).lower(), str(item.source).lower()) for item in failed_items}
            self.finished_runs = [
                run
                for run in self.finished_runs
                if not (
                    run.status == "failed"
                    and (str(run.item.plan_path).lower(), str(run.item.source).lower()) in failed_keys
                )
            ]
            for item in failed_items:
                self.queue.append(item)
            self.failed.clear()
            for source_dir in self.paused_by_source:
                self.paused_by_source[source_dir] = False
        self.wake_event.set()

    def rerun_current_item(self) -> None:
        with self.lock:
            if self.current is not None:
                self.rerun_after_current = True
                self.rerun_after_current_plan_run_id = self.current_plan_run_id
            elif self.last_item is not None:
                self.queue.appendleft(self.last_item)
                self.paused_by_source[self.last_item.source_dir] = False
                self.wake_event.set()

    def request_exit_when_idle(self) -> None:
        with self.lock:
            self.exit_when_idle = True
        self.wake_event.set()

    def _terminate_running_processes_locked(self) -> None:
        for proc in list(self.running_stage_processes.values()):
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass

    def request_stop(self) -> None:
        with self.lock:
            self.stop_requested = True
            self.exit_when_idle = True
            for source_dir in self.paused_by_source:
                self.paused_by_source[source_dir] = False
                self.pause_after_current_by_source[source_dir] = False
            self.rerun_after_current = False
            self.rerun_after_current_plan_run_id = ""
            self.queue.clear()
            self._terminate_running_processes_locked()
        self.wake_event.set()

    def handle_command(self, command: str, source_dir: str = "") -> str:
        name = str(command or "").strip().lower().replace("-", "_").replace(" ", "_")
        if name in ("status", "snapshot"):
            return "status sent"
        if name in ("pause", "pause_after_current"):
            self.request_pause_after_current(source_dir)
            return "will pause after current active plan"
        if name == "resume":
            self.resume(source_dir)
            return "resumed"
        if name == "retry_failed":
            self.retry_failed()
            return "failed plans re-queued"
        if name in ("rerun_current", "rerun_current_item"):
            self.rerun_current_item()
            return "rerun requested"
        if name == "exit_when_idle":
            self.request_exit_when_idle()
            return "will exit when idle"
        if name in ("stop_after_current", "quit", "exit"):
            if self.is_busy():
                self.request_exit_when_idle()
                return "busy; will exit when idle"
            self.request_stop()
            return "stopped"
        raise ValueError(f"unknown command: {command}")

    def _write_item_state(self, item: QueueItem, event: RunnerEvent) -> None:
        meta_dir = item.workdir / "00_meta"
        ensure_dir(meta_dir)
        state = {
            "session_id": self.session_id,
            "plan_run_id": event.plan_run_id,
            "plan": str(item.plan_path),
            "source": str(item.source),
            "workdir": str(item.workdir),
            "mode": item.mode,
            "stage": event.stage,
            "status": event.status,
            "message": event.message,
            "timestamp": event.timestamp,
            "active": self.snapshot().get("active", []),
        }
        with self.event_io_lock:
            (meta_dir / "runner_state.json").write_text(
                json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )

    def _active_state_for_item(self, item: QueueItem) -> Optional[ActivePlanState]:
        for state in self.active.values():
            if state.item.plan_path == item.plan_path:
                return state
        return None

    def _reset_cached_following_stages(self, active: ActivePlanState, stage: str) -> None:
        stage_names = [state.name for state in active.stages]
        following = set(downstream_stage_names(stage, stage_names, self.stage_bank))
        for state in active.stages:
            if state.name not in following:
                continue
            if state.status in ("pending", "started"):
                continue
            state.status = "pending"
            state.started_at = 0.0
            state.ended_at = 0.0
            state.message = ""
            state.progress = None
            state.details.clear()

    def _update_active_event(
        self,
        item: QueueItem,
        stage: str,
        status: str,
        message: str,
        timestamp: float,
        progress: Optional[float] = None,
    ) -> tuple[str, float, float, float]:
        with self.lock:
            active = self._active_state_for_item(item)
            if active is None:
                return "", 0.0, 0.0, 0.0

            if stage == STAGE_ITEM:
                active.status = status
                active.message = message
                if status == "started" and not active.started_at:
                    active.started_at = timestamp
                if status in ("completed", "failed", "skipped"):
                    active.ended_at = timestamp
                started_at = active.started_at
                ended_at = active.ended_at
            else:
                stage_state = active.stage(stage)
                if (
                    stage in (STAGE_FASTPASS, STAGE_SSIMU2)
                    and status in ("started", "completed", "failed")
                ):
                    for parent_name in (STAGE_AUTOBOOST_SCENE, STAGE_AUTOBOOST_PSD_SCENE):
                        parent_state = next((item for item in active.stages if item.name == parent_name), None)
                        if parent_state is None or parent_state.status != "started":
                            continue
                        parent_cached = (
                            is_cached_stage_message(parent_state.message)
                            or is_cached_stage_message(message)
                        )
                        if not parent_cached:
                            parent_state.status = "completed"
                            parent_state.ended_at = timestamp
                        else:
                            parent_state.status = "completed"
                            parent_state.message = CACHED_STAGE_MESSAGE
                            parent_state.started_at = 0.0
                            parent_state.ended_at = 0.0
                if (
                    stage in (STAGE_AUTOBOOST_SCENE, STAGE_AUTOBOOST_PSD_SCENE)
                    and status == "completed"
                    and stage_state.status == "completed"
                    and any(
                        child.name in (STAGE_FASTPASS, STAGE_SSIMU2) and child.status != "pending"
                        for child in active.stages
                    )
                ):
                    started_at = stage_state.started_at
                    ended_at = stage_state.ended_at
                    elapsed = 0.0
                    if started_at:
                        elapsed = (ended_at or timestamp) - started_at
                    return active.plan_run_id, started_at, ended_at, elapsed
                previous_status = stage_state.status
                stage_state.status = status
                stage_state.message = message
                if status == "started":
                    if previous_status != "started" or not stage_state.started_at:
                        stage_state.started_at = timestamp
                        stage_state.ended_at = 0.0
                        stage_state.progress = None
                        stage_state.details.clear()
                    if not is_cached_stage_message(message):
                        self._reset_cached_following_stages(active, stage)
                if status in ("completed", "failed", "skipped"):
                    if is_cached_stage_message(message):
                        stage_state.started_at = 0.0
                        stage_state.ended_at = 0.0
                    else:
                        stage_state.ended_at = timestamp
                if (
                    stage in (STAGE_AUTOBOOST_SCENE, STAGE_AUTOBOOST_PSD_SCENE)
                    and stage_state.status == "started"
                    and any(
                        child.name in (STAGE_FASTPASS, STAGE_SSIMU2) and child.status == "started"
                        for child in active.stages
                    )
                ):
                    stage_state.status = "completed"
                    stage_state.ended_at = timestamp
                if progress is not None and progress >= 0:
                    stage_state.progress = progress
                started_at = stage_state.started_at
                ended_at = stage_state.ended_at

            elapsed = 0.0
            if started_at:
                elapsed = (ended_at or timestamp) - started_at
            return active.plan_run_id, started_at, ended_at, elapsed

    def _emit(self, item: QueueItem, stage: str, status: str, message: str = "") -> None:
        timestamp = time.time()
        plan_run_id, started_at, ended_at, elapsed = self._update_active_event(item, stage, status, message, timestamp)
        event = RunnerEvent(
            event="runner",
            plan=str(item.plan_path),
            mode=item.mode,
            stage=stage,
            status=status,
            message=message,
            timestamp=timestamp,
            session_id=self.session_id,
            plan_run_id=plan_run_id,
            source=str(item.source),
            workdir=str(item.workdir),
            started_at=started_at,
            ended_at=ended_at,
            elapsed_seconds=round(elapsed, 3),
        )
        text = f"[runner] {item.name} | {stage} | {status}"
        if message:
            text += f" | {message}"
        print(text, flush=True)

        meta_dir = item.workdir / "00_meta"
        ensure_dir(meta_dir)
        event_line = json.dumps(event.__dict__, ensure_ascii=False)
        with self.event_io_lock:
            with (meta_dir / "runner_events.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(event_line + "\n")
            if self.events_jsonl is not None:
                self.events_jsonl.parent.mkdir(parents=True, exist_ok=True)
                with self.events_jsonl.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(event_line + "\n")
            self._write_item_state(item, event)
        self._notify_event_sinks(dict(event.__dict__))

    def _notify_event_sinks(self, payload: Dict[str, Any]) -> None:
        if self.event_sinks:
            snapshot = self.snapshot()
            for sink in list(self.event_sinks):
                try:
                    sink(payload, snapshot)
                except Exception as exc:
                    print(f"[runner] event sink failed: {exc}", file=sys.stderr, flush=True)

    def _ingest_child_event(self, item: QueueItem, payload: Dict[str, Any]) -> None:
        plan_run_id = str(payload.get("plan_run_id") or "")
        if plan_run_id:
            with self.lock:
                active = self._active_state_for_item(item)
                if active is None or active.plan_run_id != plan_run_id:
                    return
        stage = str(payload.get("stage") or "").strip()
        status = str(payload.get("status") or "").strip()
        if not stage or not status:
            return
        timestamp = float(payload.get("timestamp") or time.time())
        message = str(payload.get("message") or "")
        raw_progress = payload.get("progress")
        progress = None
        try:
            progress = float(raw_progress)
        except Exception:
            progress = None
        plan_run_id, started_at, ended_at, elapsed = self._update_active_event(
            item,
            stage,
            status,
            message,
            timestamp,
            progress=progress,
        )
        event = RunnerEvent(
            event=str(payload.get("event") or "runner_child"),
            plan=str(item.plan_path),
            mode=item.mode,
            stage=stage,
            status=status,
            message=message,
            timestamp=timestamp,
            session_id=self.session_id,
            plan_run_id=plan_run_id,
            source=str(item.source),
            workdir=str(item.workdir),
            progress=progress if progress is not None else -1.0,
            started_at=started_at,
            ended_at=ended_at,
            elapsed_seconds=round(elapsed, 3),
        )
        print(f"[runner-child] {item.name} | {stage} | {status}", flush=True)
        self._write_item_state(item, event)
        self._notify_event_sinks(dict(event.__dict__))

    def _forward_child_events(self, path: Path, offset: int, item: QueueItem) -> int:
        if not path.exists():
            return offset
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(offset)
                for line in fh:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        payload = json.loads(text)
                    except Exception:
                        continue
                    if str(payload.get("event") or "") != "runner_child":
                        continue
                    self._ingest_child_event(item, payload)
                return fh.tell()
        except Exception as exc:
            print(f"[runner] failed to forward child events from {path}: {exc}", file=sys.stderr, flush=True)
            return offset

    @staticmethod
    def _parse_av1an_progress_text(text: str) -> Dict[str, Any]:
        clean = ANSI_RE.sub("", str(text or "")).replace("\r", "\n")
        result: Dict[str, Any] = {}
        matches = list(AV1AN_PROGRESS_RE.finditer(clean))
        if matches:
            match = matches[-1]
            try:
                result["progress"] = max(0.0, min(100.0, float(match.group("percent"))))
            except Exception:
                pass
            try:
                result["pos"] = int(match.group("pos"))
                result["total"] = int(match.group("total"))
            except Exception:
                pass
            try:
                if match.group("fps") is not None:
                    result["fps"] = float(match.group("fps"))
                elif match.group("spf") is not None:
                    result["spf"] = float(match.group("spf"))
            except Exception:
                pass
            try:
                if match.group("kbps") is not None:
                    result["kbps"] = float(match.group("kbps"))
            except Exception:
                pass
            eta = str(match.group("eta") or "").strip()
            if eta:
                result["eta"] = eta
            chunk_matches = list(AV1AN_CHUNK_RE.finditer(clean[: match.start()]))
            if chunk_matches:
                chunk_match = chunk_matches[-1]
                try:
                    result["chunks_done"] = int(chunk_match.group("done"))
                    result["chunks_total"] = int(chunk_match.group("total"))
                except Exception:
                    pass
            return result

        percent_matches = PERCENT_RE.findall(clean)
        if percent_matches:
            try:
                result["progress"] = max(0.0, min(100.0, float(percent_matches[-1])))
            except Exception:
                pass
        return result

    def _last_av1an_progress_from_log(self, path: Path) -> Dict[str, Any]:
        if not path.exists() or not path.is_file():
            return {}
        try:
            with path.open("rb") as fh:
                size = fh.seek(0, os.SEEK_END)
                fh.seek(max(0, size - 128 * 1024))
                text = fh.read().decode("utf-8", errors="replace")
        except Exception:
            return {}
        return self._parse_av1an_progress_text(text)

    def _last_av1an_progress_from_jsonl(self, path: Path) -> Dict[str, Any]:
        if not path.exists() or not path.is_file():
            return {}
        try:
            with path.open("rb") as fh:
                size = fh.seek(0, os.SEEK_END)
                fh.seek(max(0, size - 128 * 1024))
                text = fh.read().decode("utf-8", errors="replace")
        except Exception:
            return {}
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            result: Dict[str, Any] = {}
            for source_key, target_key in (
                ("progress", "progress"),
                ("percent", "progress"),
                ("fps", "fps"),
                ("spf", "spf"),
                ("kbps", "kbps"),
                ("pos", "pos"),
                ("frame", "pos"),
                ("total", "total"),
                ("total_frames", "total"),
                ("chunks_done", "chunks_done"),
                ("chunks_total", "chunks_total"),
                ("eta", "eta"),
            ):
                if source_key in payload and payload[source_key] not in (None, ""):
                    result[target_key] = payload[source_key]
            try:
                if "progress" in result:
                    result["progress"] = max(0.0, min(100.0, float(result["progress"])))
                if "fps" in result:
                    result["fps"] = float(result["fps"])
                if "spf" in result:
                    result["spf"] = float(result["spf"])
                if "kbps" in result:
                    result["kbps"] = float(result["kbps"])
            except Exception:
                pass
            return result
        return {}

    def _refresh_running_stage_progress(self, item: QueueItem) -> None:
        log_dir = item.workdir / "00_logs"
        progress_sources = {
            STAGE_FASTPASS: log_dir / "03.1_fastpass.log",
            STAGE_MAINPASS: log_dir / "06_av1an_mainpass.log",
        }
        progress_jsonl_sources = {
            STAGE_FASTPASS: log_dir / "03.1_fastpass.progress.jsonl",
            STAGE_MAINPASS: log_dir / "06_av1an_mainpass.progress.jsonl",
        }
        with self.lock:
            active = self._active_state_for_item(item)
            if active is None:
                return
            running = [stage.name for stage in active.stages if stage.status == "started"]
        updates: Dict[str, Dict[str, Any]] = {}
        for stage_name in running:
            log_path = progress_sources.get(stage_name)
            if log_path is None:
                continue
            progress = self._last_av1an_progress_from_jsonl(progress_jsonl_sources.get(stage_name, Path()))
            if not progress:
                progress = self._last_av1an_progress_from_log(log_path)
            if progress:
                updates[stage_name] = progress
        if not updates:
            return
        with self.lock:
            active = self._active_state_for_item(item)
            if active is None:
                return
            for stage in active.stages:
                if stage.name in updates and stage.status == "started":
                    stage_update = updates[stage.name]
                    if "progress" in stage_update:
                        stage.progress = stage_update["progress"]
                    stage.details.update({key: value for key, value in stage_update.items() if key != "progress"})

    def _stage_cached_message(self, item: QueueItem, stage: str) -> str:
        resume_info = stage_resume_info(item, stage)
        if item_inputs_changed(item):
            clear_stage_marker(item, stage)
            return ""
        elif resume_info.marker_exists and not resume_info.marker_valid:
            clear_stage_marker(item, stage)
            print(f"[runner] {item.name} | {stage} | stale marker ignored | {resume_info.reason}", flush=True)
            return CACHED_STAGE_MESSAGE if resume_info.completed else ""
        return CACHED_STAGE_MESSAGE if resume_info.completed else ""

    @staticmethod
    def _stage_child_events_path(item: QueueItem, plan_run_id: str, stage: str) -> Path:
        safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage).strip("_") or "stage"
        return item.workdir / "00_meta" / f"runner_child_{plan_run_id}_{safe_stage}.jsonl"

    def _run_stage(
        self,
        item: QueueItem,
        stage: str,
        cmd: List[str],
        *,
        child_events: Optional[Path] = None,
    ) -> None:
        stage_message = self._stage_cached_message(item, stage)
        cached_before = bool(stage_message)
        with self.lock:
            self.current_stage = stage
            active = self._active_state_for_item(item)
            plan_run_id = active.plan_run_id if active is not None else self.current_plan_run_id
        if cached_before:
            self._emit(item, stage, "completed", stage_message)
            return
        self._emit(item, stage, "started", stage_message)
        print("[cmd]", subprocess.list2cmdline(cmd), flush=True)
        child_events = child_events or self._stage_child_events_path(item, plan_run_id, stage)
        ensure_dir(child_events.parent)
        env = os.environ.copy()
        env[CHILD_EVENT_ENV] = str(child_events)
        env[SESSION_ID_ENV] = self.session_id
        env[PLAN_RUN_ID_ENV] = plan_run_id
        env[RUNNER_MANAGED_STATE_ENV] = "1"
        offset = child_events.stat().st_size if child_events.exists() else 0
        proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR), env=env)
        process_key = (plan_run_id, stage)
        with self.lock:
            self.running_stage_processes[process_key] = proc
            stop_already_requested = self.stop_requested
        if stop_already_requested and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        last_heartbeat = 0.0
        stop_terminate_at = time.time() if stop_already_requested else 0.0
        try:
            while proc.poll() is None:
                offset = self._forward_child_events(child_events, offset, item)
                now = time.time()
                with self.lock:
                    stop_now = self.stop_requested
                if stop_now:
                    if not stop_terminate_at:
                        stop_terminate_at = now
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                    elif now - stop_terminate_at >= 5.0:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                if now - last_heartbeat >= 5.0:
                    self._refresh_running_stage_progress(item)
                    self._notify_event_sinks(
                        {
                            "event": "runner_heartbeat",
                            "session_id": self.session_id,
                            "plan_run_id": plan_run_id,
                            "stage": stage,
                            "status": "started",
                            "timestamp": now,
                            "source": str(item.source),
                            "workdir": str(item.workdir),
                        }
                    )
                    last_heartbeat = now
                time.sleep(0.5)
        finally:
            with self.lock:
                if self.running_stage_processes.get(process_key) is proc:
                    self.running_stage_processes.pop(process_key, None)
        offset = self._forward_child_events(child_events, offset, item)
        rc = int(proc.returncode or 0)
        if stop_terminate_at:
            clear_stage_marker(item, stage)
            self._emit(item, stage, "failed", "stop_requested")
            raise RuntimeError(f"{stage}_stop_requested")
        if rc != 0:
            clear_stage_marker(item, stage)
            message = "stop_requested" if stop_terminate_at else f"exit_code={rc}"
            self._emit(item, stage, "failed", message)
            raise RuntimeError(f"{stage}_{message}")
        if not stage_completion_artifacts_valid(item, stage):
            clear_stage_marker(item, stage)
            reason = "stage_artifact_validation_failed"
            self._emit(item, stage, "failed", reason)
            raise RuntimeError(f"{stage}_{reason}")
        write_stage_marker(item, stage)
        self._emit(item, stage, "completed", stage_message)

    def _build_item_commands(self, item: QueueItem) -> List[tuple[str, List[str]]]:
        plan = item.resolved.plan
        paths = item.resolved.paths
        primary = plan.video.primary
        details = plan.video.details
        plan_path = str(paths.plan_path)
        workdir = paths.workdir
        log_dir = workdir / "00_logs"
        ensure_dir(log_dir)
        ensure_dir(workdir / "00_meta")
        ensure_dir(workdir / "audio")
        ensure_dir(workdir / "video")
        ensure_dir(workdir / "sub")
        ensure_dir(workdir / "attachments")
        ensure_dir(workdir / "chapters")
        ensure_dir(paths.zone_file.parent)
        ensure_zoned_command_file(workdir, paths.zone_file)

        commands: List[tuple[str, List[str]]] = [
            (
                STAGE_DEMUX,
                [
                    self.toolchain.python_exe,
                    str(UTILS_DIR / "demux.py"),
                    "--plan",
                    plan_path,
                    "--log",
                    str(log_dir / "01_demux.log"),
                ],
            ),
            (
                STAGE_ATTACHMENTS,
                [
                    self.toolchain.python_exe,
                    str(UTILS_DIR / "attachments-cleaner.py"),
                    "--subs",
                    str(workdir / "sub"),
                    "--attachments",
                    str(workdir / "attachments"),
                    "--log",
                    str(log_dir / "02_att_clean.log"),
                ],
            ),
        ]

        if item.mode == "fastpass" and not item.resolved.has_video_edit():
            return commands

        if item.resolved.has_video_edit():
            fast_vpy = resolve_optional_path(details.fast_vpy, paths.plan_path)
            main_vpy = resolve_optional_path(details.main_vpy, paths.plan_path)
            proxy_vpy = resolve_optional_path(details.proxy_vpy, paths.plan_path)
            experimental = plan.video.experimental
            use_wrapper = bool(experimental.vpy_wrapper or experimental.crop_resize_enabled)
            fastpass_input_vpy = str(WRAPPER_VPY) if use_wrapper else fast_vpy
            mainpass_input_vpy = str(WRAPPER_VPY) if use_wrapper else main_vpy
            fastpass_vspipe_args = (
                build_wrapper_vspipe_args(item, user_vpy=fast_vpy, pass_name="fast")
                if use_wrapper
                else ([f"src={paths.source}"] if fast_vpy or proxy_vpy else [])
            )
            mainpass_vspipe_args = (
                build_wrapper_vspipe_args(item, user_vpy=main_vpy, pass_name="main")
                if use_wrapper
                else ([f"src={paths.source}"] if main_vpy or proxy_vpy else [])
            )
            auto_boost_cmd = [
                self.toolchain.vs_python_exe,
                str(AUTOBOOST_DIR / "auto_boost.py"),
                "--av1an",
                self.toolchain.av1an_exe,
                "--input",
                str(paths.source),
                "--out-scenes",
                str(workdir / "video" / "scenes.json"),
                "--temp",
                str(workdir / "video"),
                "--log",
                str(log_dir / "03_autoboost.log"),
                "--sdm",
                str(primary.scene_detection or "av1an"),
                "--encoder",
                str(primary.encoder),
                "--workers",
                str(int(primary.fastpass_workers)),
                "--av1an-log-file",
                str(log_dir / "03.1_fastpass.log"),
                "--av1an-log-level",
                "info",
                "--quality",
                str(primary.quality),
                "-v",
                item.resolved.build_fastpass_params_text(),
                "--final-override",
                item.resolved.build_mainpass_params_text(),
                "--keep",
                "--verbose",
            ]
            if self.av1an_progress_jsonl_enabled:
                auto_boost_cmd.extend(["--av1an-progress-jsonl", str(log_dir / "03.1_fastpass.progress.jsonl")])
            if self.av1an_fork_enabled:
                auto_boost_cmd.extend(["--chunk-order", str(primary.chunk_order or "")])
                if str(primary.encoder_path or "").strip():
                    auto_boost_cmd.extend(["--encoder-path", str(primary.encoder_path)])
                if FAST_INTERRUPT:
                    auto_boost_cmd.append("--fast-interrupt")
            if str(primary.scene_detection or "").strip().lower() == "psd":
                auto_boost_cmd.extend(["--psd-script", self.toolchain.psd_script])
            if primary.no_fastpass:
                auto_boost_cmd.append("--no-fastpass")
            if primary.fastpass_hdr:
                auto_boost_cmd.append("--fastpass-hdr")
            if fastpass_input_vpy:
                auto_boost_cmd.extend(["--fastpass-vpy", fastpass_input_vpy])
            for arg in fastpass_vspipe_args:
                auto_boost_cmd.extend(["--fastpass-vspipe-arg", arg])
            if proxy_vpy:
                auto_boost_cmd.extend(["--fastpass-proxy", proxy_vpy])
            if primary.fastpass_preset:
                auto_boost_cmd.extend(["--fast-preset", str(primary.fastpass_preset)])
            if primary.preset:
                auto_boost_cmd.extend(["--preset", str(primary.preset)])
            if str(primary.ab_multiplier).strip():
                auto_boost_cmd.extend(["-a", str(primary.ab_multiplier)])
            elif str(primary.ab_pos_multiplier).strip() and str(primary.ab_neg_multiplier).strip():
                auto_boost_cmd.extend(["--pos-dev-multiplier", str(primary.ab_pos_multiplier)])
                auto_boost_cmd.extend(["--neg-dev-multiplier", str(primary.ab_neg_multiplier)])
            if str(primary.ab_pos_dev).strip():
                auto_boost_cmd.extend(["--max-positive-dev", str(primary.ab_pos_dev)])
            if str(primary.ab_neg_dev).strip():
                auto_boost_cmd.extend(["--max-negative-dev", str(primary.ab_neg_dev)])
            if str(primary.avg_func).strip():
                auto_boost_cmd.extend(["--avg-func", str(primary.avg_func).strip()])
            if details.fastpass_filter:
                auto_boost_cmd.extend(["-f", str(details.fastpass_filter)])

            def auto_boost_cmd_for(*run_stages: str) -> List[str]:
                cmd = list(auto_boost_cmd)
                if item.mode == "fastpass":
                    cmd.append("--stop-before-stage4")
                cmd.extend(["--run-stages", ",".join(run_stages)])
                return cmd

            scene_detection = str(primary.scene_detection or "").strip().lower()
            if primary.no_fastpass:
                if scene_detection == "psd":
                    commands.append((STAGE_AUTOBOOST_PSD_SCENE, auto_boost_cmd_for("psd", "base-scenes")))
                else:
                    commands.append((STAGE_AUTOBOOST_SCENE, auto_boost_cmd_for("fastpass", "base-scenes")))
            else:
                if scene_detection == "psd":
                    commands.append((STAGE_AUTOBOOST_PSD_SCENE, auto_boost_cmd_for("psd")))
                commands.append((STAGE_FASTPASS, auto_boost_cmd_for("fastpass")))
                commands.append((STAGE_SSIMU2, auto_boost_cmd_for("ssimu2", "base-scenes")))

            if item.mode == "full":
                commands.append(
                    (
                        STAGE_ZONE_EDIT,
                        [
                            self.toolchain.vs_python_exe,
                            str(UTILS_DIR / "zone-editor.py"),
                            "--source",
                            str(paths.source),
                            "--scenes",
                            str(workdir / "video" / "scenes.json"),
                            "--out",
                            str(workdir / "video" / "scenes-zoned.json"),
                            "--command",
                            str(paths.zone_file),
                            "--log",
                            str(log_dir / "04_zone_edit.log"),
                        ],
                    )
                )

                hdr_cmd = [
                    self.toolchain.vs_python_exe,
                    str(UTILS_DIR / "av1an_hdr_metadata_patch_v2.py"),
                    "--source",
                    str(paths.source),
                    "--scenes",
                    str(workdir / "video" / "scenes-zoned.json"),
                    "--output",
                    str(workdir / "video" / "scenes-final.json"),
                    "--workdir",
                    str(workdir / "video" / "hdr_tmp"),
                    "--encoder",
                    str(primary.encoder),
                    "--log",
                    str(log_dir / "05_hdr_patch.log"),
                ]
                if primary.strict_sdr_8bit:
                    hdr_cmd.append("--no-hdr10")
                if primary.no_hdr10plus or primary.strict_sdr_8bit:
                    hdr_cmd.append("--no-hdr10plus")
                if primary.no_dolby_vision or primary.strict_sdr_8bit:
                    hdr_cmd.append("--no-dv")
                commands.append((STAGE_HDR_PATCH, hdr_cmd))

                main_input = mainpass_input_vpy or str(paths.source)
                mainpass_cmd = [
                    self.toolchain.av1an_exe,
                    "-i",
                    main_input,
                    "-o",
                    str(workdir / "video" / "video-final.mkv"),
                    "--scenes",
                    str(workdir / "video" / "scenes-final.json"),
                    "--workers",
                    str(int(primary.mainpass_workers)),
                    "--temp",
                    str(workdir / "video" / "mainpass"),
                    "-n",
                    "--keep",
                    "--verbose",
                    "--resume",
                    "--cache-mode",
                    "temp",
                    "--log-file",
                    str(log_dir / "06_av1an_mainpass.log"),
                    "--log-level",
                    "info",
                    "--chunk-method",
                    "ffms2",
                    "-e",
                    av1an_encoder_name(primary.encoder),
                    "--pix-format",
                    "yuv420p" if primary.strict_sdr_8bit else "yuv420p10le",
                    "--no-defaults",
                    "-a=-an -sn",
                ]
                if self.av1an_progress_jsonl_enabled:
                    mainpass_cmd.extend(["--progress-jsonl", str(log_dir / "06_av1an_mainpass.progress.jsonl")])
                if self.av1an_fork_enabled:
                    if str(primary.chunk_order or "").strip():
                        mainpass_cmd.extend(["--chunk-order", str(primary.chunk_order)])
                    if str(primary.encoder_path or "").strip():
                        mainpass_cmd.extend(["--encoder-path", str(primary.encoder_path)])
                    if FAST_INTERRUPT:
                        mainpass_cmd.append("--fast-interrupt")
                if proxy_vpy:
                    mainpass_cmd.extend(["--proxy", proxy_vpy])
                if mainpass_vspipe_args:
                    mainpass_cmd.append("--vspipe-args")
                    mainpass_cmd.extend(mainpass_vspipe_args)
                if details.mainpass_filter:
                    mainpass_cmd.extend(["-f", f"-vf {details.mainpass_filter}"])
                commands.append((STAGE_MAINPASS, mainpass_cmd))

        if item.mode == "full":
            commands.append(
                (
                    STAGE_AUDIO,
                    [
                        self.toolchain.python_exe,
                        str(UTILS_DIR / "audio-tool-v2.py"),
                        "--plan",
                        plan_path,
                        "--copy-container",
                        "mka",
                        "--no-preserve-special",
                        "--log",
                        str(log_dir / "07_audio.log"),
                    ],
                )
            )
            commands.append(
                (
                    STAGE_VERIFY,
                    [
                        self.toolchain.python_exe,
                        str(UTILS_DIR / "verify.py"),
                        "--plan",
                        plan_path,
                        "--log",
                        str(log_dir / "08_verify.log"),
                    ],
                )
            )
            mux_cmd = [
                self.toolchain.python_exe,
                str(UTILS_DIR / "mux.py"),
                "--plan",
                plan_path,
                "--log",
                str(log_dir / "09_mux.log"),
            ]
            if not self.add_source_bitrate:
                mux_cmd.append("--no-source-bitrate")
            commands.append((STAGE_MUX, mux_cmd))
        return commands

    def _output_reusable(self, item: QueueItem) -> bool:
        output_path = output_path_for_item(item)
        if item.mode != "full" or item_inputs_changed(item):
            return False
        if safe_file_size(output_path) < MIN_REUSABLE_OUTPUT_BYTES:
            return False
        if not stage_completion_artifacts_valid(item, STAGE_MUX):
            return False
        return file_not_older_than(output_path, item.source) and file_not_older_than(output_path, item.plan_path)

    def _refresh_current_pointer_locked(self) -> None:
        if self.active:
            active = next(iter(self.active.values()))
            self.current = active.item
            self.current_plan_run_id = active.plan_run_id
            self.current_stage = self._current_stage_names(active)
            return
        self.current = None
        self.current_stage = ""
        self.current_plan_run_id = ""

    def _stage_status(self, active: ActivePlanState, stage: str) -> str:
        for state in active.stages:
            if state.name == stage:
                return str(state.status or "pending").lower()
        return "pending"

    def _stage_terminal(self, active: ActivePlanState, stage: str) -> bool:
        return self._stage_status(active, stage) in TERMINAL_STAGE_STATUSES

    def _stage_ready(self, execution: PlanExecution, stage: str) -> bool:
        active = self.active.get(execution.plan_run_id)
        if active is None or self._stage_status(active, stage) != "pending":
            return False
        for dependency in effective_stage_dependencies(stage, execution.stage_names, self.stage_bank):
            if self._stage_status(active, dependency) != "completed":
                return False
        return True

    def _running_stage_load(self) -> int:
        return sum(task.cost for task in self.running_stage_tasks.values() if not task.done.is_set())

    def _running_stage_count(self) -> int:
        return sum(1 for task in self.running_stage_tasks.values() if not task.done.is_set())

    def _available_stage_capacity(self) -> int:
        return max(0, self.stage_bank.capacity - self._running_stage_load())

    def _available_stage_slots(self) -> int:
        return max(0, self.stage_bank.max_running_stages - self._running_stage_count())

    def _available_active_plan_slots(self) -> int:
        return max(0, self.stage_bank.max_active_plans - len(self.active))

    @staticmethod
    def _workdir_key(item: QueueItem) -> str:
        return str(item.workdir.resolve()).casefold()

    def _workdir_in_use_locked(self, item: QueueItem) -> bool:
        candidate_key = self._workdir_key(item)
        return any(self._workdir_key(state.item) == candidate_key for state in self.active.values())

    def _has_running_stage_for_source(self, source_dir: str) -> bool:
        return any(
            task.item.source_dir == source_dir and not task.done.is_set()
            for task in self.running_stage_tasks.values()
        )

    def _can_launch_for_source(self, source_dir: str) -> bool:
        with self.lock:
            return (
                not self.stop_requested
                and not self.paused_by_source.get(source_dir, False)
                and not self.pause_after_current_by_source.get(source_dir, False)
            )

    def _mark_pause_after_current_sources(self) -> bool:
        paused_sources: List[str] = []
        with self.lock:
            for source_dir, requested in list(self.pause_after_current_by_source.items()):
                if not requested or self._has_running_stage_for_source(source_dir):
                    continue
                self.paused_by_source[source_dir] = True
                self.pause_after_current_by_source[source_dir] = False
                paused_sources.append(source_dir)
            if paused_sources:
                self._refresh_current_pointer_locked()
        for source_dir in paused_sources:
            item = self._representative_item_for_source(source_dir)
            self._notify_event_sinks(
                {
                    "event": "runner_pause",
                    "session_id": self.session_id,
                    "plan_run_id": "",
                    "stage": "",
                    "status": "paused",
                    "timestamp": time.time(),
                    "source": str(item.source) if item is not None else "",
                    "workdir": str(item.workdir) if item is not None else "",
                }
            )
            print(f"[runner] paused source: {source_dir}", flush=True)
        return bool(paused_sources)

    def _representative_item_for_source(self, source_dir: str) -> Optional[QueueItem]:
        with self.lock:
            for state in self.active.values():
                if state.item.source_dir == source_dir:
                    return state.item
            for item in self.queue:
                if item.source_dir == source_dir:
                    return item
        return None

    def _activate_item(self, item: QueueItem) -> bool:
        plan_run_id = f"{self.session_id}-{uuid.uuid4().hex[:12]}"
        active_state = ActivePlanState(
            plan_run_id=plan_run_id,
            item=item,
            status="queued",
            started_at=time.time(),
            stages=initial_stage_states(item),
        )
        with self.lock:
            self.active[plan_run_id] = active_state
            self._refresh_current_pointer_locked()

        output_path = output_path_for_item(item)
        if self._output_reusable(item):
            self._finish_plan_execution(
                plan_run_id,
                final_status="skipped",
                message=f"output_exists={output_path.name}",
            )
            return True
        if item.mode == "fastpass" and not item.resolved.has_video_edit():
            self._finish_plan_execution(
                plan_run_id,
                final_status="skipped",
                message="fastpass_mode_without_video_edit",
            )
            return True

        try:
            commands = self._build_item_commands(item)
        except Exception as exc:
            self._finish_plan_execution(
                plan_run_id,
                final_status="failed",
                failed_stage=STAGE_ITEM,
                message=str(exc),
            )
            return True
        execution = PlanExecution(plan_run_id=plan_run_id, item=item, commands=commands)
        with self.lock:
            self.executions[plan_run_id] = execution
            self._refresh_current_pointer_locked()
        self._emit(item, STAGE_ITEM, "started")
        return True

    def _activate_next_queued_item(self) -> bool:
        if self._available_stage_capacity() <= 0:
            return False
        if self._available_active_plan_slots() <= 0:
            return False
        with self.lock:
            if self.stop_requested:
                return False
            if self._available_active_plan_slots() <= 0:
                return False
            selected_index = -1
            for index, candidate in enumerate(list(self.queue)):
                if self.paused_by_source.get(candidate.source_dir, False):
                    continue
                if self.pause_after_current_by_source.get(candidate.source_dir, False):
                    continue
                if self._workdir_in_use_locked(candidate):
                    continue
                selected_index = index
                break
            if selected_index < 0:
                return False
            item = self.queue[selected_index]
            del self.queue[selected_index]
        return self._activate_item(item)

    def _mark_blocked_stages(self) -> bool:
        made_progress = False
        for execution in list(self.executions.values()):
            active = self.active.get(execution.plan_run_id)
            if active is None:
                continue
            for stage in execution.stage_names:
                if self._stage_status(active, stage) != "pending":
                    continue
                blocked_by = ""
                for dependency in effective_stage_dependencies(stage, execution.stage_names, self.stage_bank):
                    dependency_status = self._stage_status(active, dependency)
                    if dependency_status in ("failed", "skipped"):
                        blocked_by = dependency
                        break
                if not blocked_by:
                    continue
                message = f"blocked_by={blocked_by}"
                execution.blocked_stages[stage] = message
                self._emit(execution.item, stage, "skipped", message)
                made_progress = True
        return made_progress

    def _complete_cached_ready_stages(self) -> bool:
        made_progress = False
        for execution in list(self.executions.values()):
            active = self.active.get(execution.plan_run_id)
            if active is None:
                continue
            for stage in execution.stage_names:
                if not self._stage_ready(execution, stage):
                    continue
                stage_message = self._stage_cached_message(execution.item, stage)
                if not stage_message:
                    continue
                self._emit(execution.item, stage, "completed", stage_message)
                made_progress = True
        return made_progress

    def _ready_stage_candidates(self) -> List[Tuple[int, int, float, int, PlanExecution, str]]:
        candidates: List[Tuple[int, int, float, int, PlanExecution, str]] = []
        for execution in list(self.executions.values()):
            active = self.active.get(execution.plan_run_id)
            if active is None:
                continue
            if not self._can_launch_for_source(execution.item.source_dir):
                continue
            for stage_index, stage in enumerate(execution.stage_names):
                if not self._stage_ready(execution, stage):
                    continue
                cost = self.stage_bank.stage_cost(stage)
                priority = self.stage_bank.stage_priority(stage)
                candidates.append((priority, cost, active.started_at or time.time(), stage_index, execution, stage))
        candidates.sort(key=lambda item: (item[0], -item[1], item[2], item[3]))
        return candidates

    def _start_stage_task(self, execution: PlanExecution, stage: str) -> None:
        cmd = execution.command_for_stage(stage)
        cost = self.stage_bank.stage_cost(stage)
        task = RunningStageTask(
            plan_run_id=execution.plan_run_id,
            item=execution.item,
            stage=stage,
            cmd=cmd,
            cost=cost,
        )

        def run_task() -> None:
            try:
                self._run_stage(
                    execution.item,
                    stage,
                    cmd,
                    child_events=self._stage_child_events_path(execution.item, execution.plan_run_id, stage),
                )
            except BaseException as exc:
                task.error = exc
            finally:
                task.done.set()
                self.wake_event.set()

        task.thread = threading.Thread(
            target=run_task,
            name=f"runner-stage-{stage}-{execution.plan_run_id[-6:]}",
            daemon=True,
        )
        self.running_stage_tasks[(execution.plan_run_id, stage)] = task
        task.thread.start()

    def _launch_ready_stages(self) -> bool:
        if self._available_stage_capacity() <= 0:
            return False
        if self._available_stage_slots() <= 0:
            return False
        made_progress = False
        while True:
            available = self._available_stage_capacity()
            if available <= 0:
                return made_progress
            if self._available_stage_slots() <= 0:
                return made_progress
            launched = False
            for _priority, cost, _started_at, _stage_index, execution, stage in self._ready_stage_candidates():
                if cost > available:
                    continue
                if (execution.plan_run_id, stage) in self.running_stage_tasks:
                    continue
                self._start_stage_task(execution, stage)
                made_progress = True
                launched = True
                break
            if not launched:
                return made_progress

    def _collect_finished_stage_tasks(self) -> bool:
        made_progress = False
        for key, task in list(self.running_stage_tasks.items()):
            if not task.done.is_set():
                continue
            self.running_stage_tasks.pop(key, None)
            if task.thread is not None:
                task.thread.join(timeout=0.1)
            execution = self.executions.get(task.plan_run_id)
            if execution is not None and task.error is not None:
                execution.failed_stages[task.stage] = str(task.error)
                active = self.active.get(task.plan_run_id)
                if active is not None and self._stage_status(active, task.stage) != "failed":
                    self._emit(task.item, task.stage, "failed", str(task.error))
            made_progress = True
        return made_progress

    def _has_unfinished_stage_task(self, plan_run_id: str) -> bool:
        return any(
            task.plan_run_id == plan_run_id and not task.done.is_set()
            for task in self.running_stage_tasks.values()
        )

    def _cancel_stopped_pending_executions(self) -> bool:
        with self.lock:
            if not self.stop_requested:
                return False
            cancellations = [
                execution
                for execution in list(self.executions.values())
                if not execution.failed_stages and not self._has_unfinished_stage_task(execution.plan_run_id)
            ]
        made_progress = False
        for execution in cancellations:
            active = self.active.get(execution.plan_run_id)
            if active is None:
                continue
            if all(self._stage_terminal(active, stage) for stage in execution.stage_names):
                continue
            self._finish_plan_execution(
                execution.plan_run_id,
                final_status="failed",
                failed_stage=STAGE_ITEM,
                message="stop_requested",
            )
            made_progress = True
        return made_progress

    def _execution_is_terminal(self, execution: PlanExecution) -> bool:
        if any(task.plan_run_id == execution.plan_run_id for task in self.running_stage_tasks.values()):
            return False
        active = self.active.get(execution.plan_run_id)
        if active is None:
            return True
        return all(self._stage_terminal(active, stage) for stage in execution.stage_names)

    def _finish_terminal_executions(self) -> bool:
        made_progress = False
        for execution in list(self.executions.values()):
            if not self._execution_is_terminal(execution):
                continue
            if execution.failed_stages:
                failed_stage, message = next(iter(execution.failed_stages.items()))
                self._finish_plan_execution(
                    execution.plan_run_id,
                    final_status="failed",
                    failed_stage=failed_stage,
                    message=message,
                )
            else:
                self._finish_plan_execution(execution.plan_run_id, final_status="completed")
            made_progress = True
        return made_progress

    def _finish_plan_execution(
        self,
        plan_run_id: str,
        *,
        final_status: str,
        failed_stage: str = "",
        message: str = "",
    ) -> None:
        active_state = self.active.get(plan_run_id)
        item = active_state.item if active_state is not None else None
        if item is None:
            execution = self.executions.get(plan_run_id)
            item = execution.item if execution is not None else None
        if item is None:
            return

        ended_at = time.time()
        event_plan_run_id, event_started_at, event_ended_at, event_elapsed = self._update_active_event(
            item,
            STAGE_ITEM,
            final_status,
            message,
            ended_at,
        )
        event_plan_run_id = event_plan_run_id or plan_run_id
        mark_clean = final_status in ("completed", "skipped")
        with self.lock:
            active_state = self.active.pop(plan_run_id, None)
            self.executions.pop(plan_run_id, None)
            started_at = active_state.started_at if active_state is not None else ended_at
            if final_status == "failed":
                self.failed = [
                    failed_item
                    for failed_item in self.failed
                    if (str(failed_item.plan_path).lower(), str(failed_item.source).lower())
                    != (str(item.plan_path).lower(), str(item.source).lower())
                ]
                self.failed.append(item)
            else:
                self.failed = [
                    failed_item
                    for failed_item in self.failed
                    if (str(failed_item.plan_path).lower(), str(failed_item.source).lower())
                    != (str(item.plan_path).lower(), str(item.source).lower())
                ]
                self.completed.append(item)
            self.finished_runs.append(
                FinishedPlanState(
                    plan_run_id=plan_run_id,
                    item=item,
                    status=final_status,
                    started_at=started_at,
                    ended_at=ended_at,
                    stage=failed_stage,
                    message=message,
                )
            )
            self.last_item = item
            if self.rerun_after_current_plan_run_id == plan_run_id or (
                self.rerun_after_current and not self.rerun_after_current_plan_run_id
            ):
                self.queue.appendleft(item)
                self.rerun_after_current = False
                self.rerun_after_current_plan_run_id = ""
            self._refresh_current_pointer_locked()

        if not event_started_at:
            event_started_at = started_at
        if not event_ended_at:
            event_ended_at = ended_at
        if event_started_at:
            event_elapsed = max(0.0, event_ended_at - event_started_at)
        event = RunnerEvent(
            event="runner",
            plan=str(item.plan_path),
            mode=item.mode,
            stage=STAGE_ITEM,
            status=final_status,
            message=message,
            timestamp=ended_at,
            session_id=self.session_id,
            plan_run_id=event_plan_run_id,
            source=str(item.source),
            workdir=str(item.workdir),
            started_at=event_started_at,
            ended_at=event_ended_at,
            elapsed_seconds=round(event_elapsed, 3),
        )
        text = f"[runner] {item.name} | {STAGE_ITEM} | {final_status}"
        if message:
            text += f" | {message}"
        print(text, flush=True)

        meta_dir = item.workdir / "00_meta"
        ensure_dir(meta_dir)
        event_line = json.dumps(event.__dict__, ensure_ascii=False)
        with self.event_io_lock:
            with (meta_dir / "runner_events.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(event_line + "\n")
            if self.events_jsonl is not None:
                self.events_jsonl.parent.mkdir(parents=True, exist_ok=True)
                with self.events_jsonl.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(event_line + "\n")
            self._write_item_state(item, event)
        self._notify_event_sinks(dict(event.__dict__))
        if mark_clean:
            mark_source_info_clean(item)

    def _worker_main(self) -> None:
        while True:
            made_progress = False
            made_progress = self._collect_finished_stage_tasks() or made_progress
            made_progress = self._cancel_stopped_pending_executions() or made_progress
            made_progress = self._mark_pause_after_current_sources() or made_progress
            while True:
                blocked = self._mark_blocked_stages()
                cached = self._complete_cached_ready_stages()
                if not blocked and not cached:
                    break
                made_progress = True
            made_progress = self._finish_terminal_executions() or made_progress
            made_progress = self._launch_ready_stages() or made_progress
            if self._available_stage_capacity() > 0:
                made_progress = self._activate_next_queued_item() or made_progress

            with self.lock:
                no_active_work = not self.active and not self.executions and not self.running_stage_tasks
                if self.stop_requested and no_active_work and not self.queue:
                    self.worker_done = True
                    return
                if no_active_work and not self.queue:
                    if self.exit_when_idle:
                        self.worker_done = True
                        return
                should_wait = not made_progress
            if should_wait:
                self.wake_event.wait(0.25)
                self.wake_event.clear()


def print_help() -> None:
    print(
        "Commands:\n"
        "  status\n"
        "  pause after current\n"
        "  resume\n"
        "  retry failed\n"
        "  rerun current item\n"
        "  exit when idle\n"
        "  quit",
        flush=True,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Session runner for file and batch .plan files.")
    parser.add_argument("--mode", choices=["full", "fastpass"], default="")
    parser.add_argument("--events-jsonl", default="")
    parser.add_argument("--add-source-bitrate", action="store_true", help="Include source bitrate metadata in mux output.")
    parser.add_argument(
        "--no-source-bitrate",
        dest="add_source_bitrate",
        action="store_false",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-interactive", action="store_true")
    parser.add_argument("--exit-when-idle", action="store_true")
    parser.add_argument("--session-id", default="")
    discord_group = parser.add_mutually_exclusive_group()
    discord_group.add_argument(
        "--discord",
        dest="discord_verbose",
        action="store_true",
        default=False,
        help="Enable Discord integration and print Discord connection errors.",
    )
    discord_group.add_argument(
        "--no-discord",
        dest="discord_enabled",
        action="store_false",
        default=True,
        help="Disable registration in the local Discord bot service.",
    )
    parser.add_argument(
        "--discord-service-url",
        default=discord_config_value("PBBATCH_DISCORD_SERVICE_URL", "http://127.0.0.1:8794"),
    )
    parser.add_argument(
        "--discord-shared-secret",
        default=discord_config_value("PBBATCH_DISCORD_SHARED_SECRET", ""),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("plans", nargs="+")
    args = parser.parse_args(argv)

    queue = build_queue(args.plans, args.mode)
    if not queue:
        print("[runner] no plans resolved", file=sys.stderr)
        return 1
    source_dirs = sorted({str(item.source.parent.resolve()) for item in queue}, key=str.lower)
    if args.discord_enabled and len(source_dirs) > 1:
        print(f"[discord] multi-folder runner: {len(source_dirs)} folder sessions will be published", flush=True)

    print(f"[runner] queue size: {len(queue)}", flush=True)
    for index, item in enumerate(queue, start=1):
        print(f"  {index}. {item.mode} | {item.plan_path}", flush=True)

    controller = SessionController(
        items=queue,
        events_jsonl=args.events_jsonl,
        add_source_bitrate=args.add_source_bitrate,
        exit_when_idle=(args.exit_when_idle or args.no_interactive),
        session_id=args.session_id,
    )
    def discord_session_id_for_source(source_dir: str) -> str:
        if len(source_dirs) <= 1:
            return controller.session_id
        suffix = hashlib.sha1(source_dir.lower().encode("utf-8", errors="ignore")).hexdigest()[:8]
        return f"{controller.session_id}-{suffix}"

    bridges: List[tuple[str, DiscordBridge]] = []
    if args.discord_enabled:
        for source_dir in source_dirs:
            discord_session_id = discord_session_id_for_source(source_dir)
            bridge = DiscordBridge(
                service_url=args.discord_service_url,
                session_id=discord_session_id,
                enabled=True,
                shared_secret=args.discord_shared_secret,
            )
            bridge.attach(
                snapshot_provider=lambda sd=source_dir, sid=discord_session_id: controller.snapshot(sd, session_id=sid),
                command_handler=lambda command, sd=source_dir: controller.handle_command(command, source_dir=sd),
            )
            if args.discord_verbose:
                bridge.set_error_callback(
                    lambda message, sd=source_dir: print(f"[discord] bridge unavailable for {sd}: {message}", flush=True)
                )
            bridges.append((source_dir, bridge))

        bridge_by_source = {source_dir: bridge for source_dir, bridge in bridges}

        def notify_discord_event(payload: Dict[str, Any], _snapshot: Dict[str, Any]) -> None:
            source = str(payload.get("source") or "")
            source_dir = str(Path(source).parent.resolve()) if source else ""
            bridge = bridge_by_source.get(source_dir)
            if bridge is None:
                return
            bridge.notify_event(payload, controller.snapshot(source_dir, session_id=bridge.session_id))

        controller.add_event_sink(notify_discord_event)
    folder_locks = [SourceDirLock(source_dir=source_dir, session_id=controller.session_id, enabled=True) for source_dir in source_dirs]
    if args.discord_enabled:
        print(f"[discord] enabled: session_id={controller.session_id}", flush=True)
        print(f"[discord] service_url={args.discord_service_url}", flush=True)
        for source_dir, bridge in bridges:
            print(f"[discord] source_dir={source_dir} | discord_session_id={bridge.session_id}", flush=True)
    else:
        print("[discord] disabled", flush=True)
    acquired_locks: List[SourceDirLock] = []
    try:
        for folder_lock in folder_locks:
            folder_lock.acquire()
            acquired_locks.append(folder_lock)
    except RuntimeError as exc:
        for folder_lock in acquired_locks:
            folder_lock.release()
        print(f"[runner] {exc}", file=sys.stderr)
        return 2
    for folder_lock in folder_locks:
        print(f"[runner] folder lock acquired: {folder_lock.path}", flush=True)
    try:
        for item in queue:
            prepare_source_info(item)
    except Exception as exc:
        for folder_lock in folder_locks:
            folder_lock.release()
        print(f"[runner] source info preparation failed: {exc}", file=sys.stderr)
        return 2
    controller.start()
    for _, bridge in bridges:
        bridge.start()
    if args.discord_enabled:
        connected_count = sum(1 for _, bridge in bridges if bridge.connected)
        if connected_count == len(bridges):
            print("[discord] runner registered in bot service", flush=True)
        elif args.discord_verbose:
            print(f"[discord] registered {connected_count}/{len(bridges)} folder sessions; runner will keep working locally", flush=True)

    if args.no_interactive:
        try:
            controller.join()
            return 1 if controller.failed else 0
        finally:
            for _, bridge in bridges:
                bridge.stop()
            for folder_lock in folder_locks:
                folder_lock.release()
            if args.discord_enabled:
                print("[discord] bridge stopped", flush=True)
            print("[runner] folder lock released", flush=True)

    def request_interrupt_shutdown() -> None:
        if controller.is_busy():
            controller.request_stop()
            print("[runner] interrupt received; stopping active work", flush=True)
        else:
            controller.request_stop()
            print("[runner] interrupt received; stopping", flush=True)

    print_help()
    try:
        try:
            while not controller.is_finished():
                try:
                    raw = input("runner> ").strip().lower()
                except EOFError:
                    controller.request_exit_when_idle()
                    break
                if raw in ("", "status"):
                    print(controller.status_text(), flush=True)
                    continue
                if raw == "pause after current":
                    controller.request_pause_after_current()
                    print("[runner] will pause after current item", flush=True)
                    continue
                if raw == "resume":
                    controller.resume()
                    print("[runner] resumed", flush=True)
                    continue
                if raw == "retry failed":
                    controller.retry_failed()
                    print("[runner] failed items re-queued", flush=True)
                    continue
                if raw == "rerun current item":
                    controller.rerun_current_item()
                    print("[runner] rerun requested", flush=True)
                    continue
                if raw == "exit when idle":
                    controller.request_exit_when_idle()
                    print("[runner] will exit when idle", flush=True)
                    continue
                if raw in ("quit", "exit"):
                    if controller.is_busy():
                        controller.request_exit_when_idle()
                        print("[runner] busy; will exit when idle", flush=True)
                    else:
                        controller.request_stop()
                    continue
                if raw == "help":
                    print_help()
                    continue
                print("[runner] unknown command", flush=True)
        except KeyboardInterrupt:
            request_interrupt_shutdown()
        try:
            controller.join()
        except KeyboardInterrupt:
            request_interrupt_shutdown()
            controller.join()
        return 1 if controller.failed else 0
    finally:
        for _, bridge in bridges:
            bridge.stop()
        for folder_lock in folder_locks:
            folder_lock.release()
        if args.discord_enabled:
            print("[discord] bridge stopped", flush=True)
        print("[runner] folder lock released", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
