from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

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
from utils.plan_model import BatchPlan, FilePlan, ResolvedFilePlan, RunnerEvent, load_plan, resolve_file_plan
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
    autoboost_fastpass_output,
    autoboost_scene_stage,
    clear_stage_marker,
    display_stage_plan,
    is_cached_stage_message,
    stage_completion_artifacts_valid,
    stage_resume_info,
    stage_resume_marker_exists,
    write_stage_marker,
)
from utils.zoned_commands import ensure_zoned_command_file

FAST_INTERRUPT = False
WRAPPER_VPY = ROOT_DIR / "wrapper.vpy"

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
_DURATION_CACHE: Dict[str, float] = {}
_AV1AN_PROGRESS_JSONL_SUPPORT: Dict[str, bool] = {}

CHILD_EVENT_ENV = "PBBATCH_RUNNER_CHILD_EVENTS_JSONL"
SESSION_ID_ENV = "PBBATCH_RUNNER_SESSION_ID"
PLAN_RUN_ID_ENV = "PBBATCH_RUNNER_PLAN_RUN_ID"


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
        source_size = self.item.source.stat().st_size if self.item.source.exists() else 0
        output_path = self.item.source.parent / f"{self.item.source.stem}-av1.mkv"
        output_size = output_path.stat().st_size if output_path.exists() else 0
        fastpass_output_path = fastpass_output_path_for_item(self.item)
        fastpass_output_size = fastpass_output_path.stat().st_size if fastpass_output_path.exists() else 0
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
            "duration_seconds": probe_source_duration(self.item.source),
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
        output_path = self.item.source.parent / f"{self.item.source.stem}-av1.mkv"
        output_size = output_path.stat().st_size if output_path.exists() else 0
        fastpass_output_path = fastpass_output_path_for_item(self.item)
        fastpass_output_size = fastpass_output_path.stat().st_size if fastpass_output_path.exists() else 0
        source_size = self.item.source.stat().st_size if self.item.source.exists() else 0
        return {
            "plan_run_id": self.plan_run_id,
            "status": self.status,
            "mode": self.item.mode,
            "name": self.item.name,
            "plan": str(self.item.plan_path),
            "source": str(self.item.source),
            "source_size": source_size,
            "duration_seconds": probe_source_duration(self.item.source),
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


class SourceDirLock:
    def __init__(self, *, source_dir: str, session_id: str, enabled: bool) -> None:
        self.enabled = bool(enabled and source_dir)
        self.session_id = session_id
        self.path = Path(source_dir) / ".pbbatch_runner.lock" if source_dir else Path()
        self.token = uuid.uuid4().hex

    def acquire(self) -> None:
        if not self.enabled:
            return
        payload = {
            "session_id": self.session_id,
            "pid": os.getpid(),
            "token": self.token,
            "started_at": time.time(),
        }
        while True:
            try:
                with self.path.open("x", encoding="utf-8", newline="\n") as fh:
                    fh.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
                return
            except FileExistsError:
                existing = self._read_existing()
                pid = int(existing.get("pid") or 0)
                if pid and self._pid_alive(pid):
                    other_session = str(existing.get("session_id") or "")
                    raise RuntimeError(f"source folder is already locked by active runner session {other_session or pid}")
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    continue

    def release(self) -> None:
        if not self.enabled:
            return
        existing = self._read_existing()
        if str(existing.get("token") or "") != self.token:
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            return

    def _read_existing(self) -> Dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


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


def fastpass_output_path_for_item(item: QueueItem) -> Path:
    return autoboost_fastpass_output(item)


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


def probe_source_duration(source: Path) -> float:
    key = str(source.resolve()).lower()
    if key in _DURATION_CACHE:
        return _DURATION_CACHE[key]
    duration = 0.0
    ffprobe = shutil.which("ffprobe")
    if ffprobe and source.exists():
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source),
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
            if proc.returncode == 0:
                duration = max(0.0, float(str(proc.stdout or "0").strip() or "0"))
        except Exception:
            duration = 0.0
    _DURATION_CACHE[key] = duration
    return duration


def av1an_supports_progress_jsonl(av1an_exe: str) -> bool:
    key = str(av1an_exe or "").strip().lower()
    if key in _AV1AN_PROGRESS_JSONL_SUPPORT:
        return _AV1AN_PROGRESS_JSONL_SUPPORT[key]
    help_text = read_command_output([str(av1an_exe or "av1an"), "--help"], timeout=5.0)
    supported = "--progress-jsonl" in help_text
    _AV1AN_PROGRESS_JSONL_SUPPORT[key] = supported
    return supported


def item_short_snapshot(item: QueueItem) -> Dict[str, Any]:
    source_size = item.source.stat().st_size if item.source.exists() else 0
    return {
        "mode": item.mode,
        "name": item.name,
        "plan": str(item.plan_path),
        "source": str(item.source),
        "source_size": source_size,
        "duration_seconds": probe_source_duration(item.source),
        "workdir": str(item.workdir),
    }


def initial_stage_states(item: QueueItem) -> List[StageState]:
    states: List[StageState] = []
    for name in display_stage_plan(item):
        if stage_resume_marker_exists(item, name):
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
        self.queue: Deque[QueueItem] = deque(items)
        self.failed: List[QueueItem] = []
        self.completed: List[QueueItem] = []
        self.current: Optional[QueueItem] = None
        self.current_stage = ""
        self.current_plan_run_id = ""
        self.active: Dict[str, ActivePlanState] = {}
        self.finished_runs: List[FinishedPlanState] = []
        self.last_item: Optional[QueueItem] = None
        self.events_jsonl = Path(events_jsonl).expanduser().resolve() if events_jsonl else None
        self.add_source_bitrate = bool(add_source_bitrate)
        self.source_dirs = sorted({item.source_dir for item in items}, key=str.lower)
        self.pause_after_current_by_source: Dict[str, bool] = {source_dir: False for source_dir in self.source_dirs}
        self.paused_by_source: Dict[str, bool] = {source_dir: False for source_dir in self.source_dirs}
        self.exit_when_idle = bool(exit_when_idle)
        self.rerun_after_current = False
        self.stop_requested = False
        self.worker_done = False
        self.lock = threading.Lock()
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
                f"- {state.item.name} [{self._current_stage_name(state) or 'pending'}]"
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

    def _pause_between_stages_if_requested(self, item: QueueItem, stage: str) -> None:
        source_dir = item.source_dir
        paused_now = False
        with self.lock:
            if self.pause_after_current_by_source.get(source_dir, False):
                self.paused_by_source[source_dir] = True
                self.pause_after_current_by_source[source_dir] = False
                paused_now = True
        if paused_now:
            print(f"[runner] {item.name} | paused after stage {stage}", flush=True)
            self._notify_event_sinks(
                {
                    "event": "runner_pause",
                    "session_id": self.session_id,
                    "plan_run_id": self.current_plan_run_id,
                    "stage": stage,
                    "status": "paused",
                    "timestamp": time.time(),
                    "source": str(item.source),
                    "workdir": str(item.workdir),
                }
            )
        while True:
            with self.lock:
                if self.stop_requested:
                    return
                if not self.paused_by_source.get(source_dir, False):
                    return
            self.wake_event.wait(0.25)
            self.wake_event.clear()

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
            elif self.last_item is not None:
                self.queue.appendleft(self.last_item)
                self.paused_by_source[self.last_item.source_dir] = False
                self.wake_event.set()

    def request_exit_when_idle(self) -> None:
        with self.lock:
            self.exit_when_idle = True
        self.wake_event.set()

    def request_stop(self) -> None:
        with self.lock:
            self.stop_requested = True
            self.exit_when_idle = True
            for source_dir in self.paused_by_source:
                self.paused_by_source[source_dir] = False
                self.pause_after_current_by_source[source_dir] = False
            self.rerun_after_current = False
            self.queue.clear()
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

    @staticmethod
    def _reset_cached_following_stages(active: ActivePlanState, stage: str) -> None:
        try:
            stage_names = display_stage_plan(active.item)
            index = stage_names.index(stage)
        except ValueError:
            return
        following = set(stage_names[index + 1 :])
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
                    and status in ("completed", "failed")
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
                except Exception:
                    pass

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
        except Exception:
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

    def _run_stage(self, item: QueueItem, stage: str, cmd: List[str]) -> None:
        resume_info = stage_resume_info(item, stage)
        if resume_info.marker_exists and not resume_info.marker_valid:
            clear_stage_marker(item, stage)
            print(f"[runner] {item.name} | {stage} | stale marker ignored | {resume_info.reason}", flush=True)
        cached_before = resume_info.completed
        stage_message = CACHED_STAGE_MESSAGE if cached_before else ""
        with self.lock:
            self.current_stage = stage
            active = self._active_state_for_item(item)
            plan_run_id = active.plan_run_id if active is not None else self.current_plan_run_id
        if cached_before:
            self._emit(item, stage, "completed", stage_message)
            return
        self._emit(item, stage, "started", stage_message)
        print("[cmd]", subprocess.list2cmdline(cmd), flush=True)
        child_events = self.events_jsonl or (item.workdir / "00_meta" / "runner_events.jsonl")
        ensure_dir(child_events.parent)
        env = os.environ.copy()
        env[CHILD_EVENT_ENV] = str(child_events)
        env[SESSION_ID_ENV] = self.session_id
        env[PLAN_RUN_ID_ENV] = plan_run_id
        env[RUNNER_MANAGED_STATE_ENV] = "1"
        offset = child_events.stat().st_size if child_events.exists() else 0
        proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR), env=env)
        last_heartbeat = 0.0
        while proc.poll() is None:
            offset = self._forward_child_events(child_events, offset, item)
            now = time.time()
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
        offset = self._forward_child_events(child_events, offset, item)
        rc = int(proc.returncode or 0)
        if rc != 0:
            clear_stage_marker(item, stage)
            self._emit(item, stage, "failed", f"exit_code={rc}")
            raise RuntimeError(f"{stage}_failed_rc_{rc}")
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

    def _process_item(self, item: QueueItem) -> None:
        output_path = item.source.parent / f"{item.source.stem}-av1.mkv"
        if item.mode == "full" and output_path.exists():
            self._emit(item, STAGE_ITEM, "skipped", f"output_exists={output_path.name}")
            return
        if item.mode == "fastpass" and not item.resolved.has_video_edit():
            self._emit(item, STAGE_ITEM, "skipped", "fastpass_mode_without_video_edit")
            return

        self._emit(item, STAGE_ITEM, "started")
        for stage, cmd in self._build_item_commands(item):
            self._run_stage(item, stage, cmd)
            self._pause_between_stages_if_requested(item, stage)
        self._emit(item, STAGE_ITEM, "completed")

    def _worker_main(self) -> None:
        while True:
            with self.lock:
                if self.stop_requested and self.current is None and not self.queue:
                    self.worker_done = True
                    return
                if self.current is None and not self.queue:
                    if self.exit_when_idle:
                        self.worker_done = True
                        return
                    should_wait = True
                else:
                    item = None
                    for index, candidate in enumerate(list(self.queue)):
                        if self.paused_by_source.get(candidate.source_dir, False):
                            continue
                        item = candidate
                        del self.queue[index]
                        break
                    if item is None:
                        should_wait = True
                    else:
                        should_wait = False
                        plan_run_id = f"{self.session_id}-{uuid.uuid4().hex[:12]}"
                        active_state = ActivePlanState(
                            plan_run_id=plan_run_id,
                            item=item,
                            status="queued",
                            started_at=time.time(),
                            stages=initial_stage_states(item),
                        )
                        self.active[plan_run_id] = active_state
                        self.current_plan_run_id = plan_run_id
                        self.current = item
                        self.current_stage = ""
            if should_wait:
                self.wake_event.wait(0.25)
                self.wake_event.clear()
                continue

            assert self.current is not None
            item = self.current
            plan_run_id = self.current_plan_run_id
            started_at = time.time()
            failed_stage = ""
            failed_message = ""
            try:
                self._process_item(item)
            except Exception as exc:
                failed_stage = self.current_stage
                failed_message = str(exc)
                self._emit(item, STAGE_ITEM, "failed", failed_message)
                with self.lock:
                    self.failed.append(item)
            else:
                with self.lock:
                    self.completed.append(item)
            finally:
                ended_at = time.time()
                with self.lock:
                    active_state = self.active.pop(plan_run_id, None)
                    if active_state is not None:
                        started_at = active_state.started_at or started_at
                    if failed_message:
                        self.finished_runs.append(
                            FinishedPlanState(
                                plan_run_id=plan_run_id,
                                item=item,
                                status="failed",
                                started_at=started_at,
                                ended_at=ended_at,
                                stage=failed_stage,
                                message=failed_message,
                            )
                        )
                    else:
                        final_status = "skipped"
                        if active_state is not None and active_state.status == "skipped":
                            final_status = "skipped"
                        elif active_state is not None and any(stage.status == "completed" for stage in active_state.stages):
                            final_status = "completed"
                        self.finished_runs.append(
                            FinishedPlanState(
                                plan_run_id=plan_run_id,
                                item=item,
                                status=final_status,
                                started_at=started_at,
                                ended_at=ended_at,
                            )
                        )
                    self.last_item = item
                    if self.rerun_after_current:
                        self.queue.appendleft(item)
                        self.rerun_after_current = False
                    source_dir = item.source_dir
                    if self.pause_after_current_by_source.get(source_dir, False):
                        self.paused_by_source[source_dir] = True
                        self.pause_after_current_by_source[source_dir] = False
                    self.current = None
                    self.current_stage = ""
                    self.current_plan_run_id = ""
                self.wake_event.set()


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

    print_help()
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


if __name__ == "__main__":
    raise SystemExit(main())
