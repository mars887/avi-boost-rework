from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from utils.runner_lock import SourceDirLock
from utils.runner_source_info import prepare_source_info

from .helpers import build_queue
from .logs import RunnerLogLine
from .models import QueueItem
from .session import SessionController
from .stage_bank import StageBankConfig


EventSink = Callable[[Dict[str, Any], Dict[str, Any]], None]
LogSink = Callable[[RunnerLogLine], None]


class RunnerIntegration(Protocol):
    connected: bool
    session_id: str
    source_dir: str

    def start(self, runtime: "RunnerRuntime") -> None:
        ...

    def stop(self) -> None:
        ...

    def notify_event(self, event: Dict[str, Any], snapshot: Dict[str, Any]) -> None:
        ...

    def notify_snapshot(self) -> None:
        ...


@dataclass
class RunnerLaunchConfig:
    plans: List[str] = field(default_factory=list)
    mode: str = ""
    plan_modes: Dict[str, str] = field(default_factory=dict)
    events_jsonl: str = ""
    add_source_bitrate: bool = False
    exit_when_idle: bool = False
    no_interactive: bool = False
    session_id: str = ""
    stage_bank_config: Optional[StageBankConfig] = None
    integrations: List[RunnerIntegration] = field(default_factory=list)


class RunnerRuntime:
    def __init__(self, config: RunnerLaunchConfig) -> None:
        self.config = config
        self.queue: List[QueueItem] = build_queue(config.plans, config.mode, config.plan_modes) if config.plans else []
        self.source_dirs = sorted({item.source_dir for item in self.queue}, key=str.lower)
        self.controller = SessionController(
            items=self.queue,
            events_jsonl=config.events_jsonl,
            add_source_bitrate=config.add_source_bitrate,
            exit_when_idle=(config.exit_when_idle or config.no_interactive),
            session_id=config.session_id,
            stage_bank_config=config.stage_bank_config,
        )
        self.session_id = self.controller.session_id
        self.integrations: List[RunnerIntegration] = list(config.integrations)
        self.event_sinks: List[EventSink] = []
        self.folder_locks = [
            SourceDirLock(source_dir=source_dir, session_id=self.session_id, enabled=True)
            for source_dir in self.source_dirs
        ]
        self._started = False
        self._closed = False
        self.controller.add_event_sink(self._notify_event)

    def add_integration(self, integration: RunnerIntegration) -> None:
        self.integrations.append(integration)

    def add_event_sink(self, sink: EventSink) -> None:
        self.event_sinks.append(sink)

    def add_log_sink(self, sink: LogSink) -> None:
        self.controller.add_log_sink(sink)

    def integration_session_id_for_source(self, source_dir: str) -> str:
        if len(self.source_dirs) <= 1:
            return self.session_id
        suffix = hashlib.sha1(source_dir.lower().encode("utf-8", errors="ignore")).hexdigest()[:8]
        return f"{self.session_id}-{suffix}"

    def snapshot(self, source_dir: str = "", session_id: str = "") -> Dict[str, Any]:
        return self.controller.snapshot(source_dir, session_id=session_id)

    def handle_command(self, command: str, source_dir: str = "") -> str:
        return self.controller.handle_command(command, source_dir=source_dir)

    def request_pause_after_plans(self, source_dir: str = "") -> None:
        self.controller.request_pause_after_plans(source_dir)

    def request_pause_plan(self, plan_run_id: str) -> None:
        self.controller.request_pause_plan(plan_run_id)

    def request_stop_plan(self, plan_run_id: str) -> None:
        self.controller.request_stop_plan(plan_run_id)

    def retry_failed_item(self, plan: str, source: str = "") -> bool:
        return self.controller.retry_failed_item(plan, source)

    def prioritize_queued_item(self, plan: str, source: str = "") -> bool:
        return self.controller.prioritize_queued_item(plan, source)

    def remove_queued_item(self, plan: str, source: str = "") -> bool:
        return self.controller.remove_queued_item(plan, source)

    def start(self) -> None:
        if self._started:
            return
        if not self.queue:
            raise RuntimeError("no plans resolved")
        acquired: List[SourceDirLock] = []
        try:
            for folder_lock in self.folder_locks:
                folder_lock.acquire()
                acquired.append(folder_lock)
            for item in self.queue:
                prepare_source_info(item)
            self.controller.start()
            self._started = True
            for integration in self.integrations:
                integration.start(self)
        except Exception:
            if self._started:
                try:
                    self.controller.request_stop()
                    self.controller.join()
                except Exception:
                    pass
                self._started = False
            for integration in self.integrations:
                try:
                    integration.stop()
                except Exception:
                    pass
            for folder_lock in acquired:
                folder_lock.release()
            raise

    def request_stop(self) -> None:
        self.controller.request_stop()

    def stop(self) -> None:
        self.request_stop()

    def join(self) -> int:
        try:
            if self._started:
                self.controller.join()
            return 1 if self.controller.failed else 0
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for integration in self.integrations:
            try:
                integration.stop()
            except Exception as exc:
                print(f"[runner] integration stop failed: {exc}", file=sys.stderr, flush=True)
        for folder_lock in self.folder_locks:
            folder_lock.release()

    def is_finished(self) -> bool:
        return self.controller.is_finished()

    def is_busy(self) -> bool:
        return self.controller.is_busy()

    @property
    def failed(self) -> List[QueueItem]:
        return self.controller.failed

    def _notify_event(self, payload: Dict[str, Any], snapshot: Dict[str, Any]) -> None:
        for sink in list(self.event_sinks):
            try:
                sink(payload, snapshot)
            except Exception as exc:
                print(f"[runner] runtime event sink failed: {exc}", file=sys.stderr, flush=True)
        for integration in list(self.integrations):
            try:
                integration.notify_event(payload, snapshot)
            except Exception as exc:
                print(f"[runner] integration event sink failed: {exc}", file=sys.stderr, flush=True)
