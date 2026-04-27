from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol


CommandHandler = Callable[[str], str]
SnapshotProvider = Callable[[], Dict[str, Any]]
DEFAULT_MAX_OUTBOX_ITEMS = 200
DEFAULT_SNAPSHOT_INTERVAL_SECONDS = 10.0
DEFAULT_STOP_FLUSH_SECONDS = 20.0
RUNNER_SECRET_HEADER = "X-PBBATCH-Runner-Secret"
DISCORD_SECRET_HEADER = "X-PBBATCH-Discord-Secret"


class RunnerRuntimeProtocol(Protocol):
    session_id: str

    def snapshot(self, source_dir: str = "", session_id: str = "") -> Dict[str, Any]:
        ...

    def handle_command(self, command: str, source_dir: str = "") -> str:
        ...


class HttpRunnerIntegrationBridge:
    """Best-effort local HTTP bridge from runner to an external integration service."""

    def __init__(
        self,
        *,
        service_url: str,
        session_id: str,
        enabled: bool,
        source_dir: str = "",
        shared_secret: str = "",
        secret_header: str = RUNNER_SECRET_HEADER,
        name: str = "http",
        max_outbox_items: int = DEFAULT_MAX_OUTBOX_ITEMS,
        snapshot_interval_seconds: float = DEFAULT_SNAPSHOT_INTERVAL_SECONDS,
        stop_flush_seconds: float = DEFAULT_STOP_FLUSH_SECONDS,
    ) -> None:
        self.service_url = service_url.rstrip("/")
        self.session_id = session_id
        self.source_dir = str(source_dir or "")
        self.enabled = bool(enabled and self.service_url)
        self.shared_secret = str(shared_secret or "")
        self.secret_header = str(secret_header or RUNNER_SECRET_HEADER)
        self.name = str(name or "http")
        self.snapshot_provider: Optional[SnapshotProvider] = None
        self.command_handler: Optional[CommandHandler] = None
        self.runtime: Optional[RunnerRuntimeProtocol] = None
        self.stop_event = threading.Event()
        self.outbox: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max(1, int(max_outbox_items)))
        self.sender = threading.Thread(target=self._sender_main, name=f"{self.name}-bridge-sender", daemon=True)
        self.poller = threading.Thread(target=self._poller_main, name=f"{self.name}-bridge-poller", daemon=True)
        self.snapshooter = threading.Thread(target=self._snapshot_main, name=f"{self.name}-bridge-snapshot", daemon=True)
        self.snapshot_interval_seconds = max(2.0, float(snapshot_interval_seconds or DEFAULT_SNAPSHOT_INTERVAL_SECONDS))
        self.stop_flush_seconds = max(0.0, float(stop_flush_seconds or 0.0))
        self.stop_drain_deadline = 0.0
        self.connected = False
        self.last_error = ""
        self.last_error_at = 0.0
        self.ever_connected = False
        self.error_callback: Optional[Callable[[str], None]] = None
        self.executed_commands: Dict[str, tuple[str, str]] = {}

    def attach(self, *, snapshot_provider: SnapshotProvider, command_handler: CommandHandler) -> None:
        self.snapshot_provider = snapshot_provider
        self.command_handler = command_handler

    def set_error_callback(self, callback: Callable[[str], None]) -> None:
        self.error_callback = callback

    def start(self, runtime: Optional[RunnerRuntimeProtocol] = None) -> None:
        if runtime is not None:
            self.runtime = runtime
            self.attach(
                snapshot_provider=lambda: runtime.snapshot(self.source_dir, session_id=self.session_id),
                command_handler=lambda command: runtime.handle_command(command, source_dir=self.source_dir),
            )
        if not self.enabled or self.snapshot_provider is None:
            return
        result = self._post("/api/sessions/register", {"snapshot": self.snapshot_provider()}, timeout=10.0)
        self.connected = isinstance(result, dict) and str(result.get("status") or "").lower() == "ok"
        self.ever_connected = self.ever_connected or self.connected
        self.sender.start()
        self.poller.start()
        self.snapshooter.start()

    def stop(self) -> None:
        if not self.enabled:
            self.stop_event.set()
            return
        self.notify_snapshot()
        self.stop_drain_deadline = time.monotonic() + self.stop_flush_seconds
        self.stop_event.set()
        self.poller.join(timeout=2.0)
        self.snapshooter.join(timeout=2.0)
        self.sender.join(timeout=max(2.0, self.stop_flush_seconds + 1.0))

    def notify_event(self, event: Dict[str, Any], snapshot: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        if not self._event_matches_source(event):
            return
        selected_snapshot = snapshot
        if self.runtime is not None:
            selected_snapshot = self.runtime.snapshot(self.source_dir, session_id=self.session_id)
        low_priority = str(event.get("event") or "") == "runner_heartbeat"
        self._enqueue(
            {
                "path": f"/api/sessions/{self.session_id}/events",
                "payload": {
                    "event": event,
                    "snapshot": selected_snapshot,
                },
            },
            drop_if_full=low_priority,
        )

    def notify_snapshot(self) -> None:
        if not self.enabled or self.snapshot_provider is None:
            return
        self._enqueue(
            {
                "path": f"/api/sessions/{self.session_id}/snapshot",
                "payload": {"snapshot": self.snapshot_provider()},
            }
        )

    def _event_matches_source(self, event: Dict[str, Any]) -> bool:
        if not self.source_dir:
            return True
        source = str(event.get("source") or "")
        if not source:
            return True
        try:
            event_source_dir = str(Path(source).parent.resolve())
        except Exception:
            event_source_dir = ""
        return event_source_dir.lower() == self.source_dir.lower()

    def _enqueue(self, item: Dict[str, Any], *, drop_if_full: bool = False) -> None:
        try:
            self.outbox.put_nowait(item)
            return
        except queue.Full:
            if drop_if_full:
                return
        try:
            self.outbox.get_nowait()
        except queue.Empty:
            pass
        try:
            self.outbox.put_nowait(item)
        except queue.Full:
            pass

    def _sender_main(self) -> None:
        while not self.stop_event.is_set() or not self.outbox.empty():
            try:
                item = self.outbox.get(timeout=0.25)
            except queue.Empty:
                continue
            now = time.monotonic()
            next_attempt_at = float(item.get("_next_attempt_at") or 0.0)
            if next_attempt_at > now:
                if not self._retry_deadline_expired():
                    self._requeue_for_retry(item)
                    time.sleep(min(0.25, max(0.0, next_attempt_at - now)))
                continue
            result = self._post(str(item["path"]), dict(item["payload"]))
            if isinstance(result, dict) and str(result.get("status") or "").lower() == "ok":
                self.connected = True
                self.ever_connected = True
                continue
            self.connected = False
            if self._should_retry_item(item):
                self._schedule_retry(item)

    def _poller_main(self) -> None:
        while not self.stop_event.wait(1.0):
            if self.command_handler is None:
                continue
            commands = self._get(f"/api/sessions/{self.session_id}/commands")
            if not isinstance(commands, list):
                continue
            self.connected = True
            self.ever_connected = True
            for command in commands:
                if not isinstance(command, dict):
                    continue
                command_id = str(command.get("command_id") or "")
                name = str(command.get("name") or "")
                if not command_id or not name:
                    continue
                status = "ok"
                message = ""
                if command_id in self.executed_commands:
                    status, message = self.executed_commands[command_id]
                else:
                    try:
                        message = self.command_handler(name)
                    except Exception as exc:
                        status = "error"
                        message = str(exc)
                    self.executed_commands[command_id] = (status, message)
                    if len(self.executed_commands) > 500:
                        for old_id in list(self.executed_commands)[:100]:
                            self.executed_commands.pop(old_id, None)
                self._post(
                    f"/api/commands/{command_id}/ack",
                    {
                        "status": status,
                        "message": message,
                        "snapshot": self.snapshot_provider() if self.snapshot_provider is not None else {},
                    },
                )

    def _snapshot_main(self) -> None:
        while not self.stop_event.wait(self.snapshot_interval_seconds):
            self.notify_snapshot()

    @staticmethod
    def _item_low_priority(item: Dict[str, Any]) -> bool:
        payload = dict(item.get("payload") or {})
        event = dict(payload.get("event") or {})
        return str(event.get("event") or "") == "runner_heartbeat"

    def _retry_deadline_expired(self) -> bool:
        return bool(
            self.stop_event.is_set()
            and self.stop_drain_deadline
            and time.monotonic() >= self.stop_drain_deadline
        )

    def _should_retry_item(self, item: Dict[str, Any]) -> bool:
        if self._item_low_priority(item):
            return False
        return not self._retry_deadline_expired()

    @staticmethod
    def _retry_delay_seconds(attempts: int) -> float:
        return min(10.0, 0.5 * (2 ** max(0, min(int(attempts), 5))))

    def _schedule_retry(self, item: Dict[str, Any]) -> None:
        attempts = int(item.get("_attempts") or 0) + 1
        delay = self._retry_delay_seconds(attempts)
        if self.stop_event.is_set() and self.stop_drain_deadline:
            remaining = self.stop_drain_deadline - time.monotonic()
            if remaining <= 0:
                return
            delay = min(delay, max(0.0, remaining))
        item["_attempts"] = attempts
        item["_next_attempt_at"] = time.monotonic() + delay
        self._requeue_for_retry(item)

    def _requeue_for_retry(self, item: Dict[str, Any]) -> None:
        try:
            self.outbox.put_nowait(item)
            return
        except queue.Full:
            if self._item_low_priority(item):
                return
        try:
            self.outbox.get_nowait()
        except queue.Empty:
            pass
        try:
            self.outbox.put_nowait(item)
        except queue.Full:
            pass

    def _auth_headers(self) -> Dict[str, str]:
        if not self.shared_secret:
            return {}
        return {self.secret_header: self.shared_secret}

    def _get(self, path: str, *, timeout: float = 2.0) -> Any:
        url = self.service_url + path
        request = urllib.request.Request(url, headers=self._auth_headers(), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._record_error(exc)
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def _post(self, path: str, payload: Dict[str, Any], *, timeout: float = 2.0) -> Any:
        url = self.service_url + path
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", **self._auth_headers()},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._record_error(exc)
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def _record_error(self, exc: Exception) -> None:
        message = str(exc)
        if self.ever_connected and "timed out" in message.lower():
            self.last_error = message
            self.last_error_at = time.time()
            return
        self.connected = False
        if message == self.last_error and (time.time() - self.last_error_at) < 30.0:
            return
        self.last_error = message
        self.last_error_at = time.time()
        if self.error_callback is not None:
            try:
                self.error_callback(message)
            except Exception:
                pass


def attach_discord_integrations(
    runtime: Any,
    *,
    service_url: str,
    shared_secret: str = "",
    enabled: bool = True,
    verbose: bool = False,
    logger: Optional[Callable[[str], None]] = None,
) -> list[HttpRunnerIntegrationBridge]:
    if not enabled:
        return []
    log = logger or (lambda message: print(message, flush=True))
    bridges: list[HttpRunnerIntegrationBridge] = []
    for source_dir in list(getattr(runtime, "source_dirs", []) or []):
        session_id = runtime.integration_session_id_for_source(source_dir)
        bridge = HttpRunnerIntegrationBridge(
            service_url=service_url,
            session_id=session_id,
            source_dir=source_dir,
            enabled=True,
            shared_secret=shared_secret,
            secret_header=DISCORD_SECRET_HEADER,
            name="discord",
        )
        if verbose:
            bridge.set_error_callback(
                lambda message, sd=source_dir: log(f"[discord] bridge unavailable for {sd}: {message}")
            )
        runtime.add_integration(bridge)
        bridges.append(bridge)
    return bridges

