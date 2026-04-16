from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional


CommandHandler = Callable[[str], str]
SnapshotProvider = Callable[[], Dict[str, Any]]


class DiscordBridge:
    """Best-effort local bridge from runner to the Discord bot service."""

    def __init__(self, *, service_url: str, session_id: str, enabled: bool) -> None:
        self.service_url = service_url.rstrip("/")
        self.session_id = session_id
        self.enabled = bool(enabled and self.service_url)
        self.snapshot_provider: Optional[SnapshotProvider] = None
        self.command_handler: Optional[CommandHandler] = None
        self.stop_event = threading.Event()
        self.outbox: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.sender = threading.Thread(target=self._sender_main, name="discord-bridge-sender", daemon=True)
        self.poller = threading.Thread(target=self._poller_main, name="discord-bridge-poller", daemon=True)
        self.connected = False
        self.last_error = ""
        self.last_error_at = 0.0
        self.error_callback: Optional[Callable[[str], None]] = None

    def attach(self, *, snapshot_provider: SnapshotProvider, command_handler: CommandHandler) -> None:
        self.snapshot_provider = snapshot_provider
        self.command_handler = command_handler

    def set_error_callback(self, callback: Callable[[str], None]) -> None:
        self.error_callback = callback

    def start(self) -> None:
        if not self.enabled or self.snapshot_provider is None:
            return
        result = self._post("/api/sessions/register", {"snapshot": self.snapshot_provider()})
        self.connected = isinstance(result, dict) and str(result.get("status") or "").lower() == "ok"
        self.sender.start()
        self.poller.start()

    def stop(self) -> None:
        if not self.enabled:
            self.stop_event.set()
            return
        self.notify_snapshot()
        self.stop_event.set()
        self.sender.join(timeout=2.0)
        self.poller.join(timeout=2.0)

    def notify_event(self, event: Dict[str, Any], snapshot: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.outbox.put(
            {
                "path": f"/api/sessions/{self.session_id}/events",
                "payload": {
                    "event": event,
                    "snapshot": snapshot,
                },
            }
        )

    def notify_snapshot(self) -> None:
        if not self.enabled or self.snapshot_provider is None:
            return
        self.outbox.put(
            {
                "path": f"/api/sessions/{self.session_id}/snapshot",
                "payload": {"snapshot": self.snapshot_provider()},
            }
        )

    def _sender_main(self) -> None:
        while not self.stop_event.is_set() or not self.outbox.empty():
            try:
                item = self.outbox.get(timeout=0.25)
            except queue.Empty:
                continue
            result = self._post(str(item["path"]), dict(item["payload"]))
            if isinstance(result, dict) and str(result.get("status") or "").lower() == "ok":
                self.connected = True

    def _poller_main(self) -> None:
        while not self.stop_event.wait(1.0):
            if self.command_handler is None:
                continue
            commands = self._get(f"/api/sessions/{self.session_id}/commands")
            if not isinstance(commands, list):
                continue
            self.connected = True
            for command in commands:
                if not isinstance(command, dict):
                    continue
                command_id = str(command.get("command_id") or "")
                name = str(command.get("name") or "")
                if not command_id or not name:
                    continue
                status = "ok"
                message = ""
                try:
                    message = self.command_handler(name)
                except Exception as exc:
                    status = "error"
                    message = str(exc)
                self._post(
                    f"/api/commands/{command_id}/ack",
                    {
                        "status": status,
                        "message": message,
                        "snapshot": self.snapshot_provider() if self.snapshot_provider is not None else {},
                    },
                )

    def _get(self, path: str) -> Any:
        url = self.service_url + path
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
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

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        url = self.service_url + path
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=2.0) as response:
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
        self.connected = False
        message = str(exc)
        if message == self.last_error and (time.time() - self.last_error_at) < 30.0:
            return
        self.last_error = message
        self.last_error_at = time.time()
        if self.error_callback is not None:
            try:
                self.error_callback(message)
            except Exception:
                pass
