from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict


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
