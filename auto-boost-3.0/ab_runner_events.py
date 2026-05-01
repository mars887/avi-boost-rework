"""Structured progress/event bridge for runner-managed auto-boost stages."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


def emit_runner_child_event(
    stage: str,
    status: str,
    *,
    message: str = "",
    source: Optional[Path] = None,
    workdir: Optional[Path] = None,
    progress: Optional[float] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    events_path = os.environ.get("PBBATCH_RUNNER_CHILD_EVENTS_JSONL", "").strip()
    if not events_path:
        return
    event: Dict[str, Any] = {
        "event": "runner_child",
        "session_id": os.environ.get("PBBATCH_RUNNER_SESSION_ID", ""),
        "plan_run_id": os.environ.get("PBBATCH_RUNNER_PLAN_RUN_ID", ""),
        "plan": "",
        "mode": "",
        "stage": stage,
        "status": status,
        "message": message,
        "timestamp": time.time(),
        "source": str(source or ""),
        "workdir": str(workdir or ""),
        "progress": -1.0 if progress is None else float(progress),
        "started_at": 0.0,
        "ended_at": 0.0,
        "elapsed_seconds": 0.0,
    }
    if details:
        event["details"] = dict(details)
    try:
        path = Path(events_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        return
