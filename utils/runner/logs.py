from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunnerLogLine:
    timestamp: float
    session_id: str
    plan_run_id: str
    source: str
    plan: str
    stage: str
    stream: str
    text: str
    raw_text: str = ""
    log_path: str = ""
