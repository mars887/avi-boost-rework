"""Logging, tee streams, and progress filtering."""

from __future__ import annotations

import atexit
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

PROGRESS_RE = re.compile(
    r"^(progress|processed)\s*[:\-]?\s*\d+%\s*$|^frame\s+\d+\s*/.*fps$|^creating lwi index file\s+\d+%\s*$",
    re.IGNORECASE,
)

def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)

def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)

def is_progress_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if PROGRESS_RE.match(s):
        return True
    if s.endswith("%") and any(ch.isdigit() for ch in s) and len(s) <= 60:
        return True
    if s.lower().startswith("frame ") and "fps" in s.lower():
        return True
    return False

class TeeStream:
    def __init__(self, stream: TextIO, log_file: TextIO) -> None:
        self._stream = stream
        self._log: Optional[TextIO] = log_file

    def write(self, s: str) -> int:
        try:
            self._stream.write(s)
            self._stream.flush()
        except Exception:
            pass
        if self._log is not None:
            try:
                self._log.write(s)
                self._log.flush()
            except Exception:
                self._log = None
        return len(s)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception:
            pass
        if self._log is not None:
            try:
                self._log.flush()
            except Exception:
                self._log = None

    def close_log(self) -> None:
        if self._log is None:
            return
        try:
            self._log.flush()
        except Exception:
            pass
        try:
            self._log.close()
        except Exception:
            pass
        self._log = None

    def isatty(self) -> bool:
        return bool(getattr(self._stream, "isatty", lambda: False)())

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8")

def setup_logging(log_path: str, workdir: Optional[Path] = None) -> None:
    """Enable tee logging to a file for stdout and stderr."""
    if not log_path:
        return
    p = Path(log_path)
    if not p.is_absolute() and workdir is not None:
        p = workdir / p
    p.parent.mkdir(parents=True, exist_ok=True)
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    log_fh = p.open("a", encoding=enc, errors="replace")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        log_fh.write(f"=== START auto-boost {ts} ===\n")
        log_fh.flush()
    except Exception:
        pass
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    tee_out = TeeStream(orig_stdout, log_fh)
    tee_err = TeeStream(orig_stderr, log_fh)
    sys.stdout = tee_out
    sys.stderr = tee_err

    def _cleanup() -> None:
        ts_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            log_fh.write(f"=== END auto-boost {ts_end} ===\n")
            log_fh.flush()
        except Exception:
            pass
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        tee_out.close_log()
        tee_err.close_log()

    atexit.register(_cleanup)
