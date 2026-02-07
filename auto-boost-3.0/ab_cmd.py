"""Subprocess execution helpers with progress-aware output."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from ab_logging import is_progress_line, strip_ansi

def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    check: bool = True,
    inherit_output: bool = False,
) -> subprocess.CompletedProcess:
    """Run a command and stream output with progress-line handling."""
    cmd_str = " ".join(shlex.quote(str(x)) for x in cmd)
    print(f"[cmd] {cmd_str}")
    if inherit_output:
        return subprocess.run(
            list(map(str, cmd)),
            cwd=str(cwd) if cwd else None,
            check=check,
        )
    p = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    buf = ""
    progress_width = 0
    had_progress = False
    last_was_cr = False
    if p.stdout is not None:
        while True:
            ch = p.stdout.read(1)
            if ch == "":
                break
            if last_was_cr and ch == "\n":
                last_was_cr = False
                continue
            if ch in ("\n", "\r"):
                line = buf
                buf = ""
                clean = strip_ansi(line)
                if ch == "\r" or is_progress_line(clean):
                    s = clean.strip()
                    if len(s) < progress_width:
                        s = s + (" " * (progress_width - len(s)))
                    progress_width = max(progress_width, len(s))
                    sys.stdout.write(s + "\r")
                    sys.stdout.flush()
                    had_progress = True
                    last_was_cr = (ch == "\r")
                    continue

                if had_progress:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    had_progress = False
                    progress_width = 0
                last_was_cr = False

                sys.stdout.write(clean)
                sys.stdout.write("\n")
                sys.stdout.flush()
                continue

            buf += ch
            last_was_cr = False
            if len(buf) >= 2048:
                sys.stdout.write(strip_ansi(buf))
                sys.stdout.flush()
                buf = ""
    if buf:
        clean = strip_ansi(buf)
        if is_progress_line(clean):
            s = clean.strip()
            if len(s) < progress_width:
                s = s + (" " * (progress_width - len(s)))
            sys.stdout.write(s + "\r")
            sys.stdout.flush()
            had_progress = True
        else:
            if had_progress:
                sys.stdout.write("\n")
                sys.stdout.flush()
                had_progress = False
                progress_width = 0
            sys.stdout.write(clean)
            sys.stdout.flush()
    rc = p.wait()
    if had_progress:
        sys.stdout.write("\n")
        sys.stdout.flush()
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, list(cmd))
    return subprocess.CompletedProcess(list(cmd), rc)
