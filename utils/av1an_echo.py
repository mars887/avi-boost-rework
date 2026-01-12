#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a command and mirror its output as-is.

Usage (recommended):
  python utils/av1an_echo.py -- av1an -i ... -o ...

Optional:
  --inherit  Run without piping (stdout/stderr inherited by console).
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Mirror command output (raw bytes).")
    ap.add_argument(
        "--inherit",
        action="store_true",
        help="Run with inherited stdout/stderr (no capture).",
    )
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run")
    args = ap.parse_args()

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("No command provided. Example: python utils/av1an_echo.py -- av1an ...", file=sys.stderr)
        return 2

    print("[run]", " ".join(cmd))

    if args.inherit:
        p = subprocess.Popen(cmd)
        return int(p.wait())

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert p.stdout is not None
    try:
        while True:
            chunk = p.stdout.read(1024)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
    except KeyboardInterrupt:
        try:
            p.terminate()
        except Exception:
            pass
        return 130
    return int(p.wait())


if __name__ == "__main__":
    raise SystemExit(main())
