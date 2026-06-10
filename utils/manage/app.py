from __future__ import annotations

import sys
from typing import List, Sequence

from utils.manage.discovery import WorkdirRef, discover_workdirs


def run_manage(argv: Sequence[str]) -> int:
    """Entry point for ``python main.py manage [folder|plan ...]``."""
    args: List[str] = [str(arg) for arg in argv if str(arg).strip()]
    try:
        refs: List[WorkdirRef] = discover_workdirs(args)
    except RuntimeError as exc:
        print(f"[manage] {exc}", file=sys.stderr)
        print(
            "[manage] pass a workdir, a source directory with .plan files, or a .plan/batch .plan",
            file=sys.stderr,
        )
        return 2

    print(f"[manage] discovered {len(refs)} workdir(s):", flush=True)
    for ref in refs:
        suffix = f" (plan: {ref.plan_path})" if ref.plan_path else " (no plan)"
        print(f"  - {ref.workdir}{suffix}", flush=True)

    try:
        from utils.manage.gui_qt import run_gui
    except ImportError as exc:
        print(f"[manage] Qt GUI is unavailable: {exc}", file=sys.stderr)
        print('[manage] install it with: python -m pip install "PySide6>=6.10,<7"', file=sys.stderr)
        return 2
    return run_gui(refs)


__all__ = ["run_manage"]
