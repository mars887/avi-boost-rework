"""PSD scene detection stage helpers."""

from __future__ import annotations

import re
import shlex
import sys
import time
from pathlib import Path
from typing import Optional

from ab_cmd import run_cmd
from ab_fs import ensure_dir, ensure_exists, load_json, save_json
from ab_runner_events import emit_runner_child_event
from ab_scenes_io import sanitize_scenes_json

PSD_PROGRESS_RE = re.compile(
    r"^(?:\[progress\]\s*)?Frame\s+(?P<frame>\d+)\s*/\s*Detecting scenes\s*/\s*(?P<fps>[0-9]+(?:\.[0-9]+)?)\s*fps$",
    re.IGNORECASE,
)


def run_psd(psd_script: Path, psd_python: Optional[Path], input_file: Path, base_scenes_path: Path, extra_args: str) -> None:
    """Run PSD scene detection and normalize scenes.json output."""
    ensure_exists(psd_script, "PSD script")
    ensure_dir(base_scenes_path.parent)

    py = str(psd_python) if psd_python else sys.executable
    cmd = [py, str(psd_script), "-i", str(input_file), "-o", str(base_scenes_path)]
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    last_progress_at = 0.0

    def emit_psd_progress(line: str) -> None:
        nonlocal last_progress_at
        match = PSD_PROGRESS_RE.match(str(line or "").strip())
        if not match:
            return
        now = time.monotonic()
        if now - last_progress_at < 1.0:
            return
        last_progress_at = now
        try:
            fps = float(match.group("fps"))
        except Exception:
            return
        emit_runner_child_event(
            "Auto-Boost: PSD Scene Detection",
            "progress",
            source=input_file,
            workdir=base_scenes_path.parent,
            details={"fps": fps, "frame": int(match.group("frame"))},
        )

    run_cmd(cmd, check=True, cwd=base_scenes_path.parent, progress_callback=emit_psd_progress)

    # Normalise to the exact "base scenes" contract we want downstream.
    raw = load_json(base_scenes_path)
    norm = sanitize_scenes_json(raw)
    save_json(base_scenes_path, norm)
    print(f"[ok] PSD scenes written: {base_scenes_path}")
