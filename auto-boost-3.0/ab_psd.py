"""PSD scene detection stage helpers."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Optional

from ab_cmd import run_cmd
from ab_fs import ensure_dir, ensure_exists, load_json, save_json
from ab_scenes_io import sanitize_scenes_json

def run_psd(psd_script: Path, psd_python: Optional[Path], input_file: Path, base_scenes_path: Path, extra_args: str) -> None:
    """Run PSD scene detection and normalize scenes.json output."""
    ensure_exists(psd_script, "PSD script")
    ensure_dir(base_scenes_path.parent)

    py = str(psd_python) if psd_python else sys.executable
    cmd = [py, str(psd_script), "-i", str(input_file), "-o", str(base_scenes_path)]
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    run_cmd(cmd, check=True, cwd=base_scenes_path.parent)

    # Normalise to the exact "base scenes" contract we want downstream.
    raw = load_json(base_scenes_path)
    norm = sanitize_scenes_json(raw)
    save_json(base_scenes_path, norm)
    print(f"[ok] PSD scenes written: {base_scenes_path}")
