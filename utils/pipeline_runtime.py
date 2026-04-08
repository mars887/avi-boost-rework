from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


ROOT_DIR = Path(__file__).resolve().parent.parent
UTILS_DIR = ROOT_DIR / "utils"
AUTOBOOST_DIR = ROOT_DIR / "auto-boost-3.0"

LEGACY_PORTABLE_DIR = Path(r"C:\vapoursynth\vapoursynth-portable")
LEGACY_PYTHON_EXE = Path(r"C:\Python313\python.exe")
LEGACY_VS_PYTHON_EXE = Path(r"C:\vapoursynth\vapoursynth-portable\python.exe")
LEGACY_AV1AN_EXE = Path(r"C:\vapoursynth\vapoursynth-portable\av1an.exe")
LEGACY_PSD_SCRIPT = Path(r"C:\vapoursynth\vapoursynth-portable\Progressive-Scene-Detection.py")
AV1AN_FORK_REPO = "https://github.com/mars887/Av1an.git"


@dataclass(frozen=True)
class Toolchain:
    python_exe: str
    vs_python_exe: str
    av1an_exe: str
    psd_script: str


def _first_existing(candidates: Sequence[str | Path]) -> Optional[str]:
    for candidate in candidates:
        text = str(candidate).strip()
        if not text:
            continue
        path = Path(text)
        if path.is_absolute() and path.exists():
            return str(path)
        if shutil.which(text):
            return text
    return None


def load_toolchain() -> Toolchain:
    python_exe = (
        os.environ.get("PBBATCH_PYTHON")
        or _first_existing([LEGACY_PYTHON_EXE, sys.executable])
        or sys.executable
    )
    vs_python_exe = (
        os.environ.get("PBBATCH_VS_PYTHON")
        or _first_existing([LEGACY_VS_PYTHON_EXE, python_exe, sys.executable])
        or sys.executable
    )
    av1an_exe = (
        os.environ.get("PBBATCH_AV1AN")
        or _first_existing([LEGACY_AV1AN_EXE, "av1an"])
        or "av1an"
    )
    psd_script = (
        os.environ.get("PBBATCH_PSD_SCRIPT")
        or _first_existing([LEGACY_PSD_SCRIPT, "Progressive-Scene-Detection.py"])
        or "Progressive-Scene-Detection.py"
    )
    return Toolchain(
        python_exe=str(python_exe),
        vs_python_exe=str(vs_python_exe),
        av1an_exe=str(av1an_exe),
        psd_script=str(psd_script),
    )


def read_command_output(cmd: List[str], *, timeout: float = 5.0) -> str:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except Exception:
        return ""
    return str(proc.stdout or "")


def is_mars_av1an_fork(av1an_exe: str) -> bool:
    text = read_command_output([str(av1an_exe), "--version"])
    return AV1AN_FORK_REPO.lower() in text.lower()


def list_portable_encoder_binaries(encoder: str, *, portable_dir: Optional[Path] = None) -> List[str]:
    base_dir = Path(portable_dir or LEGACY_PORTABLE_DIR)
    if not base_dir.exists() or not base_dir.is_dir():
        return []

    normalized = str(encoder or "").strip().lower().replace("_", "-")
    prefix = "x265" if normalized in ("x265", "libx265") else "SvtAv1EncApp"
    out: List[str] = []
    for path in sorted(base_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file():
            continue
        if not path.name.lower().startswith(prefix.lower()):
            continue
        out.append(str(path.resolve()))
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def windows_bat_lines(lines: Iterable[str]) -> str:
    return "\r\n".join(lines).rstrip() + "\r\n"


def write_windows_bat(path: Path, lines: Iterable[str], *, encoding: str = "cp1251") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = windows_bat_lines(lines)
    try:
        path.write_text(text, encoding=encoding, newline="")
    except UnicodeEncodeError:
        path.write_text(text, encoding="utf-8", newline="")


def run_process(cmd: List[str], *, cwd: Optional[Path] = None) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None)
    return int(proc.returncode)
