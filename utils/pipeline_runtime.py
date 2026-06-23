from __future__ import annotations

import os
import json
import re
import shutil
import subprocess
import sys
import atexit
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, TextIO

from utils.workdir_layout import final_output_path_for_source as _final_output_path_for_source


ROOT_DIR = Path(__file__).resolve().parent.parent
UTILS_DIR = ROOT_DIR / "utils"
AUTOBOOST_DIR = ROOT_DIR / "auto-boost-3.0"
PROJECT_PSD_SCRIPT = ROOT_DIR / "Progressive-Scene-Detection.py"

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
        or _first_existing([PROJECT_PSD_SCRIPT, LEGACY_PSD_SCRIPT, "Progressive-Scene-Detection.py"])
        or str(PROJECT_PSD_SCRIPT)
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


# Final-output naming lives in utils.workdir_layout (the single source of truth
# for on-disk paths); re-exported here for the modules that already import it
# from pipeline_runtime.
final_output_path_for_source = _final_output_path_for_source


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


def setup_stage_logging(log_path: str | Path, *, stage_name: str, base_dir: Optional[Path] = None) -> None:
    if not log_path:
        return
    path = Path(log_path)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    ensure_dir(path.parent)
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    log_fh = path.open("a", encoding=enc, errors="replace")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        log_fh.write(f"=== START {stage_name} {ts} ===\n")
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
            log_fh.write(f"=== END {stage_name} {ts_end} ===\n")
            log_fh.flush()
        except Exception:
            pass
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        tee_out.close_log()
        tee_err.close_log()

    atexit.register(_cleanup)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


def which_or(name: str, fallback: str = "") -> str:
    return shutil.which(name) or fallback or name


def eprint(*args: Any) -> None:
    """Print to stderr. Shared by the standalone stage tools."""
    print(*args, file=sys.stderr)


# Matches the bare "progress: NN%" / "processed - NN%" status lines emitted by
# ffmpeg/mkvtoolnix wrappers so they can be collapsed onto a single console row.
PROGRESS_RE = re.compile(r"^(progress|processed)\s*[:\-]?\s*\d+%\s*$", re.IGNORECASE)


def is_progress_line(line: str, *, max_len: int = 40) -> bool:
    """True for short, percentage-style status lines that should overwrite in place."""
    s = line.strip()
    if not s:
        return False
    if PROGRESS_RE.match(s):
        return True
    if s.endswith("%") and any(ch.isdigit() for ch in s) and len(s) <= max_len:
        return True
    return False


def run_cmd_streamed(
    cmd: Sequence[str],
    *,
    check: bool = True,
    cwd: Optional[Path] = None,
    echo: bool = True,
) -> int:
    """Run a command, streaming stdout live with in-place progress-line handling.

    Progress lines (see :func:`is_progress_line`) are collapsed onto one
    carriage-return-updated row; every other line is printed normally. Raises
    ``RuntimeError`` on a nonzero exit when ``check`` is true. This is the single
    streaming runner shared by the standalone stage tools (e.g. demux); use
    :func:`run_cmd` instead when you need a captured, non-streaming result.
    """
    if echo:
        print("[cmd]", " ".join(str(x) for x in cmd))
    proc = subprocess.Popen(
        [str(x) for x in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(cwd) if cwd is not None else None,
    )
    progress_width = 0
    had_progress = False
    if proc.stdout is not None:
        for line in proc.stdout:
            if is_progress_line(line):
                s = line.strip()
                if len(s) < progress_width:
                    s = s + (" " * (progress_width - len(s)))
                progress_width = max(progress_width, len(s))
                sys.stdout.write(s + "\r")
                sys.stdout.flush()
                had_progress = True
                continue
            if had_progress:
                sys.stdout.write("\n")
                sys.stdout.flush()
                had_progress = False
                progress_width = 0
            if line:
                sys.stdout.write(line)
                if not line.endswith("\n"):
                    sys.stdout.write("\n")
                sys.stdout.flush()
    rc = proc.wait()
    if had_progress:
        sys.stdout.write("\n")
        sys.stdout.flush()
    if check and rc != 0:
        raise RuntimeError(f"Command failed with code {rc}")
    return rc


def run_cmd(
    cmd: List[str],
    *,
    check: bool = False,
    capture: bool = False,
    cwd: Optional[Path] = None,
    text: bool = True,
    encoding: str = "utf-8",
) -> subprocess.CompletedProcess:
    kwargs: dict[str, Any] = {
        "check": check,
        "text": text,
    }
    if cwd is not None:
        kwargs["cwd"] = str(cwd)
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    if text:
        kwargs["encoding"] = encoding
        kwargs["errors"] = "replace"
    return subprocess.run(cmd, **kwargs)


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
