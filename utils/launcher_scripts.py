from __future__ import annotations

from pathlib import Path

from utils.pipeline_runtime import write_windows_bat


def _quoted(path: Path | str) -> str:
    return f'"{path}"'


def build_local_runner_lines(*, python_exe: str, project_root: Path) -> list[str]:
    runner_py = project_root / "runner.py"
    return [
        "@echo off",
        "setlocal EnableExtensions DisableDelayedExpansion",
        "chcp 1251 >nul",
        'pushd "%~dp0"',
        f"set \"PYTHON_EXE={python_exe}\"",
        f"set \"RUNNER_PY={runner_py}\"",
        'if "%~1"=="" (',
        '  if exist "%~dp0full-batch.plan" (',
        '    "%PYTHON_EXE%" "%RUNNER_PY%" "%~dp0full-batch.plan"',
        "  ) else (",
        '    echo [err] full-batch.plan not found in "%~dp0"',
        "    exit /b 1",
        "  )",
        ") else (",
        '  "%PYTHON_EXE%" "%RUNNER_PY%" %*',
        ")",
        "exit /b %errorlevel%",
    ]


def build_local_batch_manager_lines(*, python_exe: str, project_root: Path) -> list[str]:
    batch_manager_py = project_root / "utils" / "batch-manager.py"
    return [
        "@echo off",
        "setlocal EnableExtensions DisableDelayedExpansion",
        "chcp 1251 >nul",
        'set "CALLER_DIR=%cd%"',
        'pushd "%~dp0"',
        f"set \"PYTHON_EXE={python_exe}\"",
        f"set \"BATCH_MANAGER_PY={batch_manager_py}\"",
        'if "%~1"=="" (',
        '  if exist "%CALLER_DIR%\\full-batch.plan" (',
        '    "%PYTHON_EXE%" "%BATCH_MANAGER_PY%" "%CALLER_DIR%\\full-batch.plan"',
        "  ) else (",
        '    "%PYTHON_EXE%" "%BATCH_MANAGER_PY%" "%CALLER_DIR%"',
        "  )",
        ") else (",
        '  "%PYTHON_EXE%" "%BATCH_MANAGER_PY%" %*',
        ")",
        "exit /b %errorlevel%",
    ]


def write_directory_launchers(*, source_dir: Path, python_exe: str, project_root: Path) -> None:
    write_windows_bat(source_dir / "runner.bat", build_local_runner_lines(python_exe=python_exe, project_root=project_root))
    write_windows_bat(
        source_dir / "Batch Manager.bat",
        build_local_batch_manager_lines(python_exe=python_exe, project_root=project_root),
    )
