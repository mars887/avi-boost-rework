from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from utils.pipeline_runtime import ROOT_DIR, UTILS_DIR, load_toolchain
from utils.plan_model import FilePlan, load_plan, resolve_batch_plan, resolve_paths
from utils.plan_support import collect_file_plan_paths, refresh_support_for_plan_paths
from utils.main_select_gui import apply_template_default_groups, run_main_selection_gui
from utils.track_summary import summarize_files
from utils.track_config_gui import build_default_defaults_dict, load_gui_data_from_paths


VIDEO_EXTS = {".mkv", ".mp4"}
VIDEO_EXTRACT_EXTS = {".mkv", ".mp4", ".avi", ".mov"}


def enter_numbers(raw: str, min_value: int, max_value: int) -> List[int]:
    selected: List[int] = []
    for token in raw.split():
        try:
            if token == "*":
                selected.extend(range(min_value, max_value + 1))
                continue
            if ".." in token:
                a_raw, b_raw = token.split("..", 1)
                a = int(a_raw)
                b = int(b_raw)
                lo = min(a, b)
                hi = max(a, b)
                selected.extend(range(lo, hi + 1))
                continue
            if token.startswith("/") and token[1:].isdigit():
                value = int(token[1:])
                selected = [item for item in selected if item != value]
                continue
            if token.isdigit():
                selected.append(int(token))
                continue
        except Exception:
            continue

    out: List[int] = []
    seen: set[int] = set()
    for value in selected:
        if min_value <= value <= max_value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def is_supported_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS and not path.name.lower().endswith(("-av1.mkv", "-av1.mp4"))

def is_supported_video_file_for_extract(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTRACT_EXTS


def prompt_for_path(prompt: str) -> Path:
    raw = input(prompt).strip()
    if not raw:
        raise RuntimeError("No input provided")
    return Path(raw).expanduser().resolve()


def select_files_from_directory(directory: Path) -> List[Path]:
    files = sorted([path for path in directory.iterdir() if is_supported_video_file(path)], key=lambda item: item.name.lower())
    if not files:
        raise RuntimeError(f"No supported video files found in {directory}")
    print("Files:")
    for index, path in enumerate(files, start=1):
        print(f"  {index}) {path.name}")
    raw = input("Select files (e.g. 1 2 5..7 or *): ").strip()
    indexes = enter_numbers(raw or "*", 1, len(files))
    if not indexes:
        raise RuntimeError("No files selected")
    return [files[index - 1].resolve() for index in indexes]


def _add_input_item(
    items: List[Tuple[Path, Path]],
    seen: set[str],
    *,
    gui_path: Path,
    source_path: Path,
) -> None:
    key = str(source_path.resolve()).lower()
    if key in seen:
        return
    seen.add(key)
    items.append((gui_path.resolve(), source_path.resolve()))


def resolve_input_items(raw_paths: Sequence[str]) -> List[Tuple[Path, Path]]:
    if not raw_paths:
        target = prompt_for_path("Enter file or directory path: ")
        raw_paths = [str(target)]

    items: List[Tuple[Path, Path]] = []
    seen: set[str] = set()
    for raw in raw_paths:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            for source in select_files_from_directory(path):
                _add_input_item(items, seen, gui_path=source, source_path=source)
            continue
        if path.suffix.lower() == ".plan":
            plan = load_plan(path)
            if isinstance(plan, FilePlan):
                _add_input_item(items, seen, gui_path=path, source_path=resolve_paths(plan, path).source)
                continue
            for resolved in resolve_batch_plan(path):
                _add_input_item(items, seen, gui_path=resolved.paths.plan_path, source_path=resolved.paths.source)
            continue
        if path.suffix.lower() == ".bat" and path.name.lower() in ("runner.bat", "batch manager.bat"):
            for plan_path in collect_file_plan_paths(path.parent):
                plan = load_plan(plan_path)
                if isinstance(plan, FilePlan):
                    _add_input_item(items, seen, gui_path=plan_path, source_path=resolve_paths(plan, plan_path).source)
            continue
        if not is_supported_video_file_for_extract(path):
            raise RuntimeError(f"Unsupported input path: {path}")
        _add_input_item(items, seen, gui_path=path, source_path=path)
    return items


def resolve_source_files(raw_paths: Sequence[str]) -> List[Path]:
    return [source for _, source in resolve_input_items(raw_paths)]


def build_gui_json_input(files: Sequence[Path]) -> Dict[str, Any]:
    summary = summarize_files(files)
    return {
        "files": [str(path) for path in files],
        "summary": summary,
        "defaults": build_default_defaults_dict(),
        "outputMode": "json",
    }


def run_track_config_gui_json(files: Sequence[Path]) -> Dict[str, List[Dict[str, Any]]]:
    toolchain = load_toolchain()
    payload = json.dumps(build_gui_json_input(files), ensure_ascii=False)
    cmd = [toolchain.python_exe, str(UTILS_DIR / "track_config_gui.py")]
    proc = subprocess.run(
        cmd,
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT_DIR),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"track_config_gui failed with code {proc.returncode}\n{proc.stderr}")
    data = json.loads(proc.stdout or "{}")
    if str(data.get("status") or "").lower() != "ok":
        return {}
    result = data.get("result") or {}
    if not isinstance(result, dict):
        return {}
    return result


def run_gui_plan_save(input_paths: Sequence[Path]) -> List[Path]:
    toolchain = load_toolchain()
    cmd = [toolchain.python_exe, str(UTILS_DIR / "track_config_gui.py"), *[str(path) for path in input_paths]]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT_DIR),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"track_config_gui failed with code {proc.returncode}\n{proc.stderr}")
    data = json.loads(proc.stdout or "{}")
    if str(data.get("status") or "").lower() != "ok":
        return []
    saved = [Path(item).resolve() for item in (data.get("savedPlans") or [])]
    return saved


def run_gui_plan_save_from_data(data: Dict[str, Any]) -> List[Path]:
    toolchain = load_toolchain()
    payload_data = dict(data)
    payload_data["outputMode"] = "plans"
    payload = json.dumps(payload_data, ensure_ascii=False)
    cmd = [toolchain.python_exe, str(UTILS_DIR / "track_config_gui.py")]
    proc = subprocess.run(
        cmd,
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT_DIR),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"track_config_gui failed with code {proc.returncode}\n{proc.stderr}")
    data = json.loads(proc.stdout or "{}")
    if str(data.get("status") or "").lower() != "ok":
        return []
    saved = [Path(item).resolve() for item in (data.get("savedPlans") or [])]
    return saved


def run_batch_track_extract(files: Sequence[Path], *, overwrite: bool) -> int:
    result = run_track_config_gui_json(files)
    if not result:
        print("No GUI result received.")
        return 0

    items: List[Dict[str, Any]] = []
    for source in files:
        track_list = result.get(str(source)) or []
        selected = [track for track in track_list if str(track.get("trackStatus") or "").upper() != "SKIP"]
        if not selected:
            continue
        items.append(
            {
                "source": str(source),
                "tracks": [
                    {
                        "trackId": track.get("trackId"),
                        "type": track.get("type"),
                        "trackStatus": track.get("trackStatus"),
                        "origName": track.get("origName"),
                        "origLang": track.get("origLang"),
                        "trackMux": track.get("trackMux") or {},
                    }
                    for track in selected
                ],
            }
        )

    if not items:
        print("No tracks selected for extraction.")
        return 0

    output_root = files[0].parent
    request = {
        "outputRoot": str(output_root),
        "overwrite": bool(overwrite),
        "items": items,
    }
    toolchain = load_toolchain()
    cmd = [toolchain.python_exe, str(UTILS_DIR / "batch-track-extract.py")]
    proc = subprocess.run(
        cmd,
        input=json.dumps(request, ensure_ascii=False),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT_DIR),
    )
    if proc.stderr.strip():
        print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise RuntimeError(f"batch-track-extract failed with code {proc.returncode}")
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception:
            print(proc.stdout)
    return 0


def run_extract_attachments(files: Sequence[Path], *, outdir: str = "") -> int:
    if not files:
        resolved_files = resolve_source_files([])
    else:
        resolved_files = resolve_source_files([str(item) for item in files])
    toolchain = load_toolchain()
    script = ROOT_DIR / "additional" / "extract_attachments.py"
    if not script.exists():
        raise RuntimeError(f"Attachment extractor not found: {script}")
    for source in resolved_files:
        cmd = [toolchain.python_exe, str(script), str(source)]
        if outdir and len(resolved_files) == 1:
            cmd.extend(["--outdir", outdir])
        print("[cmd]", subprocess.list2cmdline(cmd))
        rc = subprocess.run(cmd, cwd=str(ROOT_DIR)).returncode
        if rc != 0:
            return int(rc)
    return 0


def run_main(paths: Sequence[str]) -> int:
    selection = run_main_selection_gui(paths)
    if selection is None:
        print("No files selected.")
        return 0
    input_items = selection.input_items
    files = [source for _, source in input_items]
    if not files:
        print("No files selected.")
        return 1
    gui_data = load_gui_data_from_paths([str(gui_path) for gui_path, _ in input_items])
    gui_data = apply_template_default_groups(gui_data, selection.template_plan_path, selection.default_groups)
    print("Files:")
    for path in files:
        print(f"  {path.name}")
    summary = gui_data.get("summary") or summarize_files(files)
    print("\nTracks summary:")
    for row in summary:
        print(f"  {row.get('line')}")
    saved_plans = run_gui_plan_save_from_data(gui_data)
    if not saved_plans:
        print("No plans saved.")
        return 0
    toolchain = load_toolchain()
    written = refresh_support_for_plan_paths(plan_paths=saved_plans, python_exe=toolchain.python_exe, project_root=ROOT_DIR)
    for plan in saved_plans:
        print(f"Saved: {plan}")
    for source_dir, entries in written.items():
        for kind, path in entries.items():
            print(f"Generated {kind}: {path}")
        print(f"Generated: {source_dir / 'runner.bat'}")
        print(f"Generated: {source_dir / 'Batch Manager.bat'}")
    return 0


def run_extract(paths: Sequence[str], *, overwrite: bool) -> int:
    files = resolve_source_files(paths)
    if not files:
        print("No files selected.")
        return 1
    print("Files:")
    for path in files:
        print(f"  {path.name}")
    summary = summarize_files(files)
    print("\nTracks summary:")
    for row in summary:
        print(f"  {row.get('line')}")
    return run_batch_track_extract(files, overwrite=overwrite)
