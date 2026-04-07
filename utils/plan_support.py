from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from utils.launcher_scripts import write_directory_launchers
from utils.plan_model import FILE_PLAN_TYPE, load_plan, make_batch_plan, save_plan


def collect_file_plan_paths(source_dir: Path) -> List[Path]:
    out: List[Path] = []
    for path in sorted(source_dir.glob("*.plan"), key=lambda item: item.name.lower()):
        try:
            plan = load_plan(path)
        except Exception:
            continue
        if getattr(plan, "plan_type", "") == FILE_PLAN_TYPE:
            out.append(path.resolve())
    return out


def write_batch_plans_for_paths(plan_paths: Sequence[Path]) -> Dict[Path, Dict[str, Path]]:
    by_dir: Dict[Path, List[Path]] = {}
    for plan_path in plan_paths:
        by_dir.setdefault(plan_path.parent.resolve(), []).append(plan_path.resolve())

    written: Dict[Path, Dict[str, Path]] = {}
    for source_dir, items in sorted(by_dir.items(), key=lambda entry: str(entry[0]).lower()):
        ordered = sorted({item.resolve() for item in items}, key=lambda item: item.name.lower())
        full_plan = make_batch_plan(name="full-batch", mode="full", plans=ordered, base_dir=source_dir)
        fastpass_plan = make_batch_plan(name="fastpass-batch", mode="fastpass", plans=ordered, base_dir=source_dir)
        full_path = source_dir / "full-batch.plan"
        fastpass_path = source_dir / "fastpass-batch.plan"
        save_plan(full_plan, full_path)
        save_plan(fastpass_plan, fastpass_path)
        written[source_dir] = {"full": full_path, "fastpass": fastpass_path}
    return written


def refresh_directory_support_files(*, source_dir: Path, python_exe: str, project_root: Path) -> Dict[str, Path]:
    plans = collect_file_plan_paths(source_dir)
    written = write_batch_plans_for_paths(plans).get(source_dir.resolve(), {})
    write_directory_launchers(source_dir=source_dir.resolve(), python_exe=python_exe, project_root=project_root.resolve())
    return written


def refresh_support_for_plan_paths(*, plan_paths: Iterable[Path], python_exe: str, project_root: Path) -> Dict[Path, Dict[str, Path]]:
    written = write_batch_plans_for_paths(list(plan_paths))
    for source_dir in written.keys():
        write_directory_launchers(source_dir=source_dir, python_exe=python_exe, project_root=project_root.resolve())
    return written
