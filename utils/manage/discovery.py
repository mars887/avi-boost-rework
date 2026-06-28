from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from utils import workdir_layout as layout
from utils.plan import BatchPlan, FilePlan, load_plan, resolve_batch_plan, resolve_paths
from utils.zoned_commands import (
    LEGACY_CROP_RESIZE_COMMAND_NAME,
    LEGACY_ZONE_COMMAND_NAME,
    ZONED_COMMAND_NAME,
)

from utils.manage.context import WorkdirContext, context_from_plan, context_from_workdir

PLAN_SUFFIX = ".plan"
BATCH_PLAN_NAMES = ("full-batch.plan", "fastpass-batch.plan")


@dataclass(frozen=True)
class WorkdirRef:
    workdir: Path
    plan_path: Optional[Path] = None
    batch_plan_path: Optional[Path] = None
    mode: str = ""

    @property
    def label(self) -> str:
        return self.workdir.name


def is_workdir(path: Path) -> bool:
    path = Path(path)
    if not path.is_dir():
        return False
    return (
        layout.meta_dir(path).is_dir()
        or layout.state_dir(path).is_dir()
        or layout.video_dir(path).is_dir()
        or (path / ZONED_COMMAND_NAME).is_file()
        or (path / LEGACY_ZONE_COMMAND_NAME).is_file()
        or (path / LEGACY_CROP_RESIZE_COMMAND_NAME).is_file()
    )


def find_plan_files(folder: Path) -> List[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        (entry for entry in folder.iterdir() if entry.is_file() and entry.suffix.lower() == PLAN_SUFFIX),
        key=lambda entry: entry.name.lower(),
    )


def _refs_from_plan(plan_path: Path, *, mode: str = "") -> List[WorkdirRef]:
    plan_path = Path(plan_path).absolute()
    plan = load_plan(plan_path)
    if isinstance(plan, FilePlan):
        resolved = resolve_paths(plan, plan_path)
        effective_mode = mode or str(plan.meta.mode or "").strip().lower()
        return [WorkdirRef(workdir=resolved.workdir, plan_path=plan_path, mode=effective_mode)]
    if isinstance(plan, BatchPlan):
        batch_mode = mode or str(plan.meta.mode or "").strip().lower()
        refs: List[WorkdirRef] = []
        for resolved in resolve_batch_plan(plan_path):
            refs.append(
                WorkdirRef(
                    workdir=resolved.paths.workdir,
                    plan_path=resolved.paths.plan_path,
                    batch_plan_path=plan_path,
                    mode=batch_mode,
                )
            )
        return refs
    return []


def resolve_argument_to_refs(path: Path) -> List[WorkdirRef]:
    path = Path(path).expanduser().absolute()
    if path.is_file():
        if path.suffix.lower() != PLAN_SUFFIX:
            raise RuntimeError(f"unsupported manage argument (expected .plan or directory): {path}")
        return _refs_from_plan(path)
    if not path.is_dir():
        raise RuntimeError(f"manage argument does not exist: {path}")

    if is_workdir(path):
        plan_candidates = [
            entry
            for entry in find_plan_files(path.parent)
            if entry.stem.lower() == path.name.lower()
        ]
        plan_path = plan_candidates[0] if plan_candidates else None
        if plan_path is not None:
            try:
                refs = _refs_from_plan(plan_path)
            except Exception:
                refs = []
            for ref in refs:
                if _same_path(ref.workdir, path):
                    return [ref]
        return [WorkdirRef(workdir=path)]

    refs: List[WorkdirRef] = []
    plans = find_plan_files(path)
    batch_plans = [entry for entry in plans if entry.name.lower() in BATCH_PLAN_NAMES]
    file_plans = [entry for entry in plans if entry.name.lower() not in BATCH_PLAN_NAMES]
    for plan_path in file_plans:
        try:
            refs.extend(_refs_from_plan(plan_path))
        except Exception:
            continue
    if not refs:
        for plan_path in batch_plans:
            try:
                refs.extend(_refs_from_plan(plan_path))
            except Exception:
                continue
    if not refs:
        raise RuntimeError(f"no workdir or .plan files found in: {path}")
    return refs


def _same_path(left: Path, right: Path) -> bool:
    return str(Path(left).absolute()).lower() == str(Path(right).absolute()).lower()


def discover_workdirs(args: Sequence[str]) -> List[WorkdirRef]:
    raw_args = [str(arg).strip() for arg in args if str(arg).strip()]
    refs: List[WorkdirRef] = []
    if raw_args:
        for raw in raw_args:
            refs.extend(resolve_argument_to_refs(Path(raw)))
    else:
        refs.extend(resolve_argument_to_refs(Path.cwd()))

    unique: List[WorkdirRef] = []
    seen: set[str] = set()
    for ref in refs:
        key = str(ref.workdir.absolute()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(ref)
    if not unique:
        raise RuntimeError("no workdirs discovered")
    return unique


def load_ref(ref: WorkdirRef) -> WorkdirContext:
    if ref.plan_path is not None and Path(ref.plan_path).exists():
        return context_from_plan(ref.plan_path, mode=ref.mode, batch_plan_path=ref.batch_plan_path)
    return context_from_workdir(ref.workdir, mode=ref.mode)


__all__ = [
    "BATCH_PLAN_NAMES",
    "WorkdirRef",
    "discover_workdirs",
    "find_plan_files",
    "is_workdir",
    "load_ref",
    "resolve_argument_to_refs",
]
