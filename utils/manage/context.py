from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.plan import (
    FilePlan,
    PlanMeta,
    PlanPaths,
    ResolvedFilePlan,
    ResolvedPaths,
    VideoPlan,
    load_plan,
    resolve_paths,
)
from utils import workdir_layout as layout
from utils.runner.stage_bank import StageBankConfig, load_stage_bank_config
from utils.runner_state import display_stage_plan
from utils.zoned_commands import zoned_command_path

MODE_FULL = "full"
MODE_FASTPASS = "fastpass"


@dataclass(frozen=True)
class RunnerItemAdapter:
    """Duck-typed stand-in for runner ``QueueItem``.

    Carries the same attributes that ``utils.runner_state`` helpers read
    (``resolved``/``mode``/``workdir``/``source``/``plan_path``) so manage can
    reuse stage plan/marker/artifact logic without copying it.
    """

    resolved: ResolvedFilePlan
    mode: str
    workdir: Path
    source: Path
    plan_path: Path

    @property
    def source_dir(self) -> str:
        return str(self.source.parent)

    @property
    def name(self) -> str:
        return self.resolved.plan.meta.name or self.source.stem


@dataclass
class WorkdirContext:
    source: Optional[Path]
    plan_path: Optional[Path]
    batch_plan_path: Optional[Path]
    workdir: Path
    zone_file: Optional[Path]
    mode: str
    resolved_plan: Optional[ResolvedFilePlan]
    stage_names: List[str]
    stage_bank: StageBankConfig
    warnings: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        if self.resolved_plan is not None and self.resolved_plan.plan.meta.name:
            return self.resolved_plan.plan.meta.name
        if self.source is not None:
            return self.source.stem
        return self.workdir.name

    @property
    def meta_dir(self) -> Path:
        return layout.meta_dir(self.workdir)

    @property
    def video_dir(self) -> Path:
        return layout.video_dir(self.workdir)

    def has_video_edit(self) -> bool:
        if self.resolved_plan is not None:
            return self.resolved_plan.has_video_edit()
        return self.video_dir.is_dir()


def make_runner_item(ctx: WorkdirContext, *, mode: Optional[str] = None) -> RunnerItemAdapter:
    resolved = ctx.resolved_plan or _synthetic_resolved_plan(ctx.workdir, ctx.source, ctx.zone_file)
    source = ctx.source or resolved.paths.source
    effective_mode = str(mode if mode is not None else (ctx.mode or MODE_FULL)).strip().lower() or MODE_FULL
    return RunnerItemAdapter(
        resolved=resolved,
        mode=effective_mode,
        workdir=ctx.workdir,
        source=source,
        plan_path=ctx.plan_path or resolved.paths.plan_path,
    )


def load_json_file(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_meta_paths(workdir: Path) -> Dict[str, str]:
    """Best-effort recovery of source/plan/mode hints from 00_meta files."""
    out: Dict[str, str] = {}
    info = load_json_file(layout.source_info_path(workdir))
    if isinstance(info, dict):
        if str(info.get("source") or "").strip():
            out["source"] = str(info["source"])
        if str(info.get("plan") or "").strip():
            out["plan"] = str(info["plan"])
    state = load_json_file(layout.runner_state_path(workdir))
    if isinstance(state, dict):
        for key in ("source", "plan", "mode"):
            value = str(state.get(key) or "").strip()
            if value and key not in out:
                out[key] = value
    if "mode" not in out or "source" not in out or "plan" not in out:
        last = _last_runner_event(layout.runner_events_path(workdir))
        if last:
            if "mode" not in out and str(last.get("mode") or "").strip():
                out["mode"] = str(last["mode"])
            if "source" not in out and str(last.get("source") or "").strip():
                out["source"] = str(last["source"])
            if "plan" not in out and str(last.get("plan") or "").strip():
                out["plan"] = str(last["plan"])
    return {key: value for key, value in out.items() if str(value).strip()}


def _last_runner_event(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    last: Optional[Dict[str, Any]] = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    last = payload
    except Exception:
        return None
    return last


def _guess_mode(workdir: Path) -> str:
    preview_scenes = layout.stage4_scenes(workdir, MODE_FASTPASS)
    if preview_scenes.exists() and not layout.mainpass_dir(workdir).is_dir():
        return MODE_FASTPASS
    return MODE_FULL


def _synthetic_file_plan(workdir: Path, source: Optional[Path]) -> FilePlan:
    video_dir = layout.video_dir(workdir)
    plan = FilePlan(
        meta=PlanMeta(name=Path(workdir).name, created_by="manage (synthetic)"),
        paths=PlanPaths(source=str(source) if source else ""),
        video=VideoPlan(track_id=0, action="edit" if video_dir.is_dir() else "copy"),
    )
    if layout.psd_dir(workdir).is_dir():
        plan.video.primary.scene_detection = "psd"
    return plan


def _synthetic_resolved_plan(
    workdir: Path,
    source: Optional[Path],
    zone_file: Optional[Path],
) -> ResolvedFilePlan:
    workdir = Path(workdir)
    guessed_source = source or _guess_source(workdir)
    plan = _synthetic_file_plan(workdir, guessed_source)
    plan_path = workdir.parent / f"{guessed_source.stem}.plan" if guessed_source else workdir / "_synthetic.plan"
    return ResolvedFilePlan(
        plan=plan,
        paths=ResolvedPaths(
            plan_path=plan_path,
            source=guessed_source or (workdir.parent / f"{workdir.name}.mkv"),
            workdir=workdir,
            zone_file=zone_file or zoned_command_path(workdir),
            crop_resize_file=zone_file or zoned_command_path(workdir),
        ),
    )


def _guess_source(workdir: Path) -> Optional[Path]:
    hints = read_meta_paths(workdir)
    raw = hints.get("source")
    if raw:
        return Path(raw)
    for suffix in (".mkv", ".mp4"):
        candidate = workdir.parent / f"{workdir.name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _paths_equal(left: Path, right: Path) -> bool:
    try:
        return str(Path(left).resolve()).lower() == str(Path(right).resolve()).lower()
    except Exception:
        return str(left).lower() == str(right).lower()


def context_from_plan(
    plan_path: Path,
    *,
    mode: str = "",
    batch_plan_path: Optional[Path] = None,
    stage_bank: Optional[StageBankConfig] = None,
) -> WorkdirContext:
    plan_path = Path(plan_path).absolute()
    plan = load_plan(plan_path)
    if not isinstance(plan, FilePlan):
        raise RuntimeError(f"{plan_path} is not a file plan")
    resolved = ResolvedFilePlan(plan=plan, paths=resolve_paths(plan, plan_path))
    effective_mode = (
        str(mode or "").strip().lower()
        or str(plan.meta.mode or "").strip().lower()
        or read_meta_paths(resolved.paths.workdir).get("mode", "")
        or MODE_FULL
    )
    bank = stage_bank or load_stage_bank_config()
    ctx = WorkdirContext(
        source=resolved.paths.source,
        plan_path=plan_path,
        batch_plan_path=Path(batch_plan_path).absolute() if batch_plan_path else None,
        workdir=resolved.paths.workdir,
        zone_file=resolved.paths.zone_file,
        mode=effective_mode,
        resolved_plan=resolved,
        stage_names=[],
        stage_bank=bank,
    )
    ctx.stage_names = display_stage_plan(make_runner_item(ctx, mode=MODE_FULL))
    if not resolved.paths.source.exists():
        ctx.warnings.append(f"source file not found: {resolved.paths.source}")
    return ctx


def context_from_workdir(
    workdir: Path,
    *,
    mode: str = "",
    stage_bank: Optional[StageBankConfig] = None,
) -> WorkdirContext:
    workdir = Path(workdir).absolute()
    hints = read_meta_paths(workdir)

    plan_path = Path(hints["plan"]) if hints.get("plan") else None
    if plan_path is not None and plan_path.exists():
        try:
            ctx = context_from_plan(plan_path, mode=mode or hints.get("mode", ""), stage_bank=stage_bank)
        except Exception as exc:
            ctx = None  # fall through to best-effort context
            plan_error = str(exc)
        if ctx is not None:
            if not _paths_equal(ctx.workdir, workdir):
                ctx.warnings.append(
                    f"plan {plan_path} resolves workdir to {ctx.workdir}, but {workdir} was requested; using {workdir}"
                )
                ctx.workdir = workdir
                ctx.stage_names = display_stage_plan(make_runner_item(ctx, mode=MODE_FULL))
            return ctx
    else:
        plan_error = ""

    source = Path(hints["source"]) if hints.get("source") else _guess_source(workdir)
    bank = stage_bank or load_stage_bank_config()
    effective_mode = (
        str(mode or "").strip().lower()
        or hints.get("mode", "")
        or _guess_mode(workdir)
    )
    ctx = WorkdirContext(
        source=source,
        plan_path=None,
        batch_plan_path=None,
        workdir=workdir,
        zone_file=zoned_command_path(workdir),
        mode=effective_mode,
        resolved_plan=None,
        stage_names=[],
        stage_bank=bank,
    )
    ctx.warnings.append("workdir opened without .plan; context restored best-effort from 00_meta")
    if plan_path is not None and not plan_path.exists():
        ctx.warnings.append(f"plan recorded in 00_meta is missing: {plan_path}")
    if plan_error:
        ctx.warnings.append(f"failed to load plan {plan_path}: {plan_error}")
    ctx.stage_names = display_stage_plan(make_runner_item(ctx, mode=MODE_FULL))
    return ctx


__all__ = [
    "MODE_FASTPASS",
    "MODE_FULL",
    "RunnerItemAdapter",
    "WorkdirContext",
    "context_from_plan",
    "context_from_workdir",
    "load_json_file",
    "make_runner_item",
    "read_meta_paths",
]
