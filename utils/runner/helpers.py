from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from utils.pipeline_runtime import read_command_output
from utils.plan_model import FilePlan, load_plan, resolve_file_plan
from utils.runner_state import CACHED_STAGE_MESSAGE, display_stage_plan, stage_resume_marker_exists
from utils.runner_source_info import item_inputs_changed, item_source_duration, item_source_size

from .models import QueueItem, StageState

_AV1AN_PROGRESS_JSONL_SUPPORT: Dict[str, bool] = {}

def normalize_mode(value: str) -> str:
    return "fastpass" if str(value or "").strip().lower() == "fastpass" else "full"


def av1an_encoder_name(value: str) -> str:
    return "x265" if str(value or "").strip().lower() in ("libx265", "x265") else "svt-av1"


def resolve_optional_path(raw_value: str, plan_path: Path) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (plan_path.parent / path).resolve()
    return str(path)


def bool_arg(value: bool) -> str:
    return "1" if bool(value) else "0"


def av1an_supports_progress_jsonl(av1an_exe: str) -> bool:
    key = str(av1an_exe or "").strip().lower()
    if key in _AV1AN_PROGRESS_JSONL_SUPPORT:
        return _AV1AN_PROGRESS_JSONL_SUPPORT[key]
    help_text = read_command_output([str(av1an_exe or "av1an"), "--help"], timeout=5.0)
    supported = "--progress-jsonl" in help_text
    _AV1AN_PROGRESS_JSONL_SUPPORT[key] = supported
    return supported


def item_short_snapshot(item: QueueItem) -> Dict[str, Any]:
    return {
        "mode": item.mode,
        "name": item.name,
        "plan": str(item.plan_path),
        "source": str(item.source),
        "source_size": item_source_size(item),
        "duration_seconds": item_source_duration(item),
        "workdir": str(item.workdir),
    }


def initial_stage_states(item: QueueItem) -> List[StageState]:
    states: List[StageState] = []
    for name in display_stage_plan(item):
        if not item_inputs_changed(item) and stage_resume_marker_exists(item, name):
            states.append(StageState(name=name, status="completed", message=CACHED_STAGE_MESSAGE))
        else:
            states.append(StageState(name=name))
    return states


def build_wrapper_vspipe_args(
    item: "QueueItem",
    *,
    user_vpy: str,
    pass_name: str,
) -> List[str]:
    plan = item.resolved.plan
    experimental = plan.video.experimental
    return [
        f"src={item.source}",
        f"vpy={user_vpy or ''}",
        f"rules={item.resolved.paths.crop_resize_file}",
        f"pass_name={pass_name}",
        f"source_loader={experimental.source_loader or 'ffms2'}",
        f"crop_enabled={bool_arg(experimental.crop_resize_enabled)}",
        f"plan_path={item.plan_path}",
        f"workdir={item.workdir}",
    ]


def build_queue(plan_args: List[str], cli_mode: str, mode_overrides: Dict[str, str] | None = None) -> List[QueueItem]:
    queue: List[QueueItem] = []
    seen: set[str] = set()
    visiting: set[str] = set()
    overrides = {
        str(Path(path).expanduser().resolve()).lower(): normalize_mode(mode)
        for path, mode in dict(mode_overrides or {}).items()
        if str(path or "").strip()
    }

    def visit(path: Path, inherited_mode: str) -> None:
        plan_path = path.expanduser().resolve()
        key = str(plan_path).lower()
        if key in visiting:
            raise RuntimeError(f"batch plan cycle detected at {plan_path}")
        plan = load_plan(plan_path)
        if isinstance(plan, FilePlan):
            if key in seen:
                return
            seen.add(key)
            mode = normalize_mode(overrides.get(key) or cli_mode or inherited_mode or plan.meta.mode or "full")
            queue.append(QueueItem(resolved=resolve_file_plan(plan_path), mode=mode))
            return

        visiting.add(key)
        try:
            batch_mode = normalize_mode(overrides.get(key) or cli_mode or plan.meta.mode or inherited_mode)
            for item in plan.items:
                nested = Path(item.plan).expanduser()
                if not nested.is_absolute():
                    nested = (plan_path.parent / nested).resolve()
                visit(nested, batch_mode)
        finally:
            visiting.remove(key)

    for raw in plan_args:
        visit(Path(raw), cli_mode or "")
    return queue
