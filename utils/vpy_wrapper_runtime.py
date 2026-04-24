from __future__ import annotations

import os
import re
import runpy
import sys
import ast
import keyword
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from utils.crop_resize import (
    FrameRange,
    apply_crop_resize_to_clip,
    load_chapters,
    load_crop_resize_lines,
    load_crop_resize_plan,
    resolve_selector,
)
from utils.zoned_commands import project_vpy_function_lines


SOURCE_LOADERS = ("auto", "ffms2", "bs", "lsmas")
_RX_FRAME_SELECTOR = re.compile(r"^(\d+)f(\d+)$", re.IGNORECASE)
_RX_TIME_SELECTOR = re.compile(r"^([0-9:.,]+)t([0-9:.,]+)$", re.IGNORECASE)


@dataclass(frozen=True)
class VpyFunctionRule:
    selectors: Tuple[str, ...]
    function_name: str
    raw_line: str


@dataclass(frozen=True)
class VpyFunctionPlan:
    default_function: str
    rules: Tuple[VpyFunctionRule, ...]


EMPTY_VPY_FUNCTION_PLAN = VpyFunctionPlan(default_function="", rules=())


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        for encoding in ("utf-8", "utf-8-sig", "cp1251"):
            try:
                return value.decode(encoding)
            except Exception:
                continue
        return value.decode("utf-8", errors="replace")

    text = str(value)
    stripped = text.strip()
    if (stripped.startswith("b'") or stripped.startswith('b"')) and stripped[-1:] in ("'", '"'):
        try:
            parsed = ast.literal_eval(stripped)
        except Exception:
            parsed = None
        if isinstance(parsed, bytes):
            return _coerce_text(parsed)
    return text


def _bool_arg(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = _coerce_text(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


def _arg_text(raw: Dict[str, Any], key: str, default: str = "") -> str:
    value = raw.get(key, default)
    return _coerce_text(value).strip()


def build_args(raw_globals: Dict[str, Any]) -> SimpleNamespace:
    raw = dict(raw_globals)
    source_loader = _arg_text(raw, "source_loader", "ffms2").lower()
    if source_loader in ("bestsource", "best-source"):
        source_loader = "bs"
    if source_loader in ("lsmash", "lwlibavsource"):
        source_loader = "lsmas"
    if source_loader not in SOURCE_LOADERS:
        source_loader = "ffms2"

    src = _arg_text(raw, "src")
    vpy = _arg_text(raw, "vpy")
    rules = _arg_text(raw, "rules")
    plan_path = _arg_text(raw, "plan_path")
    workdir = _arg_text(raw, "workdir")
    pass_name = _arg_text(raw, "pass_name", "main") or "main"

    return SimpleNamespace(
        src=src,
        vpy=vpy,
        rules=rules,
        pass_name=pass_name,
        source_loader=source_loader,
        crop_enabled=_bool_arg(raw.get("crop_enabled"), False),
        plan_path=plan_path,
        workdir=workdir,
        raw=raw,
    )


def _load_default_source(args: SimpleNamespace) -> Any:
    import vapoursynth as vs  # type: ignore

    src_path = Path(args.src).expanduser()
    if not src_path.exists():
        raise RuntimeError(f"wrapper.vpy source file not found: {src_path}")

    core = vs.core
    providers = []
    requested = str(args.source_loader or "ffms2").strip().lower()
    if requested == "auto":
        if hasattr(core, "ffms2"):
            providers.append(("ffms2", core.ffms2.Source))
        if hasattr(core, "bs"):
            providers.append(("bs", core.bs.VideoSource))
        if hasattr(core, "lsmas"):
            providers.append(("lsmas", core.lsmas.LWLibavSource))
    elif requested == "ffms2" and hasattr(core, "ffms2"):
        providers.append(("ffms2", core.ffms2.Source))
    elif requested == "bs" and hasattr(core, "bs"):
        providers.append(("bs", core.bs.VideoSource))
    elif requested == "lsmas" and hasattr(core, "lsmas"):
        providers.append(("lsmas", core.lsmas.LWLibavSource))

    if not providers:
        raise RuntimeError(f"VapourSynth source loader is not available: {requested}")

    last_error: Optional[Exception] = None
    for _name, loader in providers:
        try:
            return loader(str(src_path))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to load source with {requested}: {last_error}")


def _load_user_vpy(args: SimpleNamespace) -> Dict[str, Any]:
    if not str(args.vpy or "").strip():
        return {}
    vpy_path = Path(args.vpy).expanduser()
    if not vpy_path.exists():
        raise RuntimeError(f"user vpy not found: {vpy_path}")

    old_cwd = os.getcwd()
    old_sys_path = list(sys.path)
    try:
        os.chdir(str(vpy_path.parent))
        if str(vpy_path.parent) not in sys.path:
            sys.path.insert(0, str(vpy_path.parent))
        return runpy.run_path(
            str(vpy_path),
            init_globals={
                "__file__": str(vpy_path),
                "__name__": "__vapoursynth_user__",
                "args": args,
                "src": args.src,
            },
        )
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path


def _create_pipe(args: SimpleNamespace, user_ns: Dict[str, Any]) -> Any:
    create_pipe = user_ns.get("create_pipe")
    if callable(create_pipe):
        return create_pipe(args)
    return _load_default_source(args)


def _strip_rule_line(raw: str) -> str:
    line = str(raw or "").strip()
    if not line or line.startswith("#") or line.startswith("//"):
        return ""
    return line


def _split_rule_line(line: str) -> Tuple[str, str]:
    if " - " in line:
        left, right = line.split(" - ", 1)
    else:
        parts = line.split(None, 2)
        if len(parts) < 3 or parts[1] != "-":
            raise ValueError("expected '<selector> - <function>'")
        left, right = parts[0], parts[2]
    left = left.strip()
    right = right.strip()
    if not left:
        raise ValueError("missing selector")
    if not right:
        raise ValueError("missing function")
    return left, right


def _is_function_name(value: str) -> bool:
    text = str(value or "").strip()
    return bool(text and text.isidentifier() and not keyword.iskeyword(text))


def _parse_vpy_function_lines(lines: Iterable[str]) -> VpyFunctionPlan:
    default_function = ""
    rules: List[VpyFunctionRule] = []

    for raw in project_vpy_function_lines(lines):
        line = _strip_rule_line(raw)
        if not line:
            continue
        try:
            selector_text, function_name = _split_rule_line(line)
        except ValueError:
            continue
        if not _is_function_name(function_name):
            continue
        if selector_text == "@default":
            default_function = function_name
            continue
        selectors = tuple(part.strip() for part in selector_text.split(",") if part.strip())
        if not selectors:
            raise ValueError(f"no selectors in line: {line}")
        rules.append(VpyFunctionRule(selectors=selectors, function_name=function_name, raw_line=line))

    if not default_function and not rules:
        return EMPTY_VPY_FUNCTION_PLAN
    return VpyFunctionPlan(default_function=default_function, rules=tuple(rules))


def _load_vpy_function_plan(args: SimpleNamespace) -> VpyFunctionPlan:
    rules_path = Path(args.rules).expanduser() if str(args.rules or "").strip() else None
    if rules_path is None or not rules_path.exists():
        return EMPTY_VPY_FUNCTION_PLAN
    return _parse_vpy_function_lines(load_crop_resize_lines(rules_path))


def _fps_from_clip(clip: Any) -> Fraction:
    num = int(getattr(clip, "fps_num", 0) or 0)
    den = int(getattr(clip, "fps_den", 0) or 0)
    if num > 0 and den > 0:
        return Fraction(num, den)
    return Fraction(24000, 1001)


def _resolve_vpy_function_rule_ranges(
    plan: VpyFunctionPlan,
    *,
    clip: Any,
    source: Optional[Path] = None,
) -> List[List[FrameRange]]:
    total_frames = int(getattr(clip, "num_frames", 0) or 0)
    fps = _fps_from_clip(clip)
    chapters = []
    needs_chapters = any(
        not _RX_FRAME_SELECTOR.match(selector.strip()) and not _RX_TIME_SELECTOR.match(selector.strip())
        for rule in plan.rules
        for selector in rule.selectors
    )
    if needs_chapters:
        if source is None:
            raise ValueError("chapter selectors require source path")
        chapters = load_chapters(source)

    resolved: List[List[FrameRange]] = []
    for rule in plan.rules:
        ranges: List[FrameRange] = []
        for selector in rule.selectors:
            ranges.extend(resolve_selector(selector, fps=fps, total_frames=total_frames, chapters=chapters))
        resolved.append(ranges)
    return resolved


def _format_name(clip: Any) -> str:
    fmt = getattr(clip, "format", None)
    name = getattr(fmt, "name", None)
    if name:
        return str(name)
    fmt_id = getattr(fmt, "id", None)
    return str(fmt_id) if fmt_id is not None else ""


def _validate_function_clip(base: Any, variant: Any, function_name: str) -> None:
    base_width = int(getattr(base, "width", 0) or 0)
    base_height = int(getattr(base, "height", 0) or 0)
    variant_width = int(getattr(variant, "width", 0) or 0)
    variant_height = int(getattr(variant, "height", 0) or 0)
    if variant_width != base_width or variant_height != base_height:
        raise RuntimeError(
            f"wrapper.vpy function {function_name!r} produced invalid geometry: "
            f"{variant_width}x{variant_height}, expected {base_width}x{base_height}"
        )

    base_frames = int(getattr(base, "num_frames", 0) or 0)
    variant_frames = int(getattr(variant, "num_frames", 0) or 0)
    if base_frames and variant_frames and variant_frames != base_frames:
        raise RuntimeError(
            f"wrapper.vpy function {function_name!r} produced invalid frame count: "
            f"{variant_frames}, expected {base_frames}"
        )

    base_format = _format_name(base)
    variant_format = _format_name(variant)
    if base_format and variant_format and variant_format != base_format:
        raise RuntimeError(
            f"wrapper.vpy function {function_name!r} produced invalid format: "
            f"{variant_format}, expected {base_format}"
        )


def _run_named_pipeline(
    args: SimpleNamespace,
    user_ns: Dict[str, Any],
    clip: Any,
    function_name: str,
) -> Any:
    function = user_ns.get(function_name)
    if callable(function):
        return function(args, clip)
    if function_name == "pipeline":
        return clip
    raise RuntimeError(f"wrapper.vpy function not found: {function_name}")


def _run_pipeline(args: SimpleNamespace, user_ns: Dict[str, Any], clip: Any) -> Any:
    pipeline = user_ns.get("pipeline")
    if callable(pipeline):
        return pipeline(args, clip)
    return clip


def _run_pipeline_zones(args: SimpleNamespace, user_ns: Dict[str, Any], clip: Any) -> Any:
    if not str(args.vpy or "").strip():
        return _run_pipeline(args, user_ns, clip)

    plan = _load_vpy_function_plan(args)
    if not plan.default_function and not plan.rules:
        return _run_pipeline(args, user_ns, clip)

    import vapoursynth as vs  # type: ignore

    base_function = plan.default_function or "pipeline"
    base = _run_named_pipeline(args, user_ns, clip, base_function)

    total_frames = int(getattr(base, "num_frames", 0) or 0)
    variant_indexes = [0] * total_frames
    variants: List[Any] = [base]
    variant_by_function: Dict[str, int] = {base_function: 0}

    rule_ranges = _resolve_vpy_function_rule_ranges(
        plan,
        clip=base,
        source=Path(args.src).expanduser() if str(args.src or "").strip() else None,
    )

    for rule, ranges in zip(plan.rules, rule_ranges):
        variant_index = variant_by_function.get(rule.function_name)
        if variant_index is None:
            variant = _run_named_pipeline(args, user_ns, clip, rule.function_name)
            _validate_function_clip(base, variant, rule.function_name)
            variant_index = len(variants)
            variant_by_function[rule.function_name] = variant_index
            variants.append(variant)
        for frame_range in ranges:
            for frame_no in range(frame_range.start, frame_range.end):
                variant_indexes[frame_no] = variant_index

    if len(variants) == 1:
        return base

    def select_frame(n: int) -> Any:
        return variants[variant_indexes[n]]

    return vs.core.std.FrameEval(base, select_frame)


def _run_pre_pipeline(args: SimpleNamespace, user_ns: Dict[str, Any], clip: Any) -> Any:
    pre_pipeline = user_ns.get("pre_pipeline")
    if callable(pre_pipeline):
        return pre_pipeline(args, clip)
    return clip

def _run_post_pipeline(args: SimpleNamespace, user_ns: Dict[str, Any], clip: Any) -> Any:
    post_pipeline = user_ns.get("post_pipeline")
    if callable(post_pipeline):
        return post_pipeline(args, clip)
    return clip


def build_clip(raw_globals: Dict[str, Any]) -> Any:
    args = build_args(raw_globals)
    user_ns = _load_user_vpy(args)
    clip = _create_pipe(args, user_ns)

    crop_plan = None
    if args.crop_enabled:
        rules_path = Path(args.rules).expanduser() if str(args.rules or "").strip() else None
        if rules_path is not None and rules_path.exists():
            crop_plan = load_crop_resize_plan(rules_path)
            if crop_plan.enabled:
                clip = apply_crop_resize_to_clip(clip, crop_plan, source=Path(args.src).expanduser())
                if int(clip.width) != int(crop_plan.target_width) or int(clip.height) != int(crop_plan.target_height):
                    raise RuntimeError(
                        "wrapper.vpy dynamic crop/resize produced invalid geometry: "
                        f"{clip.width}x{clip.height}, expected {crop_plan.target_width}x{crop_plan.target_height}"
                    )

    clip = _run_pre_pipeline(args, user_ns, clip)
    clip = _run_pipeline_zones(args, user_ns, clip)
    clip = _run_post_pipeline(args, user_ns, clip)
    return clip


__all__ = ["build_args", "build_clip"]
