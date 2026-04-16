from __future__ import annotations

import os
import runpy
import sys
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

from utils.crop_resize import apply_crop_resize_to_clip, load_crop_resize_plan


SOURCE_LOADERS = ("auto", "ffms2", "bs", "lsmas")


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


def _run_pipeline(args: SimpleNamespace, user_ns: Dict[str, Any], clip: Any) -> Any:
    pipeline = user_ns.get("pipeline")
    if callable(pipeline):
        return pipeline(args, clip)
    return clip


def _run_pre_pipeline(args: SimpleNamespace, user_ns: Dict[str, Any], clip: Any) -> Any:
    pre_pipeline = user_ns.get("pre_pipeline")
    if callable(pre_pipeline):
        return pre_pipeline(args, clip)
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
    clip = _run_pipeline(args, user_ns, clip)
    return clip


__all__ = ["build_args", "build_clip"]
