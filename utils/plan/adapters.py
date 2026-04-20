from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from utils.plan.io import load_file_plan, plan_path_for_source, resolve_paths
from utils.plan.types import (
    DEFAULT_QUALITY,
    DEFAULT_SCENE_DETECTION,
    DEFAULT_CHUNK_ORDER,
    DEFAULT_SOURCE_LOADER,
    AudioPlan,
    FilePlan,
    PlanMeta,
    PlanPaths,
    SubPlan,
    VideoDetails,
    VideoExperimental,
    VideoPlan,
    VideoPrimary,
    coerce_scalar,
    format_value,
    normalize_encoder,
    normalize_chunk_order,
    normalize_source_loader,
    normalize_track_type,
    params_map_to_tokens,
    parse_bool_value,
    to_float,
)


def gui_defaults_from_file_plan(plan: FilePlan) -> Dict[str, Any]:
    video = plan.video
    primary = video.primary
    details = video.details
    experimental = video.experimental
    fastpass_tokens = params_map_to_tokens(video.fastpass_params)
    mainpass_tokens = params_map_to_tokens(video.mainpass_params)
    fastpass_tokens.extend(["--crf", format_value(primary.quality)])
    mainpass_tokens.extend(["--crf", format_value(primary.quality)])
    if primary.fastpass_preset:
        fastpass_tokens.extend(["--preset", str(primary.fastpass_preset)])
    if primary.preset:
        mainpass_tokens.extend(["--preset", str(primary.preset)])
    return {
        "params": subprocess.list2cmdline([str(x) for x in fastpass_tokens]),
        "last_params": subprocess.list2cmdline([str(x) for x in mainpass_tokens]),
        "encoder": primary.encoder,
        "scene_detection": primary.scene_detection,
        "chunk_order": primary.chunk_order,
        "encoder_path": primary.encoder_path,
        "quality": primary.quality,
        "fastpass_preset": primary.fastpass_preset,
        "preset": primary.preset,
        "fastpass_workers": str(primary.fastpass_workers),
        "mainpass_workers": str(primary.mainpass_workers),
        "no_fastpass": primary.no_fastpass,
        "fastpass_hdr": primary.fastpass_hdr,
        "strict_sdr_8bit": primary.strict_sdr_8bit,
        "no_dolby_vision": primary.no_dolby_vision,
        "no_hdr10plus": primary.no_hdr10plus,
        "ab_multiplier": format_value(primary.ab_multiplier),
        "ab_pos_dev": format_value(primary.ab_pos_dev),
        "ab_neg_dev": format_value(primary.ab_neg_dev),
        "ab_pos_multiplier": str(primary.ab_pos_multiplier or ""),
        "ab_neg_multiplier": str(primary.ab_neg_multiplier or ""),
        "avg_func": str(primary.avg_func or ""),
        "attach_encode_info": primary.attach_encode_info,
        "fastpass": details.fastpass_filter,
        "mainpass": details.mainpass_filter,
        "main_vpy": details.main_vpy,
        "fast_vpy": details.fast_vpy,
        "proxy_vpy": details.proxy_vpy,
        "vpy_wrapper": experimental.vpy_wrapper,
        "source_loader": experimental.source_loader,
        "crop_resize_enabled": experimental.crop_resize_enabled,
        "note": details.note,
    }


def gui_settings_from_file_plan(plan: FilePlan) -> List[Dict[str, Any]]:
    settings: List[Dict[str, Any]] = []
    video = plan.video
    settings.append(
        {
            "id": str(video.track_id),
            "type": "video",
            "mode": video.action.upper(),
            # Video params are edited through the global Video tab, not via per-track rules.
            "params": "",
            "last_params": "",
            "bitrate": "",
            "channels": "",
            "name": "",
            "lang": video.source_lang,
            "default": None,
        }
    )
    for track in plan.audio:
        settings.append(
            {
                "id": str(track.track_id),
                "type": "audio",
                "mode": track.action.upper(),
                "params": "",
                "last_params": "",
                "bitrate": str(track.bitrate_kbps),
                "channels": str(track.channels),
                "name": track.name,
                "lang": track.lang or track.source_lang,
                "default": bool(track.default),
            }
        )
    for track in plan.sub:
        settings.append(
            {
                "id": str(track.track_id),
                "type": "sub",
                "mode": track.action.upper(),
                "params": "",
                "last_params": "",
                "bitrate": "",
                "channels": "",
                "name": track.name,
                "lang": track.lang or track.source_lang,
                "default": bool(track.default),
            }
        )
    return settings


def _parse_video_config(raw_config: Dict[str, Any], default_quality: float) -> tuple[float, str, str, Dict[str, Any], Dict[str, Any]]:
    fastpass_raw = dict(raw_config.get("fastpass_params") or {})
    mainpass_raw = dict(raw_config.get("mainpass_params") or {})

    fastpass_quality = fastpass_raw.pop("--crf", None)
    mainpass_quality = mainpass_raw.pop("--crf", None)
    fastpass_preset = str(raw_config.get("fastpass_preset") or fastpass_raw.pop("--preset", "") or "")
    preset = str(raw_config.get("preset") or mainpass_raw.pop("--preset", "") or "")
    quality = to_float(raw_config.get("quality") or mainpass_quality or fastpass_quality, default_quality)

    fastpass_params = {str(key): coerce_scalar(value) for key, value in fastpass_raw.items() if str(key).startswith("--")}
    mainpass_params = {str(key): coerce_scalar(value) for key, value in mainpass_raw.items() if str(key).startswith("--")}
    return quality, fastpass_preset, preset, fastpass_params, mainpass_params


def file_plan_from_gui_result(
    *,
    source: Path,
    defaults: Dict[str, Any],
    track_results: Sequence[Dict[str, Any]],
    plan_path: Optional[Path] = None,
) -> FilePlan:
    src = Path(source).absolute()
    plan_file = Path(plan_path).absolute() if plan_path is not None else plan_path_for_source(src).absolute()
    existing_plan = load_file_plan(plan_file) if plan_file.exists() else None

    video_result = next((track for track in track_results if normalize_track_type(track.get("type")) == "video"), None)
    if video_result is None:
        raise RuntimeError(f"GUI result has no video track for {src}")

    video_primary = VideoPrimary(
        encoder=normalize_encoder(defaults.get("encoder")),
        scene_detection=str(defaults.get("scene_detection") or DEFAULT_SCENE_DETECTION),
        quality=float(defaults.get("quality") or DEFAULT_QUALITY),
        chunk_order=normalize_chunk_order(defaults.get("chunk_order") or DEFAULT_CHUNK_ORDER),
        encoder_path=str(defaults.get("encoder_path") or ""),
        fastpass_preset=str(defaults.get("fastpass_preset") or ""),
        preset=str(defaults.get("preset") or ""),
        fastpass_workers=int(defaults.get("fastpass_workers") or 8),
        mainpass_workers=int(defaults.get("mainpass_workers") or 8),
        no_fastpass=bool(defaults.get("no_fastpass")),
        fastpass_hdr=bool(defaults.get("fastpass_hdr", True)),
        strict_sdr_8bit=bool(defaults.get("strict_sdr_8bit")),
        no_dolby_vision=bool(defaults.get("no_dolby_vision")),
        no_hdr10plus=bool(defaults.get("no_hdr10plus")),
        attach_encode_info=bool(defaults.get("attach_encode_info")),
        ab_multiplier=float(defaults.get("ab_multiplier") or 0.7),
        ab_pos_dev=float(defaults.get("ab_pos_dev") or 5.0),
        ab_neg_dev=float(defaults.get("ab_neg_dev") or 4.0),
        ab_pos_multiplier=str(defaults.get("ab_pos_multiplier") or ""),
        ab_neg_multiplier=str(defaults.get("ab_neg_multiplier") or ""),
        avg_func=str(
            defaults.get("avg_func")
            or defaults.get("avgFunc")
            or defaults.get("ab_avg_func")
            or defaults.get("abAvgFunc")
            or ""
        ),
    )
    video_details = VideoDetails(
        fastpass_filter=str(defaults.get("fastpass_filter") or ""),
        mainpass_filter=str(defaults.get("mainpass_filter") or ""),
        main_vpy=str(defaults.get("main_vpy") or ""),
        fast_vpy=str(defaults.get("fast_vpy") or ""),
        proxy_vpy=str(defaults.get("proxy_vpy") or ""),
        note=str(defaults.get("note") or ""),
    )
    video_experimental = VideoExperimental(
        vpy_wrapper=parse_bool_value(defaults.get("vpy_wrapper"), False),
        source_loader=normalize_source_loader(defaults.get("source_loader") or DEFAULT_SOURCE_LOADER),
        crop_resize_enabled=parse_bool_value(defaults.get("crop_resize_enabled"), False),
    )

    video_config = dict(video_result.get("videoConfig") or {})
    quality, fastpass_preset, preset, fastpass_params, mainpass_params = _parse_video_config(
        video_config,
        video_primary.quality,
    )
    video_primary.quality = quality
    video_primary.fastpass_preset = fastpass_preset
    video_primary.preset = preset

    video_plan = VideoPlan(
        track_id=int(video_result.get("trackId")),
        source_name=str(video_result.get("origName") or ""),
        source_lang=str(video_result.get("origLang") or ""),
        action=str(video_result.get("trackStatus") or "EDIT").lower(),
        primary=video_primary,
        details=video_details,
        experimental=video_experimental,
        fastpass_params=fastpass_params,
        mainpass_params=mainpass_params,
    )

    audio_plans: List[AudioPlan] = []
    sub_plans: List[SubPlan] = []
    for item in track_results:
        track_type = normalize_track_type(item.get("type"))
        if track_type == "audio":
            mux = item.get("trackMux") or {}
            param = item.get("trackParam") or {}
            audio_plans.append(
                AudioPlan(
                    track_id=int(item.get("trackId")),
                    source_name=str(item.get("origName") or ""),
                    source_lang=str(item.get("origLang") or ""),
                    action=str(item.get("trackStatus") or "COPY").lower(),
                    name=str(mux.get("name") or ""),
                    lang=str(mux.get("lang") or item.get("origLang") or ""),
                    default=parse_bool_value(mux.get("default"), False),
                    forced=parse_bool_value(mux.get("forced"), False),
                    bitrate_kbps=int(param.get("bitrate") or 128),
                    channels=int(param.get("channels") or 2),
                )
            )
        elif track_type == "sub":
            mux = item.get("trackMux") or {}
            sub_plans.append(
                SubPlan(
                    track_id=int(item.get("trackId")),
                    source_name=str(item.get("origName") or ""),
                    source_lang=str(item.get("origLang") or ""),
                    action=str(item.get("trackStatus") or "COPY").lower(),
                    name=str(mux.get("name") or ""),
                    lang=str(mux.get("lang") or item.get("origLang") or ""),
                    default=parse_bool_value(mux.get("default"), False),
                    forced=parse_bool_value(mux.get("forced"), False),
                )
            )

    meta_name = existing_plan.meta.name if existing_plan is not None else src.stem
    meta_mode = existing_plan.meta.mode if existing_plan is not None else ""
    meta_created_by = existing_plan.meta.created_by if existing_plan is not None else "track_config_gui.py"
    return FilePlan(
        meta=PlanMeta(name=meta_name, created_by=meta_created_by, mode=meta_mode),
        paths=PlanPaths(source=_relative_source_for_plan(src, plan_file)),
        video=video_plan,
        audio=audio_plans,
        sub=sub_plans,
    )


def _relative_source_for_plan(source: Path, plan_file: Path) -> str:
    try:
        return str(source.absolute().relative_to(plan_file.parent.absolute()))
    except Exception:
        return str(source.absolute())


__all__ = [
    "gui_defaults_from_file_plan",
    "gui_settings_from_file_plan",
    "file_plan_from_gui_result",
]
