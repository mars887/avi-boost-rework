from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from utils.plan.io import load_file_plan, plan_path_for_source, resolve_paths
from utils.plan.types import (
    DEFAULT_QUALITY,
    DEFAULT_SCENE_DETECTION,
    AudioPlan,
    FilePlan,
    PlanMeta,
    PlanPaths,
    SubPlan,
    VideoDetails,
    VideoPlan,
    VideoPrimary,
    coerce_scalar,
    format_value,
    normalize_encoder,
    normalize_track_type,
    params_map_to_tokens,
    parse_bool_value,
    to_float,
)


def gui_defaults_from_file_plan(plan: FilePlan) -> Dict[str, Any]:
    video = plan.video
    primary = video.primary
    details = video.details
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
        "quality": primary.quality,
        "fastpass_preset": primary.fastpass_preset,
        "preset": primary.preset,
        "mainpass_preset": primary.preset,
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
        "attach_encode_info": primary.attach_encode_info,
        "fastpass": details.fastpass_filter,
        "mainpass": details.mainpass_filter,
        "main_vpy": details.main_vpy,
        "fast_vpy": details.fast_vpy,
        "proxy_vpy": details.proxy_vpy,
        "note": details.note,
    }


def gui_settings_from_file_plan(plan: FilePlan) -> List[Dict[str, Any]]:
    settings: List[Dict[str, Any]] = []
    video = plan.video
    fastpass_tokens = params_map_to_tokens(video.fastpass_params)
    mainpass_tokens = params_map_to_tokens(video.mainpass_params)
    fastpass_tokens.extend(["--crf", format_value(video.primary.quality)])
    mainpass_tokens.extend(["--crf", format_value(video.primary.quality)])
    if video.primary.fastpass_preset:
        fastpass_tokens.extend(["--preset", str(video.primary.fastpass_preset)])
    if video.primary.preset:
        mainpass_tokens.extend(["--preset", str(video.primary.preset)])
    settings.append(
        {
            "id": str(video.track_id),
            "type": "video",
            "mode": video.action.upper(),
            "params": subprocess.list2cmdline([str(x) for x in fastpass_tokens]),
            "last_params": subprocess.list2cmdline([str(x) for x in mainpass_tokens]),
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
        fastpass_preset=str(defaults.get("fastpass_preset") or ""),
        preset=str(defaults.get("preset") or defaults.get("mainpass_preset") or ""),
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
    )
    video_details = VideoDetails(
        fastpass_filter=str(defaults.get("fastpass_filter") or ""),
        mainpass_filter=str(defaults.get("mainpass_filter") or ""),
        main_vpy=str(defaults.get("main_vpy") or ""),
        fast_vpy=str(defaults.get("fast_vpy") or ""),
        proxy_vpy=str(defaults.get("proxy_vpy") or ""),
        note=str(defaults.get("note") or ""),
    )

    track_param = dict(video_result.get("trackParam") or {})
    fastpass_params: Dict[str, Any] = {}
    mainpass_params: Dict[str, Any] = {}
    quality = video_primary.quality
    fastpass_preset = video_primary.fastpass_preset
    preset = video_primary.preset
    for key, value in track_param.items():
        key_text = str(key)
        if key_text == "--crf":
            quality = to_float(value, DEFAULT_QUALITY)
            continue
        if key_text == "^^crf":
            quality = to_float(value, quality)
            continue
        if key_text == "--preset":
            fastpass_preset = str(value or "")
            continue
        if key_text == "^^preset":
            preset = str(value or "")
            continue
        if key_text.startswith("^^"):
            mainpass_params[key_text.replace("^^", "--", 1)] = coerce_scalar(value)
        elif key_text.startswith("--"):
            fastpass_params[key_text] = coerce_scalar(value)
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


def file_plan_from_legacy_tracks_data(
    data: Dict[str, Any],
    *,
    plan_path: Optional[Path] = None,
    zone_file: Optional[Path] = None,
) -> FilePlan:
    source = Path(str(data.get("source") or "")).expanduser()
    if not source:
        raise RuntimeError("Legacy tracks data has no source path")
    if not source.is_absolute():
        source = source.absolute()

    workdir_raw = str(data.get("workdir") or "")
    workdir = Path(workdir_raw).expanduser() if workdir_raw else (source.parent / source.stem)
    if not workdir.is_absolute():
        workdir = workdir.absolute()

    plan_file = Path(plan_path).absolute() if plan_path is not None else plan_path_for_source(source).absolute()
    _ = zone_file
    tracks = [item for item in (data.get("tracks") or []) if isinstance(item, dict)]
    video_track = next(
        (
            item
            for item in tracks
            if normalize_track_type(item.get("type")) == "video"
            and str(item.get("trackStatus") or "").strip().lower() in ("edit", "copy")
        ),
        None,
    )
    if video_track is None:
        raise RuntimeError("Legacy tracks data has no usable video track")

    video_mux = dict(video_track.get("trackMux") or {})
    video_param = dict(video_track.get("trackParam") or {})
    fastpass_params: Dict[str, Any] = {}
    mainpass_params: Dict[str, Any] = {}
    quality = DEFAULT_QUALITY
    fastpass_preset = ""
    preset = ""

    for key, value in video_param.items():
        key_text = str(key)
        if key_text == "--crf":
            quality = to_float(value, DEFAULT_QUALITY)
            continue
        if key_text == "^^crf":
            quality = to_float(value, quality)
            continue
        if key_text == "--preset":
            fastpass_preset = str(value or "")
            continue
        if key_text == "^^preset":
            preset = str(value or "")
            continue
        if key_text.startswith("^^"):
            mainpass_params[key_text.replace("^^", "--", 1)] = coerce_scalar(value)
        elif key_text.startswith("--"):
            fastpass_params[key_text] = coerce_scalar(value)

    workers_text = str(video_mux.get("workers") or "8")
    video_plan = VideoPlan(
        track_id=int(video_track.get("trackId") or 0),
        source_name=str(video_track.get("origName") or ""),
        source_lang=str(video_track.get("origLang") or ""),
        action=str(video_track.get("trackStatus") or "EDIT").strip().lower(),
        primary=VideoPrimary(
            encoder=normalize_encoder(video_mux.get("encoder")),
            scene_detection=str(video_mux.get("sceneDetection") or DEFAULT_SCENE_DETECTION),
            quality=quality,
            fastpass_preset=fastpass_preset,
            preset=preset,
            fastpass_workers=int(video_mux.get("fastpassWorkers") or workers_text or 8),
            mainpass_workers=int(video_mux.get("mainpassWorkers") or workers_text or 8),
            no_fastpass=parse_bool_value(video_mux.get("noFastpass"), False),
            fastpass_hdr=parse_bool_value(video_mux.get("fastpassHdr"), True),
            strict_sdr_8bit=parse_bool_value(video_mux.get("strictSdr8bit"), False),
            no_dolby_vision=parse_bool_value(video_mux.get("noDolbyVision"), False),
            no_hdr10plus=parse_bool_value(video_mux.get("noHdr10Plus"), False),
            attach_encode_info=parse_bool_value(video_mux.get("attachEncodeInfo"), False),
            ab_multiplier=to_float(video_mux.get("abMultiplier"), 0.7),
            ab_pos_dev=to_float(video_mux.get("abPosDev"), 5.0),
            ab_neg_dev=to_float(video_mux.get("abNegDev"), 4.0),
            ab_pos_multiplier=str(video_mux.get("abPosMultiplier") or ""),
            ab_neg_multiplier=str(video_mux.get("abNegMultiplier") or ""),
        ),
        details=VideoDetails(
            fastpass_filter=str(video_mux.get("fastpass") or ""),
            mainpass_filter=str(video_mux.get("mainpass") or ""),
            main_vpy=str(video_mux.get("mainVpy") or ""),
            fast_vpy=str(video_mux.get("fastVpy") or ""),
            proxy_vpy=str(video_mux.get("proxyVpy") or ""),
            note=str(video_mux.get("note") or ""),
        ),
        fastpass_params=fastpass_params,
        mainpass_params=mainpass_params,
    )

    audio_plans: List[AudioPlan] = []
    sub_plans: List[SubPlan] = []
    for item in tracks:
        track_type = normalize_track_type(item.get("type"))
        if track_type == "video":
            continue
        mux = dict(item.get("trackMux") or {})
        param = dict(item.get("trackParam") or {})
        if track_type == "audio":
            audio_plans.append(
                AudioPlan(
                    track_id=int(item.get("trackId") or 0),
                    source_name=str(item.get("origName") or ""),
                    source_lang=str(item.get("origLang") or ""),
                    action=str(item.get("trackStatus") or "COPY").strip().lower(),
                    name=str(mux.get("name") or ""),
                    lang=str(mux.get("lang") or item.get("origLang") or ""),
                    default=parse_bool_value(mux.get("default"), False),
                    forced=parse_bool_value(mux.get("forced"), False),
                    bitrate_kbps=int(param.get("bitrate") or 128),
                    channels=int(param.get("channels") or 2),
                )
            )
        elif track_type == "sub":
            sub_plans.append(
                SubPlan(
                    track_id=int(item.get("trackId") or 0),
                    source_name=str(item.get("origName") or ""),
                    source_lang=str(item.get("origLang") or ""),
                    action=str(item.get("trackStatus") or "COPY").strip().lower(),
                    name=str(mux.get("name") or ""),
                    lang=str(mux.get("lang") or item.get("origLang") or ""),
                    default=parse_bool_value(mux.get("default"), False),
                    forced=parse_bool_value(mux.get("forced"), False),
                )
            )

    meta_name = workdir.name or source.stem
    return FilePlan(
        meta=PlanMeta(name=meta_name, created_by="legacy-tracks.json"),
        paths=PlanPaths(source=_relative_source_for_plan(source, plan_file)),
        video=video_plan,
        audio=audio_plans,
        sub=sub_plans,
    )


def _relative_source_for_plan(source: Path, plan_file: Path) -> str:
    try:
        return str(source.absolute().relative_to(plan_file.parent.absolute()))
    except Exception:
        return str(source.absolute())


def load_legacy_tracks_json(path: Path, *, plan_path: Optional[Path] = None) -> FilePlan:
    tracks_path = Path(path).absolute()
    with tracks_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return file_plan_from_legacy_tracks_data(data, plan_path=plan_path, zone_file=tracks_path.parent / "zone_edit_command.txt")


__all__ = [
    "gui_defaults_from_file_plan",
    "gui_settings_from_file_plan",
    "file_plan_from_gui_result",
    "file_plan_from_legacy_tracks_data",
    "load_legacy_tracks_json",
]
