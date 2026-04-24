from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.media_helpers import normalize_track_type, sanitize_component
from utils.param_utils import apply_override, find_last_option, is_param_key, strip_param_tokens


PLAN_FORMAT_VERSION = 1
FILE_PLAN_TYPE = "file"
BATCH_PLAN_TYPE = "batch"
PLAN_SUFFIX = ".plan"
DEFAULT_VIDEO_ENCODER = "svt-av1"
DEFAULT_SCENE_DETECTION = "av1an"
DEFAULT_QUALITY = 30.0
DEFAULT_CHUNK_ORDER = "long-biased-random"
DEFAULT_SOURCE_LOADER = "ffms2"
CHUNK_ORDER_OPTIONS = (
    "long-biased-random",
    "random",
    "sequential",
    "short-to-long",
    "long-to-short",
)
SOURCE_LOADER_OPTIONS = ("auto", "ffms2", "bs", "lsmas")

DEFAULT_FASTPASS_PARAMS: Dict[str, Any] = {
    "--variance-boost-strength": 2,
    "--variance-octile": 6,
    "--variance-boost-curve": 3,
    "--tune": 0,
    "--qm-min": 7,
    "--chroma-qm-min": 10,
    "--scm": 0,
    "--enable-dlf": 2,
    "--sharp-tx": 1,
    "--enable-restoration": 0,
    "--color-primaries": 9,
    "--transfer-characteristics": 16,
    "--matrix-coefficients": 9,
    "--lp": 3,
    "--sharpness": 1,
    "--hbd-mds": 1,
    "--ac-bias": 2.0,
}

DEFAULT_MAINPASS_PARAMS: Dict[str, Any] = {
    "--film-grain": 14,
    "--complex-hvs": 1,
}


@dataclass
class SourceTrack:
    track_id: int
    track_type: str
    name: str
    lang: str
    default: bool = False
    forced: bool = False
    codec: str = ""


@dataclass
class PlanMeta:
    name: str
    created_by: str = "batch-manager.py"
    mode: str = ""


@dataclass
class PlanPaths:
    source: str
    workdir: str = ""
    zone_file: str = ""


@dataclass
class VideoPrimary:
    encoder: str = DEFAULT_VIDEO_ENCODER
    scene_detection: str = DEFAULT_SCENE_DETECTION
    quality: float = DEFAULT_QUALITY
    chunk_order: str = DEFAULT_CHUNK_ORDER
    encoder_path: str = ""
    fastpass_preset: str = ""
    preset: str = ""
    fastpass_workers: int = 8
    mainpass_workers: int = 8
    no_fastpass: bool = False
    fastpass_hdr: bool = True
    strict_sdr_8bit: bool = False
    no_dolby_vision: bool = False
    no_hdr10plus: bool = False
    attach_encode_info: bool = False
    ab_multiplier: float = 0.7
    ab_pos_dev: float = 5.0
    ab_neg_dev: float = 4.0
    ab_pos_multiplier: str = ""
    ab_neg_multiplier: str = ""
    avg_func: str = ""


@dataclass
class VideoDetails:
    fastpass_filter: str = ""
    mainpass_filter: str = ""
    main_vpy: str = ""
    fast_vpy: str = ""
    proxy_vpy: str = ""
    note: str = ""


@dataclass
class VideoExperimental:
    vpy_wrapper: bool = False
    source_loader: str = DEFAULT_SOURCE_LOADER
    crop_resize_enabled: bool = False


@dataclass
class VideoPlan:
    track_id: int
    source_name: str = ""
    source_lang: str = ""
    action: str = "edit"
    primary: VideoPrimary = field(default_factory=VideoPrimary)
    details: VideoDetails = field(default_factory=VideoDetails)
    experimental: VideoExperimental = field(default_factory=VideoExperimental)
    fastpass_params: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_FASTPASS_PARAMS))
    mainpass_params: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_MAINPASS_PARAMS))


@dataclass
class AudioPlan:
    track_id: int
    source_name: str = ""
    source_lang: str = ""
    action: str = "copy"
    name: str = ""
    lang: str = ""
    default: bool = False
    forced: bool = False
    bitrate_kbps: int = 128
    channels: int = 2


@dataclass
class SubPlan:
    track_id: int
    source_name: str = ""
    source_lang: str = ""
    action: str = "copy"
    name: str = ""
    lang: str = ""
    default: bool = False
    forced: bool = False


@dataclass
class FilePlan:
    meta: PlanMeta
    paths: PlanPaths
    video: VideoPlan
    audio: List[AudioPlan] = field(default_factory=list)
    sub: List[SubPlan] = field(default_factory=list)
    format_version: int = PLAN_FORMAT_VERSION
    plan_type: str = FILE_PLAN_TYPE


@dataclass
class BatchPlanItem:
    plan: str


@dataclass
class BatchPlan:
    meta: PlanMeta
    items: List[BatchPlanItem]
    format_version: int = PLAN_FORMAT_VERSION
    plan_type: str = BATCH_PLAN_TYPE


@dataclass
class ResolvedPaths:
    plan_path: Path
    source: Path
    workdir: Path
    zone_file: Path
    crop_resize_file: Path


@dataclass
class ResolvedFilePlan:
    plan: FilePlan
    paths: ResolvedPaths

    def has_video_edit(self) -> bool:
        return self.plan.video.action.strip().lower() == "edit"

    def has_video_copy(self) -> bool:
        return self.plan.video.action.strip().lower() == "copy"

    def runtime_tracks(self) -> List[Dict[str, Any]]:
        tracks: List[Dict[str, Any]] = []
        tracks.append(self._video_runtime_track())
        tracks.extend(self._audio_runtime_tracks())
        tracks.extend(self._sub_runtime_tracks())
        return tracks

    def build_encode_params_text(self) -> Optional[str]:
        if not self.has_video_edit():
            return None
        video = self.plan.video
        tokens = params_map_to_tokens(video.fastpass_params)
        merged = apply_override(list(tokens), params_map_to_tokens(video.mainpass_params))
        merged = strip_param_tokens(merged, ["--crf", "--preset"])
        merged.extend(["--crf", format_value(video.primary.quality)])
        if video.primary.preset:
            merged.extend(["--preset", str(video.primary.preset)])
        if not merged:
            return None
        return subprocess.list2cmdline([str(x) for x in merged])

    def build_fastpass_params_text(self) -> str:
        tokens = params_map_to_tokens(self.plan.video.fastpass_params)
        if not tokens:
            return ""
        return subprocess.list2cmdline([str(x) for x in tokens])

    def build_mainpass_params_text(self) -> str:
        tokens = params_map_to_tokens(self.plan.video.mainpass_params)
        if not tokens:
            return ""
        return subprocess.list2cmdline([str(x) for x in tokens])

    def _video_runtime_track(self) -> Dict[str, Any]:
        video = self.plan.video
        primary = video.primary
        details = video.details
        experimental = video.experimental
        video_config = {
            "quality": format_value(primary.quality),
            "chunk_order": str(primary.chunk_order or ""),
            "encoder_path": str(primary.encoder_path or ""),
            "fastpass_preset": str(primary.fastpass_preset or ""),
            "preset": str(primary.preset or ""),
            "fastpass_params": {key: format_value(value) for key, value in video.fastpass_params.items()},
            "mainpass_params": {key: format_value(value) for key, value in video.mainpass_params.items()},
        }

        track_mux = {
            "encoder": primary.encoder,
            "sceneDetection": primary.scene_detection,
            "chunkOrder": str(primary.chunk_order or ""),
            "encoderPath": str(primary.encoder_path or ""),
            "noFastpass": bool_text(primary.no_fastpass),
            "fastpassHdr": bool_text(primary.fastpass_hdr),
            "strictSdr8bit": bool_text(primary.strict_sdr_8bit),
            "noDolbyVision": bool_text(primary.no_dolby_vision or primary.strict_sdr_8bit),
            "noHdr10Plus": bool_text(primary.no_hdr10plus or primary.strict_sdr_8bit),
            "fastpassWorkers": str(primary.fastpass_workers),
            "mainpassWorkers": str(primary.mainpass_workers),
            "workers": str(primary.fastpass_workers),
            "abMultiplier": format_value(primary.ab_multiplier),
            "abPosDev": format_value(primary.ab_pos_dev),
            "abNegDev": format_value(primary.ab_neg_dev),
            "abPosMultiplier": str(primary.ab_pos_multiplier or ""),
            "abNegMultiplier": str(primary.ab_neg_multiplier or ""),
            "avgFunc": str(primary.avg_func or ""),
            "fastpass": details.fastpass_filter,
            "mainpass": details.mainpass_filter,
            "mainVpy": details.main_vpy,
            "fastVpy": details.fast_vpy,
            "proxyVpy": details.proxy_vpy,
            "attachEncodeInfo": bool_text(primary.attach_encode_info),
            "vpyWrapper": bool_text(experimental.vpy_wrapper),
            "sourceLoader": str(experimental.source_loader or DEFAULT_SOURCE_LOADER),
            "cropResizeEnabled": bool_text(experimental.crop_resize_enabled),
        }
        if details.note:
            track_mux["note"] = details.note

        return {
            "trackId": int(video.track_id),
            "type": "video",
            "trackStatus": video.action.upper(),
            "origName": video.source_name,
            "origLang": video.source_lang,
            "videoConfig": video_config,
            "trackMux": track_mux,
            "fileBase": build_file_base(
                track_id=video.track_id,
                default=False,
                lang=video.source_lang or "und",
                name=video.source_name or f"track{video.track_id}",
            ),
        }

    def _audio_runtime_tracks(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for track in self.plan.audio:
            track_mux = {
                "name": track.name,
                "lang": track.lang or track.source_lang or "und",
                "default": bool_text(track.default),
                "forced": bool_text(track.forced),
            }
            track_param: Dict[str, str] = {}
            if track.action.strip().lower() == "edit":
                track_param["bitrate"] = str(track.bitrate_kbps)
                track_param["channels"] = str(track.channels)
            out.append(
                {
                    "trackId": int(track.track_id),
                    "type": "audio",
                    "trackStatus": track.action.upper(),
                    "origName": track.source_name,
                    "origLang": track.source_lang,
                    "trackParam": track_param,
                    "trackMux": track_mux,
                    "fileBase": build_file_base(
                        track_id=track.track_id,
                        default=track.default,
                        lang=track.lang or track.source_lang or "und",
                        name=track.name or track.source_name or f"track{track.track_id}",
                    ),
                }
            )
        return out

    def _sub_runtime_tracks(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for track in self.plan.sub:
            track_mux = {
                "name": track.name,
                "lang": track.lang or track.source_lang or "und",
                "default": bool_text(track.default),
                "forced": bool_text(track.forced),
            }
            out.append(
                {
                    "trackId": int(track.track_id),
                    "type": "sub",
                    "trackStatus": track.action.upper(),
                    "origName": track.source_name,
                    "origLang": track.source_lang,
                    "trackParam": {},
                    "trackMux": track_mux,
                    "fileBase": build_file_base(
                        track_id=track.track_id,
                        default=track.default,
                        lang=track.lang or track.source_lang or "und",
                        name=track.name or track.source_name or f"track{track.track_id}",
                    ),
                }
            )
        return out


@dataclass
class RunnerEvent:
    event: str
    plan: str
    mode: str
    stage: str
    status: str
    message: str = ""
    timestamp: float = 0.0
    session_id: str = ""
    plan_run_id: str = ""
    source: str = ""
    workdir: str = ""
    progress: float = -1.0
    started_at: float = 0.0
    ended_at: float = 0.0
    elapsed_seconds: float = 0.0


def default_video_primary() -> VideoPrimary:
    return VideoPrimary()


def default_video_details() -> VideoDetails:
    return VideoDetails()


def default_video_experimental() -> VideoExperimental:
    return VideoExperimental()


def normalize_encoder(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("_", "-")
    if raw in ("", "auto", "default", "svt-av1"):
        return "svt-av1"
    if raw in ("x265", "libx265"):
        return "libx265"
    return DEFAULT_VIDEO_ENCODER


def normalize_chunk_order(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in CHUNK_ORDER_OPTIONS:
        return raw
    return DEFAULT_CHUNK_ORDER


def normalize_source_loader(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in ("bestsource", "best-source"):
        raw = "bs"
    if raw in ("lsmash", "lwlibavsource"):
        raw = "lsmas"
    if raw in SOURCE_LOADER_OPTIONS:
        return raw
    return DEFAULT_SOURCE_LOADER


def bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def parse_bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text if text else "0"
    return str(value)


def coerce_scalar(value: Any) -> Any:
    if isinstance(value, (bool, int, float)):
        return value
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        if "." in text:
            return float(text)
        return int(text)
    except Exception:
        return text


def sanitize_lang(lang: str) -> str:
    value = str(lang or "").strip() or "und"
    cleaned = []
    for char in value:
        if char.isalnum() or char in ("_", "-"):
            cleaned.append(char)
        else:
            cleaned.append("_")
    result = "".join(cleaned)[:16]
    return result or "und"


def build_file_base(*, track_id: int, default: bool, lang: str, name: str) -> str:
    d_flag = "d1" if bool(default) else "d0"
    return f"{track_id}-{d_flag}-{sanitize_lang(lang)}-{sanitize_component(name)}"


def params_map_to_tokens(params: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []
    for key, value in params.items():
        tokens.append(str(key))
        value_text = str(value).strip()
        if value_text:
            tokens.append(value_text)
    return tokens


__all__ = [
    "PLAN_FORMAT_VERSION",
    "FILE_PLAN_TYPE",
    "BATCH_PLAN_TYPE",
    "PLAN_SUFFIX",
    "DEFAULT_VIDEO_ENCODER",
    "DEFAULT_SCENE_DETECTION",
    "DEFAULT_QUALITY",
    "DEFAULT_CHUNK_ORDER",
    "DEFAULT_SOURCE_LOADER",
    "CHUNK_ORDER_OPTIONS",
    "SOURCE_LOADER_OPTIONS",
    "DEFAULT_FASTPASS_PARAMS",
    "DEFAULT_MAINPASS_PARAMS",
    "SourceTrack",
    "PlanMeta",
    "PlanPaths",
    "VideoPrimary",
    "VideoDetails",
    "VideoExperimental",
    "VideoPlan",
    "AudioPlan",
    "SubPlan",
    "FilePlan",
    "BatchPlanItem",
    "BatchPlan",
    "ResolvedPaths",
    "ResolvedFilePlan",
    "RunnerEvent",
    "default_video_primary",
    "default_video_details",
    "default_video_experimental",
    "normalize_track_type",
    "normalize_encoder",
    "normalize_chunk_order",
    "normalize_source_loader",
    "bool_text",
    "parse_bool_value",
    "to_float",
    "format_value",
    "coerce_scalar",
    "sanitize_component",
    "sanitize_lang",
    "build_file_base",
    "params_map_to_tokens",
    "is_param_key",
    "find_last_option",
    "apply_override",
    "strip_param_tokens",
]
