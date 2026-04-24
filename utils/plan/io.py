from __future__ import annotations

import shlex
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from utils.plan.summary import probe_source_tracks
from utils.plan.types import (
    BATCH_PLAN_TYPE,
    DEFAULT_MAINPASS_PARAMS,
    DEFAULT_CHUNK_ORDER,
    DEFAULT_SOURCE_LOADER,
    DEFAULT_QUALITY,
    DEFAULT_SCENE_DETECTION,
    DEFAULT_VIDEO_ENCODER,
    DEFAULT_FASTPASS_PARAMS,
    FILE_PLAN_TYPE,
    PLAN_FORMAT_VERSION,
    PLAN_SUFFIX,
    AudioPlan,
    BatchPlan,
    BatchPlanItem,
    FilePlan,
    PlanMeta,
    PlanPaths,
    ResolvedFilePlan,
    ResolvedPaths,
    SubPlan,
    SourceTrack,
    VideoDetails,
    VideoExperimental,
    VideoPlan,
    VideoPrimary,
    coerce_scalar,
    format_value,
    normalize_encoder,
    normalize_chunk_order,
    normalize_source_loader,
    parse_bool_value,
    sanitize_component,
    to_float,
)
from utils.zoned_commands import zoned_command_path


_PARAM_SECTION_HEADERS = {"[video.fastpass.params]", "[video.mainpass.params]"}


def plan_path_for_source(source: Path) -> Path:
    return Path(source).with_suffix(PLAN_SUFFIX)


def _relative_or_absolute(path: Path, base_dir: Path) -> str:
    try:
        return str(Path(path).absolute().relative_to(base_dir.absolute()))
    except Exception:
        return str(Path(path).absolute())


def _resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _workdir_name(plan: FilePlan, source: Path) -> str:
    name = str(plan.meta.name or "").strip()
    if name:
        return name
    return source.stem


def create_default_file_plan(source: Path, *, tracks: Optional[Sequence[SourceTrack]] = None) -> FilePlan:
    src = Path(source).absolute()
    plan_path = plan_path_for_source(src)
    source_tracks = list(tracks or probe_source_tracks(src))

    video_track = next((track for track in source_tracks if track.track_type == "video"), None)
    if video_track is None:
        raise RuntimeError(f"No video track found in source: {src}")

    audio_tracks = [track for track in source_tracks if track.track_type == "audio"]
    sub_tracks = [track for track in source_tracks if track.track_type == "sub"]

    return FilePlan(
        meta=PlanMeta(name=src.stem),
        paths=PlanPaths(source=_relative_or_absolute(src, plan_path.parent)),
        video=VideoPlan(
            track_id=video_track.track_id,
            source_name=video_track.name,
            source_lang=video_track.lang,
            action="edit",
        ),
        audio=[
            AudioPlan(
                track_id=track.track_id,
                source_name=track.name,
                source_lang=track.lang,
                action="copy",
                lang=track.lang,
                default=track.default,
                forced=track.forced,
            )
            for track in audio_tracks
        ],
        sub=[
            SubPlan(
                track_id=track.track_id,
                source_name=track.name,
                source_lang=track.lang,
                action="copy",
                lang=track.lang,
                default=track.default,
                forced=track.forced,
            )
            for track in sub_tracks
        ],
    )


def resolve_paths(plan: FilePlan, plan_path: Path) -> ResolvedPaths:
    plan_file = Path(plan_path).absolute()
    base_dir = plan_file.parent
    source = _resolve_path(plan.paths.source, base_dir)
    workdir_raw = str(plan.paths.workdir or "").strip()
    workdir = _resolve_path(workdir_raw, base_dir) if workdir_raw else (base_dir / _workdir_name(plan, source)).resolve()
    zone_raw = str(plan.paths.zone_file or "").strip()
    if zone_raw:
        zone_path = Path(zone_raw).expanduser()
        zoned_file = zone_path.resolve() if zone_path.is_absolute() else (workdir / zone_path).resolve()
    else:
        zoned_file = zoned_command_path(workdir).resolve()
    return ResolvedPaths(
        plan_path=plan_file,
        source=source,
        workdir=workdir,
        zone_file=zoned_file,
        crop_resize_file=zoned_file,
    )


def _decode_plan_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return raw.decode(encoding)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def _toml_key(key: str) -> str:
    escaped = key.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_string(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text if text else "0"
    return _toml_string(str(value))


def _convert_param_line_to_toml(raw_line: str) -> str:
    indent = raw_line[: len(raw_line) - len(raw_line.lstrip())]
    stripped = raw_line.strip()
    if not stripped or "=" in stripped:
        return raw_line
    tokens = shlex.split(stripped, posix=False)
    if not tokens or not tokens[0].startswith("-"):
        return raw_line
    key = tokens[0]
    value = " ".join(tokens[1:]).strip()
    return f"{indent}{_toml_key(key)} = {_toml_scalar(coerce_scalar(value) if value else '')}"


def _reorder_file_plan_blocks(lines: List[str]) -> List[str]:
    blocks: List[tuple[Optional[str], List[str]]] = []
    current_header: Optional[str] = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_lines, current_header
        blocks.append((current_header, list(current_lines)))
        current_lines = []

    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if current_header is not None or current_lines:
                flush()
            current_header = stripped
            current_lines = [raw]
            continue
        current_lines.append(raw)
    if current_header is not None or current_lines:
        flush()

    if not any((header or "").lower() == "[video]" for header, _ in blocks):
        return lines

    preferred = [
        None,
        "[paths]",
        "[video]",
        "[video.primary]",
        "[video.pipeline]",
        "[video.color]",
        "[video.pipe]",
        "[video.experimental]",
        "[video.fastpass.params]",
        "[video.mainpass.params]",
        "[mux]",
    ]
    used: set[int] = set()
    ordered: List[str] = []

    def append_block(index: int) -> None:
        used.add(index)
        ordered.extend(blocks[index][1])

    for wanted in preferred:
        for index, (header, _) in enumerate(blocks):
            if index in used:
                continue
            if wanted is None:
                if header is None:
                    append_block(index)
            elif (header or "").lower() == wanted.lower():
                append_block(index)

    for index, (header, _) in enumerate(blocks):
        if index in used:
            continue
        if (header or "").lower() in ("[[audio]]", "[[sub]]", "[meta]"):
            append_block(index)

    for index in range(len(blocks)):
        if index not in used:
            append_block(index)

    return ordered


def _preprocess_plan_text(text: str) -> str:
    stripped_lines: List[str] = []
    in_param_section = False
    for raw in text.splitlines():
        if raw.lstrip().startswith("//"):
            continue
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_param_section = stripped.lower() in _PARAM_SECTION_HEADERS
            stripped_lines.append(raw)
            continue
        if in_param_section and stripped and not stripped.startswith("#"):
            stripped_lines.append(_convert_param_line_to_toml(raw))
            continue
        stripped_lines.append(raw)
    return "\n".join(_reorder_file_plan_blocks(stripped_lines))


def load_plan(path: Path) -> FilePlan | BatchPlan:
    plan_path = Path(path).absolute()
    raw = plan_path.read_bytes()
    data = tomllib.loads(_preprocess_plan_text(_decode_plan_text(raw)))
    meta_data = dict(data.get("meta") or {})
    plan_type = str(data.get("plan_type") or meta_data.get("plan_type") or "").strip().lower()
    if plan_type == FILE_PLAN_TYPE:
        return _load_file_plan(data)
    if plan_type == BATCH_PLAN_TYPE:
        return _load_batch_plan(data)
    if "source" in data or "video" in data:
        return _load_file_plan(data)
    if "items" in data:
        return _load_batch_plan(data)
    raise RuntimeError(f"Unsupported plan_type in {plan_path}: {plan_type!r}")


def load_file_plan(path: Path) -> FilePlan:
    plan = load_plan(path)
    if not isinstance(plan, FilePlan):
        raise RuntimeError(f"{path} is not a file plan")
    return plan


def load_batch_plan(path: Path) -> BatchPlan:
    plan = load_plan(path)
    if not isinstance(plan, BatchPlan):
        raise RuntimeError(f"{path} is not a batch plan")
    return plan


def resolve_file_plan(path: Path) -> ResolvedFilePlan:
    file_plan = load_file_plan(path)
    return ResolvedFilePlan(plan=file_plan, paths=resolve_paths(file_plan, Path(path)))


def save_plan(plan: FilePlan | BatchPlan, path: Path) -> None:
    plan_path = Path(path)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(dump_plan(plan), encoding="utf-8", newline="\n")


def dump_plan(plan: FilePlan | BatchPlan) -> str:
    if isinstance(plan, FilePlan):
        return _dump_file_plan(plan)
    if isinstance(plan, BatchPlan):
        return _dump_batch_plan(plan)
    raise TypeError(f"Unsupported plan instance: {type(plan)!r}")


def make_batch_plan(*, name: str, mode: str, plans: Sequence[Path], base_dir: Path) -> BatchPlan:
    items = [BatchPlanItem(plan=_relative_or_absolute(Path(plan), base_dir)) for plan in plans]
    return BatchPlan(meta=PlanMeta(name=name, mode=mode), items=items)


def resolve_batch_plan(batch_path: Path) -> List[ResolvedFilePlan]:
    plan_file = Path(batch_path).absolute()
    resolved: List[ResolvedFilePlan] = []
    seen: set[str] = set()
    visiting: set[str] = set()

    def visit_plan(path: Path) -> None:
        key = str(path.absolute()).lower()
        if key in visiting:
            raise RuntimeError(f"batch plan cycle detected at {path}")
        plan = load_plan(path)
        if isinstance(plan, FilePlan):
            if key in seen:
                return
            seen.add(key)
            resolved.append(ResolvedFilePlan(plan=plan, paths=resolve_paths(plan, path)))
            return

        visiting.add(key)
        try:
            for item in plan.items:
                nested = _resolve_path(item.plan, path.parent)
                visit_plan(nested)
        finally:
            visiting.remove(key)

    batch = load_batch_plan(plan_file)
    for item in batch.items:
        visit_plan(_resolve_path(item.plan, plan_file.parent))
    return resolved


def _load_file_plan(data: Dict[str, Any]) -> FilePlan:
    meta_data = dict(data.get("meta") or {})
    paths_data = dict(data.get("paths") or {})
    video_data = dict(data.get("video") or {})
    video_primary_data = dict(video_data.get("primary") or {})
    video_pipeline_data = dict(video_data.get("pipeline") or {})
    video_color_data = dict(video_data.get("color") or {})
    video_pipe_data = dict(video_data.get("pipe") or {})
    video_experimental_data = dict(video_data.get("experimental") or {})
    video_details_data = dict(video_data.get("details") or {})
    video_fastpass_section = dict(video_data.get("fastpass") or {})
    video_mainpass_section = dict(video_data.get("mainpass") or {})
    video_fastpass_params = dict(video_fastpass_section.get("params") or {})
    video_mainpass_params = dict(video_mainpass_section.get("params") or {})
    mux_data = dict(data.get("mux") or {})

    format_version = int(meta_data.get("format_version") or data.get("format_version") or PLAN_FORMAT_VERSION)
    plan_type = str(meta_data.get("plan_type") or data.get("plan_type") or FILE_PLAN_TYPE)

    source_text = str(data.get("source") or paths_data.get("source") or "")
    quality_value = data.get("quality")
    if quality_value is None:
        quality_value = video_primary_data.get("quality")
    preset_value = data.get("preset")
    if preset_value is None:
        preset_value = video_primary_data.get("preset")

    return FilePlan(
        format_version=format_version,
        plan_type=plan_type,
        meta=PlanMeta(
            name=str(meta_data.get("name") or ""),
            created_by=str(meta_data.get("created_by") or "batch-manager.py"),
            mode=str(meta_data.get("mode") or data.get("mode") or ""),
        ),
        paths=PlanPaths(
            source=source_text,
            workdir=str(paths_data.get("workdir") or ""),
            zone_file=str(paths_data.get("zone_file") or ""),
        ),
        video=VideoPlan(
            track_id=int(video_data.get("track_id") or 0),
            source_name=str(video_data.get("source_name") or ""),
            source_lang=str(video_data.get("source_lang") or ""),
            action=str(video_data.get("action") or "edit").lower(),
            primary=VideoPrimary(
                encoder=normalize_encoder(video_pipeline_data.get("encoder") or video_primary_data.get("encoder")),
                scene_detection=str(video_pipeline_data.get("scene_detection") or video_primary_data.get("scene_detection") or DEFAULT_SCENE_DETECTION),
                quality=to_float(quality_value, DEFAULT_QUALITY),
                chunk_order=normalize_chunk_order(video_pipeline_data.get("chunk_order") or video_primary_data.get("chunk_order") or DEFAULT_CHUNK_ORDER),
                encoder_path=str(video_pipeline_data.get("encoder_path") or video_primary_data.get("encoder_path") or ""),
                fastpass_preset=str(video_primary_data.get("fastpass_preset") or ""),
                preset=str(preset_value or ""),
                fastpass_workers=int(video_primary_data.get("fastpass_workers") or 8),
                mainpass_workers=int(video_primary_data.get("mainpass_workers") or 8),
                no_fastpass=parse_bool_value(video_pipeline_data.get("no_fastpass", video_primary_data.get("no_fastpass")), False),
                fastpass_hdr=parse_bool_value(video_color_data.get("fastpass_hdr", video_primary_data.get("fastpass_hdr")), True),
                strict_sdr_8bit=parse_bool_value(video_color_data.get("strict_sdr_8bit", video_primary_data.get("strict_sdr_8bit")), False),
                no_dolby_vision=parse_bool_value(video_color_data.get("no_dolby_vision", video_primary_data.get("no_dolby_vision")), False),
                no_hdr10plus=parse_bool_value(video_color_data.get("no_hdr10plus", video_primary_data.get("no_hdr10plus")), False),
                attach_encode_info=parse_bool_value(mux_data.get("attach_encode_info", video_primary_data.get("attach_encode_info")), False),
                ab_multiplier=to_float(video_primary_data.get("ab_multiplier"), 0.7),
                ab_pos_dev=to_float(video_primary_data.get("ab_pos_dev"), 5.0),
                ab_neg_dev=to_float(video_primary_data.get("ab_neg_dev"), 4.0),
                ab_pos_multiplier=str(video_primary_data.get("ab_pos_multiplier") or ""),
                ab_neg_multiplier=str(video_primary_data.get("ab_neg_multiplier") or ""),
                avg_func=str(
                    video_primary_data.get("avg_func")
                    or video_primary_data.get("avgFunc")
                    or video_primary_data.get("ab_avg_func")
                    or video_primary_data.get("abAvgFunc")
                    or ""
                ),
            ),
            details=VideoDetails(
                fastpass_filter=str(video_pipe_data.get("fastpass_filter") or video_details_data.get("fastpass_filter") or ""),
                mainpass_filter=str(video_pipe_data.get("mainpass_filter") or video_details_data.get("mainpass_filter") or ""),
                main_vpy=str(video_pipe_data.get("main_vpy") or video_details_data.get("main_vpy") or ""),
                fast_vpy=str(video_pipe_data.get("fast_vpy") or video_details_data.get("fast_vpy") or ""),
                proxy_vpy=str(video_pipe_data.get("proxy_vpy") or video_details_data.get("proxy_vpy") or ""),
                note=str(mux_data.get("note") or video_details_data.get("note") or ""),
            ),
            experimental=VideoExperimental(
                vpy_wrapper=parse_bool_value(video_experimental_data.get("vpy_wrapper"), False),
                source_loader=normalize_source_loader(video_experimental_data.get("source_loader") or DEFAULT_SOURCE_LOADER),
                crop_resize_enabled=parse_bool_value(video_experimental_data.get("crop_resize_enabled"), False),
            ),
            fastpass_params=(
                {
                    str(key): coerce_scalar(value)
                    for key, value in video_fastpass_params.items()
                    if str(key) != "__empty__"
                }
                if "params" in video_fastpass_section
                else dict(DEFAULT_FASTPASS_PARAMS)
            ),
            mainpass_params=(
                {
                    str(key): coerce_scalar(value)
                    for key, value in video_mainpass_params.items()
                    if str(key) != "__empty__"
                }
                if "params" in video_mainpass_section
                else dict(DEFAULT_MAINPASS_PARAMS)
            ),
        ),
        audio=[
            AudioPlan(
                track_id=int(item.get("track_id") or 0),
                source_name=str(item.get("source_name") or ""),
                source_lang=str(item.get("source_lang") or ""),
                action=str(item.get("action") or "copy").lower(),
                name=str(item.get("name") or ""),
                lang=str(item.get("lang") or ""),
                default=parse_bool_value(item.get("default"), False),
                forced=parse_bool_value(item.get("forced"), False),
                bitrate_kbps=int(item.get("bitrate_kbps") or 128),
                channels=int(item.get("channels") or 2),
            )
            for item in (data.get("audio") or [])
            if isinstance(item, dict)
        ],
        sub=[
            SubPlan(
                track_id=int(item.get("track_id") or 0),
                source_name=str(item.get("source_name") or ""),
                source_lang=str(item.get("source_lang") or ""),
                action=str(item.get("action") or "copy").lower(),
                name=str(item.get("name") or ""),
                lang=str(item.get("lang") or ""),
                default=parse_bool_value(item.get("default"), False),
                forced=parse_bool_value(item.get("forced"), False),
            )
            for item in (data.get("sub") or [])
            if isinstance(item, dict)
        ],
    )


def _load_batch_plan(data: Dict[str, Any]) -> BatchPlan:
    meta_data = dict(data.get("meta") or {})
    return BatchPlan(
        format_version=int(meta_data.get("format_version") or data.get("format_version") or PLAN_FORMAT_VERSION),
        plan_type=str(meta_data.get("plan_type") or data.get("plan_type") or BATCH_PLAN_TYPE),
        meta=PlanMeta(
            name=str(meta_data.get("name") or ""),
            created_by=str(meta_data.get("created_by") or "batch-manager.py"),
            mode=str(meta_data.get("mode") or data.get("mode") or "full"),
        ),
        items=[
            BatchPlanItem(plan=str(item.get("plan") or ""))
            for item in (data.get("items") or [])
            if isinstance(item, dict)
        ],
    )


def _comment_block(title: str) -> List[str]:
    return [
        f"// --------------------------------------------",
        f"// {title.center(42)}",
        f"// --------------------------------------------",
        "",
    ]


def _render_param_line(key: str, value: Any) -> str:
    value_text = str(format_value(value)).strip()
    if not value_text:
        return str(key)
    return subprocess.list2cmdline([str(key), value_text])


def _dump_file_plan(plan: FilePlan) -> str:
    lines: List[str] = []

    lines.extend(_comment_block("MAIN"))
    lines.append(f"source = {_toml_string(plan.paths.source)}")
    lines.append(f"quality = {_toml_scalar(plan.video.primary.quality)}")
    lines.append(f"preset = {_toml_scalar(coerce_scalar(plan.video.primary.preset))}")
    lines.append("")
    if str(plan.paths.workdir or "").strip() or str(plan.paths.zone_file or "").strip():
        lines.append("[paths]")
        if str(plan.paths.workdir or "").strip():
            lines.append(f"workdir = {_toml_string(plan.paths.workdir)}")
        if str(plan.paths.zone_file or "").strip():
            lines.append(f"zone_file = {_toml_string(plan.paths.zone_file)}")
        lines.append("")

    lines.extend(_comment_block("PARAMS"))
    lines.append("[video.fastpass.params]")
    for key, value in plan.video.fastpass_params.items():
        lines.append(_render_param_line(str(key), value))
    lines.append("")
    lines.append("[video.mainpass.params]")
    for key, value in plan.video.mainpass_params.items():
        lines.append(_render_param_line(str(key), value))
    lines.append("")

    lines.extend(_comment_block("PROCESS"))
    lines.extend(
        [
            "[video.primary]",
            f"fastpass_workers = {int(plan.video.primary.fastpass_workers)}",
            f"mainpass_workers = {int(plan.video.primary.mainpass_workers)}",
            f"ab_pos_dev = {_toml_scalar(plan.video.primary.ab_pos_dev)}",
            f"ab_pos_multiplier = {_toml_string(str(plan.video.primary.ab_pos_multiplier or ''))}",
            f"ab_multiplier = {_toml_scalar(plan.video.primary.ab_multiplier)}",
            f"ab_neg_multiplier = {_toml_string(str(plan.video.primary.ab_neg_multiplier or ''))}",
            f"ab_neg_dev = {_toml_scalar(plan.video.primary.ab_neg_dev)}",
            f"avg_func = {_toml_string(str(plan.video.primary.avg_func or ''))}",
            f"fastpass_preset = {_toml_scalar(coerce_scalar(plan.video.primary.fastpass_preset))}",
            "",
            "[video.pipeline]",
            f"encoder = {_toml_string(plan.video.primary.encoder)}",
            f"scene_detection = {_toml_string(plan.video.primary.scene_detection)}",
            f"chunk_order = {_toml_string(plan.video.primary.chunk_order)}",
            f"encoder_path = {_toml_string(plan.video.primary.encoder_path)}",
            f"no_fastpass = {_toml_scalar(plan.video.primary.no_fastpass)}",
            "",
            "[video.color]",
            f"fastpass_hdr = {_toml_scalar(plan.video.primary.fastpass_hdr)}",
            f"strict_sdr_8bit = {_toml_scalar(plan.video.primary.strict_sdr_8bit)}",
            f"no_dolby_vision = {_toml_scalar(plan.video.primary.no_dolby_vision)}",
            f"no_hdr10plus = {_toml_scalar(plan.video.primary.no_hdr10plus)}",
            "",
            "[video.pipe]",
            f"fastpass_filter = {_toml_string(plan.video.details.fastpass_filter)}",
            f"mainpass_filter = {_toml_string(plan.video.details.mainpass_filter)}",
            f"main_vpy = {_toml_string(plan.video.details.main_vpy)}",
            f"fast_vpy = {_toml_string(plan.video.details.fast_vpy)}",
            f"proxy_vpy = {_toml_string(plan.video.details.proxy_vpy)}",
            "",
            "[video.experimental]",
            f"vpy_wrapper = {_toml_scalar(plan.video.experimental.vpy_wrapper)}",
            f"source_loader = {_toml_string(normalize_source_loader(plan.video.experimental.source_loader))}",
            f"crop_resize_enabled = {_toml_scalar(plan.video.experimental.crop_resize_enabled)}",
            "",
            "[mux]",
            f"attach_encode_info = {_toml_scalar(plan.video.primary.attach_encode_info)}",
            f"note = {_toml_string(plan.video.details.note)}",
            "",
        ]
    )

    lines.extend(_comment_block("TRACKS"))
    lines.extend(
        [
            "[video]",
            f"track_id = {int(plan.video.track_id)}",
            f"source_name = {_toml_string(plan.video.source_name)}",
            f"source_lang = {_toml_string(plan.video.source_lang)}",
            f"action = {_toml_string(plan.video.action)}",
        ]
    )
    for track in plan.audio:
        lines.extend(
            [
                "",
                "[[audio]]",
                f"track_id = {int(track.track_id)}",
                f"source_name = {_toml_string(track.source_name)}",
                f"source_lang = {_toml_string(track.source_lang)}",
                f"action = {_toml_string(track.action)}",
                f"name = {_toml_string(track.name)}",
                f"lang = {_toml_string(track.lang)}",
                f"default = {_toml_scalar(track.default)}",
                f"forced = {_toml_scalar(track.forced)}",
                f"bitrate_kbps = {int(track.bitrate_kbps)}",
                f"channels = {int(track.channels)}",
            ]
        )
    for track in plan.sub:
        lines.extend(
            [
                "",
                "[[sub]]",
                f"track_id = {int(track.track_id)}",
                f"source_name = {_toml_string(track.source_name)}",
                f"source_lang = {_toml_string(track.source_lang)}",
                f"action = {_toml_string(track.action)}",
                f"name = {_toml_string(track.name)}",
                f"lang = {_toml_string(track.lang)}",
                f"default = {_toml_scalar(track.default)}",
                f"forced = {_toml_scalar(track.forced)}",
            ]
        )
    lines.append("")

    lines.extend(_comment_block("META"))
    lines.extend(
        [
            "[meta]",
            f"name = {_toml_string(plan.meta.name)}",
            f"created_by = {_toml_string(plan.meta.created_by)}",
        ]
    )
    if str(plan.meta.mode or "").strip():
        lines.append(f"mode = {_toml_string(plan.meta.mode)}")
    lines.extend(
        [
            f"format_version = {int(plan.format_version)}",
            f"plan_type = {_toml_string(plan.plan_type)}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _dump_batch_plan(plan: BatchPlan) -> str:
    lines: List[str] = []
    lines.extend(_comment_block("MAIN"))
    lines.append(f"mode = {_toml_string(plan.meta.mode or 'full')}")
    lines.append("")

    lines.extend(_comment_block("ITEMS"))
    for item in plan.items:
        lines.extend(["[[items]]", f"plan = {_toml_string(item.plan)}", ""])

    lines.extend(_comment_block("META"))
    lines.extend(
        [
            "[meta]",
            f"name = {_toml_string(plan.meta.name)}",
            f"created_by = {_toml_string(plan.meta.created_by)}",
            f"format_version = {int(plan.format_version)}",
            f"plan_type = {_toml_string(plan.plan_type)}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "plan_path_for_source",
    "create_default_file_plan",
    "resolve_paths",
    "load_plan",
    "load_file_plan",
    "load_batch_plan",
    "resolve_file_plan",
    "save_plan",
    "dump_plan",
    "make_batch_plan",
    "resolve_batch_plan",
]
