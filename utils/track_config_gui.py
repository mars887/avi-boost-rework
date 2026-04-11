import argparse
import json
import os
import re
import shlex
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plan_model import (
    CHUNK_ORDER_OPTIONS,
    DEFAULT_CHUNK_ORDER,
    build_summary_rows,
    create_default_file_plan,
    file_plan_from_gui_result,
    gui_defaults_from_file_plan,
    gui_settings_from_file_plan,
    load_file_plan,
    plan_path_for_source,
    probe_source_tracks,
    resolve_paths,
    save_plan,
)
from utils.pipeline_runtime import is_mars_av1an_fork, list_portable_encoder_binaries, load_toolchain


TYPE_OPTIONS = ["auto", "video", "audio", "sub"]
MODE_OPTIONS = ["COPY", "EDIT", "SKIP"]
VIDEO_MODE_OPTIONS = ["COPY", "EDIT"]
AUDIO_MODE_OPTIONS = ["COPY", "EDIT", "SKIP"]
SUB_MODE_OPTIONS = ["COPY", "SKIP"]
DEFAULT_OPTIONS = ["auto", "true", "false"]
VIDEO_ENCODER_OPTIONS = ["svt-av1", "libx265"]
TYPE_ORDER = {"video": 0, "audio": 1, "sub": 2}
DEFAULT_VIDEO_ENCODER = "svt-av1"
APP_BG = "#07111a"
APP_SURFACE = "#0d1a29"
APP_SURFACE_ALT = "#122334"
APP_CARD = "#0f2232"
APP_INPUT = "#081722"
APP_BORDER = "#21405a"
APP_TEXT = "#f0fbff"
APP_MUTED = "#89a9bf"
APP_ACCENT = "#35f2ff"
APP_ACCENT_SOFT = "#193e4f"
APP_ACCENT_ALT = "#ff5fd2"
APP_SUCCESS = "#73ffbe"
APP_WARNING = "#ffc66d"
APP_MONO_FONT = ("Consolas", 11)
APP_BODY_FONT = ("Segoe UI", 11)
APP_TITLE_FONT = ("Segoe UI Semibold", 24)
APP_SUBTITLE_FONT = ("Segoe UI", 11)
APP_SECTION_FONT = ("Segoe UI Semibold", 12)
APP_LABEL_FONT = ("Segoe UI Semibold", 11)
APP_BUTTON_FONT = ("Segoe UI Semibold", 11)
STRICT_SDR_8BIT_PARAMS = {
    "svt-av1": {
        "--matrix-coefficients": "1",
        "--transfer-characteristics": "1",
        "--color-primaries": "1",
        "--input-depth": "8",
        "--hbd-mds": "0",
    },
    "libx265": {
        "--colormatrix": "bt709",
        "--transfer": "bt709",
        "--colorprim": "bt709",
    },
}
LIBX265_UNSUPPORTED_FLAGS = {
    "--ac-bias",
    "--cdef-scaling",
    "--chroma-qm-min",
    "--color-primaries",
    "--complex-hvs",
    "--enable-dlf",
    "--enable-restoration",
    "--fast-decode",
    "--film-grain",
    "--hbd-mds",
    "--lp",
    "--matrix-coefficients",
    "--noise-adaptive-filtering",
    "--qm-min",
    "--scm",
    "--sharpness",
    "--sharp-tx",
    "--transfer-characteristics",
    "--variance-boost-curve",
    "--variance-boost-strength",
    "--variance-octile",
}


class DefaultSettings:
    def __init__(
        self,
        params="",
        last_params="",
        zoning="",
        fastpass="",
        mainpass="",
        scene_detection="",
        chunk_order=DEFAULT_CHUNK_ORDER,
        encoder_path="",
        no_fastpass=False,
        fastpass_hdr=True,
        strict_sdr_8bit=False,
        no_dolby_vision=False,
        no_hdr10plus=False,
        fastpass_workers="",
        mainpass_workers="",
        workers="",
        encoder=DEFAULT_VIDEO_ENCODER,
        ab_multiplier="",
        ab_pos_dev="",
        ab_neg_dev="",
        ab_pos_multiplier="",
        ab_neg_multiplier="",
        main_vpy="",
        fast_vpy="",
        proxy_vpy="",
        attach_encode_info=False,
        note="",
    ):
        self.params = params or ""
        self.last_params = last_params or ""
        self.zoning = zoning or ""
        self.fastpass = fastpass or ""
        self.mainpass = mainpass or ""
        sd = (scene_detection or "").strip().lower()
        if sd not in ("psd", "av1an"):
            sd = "av1an"
        self.scene_detection = sd
        self.chunk_order = str(chunk_order or DEFAULT_CHUNK_ORDER).strip() or DEFAULT_CHUNK_ORDER
        if self.chunk_order not in CHUNK_ORDER_OPTIONS:
            self.chunk_order = DEFAULT_CHUNK_ORDER
        self.encoder_path = str(encoder_path or "").strip()
        self.no_fastpass = parse_bool_value(no_fastpass, default=False)
        self.fastpass_hdr = parse_bool_value(fastpass_hdr, default=True)
        self.strict_sdr_8bit = parse_bool_value(strict_sdr_8bit, default=False)
        self.no_dolby_vision = parse_bool_value(no_dolby_vision, default=False)
        self.no_hdr10plus = parse_bool_value(no_hdr10plus, default=False)
        self.workers = workers or ""
        self.fastpass_workers = fastpass_workers or self.workers or ""
        self.mainpass_workers = mainpass_workers or self.workers or ""
        self.encoder = normalize_encoder(encoder)
        self.ab_multiplier = ab_multiplier or ""
        self.ab_pos_dev = ab_pos_dev or ""
        self.ab_neg_dev = ab_neg_dev or ""
        self.ab_pos_multiplier = ab_pos_multiplier or ""
        self.ab_neg_multiplier = ab_neg_multiplier or ""
        self.main_vpy = main_vpy or ""
        self.fast_vpy = fast_vpy or ""
        self.proxy_vpy = proxy_vpy or ""
        self.attach_encode_info = parse_bool_value(attach_encode_info, default=False)
        self.note = note or ""


def normalize_type(value):
    if not value:
        return ""
    val = value.strip().lower()
    if val.startswith("sub"):
        return "sub"
    if val.startswith("vid"):
        return "video"
    if val.startswith("aud"):
        return "audio"
    return val


def parse_id_rule(raw_value):
    if raw_value is None:
        return ("any", None)
    value = str(raw_value).strip()
    if not value or value.lower() == "none":
        return ("any", None)

    lowered = value.lower()
    if lowered.startswith("id="):
        value = value[3:].strip()
        lowered = value.lower()

    if value.isdigit():
        return ("id", int(value))

    if lowered.startswith("name="):
        return ("name", lowered[5:].strip())

    if lowered.startswith("lang="):
        return ("lang", lowered[5:].strip())

    return ("name", lowered)


def parse_params_string(raw_value):
    if not raw_value:
        return {}
    try:
        tokens = shlex.split(str(raw_value))
    except ValueError:
        tokens = str(raw_value).split()
    result = {}
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith("--") and len(token) > 2:
            if "=" in token:
                key, value = token.split("=", 1)
            else:
                key = token
                value = ""
                if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--"):
                    value = tokens[idx + 1]
                    idx += 1
            result[key] = value
        idx += 1
    return result


def normalize_encoder(value):
    raw = str(value or "").strip().lower().replace("_", "-")
    if raw in ("", "auto", "default", "svt-av1"):
        return "svt-av1"
    if raw in ("x265", "libx265"):
        return "libx265"
    return DEFAULT_VIDEO_ENCODER


def get_param_value(params_map, key):
    if not params_map:
        return ""
    value = params_map.get(key)
    if value is None:
        return ""
    return str(value).strip()


def apply_strict_sdr_8bit_params(params_map, encoder):
    result = dict(params_map or {})
    for key, value in STRICT_SDR_8BIT_PARAMS.get(normalize_encoder(encoder), {}).items():
        result[key] = value
    return result


def find_encoder_param_conflicts(encoder, params_map):
    normalized = normalize_encoder(encoder)
    if normalized != "libx265":
        return []

    normalized_items = {
        str(key).replace("^", "-"): str(value).strip()
        for key, value in (params_map or {}).items()
    }
    conflicts = {flag for flag in LIBX265_UNSUPPORTED_FLAGS if flag in normalized_items}

    preset_value = normalized_items.get("--preset", "")
    tune_value = normalized_items.get("--tune", "")
    if preset_value and preset_value.replace(".", "", 1).isdigit():
        conflicts.add("--preset=<numeric>")
    if tune_value and tune_value.replace(".", "", 1).isdigit():
        conflicts.add("--tune=<numeric>")

    return sorted(conflicts)


def parse_int_value(raw_value, default_value):
    if raw_value is None:
        return default_value
    try:
        return int(str(raw_value).strip())
    except ValueError:
        return default_value


def parse_bool_value(raw_value, default=False):
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    text = str(raw_value).strip().lower()
    if not text:
        return default
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


def rule_matches_track(rule, track):
    kind, value = rule
    if kind == "any":
        return True
    if kind == "id":
        return track["track_id"] == value
    if kind == "name":
        return (track.get("name") or "").lower() == value
    if kind == "lang":
        return (track.get("lang") or "").lower() == value
    return False


def setting_matches_track(setting, track):
    rule = parse_id_rule(setting.get("id"))
    if not rule_matches_track(rule, track):
        return False

    stype = setting.get("type")
    if not stype:
        return True

    return normalize_type(track.get("type")) == stype


def infer_type_from_id(raw_value, tracks):
    rule = parse_id_rule(raw_value)
    if rule[0] == "any":
        return None
    types = {normalize_type(t.get("type")) for t in tracks if rule_matches_track(rule, t)}
    return types.pop() if len(types) == 1 else None


def build_result_line(track, mode, name_display, lang_display, default_display, params_display, applied_note, overlap_note):
    name_display = name_display if name_display else "-"
    lang_display = lang_display if lang_display else "-"
    parts = [
        f"{track['track_id']}",
        track.get("type") or "-",
        name_display,
        lang_display,
        mode,
    ]
    if params_display:
        parts.append(params_display)
    if default_display is not None:
        parts.append(f"default={default_display}")
    if applied_note:
        parts.append(applied_note)
    if overlap_note:
        parts.append(overlap_note)
    return " | ".join(parts)


def align_pipe_table(lines):
    parsed = []
    widths = []
    for line in lines:
        raw = str(line or "")
        indent_length = len(raw) - len(raw.lstrip(" "))
        indent = raw[:indent_length]
        body = raw[indent_length:]
        if " | " not in body:
            parsed.append((raw, None, None))
            continue
        parts = body.split(" | ")
        for idx, part in enumerate(parts):
            if idx >= len(widths):
                widths.append(len(part))
            else:
                widths[idx] = max(widths[idx], len(part))
        parsed.append((indent, parts, body))

    out = []
    for first, parts, _body in parsed:
        if parts is None:
            out.append(first)
            continue
        padded = []
        for idx, part in enumerate(parts):
            if idx == len(parts) - 1:
                padded.append(part)
            else:
                padded.append(part.ljust(widths[idx]))
        out.append(first + " | ".join(padded))
    return out


def build_results(files, tracks_by_file, settings, defaults):
    result = {}
    lines = []

    default_params_map = parse_params_string(defaults.params)
    default_last_params_map = parse_params_string(defaults.last_params)
    default_zoning = defaults.zoning
    default_fastpass = defaults.fastpass
    default_mainpass = defaults.mainpass
    default_scene_detection = defaults.scene_detection
    default_no_fastpass = defaults.no_fastpass
    default_fastpass_hdr = defaults.fastpass_hdr
    default_strict_sdr_8bit = defaults.strict_sdr_8bit
    default_no_dolby_vision = defaults.no_dolby_vision
    default_no_hdr10plus = defaults.no_hdr10plus
    default_fastpass_workers = defaults.fastpass_workers or defaults.workers
    default_mainpass_workers = defaults.mainpass_workers or defaults.workers
    default_encoder = normalize_encoder(defaults.encoder)
    default_ab_multiplier = defaults.ab_multiplier
    default_ab_pos_dev = defaults.ab_pos_dev
    default_ab_neg_dev = defaults.ab_neg_dev
    default_ab_pos_multiplier = defaults.ab_pos_multiplier
    default_ab_neg_multiplier = defaults.ab_neg_multiplier
    default_main_vpy = defaults.main_vpy
    default_fast_vpy = defaults.fast_vpy
    default_proxy_vpy = defaults.proxy_vpy
    default_attach_encode_info = defaults.attach_encode_info
    default_note = defaults.note

    for file_index, file_path in enumerate(files, start=1):
        lines.append(f"[{file_index}] {os.path.basename(file_path)}")
        tracks = tracks_by_file.get(file_index, [])

        def sort_key(track):
            return (
                track["track_id"],
                TYPE_ORDER.get(normalize_type(track.get("type")), 9),
                (track.get("name") or "").lower(),
                (track.get("lang") or "").lower(),
            )

        entries = []
        for track in sorted(tracks, key=sort_key):
            match_indexes = [idx for idx, setting in enumerate(settings) if setting_matches_track(setting, track)]
            applied_idx = match_indexes[-1] if match_indexes else None
            applied = settings[applied_idx] if applied_idx is not None else None

            entries.append(
                {
                    "track": track,
                    "type_norm": normalize_type(track.get("type")),
                    "mode": (applied.get("mode") if applied else None) or "SKIP",
                    "params": applied.get("params") if applied else "",
                    "last_params": applied.get("last_params") if applied else "",
                    "bitrate": applied.get("bitrate") if applied else "",
                    "channels": applied.get("channels") if applied else "",
                    "name": applied.get("name") if applied else "",
                    "lang": applied.get("lang") if applied else "",
                    "default_raw": applied.get("default") if applied else None,
                    "applied_idx": applied_idx,
                    "match_indexes": match_indexes,
                }
            )

        def apply_default(entries_for_type):
            for idx, entry in enumerate(entries_for_type):
                raw = entry["default_raw"]
                if raw is True or raw is False:
                    entry["default_final"] = raw
                    continue
                if idx == 0:
                    ok = all(sub["default_raw"] in (False, None) for sub in entries_for_type[1:])
                    entry["default_final"] = True if ok else False
                else:
                    entry["default_final"] = False

        audio_entries = sorted(
            [entry for entry in entries if entry["type_norm"] == "audio"],
            key=lambda entry: entry["track"]["track_id"],
        )
        sub_entries = sorted(
            [entry for entry in entries if entry["type_norm"] == "sub"],
            key=lambda entry: entry["track"]["track_id"],
        )
        apply_default(audio_entries)
        apply_default(sub_entries)

        track_results = []
        for entry in entries:
            track = entry["track"]
            track_type = entry["type_norm"]
            mode = entry["mode"]

            final_name = track.get("name") or ""
            final_lang = track.get("lang") or ""
            if track_type == "audio" and mode in ("COPY", "EDIT"):
                if entry["name"]:
                    final_name = entry["name"]
                if entry["lang"]:
                    final_lang = entry["lang"]
            elif track_type == "sub" and mode == "COPY":
                if entry["name"]:
                    final_name = entry["name"]
                if entry["lang"]:
                    final_lang = entry["lang"]
            elif track_type == "video":
                if entry["lang"]:
                    final_lang = entry["lang"]

            default_display = None
            default_value = None
            if track_type in ("audio", "sub"):
                default_value = entry.get("default_final", False)
                default_display = "true" if default_value else "false"

            track_param = {}
            video_config = None
            params_display = ""
            if track_type == "video" and mode == "EDIT":
                # Video params are global for the current GUI flow and come from the Video tab.
                # Per-track video param overrides are no longer part of the active contract.
                param_map = dict(default_params_map)
                last_param_map = dict(default_last_params_map)
                if default_strict_sdr_8bit:
                    last_param_map = apply_strict_sdr_8bit_params(last_param_map, default_encoder)
                fastpass_crf = get_param_value(param_map, "--crf")
                mainpass_crf = get_param_value(last_param_map, "--crf")
                video_config = {
                    "quality": mainpass_crf or fastpass_crf or "",
                    "fastpass_crf": fastpass_crf,
                    "mainpass_crf": mainpass_crf,
                    "fastpass_preset": get_param_value(param_map, "--preset"),
                    "preset": get_param_value(last_param_map, "--preset"),
                    "fastpass_params": dict(param_map),
                    "mainpass_params": dict(last_param_map),
                }
                param_parts = []
                param_parts.append(f"encoder={default_encoder}")
                params_display = ", ".join(param_parts)
            elif track_type == "audio" and mode == "EDIT":
                bitrate = parse_int_value(entry["bitrate"], 2)
                channels = parse_int_value(entry["channels"], 2)
                track_param["bitrate"] = str(bitrate)
                track_param["channels"] = str(channels)
                params_display = f"bitrate={bitrate}, channels={channels}"

            track_mux = {}
            if track_type == "video":
                track_mux["encoder"] = default_encoder
                track_mux["zoning"] = default_zoning
                track_mux["fastpass"] = default_fastpass
                track_mux["mainpass"] = default_mainpass
                track_mux["sceneDetection"] = default_scene_detection
                track_mux["noFastpass"] = "true" if default_no_fastpass else "false"
                track_mux["fastpassHdr"] = "true" if default_fastpass_hdr else "false"
                track_mux["strictSdr8bit"] = "true" if default_strict_sdr_8bit else "false"
                track_mux["noDolbyVision"] = "true" if default_no_dolby_vision or default_strict_sdr_8bit else "false"
                track_mux["noHdr10Plus"] = "true" if default_no_hdr10plus or default_strict_sdr_8bit else "false"
                track_mux["fastpassWorkers"] = default_fastpass_workers
                track_mux["mainpassWorkers"] = default_mainpass_workers
                track_mux["workers"] = default_fastpass_workers
                track_mux["abMultiplier"] = default_ab_multiplier
                track_mux["abPosDev"] = default_ab_pos_dev
                track_mux["abNegDev"] = default_ab_neg_dev
                track_mux["abPosMultiplier"] = default_ab_pos_multiplier
                track_mux["abNegMultiplier"] = default_ab_neg_multiplier
                track_mux["mainVpy"] = default_main_vpy
                track_mux["fastVpy"] = default_fast_vpy
                track_mux["proxyVpy"] = default_proxy_vpy
                track_mux["attachEncodeInfo"] = "true" if default_attach_encode_info else "false"
                if default_note:
                    track_mux["note"] = default_note
            elif track_type in ("audio", "sub"):
                if final_name:
                    track_mux["name"] = final_name
                if final_lang:
                    track_mux["lang"] = final_lang
                if default_value is not None:
                    track_mux["default"] = "true" if default_value else "false"

            orig_name = track.get("name") or ""
            orig_lang = track.get("lang") or ""
            if track_type == "video" and entry["lang"]:
                orig_lang = entry["lang"]

            track_results.append(
                {
                    "fileIndex": file_index,
                    "trackId": track["track_id"],
                    "type": track.get("type") or "",
                    "origName": orig_name,
                    "origLang": orig_lang,
                    "trackStatus": mode,
                    "trackParam": track_param,
                    "videoConfig": video_config,
                    "trackMux": track_mux,
                }
            )

            applied_note = ""
            if entry["applied_idx"] is not None:
                applied_note = f"setting #{entry['applied_idx'] + 1}"
            else:
                applied_note = "no setting"

            overlap_note = ""
            if len(entry["match_indexes"]) > 1:
                overlap_note = "overlap: " + ", ".join(f"#{idx + 1}" for idx in entry["match_indexes"])

            name_display = final_name if track_type in ("audio", "sub") else "-"
            lang_display = final_lang
            line = build_result_line(
                track,
                mode,
                name_display,
                lang_display,
                default_display,
                params_display,
                applied_note,
                overlap_note,
            )
            lines.append("  " + line)

        result[file_path] = track_results
        lines.append("")

    return result, lines


def build_default_defaults_dict():
    return {
        "params": "--variance-boost-strength 2 --variance-octile 6 --variance-boost-curve 3 --tune 0 --qm-min 7 --chroma-qm-min 10 --scm 0 --enable-dlf 2 --sharp-tx 1 --enable-restoration 0 --lp 3 --sharpness 1 --hbd-mds 1 --ac-bias 2.00 --crf 30",
        "last_params": "--film-grain 14 --complex-hvs 1 --crf 30",
        "zoning": "",
        "fastpass": "",
        "mainpass": "",
        "scene_detection": "av1an",
        "chunk_order": DEFAULT_CHUNK_ORDER,
        "encoder_path": "",
        "no_fastpass": False,
        "fastpass_hdr": True,
        "strict_sdr_8bit": False,
        "no_dolby_vision": False,
        "no_hdr10plus": False,
        "fastpass_workers": "8",
        "mainpass_workers": "8",
        "workers": "",
        "encoder": DEFAULT_VIDEO_ENCODER,
        "ab_multiplier": "0.7",
        "ab_pos_dev": "5",
        "ab_neg_dev": "4",
        "ab_pos_multiplier": "",
        "ab_neg_multiplier": "",
        "main_vpy": "",
        "fast_vpy": "",
        "proxy_vpy": "",
        "attach_encode_info": False,
        "note": "",
    }


def load_gui_data_from_paths(raw_paths):
    files = []
    plan_paths = {}
    tracks_by_file = {}
    defaults = build_default_defaults_dict()
    settings = []

    normalized_paths = [Path(item).expanduser().resolve() for item in raw_paths]
    if not normalized_paths:
        raise RuntimeError("No input paths provided")

    for index, path in enumerate(normalized_paths, start=1):
        if path.suffix.lower() == ".plan":
            plan = load_file_plan(path)
            source = resolve_paths(plan, path).source
            if len(normalized_paths) == 1:
                defaults.update(gui_defaults_from_file_plan(plan))
                settings = gui_settings_from_file_plan(plan)
            plan_paths[str(source)] = str(path)
            files.append(str(source))
            tracks_by_file[index] = list(probe_source_tracks(source))
            continue

        source = path
        if not source.exists():
            raise RuntimeError(f"Input file not found: {source}")
        existing_plan = plan_path_for_source(source)
        if existing_plan.exists() and len(normalized_paths) == 1:
            plan = load_file_plan(existing_plan)
            defaults.update(gui_defaults_from_file_plan(plan))
            settings = gui_settings_from_file_plan(plan)
        files.append(str(source))
        plan_paths[str(source)] = str(existing_plan)
        tracks_by_file[index] = list(probe_source_tracks(source))

    summary = build_summary_rows([Path(item) for item in files], tracks_by_file)
    return {
        "files": files,
        "summary": summary,
        "defaults": defaults,
        "settings": settings,
        "planPaths": plan_paths,
        "outputMode": "plans",
    }


class TrackConfigGui:
    def __init__(self, data):
        self.files = data.get("files") or []
        self.summary = data.get("summary") or []
        defaults_raw = data.get("defaults") or {}
        self.defaults = DefaultSettings(
            params=defaults_raw.get("params") or "",
            last_params=defaults_raw.get("lastParams") or defaults_raw.get("last_params") or "",
            zoning=defaults_raw.get("zoning") or "",
            fastpass=defaults_raw.get("fastpass") or "",
            mainpass=defaults_raw.get("mainpass") or "",
            scene_detection=defaults_raw.get("sceneDetection") or defaults_raw.get("scene_detection") or "",
            chunk_order=defaults_raw.get("chunkOrder") or defaults_raw.get("chunk_order") or DEFAULT_CHUNK_ORDER,
            encoder_path=defaults_raw.get("encoderPath") or defaults_raw.get("encoder_path") or "",
            no_fastpass=defaults_raw.get("noFastpass") or defaults_raw.get("no_fastpass") or False,
            fastpass_hdr=defaults_raw["fastpassHdr"] if "fastpassHdr" in defaults_raw else defaults_raw.get("fastpass_hdr", True),
            strict_sdr_8bit=defaults_raw.get("strictSdr8bit") if "strictSdr8bit" in defaults_raw else defaults_raw.get("strict_sdr_8bit", False),
            no_dolby_vision=defaults_raw.get("noDolbyVision") or defaults_raw.get("no_dolby_vision") or False,
            no_hdr10plus=defaults_raw.get("noHdr10Plus") or defaults_raw.get("no_hdr10plus") or False,
            fastpass_workers=defaults_raw.get("fastpassWorkers") or defaults_raw.get("fastpass_workers") or "",
            mainpass_workers=defaults_raw.get("mainpassWorkers") or defaults_raw.get("mainpass_workers") or "",
            workers=defaults_raw.get("workers") or "",
            encoder=defaults_raw.get("encoder") or DEFAULT_VIDEO_ENCODER,
            ab_multiplier=defaults_raw.get("abMultiplier") or defaults_raw.get("ab_multiplier") or "",
            ab_pos_dev=defaults_raw.get("abPosDev") or defaults_raw.get("ab_pos_dev") or "",
            ab_neg_dev=defaults_raw.get("abNegDev") or defaults_raw.get("ab_neg_dev") or "",
            ab_pos_multiplier=defaults_raw.get("abPosMultiplier") or defaults_raw.get("ab_pos_multiplier") or "",
            ab_neg_multiplier=defaults_raw.get("abNegMultiplier") or defaults_raw.get("ab_neg_multiplier") or "",
            main_vpy=defaults_raw.get("mainVpy") or defaults_raw.get("main_vpy") or "",
            fast_vpy=defaults_raw.get("fastVpy") or defaults_raw.get("fast_vpy") or "",
            proxy_vpy=defaults_raw.get("proxyVpy") or defaults_raw.get("proxy_vpy") or "",
            attach_encode_info=defaults_raw["attachEncodeInfo"] if "attachEncodeInfo" in defaults_raw else defaults_raw.get("attach_encode_info", False),
            note=defaults_raw.get("note") or "",
        )
        self.plan_paths = data.get("planPaths") or {}
        self.output_mode = data.get("outputMode") or "plans"
        self.settings = list(data.get("settings") or [])
        toolchain = load_toolchain()
        self.av1an_exe = toolchain.av1an_exe
        self.av1an_fork_enabled = is_mars_av1an_fork(self.av1an_exe)

        self.match_tracks = []
        for row in self.summary:
            self.match_tracks.append(
                {
                    "track_id": row.get("index"),
                    "type": row.get("type"),
                    "name": row.get("displayName") or "",
                    "lang": row.get("lang") or "",
                }
            )

        self.tracks_by_file = {idx + 1: [] for idx in range(len(self.files))}
        for row in self.summary:
            for file_index in row.get("presentIn") or []:
                self.tracks_by_file.setdefault(file_index, []).append(
                    {
                        "file_index": file_index,
                        "track_id": row.get("index"),
                        "type": row.get("type"),
                        "name": row.get("displayName") or "",
                        "lang": row.get("lang") or "",
                    }
                )

        self.root = tk.Tk()
        self.root.title("Track Config")
        self.root.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self._int_vcmd = (self.root.register(self._validate_number_entry), "%P", "int")
        self._float_vcmd = (self.root.register(self._validate_number_entry), "%P", "float")
        self._configure_theme()

        self._build_ui()
        self._refresh_summary()
        self._refresh_settings()
        self._refresh_results()

    def _configure_theme(self):
        self.root.configure(bg=APP_BG)
        try:
            self.root.tk.call("tk", "scaling", 1.10)
        except Exception:
            pass
        self.root.option_add("*Font", APP_BODY_FONT)
        self.root.option_add("*TCombobox*Listbox*Background", APP_INPUT)
        self.root.option_add("*TCombobox*Listbox*Foreground", APP_TEXT)
        self.root.option_add("*TCombobox*Listbox*selectBackground", APP_ACCENT_SOFT)
        self.root.option_add("*TCombobox*Listbox*selectForeground", APP_TEXT)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background=APP_BG, foreground=APP_TEXT, fieldbackground=APP_INPUT)
        style.configure("App.TFrame", background=APP_BG)
        style.configure("Surface.TFrame", background=APP_SURFACE)
        style.configure("Card.TFrame", background=APP_CARD, relief="flat")
        style.configure("CardHeader.TFrame", background=APP_CARD)
        style.configure("Toolbar.TFrame", background=APP_SURFACE_ALT)
        style.configure("Inline.TFrame", background=APP_CARD)
        style.configure("TLabel", background=APP_BG, foreground=APP_TEXT, font=APP_BODY_FONT)
        style.configure("Surface.TLabel", background=APP_SURFACE, foreground=APP_TEXT, font=APP_BODY_FONT)
        style.configure("Card.TLabel", background=APP_CARD, foreground=APP_TEXT, font=APP_BODY_FONT)
        style.configure("Title.TLabel", background=APP_BG, foreground=APP_TEXT, font=APP_TITLE_FONT)
        style.configure("Subtitle.TLabel", background=APP_BG, foreground=APP_MUTED, font=APP_SUBTITLE_FONT)
        style.configure("CardTitle.TLabel", background=APP_CARD, foreground=APP_TEXT, font=APP_SECTION_FONT)
        style.configure("CardSubtitle.TLabel", background=APP_CARD, foreground=APP_MUTED, font=APP_SUBTITLE_FONT)
        style.configure("Toolbar.TLabel", background=APP_SURFACE_ALT, foreground=APP_MUTED, font=APP_SUBTITLE_FONT)
        style.configure("SectionLabel.TLabel", background=APP_CARD, foreground=APP_ACCENT, font=APP_LABEL_FONT)
        style.configure("Hint.TLabel", background=APP_CARD, foreground=APP_MUTED, font=APP_SUBTITLE_FONT)
        style.configure("Chip.TLabel", background=APP_ACCENT_SOFT, foreground=APP_ACCENT, font=("Segoe UI Semibold", 10), padding=(10, 5))
        style.configure("GoodChip.TLabel", background="#133426", foreground=APP_SUCCESS, font=("Segoe UI Semibold", 10), padding=(10, 5))
        style.configure("WarnChip.TLabel", background="#3a2a12", foreground=APP_WARNING, font=("Segoe UI Semibold", 10), padding=(10, 5))
        style.configure("TEntry", fieldbackground=APP_INPUT, foreground=APP_TEXT, insertcolor=APP_TEXT, bordercolor=APP_BORDER, lightcolor=APP_BORDER, darkcolor=APP_BORDER, padding=6)
        style.configure("TCombobox", fieldbackground=APP_INPUT, foreground=APP_TEXT, background=APP_INPUT, arrowcolor=APP_ACCENT, bordercolor=APP_BORDER, lightcolor=APP_BORDER, darkcolor=APP_BORDER, padding=4)
        style.map("TCombobox", fieldbackground=[("readonly", APP_INPUT)], foreground=[("readonly", APP_TEXT)], selectbackground=[("readonly", APP_INPUT)])
        style.configure("TCheckbutton", background=APP_CARD, foreground=APP_TEXT, font=APP_BODY_FONT)
        style.map("TCheckbutton", background=[("active", APP_CARD)], foreground=[("active", APP_ACCENT)])
        style.configure("TNotebook", background=APP_SURFACE, borderwidth=0, tabmargins=(0, 0, 0, 0))
        style.configure("TNotebook.Tab", background=APP_SURFACE_ALT, foreground=APP_MUTED, padding=(18, 10), font=("Segoe UI Semibold", 11))
        style.map("TNotebook.Tab", background=[("selected", APP_CARD), ("active", APP_SURFACE_ALT)], foreground=[("selected", APP_TEXT), ("active", APP_TEXT)])
        style.configure("Treeview", background=APP_INPUT, fieldbackground=APP_INPUT, foreground=APP_TEXT, bordercolor=APP_BORDER, lightcolor=APP_BORDER, darkcolor=APP_BORDER, rowheight=30)
        style.map("Treeview", background=[("selected", APP_ACCENT_SOFT)], foreground=[("selected", APP_TEXT)])
        style.configure("Treeview.Heading", background=APP_SURFACE_ALT, foreground=APP_ACCENT, relief="flat", font=("Segoe UI Semibold", 10))
        style.map("Treeview.Heading", background=[("active", APP_SURFACE_ALT)])
        style.configure("TScrollbar", background=APP_SURFACE_ALT, troughcolor=APP_BG, bordercolor=APP_BG, arrowcolor=APP_ACCENT)
        style.configure("Accent.TButton", background=APP_ACCENT_SOFT, foreground=APP_ACCENT, bordercolor=APP_ACCENT, lightcolor=APP_ACCENT_SOFT, darkcolor=APP_ACCENT_SOFT, font=APP_BUTTON_FONT, padding=(14, 8))
        style.map("Accent.TButton", background=[("active", "#255469"), ("pressed", "#21495c")], foreground=[("disabled", APP_MUTED)])
        style.configure("Secondary.TButton", background=APP_SURFACE_ALT, foreground=APP_TEXT, bordercolor=APP_BORDER, lightcolor=APP_SURFACE_ALT, darkcolor=APP_SURFACE_ALT, font=APP_BUTTON_FONT, padding=(12, 8))
        style.map("Secondary.TButton", background=[("active", "#1b344c"), ("pressed", "#162c40")])
        style.configure("Ghost.TButton", background=APP_CARD, foreground=APP_MUTED, bordercolor=APP_BORDER, lightcolor=APP_CARD, darkcolor=APP_CARD, font=APP_BUTTON_FONT, padding=(10, 8))
        style.map("Ghost.TButton", foreground=[("active", APP_ACCENT)], background=[("active", APP_SURFACE_ALT)])
        style.configure("Danger.TButton", background="#3a1832", foreground=APP_ACCENT_ALT, bordercolor=APP_ACCENT_ALT, lightcolor="#3a1832", darkcolor="#3a1832", font=APP_BUTTON_FONT, padding=(10, 8))
        style.map("Danger.TButton", background=[("active", "#512147")])
        style.configure("Section.TLabelframe", background=APP_CARD, bordercolor=APP_BORDER, lightcolor=APP_BORDER, darkcolor=APP_BORDER, relief="solid", borderwidth=1)
        style.configure("Section.TLabelframe.Label", background=APP_CARD, foreground=APP_ACCENT, font=APP_LABEL_FONT)
        style.configure("TPanedwindow", background=APP_BG, sashthickness=8)

    def _style_text_widget(self, widget, *, monospace=False):
        widget.configure(
            bg=APP_INPUT,
            fg=APP_TEXT,
            insertbackground=APP_ACCENT,
            selectbackground=APP_ACCENT_SOFT,
            selectforeground=APP_TEXT,
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=APP_BORDER,
            highlightcolor=APP_ACCENT,
            padx=10,
            pady=10,
            font=APP_MONO_FONT if monospace else APP_BODY_FONT,
        )

    def _make_card(self, parent, *, title, subtitle="", style="Card.TFrame", padding=14):
        card = ttk.Frame(parent, style=style, padding=padding)
        header = ttk.Frame(card, style="CardHeader.TFrame")
        header.pack(fill=tk.X)
        ttk.Label(header, text=title, style="CardTitle.TLabel").pack(anchor=tk.W)
        body = ttk.Frame(card, style=style)
        body.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        return card, body

    def _add_stat_chip(self, parent, label, value, *, style="Chip.TLabel"):
        text = f"{label}: {value}"
        ttk.Label(parent, text=text, style=style).pack(side=tk.LEFT, padx=(0, 8))

    def _create_scrolled_text_panel(self, parent, *, title, subtitle="", monospace=True):
        card, body = self._make_card(parent, title=title, subtitle=subtitle)
        card.pack(fill=tk.BOTH, expand=True)
        text_wrap = ttk.Frame(body, style="Card.TFrame")
        text_wrap.pack(fill=tk.BOTH, expand=True)
        text = tk.Text(text_wrap, height=10, wrap="none")
        self._style_text_widget(text, monospace=monospace)
        text.configure(state="disabled")
        text.grid(row=0, column=0, sticky="nsew")
        text_wrap.columnconfigure(0, weight=1)
        text_wrap.rowconfigure(0, weight=1)
        self._bind_vertical_scroll(text)
        return card, text

    def _build_section(self, parent, *, title, description="", columns=2):
        section = ttk.LabelFrame(parent, text=title, style="Section.TLabelframe", padding=14)
        start_row = 0
        for col in range(columns):
            section.columnconfigure(col * 2 + 1, weight=1)
        return section, start_row

    def _build_labeled_entry(self, parent, row, column, label, variable, *, width=28, numeric=None):
        ttk.Label(parent, text=label, style="SectionLabel.TLabel").grid(row=row, column=column * 2, sticky=tk.W, padx=(0, 8), pady=6)
        entry_options = {"textvariable": variable, "width": width}
        if numeric == "int":
            entry_options["validate"] = "key"
            entry_options["validatecommand"] = self._int_vcmd
        elif numeric == "float":
            entry_options["validate"] = "key"
            entry_options["validatecommand"] = self._float_vcmd
        entry = ttk.Entry(parent, **entry_options)
        entry.grid(row=row, column=column * 2 + 1, sticky=tk.EW, pady=6)
        return entry

    def _build_labeled_combo(self, parent, row, column, label, variable, values, *, width=26, state="readonly"):
        ttk.Label(parent, text=label, style="SectionLabel.TLabel").grid(row=row, column=column * 2, sticky=tk.W, padx=(0, 8), pady=6)
        combo = ttk.Combobox(parent, textvariable=variable, values=values, width=width, state=state)
        combo.grid(row=row, column=column * 2 + 1, sticky=tk.EW, pady=6)
        return combo

    def _build_labeled_text(self, parent, row, title, widget, *, description="", height=4):
        header = ttk.Frame(parent, style="Card.TFrame")
        header.grid(row=row, column=0, columnspan=4, sticky=tk.EW, pady=(0, 6))
        ttk.Label(header, text=title, style="SectionLabel.TLabel").pack(anchor=tk.W)
        widget.configure(height=height)
        self._style_text_widget(widget, monospace=True)
        widget.grid(row=row + 1, column=0, columnspan=4, sticky=tk.EW, pady=(0, 12))

    def _build_stacked_entry(self, parent, row, label, variable, *, width=36):
        ttk.Label(parent, text=label, style="SectionLabel.TLabel").grid(row=row, column=0, sticky=tk.W, pady=(0, 4))
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row + 1, column=0, sticky=tk.EW, pady=(0, 10))
        return entry

    def _bind_vertical_scroll(self, widget, *, target=None):
        scroll_target = target or widget

        def _on_mousewheel(event):
            try:
                first, last = scroll_target.yview()
            except Exception:
                return None
            if first <= 0.0 and last >= 1.0:
                return "break"
            delta = -1 if event.delta > 0 else 1
            scroll_target.yview_scroll(delta, "units")
            return "break"

        widget.bind("<MouseWheel>", _on_mousewheel, add="+")

    def _validate_number_entry(self, proposed, mode):
        if proposed == "":
            return True
        if mode == "int":
            return proposed.isdigit()
        if mode == "float":
            return bool(re.fullmatch(r"\d*(?:\.\d*)?", proposed))
        return True

    def _create_scrollable_area(self, parent):
        wrap = ttk.Frame(parent, style="Surface.TFrame")
        wrap.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(wrap, background=APP_SURFACE, highlightthickness=0, borderwidth=0)
        inner = ttk.Frame(canvas, style="Surface.TFrame")
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _sync_inner(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(window_id, width=canvas.winfo_width())

        inner.bind("<Configure>", _sync_inner)
        canvas.bind("<Configure>", _sync_inner)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def _on_mousewheel(event):
            bbox = canvas.bbox("all")
            if not bbox:
                return "break"
            content_height = bbox[3] - bbox[1]
            if content_height <= canvas.winfo_height() + 2:
                return "break"
            delta = -1 if event.delta > 0 else 1
            canvas.yview_scroll(delta, "units")
            return "break"

        for widget in (wrap, canvas, inner):
            widget.bind("<MouseWheel>", _on_mousewheel, add="+")
        return inner

    def _build_ui(self):
        self.root.geometry("1560x920")
        self.root.minsize(1320, 760)

        shell = ttk.Frame(self.root, style="App.TFrame", padding=(18, 16, 18, 18))
        shell.pack(fill=tk.BOTH, expand=True)
        shell.rowconfigure(0, weight=1)
        shell.columnconfigure(0, weight=1)

        body = ttk.Panedwindow(shell, orient=tk.HORIZONTAL, style="TPanedwindow")
        body.grid(row=0, column=0, sticky="nsew")

        left_frame = ttk.Frame(body, style="App.TFrame")
        center_frame = ttk.Frame(body, style="App.TFrame")
        right_frame = ttk.Frame(body, style="App.TFrame")
        body.add(left_frame, weight=24)
        body.add(center_frame, weight=52)
        body.add(right_frame, weight=24)

        summary_card, self.summary_text = self._create_scrolled_text_panel(
            left_frame,
            title="Tracks Summary",
            subtitle="Discovered tracks across selected sources. Use this as the routing map for your rules.",
            monospace=True,
        )
        summary_meta = ttk.Frame(left_frame, style="App.TFrame")
        summary_meta.pack(fill=tk.X, pady=(10, 0))
        self._add_stat_chip(summary_meta, "Files", len(self.files))
        self._add_stat_chip(summary_meta, "Rules", len(self.settings))
        fork_style = "GoodChip.TLabel" if self.av1an_fork_enabled else "WarnChip.TLabel"
        fork_value = "mars887 fork" if self.av1an_fork_enabled else "standard av1an"
        self._add_stat_chip(summary_meta, "av1an", fork_value, style=fork_style)
        self._build_center(center_frame)
        _result_card, self.result_text = self._create_scrolled_text_panel(
            right_frame,
            title="Result Preview",
            subtitle="Live preview of the normalized output contract that will be written into .plan files.",
            monospace=True,
        )

    def _build_center(self, frame):
        card, body = self._make_card(
            frame,
            title="Pipeline Editor",
            subtitle="Split by intent: detailed routing in Track Rules, global encode behavior in Video Pipeline.",
            style="Surface.TFrame",
            padding=16,
        )
        card.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(body)
        notebook.pack(fill=tk.BOTH, expand=True)

        video_tab = ttk.Frame(notebook, style="Surface.TFrame")
        tracks_tab = ttk.Frame(notebook, style="Surface.TFrame")

        notebook.add(video_tab, text="Video Pipeline")
        notebook.add(tracks_tab, text="Track Rules")

        self._build_video_tab(video_tab)
        self._build_tracks_tab(tracks_tab)

    def _build_video_tab(self, frame):
        content = self._create_scrollable_area(frame)
        grid = ttk.Frame(content, style="Surface.TFrame", padding=(6, 6, 6, 16))
        grid.pack(fill=tk.BOTH, expand=True)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        self.default_fastpass_var = tk.StringVar(value=self.defaults.fastpass)
        self.default_mainpass_var = tk.StringVar(value=self.defaults.mainpass)
        self.scene_detection_var = tk.StringVar(value=self.defaults.scene_detection or "av1an")
        self.encoder_var = tk.StringVar(value=self.defaults.encoder or DEFAULT_VIDEO_ENCODER)
        self.chunk_order_var = tk.StringVar(value=self.defaults.chunk_order or DEFAULT_CHUNK_ORDER)
        self.encoder_path_var = tk.StringVar(value=self.defaults.encoder_path)
        self.no_fastpass_var = tk.BooleanVar(value=bool(self.defaults.no_fastpass))
        self.fastpass_hdr_var = tk.BooleanVar(value=bool(self.defaults.fastpass_hdr))
        self.strict_sdr_8bit_var = tk.BooleanVar(value=bool(self.defaults.strict_sdr_8bit))
        self.no_dolby_vision_var = tk.BooleanVar(value=bool(self.defaults.no_dolby_vision))
        self.no_hdr10plus_var = tk.BooleanVar(value=bool(self.defaults.no_hdr10plus))
        self.fastpass_workers_var = tk.StringVar(value=self.defaults.fastpass_workers or self.defaults.workers)
        self.mainpass_workers_var = tk.StringVar(value=self.defaults.mainpass_workers or self.defaults.workers)
        self.ab_multiplier_var = tk.StringVar(value=self.defaults.ab_multiplier)
        self.ab_pos_multiplier_var = tk.StringVar(value=self.defaults.ab_pos_multiplier)
        self.ab_neg_multiplier_var = tk.StringVar(value=self.defaults.ab_neg_multiplier)
        self.ab_pos_dev_var = tk.StringVar(value=self.defaults.ab_pos_dev)
        self.ab_neg_dev_var = tk.StringVar(value=self.defaults.ab_neg_dev)
        self.main_vpy_var = tk.StringVar(value=self.defaults.main_vpy)
        self.fast_vpy_var = tk.StringVar(value=self.defaults.fast_vpy)
        self.proxy_vpy_var = tk.StringVar(value=self.defaults.proxy_vpy)
        self.attach_encode_info_var = tk.BooleanVar(value=bool(self.defaults.attach_encode_info))
        self.note_var = tk.StringVar(value=self.defaults.note)

        params_section, row = self._build_section(
            grid,
            title="Encoder Params",
            columns=2,
        )
        params_section.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=6, pady=6)
        self.default_params_text = tk.Text(params_section, height=4, width=60, wrap="char")
        self.default_last_params_text = tk.Text(params_section, height=2, width=60, wrap="char")
        self.default_zoning_text = tk.Text(params_section, height=5, width=60, wrap="none")
        self._build_labeled_text(
            params_section,
            row,
            "Fast-pass params",
            self.default_params_text,
            height=5,
        )
        self.default_params_text.insert("1.0", self.defaults.params)
        self.default_params_text.edit_modified(False)
        self.default_params_text.bind("<Return>", lambda _event: "break")
        self.default_params_text.bind("<KP_Enter>", lambda _event: "break")
        self.default_params_text.bind("<<Modified>>", self.on_default_text_change)
        row += 2
        self._build_labeled_text(
            params_section,
            row,
            "Main-pass params",
            self.default_last_params_text,
            height=4,
        )
        self.default_last_params_text.insert("1.0", self.defaults.last_params)
        self.default_last_params_text.edit_modified(False)
        self.default_last_params_text.bind("<Return>", lambda _event: "break")
        self.default_last_params_text.bind("<KP_Enter>", lambda _event: "break")
        self.default_last_params_text.bind("<<Modified>>", self.on_default_text_change)
        row += 2
        self._build_labeled_text(
            params_section,
            row,
            "Zone commands",
            self.default_zoning_text,
            height=6,
        )
        self.default_zoning_text.insert("1.0", self.defaults.zoning)
        self.default_zoning_text.edit_modified(False)
        self.default_zoning_text.bind("<<Modified>>", self.on_zoning_change)

        core_section, row = self._build_section(
            grid,
            title="Pipeline Core",
            columns=2,
        )
        core_section.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self._build_labeled_combo(core_section, row, 0, "Scene detection", self.scene_detection_var, ["psd", "av1an"], width=24)
        self._build_labeled_combo(core_section, row, 1, "Encoder", self.encoder_var, VIDEO_ENCODER_OPTIONS, width=24)
        row += 1
        self._build_labeled_entry(core_section, row, 0, "Fast-pass workers", self.fastpass_workers_var, numeric="int")
        self._build_labeled_entry(core_section, row, 1, "Main-pass workers", self.mainpass_workers_var, numeric="int")
        row += 1
        self._build_labeled_entry(core_section, row, 0, "Fast-pass filter", self.default_fastpass_var)
        self._build_labeled_entry(core_section, row, 1, "Main-pass filter", self.default_mainpass_var)
        row += 1
        if self.av1an_fork_enabled:
            self._build_labeled_combo(core_section, row, 0, "Chunk order", self.chunk_order_var, CHUNK_ORDER_OPTIONS, width=24)
            self.encoder_path_combo = self._build_labeled_combo(core_section, row, 1, "Encoder path", self.encoder_path_var, [""], width=24)
            self._refresh_encoder_path_options(preserve_missing=True)

        feature_section, row = self._build_section(
            grid,
            title="Feature Switches",
            columns=1,
        )
        feature_section.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)
        ttk.Checkbutton(feature_section, text="No fast-pass", variable=self.no_fastpass_var).grid(row=row, column=0, sticky=tk.W, pady=4)
        row += 1
        ttk.Checkbutton(feature_section, text="Fast-pass HDR", variable=self.fastpass_hdr_var).grid(row=row, column=0, sticky=tk.W, pady=4)
        row += 1
        ttk.Checkbutton(feature_section, text="Strict SDR 8bit", variable=self.strict_sdr_8bit_var).grid(row=row, column=0, sticky=tk.W, pady=4)
        row += 1
        ttk.Checkbutton(feature_section, text="No Dolby Vision", variable=self.no_dolby_vision_var).grid(row=row, column=0, sticky=tk.W, pady=4)
        row += 1
        ttk.Checkbutton(feature_section, text="No HDR10+", variable=self.no_hdr10plus_var).grid(row=row, column=0, sticky=tk.W, pady=4)
        row += 1
        ttk.Checkbutton(feature_section, text="Attach Encode Info", variable=self.attach_encode_info_var).grid(row=row, column=0, sticky=tk.W, pady=4)

        tuning_section, row = self._build_section(
            grid,
            title="Auto-Boost Tuning",
            columns=2,
        )
        tuning_section.grid(row=2, column=0, sticky="nsew", padx=6, pady=6)
        self._build_labeled_entry(tuning_section, row, 0, "Max + dev", self.ab_pos_dev_var, numeric="int")
        self._build_labeled_entry(tuning_section, row, 1, "Pos multiplier", self.ab_pos_multiplier_var, numeric="float")
        row += 1
        self._build_labeled_entry(tuning_section, row, 0, "Shared multiplier", self.ab_multiplier_var, numeric="float")
        self._build_labeled_entry(tuning_section, row, 1, "Neg multiplier", self.ab_neg_multiplier_var, numeric="float")
        row += 1
        self._build_labeled_entry(tuning_section, row, 0, "Max - dev", self.ab_neg_dev_var, numeric="int")

        script_section, row = self._build_section(
            grid,
            title="Script Paths",
            columns=1,
        )
        script_section.grid(row=2, column=1, sticky="nsew", padx=6, pady=6)
        script_section.columnconfigure(0, weight=1)
        self._build_stacked_entry(script_section, row, "Main vpy", self.main_vpy_var, width=36)
        row += 2
        self._build_stacked_entry(script_section, row, "Fast vpy", self.fast_vpy_var, width=36)
        row += 2
        self._build_stacked_entry(script_section, row, "Proxy vpy", self.proxy_vpy_var, width=36)
        row += 2
        self._build_stacked_entry(script_section, row, "Note", self.note_var, width=36)

        self.default_fastpass_var.trace_add("write", self.on_defaults_change)
        self.default_mainpass_var.trace_add("write", self.on_defaults_change)
        self.scene_detection_var.trace_add("write", self.on_defaults_change)
        self.encoder_var.trace_add("write", self.on_encoder_change)
        if self.av1an_fork_enabled:
            self.chunk_order_var.trace_add("write", self.on_defaults_change)
            self.encoder_path_var.trace_add("write", self.on_defaults_change)
        self.no_fastpass_var.trace_add("write", self.on_defaults_change)
        self.fastpass_hdr_var.trace_add("write", self.on_defaults_change)
        self.strict_sdr_8bit_var.trace_add("write", self.on_defaults_change)
        self.no_dolby_vision_var.trace_add("write", self.on_defaults_change)
        self.no_hdr10plus_var.trace_add("write", self.on_defaults_change)
        self.fastpass_workers_var.trace_add("write", self.on_defaults_change)
        self.mainpass_workers_var.trace_add("write", self.on_defaults_change)
        self.ab_pos_dev_var.trace_add("write", self.on_defaults_change)
        self.ab_neg_dev_var.trace_add("write", self.on_defaults_change)
        self.main_vpy_var.trace_add("write", self.on_defaults_change)
        self.fast_vpy_var.trace_add("write", self.on_defaults_change)
        self.proxy_vpy_var.trace_add("write", self.on_defaults_change)
        self.attach_encode_info_var.trace_add("write", self.on_defaults_change)
        self.note_var.trace_add("write", self.on_defaults_change)

        self.ab_multiplier_var.trace_add("write", self.on_ab_multiplier_change)
        self.ab_pos_multiplier_var.trace_add("write", self.on_ab_pos_multiplier_change)
        self.ab_neg_multiplier_var.trace_add("write", self.on_ab_pos_multiplier_change)

    def _build_tracks_tab(self, frame):
        frame.configure(padding=6)
        workspace = ttk.Frame(frame, style="Surface.TFrame")
        workspace.pack(fill=tk.BOTH, expand=True)
        workspace.rowconfigure(0, weight=1)
        workspace.columnconfigure(0, weight=1)

        split = ttk.Panedwindow(workspace, orient=tk.HORIZONTAL, style="TPanedwindow")
        split.grid(row=0, column=0, sticky="nsew")

        left_host = ttk.Frame(split, style="Surface.TFrame")
        right_host = ttk.Frame(split, style="Surface.TFrame")
        split.add(left_host, weight=60)
        split.add(right_host, weight=40)

        table_card, table_body = self._make_card(
            left_host,
            title="Track Rules",
            style="Card.TFrame",
        )
        table_card.pack(fill=tk.BOTH, expand=True, padx=(0, 8))

        columns = (
            "idx",
            "id",
            "type",
            "mode",
            "bitrate",
            "channels",
            "name",
            "lang",
            "default",
        )
        tree_wrap = ttk.Frame(table_body, style="Card.TFrame")
        tree_wrap.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(tree_wrap, columns=columns, show="headings", height=12)
        self.tree.heading("idx", text="#")
        self.tree.heading("id", text="id")
        self.tree.heading("type", text="type")
        self.tree.heading("mode", text="mode")
        self.tree.heading("bitrate", text="bitrate")
        self.tree.heading("channels", text="channels")
        self.tree.heading("name", text="name")
        self.tree.heading("lang", text="lang")
        self.tree.heading("default", text="default")

        self.tree.column("idx", width=40, anchor=tk.CENTER)
        self.tree.column("id", width=120)
        self.tree.column("type", width=70, anchor=tk.CENTER)
        self.tree.column("mode", width=70, anchor=tk.CENTER)
        self.tree.column("bitrate", width=70, anchor=tk.CENTER)
        self.tree.column("channels", width=70, anchor=tk.CENTER)
        self.tree.column("name", width=120)
        self.tree.column("lang", width=90)
        self.tree.column("default", width=70, anchor=tk.CENTER)
        self.tree.tag_configure("video", background="#0a1b28")
        self.tree.tag_configure("audio", background="#0e1f2f")
        self.tree.tag_configure("sub", background="#102536")

        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_wrap.rowconfigure(0, weight=1)
        tree_wrap.columnconfigure(0, weight=1)
        self._bind_vertical_scroll(self.tree)

        self.tree.bind("<<TreeviewSelect>>", self.on_select_setting)

        editor_card, editor_body = self._make_card(
            right_host,
            title="Rule Inspector",
            style="Card.TFrame",
        )
        editor_card.pack(fill=tk.BOTH, expand=True, padx=(8, 0))

        self.id_var = tk.StringVar()
        self.type_var = tk.StringVar(value="auto")
        self.mode_var = tk.StringVar(value="COPY")
        self.bitrate_var = tk.StringVar()
        self.channels_var = tk.StringVar()
        self.name_var = tk.StringVar()
        self.lang_var = tk.StringVar()
        self.default_var = tk.StringVar(value="auto")

        self.id_var.trace_add("write", self.on_id_change)
        self.type_var.trace_add("write", self.on_type_change)
        self.mode_var.trace_add("write", self.on_mode_change)

        match_section, row = self._build_section(
            editor_body,
            title="Matcher",
            columns=1,
        )
        match_section.pack(fill=tk.X)
        self._add_form_row(match_section, row, "id", self.id_var)
        row += 1
        self._add_combo_row(match_section, row, "type", self.type_var, TYPE_OPTIONS)
        row += 1
        self._add_combo_row(match_section, row, "mode", self.mode_var, MODE_OPTIONS)
        row += 1
        self._add_combo_row(match_section, row, "default", self.default_var, DEFAULT_OPTIONS)

        details_section, row = self._build_section(
            editor_body,
            title="Output Details",
            columns=1,
        )
        details_section.pack(fill=tk.X, pady=(12, 0))
        self._add_form_row(details_section, row, "bitrate", self.bitrate_var, numeric="int")
        row += 1
        self._add_form_row(details_section, row, "channels", self.channels_var, numeric="int")
        row += 1
        self._add_form_row(details_section, row, "name", self.name_var)
        row += 1
        self._add_form_row(details_section, row, "lang", self.lang_var)

        buttons = ttk.Frame(editor_body, style="Card.TFrame")
        buttons.pack(fill=tk.X, pady=(14, 0))
        ttk.Button(buttons, text="Add Rule", command=self.on_add, style="Accent.TButton").pack(side=tk.LEFT)
        ttk.Button(buttons, text="Update Rule", command=self.on_update, style="Secondary.TButton").pack(side=tk.LEFT, padx=8)
        ttk.Button(buttons, text="Remove", command=self.on_remove, style="Danger.TButton").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(buttons, text="Move Up", command=self.on_move_up, style="Ghost.TButton").pack(side=tk.LEFT)
        ttk.Button(buttons, text="Move Down", command=self.on_move_down, style="Ghost.TButton").pack(side=tk.LEFT, padx=8)

        footer = ttk.Frame(workspace, style="Toolbar.TFrame", padding=(14, 12))
        footer.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        actions = ttk.Frame(footer, style="Toolbar.TFrame")
        actions.pack(side=tk.RIGHT)
        ttk.Button(actions, text="Cancel", command=self.on_cancel, style="Ghost.TButton").pack(side=tk.RIGHT)
        ttk.Button(actions, text="Apply and Close", command=self.on_apply, style="Accent.TButton").pack(side=tk.RIGHT, padx=(0, 8))

        self._refresh_field_states()

    def _add_form_row(self, frame, row, label, variable, *, numeric=None):
        ttk.Label(frame, text=label, style="SectionLabel.TLabel").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=6)
        entry_options = {"textvariable": variable, "width": 40}
        if numeric == "int":
            entry_options["validate"] = "key"
            entry_options["validatecommand"] = self._int_vcmd
        elif numeric == "float":
            entry_options["validate"] = "key"
            entry_options["validatecommand"] = self._float_vcmd
        entry = ttk.Entry(frame, **entry_options)
        entry.grid(row=row, column=1, sticky=tk.EW, pady=6)
        if label == "bitrate":
            self.bitrate_entry = entry
        elif label == "channels":
            self.channels_entry = entry
        elif label == "name":
            self.name_entry = entry
        elif label == "lang":
            self.lang_entry = entry

    def _add_combo_row(self, frame, row, label, variable, values):
        ttk.Label(frame, text=label, style="SectionLabel.TLabel").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=6)
        combo = ttk.Combobox(frame, textvariable=variable, values=values, width=37, state="readonly")
        combo.grid(row=row, column=1, sticky=tk.EW, pady=6)
        if label == "mode":
            self.mode_combo = combo
        elif label == "default":
            self.default_combo = combo

    def _refresh_summary(self):
        lines = align_pipe_table([row.get("line") or "" for row in self.summary])
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, "\n".join(lines))
        self.summary_text.configure(state="disabled")

    def _refresh_settings(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for idx, setting in enumerate(self.settings, start=1):
            default_display = "auto"
            if setting.get("default") is True:
                default_display = "true"
            elif setting.get("default") is False:
                default_display = "false"
            values = (
                idx,
                setting.get("id") or "",
                setting.get("type") or "auto",
                setting.get("mode") or "",
                setting.get("bitrate") or "",
                setting.get("channels") or "",
                setting.get("name") or "",
                setting.get("lang") or "",
                default_display,
            )
            tag = normalize_type(setting.get("type") or "")
            if tag not in ("video", "audio", "sub"):
                tag = "audio" if idx % 2 == 0 else "sub"
            self.tree.insert("", tk.END, values=values, tags=(tag,))

    def _refresh_results(self):
        self.defaults = self._current_defaults()
        result, lines = build_results(self.files, self.tracks_by_file, self.settings, self.defaults)
        self.latest_result = result
        aligned_lines = align_pipe_table(lines)
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "\n".join(aligned_lines))
        self.result_text.configure(state="disabled")

    def _current_setting(self):
        default_raw = self.default_var.get().strip().lower()
        if default_raw == "true":
            default_val = True
        elif default_raw == "false":
            default_val = False
        else:
            default_val = None

        setting_type = self.type_var.get().strip().lower()
        if setting_type == "auto":
            setting_type = None

        return {
            "id": self.id_var.get().strip(),
            "type": setting_type,
            "mode": self.mode_var.get().strip().upper(),
            "params": "",
            "last_params": "",
            "bitrate": self.bitrate_var.get().strip(),
            "channels": self.channels_var.get().strip(),
            "name": self.name_var.get().strip(),
            "lang": self.lang_var.get().strip(),
            "default": default_val,
        }

    def _get_single_line_text(self, widget):
        if not widget:
            return ""
        text = widget.get("1.0", tk.END).rstrip("\n")
        if "\n" in text:
            text = " ".join(line.strip() for line in text.splitlines())
        return text.strip()

    def _get_var_value(self, name):
        var = getattr(self, name, None)
        if not var:
            return ""
        return var.get().strip()

    def _current_defaults(self):
        zoning_value = ""
        if hasattr(self, "default_zoning_text"):
            zoning_value = self.default_zoning_text.get("1.0", tk.END).rstrip("\n")
        chunk_order_var = getattr(self, "chunk_order_var", None)
        encoder_path_var = getattr(self, "encoder_path_var", None)
        chunk_order_value = chunk_order_var.get().strip() if chunk_order_var is not None else self.defaults.chunk_order
        encoder_path_value = encoder_path_var.get().strip() if encoder_path_var is not None else self.defaults.encoder_path
        return DefaultSettings(
            params=self._get_single_line_text(getattr(self, "default_params_text", None)),
            last_params=self._get_single_line_text(getattr(self, "default_last_params_text", None)),
            zoning=zoning_value,
            fastpass=self._get_var_value("default_fastpass_var"),
            mainpass=self._get_var_value("default_mainpass_var"),
            scene_detection=self._get_var_value("scene_detection_var"),
            encoder=self._get_var_value("encoder_var"),
            chunk_order=chunk_order_value,
            encoder_path=encoder_path_value,
            no_fastpass=bool(getattr(self, "no_fastpass_var", tk.BooleanVar(value=False)).get()),
            fastpass_hdr=bool(getattr(self, "fastpass_hdr_var", tk.BooleanVar(value=True)).get()),
            strict_sdr_8bit=bool(getattr(self, "strict_sdr_8bit_var", tk.BooleanVar(value=False)).get()),
            no_dolby_vision=bool(getattr(self, "no_dolby_vision_var", tk.BooleanVar(value=False)).get()),
            no_hdr10plus=bool(getattr(self, "no_hdr10plus_var", tk.BooleanVar(value=False)).get()),
            fastpass_workers=self._get_var_value("fastpass_workers_var"),
            mainpass_workers=self._get_var_value("mainpass_workers_var"),
            workers=self.defaults.workers,
            ab_multiplier=self._get_var_value("ab_multiplier_var"),
            ab_pos_dev=self._get_var_value("ab_pos_dev_var"),
            ab_neg_dev=self._get_var_value("ab_neg_dev_var"),
            ab_pos_multiplier=self._get_var_value("ab_pos_multiplier_var"),
            ab_neg_multiplier=self._get_var_value("ab_neg_multiplier_var"),
            main_vpy=self._get_var_value("main_vpy_var"),
            fast_vpy=self._get_var_value("fast_vpy_var"),
            proxy_vpy=self._get_var_value("proxy_vpy_var"),
            attach_encode_info=bool(getattr(self, "attach_encode_info_var", tk.BooleanVar(value=False)).get()),
            note=self._get_var_value("note_var"),
        )

    def _apply_setting_to_form(self, setting):
        self.id_var.set(setting.get("id") or "")
        self.type_var.set(setting.get("type") or "auto")
        self.mode_var.set(setting.get("mode") or "COPY")
        self.bitrate_var.set(setting.get("bitrate") or "")
        self.channels_var.set(setting.get("channels") or "")
        self.name_var.set(setting.get("name") or "")
        self.lang_var.set(setting.get("lang") or "")
        if setting.get("default") is True:
            self.default_var.set("true")
        elif setting.get("default") is False:
            self.default_var.set("false")
        else:
            self.default_var.set("auto")
        self._refresh_field_states()

    def _set_entry_state(self, entry, enabled):
        if not entry:
            return
        entry.configure(state="normal" if enabled else "disabled")

    def _refresh_field_states(self):
        if getattr(self, "_state_refreshing", False):
            return
        self._state_refreshing = True
        try:
            track_type = self.type_var.get().strip().lower()
            mode = self.mode_var.get().strip().upper()

            allowed_modes = MODE_OPTIONS
            enable_bitrate = False
            enable_channels = False
            enable_name = False
            enable_lang = False
            enable_default = False

            if track_type == "video":
                allowed_modes = VIDEO_MODE_OPTIONS
                if mode not in allowed_modes:
                    mode = allowed_modes[0]
                    self.mode_var.set(mode)
                enable_lang = True
            elif track_type == "audio":
                allowed_modes = AUDIO_MODE_OPTIONS
                if mode not in allowed_modes:
                    mode = allowed_modes[0]
                    self.mode_var.set(mode)
                if mode in ("COPY", "EDIT"):
                    enable_name = True
                    enable_lang = True
                    enable_default = True
                if mode == "EDIT":
                    enable_bitrate = True
                    enable_channels = True
            elif track_type == "sub":
                allowed_modes = SUB_MODE_OPTIONS
                if mode not in allowed_modes:
                    mode = allowed_modes[0]
                    self.mode_var.set(mode)
                if mode == "COPY":
                    enable_name = True
                    enable_lang = True
                    enable_default = True
            else:
                allowed_modes = MODE_OPTIONS
                if mode not in allowed_modes:
                    mode = allowed_modes[0]
                    self.mode_var.set(mode)
                enable_name = True
                enable_lang = True
                if mode == "EDIT":
                    enable_bitrate = True
                    enable_channels = True

            if hasattr(self, "mode_combo"):
                self.mode_combo.configure(values=allowed_modes)

            self._set_entry_state(getattr(self, "bitrate_entry", None), enable_bitrate)
            self._set_entry_state(getattr(self, "channels_entry", None), enable_channels)
            self._set_entry_state(getattr(self, "name_entry", None), enable_name)
            self._set_entry_state(getattr(self, "lang_entry", None), enable_lang)

            if hasattr(self, "default_combo"):
                if enable_default:
                    self.default_combo.configure(state="readonly")
                else:
                    self.default_var.set("auto")
                    self.default_combo.configure(state="disabled")

            if enable_bitrate and not self.bitrate_var.get().strip():
                self.bitrate_var.set("2")
            if enable_channels and not self.channels_var.get().strip():
                self.channels_var.set("2")
        finally:
            self._state_refreshing = False

    def _selected_index(self):
        selection = self.tree.selection()
        if not selection:
            return None
        item = selection[0]
        values = self.tree.item(item, "values")
        if not values:
            return None
        try:
            return int(values[0]) - 1
        except (ValueError, IndexError):
            return None

    def on_select_setting(self, _event=None):
        idx = self._selected_index()
        if idx is None:
            return
        if 0 <= idx < len(self.settings):
            self._apply_setting_to_form(self.settings[idx])

    def on_id_change(self, *_args):
        inferred = infer_type_from_id(self.id_var.get(), self.match_tracks)
        if inferred:
            self.type_var.set(inferred)

    def on_type_change(self, *_args):
        self._refresh_field_states()

    def on_mode_change(self, *_args):
        self._refresh_field_states()

    def on_encoder_change(self, *_args):
        self._refresh_encoder_path_options(preserve_missing=False)
        self.on_defaults_change()

    def on_ab_multiplier_change(self, *_args):
        if getattr(self, "_ab_syncing", False):
            return
        self._ab_syncing = True
        try:
            value = self._get_var_value("ab_multiplier_var")
            if self._get_var_value("ab_pos_multiplier_var") != value:
                self.ab_pos_multiplier_var.set(value)
            if self._get_var_value("ab_neg_multiplier_var") != value:
                self.ab_neg_multiplier_var.set(value)
            self.on_defaults_change()
        finally:
            self._ab_syncing = False

    def on_ab_pos_multiplier_change(self, *_args):
        if getattr(self, "_ab_syncing", False):
            return
        self._ab_syncing = True
        try:
            multiplier = self._get_var_value("ab_multiplier_var")
            if multiplier:
                pos_val = self._get_var_value("ab_pos_multiplier_var")
                neg_val = self._get_var_value("ab_neg_multiplier_var")
                if pos_val != multiplier or neg_val != multiplier:
                    self.ab_multiplier_var.set("")
            self.on_defaults_change()
        finally:
            self._ab_syncing = False

    def on_default_text_change(self, event=None):
        widget = event.widget if event else None
        if not widget or not widget.edit_modified():
            return
        if getattr(self, "_defaults_refreshing", False):
            widget.edit_modified(False)
            return
        self._defaults_refreshing = True
        try:
            current = widget.get("1.0", tk.END).rstrip("\n")
            if "\n" in current:
                cleaned = " ".join(part.strip() for part in current.splitlines() if part.strip())
                widget.delete("1.0", tk.END)
                widget.insert("1.0", cleaned)
            self.defaults = self._current_defaults()
            self._refresh_results()
        finally:
            widget.edit_modified(False)
            self._defaults_refreshing = False

    def on_defaults_change(self, *_args):
        self.defaults = self._current_defaults()
        self._refresh_results()

    def _refresh_encoder_path_options(self, *, preserve_missing):
        combo = getattr(self, "encoder_path_combo", None)
        if combo is None:
            return
        current = self._get_var_value("encoder_path_var")
        options = ["", *list_portable_encoder_binaries(self._get_var_value("encoder_var"))]
        if preserve_missing and current and current not in options:
            options.append(current)
        combo.configure(values=options)
        if current in options:
            return
        self.encoder_path_var.set("")

    def on_zoning_change(self, _event=None):
        if not hasattr(self, "default_zoning_text"):
            return
        if self.default_zoning_text.edit_modified():
            self.defaults = self._current_defaults()
            self._refresh_results()
            self.default_zoning_text.edit_modified(False)

    def on_add(self):
        self.settings.append(self._current_setting())
        self._refresh_settings()
        self._refresh_results()

    def on_update(self):
        idx = self._selected_index()
        if idx is None:
            return
        if 0 <= idx < len(self.settings):
            self.settings[idx] = self._current_setting()
            self._refresh_settings()
            self._refresh_results()

    def on_remove(self):
        idx = self._selected_index()
        if idx is None:
            return
        if 0 <= idx < len(self.settings):
            self.settings.pop(idx)
            self._refresh_settings()
            self._refresh_results()

    def on_move_up(self):
        idx = self._selected_index()
        if idx is None or idx <= 0:
            return
        self.settings[idx - 1], self.settings[idx] = self.settings[idx], self.settings[idx - 1]
        self._refresh_settings()
        self._refresh_results()

    def on_move_down(self):
        idx = self._selected_index()
        if idx is None or idx >= len(self.settings) - 1:
            return
        self.settings[idx + 1], self.settings[idx] = self.settings[idx], self.settings[idx + 1]
        self._refresh_settings()
        self._refresh_results()

    def on_apply(self):
        self.defaults = self._current_defaults()
        result, _ = build_results(self.files, self.tracks_by_file, self.settings, self.defaults)
        missing = []
        mismatch = []
        encoder_conflicts = []
        for file_path, tracks in result.items():
            for t in tracks:
                t_type = normalize_type(t.get("type") or "")
                if t_type != "video":
                    continue
                if str(t.get("trackStatus") or "").upper() != "EDIT":
                    continue
                video_config = t.get("videoConfig") or {}
                fastpass_params = dict(video_config.get("fastpass_params") or {})
                mainpass_params = dict(video_config.get("mainpass_params") or {})
                fast_crf = str(video_config.get("fastpass_crf") or "").strip()
                main_crf = str(video_config.get("mainpass_crf") or "").strip()
                if not fast_crf and not main_crf:
                    missing.append(f"{os.path.basename(file_path)} (trackId={t.get('trackId')})")
                if fast_crf and main_crf:
                    try:
                        same = abs(float(fast_crf) - float(main_crf)) < 1e-6
                    except ValueError:
                        same = (fast_crf == main_crf)
                    if not same:
                        mismatch.append(f"{os.path.basename(file_path)} (trackId={t.get('trackId')})")
                encoder = normalize_encoder((t.get("trackMux") or {}).get("encoder"))
                conflicts = sorted(
                    {
                        *find_encoder_param_conflicts(encoder, fastpass_params),
                        *find_encoder_param_conflicts(encoder, mainpass_params),
                    }
                )
                if conflicts:
                    encoder_conflicts.append(
                        f"{os.path.basename(file_path)} (trackId={t.get('trackId')}): " + ", ".join(conflicts)
                    )

        if missing:
            messagebox.showwarning(
                "CRF required",
                "Specify --crf in params or last params for video EDIT.\n"
                "Missing --crf for:\n  " + "\n  ".join(missing),
            )
            return
        if encoder_conflicts:
            messagebox.showwarning(
                "Encoder params mismatch",
                "Selected encoder has incompatible video params.\n"
                "Fix or remove these params before applying:\n  " + "\n  ".join(encoder_conflicts),
            )
            return
        if mismatch:
            messagebox.showwarning(
                "CRF mismatch",
                "Different --crf values in params and last params for:\n  " + "\n  ".join(mismatch),
            )
        if self.output_mode == "json":
            payload = {"status": "ok", "result": result}
            sys.stdout.write(json.dumps(payload, ensure_ascii=False))
            sys.stdout.flush()
            self.root.destroy()
            return

        defaults_map = {
            "params": self.defaults.params,
            "last_params": self.defaults.last_params,
            "zoning": self.defaults.zoning,
            "fastpass_filter": self.defaults.fastpass,
            "mainpass_filter": self.defaults.mainpass,
            "scene_detection": self.defaults.scene_detection,
            "chunk_order": self.defaults.chunk_order,
            "encoder_path": self.defaults.encoder_path,
            "no_fastpass": self.defaults.no_fastpass,
            "fastpass_hdr": self.defaults.fastpass_hdr,
            "strict_sdr_8bit": self.defaults.strict_sdr_8bit,
            "no_dolby_vision": self.defaults.no_dolby_vision,
            "no_hdr10plus": self.defaults.no_hdr10plus,
            "fastpass_workers": self.defaults.fastpass_workers,
            "mainpass_workers": self.defaults.mainpass_workers,
            "encoder": self.defaults.encoder,
            "ab_multiplier": self.defaults.ab_multiplier,
            "ab_pos_dev": self.defaults.ab_pos_dev,
            "ab_neg_dev": self.defaults.ab_neg_dev,
            "ab_pos_multiplier": self.defaults.ab_pos_multiplier,
            "ab_neg_multiplier": self.defaults.ab_neg_multiplier,
            "main_vpy": self.defaults.main_vpy,
            "fast_vpy": self.defaults.fast_vpy,
            "proxy_vpy": self.defaults.proxy_vpy,
            "attach_encode_info": self.defaults.attach_encode_info,
            "note": self.defaults.note,
        }
        saved = []
        for file_path, tracks in result.items():
            source_path = Path(file_path)
            plan_path = Path(self.plan_paths.get(file_path) or plan_path_for_source(source_path))
            plan = file_plan_from_gui_result(
                source=source_path,
                defaults=defaults_map,
                track_results=tracks,
                plan_path=plan_path,
            )
            save_plan(plan, plan_path)
            resolved = resolve_paths(plan, plan_path)
            resolved.zone_file.parent.mkdir(parents=True, exist_ok=True)
            resolved.zone_file.write_text(self.defaults.zoning or "", encoding="utf-8", newline="\n")
            saved.append(str(plan_path))
        payload = {"status": "ok", "savedPlans": saved}
        sys.stdout.write(json.dumps(payload, ensure_ascii=False))
        sys.stdout.flush()
        self.root.destroy()

    def on_cancel(self):
        payload = {"status": "cancel"}
        sys.stdout.write(json.dumps(payload))
        sys.stdout.flush()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Track config GUI for .plan files.")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    if args.paths:
        data = load_gui_data_from_paths(args.paths)
    else:
        raw = sys.stdin.read()
        if not raw.strip():
            sys.stderr.write("No input provided\n")
            sys.exit(1)
        data = json.loads(raw)
        data.setdefault("outputMode", "json")
    gui = TrackConfigGui(data)
    gui.run()


if __name__ == "__main__":
    main()
