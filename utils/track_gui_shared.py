import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plan_model import (
    CHUNK_ORDER_OPTIONS,
    DEFAULT_CHUNK_ORDER,
    DEFAULT_SOURCE_LOADER,
    SOURCE_LOADER_OPTIONS,
    build_summary_rows,
    file_plan_from_gui_result,
    gui_defaults_from_file_plan,
    gui_settings_from_file_plan,
    load_file_plan,
    plan_path_for_source,
    probe_source_tracks,
    resolve_paths,
    save_plan,
)
from utils.pipeline_runtime import (
    LEGACY_PORTABLE_DIR,
    is_mars_av1an_fork,
    list_portable_encoder_binaries,
    load_toolchain,
)


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
PARAM_GLUE = "\u00a0"
ENCODER_PATH_INFO_PREFIX = "[scan]"

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
        avg_func="",
        main_vpy="",
        fast_vpy="",
        proxy_vpy="",
        vpy_wrapper=False,
        source_loader=DEFAULT_SOURCE_LOADER,
        crop_resize_enabled=False,
        crop_resize_commands="",
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
        self.avg_func = avg_func or ""
        self.main_vpy = main_vpy or ""
        self.fast_vpy = fast_vpy or ""
        self.proxy_vpy = proxy_vpy or ""
        self.vpy_wrapper = parse_bool_value(vpy_wrapper, default=False)
        source_loader_value = str(source_loader or DEFAULT_SOURCE_LOADER).strip().lower()
        if source_loader_value in ("bestsource", "best-source"):
            source_loader_value = "bs"
        if source_loader_value in ("lsmash", "lwlibavsource"):
            source_loader_value = "lsmas"
        if source_loader_value not in SOURCE_LOADER_OPTIONS:
            source_loader_value = DEFAULT_SOURCE_LOADER
        self.source_loader = source_loader_value
        self.crop_resize_enabled = parse_bool_value(crop_resize_enabled, default=False)
        if self.crop_resize_enabled:
            self.vpy_wrapper = True
        self.crop_resize_commands = crop_resize_commands or ""
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


def format_params_for_display(raw_value):
    text = str(raw_value or "").replace(PARAM_GLUE, " ").strip()
    if not text:
        return ""
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()
    groups = []
    idx = 0
    while idx < len(tokens):
        token = str(tokens[idx])
        if token.startswith("--") and idx + 1 < len(tokens) and not str(tokens[idx + 1]).startswith("--"):
            groups.append(f"{token}{PARAM_GLUE}{tokens[idx + 1]}")
            idx += 2
            continue
        groups.append(token)
        idx += 1
    return " ".join(groups)


def sanitize_params_display_text(raw_value):
    return str(raw_value or "").replace(PARAM_GLUE, " ")


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
    default_avg_func = defaults.avg_func
    default_main_vpy = defaults.main_vpy
    default_fast_vpy = defaults.fast_vpy
    default_proxy_vpy = defaults.proxy_vpy
    default_vpy_wrapper = defaults.vpy_wrapper
    default_source_loader = defaults.source_loader
    default_crop_resize_enabled = defaults.crop_resize_enabled
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
            for entry in entries_for_type:
                entry["default_final"] = False

            eligible = [entry for entry in entries_for_type if entry["mode"] in ("COPY", "EDIT")]
            if not eligible:
                return

            explicit_true = [entry for entry in eligible if entry["default_raw"] is True]
            if explicit_true:
                for entry in eligible:
                    entry["default_final"] = entry["default_raw"] is True
                return

            first_auto = None
            for idx, entry in enumerate(eligible):
                raw = entry["default_raw"]
                if raw is False:
                    entry["default_final"] = False
                    continue
                if first_auto is None:
                    first_auto = idx
            if first_auto is not None:
                eligible[first_auto]["default_final"] = True

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
                params_display = f"encoder={default_encoder}"
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
                track_mux["avgFunc"] = default_avg_func
                track_mux["mainVpy"] = default_main_vpy
                track_mux["fastVpy"] = default_fast_vpy
                track_mux["proxyVpy"] = default_proxy_vpy
                track_mux["vpyWrapper"] = "true" if default_vpy_wrapper else "false"
                track_mux["sourceLoader"] = default_source_loader
                track_mux["cropResizeEnabled"] = "true" if default_crop_resize_enabled else "false"
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

            applied_note = f"setting #{entry['applied_idx'] + 1}" if entry["applied_idx"] is not None else "no setting"
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
        "avg_func": "",
        "main_vpy": "",
        "fast_vpy": "",
        "proxy_vpy": "",
        "vpy_wrapper": False,
        "source_loader": DEFAULT_SOURCE_LOADER,
        "crop_resize_enabled": False,
        "crop_resize_commands": "",
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
            resolved = resolve_paths(plan, path)
            source = resolved.source
            if len(normalized_paths) == 1:
                defaults.update(gui_defaults_from_file_plan(plan))
                if resolved.crop_resize_file.exists():
                    defaults["crop_resize_commands"] = resolved.crop_resize_file.read_text(encoding="utf-8")
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
            resolved = resolve_paths(plan, existing_plan)
            defaults.update(gui_defaults_from_file_plan(plan))
            if resolved.crop_resize_file.exists():
                defaults["crop_resize_commands"] = resolved.crop_resize_file.read_text(encoding="utf-8")
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


__all__ = [
    "APP_ACCENT",
    "APP_ACCENT_ALT",
    "APP_ACCENT_SOFT",
    "APP_BG",
    "APP_BODY_FONT",
    "APP_BORDER",
    "APP_BUTTON_FONT",
    "APP_CARD",
    "APP_INPUT",
    "APP_LABEL_FONT",
    "APP_MONO_FONT",
    "APP_MUTED",
    "APP_SECTION_FONT",
    "APP_SUBTITLE_FONT",
    "APP_SUCCESS",
    "APP_SURFACE",
    "APP_SURFACE_ALT",
    "APP_TEXT",
    "APP_TITLE_FONT",
    "APP_WARNING",
    "AUDIO_MODE_OPTIONS",
    "CHUNK_ORDER_OPTIONS",
    "DEFAULT_CHUNK_ORDER",
    "DEFAULT_SOURCE_LOADER",
    "DEFAULT_OPTIONS",
    "DEFAULT_VIDEO_ENCODER",
    "DefaultSettings",
    "ENCODER_PATH_INFO_PREFIX",
    "LEGACY_PORTABLE_DIR",
    "MODE_OPTIONS",
    "PARAM_GLUE",
    "STRICT_SDR_8BIT_PARAMS",
    "SOURCE_LOADER_OPTIONS",
    "SUB_MODE_OPTIONS",
    "TYPE_OPTIONS",
    "TYPE_ORDER",
    "VIDEO_ENCODER_OPTIONS",
    "VIDEO_MODE_OPTIONS",
    "align_pipe_table",
    "apply_strict_sdr_8bit_params",
    "build_default_defaults_dict",
    "build_result_line",
    "build_results",
    "file_plan_from_gui_result",
    "find_encoder_param_conflicts",
    "format_params_for_display",
    "get_param_value",
    "gui_defaults_from_file_plan",
    "gui_settings_from_file_plan",
    "infer_type_from_id",
    "is_mars_av1an_fork",
    "list_portable_encoder_binaries",
    "load_file_plan",
    "load_gui_data_from_paths",
    "load_toolchain",
    "normalize_encoder",
    "normalize_type",
    "parse_bool_value",
    "parse_id_rule",
    "parse_int_value",
    "parse_params_string",
    "plan_path_for_source",
    "probe_source_tracks",
    "resolve_paths",
    "rule_matches_track",
    "sanitize_params_display_text",
    "save_plan",
    "setting_matches_track",
]
