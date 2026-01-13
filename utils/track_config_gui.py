import json
import os
import shlex
import sys
import tkinter as tk
from tkinter import ttk, messagebox


TYPE_OPTIONS = ["auto", "video", "audio", "sub"]
MODE_OPTIONS = ["COPY", "EDIT", "SKIP"]
VIDEO_MODE_OPTIONS = ["COPY", "EDIT"]
AUDIO_MODE_OPTIONS = ["COPY", "EDIT", "SKIP"]
SUB_MODE_OPTIONS = ["COPY", "SKIP"]
DEFAULT_OPTIONS = ["auto", "true", "false"]
TYPE_ORDER = {"video": 0, "audio": 1, "sub": 2}


class DefaultSettings:
    def __init__(
        self,
        params="",
        last_params="",
        zoning="",
        fastpass="",
        mainpass="",
        workers="",
        ab_multiplier="",
        ab_pos_dev="",
        ab_neg_dev="",
    ):
        self.params = params or ""
        self.last_params = last_params or ""
        self.zoning = zoning or ""
        self.fastpass = fastpass or ""
        self.mainpass = mainpass or ""
        self.workers = workers or ""
        self.ab_multiplier = ab_multiplier or ""
        self.ab_pos_dev = ab_pos_dev or ""
        self.ab_neg_dev = ab_neg_dev or ""


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


def get_param_value(params_map, key):
    if not params_map:
        return ""
    value = params_map.get(key)
    if value is None:
        return ""
    return str(value).strip()


def parse_int_value(raw_value, default_value):
    if raw_value is None:
        return default_value
    try:
        return int(str(raw_value).strip())
    except ValueError:
        return default_value


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


def build_results(files, tracks_by_file, settings, defaults):
    result = {}
    lines = []

    default_params_map = parse_params_string(defaults.params)
    default_last_params_map = parse_params_string(defaults.last_params)
    default_zoning = defaults.zoning
    default_fastpass = defaults.fastpass
    default_mainpass = defaults.mainpass
    default_workers = defaults.workers
    default_ab_multiplier = defaults.ab_multiplier
    default_ab_pos_dev = defaults.ab_pos_dev
    default_ab_neg_dev = defaults.ab_neg_dev

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
            params_display = ""
            if track_type == "video" and mode == "EDIT":
                param_map = dict(default_params_map)
                last_param_map = dict(default_last_params_map)
                track_param_map = parse_params_string(entry["params"])
                track_last_param_map = parse_params_string(entry["last_params"])
                if track_param_map:
                    param_map.update(track_param_map)
                if track_last_param_map:
                    last_param_map.update(track_last_param_map)
                if param_map:
                    track_param.update(param_map)
                if last_param_map:
                    track_param.update({key.replace("-", "^"): value for key, value in last_param_map.items()})
                param_parts = []
                if entry["params"]:
                    param_parts.append(f"params={entry['params']}")
                if entry["last_params"]:
                    param_parts.append(f"last={entry['last_params']}")
                params_display = ", ".join(param_parts)
            elif track_type == "audio" and mode == "EDIT":
                bitrate = parse_int_value(entry["bitrate"], 2)
                channels = parse_int_value(entry["channels"], 2)
                track_param["bitrate"] = str(bitrate)
                track_param["channels"] = str(channels)
                params_display = f"bitrate={bitrate}, channels={channels}"

            track_mux = {}
            if track_type == "video":
                track_mux["zoning"] = default_zoning
                track_mux["fastpass"] = default_fastpass
                track_mux["mainpass"] = default_mainpass
                track_mux["workers"] = default_workers
                track_mux["abMultiplier"] = default_ab_multiplier
                track_mux["abPosDev"] = default_ab_pos_dev
                track_mux["abNegDev"] = default_ab_neg_dev
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


class TrackConfigGui:
    def __init__(self, data):
        self.files = data.get("files") or []
        self.summary = data.get("summary") or []
        self.settings = []
        defaults_raw = data.get("defaults") or {}
        self.defaults = DefaultSettings(
            params=defaults_raw.get("params") or "",
            last_params=defaults_raw.get("lastParams") or defaults_raw.get("last_params") or "",
            zoning=defaults_raw.get("zoning") or "",
            fastpass=defaults_raw.get("fastpass") or "",
            mainpass=defaults_raw.get("mainpass") or "",
            workers=defaults_raw.get("workers") or "",
            ab_multiplier=defaults_raw.get("abMultiplier") or defaults_raw.get("ab_multiplier") or "",
            ab_pos_dev=defaults_raw.get("abPosDev") or defaults_raw.get("ab_pos_dev") or "",
            ab_neg_dev=defaults_raw.get("abNegDev") or defaults_raw.get("ab_neg_dev") or "",
        )

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

        self._build_ui()
        self._refresh_summary()
        self._refresh_settings()
        self._refresh_results()

    def _build_ui(self):
        self.root.geometry("1200x700")
        self.root.minsize(1100, 600)

        pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(pane)
        center_frame = ttk.Frame(pane)
        right_frame = ttk.Frame(pane)

        pane.add(left_frame, weight=1)
        pane.add(center_frame, weight=2)
        pane.add(right_frame, weight=1)

        summary_label = ttk.Label(left_frame, text="Tracks summary")
        summary_label.pack(anchor=tk.W, padx=6, pady=(6, 0))
        self.summary_text = tk.Text(left_frame, height=10, width=40, wrap="none")
        summary_scroll = ttk.Scrollbar(left_frame, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scroll.set, state="disabled")
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=6)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 6), pady=6)

        result_label = ttk.Label(right_frame, text="Result preview")
        result_label.pack(anchor=tk.W, padx=6, pady=(6, 0))
        self.result_text = tk.Text(right_frame, height=10, width=40, wrap="none")
        result_scroll = ttk.Scrollbar(right_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scroll.set, state="disabled")
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0), pady=6)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 6), pady=6)

        self._build_center(center_frame)

    def _build_center(self, frame):
        defaults_frame = ttk.LabelFrame(frame, text="Defaults")
        defaults_frame.pack(fill=tk.X, padx=6, pady=(6, 0))

        ttk.Label(defaults_frame, text="params").grid(row=0, column=0, sticky=tk.W, padx=(0, 6), pady=2)
        self.default_params_text = tk.Text(defaults_frame, height=2, width=60, wrap="char")
        self.default_params_text.grid(row=0, column=1, sticky=tk.EW, pady=2)
        self.default_params_text.insert("1.0", self.defaults.params)
        self.default_params_text.edit_modified(False)
        self.default_params_text.bind("<Return>", lambda _event: "break")
        self.default_params_text.bind("<KP_Enter>", lambda _event: "break")
        self.default_params_text.bind("<<Modified>>", self.on_default_text_change)
        ttk.Label(defaults_frame, text="last params").grid(row=1, column=0, sticky=tk.W, padx=(0, 6), pady=2)
        self.default_last_params_text = tk.Text(defaults_frame, height=2, width=60, wrap="char")
        self.default_last_params_text.grid(row=1, column=1, sticky=tk.EW, pady=2)
        self.default_last_params_text.insert("1.0", self.defaults.last_params)
        self.default_last_params_text.edit_modified(False)
        self.default_last_params_text.bind("<Return>", lambda _event: "break")
        self.default_last_params_text.bind("<KP_Enter>", lambda _event: "break")
        self.default_last_params_text.bind("<<Modified>>", self.on_default_text_change)
        ttk.Label(defaults_frame, text="zoning").grid(row=2, column=0, sticky=tk.NW, padx=(0, 6), pady=2)
        self.default_zoning_text = tk.Text(defaults_frame, height=3, width=60, wrap="none")
        self.default_zoning_text.grid(row=2, column=1, sticky=tk.EW, pady=2)
        self.default_zoning_text.insert("1.0", self.defaults.zoning)
        self.default_zoning_text.edit_modified(False)

        ttk.Label(defaults_frame, text="Filters").grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(6, 2))

        self.default_fastpass_var = tk.StringVar(value=self.defaults.fastpass)
        self.default_mainpass_var = tk.StringVar(value=self.defaults.mainpass)

        ttk.Label(defaults_frame, text="fastpass").grid(row=4, column=0, sticky=tk.W, padx=(0, 6), pady=2)
        ttk.Entry(defaults_frame, textvariable=self.default_fastpass_var, width=60).grid(
            row=4, column=1, sticky=tk.EW, pady=2
        )
        ttk.Label(defaults_frame, text="mainpass").grid(row=5, column=0, sticky=tk.W, padx=(0, 6), pady=2)
        ttk.Entry(defaults_frame, textvariable=self.default_mainpass_var, width=60).grid(
            row=5, column=1, sticky=tk.EW, pady=2
        )

        defaults_frame.columnconfigure(1, weight=1)

        top_frame = ttk.Frame(frame)
        top_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        columns = (
            "idx",
            "id",
            "type",
            "mode",
            "params",
            "last_params",
            "bitrate",
            "channels",
            "name",
            "lang",
            "default",
        )
        self.tree = ttk.Treeview(top_frame, columns=columns, show="headings", height=8)
        self.tree.heading("idx", text="#")
        self.tree.heading("id", text="id")
        self.tree.heading("type", text="type")
        self.tree.heading("mode", text="mode")
        self.tree.heading("params", text="params")
        self.tree.heading("last_params", text="last params")
        self.tree.heading("bitrate", text="bitrate")
        self.tree.heading("channels", text="channels")
        self.tree.heading("name", text="name")
        self.tree.heading("lang", text="lang")
        self.tree.heading("default", text="default")

        self.tree.column("idx", width=40, anchor=tk.CENTER)
        self.tree.column("id", width=120)
        self.tree.column("type", width=70, anchor=tk.CENTER)
        self.tree.column("mode", width=70, anchor=tk.CENTER)
        self.tree.column("params", width=140)
        self.tree.column("last_params", width=140)
        self.tree.column("bitrate", width=70, anchor=tk.CENTER)
        self.tree.column("channels", width=70, anchor=tk.CENTER)
        self.tree.column("name", width=120)
        self.tree.column("lang", width=90)
        self.tree.column("default", width=70, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(top_frame, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self.on_select_setting)

        form_frame = ttk.Frame(frame)
        form_frame.pack(fill=tk.X, padx=6, pady=(0, 6))

        self.id_var = tk.StringVar()
        self.type_var = tk.StringVar(value="auto")
        self.mode_var = tk.StringVar(value="COPY")
        self.params_var = tk.StringVar()
        self.last_params_var = tk.StringVar()
        self.bitrate_var = tk.StringVar()
        self.channels_var = tk.StringVar()
        self.name_var = tk.StringVar()
        self.lang_var = tk.StringVar()
        self.default_var = tk.StringVar(value="auto")

        self.id_var.trace_add("write", self.on_id_change)
        self.type_var.trace_add("write", self.on_type_change)
        self.mode_var.trace_add("write", self.on_mode_change)
        self.default_fastpass_var.trace_add("write", self.on_defaults_change)
        self.default_mainpass_var.trace_add("write", self.on_defaults_change)
        self.default_zoning_text.bind("<<Modified>>", self.on_zoning_change)

        self._add_form_row(form_frame, 0, "id", self.id_var)
        self._add_combo_row(form_frame, 1, "type", self.type_var, TYPE_OPTIONS)
        self._add_combo_row(form_frame, 2, "mode", self.mode_var, MODE_OPTIONS)
        self._add_form_row(form_frame, 3, "params", self.params_var)
        self._add_form_row(form_frame, 4, "last params", self.last_params_var)
        self._add_form_row(form_frame, 5, "bitrate", self.bitrate_var)
        self._add_form_row(form_frame, 6, "channels", self.channels_var)
        self._add_form_row(form_frame, 7, "name", self.name_var)
        self._add_form_row(form_frame, 8, "lang", self.lang_var)
        self._add_combo_row(form_frame, 9, "default", self.default_var, DEFAULT_OPTIONS)

        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X, padx=6, pady=(0, 6))

        ttk.Button(buttons, text="Add", command=self.on_add).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Update", command=self.on_update).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Remove", command=self.on_remove).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Up", command=self.on_move_up).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Down", command=self.on_move_down).pack(side=tk.LEFT)

        actions = ttk.Frame(frame)
        actions.pack(fill=tk.X, padx=6, pady=(0, 8))
        ttk.Button(actions, text="Apply and Close", command=self.on_apply).pack(side=tk.RIGHT)
        ttk.Button(actions, text="Cancel", command=self.on_cancel).pack(side=tk.RIGHT, padx=(0, 8))

        self._refresh_field_states()

    def _add_form_row(self, frame, row, label, variable):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 6), pady=2)
        entry = ttk.Entry(frame, textvariable=variable, width=40)
        entry.grid(row=row, column=1, sticky=tk.W, pady=2)
        if label == "params":
            self.params_entry = entry
        elif label == "last params":
            self.last_params_entry = entry
        elif label == "bitrate":
            self.bitrate_entry = entry
        elif label == "channels":
            self.channels_entry = entry
        elif label == "name":
            self.name_entry = entry
        elif label == "lang":
            self.lang_entry = entry

    def _add_combo_row(self, frame, row, label, variable, values):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 6), pady=2)
        combo = ttk.Combobox(frame, textvariable=variable, values=values, width=37, state="readonly")
        combo.grid(row=row, column=1, sticky=tk.W, pady=2)
        if label == "mode":
            self.mode_combo = combo
        elif label == "default":
            self.default_combo = combo

    def _refresh_summary(self):
        lines = [row.get("line") or "" for row in self.summary]
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
                setting.get("params") or "",
                setting.get("last_params") or "",
                setting.get("bitrate") or "",
                setting.get("channels") or "",
                setting.get("name") or "",
                setting.get("lang") or "",
                default_display,
            )
            self.tree.insert("", tk.END, values=values)

    def _refresh_results(self):
        self.defaults = self._current_defaults()
        result, lines = build_results(self.files, self.tracks_by_file, self.settings, self.defaults)
        self.latest_result = result
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "\n".join(lines))
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
            "params": self.params_var.get().strip(),
            "last_params": self.last_params_var.get().strip(),
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

    def _current_defaults(self):
        zoning_value = ""
        if hasattr(self, "default_zoning_text"):
            zoning_value = self.default_zoning_text.get("1.0", tk.END).rstrip("\n")
        return DefaultSettings(
            params=self._get_single_line_text(getattr(self, "default_params_text", None)),
            last_params=self._get_single_line_text(getattr(self, "default_last_params_text", None)),
            zoning=zoning_value,
            fastpass=self.default_fastpass_var.get().strip(),
            mainpass=self.default_mainpass_var.get().strip(),
            workers=self.defaults.workers,
            ab_multiplier=self.defaults.ab_multiplier,
            ab_pos_dev=self.defaults.ab_pos_dev,
            ab_neg_dev=self.defaults.ab_neg_dev,
        )

    def _apply_setting_to_form(self, setting):
        self.id_var.set(setting.get("id") or "")
        self.type_var.set(setting.get("type") or "auto")
        self.mode_var.set(setting.get("mode") or "COPY")
        self.params_var.set(setting.get("params") or "")
        self.last_params_var.set(setting.get("last_params") or "")
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
            enable_params = False
            enable_last_params = False
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
                if mode == "EDIT":
                    enable_params = True
                    enable_last_params = True
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
                    enable_params = True
                    enable_last_params = True
                    enable_bitrate = True
                    enable_channels = True

            if hasattr(self, "mode_combo"):
                self.mode_combo.configure(values=allowed_modes)

            self._set_entry_state(getattr(self, "params_entry", None), enable_params)
            self._set_entry_state(getattr(self, "last_params_entry", None), enable_last_params)
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

    def on_default_text_change(self, event=None):
        widget = event.widget if event else None
        if not widget or not widget.edit_modified():
            return
        if getattr(self, "_defaults_refreshing", False):
            widget.edit_modified(False)
            return
        self._defaults_refreshing = True
        try:
            cleaned = self._get_single_line_text(widget)
            current = widget.get("1.0", tk.END).rstrip("\n")
            if cleaned != current:
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
        for file_path, tracks in result.items():
            for t in tracks:
                t_type = normalize_type(t.get("type") or "")
                if t_type != "video":
                    continue
                if str(t.get("trackStatus") or "").upper() != "EDIT":
                    continue
                track_param = t.get("trackParam") or {}
                fast_crf = get_param_value(track_param, "--crf")
                main_crf = get_param_value(track_param, "^^crf")
                if not fast_crf and not main_crf:
                    missing.append(f"{os.path.basename(file_path)} (trackId={t.get('trackId')})")
                if fast_crf and main_crf:
                    try:
                        same = abs(float(fast_crf) - float(main_crf)) < 1e-6
                    except ValueError:
                        same = (fast_crf == main_crf)
                    if not same:
                        mismatch.append(f"{os.path.basename(file_path)} (trackId={t.get('trackId')})")

        if missing:
            messagebox.showwarning(
                "CRF required",
                "Specify --crf in params or last params for video EDIT.\n"
                "Missing --crf for:\n  " + "\n  ".join(missing),
            )
            return
        if mismatch:
            messagebox.showwarning(
                "CRF mismatch",
                "Different --crf values in params and last params for:\n  " + "\n  ".join(mismatch),
            )
        payload = {"status": "ok", "result": result}
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
    raw = sys.stdin.read()
    if not raw.strip():
        sys.stderr.write("No input provided\n")
        sys.exit(1)
    data = json.loads(raw)
    gui = TrackConfigGui(data)
    gui.run()


if __name__ == "__main__":
    main()
