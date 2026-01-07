import json
import os
import sys
import tkinter as tk
from tkinter import ttk


TYPE_OPTIONS = ["auto", "video", "audio", "sub"]
MODE_OPTIONS = ["COPY", "EDIT", "SKIP"]
DEFAULT_OPTIONS = ["auto", "true", "false"]
TYPE_ORDER = {"video": 0, "audio": 1, "sub": 2}


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


def build_result_line(track, mode, final_name, final_lang, default_display, params, applied_note, overlap_note):
    name_display = final_name if final_name else "-"
    lang_display = final_lang if final_lang else "-"
    parts = [
        f"{track['track_id']}",
        track.get("type") or "-",
        name_display,
        lang_display,
        mode,
    ]
    if mode == "EDIT" and params:
        parts.append(f"params={params}")
    if default_display is not None:
        parts.append(f"default={default_display}")
    if applied_note:
        parts.append(applied_note)
    if overlap_note:
        parts.append(overlap_note)
    return " | ".join(parts)


def build_results(files, tracks_by_file, settings):
    result = {}
    lines = []

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

        track_results = []
        for track in sorted(tracks, key=sort_key):
            match_indexes = [idx for idx, setting in enumerate(settings) if setting_matches_track(setting, track)]
            applied_idx = match_indexes[-1] if match_indexes else None
            applied = settings[applied_idx] if applied_idx is not None else None

            mode = (applied.get("mode") if applied else None) or "SKIP"
            params = applied.get("params") if applied and mode == "EDIT" else ""

            final_name = track.get("name") or ""
            final_lang = track.get("lang") or ""
            if applied:
                if applied.get("name"):
                    final_name = applied["name"]
                if applied.get("lang"):
                    final_lang = applied["lang"]

            default_display = None
            default_value = None
            if normalize_type(track.get("type")) in ("audio", "sub"):
                if applied and applied.get("default") is not None:
                    default_value = bool(applied["default"])
                    default_display = "true" if default_value else "false"
                else:
                    default_display = "orig"

            track_param = {}
            if mode == "EDIT" and params:
                track_param["params"] = params

            track_mux = {}
            if final_name:
                track_mux["name"] = final_name
            if final_lang:
                track_mux["lang"] = final_lang
            if default_value is not None:
                track_mux["default"] = "true" if default_value else "false"

            track_results.append(
                {
                    "fileIndex": file_index,
                    "trackId": track["track_id"],
                    "trackStatus": mode,
                    "trackParam": track_param,
                    "trackMux": track_mux,
                }
            )

            applied_note = ""
            if applied_idx is not None:
                applied_note = f"setting #{applied_idx + 1}"
            else:
                applied_note = "no setting"

            overlap_note = ""
            if len(match_indexes) > 1:
                overlap_note = "overlap: " + ", ".join(f"#{idx + 1}" for idx in match_indexes)

            line = build_result_line(
                track,
                mode,
                final_name,
                final_lang,
                default_display,
                params,
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
        top_frame = ttk.Frame(frame)
        top_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        columns = ("idx", "id", "type", "mode", "params", "name", "lang", "default")
        self.tree = ttk.Treeview(top_frame, columns=columns, show="headings", height=8)
        self.tree.heading("idx", text="#")
        self.tree.heading("id", text="id")
        self.tree.heading("type", text="type")
        self.tree.heading("mode", text="mode")
        self.tree.heading("params", text="params")
        self.tree.heading("name", text="name")
        self.tree.heading("lang", text="lang")
        self.tree.heading("default", text="default")

        self.tree.column("idx", width=40, anchor=tk.CENTER)
        self.tree.column("id", width=120)
        self.tree.column("type", width=70, anchor=tk.CENTER)
        self.tree.column("mode", width=70, anchor=tk.CENTER)
        self.tree.column("params", width=150)
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
        self.name_var = tk.StringVar()
        self.lang_var = tk.StringVar()
        self.default_var = tk.StringVar(value="auto")

        self.id_var.trace_add("write", self.on_id_change)
        self.type_var.trace_add("write", self.on_type_change)
        self.mode_var.trace_add("write", self.on_mode_change)

        self._add_form_row(form_frame, 0, "id", self.id_var)
        self._add_combo_row(form_frame, 1, "type", self.type_var, TYPE_OPTIONS)
        self._add_combo_row(form_frame, 2, "mode", self.mode_var, MODE_OPTIONS)
        self._add_form_row(form_frame, 3, "params", self.params_var)
        self._add_form_row(form_frame, 4, "name", self.name_var)
        self._add_form_row(form_frame, 5, "lang", self.lang_var)
        self._add_combo_row(form_frame, 6, "default", self.default_var, DEFAULT_OPTIONS)

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

        self.on_mode_change()

    def _add_form_row(self, frame, row, label, variable):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 6), pady=2)
        entry = ttk.Entry(frame, textvariable=variable, width=40)
        entry.grid(row=row, column=1, sticky=tk.W, pady=2)
        if label == "params":
            self.params_entry = entry

    def _add_combo_row(self, frame, row, label, variable, values):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 6), pady=2)
        combo = ttk.Combobox(frame, textvariable=variable, values=values, width=37, state="readonly")
        combo.grid(row=row, column=1, sticky=tk.W, pady=2)
        if label == "default":
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
                setting.get("name") or "",
                setting.get("lang") or "",
                default_display,
            )
            self.tree.insert("", tk.END, values=values)

    def _refresh_results(self):
        result, lines = build_results(self.files, self.tracks_by_file, self.settings)
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
            "name": self.name_var.get().strip(),
            "lang": self.lang_var.get().strip(),
            "default": default_val,
        }

    def _apply_setting_to_form(self, setting):
        self.id_var.set(setting.get("id") or "")
        self.type_var.set(setting.get("type") or "auto")
        self.mode_var.set(setting.get("mode") or "COPY")
        self.params_var.set(setting.get("params") or "")
        self.name_var.set(setting.get("name") or "")
        self.lang_var.set(setting.get("lang") or "")
        if setting.get("default") is True:
            self.default_var.set("true")
        elif setting.get("default") is False:
            self.default_var.set("false")
        else:
            self.default_var.set("auto")
        self.on_mode_change()

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
        if self.type_var.get().strip().lower() not in ("", "auto"):
            return
        inferred = infer_type_from_id(self.id_var.get(), self.match_tracks)
        if inferred:
            self.type_var.set(inferred)

    def on_type_change(self, *_args):
        current = self.type_var.get().strip().lower()
        if current not in ("audio", "sub"):
            self.default_var.set("auto")
            if hasattr(self, "default_combo"):
                self.default_combo.configure(state="disabled")
        else:
            if hasattr(self, "default_combo"):
                self.default_combo.configure(state="readonly")

    def on_mode_change(self, *_args):
        if hasattr(self, "params_entry"):
            state = "normal" if self.mode_var.get().strip().upper() == "EDIT" else "disabled"
            self.params_entry.configure(state=state)

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
        result, _ = build_results(self.files, self.tracks_by_file, self.settings)
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
