import argparse
import json
import os
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

from utils.track_gui_shared import (
    APP_ACCENT,
    APP_ACCENT_ALT,
    APP_ACCENT_SOFT,
    APP_BG,
    APP_BODY_FONT,
    APP_BORDER,
    APP_BUTTON_FONT,
    APP_CARD,
    APP_INPUT,
    APP_LABEL_FONT,
    APP_MONO_FONT,
    APP_MUTED,
    APP_SECTION_FONT,
    APP_SUBTITLE_FONT,
    APP_SUCCESS,
    APP_SURFACE,
    APP_SURFACE_ALT,
    APP_TEXT,
    APP_TITLE_FONT,
    APP_WARNING,
    AUDIO_MODE_OPTIONS,
    CHUNK_ORDER_OPTIONS,
    DEFAULT_CHUNK_ORDER,
    DEFAULT_SOURCE_LOADER,
    DEFAULT_OPTIONS,
    DEFAULT_VIDEO_ENCODER,
    DefaultSettings,
    ENCODER_PATH_INFO_PREFIX,
    LEGACY_PORTABLE_DIR,
    MODE_OPTIONS,
    SOURCE_LOADER_OPTIONS,
    SUB_MODE_OPTIONS,
    TYPE_OPTIONS,
    VIDEO_ENCODER_OPTIONS,
    align_pipe_table,
    build_results,
    file_plan_from_gui_result,
    find_encoder_param_conflicts,
    format_params_for_display,
    infer_type_from_id,
    is_mars_av1an_fork,
    list_portable_encoder_binaries,
    load_gui_data_from_paths,
    load_toolchain,
    normalize_encoder,
    normalize_type,
    parse_id_rule,
    plan_path_for_source,
    resolve_paths,
    rule_matches_track,
    sanitize_params_display_text,
    save_plan,
)

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
            vpy_wrapper=defaults_raw.get("vpyWrapper") if "vpyWrapper" in defaults_raw else defaults_raw.get("vpy_wrapper", False),
            source_loader=defaults_raw.get("sourceLoader") or defaults_raw.get("source_loader") or DEFAULT_SOURCE_LOADER,
            crop_resize_enabled=defaults_raw.get("cropResizeEnabled") if "cropResizeEnabled" in defaults_raw else defaults_raw.get("crop_resize_enabled", False),
            crop_resize_commands=defaults_raw.get("cropResizeCommands") or defaults_raw.get("crop_resize_commands") or "",
            attach_encode_info=defaults_raw["attachEncodeInfo"] if "attachEncodeInfo" in defaults_raw else defaults_raw.get("attach_encode_info", False),
            note=defaults_raw.get("note") or "",
        )
        self.plan_paths = data.get("planPaths") or {}
        self.output_mode = data.get("outputMode") or "plans"
        self.settings = list(data.get("settings") or [])
        self._encoder_path_choices = []
        self._encoder_path_info_value = f"{ENCODER_PATH_INFO_PREFIX} {LEGACY_PORTABLE_DIR}"
        self._last_encoder_path_choice = ""
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

        self._ensure_default_video_rule()

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

    def _build_stacked_entry(self, parent, row, label, variable, *, width=None):
        ttk.Label(parent, text=label, style="SectionLabel.TLabel").grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 4))
        entry_options = {"textvariable": variable}
        if width is not None:
            entry_options["width"] = width
        entry = ttk.Entry(parent, **entry_options)
        entry.grid(row=row + 1, column=0, columnspan=2, sticky=tk.EW, pady=(0, 10))
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

    def _find_first_track(self, track_type):
        normalized = normalize_type(track_type)
        for file_index in sorted(self.tracks_by_file):
            for track in self.tracks_by_file.get(file_index) or []:
                if normalize_type(track.get("type")) == normalized:
                    return track
        return None

    def _ensure_default_video_rule(self):
        if any(normalize_type(setting.get("type") or "") == "video" for setting in self.settings):
            return
        first_video = self._find_first_track("video")
        if not first_video:
            return
        self.settings.insert(
            0,
            {
                "id": str(first_video.get("track_id") if first_video.get("track_id") is not None else ""),
                "type": "video",
                "mode": "EDIT",
                "params": "",
                "last_params": "",
                "bitrate": "",
                "channels": "",
                "name": "",
                "lang": "",
                "default": True,
            },
        )

    def _matching_rule_tracks(self):
        rule = parse_id_rule(self.id_var.get())
        track_type = normalize_type(self.type_var.get())
        matches = []
        for track in sorted(self.match_tracks, key=lambda item: (int(item.get("track_id") or 0), normalize_type(item.get("type") or ""))):
            if track_type and normalize_type(track.get("type") or "") != track_type:
                continue
            if rule_matches_track(rule, track):
                matches.append(track)
        return matches

    def _apply_track_metadata_suggestion(self, *, overwrite=False):
        track_type = normalize_type(self.type_var.get())
        if track_type not in ("audio", "sub"):
            return
        matches = self._matching_rule_tracks()
        if not matches:
            return
        track = matches[0]
        suggested_name = str(track.get("name") or "")
        suggested_lang = str(track.get("lang") or "")
        if overwrite or not self.name_var.get().strip():
            self.name_var.set(suggested_name)
        if overwrite or not self.lang_var.get().strip():
            self.lang_var.set(suggested_lang)

    def _resolve_vpy_directory_fill(self, raw_value):
        value = str(raw_value or "").strip().strip('"')
        if not value:
            return False
        path = Path(value).expanduser()
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if not resolved.exists() or not resolved.is_dir():
            return False

        discovered = {}
        for item in resolved.iterdir():
            if not item.is_file():
                continue
            lower_name = item.name.lower()
            if lower_name in ("main.vpy", "fast.vpy", "proxy.vpy"):
                discovered[lower_name] = str(item.resolve())
        if not discovered:
            return False

        changed = False
        mapping = (
            ("main.vpy", self.main_vpy_var),
            ("fast.vpy", self.fast_vpy_var),
            ("proxy.vpy", self.proxy_vpy_var),
        )
        for filename, variable in mapping:
            new_value = discovered.get(filename)
            if new_value and variable.get().strip() != new_value:
                variable.set(new_value)
                changed = True
        return changed

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
        try:
            self.root.state("zoomed")
        except Exception:
            try:
                self.root.attributes("-zoomed", True)
            except Exception:
                pass

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
        experimental_tab = ttk.Frame(notebook, style="Surface.TFrame")
        tracks_tab = ttk.Frame(notebook, style="Surface.TFrame")

        notebook.add(video_tab, text="Video Pipeline")
        notebook.add(experimental_tab, text="Experimental")
        notebook.add(tracks_tab, text="Track Rules")

        self._build_video_tab(video_tab)
        self._build_experimental_tab(experimental_tab)
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
        self.default_params_text = tk.Text(params_section, height=4, width=60, wrap="word")
        self.default_last_params_text = tk.Text(params_section, height=2, width=60, wrap="word")
        self.default_zoning_text = tk.Text(params_section, height=5, width=60, wrap="none")
        self._build_labeled_text(
            params_section,
            row,
            "Fast-pass params",
            self.default_params_text,
            height=5,
        )
        self.default_params_text.insert("1.0", format_params_for_display(self.defaults.params))
        self.default_params_text.edit_modified(False)
        self.default_params_text.bind("<Return>", lambda _event: "break")
        self.default_params_text.bind("<KP_Enter>", lambda _event: "break")
        self.default_params_text.bind("<<Modified>>", self.on_default_text_change)
        self.default_params_text.bind("<FocusOut>", self.on_default_text_focus_out, add="+")
        row += 2
        self._build_labeled_text(
            params_section,
            row,
            "Main-pass params",
            self.default_last_params_text,
            height=4,
        )
        self.default_last_params_text.insert("1.0", format_params_for_display(self.defaults.last_params))
        self.default_last_params_text.edit_modified(False)
        self.default_last_params_text.bind("<Return>", lambda _event: "break")
        self.default_last_params_text.bind("<KP_Enter>", lambda _event: "break")
        self.default_last_params_text.bind("<<Modified>>", self.on_default_text_change)
        self.default_last_params_text.bind("<FocusOut>", self.on_default_text_focus_out, add="+")
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
        if self.av1an_fork_enabled:
            self._build_labeled_combo(core_section, row, 1, "Chunk order", self.chunk_order_var, CHUNK_ORDER_OPTIONS, width=24)
            row += 1
            self._build_labeled_combo(core_section, row, 0, "Encoder", self.encoder_var, VIDEO_ENCODER_OPTIONS, width=24)
            self.encoder_path_combo = self._build_labeled_combo(core_section, row, 1, "Encoder path", self.encoder_path_var, [self._encoder_path_info_value], width=24)
            self.encoder_path_combo.bind("<<ComboboxSelected>>", self.on_encoder_path_select, add="+")
            self._refresh_encoder_path_options(preserve_missing=True)
            row += 1
        else:
            self._build_labeled_combo(core_section, row, 1, "Encoder", self.encoder_var, VIDEO_ENCODER_OPTIONS, width=24)
            row += 1
        self._build_labeled_entry(core_section, row, 0, "Fast-pass workers", self.fastpass_workers_var, numeric="int")
        self._build_labeled_entry(core_section, row, 1, "Main-pass workers", self.mainpass_workers_var, numeric="int")

        feature_section, row = self._build_section(
            grid,
            title="Feature Switches",
            columns=2,
        )
        feature_section.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)
        feature_section.columnconfigure(0, weight=1)
        feature_section.columnconfigure(1, weight=1)
        toggles = [
            ("No fast-pass", self.no_fastpass_var),
            ("Fast-pass HDR", self.fastpass_hdr_var),
            ("Strict SDR 8bit", self.strict_sdr_8bit_var),
            ("No Dolby Vision", self.no_dolby_vision_var),
            ("No HDR10+", self.no_hdr10plus_var),
        ]
        for index, (label, variable) in enumerate(toggles):
            current_row = row + (index // 2)
            current_col = index % 2
            ttk.Checkbutton(feature_section, text=label, variable=variable).grid(
                row=current_row, column=current_col, sticky=tk.W, pady=4, padx=(0, 12)
            )

        tuning_section, row = self._build_section(
            grid,
            title="Auto-Boost Tuning",
            columns=1,
        )
        tuning_section.grid(row=2, column=1, sticky="nsew", padx=6, pady=6)
        self._build_labeled_entry(tuning_section, row, 0, "Max + dev", self.ab_pos_dev_var, numeric="int")
        row += 1
        self._build_labeled_entry(tuning_section, row, 0, "Pos multiplier", self.ab_pos_multiplier_var, numeric="float")
        row += 1
        self._build_labeled_entry(tuning_section, row, 0, "Shared multiplier", self.ab_multiplier_var, numeric="float")
        row += 1
        self._build_labeled_entry(tuning_section, row, 0, "Neg multiplier", self.ab_neg_multiplier_var, numeric="float")
        row += 1
        self._build_labeled_entry(tuning_section, row, 0, "Max - dev", self.ab_neg_dev_var, numeric="int")

        script_section, row = self._build_section(
            grid,
            title="Filtering",
            columns=1,
        )
        script_section.grid(row=2, column=0, sticky="nsew", padx=6, pady=6)
        script_section.columnconfigure(0, weight=1)
        self._build_stacked_entry(script_section, row, "Fast-pass filter", self.default_fastpass_var)
        row += 2
        self._build_stacked_entry(script_section, row, "Main-pass filter", self.default_mainpass_var)
        row += 2
        self.main_vpy_entry = self._build_stacked_entry(script_section, row, "Main vpy", self.main_vpy_var)
        self.main_vpy_entry.bind("<FocusOut>", self.on_vpy_focus_out, add="+")
        row += 2
        self.fast_vpy_entry = self._build_stacked_entry(script_section, row, "Fast vpy", self.fast_vpy_var)
        self.fast_vpy_entry.bind("<FocusOut>", self.on_vpy_focus_out, add="+")
        row += 2
        self.proxy_vpy_entry = self._build_stacked_entry(script_section, row, "Proxy vpy", self.proxy_vpy_var)
        self.proxy_vpy_entry.bind("<FocusOut>", self.on_vpy_focus_out, add="+")

        meta_section, row = self._build_section(
            grid,
            title="Output Meta",
            columns=1,
        )
        meta_section.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=6, pady=6)
        meta_section.columnconfigure(0, weight=1)
        self._build_stacked_entry(meta_section, row, "Note In Metadata", self.note_var)
        row += 2
        ttk.Checkbutton(meta_section, text="Attach Encode Info", variable=self.attach_encode_info_var).grid(row=row, column=0, sticky=tk.W, pady=4)

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

    def _build_experimental_tab(self, frame):
        content = self._create_scrollable_area(frame)
        grid = ttk.Frame(content, style="Surface.TFrame", padding=(6, 6, 6, 16))
        grid.pack(fill=tk.BOTH, expand=True)
        grid.columnconfigure(0, weight=1)

        self.vpy_wrapper_var = tk.BooleanVar(value=bool(self.defaults.vpy_wrapper))
        self.source_loader_var = tk.StringVar(value=self.defaults.source_loader or DEFAULT_SOURCE_LOADER)
        self.crop_resize_enabled_var = tk.BooleanVar(value=bool(self.defaults.crop_resize_enabled))

        global_section, row = self._build_section(grid, title="Global", columns=2)
        global_section.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        ttk.Checkbutton(global_section, text="VPY Wrapper", variable=self.vpy_wrapper_var).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=6, padx=(0, 12)
        )
        self._build_labeled_combo(
            global_section,
            row,
            1,
            "Default source loader",
            self.source_loader_var,
            SOURCE_LOADER_OPTIONS,
            width=20,
        )

        crop_section, row = self._build_section(grid, title="Dynamic crop/resize", columns=1)
        crop_section.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        crop_section.columnconfigure(0, weight=1)
        ttk.Checkbutton(crop_section, text="Enable crop/resize", variable=self.crop_resize_enabled_var).grid(
            row=row, column=0, sticky=tk.W, pady=(0, 8)
        )
        row += 1
        self.crop_resize_text = tk.Text(crop_section, height=12, width=80, wrap="none")
        self._build_labeled_text(crop_section, row, "Commands", self.crop_resize_text, height=14)
        self.crop_resize_text.insert("1.0", self.defaults.crop_resize_commands)
        self.crop_resize_text.edit_modified(False)
        self.crop_resize_text.bind("<<Modified>>", self.on_crop_resize_change)

        self.vpy_wrapper_var.trace_add("write", self.on_experimental_change)
        self.source_loader_var.trace_add("write", self.on_defaults_change)
        self.crop_resize_enabled_var.trace_add("write", self.on_crop_resize_enabled_change)
        self._refresh_crop_resize_state()

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
        self.tree.column("id", width=40, anchor=tk.CENTER)
        self.tree.column("type", width=70, anchor=tk.CENTER)
        self.tree.column("mode", width=70, anchor=tk.CENTER)
        self.tree.column("bitrate", width=70, anchor=tk.CENTER)
        self.tree.column("channels", width=70, anchor=tk.CENTER)
        self.tree.column("name", width=120, anchor=tk.CENTER)
        self.tree.column("lang", width=70, anchor=tk.CENTER)
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
        for col in range(3):
            buttons.columnconfigure(col, weight=1)
        ttk.Button(buttons, text="Add Rule", command=self.on_add, style="Accent.TButton").grid(row=0, column=0, sticky=tk.EW, padx=(0, 6), pady=(0, 6))
        ttk.Button(buttons, text="Update Rule", command=self.on_update, style="Secondary.TButton").grid(row=0, column=1, sticky=tk.EW, padx=6, pady=(0, 6))
        ttk.Button(buttons, text="Remove", command=self.on_remove, style="Danger.TButton").grid(row=0, column=2, sticky=tk.EW, padx=(6, 0), pady=(0, 6))
        ttk.Button(buttons, text="Move Up", command=self.on_move_up, style="Ghost.TButton").grid(row=1, column=0, columnspan=2, sticky=tk.EW, padx=(0, 6))
        ttk.Button(buttons, text="Move Down", command=self.on_move_down, style="Ghost.TButton").grid(row=1, column=2, sticky=tk.EW, padx=(6, 0))

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

        mode = self.mode_var.get().strip().upper()
        bitrate = self.bitrate_var.get().strip()
        channels = self.channels_var.get().strip()
        name = self.name_var.get().strip()
        lang = self.lang_var.get().strip()

        type_norm = normalize_type(setting_type or "")
        if type_norm != "audio" or mode != "EDIT":
            bitrate = ""
            channels = ""
        if type_norm == "video":
            name = ""
        if type_norm == "audio" and mode == "SKIP":
            name = ""
            lang = ""
        if type_norm == "sub" and mode != "COPY":
            name = ""
            lang = ""

        return {
            "id": self.id_var.get().strip(),
            "type": setting_type,
            "mode": mode,
            "params": "",
            "last_params": "",
            "bitrate": bitrate,
            "channels": channels,
            "name": name,
            "lang": lang,
            "default": default_val,
        }

    def _get_single_line_text(self, widget):
        if not widget:
            return ""
        text = sanitize_params_display_text(widget.get("1.0", tk.END)).rstrip("\n")
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
        crop_resize_value = ""
        if hasattr(self, "crop_resize_text"):
            crop_resize_value = self.crop_resize_text.get("1.0", tk.END).rstrip("\n")
        chunk_order_var = getattr(self, "chunk_order_var", None)
        encoder_path_var = getattr(self, "encoder_path_var", None)
        chunk_order_value = chunk_order_var.get().strip() if chunk_order_var is not None else self.defaults.chunk_order
        if encoder_path_var is not None:
            encoder_path_value = Path(encoder_path_var.get().strip()).name
        else:
            encoder_path_value = Path(str(self.defaults.encoder_path or "")).name
        if str(encoder_path_value).startswith(ENCODER_PATH_INFO_PREFIX):
            encoder_path_value = ""
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
            vpy_wrapper=bool(getattr(self, "vpy_wrapper_var", tk.BooleanVar(value=False)).get()),
            source_loader=self._get_var_value("source_loader_var") or DEFAULT_SOURCE_LOADER,
            crop_resize_enabled=bool(getattr(self, "crop_resize_enabled_var", tk.BooleanVar(value=False)).get()),
            crop_resize_commands=crop_resize_value,
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
        self._apply_track_metadata_suggestion(overwrite=False)
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
            if not enable_bitrate and self.bitrate_var.get().strip():
                self.bitrate_var.set("")
            if not enable_channels and self.channels_var.get().strip():
                self.channels_var.set("")
            if not enable_name and self.name_var.get().strip():
                self.name_var.set("")
            if not enable_lang and self.lang_var.get().strip():
                self.lang_var.set("")
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
        self._apply_track_metadata_suggestion(overwrite=True)

    def on_type_change(self, *_args):
        self._apply_track_metadata_suggestion(overwrite=True)
        self._refresh_field_states()

    def on_mode_change(self, *_args):
        self._apply_track_metadata_suggestion(overwrite=False)
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

    def on_default_text_focus_out(self, event=None):
        widget = event.widget if event else None
        if widget is None:
            return
        current = sanitize_params_display_text(widget.get("1.0", tk.END)).rstrip("\n")
        trailing_space = " " if current.endswith(" ") else ""
        formatted = format_params_for_display(current) + trailing_space
        shown = widget.get("1.0", tk.END).rstrip("\n")
        if shown != formatted:
            widget.delete("1.0", tk.END)
            widget.insert("1.0", formatted)
            widget.edit_modified(False)

    def on_vpy_focus_out(self, event=None):
        widget = event.widget if event else None
        if widget is None:
            return
        try:
            value = widget.get()
        except Exception:
            return
        if self._resolve_vpy_directory_fill(value):
            self.on_defaults_change()

    def on_defaults_change(self, *_args):
        self.defaults = self._current_defaults()
        self._refresh_results()

    def _refresh_crop_resize_state(self):
        enabled = bool(getattr(self, "crop_resize_enabled_var", tk.BooleanVar(value=False)).get())
        text = getattr(self, "crop_resize_text", None)
        if text is not None:
            text.configure(state="normal" if enabled else "disabled")

    def on_experimental_change(self, *_args):
        wrapper_var = getattr(self, "vpy_wrapper_var", None)
        crop_var = getattr(self, "crop_resize_enabled_var", None)
        if crop_var is not None and crop_var.get() and wrapper_var is not None and not wrapper_var.get():
            wrapper_var.set(True)
            return
        self.defaults = self._current_defaults()
        self._refresh_crop_resize_state()
        self._refresh_results()

    def on_crop_resize_enabled_change(self, *_args):
        if bool(getattr(self, "crop_resize_enabled_var", tk.BooleanVar(value=False)).get()):
            wrapper_var = getattr(self, "vpy_wrapper_var", None)
            if wrapper_var is not None and not wrapper_var.get():
                wrapper_var.set(True)
                return
        self.on_experimental_change()

    def on_crop_resize_change(self, _event=None):
        text = getattr(self, "crop_resize_text", None)
        if text is None or not text.edit_modified():
            return
        self.defaults = self._current_defaults()
        self._refresh_results()
        text.edit_modified(False)

    def _refresh_encoder_path_options(self, *, preserve_missing):
        combo = getattr(self, "encoder_path_combo", None)
        if combo is None:
            return
        current = Path(self._get_var_value("encoder_path_var")).name if self._get_var_value("encoder_path_var") else ""
        discovered = []
        for item in list_portable_encoder_binaries(self._get_var_value("encoder_var")):
            name = Path(item).name
            if name not in discovered:
                discovered.append(name)
        options = [self._encoder_path_info_value, *discovered]
        if preserve_missing and current and current not in discovered:
            options.append(current)
        self._encoder_path_choices = options[1:]
        combo.configure(values=options)
        if current in self._encoder_path_choices:
            self.encoder_path_var.set(current)
            self._last_encoder_path_choice = current
            return
        self._last_encoder_path_choice = ""
        self.encoder_path_var.set("")

    def on_encoder_path_select(self, _event=None):
        value = self._get_var_value("encoder_path_var")
        if value == self._encoder_path_info_value:
            self.encoder_path_var.set(self._last_encoder_path_choice)
            return
        if value in self._encoder_path_choices:
            self._last_encoder_path_choice = value
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
            "vpy_wrapper": self.defaults.vpy_wrapper,
            "source_loader": self.defaults.source_loader,
            "crop_resize_enabled": self.defaults.crop_resize_enabled,
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
            resolved.crop_resize_file.parent.mkdir(parents=True, exist_ok=True)
            resolved.crop_resize_file.write_text(self.defaults.crop_resize_commands or "", encoding="utf-8", newline="\n")
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
