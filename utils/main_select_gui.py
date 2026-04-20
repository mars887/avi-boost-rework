from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.plan_model import FilePlan, gui_defaults_from_file_plan, load_plan, resolve_batch_plan, resolve_paths
from utils.plan_support import collect_file_plan_paths
from utils.track_gui_shared import (
    APP_ACCENT,
    APP_ACCENT_SOFT,
    APP_BG,
    APP_BODY_FONT,
    APP_BORDER,
    APP_BUTTON_FONT,
    APP_CARD,
    APP_INPUT,
    APP_LABEL_FONT,
    APP_MUTED,
    APP_SECTION_FONT,
    APP_SUBTITLE_FONT,
    APP_SURFACE,
    APP_SURFACE_ALT,
    APP_TEXT,
    build_default_defaults_dict,
)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    _DND_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # pragma: no cover - depends on local Tk extension install.
    DND_FILES = None
    TkinterDnD = None
    _DND_IMPORT_ERROR = exc


VIDEO_EXTS = {".mkv", ".mp4"}
VIDEO_EXTRACT_EXTS = {".mkv", ".mp4", ".avi", ".mov"}
RUNNER_BAT_NAMES = {"runner.bat", "batch manager.bat"}

DEFAULT_GROUP_KEYS: Dict[str, Tuple[str, ...]] = {
    "vpy": (
        "fastpass",
        "mainpass",
        "main_vpy",
        "fast_vpy",
        "proxy_vpy",
        "vpy_wrapper",
        "source_loader",
        "crop_resize_enabled",
        "crop_resize_commands",
    ),
    "params": (
        "params",
        "last_params",
        "zoning",
    ),
    "pipeline_core": (
        "encoder",
        "scene_detection",
        "chunk_order",
        "encoder_path",
        "no_fastpass",
        "fastpass_hdr",
        "strict_sdr_8bit",
        "no_dolby_vision",
        "no_hdr10plus",
        "fastpass_workers",
        "mainpass_workers",
        "attach_encode_info",
        "note",
    ),
    "ab_tuning": (
        "ab_multiplier",
        "ab_pos_dev",
        "ab_neg_dev",
        "ab_pos_multiplier",
        "ab_neg_multiplier",
        "avg_func",
    ),
}

DEFAULT_GROUP_LABELS: Tuple[Tuple[str, str], ...] = (
    ("vpy", "VPY"),
    ("params", "Params"),
    ("pipeline_core", "Pipeline core"),
    ("ab_tuning", "AB Tuning"),
)


@dataclass
class SelectionItem:
    gui_path: Path
    source_path: Path
    enabled: bool = True


@dataclass
class MainSelectionResult:
    input_items: List[Tuple[Path, Path]]
    template_plan_path: Optional[Path]
    default_groups: Dict[str, bool]


def is_supported_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS and not path.name.lower().endswith(("-av1.mkv", "-av1.mp4"))


def is_supported_video_file_for_input(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTRACT_EXTS


def normalize_path_key(path: Path) -> str:
    return str(Path(path).expanduser().resolve()).lower()


def expand_input_path(raw_path: str | Path) -> List[Tuple[Path, Path]]:
    path = Path(raw_path).expanduser().resolve()

    if path.is_dir():
        return [
            (source.resolve(), source.resolve())
            for source in sorted(path.iterdir(), key=lambda item: item.name.lower())
            if is_supported_video_file(source)
        ]

    if path.suffix.lower() == ".plan":
        plan = load_plan(path)
        if isinstance(plan, FilePlan):
            return [(path.resolve(), resolve_paths(plan, path).source.resolve())]
        return [(resolved.paths.plan_path.resolve(), resolved.paths.source.resolve()) for resolved in resolve_batch_plan(path)]

    if path.suffix.lower() == ".bat" and path.name.lower() in RUNNER_BAT_NAMES:
        items: List[Tuple[Path, Path]] = []
        for plan_path in collect_file_plan_paths(path.parent):
            plan = load_plan(plan_path)
            if isinstance(plan, FilePlan):
                items.append((plan_path.resolve(), resolve_paths(plan, plan_path).source.resolve()))
        return items

    if not is_supported_video_file_for_input(path):
        raise RuntimeError(f"Unsupported input path: {path}")
    return [(path.resolve(), path.resolve())]


def expand_input_paths(raw_paths: Iterable[str | Path]) -> List[Tuple[Path, Path]]:
    items: List[Tuple[Path, Path]] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        for gui_path, source_path in expand_input_path(raw_path):
            key = normalize_path_key(source_path)
            if key in seen:
                continue
            seen.add(key)
            items.append((gui_path.resolve(), source_path.resolve()))
    return sort_input_items(items)


def sort_input_items(items: Iterable[Tuple[Path, Path]]) -> List[Tuple[Path, Path]]:
    return sorted(
        [(Path(gui_path).resolve(), Path(source_path).resolve()) for gui_path, source_path in items],
        key=lambda item: (str(item[1].parent).lower(), item[1].name.lower()),
    )


def resolve_template_file_plan_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if path.suffix.lower() != ".plan":
        raise RuntimeError(f"Template must be a .plan file: {path}")
    plan = load_plan(path)
    if isinstance(plan, FilePlan):
        return path
    resolved = resolve_batch_plan(path)
    if not resolved:
        raise RuntimeError(f"Batch template has no file plans: {path}")
    return resolved[0].paths.plan_path.resolve()


def load_template_defaults(plan_path: str | Path) -> Dict[str, Any]:
    template_path = resolve_template_file_plan_path(plan_path)
    plan = load_plan(template_path)
    if not isinstance(plan, FilePlan):
        raise RuntimeError(f"Template must resolve to a file plan: {template_path}")

    defaults = build_default_defaults_dict()
    defaults.update(gui_defaults_from_file_plan(plan))
    resolved = resolve_paths(plan, template_path)
    if resolved.zone_file.exists():
        defaults["zoning"] = resolved.zone_file.read_text(encoding="utf-8")
    if resolved.crop_resize_file.exists():
        defaults["crop_resize_commands"] = resolved.crop_resize_file.read_text(encoding="utf-8")
    return defaults


def overlay_default_groups(
    base_defaults: Dict[str, Any],
    template_defaults: Dict[str, Any],
    groups: Dict[str, bool],
) -> Dict[str, Any]:
    out = dict(base_defaults)
    for group_name, keys in DEFAULT_GROUP_KEYS.items():
        if not groups.get(group_name):
            continue
        for key in keys:
            if key in template_defaults:
                out[key] = template_defaults[key]
    return out


def apply_template_default_groups(
    gui_data: Dict[str, Any],
    template_plan_path: Optional[str | Path],
    groups: Dict[str, bool],
) -> Dict[str, Any]:
    if template_plan_path is None:
        return gui_data
    out = dict(gui_data)
    base_defaults = dict(out.get("defaults") or build_default_defaults_dict())
    template_defaults = load_template_defaults(template_plan_path)
    out["defaults"] = overlay_default_groups(base_defaults, template_defaults, groups)
    return out


def _show_missing_dnd_error() -> None:
    message = (
        "tkinterdnd2 is required for the input selection GUI.\n\n"
        "Install it with:\n"
        "  pip install tkinterdnd2"
    )
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Missing tkinterdnd2", message, parent=root)
        root.destroy()
    except Exception:
        pass
    detail = f"\nImport error: {_DND_IMPORT_ERROR}" if _DND_IMPORT_ERROR else ""
    raise RuntimeError(message + detail)


def split_drop_paths(root: tk.Misc, raw_data: str) -> List[Path]:
    try:
        parts = root.tk.splitlist(raw_data)
    except Exception:
        parts = str(raw_data or "").split()
    return [Path(str(part)).expanduser() for part in parts if str(part).strip()]


class MainSelectionGui:
    def __init__(self, initial_paths: Sequence[str | Path]):
        if TkinterDnD is None or DND_FILES is None:
            _show_missing_dnd_error()

        self.items: List[SelectionItem] = []
        self._item_iids: Dict[str, str] = {}
        self._iid_keys: Dict[str, str] = {}
        self._next_iid = 1
        self.result: Optional[MainSelectionResult] = None
        self.template_plan_path: Optional[Path] = None

        self.root = TkinterDnD.Tk()
        self.root.title("Batch Input Selection")
        self.root.geometry("1160x720")
        self.root.minsize(940, 560)
        self.root.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self._configure_theme()

        self.status_var = tk.StringVar(value="")
        self.template_path_var = tk.StringVar(value="No template selected")
        self.group_vars = {name: tk.BooleanVar(value=False) for name, _label in DEFAULT_GROUP_LABELS}
        self.group_checks: List[ttk.Checkbutton] = []

        self._build_ui()
        self._set_template_controls_enabled(False)
        if initial_paths:
            self.add_paths(initial_paths)
        self._refresh_tree()

    def _configure_theme(self) -> None:
        self.root.configure(bg=APP_BG)
        try:
            self.root.tk.call("tk", "scaling", 1.10)
        except Exception:
            pass
        self.root.option_add("*Font", APP_BODY_FONT)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", background=APP_BG, foreground=APP_TEXT, fieldbackground=APP_INPUT)
        style.configure("App.TFrame", background=APP_BG)
        style.configure("Surface.TFrame", background=APP_SURFACE)
        style.configure("Card.TFrame", background=APP_CARD, relief="flat")
        style.configure("Toolbar.TFrame", background=APP_SURFACE_ALT)
        style.configure("TLabel", background=APP_BG, foreground=APP_TEXT, font=APP_BODY_FONT)
        style.configure("Surface.TLabel", background=APP_SURFACE, foreground=APP_TEXT, font=APP_BODY_FONT)
        style.configure("Card.TLabel", background=APP_CARD, foreground=APP_TEXT, font=APP_BODY_FONT)
        style.configure("Title.TLabel", background=APP_BG, foreground=APP_TEXT, font=("Segoe UI Semibold", 22))
        style.configure("Subtitle.TLabel", background=APP_BG, foreground=APP_MUTED, font=APP_SUBTITLE_FONT)
        style.configure("SectionTitle.TLabel", background=APP_CARD, foreground=APP_ACCENT, font=APP_SECTION_FONT)
        style.configure("SectionValue.TLabel", background=APP_CARD, foreground=APP_TEXT, font=APP_BODY_FONT)
        style.configure("SectionHint.TLabel", background=APP_CARD, foreground=APP_MUTED, font=APP_SUBTITLE_FONT)
        style.configure("TCheckbutton", background=APP_CARD, foreground=APP_TEXT, font=APP_BODY_FONT)
        style.map("TCheckbutton", background=[("active", APP_CARD)], foreground=[("active", APP_ACCENT)])
        style.configure("Treeview", background=APP_INPUT, fieldbackground=APP_INPUT, foreground=APP_TEXT, bordercolor=APP_BORDER, rowheight=30)
        style.map("Treeview", background=[("selected", APP_ACCENT_SOFT)], foreground=[("selected", APP_TEXT)])
        style.configure("Treeview.Heading", background=APP_SURFACE_ALT, foreground=APP_ACCENT, relief="flat", font=APP_LABEL_FONT)
        style.map("Treeview.Heading", background=[("active", APP_SURFACE_ALT)])
        style.configure("TScrollbar", background=APP_SURFACE_ALT, troughcolor=APP_BG, bordercolor=APP_BG, arrowcolor=APP_ACCENT)
        style.configure("Accent.TButton", background=APP_ACCENT_SOFT, foreground=APP_ACCENT, bordercolor=APP_ACCENT, font=APP_BUTTON_FONT, padding=(14, 8))
        style.map("Accent.TButton", background=[("active", "#255469"), ("pressed", "#21495c")], foreground=[("disabled", APP_MUTED)])
        style.configure("Secondary.TButton", background=APP_SURFACE_ALT, foreground=APP_TEXT, bordercolor=APP_BORDER, font=APP_BUTTON_FONT, padding=(12, 8))
        style.map("Secondary.TButton", background=[("active", "#1b344c"), ("pressed", "#162c40")])
        style.configure("Ghost.TButton", background=APP_CARD, foreground=APP_MUTED, bordercolor=APP_BORDER, font=APP_BUTTON_FONT, padding=(10, 8))
        style.map("Ghost.TButton", foreground=[("active", APP_ACCENT)], background=[("active", APP_SURFACE_ALT)])
        style.configure("Danger.TButton", background="#3a1832", foreground="#ff5fd2", bordercolor="#ff5fd2", font=APP_BUTTON_FONT, padding=(10, 8))
        style.map("Danger.TButton", background=[("active", "#512147")])
        style.configure("TPanedwindow", background=APP_BG, sashthickness=8)

    def _build_ui(self) -> None:
        shell = ttk.Frame(self.root, style="App.TFrame", padding=(18, 16, 18, 18))
        shell.pack(fill=tk.BOTH, expand=True)
        shell.rowconfigure(1, weight=1)
        shell.columnconfigure(0, weight=1)

        ttk.Label(shell, text="Batch Setup", style="Title.TLabel").grid(row=0, column=0, sticky=tk.W)

        body = ttk.Panedwindow(shell, orient=tk.HORIZONTAL, style="TPanedwindow")
        body.grid(row=1, column=0, sticky="nsew", pady=(14, 0))

        left = ttk.Frame(body, style="Card.TFrame", padding=14)
        right = ttk.Frame(body, style="Card.TFrame", padding=14)
        body.add(left, weight=66)
        body.add(right, weight=34)

        self._build_left_panel(left)
        self._build_right_panel(right)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(2, weight=1)
        parent.columnconfigure(0, weight=1)

        header = ttk.Frame(parent, style="Card.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Inputs", style="SectionTitle.TLabel").grid(row=0, column=0, sticky=tk.W)
        self.count_label = ttk.Label(header, text="", style="SectionHint.TLabel")
        self.count_label.grid(row=0, column=1, sticky=tk.E)

        controls = ttk.Frame(parent, style="Card.TFrame")
        controls.grid(row=1, column=0, sticky="ew", pady=(12, 10))
        for index in range(9):
            controls.columnconfigure(index, weight=0)
        ttk.Button(controls, text="Add files", command=self.on_add_files, style="Secondary.TButton").grid(row=0, column=0, padx=(0, 6), pady=(0, 6), sticky="ew")
        ttk.Button(controls, text="Add folder", command=self.on_add_folder, style="Secondary.TButton").grid(row=0, column=1, padx=6, pady=(0, 6), sticky="ew")
        ttk.Button(controls, text="Enable all", command=lambda: self.set_enabled_for_all(True), style="Ghost.TButton").grid(row=0, column=2, padx=6, pady=(0, 6), sticky="ew")
        ttk.Button(controls, text="Disable all", command=lambda: self.set_enabled_for_all(False), style="Ghost.TButton").grid(row=0, column=3, padx=6, pady=(0, 6), sticky="ew")
        ttk.Button(controls, text="Invert", command=self.invert_enabled, style="Ghost.TButton").grid(row=0, column=4, padx=6, pady=(0, 6), sticky="ew")
        ttk.Button(controls, text="Enable selected", command=lambda: self.set_enabled_for_selected(True), style="Ghost.TButton").grid(row=1, column=0, padx=(0, 6), sticky="ew")
        ttk.Button(controls, text="Disable selected", command=lambda: self.set_enabled_for_selected(False), style="Ghost.TButton").grid(row=1, column=1, padx=6, sticky="ew")
        ttk.Button(controls, text="Remove selected", command=self.remove_selected, style="Danger.TButton").grid(row=1, column=2, padx=6, sticky="ew")

        tree_wrap = ttk.Frame(parent, style="Card.TFrame")
        tree_wrap.grid(row=2, column=0, sticky="nsew")
        tree_wrap.rowconfigure(0, weight=1)
        tree_wrap.columnconfigure(0, weight=1)

        columns = ("enabled", "folder", "name", "source")
        self.tree = ttk.Treeview(tree_wrap, columns=columns, show="headings", selectmode="extended")
        self.tree.heading("enabled", text="Enabled")
        self.tree.heading("folder", text="Folder")
        self.tree.heading("name", text="Name")
        self.tree.heading("source", text="Source")
        self.tree.column("enabled", width=86, minwidth=76, anchor=tk.CENTER, stretch=False)
        self.tree.column("folder", width=260, minwidth=160, anchor=tk.W)
        self.tree.column("name", width=240, minwidth=160, anchor=tk.W)
        self.tree.column("source", width=360, minwidth=220, anchor=tk.W)
        yscroll = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self.tree.yview)
        xscroll = ttk.Scrollbar(tree_wrap, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self.tree.bind("<Button-1>", self.on_tree_click, add="+")
        self.tree.bind("<space>", self.on_tree_space, add="+")
        self.tree.bind("<Delete>", lambda _event: self.remove_selected(), add="+")

        footer = ttk.Frame(parent, style="Card.TFrame")
        footer.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var, style="SectionHint.TLabel").grid(row=0, column=0, sticky=tk.W)
        ttk.Button(footer, text="Cancel", command=self.on_cancel, style="Ghost.TButton").grid(row=0, column=1, padx=(0, 8))
        self.continue_button = ttk.Button(footer, text="Continue", command=self.on_continue, style="Accent.TButton")
        self.continue_button.grid(row=0, column=2)

        self._register_drop_target(parent, self.on_inputs_drop)
        self._register_drop_target(self.tree, self.on_inputs_drop)

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(3, weight=1)
        parent.columnconfigure(0, weight=1)

        ttk.Label(parent, text="Defaults Source", style="SectionTitle.TLabel").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(parent, textvariable=self.template_path_var, style="SectionValue.TLabel", wraplength=360).grid(row=1, column=0, sticky="ew", pady=(10, 8))

        actions = ttk.Frame(parent, style="Card.TFrame")
        actions.grid(row=2, column=0, sticky="ew")
        ttk.Button(actions, text="Choose .plan", command=self.on_choose_template, style="Secondary.TButton").grid(row=0, column=0, padx=(0, 8))
        ttk.Button(actions, text="Clear", command=self.clear_template, style="Ghost.TButton").grid(row=0, column=1)

        groups = ttk.Frame(parent, style="Card.TFrame")
        groups.grid(row=3, column=0, sticky="nsew", pady=(18, 0))
        groups.columnconfigure(0, weight=1)
        ttk.Label(groups, text="Copy", style="SectionTitle.TLabel").grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        for row, (group_name, label) in enumerate(DEFAULT_GROUP_LABELS, start=1):
            check = ttk.Checkbutton(groups, text=label, variable=self.group_vars[group_name])
            check.grid(row=row, column=0, sticky=tk.W, pady=5)
            self.group_checks.append(check)

        self._register_drop_target(parent, self.on_template_drop)
        self._register_drop_target(groups, self.on_template_drop)

    def _register_drop_target(self, widget: tk.Misc, callback) -> None:
        widget.drop_target_register(DND_FILES)
        widget.dnd_bind("<<Drop>>", callback)

    def add_paths(self, raw_paths: Iterable[str | Path]) -> None:
        errors: List[str] = []
        added = 0
        for raw_path in raw_paths:
            try:
                candidates = expand_input_path(raw_path)
            except Exception as exc:
                errors.append(f"{raw_path}: {exc}")
                continue
            for gui_path, source_path in candidates:
                key = normalize_path_key(source_path)
                if any(normalize_path_key(item.source_path) == key for item in self.items):
                    continue
                self.items.append(SelectionItem(gui_path=gui_path.resolve(), source_path=source_path.resolve(), enabled=True))
                added += 1
        self._refresh_tree()
        if added:
            self.status_var.set(f"Added {added} item(s).")
        if errors:
            messagebox.showwarning("Some inputs were skipped", "\n".join(errors[:12]), parent=self.root)

    def _sorted_items(self) -> List[SelectionItem]:
        return sorted(self.items, key=lambda item: (str(item.source_path.parent).lower(), item.source_path.name.lower()))

    def _refresh_tree(self) -> None:
        selected_keys = {self._iid_keys.get(iid) for iid in self.tree.selection()} if hasattr(self, "tree") else set()
        selected_keys.discard(None)
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self._item_iids.clear()
        self._iid_keys.clear()
        for item in self._sorted_items():
            key = normalize_path_key(item.source_path)
            iid = f"item-{self._next_iid}"
            self._next_iid += 1
            self._item_iids[key] = iid
            self._iid_keys[iid] = key
            self.tree.insert(
                "",
                tk.END,
                iid=iid,
                values=(
                    "ON" if item.enabled else "OFF",
                    str(item.source_path.parent),
                    item.source_path.name,
                    str(item.source_path),
                ),
            )
            if key in selected_keys:
                self.tree.selection_add(iid)
        self._refresh_counts()

    def _refresh_counts(self) -> None:
        total = len(self.items)
        enabled = sum(1 for item in self.items if item.enabled)
        self.count_label.configure(text=f"{enabled} enabled / {total} total")
        if enabled > 0:
            self.continue_button.state(["!disabled"])
        else:
            self.continue_button.state(["disabled"])

    def _item_for_iid(self, iid: str) -> Optional[SelectionItem]:
        key = self._iid_keys.get(iid)
        if not key:
            return None
        for item in self.items:
            if normalize_path_key(item.source_path) == key:
                return item
        return None

    def selected_items(self) -> List[SelectionItem]:
        out: List[SelectionItem] = []
        for iid in self.tree.selection():
            item = self._item_for_iid(iid)
            if item is not None:
                out.append(item)
        return out

    def set_enabled_for_all(self, enabled: bool) -> None:
        for item in self.items:
            item.enabled = enabled
        self._refresh_tree()

    def set_enabled_for_selected(self, enabled: bool) -> None:
        for item in self.selected_items():
            item.enabled = enabled
        self._refresh_tree()

    def invert_enabled(self) -> None:
        for item in self.items:
            item.enabled = not item.enabled
        self._refresh_tree()

    def remove_selected(self) -> None:
        selected_keys = {normalize_path_key(item.source_path) for item in self.selected_items()}
        if not selected_keys:
            return
        self.items = [item for item in self.items if normalize_path_key(item.source_path) not in selected_keys]
        self._refresh_tree()

    def on_tree_click(self, event) -> Optional[str]:
        if self.tree.identify("region", event.x, event.y) != "cell":
            return None
        if self.tree.identify_column(event.x) != "#1":
            return None
        iid = self.tree.identify_row(event.y)
        item = self._item_for_iid(iid)
        if item is None:
            return "break"
        item.enabled = not item.enabled
        self._refresh_tree()
        return "break"

    def on_tree_space(self, _event) -> str:
        for item in self.selected_items():
            item.enabled = not item.enabled
        self._refresh_tree()
        return "break"

    def on_inputs_drop(self, event) -> str:
        self.add_paths(split_drop_paths(self.root, event.data))
        return "break"

    def on_template_drop(self, event) -> str:
        paths = split_drop_paths(self.root, event.data)
        if not paths:
            return "break"
        self.set_template(paths[0])
        return "break"

    def on_add_files(self) -> None:
        selected = filedialog.askopenfilenames(
            parent=self.root,
            title="Add files",
            filetypes=[
                ("Supported inputs", "*.mkv *.mp4 *.avi *.mov *.plan *.bat"),
                ("Video files", "*.mkv *.mp4 *.avi *.mov"),
                ("Plan files", "*.plan"),
                ("Batch files", "*.bat"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.add_paths(selected)

    def on_add_folder(self) -> None:
        selected = filedialog.askdirectory(parent=self.root, title="Add folder")
        if selected:
            self.add_paths([selected])

    def on_choose_template(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Choose defaults .plan",
            filetypes=[("Plan files", "*.plan"), ("All files", "*.*")],
        )
        if selected:
            self.set_template(selected)

    def set_template(self, raw_path: str | Path) -> None:
        try:
            template_path = resolve_template_file_plan_path(raw_path)
            load_template_defaults(template_path)
        except Exception as exc:
            messagebox.showerror("Template not loaded", str(exc), parent=self.root)
            return
        self.template_plan_path = template_path
        self.template_path_var.set(str(template_path))
        for variable in self.group_vars.values():
            variable.set(True)
        self._set_template_controls_enabled(True)

    def clear_template(self) -> None:
        self.template_plan_path = None
        self.template_path_var.set("No template selected")
        for variable in self.group_vars.values():
            variable.set(False)
        self._set_template_controls_enabled(False)

    def _set_template_controls_enabled(self, enabled: bool) -> None:
        state = ["!disabled"] if enabled else ["disabled"]
        for check in self.group_checks:
            check.state(state)

    def on_continue(self) -> None:
        enabled_items = [item for item in self._sorted_items() if item.enabled]
        if not enabled_items:
            return
        self.result = MainSelectionResult(
            input_items=[(item.gui_path.resolve(), item.source_path.resolve()) for item in enabled_items],
            template_plan_path=self.template_plan_path.resolve() if self.template_plan_path is not None else None,
            default_groups={name: bool(variable.get()) for name, variable in self.group_vars.items()},
        )
        self.root.destroy()

    def on_cancel(self) -> None:
        self.result = None
        self.root.destroy()

    def run(self) -> Optional[MainSelectionResult]:
        self.root.mainloop()
        return self.result


def run_main_selection_gui(initial_paths: Sequence[str | Path]) -> Optional[MainSelectionResult]:
    gui = MainSelectionGui(initial_paths)
    return gui.run()


__all__ = [
    "DEFAULT_GROUP_KEYS",
    "MainSelectionGui",
    "MainSelectionResult",
    "SelectionItem",
    "apply_template_default_groups",
    "expand_input_path",
    "expand_input_paths",
    "is_supported_video_file",
    "is_supported_video_file_for_input",
    "load_template_defaults",
    "overlay_default_groups",
    "resolve_template_file_plan_path",
    "run_main_selection_gui",
    "sort_input_items",
]
