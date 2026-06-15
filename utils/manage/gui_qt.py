from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from utils.pipeline_runtime import ROOT_DIR, load_toolchain
from utils.plan import FilePlan

from utils.manage import (
    analytics,
    av1an_state,
    config as manage_config,
    reset as manage_reset,
    scenes as manage_scenes,
    zone_patch,
)
from utils.manage.av1an_state import FrameRange, MoveTarget
from utils.manage.backup import ManageTransaction, list_backups
from utils.manage.context import WorkdirContext
from utils.manage.discovery import WorkdirRef, load_ref
from utils.manage.status import (
    STATE_ARTIFACT_MISSING,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_NOT_STARTED,
    STATE_RUNNING,
    STATE_SKIPPED,
    STATE_STALE_MARKER,
    RunnerActivity,
    StageStatus,
    get_stage_statuses,
    load_runner_history,
    runner_activity,
    summarize_workdir,
)
from utils.manage.store import import_workdir_files, open_store

COLOR_GREEN = "#4caf50"
COLOR_BLUE = "#42a5f5"
COLOR_RED = "#ef5350"
COLOR_RED_DARK = "#d32f2f"
COLOR_ORANGE = "#ffa726"
COLOR_ORANGE_DARK = "#ef6c00"
COLOR_GREY = "#9e9e9e"
COLOR_GREY_400 = "#bdbdbd"
COLOR_GREY_500 = "#9e9e9e"
COLOR_GREY_600 = "#757575"
COLOR_GREY_800 = "#424242"
COLOR_LINK = "#64b5f6"

STATE_COLORS = {
    STATE_COMPLETED: COLOR_GREEN,
    STATE_RUNNING: COLOR_BLUE,
    STATE_FAILED: COLOR_RED,
    STATE_SKIPPED: COLOR_GREY,
    STATE_NOT_STARTED: COLOR_GREY_600,
    STATE_STALE_MARKER: COLOR_ORANGE,
    STATE_ARTIFACT_MISSING: COLOR_ORANGE_DARK,
}

TAB_NAMES = ("Status", "Config", "Scenes", "Fastpass", "Mainpass", "Logs", "Raw JSON")
MONO = "Consolas"


def fmt_size(value: Optional[int]) -> str:
    if not value:
        return "-"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return "-"


def fmt_duration(value: Optional[float]) -> str:
    if value is None or value <= 0:
        return "-"
    seconds = int(value)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def fmt_timestamp(value: Optional[float]) -> str:
    if not value:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(value))


def fmt_frame_time(frame: int, fps: float) -> str:
    if fps <= 0:
        return "-"
    total = frame / fps
    hours, rem = divmod(int(total), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{int((total % 1) * 1000):03d}"


def parse_time_or_frame(text: str, fps: float) -> Optional[int]:
    """Accepts a frame number or hh:mm:ss(.ms) / mm:ss and returns a frame."""
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    parts = raw.replace(",", ".").split(":")
    try:
        values = [float(part) for part in parts]
    except ValueError:
        return None
    seconds = 0.0
    for value in values:
        seconds = seconds * 60 + value
    if fps <= 0:
        return None
    return int(round(seconds * fps))


def _fmt_dur0(seconds: float) -> str:
    return fmt_duration(seconds) if seconds > 0 else "0:00"


def format_chunk_stats(rows: Sequence["analytics.PassChunkRow"]) -> str:
    """Progress summary for a set of chunks.

    ``done\\total - done_time\\total_time | encoded size: X(?) | avg bitrate: Y(?)``
    where size/bitrate cover only finished chunks and a trailing ``?`` marks that
    the set still has unfinished chunks (so the figure is partial).
    """
    total = len(rows)
    done_rows = [row for row in rows if row.done]
    done_count = len(done_rows)
    partial = "?" if done_count < total else ""
    done_seconds = sum(row.duration_seconds for row in done_rows)
    total_seconds = sum(row.duration_seconds for row in rows)
    done_size = sum(row.output_size for row in done_rows)
    if done_seconds > 0:
        bitrate = f"{done_size * 8 / 1000.0 / done_seconds:.0f} kbps{partial}"
    else:
        bitrate = f"-{partial}"
    return (
        f"chunks: {done_count}\\{total} - {_fmt_dur0(done_seconds)}\\{_fmt_dur0(total_seconds)} "
        f"| encoded size: {fmt_size(done_size)}{partial} | avg bitrate: {bitrate}"
    )


@dataclass
class PassTabState:
    pass_name: str
    rows: List[analytics.PassChunkRow] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    selected: set = field(default_factory=set)
    sort_field: str = "queue_position"
    sort_desc: bool = False
    policy_quarantine: Optional[bool] = None  # None -> default per pass
    filter_done: str = "all"  # all | done | not done
    filter_chapter: str = ""
    filter_crf_min: str = ""
    filter_crf_max: str = ""
    filter_outliers: bool = False

    def apply_filters(self, rows: List[analytics.PassChunkRow]) -> List[analytics.PassChunkRow]:
        out = rows
        if self.filter_done == "done":
            out = [row for row in out if row.done]
        elif self.filter_done == "not done":
            out = [row for row in out if not row.done]
        if self.filter_chapter:
            out = [row for row in out if row.chapter == self.filter_chapter]
        try:
            crf_min = float(self.filter_crf_min) if str(self.filter_crf_min).strip() else None
        except ValueError:
            crf_min = None
        try:
            crf_max = float(self.filter_crf_max) if str(self.filter_crf_max).strip() else None
        except ValueError:
            crf_max = None
        if crf_min is not None:
            out = [row for row in out if row.crf is not None and row.crf >= crf_min]
        if crf_max is not None:
            out = [row for row in out if row.crf is not None and row.crf <= crf_max]
        if self.filter_outliers:
            rates = sorted(row.bitrate_kbps for row in rows if row.bitrate_kbps)
            if rates:
                median = rates[len(rates) // 2]
                out = [
                    row
                    for row in out
                    if row.bitrate_kbps is not None
                    and (row.bitrate_kbps > 2 * median or row.bitrate_kbps < 0.5 * median)
                ]
        return out


# ---------------------------------------------------------------------- Qt helpers


def mono_font(point_size: int = 9) -> QFont:
    font = QFont(MONO)
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPointSize(point_size)
    return font


def make_label(
    text: str,
    *,
    color: Optional[str] = None,
    size: Optional[int] = None,
    bold: bool = False,
    mono: bool = False,
    selectable: bool = False,
    wrap: bool = False,
    tooltip: Optional[str] = None,
) -> QLabel:
    label = QLabel(text)
    styles: List[str] = []
    if color:
        styles.append(f"color: {color};")
    if size:
        styles.append(f"font-size: {size}px;")
    if bold:
        styles.append("font-weight: bold;")
    if styles:
        label.setStyleSheet("".join(styles))
    if mono:
        label.setFont(mono_font())
    if selectable:
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    if wrap:
        label.setWordWrap(True)
    if tooltip:
        label.setToolTip(tooltip)
    return label


def labeled(caption: str, widget: QWidget) -> QWidget:
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    layout.addWidget(make_label(caption, color=COLOR_GREY_500, size=11))
    layout.addWidget(widget)
    return host


def widget_row(*widgets: QWidget, spacing: int = 8) -> QWidget:
    host = QWidget()
    layout = QHBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for widget in widgets:
        # bottom-align so plain buttons/checkboxes line up with the input part
        # of labeled() fields instead of floating at mid-height
        layout.addWidget(widget, 0, Qt.AlignmentFlag.AlignBottom)
    layout.addStretch(1)
    return host


def link_button(text: str, handler: Callable[[], None]) -> QPushButton:
    button = QPushButton(text)
    button.setFlat(True)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setStyleSheet(f"QPushButton {{ color: {COLOR_LINK}; border: none; text-align: left; padding: 2px; }}")
    button.clicked.connect(lambda: handler())
    return button


def clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
            continue
        child = item.layout()
        if child is not None:
            clear_layout(child)


def make_table(headers: Sequence[str]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(30)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    table.setAlternatingRowColors(True)
    table.setWordWrap(False)
    return table


def fit_table(table: QTableWidget, stretch_column: Optional[int] = None) -> None:
    """Size columns to content and let one column absorb the remaining width."""
    table.resizeColumnsToContents()
    if stretch_column is not None:
        table.horizontalHeader().setSectionResizeMode(stretch_column, QHeaderView.ResizeMode.Stretch)


SORT_KEY_ROLE = Qt.ItemDataRole.UserRole + 1


class SortableItem(QTableWidgetItem):
    """Sorts by an explicit key (numbers stay numeric) with text fallback."""

    def __lt__(self, other: QTableWidgetItem) -> bool:  # type: ignore[override]
        left = self.data(SORT_KEY_ROLE)
        right = other.data(SORT_KEY_ROLE)
        if left is not None and right is not None:
            try:
                return left < right
            except TypeError:
                pass
        return super().__lt__(other)


def make_item(
    text: str,
    *,
    color: Optional[str] = None,
    bold: bool = False,
    mono: bool = False,
    tooltip: Optional[str] = None,
    user_data: Any = None,
    sort_key: Any = None,
) -> QTableWidgetItem:
    item = SortableItem(text)
    if color:
        item.setForeground(QColor(color))
    if bold or mono:
        font = mono_font() if mono else item.font()
        font.setBold(bold)
        item.setFont(font)
    if tooltip:
        item.setToolTip(tooltip)
    if user_data is not None:
        item.setData(Qt.ItemDataRole.UserRole, user_data)
    if sort_key is not None:
        item.setData(SORT_KEY_ROLE, sort_key)
    return item


class ChunkTable(QTableWidget):
    """Pass-tab table: subtle full-row selection plus Ctrl-click / Ctrl-drag to
    paint the selection checkboxes (column 0). A plain click just highlights the
    row; Ctrl-click toggles a chunk's checkbox, and holding Ctrl while dragging
    paints that same state onto every row the cursor passes over."""

    def __init__(self, headers: Sequence[str]) -> None:
        super().__init__(0, len(headers))
        self.setHorizontalHeaderLabels(list(headers))
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(28)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        # full-row highlight via a faint slate-blue Highlight. Fusion draws the
        # selection from the palette (a QSS ::item:selected rule does not win
        # here), so override the palette to a soft bluish tint.
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Highlight, QColor(50, 62, 90))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(224, 224, 224))
        self.setPalette(palette)
        self._owned = False  # we are handling the current press->release sequence
        self._ctrl_drag = False  # paint checkboxes while dragging
        self._paint_state = Qt.CheckState.Checked

    def _check_item(self, row: int) -> Optional[QTableWidgetItem]:
        return self.item(row, 0) if row >= 0 else None

    def _row_at(self, event) -> int:
        return self.indexAt(event.position().toPoint()).row()

    def _toggle_row(self, row: int) -> bool:
        item = self._check_item(row)
        if item is None:
            return False
        self._paint_state = (
            Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked
        )
        item.setCheckState(self._paint_state)
        return True

    def mousePressEvent(self, event) -> None:
        self._owned = False
        self._ctrl_drag = False
        if event.button() == Qt.MouseButton.LeftButton:
            index = self.indexAt(event.position().toPoint())
            ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            # Toggle on PRESS for a checkbox-column click or any Ctrl-click; doing
            # it here (not on release) keeps direct checkbox clicks from being
            # swallowed as a drag-select. Ctrl additionally arms drag-painting.
            if index.row() >= 0 and (ctrl or index.column() == 0) and self._toggle_row(index.row()):
                self._owned = True
                self._ctrl_drag = ctrl
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._owned:
            if self._ctrl_drag:
                item = self._check_item(self._row_at(event))
                if item is not None and item.checkState() != self._paint_state:
                    item.setCheckState(self._paint_state)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._owned:
            self._owned = False
            self._ctrl_drag = False
            event.accept()
            return
        super().mouseReleaseEvent(event)


class CollapsibleSection(QWidget):
    """Qt counterpart of flet's ExpansionTile: a toggle header over a content widget."""

    def __init__(self, title: str, content: QWidget, *, expanded: bool = False) -> None:
        super().__init__()
        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self._content = content
        content.setVisible(expanded)
        self._toggle.toggled.connect(self._on_toggled)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._toggle)
        layout.addWidget(content)

    def _on_toggled(self, checked: bool) -> None:
        self._content.setVisible(checked)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    window = QColor(32, 33, 36)
    base = QColor(24, 25, 28)
    alternate = QColor(40, 41, 45)
    text = QColor(224, 224, 224)
    disabled = QColor(120, 120, 120)
    button = QColor(45, 46, 50)
    highlight = QColor(42, 130, 218)
    palette.setColor(QPalette.ColorRole.Window, window)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, base)
    palette.setColor(QPalette.ColorRole.AlternateBase, alternate)
    palette.setColor(QPalette.ColorRole.ToolTipBase, alternate)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, button)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#ff5252"))
    palette.setColor(QPalette.ColorRole.Highlight, highlight)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("white"))
    palette.setColor(QPalette.ColorRole.Link, QColor(COLOR_LINK))
    palette.setColor(QPalette.ColorRole.PlaceholderText, disabled)
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    app.setPalette(palette)


# ---------------------------------------------------------------------- main window


class ManageWindow(QMainWindow):
    def __init__(self, refs: Sequence[WorkdirRef]) -> None:
        super().__init__()
        self.refs: List[WorkdirRef] = list(refs)
        self.current = 0
        self.ctx: Optional[WorkdirContext] = None
        self.ctx_error: str = ""
        self.activity: Optional[RunnerActivity] = None
        self.statuses: List[StageStatus] = []
        self.pass_states = {name: PassTabState(pass_name=name) for name in ("fastpass", "mainpass")}
        self.include_internal = True

        self.setWindowTitle("PBBatch Manage")
        self._build_ui()
        self.load_context()
        self.rebuild()

    # ------------------------------------------------------------------ utils

    def notify(self, message: str, *, error: bool = False) -> None:
        bar = self.statusBar()
        bar.setStyleSheet(f"color: {COLOR_RED};" if error else "")
        bar.showMessage(message, 8000)

    def guard(self, handler: Callable[[], None]) -> None:
        try:
            handler()
        except manage_reset.ResetBlockedError as exc:
            self.notify(str(exc), error=True)
        except Exception as exc:  # surface, don't crash the GUI loop
            self.notify(f"{type(exc).__name__}: {exc}", error=True)

    def confirm(
        self,
        title: str,
        lines: Sequence[str],
        on_confirm: Callable[[], None],
        *,
        confirm_label: str = "Apply",
        danger: bool = True,
    ) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        body = QPlainTextEdit("\n".join(lines))
        body.setReadOnly(True)
        body.setFont(mono_font())
        body.setMinimumSize(820, min(480, 60 + 18 * max(3, len(lines))))
        layout.addWidget(body)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(dialog.reject)
        apply_btn = QPushButton(confirm_label)
        if danger:
            apply_btn.setStyleSheet(f"background-color: {COLOR_RED_DARK}; color: white;")

        def do_apply() -> None:
            dialog.accept()
            self.guard(on_confirm)

        apply_btn.clicked.connect(do_apply)
        buttons.addWidget(cancel)
        buttons.addWidget(apply_btn)
        layout.addLayout(buttons)
        dialog.exec()

    def show_dialog_form(
        self,
        title: str,
        controls: Sequence[QWidget],
        on_submit: Callable[[], None],
        *,
        submit_label: str = "Apply",
        width: int = 560,
    ) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setMinimumWidth(width)
        layout = QVBoxLayout(dialog)
        for control in controls:
            layout.addWidget(control)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(dialog.reject)
        submit = QPushButton(submit_label)

        def do_submit() -> None:
            dialog.accept()
            self.guard(on_submit)

        submit.clicked.connect(do_submit)
        buttons.addWidget(cancel)
        buttons.addWidget(submit)
        layout.addLayout(buttons)
        dialog.exec()

    def show_text_dialog(self, title: str, text: str, *, width: int = 1000, height: int = 600) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(width, height)
        layout = QVBoxLayout(dialog)
        body = QPlainTextEdit(text)
        body.setReadOnly(True)
        body.setFont(mono_font())
        body.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(body)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(dialog.accept)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        dialog.exec()

    def destructive_allowed(self) -> bool:
        if self.activity is not None and self.activity.active:
            self.notify("runner is active for this workdir; destructive operations are blocked", error=True)
            return False
        return True

    # ------------------------------------------------------------------ data

    def load_context(self) -> None:
        self.ctx = None
        self.ctx_error = ""
        try:
            self.ctx = load_ref(self.refs[self.current])
        except Exception as exc:
            self.ctx_error = str(exc)
            return
        self.activity = runner_activity(self.ctx)
        history = load_runner_history(self.ctx)
        self.statuses = get_stage_statuses(self.ctx, include_internal=True, history=history, activity=self.activity)
        for state in self.pass_states.values():
            state.selected.clear()
            result = analytics.collect_pass_rows(self.ctx, state.pass_name)  # type: ignore[arg-type]
            state.rows = result.rows
            state.warnings = result.warnings
        try:
            with open_store(self.ctx.workdir) as store:
                import_workdir_files(store, self.ctx.workdir)
                for state in self.pass_states.values():
                    if state.rows:
                        analytics.cache_pass_rows(
                            store,
                            state.pass_name,  # type: ignore[arg-type]
                            analytics.AnalyticsResult(
                                pass_name=state.pass_name, rows=state.rows, warnings=state.warnings
                            ),
                        )
        except Exception:
            pass  # store is a derived cache; never block the GUI on it

    def refresh(self, *_args) -> None:
        self.load_context()
        self.rebuild()

    # ------------------------------------------------------------------ build

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 4)
        root.setSpacing(6)

        top_bar = QHBoxLayout()
        top_bar.addWidget(make_label("PBBatch Manage", size=18, bold=True))
        top_bar.addSpacing(14)
        info_host = QWidget()
        self.top_info = QHBoxLayout(info_host)
        self.top_info.setContentsMargins(0, 0, 0, 0)
        self.top_info.setSpacing(14)
        top_bar.addWidget(info_host, 1)
        top_bar.addWidget(
            self._icon_button(QStyle.StandardPixmap.SP_DirOpenIcon, "Open workdir folder", self.open_workdir)
        )
        top_bar.addWidget(self._icon_button(QStyle.StandardPixmap.SP_MediaPlay, "Start plan (runner)", self.start_plan))
        top_bar.addWidget(self._icon_button(QStyle.StandardPixmap.SP_BrowserReload, "Refresh", self.refresh))
        root.addLayout(top_bar)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(divider)

        body = QHBoxLayout()
        self.rail: Optional[QListWidget] = None
        if len(self.refs) > 1:
            self.rail = QListWidget()
            self.rail.setFixedWidth(260)
            for ref in self.refs:
                item = QListWidgetItem(ref.label)
                item.setToolTip(str(ref.workdir))
                self.rail.addItem(item)
            self.rail.setCurrentRow(self.current)
            self.rail.currentRowChanged.connect(self.select_workdir)
            body.addWidget(self.rail)

        self.tabs = QTabWidget()
        self.tab_layouts: Dict[str, QVBoxLayout] = {}
        for name in TAB_NAMES:
            content = QWidget()
            layout = QVBoxLayout(content)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(10)
            self.tab_layouts[name] = layout
            if name in ("Config", "Raw JSON"):
                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setFrameShape(QFrame.Shape.NoFrame)
                scroll.setWidget(content)
                self.tabs.addTab(scroll, name)
            else:
                self.tabs.addTab(content, name)
        body.addWidget(self.tabs, 1)
        root.addLayout(body, 1)

        self.setCentralWidget(central)
        self.statusBar()

    def _icon_button(self, pixmap: QStyle.StandardPixmap, tooltip: str, handler: Callable) -> QToolButton:
        button = QToolButton()
        button.setIcon(self.style().standardIcon(pixmap))
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        button.clicked.connect(lambda: handler())
        return button

    def select_workdir(self, index: int) -> None:
        if index < 0 or index == self.current:
            return
        self.current = index
        self.refresh()

    def rebuild(self) -> None:
        self.rebuild_top_info()
        for layout in self.tab_layouts.values():
            clear_layout(layout)
        if self.ctx is None:
            for layout in self.tab_layouts.values():
                layout.addWidget(make_label(f"Failed to open workdir: {self.ctx_error}", color=COLOR_RED))
                layout.addStretch(1)
            return
        self.build_status_tab(self.tab_layouts["Status"])
        self.build_config_tab(self.tab_layouts["Config"])
        self.build_scenes_tab(self.tab_layouts["Scenes"])
        self.build_pass_tab(self.tab_layouts["Fastpass"], "fastpass")
        self.build_pass_tab(self.tab_layouts["Mainpass"], "mainpass")
        self.build_logs_tab(self.tab_layouts["Logs"])
        self.build_raw_tab(self.tab_layouts["Raw JSON"])

    def rebuild_top_info(self) -> None:
        clear_layout(self.top_info)
        if self.ctx is None:
            self.top_info.addWidget(make_label(self.ctx_error or "no workdir", color=COLOR_RED))
        else:
            summary = summarize_workdir(self.ctx)
            self.top_info.addWidget(make_label(summary.name, bold=True))
            self.top_info.addWidget(make_label(f"mode: {summary.mode}", color=COLOR_GREY_400))
            if self.ctx.source is not None:
                self.top_info.addWidget(make_label(str(self.ctx.source), size=11, color=COLOR_GREY_500))
            if summary.runner.active:
                badge = make_label(f"RUNNER ACTIVE (pid {summary.runner.pid})", size=11)
                badge.setStyleSheet(
                    f"background-color: {COLOR_RED_DARK}; color: white; padding: 3px 6px; "
                    "border-radius: 4px; font-size: 11px;"
                )
                self.top_info.addWidget(badge)
            done = summary.completed_stages
            self.top_info.addWidget(make_label(f"stages: {done}/{summary.total_stages} done", color=COLOR_GREY_400))
            warnings = summary.warnings + [w.message for w in manage_config.stale_config_warnings(self.ctx)]
            if warnings:
                self.top_info.addWidget(
                    make_label("⚠", color=COLOR_ORANGE, size=16, tooltip="\n".join(warnings))
                )
        self.top_info.addStretch(1)

    # ------------------------------------------------------------------ actions

    def open_workdir(self, *_args) -> None:
        if self.ctx is not None and self.ctx.workdir.exists():
            os.startfile(str(self.ctx.workdir))

    def start_plan(self, *_args) -> None:
        if self.ctx is None:
            return
        plan_path = self.ctx.plan_path
        if plan_path is None or not plan_path.exists():
            self.notify("no .plan file for this workdir; cannot start runner", error=True)
            return
        if self.activity is not None and self.activity.active:
            self.notify("runner is already active for this source folder", error=True)
            return

        def launch() -> None:
            toolchain = load_toolchain()
            cmd = [toolchain.python_exe, str(ROOT_DIR / "runner.py"), str(plan_path)]
            flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
            subprocess.Popen(cmd, cwd=str(ROOT_DIR), creationflags=flags)
            self.notify("runner started in a separate process; use Refresh to watch progress")

        self.confirm(
            "Start plan",
            [f"Launch runner for: {plan_path}", "Manage stays an offline editor and only observes progress."],
            launch,
            confirm_label="Start",
            danger=False,
        )

    # ------------------------------------------------------------------ Status tab

    def build_status_tab(self, layout: QVBoxLayout) -> None:
        assert self.ctx is not None
        if self.activity is not None and self.activity.warning:
            layout.addWidget(make_label(self.activity.warning, color=COLOR_ORANGE))
        backups = list_backups(self.ctx.workdir)
        toolbar = QHBoxLayout()
        toolbar.addWidget(make_label(f"backups: {len(backups)}", color=COLOR_GREY_500))
        toolbar.addWidget(link_button("audit log", self.show_manage_events))
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        table = make_table(
            ["Stage", "State", "Progress", "Started", "Elapsed", "Marker", "Reason / message", "Depends on", "Actions"]
        )
        table.setRowCount(len(self.statuses))
        for row, status in enumerate(self.statuses):
            color = STATE_COLORS.get(status.state, COLOR_GREY)
            elapsed = status.elapsed_seconds
            if status.state == STATE_RUNNING and status.started_at:
                elapsed = time.time() - status.started_at
            progress_text = f"{status.progress:.1f}%" if status.progress is not None else "-"
            name_label = status.name + ("  (internal)" if status.internal else "")
            reason = status.reason or status.last_message
            table.setItem(row, 0, make_item(name_label))
            table.setItem(row, 1, make_item(status.state, color=color, bold=True))
            table.setItem(row, 2, make_item(progress_text))
            table.setItem(row, 3, make_item(fmt_timestamp(status.started_at)))
            table.setItem(row, 4, make_item(fmt_duration(elapsed)))
            table.setItem(
                row, 5, make_item("yes" if status.marker_exists else ("-" if status.marker_path is None else "no"))
            )
            table.setItem(row, 6, make_item(reason, color=COLOR_GREY_400, tooltip=reason))
            deps = ", ".join(status.dependencies)
            table.setItem(row, 7, make_item(deps, color=COLOR_GREY_600, tooltip=deps))

            actions = QWidget()
            actions_layout = QHBoxLayout(actions)
            actions_layout.setContentsMargins(2, 0, 2, 0)
            actions_layout.setSpacing(4)
            reset_btn = QToolButton()
            reset_btn.setText("reset")
            reset_btn.setToolTip("Reset stage")
            reset_btn.clicked.connect(lambda _=False, s=status.name: self.on_reset_stage(s, chain=False))
            chain_btn = QToolButton()
            chain_btn.setText("chain")
            chain_btn.setToolTip("Reset chain (stage + downstream)")
            chain_btn.clicked.connect(lambda _=False, s=status.name: self.on_reset_stage(s, chain=True))
            actions_layout.addWidget(reset_btn)
            actions_layout.addWidget(chain_btn)
            actions_layout.addStretch(1)
            table.setCellWidget(row, 8, actions)
        fit_table(table, 6)  # "Reason / message" absorbs the remaining width
        layout.addWidget(table, 1)

    def show_manage_events(self) -> None:
        assert self.ctx is not None
        path = self.ctx.meta_dir / "manage_events.jsonl"
        if not path.exists():
            self.notify("no manage events yet")
            return
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
        pretty: List[str] = []
        for line in lines:
            try:
                payload = json.loads(line)
                pretty.append(
                    f"{fmt_timestamp(payload.get('timestamp'))}  {payload.get('operation')}  "
                    f"{payload.get('message') or ''}  ({len(payload.get('changed_paths') or [])} paths)"
                )
            except Exception:
                pretty.append(line)
        self.show_text_dialog("manage_events.jsonl (last 200)", "\n".join(pretty))

    def on_reset_stage(self, stage: str, *, chain: bool) -> None:
        assert self.ctx is not None
        if not self.destructive_allowed():
            return
        try:
            plan = manage_reset.preview_stage_reset(self.ctx, stage, chain=chain)
        except Exception as exc:
            self.notify(str(exc), error=True)
            return
        lines = [f"Stages: {', '.join(plan.stages)}", ""]
        existing = plan.existing_actions
        if not existing:
            lines.append("Nothing to delete (no files exist).")
        for action in existing:
            lines.append(f"[{action.action}] {action.path}")
        lines.extend(["", *plan.notes])

        def apply() -> None:
            result = manage_reset.reset_stage(self.ctx, stage, chain=chain)
            self.notify(f"reset done: {len(result.changed_paths)} paths changed")
            self.refresh()

        self.confirm(f"Reset {'chain' if chain else 'stage'}: {stage}", lines, apply, confirm_label="Reset")

    # ------------------------------------------------------------------ Config tab

    def build_config_tab(self, layout: QVBoxLayout) -> None:
        assert self.ctx is not None
        ctx = self.ctx

        info_lines = [
            f"workdir:  {ctx.workdir}",
            f"source:   {ctx.source or '-'}",
            f"plan:     {ctx.plan_path or '- (best-effort context)'}",
            f"batch:    {ctx.batch_plan_path or '-'}",
            f"zone:     {ctx.zone_file or '-'}",
            f"mode:     {ctx.mode}",
        ]
        info_frame = QFrame()
        info_frame.setStyleSheet(f"QFrame {{ border: 1px solid {COLOR_GREY_800}; border-radius: 6px; }}")
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(8, 8, 8, 8)
        info_layout.setSpacing(2)
        for line in info_lines:
            label = make_label(line, mono=True, selectable=True)
            label.setStyleSheet("border: none;")
            info_layout.addWidget(label)
        layout.addWidget(info_frame)

        stale = manage_config.stale_config_warnings(ctx)
        fingerprint_row = QHBoxLayout()
        fingerprint_row.addWidget(
            make_label(
                "; ".join(warning.message for warning in stale) if stale else "config fingerprint is confirmed",
                color=COLOR_ORANGE if stale else COLOR_GREEN,
            )
        )
        confirm_btn = link_button("Confirm hash", lambda: self.guard(self.on_confirm_fingerprint))
        fingerprint_row.addWidget(confirm_btn)
        fingerprint_row.addStretch(1)
        layout.addLayout(fingerprint_row)

        if ctx.plan_path is not None and ctx.plan_path.exists():
            layout.addWidget(self.build_plan_editor())
        else:
            layout.addWidget(make_label("No .plan file found — plan editing disabled.", color=COLOR_GREY_500))

        layout.addWidget(self.build_zone_editor())
        layout.addWidget(self.build_manifest_viewer())
        layout.addStretch(1)

    def on_confirm_fingerprint(self) -> None:
        assert self.ctx is not None
        manage_config.confirm_fingerprint(self.ctx)
        self.notify("config fingerprint confirmed")
        self.rebuild()

    def build_plan_editor(self) -> QWidget:
        assert self.ctx is not None and self.ctx.plan_path is not None
        try:
            plan = manage_config.load_plan_config(self.ctx.plan_path)
        except Exception as exc:
            return make_label(f"failed to load plan: {exc}", color=COLOR_RED)
        if not isinstance(plan, FilePlan):
            return make_label("batch plan editing is not supported; open the file plan instead", color=COLOR_GREY_500)

        primary = plan.video.primary
        quality = QLineEdit(str(primary.quality))
        quality.setFixedWidth(140)
        preset = QLineEdit(str(primary.preset or ""))
        preset.setFixedWidth(110)
        fp_preset = QLineEdit(str(primary.fastpass_preset or ""))
        fp_preset.setFixedWidth(130)
        fp_workers = QLineEdit(str(primary.fastpass_workers))
        fp_workers.setFixedWidth(130)
        mp_workers = QLineEdit(str(primary.mainpass_workers))
        mp_workers.setFixedWidth(130)
        chunk_order = QComboBox()
        chunk_order.addItems(["long-biased-random", "random", "sequential", "short-to-long", "long-to-short"])
        chunk_order.setCurrentText(primary.chunk_order)
        chunk_order.setFixedWidth(200)
        scene_detection = QComboBox()
        scene_detection.addItems(["psd", "av1an", "none"])
        scene_detection.setCurrentText(primary.scene_detection)
        scene_detection.setFixedWidth(140)
        no_fastpass = QCheckBox("no_fastpass")
        no_fastpass.setChecked(primary.no_fastpass)

        def params_text(params: Dict[str, Any]) -> str:
            return "\n".join(f"{key} {value}".rstrip() for key, value in params.items())

        fastpass_params = QPlainTextEdit(params_text(plan.video.fastpass_params))
        fastpass_params.setFont(mono_font())
        fastpass_params.setMinimumHeight(110)
        mainpass_params = QPlainTextEdit(params_text(plan.video.mainpass_params))
        mainpass_params.setFont(mono_font())
        mainpass_params.setMinimumHeight(80)

        def parse_params(text: str) -> Dict[str, Any]:
            from utils.plan import coerce_scalar

            out: Dict[str, Any] = {}
            for line_no, raw in enumerate(str(text or "").splitlines(), start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                key = parts[0]
                if not key.startswith("-"):
                    raise RuntimeError(f"params line {line_no}: key must start with '-': {key!r}")
                out[key] = coerce_scalar(parts[1].strip()) if len(parts) > 1 else ""
            return out

        def save() -> None:
            assert self.ctx is not None and self.ctx.plan_path is not None
            primary.quality = float(quality.text().replace(",", "."))
            primary.preset = preset.text().strip()
            primary.fastpass_preset = fp_preset.text().strip()
            primary.fastpass_workers = int(fp_workers.text().strip())
            primary.mainpass_workers = int(mp_workers.text().strip())
            primary.chunk_order = chunk_order.currentText() or primary.chunk_order
            primary.scene_detection = scene_detection.currentText() or primary.scene_detection
            primary.no_fastpass = no_fastpass.isChecked()
            plan.video.fastpass_params = parse_params(fastpass_params.toPlainText())
            plan.video.mainpass_params = parse_params(mainpass_params.toPlainText())
            tx = ManageTransaction(
                workdir=self.ctx.workdir, operation="edit_plan", source=self.ctx.source, plan=self.ctx.plan_path
            )
            manage_config.save_plan_config(plan, self.ctx.plan_path, tx=tx)
            tx.commit(message="plan edited from manage GUI")
            self.notify("plan saved (backup written); markers are NOT reset automatically")
            self.refresh()

        def on_save() -> None:
            self.confirm(
                "Save .plan",
                [
                    f"Write {self.ctx.plan_path}",
                    "A backup will be stored in 00_meta/manage_backups.",
                    "Stage markers are not reset automatically — use Reset on the Status tab if needed.",
                ],
                save,
                confirm_label="Save",
                danger=False,
            )

        save_btn = QPushButton("Save plan")
        save_btn.clicked.connect(on_save)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(10)
        content_layout.addWidget(
            widget_row(
                labeled("quality (CRF)", quality),
                labeled("preset", preset),
                labeled("fastpass preset", fp_preset),
                labeled("fastpass workers", fp_workers),
                labeled("mainpass workers", mp_workers),
            )
        )
        content_layout.addWidget(
            widget_row(
                labeled("chunk order", chunk_order),
                labeled("scene detection", scene_detection),
                no_fastpass,
            )
        )
        content_layout.addWidget(labeled("video.fastpass.params (key value per line)", fastpass_params))
        content_layout.addWidget(labeled("video.mainpass.params (key value per line)", mainpass_params))
        content_layout.addWidget(widget_row(save_btn))
        return CollapsibleSection(".plan editor", content)

    def build_zone_editor(self) -> QWidget:
        assert self.ctx is not None
        zone_path = self.ctx.zone_file
        text = manage_config.load_zone_text(zone_path) if zone_path is not None else ""
        editor = QPlainTextEdit(text)
        editor.setFont(mono_font())
        editor.setMinimumHeight(140)
        status_text = QLabel("")

        def validate() -> None:
            issues = manage_config.validate_zone_text(self.ctx, editor.toPlainText())
            if issues:
                status_text.setText("; ".join(issue.message for issue in issues))
                status_text.setStyleSheet(f"color: {COLOR_RED};")
            else:
                status_text.setText("OK: no parse errors")
                status_text.setStyleSheet(f"color: {COLOR_GREEN};")

        def save() -> None:
            tx = ManageTransaction(
                workdir=self.ctx.workdir, operation="edit_zone", source=self.ctx.source, plan=self.ctx.plan_path
            )
            manage_config.save_zone_text(self.ctx, zone_path, editor.toPlainText(), tx=tx)
            tx.commit(message="zoned_command.txt edited from manage GUI")
            self.notify("zone file saved (backup written)")
            self.refresh()

        def on_save() -> None:
            self.confirm(
                "Save zone file",
                [
                    f"Write {zone_path}",
                    "Zone stages are not reset automatically; reset the Zone chain to re-apply.",
                ],
                save,
                confirm_label="Save",
                danger=False,
            )

        validate_btn = QPushButton("Validate")
        validate_btn.clicked.connect(lambda: self.guard(validate))
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(on_save)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(8)
        content_layout.addWidget(editor)
        content_layout.addWidget(widget_row(validate_btn, save_btn, status_text))
        return CollapsibleSection(f"zoned_command.txt ({zone_path})", content)

    def build_manifest_viewer(self) -> QWidget:
        assert self.ctx is not None
        refs = [ref for ref in manage_config.list_config_files(self.ctx) if ref.kind == "manifest" and ref.exists]
        if not refs:
            return make_label("no manifests found", color=COLOR_GREY_600)
        buttons = [
            link_button(ref.path.name, lambda p=ref.path: self.show_json_file(p))
            for ref in refs
        ]
        return widget_row(make_label("manifests:"), *buttons)

    def show_json_file(self, path: Path) -> None:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception as exc:
            self.notify(f"failed to read {path}: {exc}", error=True)
            return
        if len(text) > 400_000:
            text = text[:400_000] + "\n... (truncated)"
        self.show_text_dialog(str(path), text)

    # ------------------------------------------------------------------ Scenes tab

    def build_scenes_tab(self, layout: QVBoxLayout) -> None:
        assert self.ctx is not None
        refs = [ref for ref in manage_scenes.list_scene_files(self.ctx) if ref.exists]
        if not refs:
            layout.addWidget(make_label("no scene files found", color=COLOR_GREY_500))
            layout.addStretch(1)
            return

        selector = QComboBox()
        for ref in refs:
            selector.addItem(f"{ref.relative} — {ref.description}", ref.relative)
        selector.setMinimumWidth(420)
        search = QLineEdit()
        search.setPlaceholderText("find frame / time (hh:mm:ss)")
        search.setFixedWidth(220)

        host = QWidget()
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(8)

        def selected_path() -> Path:
            return self.ctx.workdir / Path(str(selector.currentData()))

        def render(*_args) -> None:
            self.guard(lambda: self._render_scene_table(host_layout, selected_path(), search.text()))

        show_btn = QPushButton("Show")
        show_btn.clicked.connect(render)

        layout.addWidget(
            widget_row(labeled("scene file", selector), labeled("find frame / time", search), show_btn)
        )
        layout.addWidget(
            make_label(
                "double-click a zone video_params cell to edit that scene's overrides; "
                "for bulk command-based edits use Patch params… on the Fastpass / Mainpass tabs",
                color=COLOR_GREY_600,
                size=11,
            )
        )
        self._render_scene_table(host_layout, selected_path(), "")
        selector.currentIndexChanged.connect(render)
        search.returnPressed.connect(render)
        layout.addWidget(host, 1)

    def _scene_fps(self) -> float:
        for state in self.pass_states.values():
            for row in state.rows:
                if row.frames > 0 and row.duration_seconds > 0:
                    return row.frames / row.duration_seconds
        return 0.0

    def _render_scene_table(self, host_layout: QVBoxLayout, path: Path, search_value: str) -> None:
        clear_layout(host_layout)
        scene_file = manage_scenes.load_scene_file(path)
        issues = manage_scenes.validate_scene_file(scene_file)
        fps = self._scene_fps()
        target_frame = parse_time_or_frame(search_value, fps) if search_value else None
        if target_frame is None and search_value and str(search_value).strip().isdigit():
            target_frame = int(str(search_value).strip())

        host_layout.addWidget(
            make_label(
                f"frames: {scene_file.frames} | scenes: {len(scene_file.scenes)} | split_scenes: {len(scene_file.split_scenes)}"
                + (f" | filtered around frame {target_frame}" if target_frame is not None else ""),
                color=COLOR_GREY_400,
            )
        )
        problems = [issue for issue in issues if issue.is_error]
        if problems:
            host_layout.addWidget(make_label("; ".join(issue.message for issue in problems), color=COLOR_RED))

        table = make_table(["#", "start", "end", "frames", "time", "CRF", "zone video_params"])
        table.verticalHeader().setDefaultSectionSize(26)
        scenes = scene_file.effective_scenes
        rows: List[List[QTableWidgetItem]] = []
        for position, scene in enumerate(scenes):
            start = int(scene.get("start_frame") or 0)
            end = int(scene.get("end_frame") or 0)
            if target_frame is not None and not (start <= target_frame < end):
                continue
            overrides = scene.get("zone_overrides") if isinstance(scene.get("zone_overrides"), dict) else {}
            params = overrides.get("video_params") if isinstance(overrides, dict) else None
            params_text = " ".join(str(token) for token in params) if isinstance(params, list) else ""
            crf = analytics.parse_crf_from_video_params([str(token) for token in (params or [])])
            rows.append(
                [
                    make_item(str(position), sort_key=position),
                    make_item(str(start), sort_key=start),
                    make_item(str(end), sort_key=end),
                    make_item(str(end - start), sort_key=end - start),
                    make_item(fmt_frame_time(start, fps) if fps else "-", sort_key=start),
                    make_item("-" if crf is None else f"{crf:g}", sort_key=-1.0 if crf is None else float(crf)),
                    make_item(
                        params_text,
                        mono=True,
                        tooltip=params_text,
                        user_data=(start, end, params_text),
                    ),
                ]
            )
        table.setRowCount(len(rows))
        for row_index, items in enumerate(rows):
            for col_index, item in enumerate(items):
                table.setItem(row_index, col_index, item)
        fit_table(table, 6)  # zone video_params absorbs the remaining width
        table.setSortingEnabled(True)

        def on_double_clicked(item: QTableWidgetItem) -> None:
            if item.column() != 6:
                return
            meta = item.data(Qt.ItemDataRole.UserRole)
            if not meta:
                return
            start, end, params_text = meta
            self.on_edit_scene_zone(path, start, end, params_text)

        table.itemDoubleClicked.connect(on_double_clicked)
        host_layout.addWidget(table, 1)

    def on_edit_scene_zone(self, scene_path: Path, start: int, end: int, current_params: str) -> None:
        if not self.destructive_allowed():
            return
        params_field = QLineEdit(current_params)
        params_field.setFont(mono_font())
        params_field.setMinimumWidth(480)

        def apply() -> None:
            params = [token for token in params_field.text().split() if token]
            tx = ManageTransaction(
                workdir=self.ctx.workdir, operation="patch_zone", source=self.ctx.source, plan=self.ctx.plan_path
            )
            result = manage_scenes.patch_scene_region(
                scene_path,
                manage_scenes.SceneSelector(start_frame=start, end_frame=end),
                manage_scenes.ZonePatch(video_params=params),
                tx=tx,
            )
            tx.commit(message=f"zone patch {start}..{end} on {scene_path.name}")
            self.notify(f"patched {len(result.patched_positions)} scene(s); downstream stages NOT reset automatically")
            self.refresh()

        self.show_dialog_form(
            f"Edit zone params — scene [{start}, {end}) — {scene_path.name}",
            [
                make_label(
                    "Replaces zone_overrides.video_params for this scene. "
                    "A backup is written; reset the affected chain afterwards.",
                    color=COLOR_GREY_400,
                    wrap=True,
                ),
                labeled("video_params (full list, e.g. --crf 25 --preset 4)", params_field),
            ],
            apply,
            submit_label="Patch",
        )

    # ------------------------------------------------------------------ Pass tabs

    def build_pass_tab(self, layout: QVBoxLayout, pass_name: str) -> None:
        assert self.ctx is not None
        state = self.pass_states[pass_name]
        pdir = av1an_state.pass_dir(self.ctx, pass_name)  # type: ignore[arg-type]
        if not (pdir / "chunks.json").exists():
            layout.addWidget(make_label(f"{pass_name}: no chunks.json (pass not started)", color=COLOR_GREY_500))
            layout.addStretch(1)
            return

        rows = analytics.sort_pass_rows(state.rows, analytics.SortSpec(state.sort_field, state.sort_desc))
        rows = state.apply_filters(rows)

        chapters = sorted({row.chapter for row in state.rows if row.chapter})
        done_dd = QComboBox()
        done_dd.addItems(["all", "done", "not done"])
        done_dd.setCurrentText(state.filter_done)
        done_dd.setFixedWidth(120)
        chapter_dd = QComboBox()
        chapter_dd.addItems(["(all)"] + chapters)
        chapter_dd.setCurrentText(state.filter_chapter or "(all)")
        chapter_dd.setFixedWidth(200)
        crf_min_tf = QLineEdit(state.filter_crf_min)
        crf_min_tf.setFixedWidth(90)
        crf_max_tf = QLineEdit(state.filter_crf_max)
        crf_max_tf.setFixedWidth(90)
        outliers_cb = QCheckBox("bitrate outliers (×2 / ÷2 of median)")
        outliers_cb.setChecked(state.filter_outliers)

        def on_filter(*_args) -> None:
            state.filter_done = done_dd.currentText() or "all"
            chapter_value = chapter_dd.currentText()
            state.filter_chapter = "" if chapter_value == "(all)" else chapter_value
            state.filter_crf_min = crf_min_tf.text()
            state.filter_crf_max = crf_max_tf.text()
            state.filter_outliers = outliers_cb.isChecked()
            self.rebuild()

        default_quarantine = av1an_state.DEFAULT_POLICY_BY_PASS[pass_name] == "quarantine"
        quarantine_cb = QCheckBox("quarantine encode output (instead of delete)")
        quarantine_cb.setChecked(default_quarantine if state.policy_quarantine is None else state.policy_quarantine)

        def policy() -> str:
            state.policy_quarantine = quarantine_cb.isChecked()
            return "quarantine" if quarantine_cb.isChecked() else "delete"

        def action_button(text: str, handler: Callable[[], None]) -> QPushButton:
            button = QPushButton(text)
            button.clicked.connect(lambda: handler())
            return button

        toolbar_widgets: List[QWidget] = [
            quarantine_cb,
            action_button("Reset chunk(s)", lambda: self.on_reset_chunks(pass_name, policy())),
            action_button("Edit params…", lambda: self.on_edit_chunk_params(pass_name, policy())),
            action_button("Patch params…", lambda: self.on_patch_pass_params(pass_name, policy())),
            action_button("Split…", lambda: self.on_split_chunk(pass_name, policy())),
            action_button("Merge", lambda: self.on_merge_chunks(pass_name, policy())),
            action_button("Reshape…", lambda: self.on_reshape_range(pass_name, policy())),
            action_button("Move…", lambda: self.on_move_chunks(pass_name)),
            action_button("Sort queue…", lambda: self.on_sort_queue(pass_name)),
        ]
        layout.addWidget(widget_row(*toolbar_widgets))

        filter_widgets: List[QWidget] = [labeled("done", done_dd)]
        if chapters:
            filter_widgets.append(labeled("chapter", chapter_dd))
        filter_widgets += [labeled("CRF ≥", crf_min_tf), labeled("CRF ≤", crf_max_tf), outliers_cb]
        layout.addWidget(widget_row(*filter_widgets))

        all_rows = state.rows
        shown_rows = rows
        stats_host = QWidget()
        stats_row = QHBoxLayout(stats_host)
        stats_row.setContentsMargins(0, 0, 0, 0)
        stats_row.setSpacing(0)
        stats_row.addWidget(make_label(format_chunk_stats(all_rows), color=COLOR_GREY_400))
        stats_row.addSpacing(48)
        subset_label = make_label("", color=COLOR_GREY_500)
        stats_row.addWidget(subset_label)
        stats_row.addStretch(1)
        layout.addWidget(stats_host)

        def update_subset_label() -> None:
            # after the gap: selected chunks (when 2+ ticked), else the filtered
            # subset (when filters hide rows), else nothing
            selected_rows = [row for row in all_rows if row.index in state.selected]
            if len(selected_rows) > 1:
                subset_label.setText("selected " + format_chunk_stats(selected_rows))
            elif len(shown_rows) != len(all_rows):
                subset_label.setText("shown " + format_chunk_stats(shown_rows))
            else:
                subset_label.setText("")

        update_subset_label()
        if state.warnings:
            layout.addWidget(make_label("; ".join(state.warnings), color=COLOR_ORANGE, wrap=True))

        is_fastpass = pass_name == "fastpass"
        headers = ["", "idx", "queue", "done", "start", "end", "frames", "dur", "size", "kbps", "CRF"]
        if is_fastpass:
            headers += ["ssimu2", "p5"]
        headers += ["chapter", ""]

        table = ChunkTable(headers)
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            select_item = QTableWidgetItem()
            select_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            select_item.setCheckState(
                Qt.CheckState.Checked if row.index in state.selected else Qt.CheckState.Unchecked
            )
            select_item.setData(Qt.ItemDataRole.UserRole, row.index)
            table.setItem(row_index, 0, select_item)
            cells = [
                make_item(str(row.index), mono=True),
                make_item(str(row.queue_position)),
                make_item("✓" if row.done else "·", color=COLOR_GREEN if row.done else COLOR_GREY_600, bold=row.done),
                make_item(str(row.start_frame)),
                make_item(str(row.end_frame)),
                make_item(str(row.frames)),
                make_item(fmt_duration(row.duration_seconds)),
                make_item(fmt_size(row.output_size)),
                make_item("-" if row.bitrate_kbps is None else f"{row.bitrate_kbps:.0f}"),
                make_item("-" if row.crf is None else f"{row.crf:g}"),
            ]
            if is_fastpass:
                cells += [
                    make_item("-" if row.ssimu2_avg is None else f"{row.ssimu2_avg:.2f}"),
                    make_item("-" if row.ssimu2_p5 is None else f"{row.ssimu2_p5:.2f}"),
                ]
            cells += [make_item(row.chapter[:24], tooltip=row.chapter)]
            for col_offset, item in enumerate(cells, start=1):
                table.setItem(row_index, col_offset, item)

            open_btn = QToolButton()
            open_btn.setText("open")
            open_btn.setToolTip("Open encode output")
            open_btn.setEnabled(row.output_exists)
            open_btn.clicked.connect(lambda _=False, r=row: self.guard(lambda: analytics.open_chunk_external(r)))
            table.setCellWidget(row_index, len(headers) - 1, open_btn)

        def on_item_changed(item: QTableWidgetItem) -> None:
            if item.column() != 0:
                return
            index = item.data(Qt.ItemDataRole.UserRole)
            if index is None:
                return
            if item.checkState() == Qt.CheckState.Checked:
                state.selected.add(index)
            else:
                state.selected.discard(index)
            update_subset_label()

        table.itemChanged.connect(on_item_changed)
        fit_table(table, headers.index("chapter"))
        layout.addWidget(table, 1)

        # Explorer-style sorting: click a column header to sort, click again to
        # flip direction. Sorting is done on the row model (analytics.sort_pass_rows)
        # so checkboxes and cell widgets stay attached to their rows.
        sort_field_by_header = {
            "idx": "index",
            "queue": "queue_position",
            "done": "done",
            "start": "start_frame",
            "end": "end_frame",
            "frames": "frames",
            "dur": "duration_seconds",
            "size": "output_size",
            "kbps": "bitrate_kbps",
            "CRF": "crf",
            "ssimu2": "ssimu2_avg",
            "p5": "ssimu2_p5",
        }
        field_by_column = {
            column: sort_field_by_header[name]
            for column, name in enumerate(headers)
            if name in sort_field_by_header
        }
        header = table.horizontalHeader()
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        sort_order = Qt.SortOrder.DescendingOrder if state.sort_desc else Qt.SortOrder.AscendingOrder
        indicator_column = next(
            (column for column, field_name in field_by_column.items() if field_name == state.sort_field), -1
        )
        header.setSortIndicator(indicator_column, sort_order)

        def on_header_clicked(column: int) -> None:
            field_name = field_by_column.get(column)
            if field_name is None:
                return
            if state.sort_field == field_name:
                state.sort_desc = not state.sort_desc
            else:
                state.sort_field = field_name
                state.sort_desc = False
            self.rebuild()

        header.sectionClicked.connect(on_header_clicked)

        def on_crf_edited(*_args) -> None:
            # editingFinished also fires on plain focus changes; skip the
            # rebuild when nothing actually changed
            if state.filter_crf_min == crf_min_tf.text() and state.filter_crf_max == crf_max_tf.text():
                return
            on_filter()

        done_dd.currentTextChanged.connect(on_filter)
        chapter_dd.currentTextChanged.connect(on_filter)
        outliers_cb.toggled.connect(on_filter)
        crf_min_tf.editingFinished.connect(on_crf_edited)
        crf_max_tf.editingFinished.connect(on_crf_edited)

    def _selected_indices(self, pass_name: str) -> List[int]:
        return sorted(self.pass_states[pass_name].selected)

    def _sync_scenes_after_geometry(self, pass_name: str, tx: ManageTransaction) -> List[str]:
        """Update split_scenes in the scene files av1an may re-read for this pass."""
        assert self.ctx is not None
        pdir = av1an_state.pass_dir(self.ctx, pass_name)  # type: ignore[arg-type]
        chunks = av1an_state.load_chunks(pdir)
        targets: List[Path] = []
        pass_scenes = pdir / "scenes.json"
        if pass_scenes.exists():
            targets.append(pass_scenes)
        if pass_name == "mainpass":
            final_scenes = self.ctx.workdir / "video" / "scenes-final.json"
            if final_scenes.exists():
                targets.append(final_scenes)
        synced: List[str] = []
        for target in targets:
            manage_scenes.rebuild_split_scenes_for_chunks(target, chunks, tx=tx)
            synced.append(str(target))
        return synced

    def _geometry_tx(self, operation: str) -> ManageTransaction:
        assert self.ctx is not None
        return ManageTransaction(
            workdir=self.ctx.workdir, operation=operation, source=self.ctx.source, plan=self.ctx.plan_path
        )

    def _post_geometry_cleanup(self, pass_name: str, tx: ManageTransaction) -> None:
        """Invalidate downstream artifacts after a chunks/done edit."""
        assert self.ctx is not None
        from utils.runner_state import STAGE_FASTPASS, STAGE_MUX, STAGE_SSIMU2, STAGE_VERIFY, stage_marker_path
        from utils.runner_state import autoboost_fastpass_output
        from utils.manage.context import make_runner_item

        item = make_runner_item(self.ctx)
        if pass_name == "fastpass":
            for path in (autoboost_fastpass_output(item), stage_marker_path(item, STAGE_FASTPASS)):
                if path is not None:
                    tx.delete_file(path)
            downstream = manage_reset.preview_stage_reset(self.ctx, STAGE_SSIMU2, chain=True)
            for action in downstream.actions:
                if action.action == "delete_tree":
                    tx.delete_tree(action.path)
                else:
                    tx.delete_file(action.path)
        else:
            for path in (
                self.ctx.workdir / "video" / "video-final.mkv",
                stage_marker_path(item, STAGE_VERIFY),
                stage_marker_path(item, STAGE_MUX),
            ):
                if path is not None:
                    tx.delete_file(path)

    def on_reset_chunks(self, pass_name: str, policy: str) -> None:
        assert self.ctx is not None
        if not self.destructive_allowed():
            return
        indices = self._selected_indices(pass_name)
        if not indices:
            self.notify("select chunk(s) first")
            return
        previews: List[str] = [f"Pass: {pass_name} | policy: {policy}", f"Chunks: {indices}", ""]
        try:
            plan = manage_reset.preview_chunk_reset(self.ctx, pass_name, indices[0], policy=policy)  # type: ignore[arg-type]
            previews += [f"[{action.action}] {action.path}" for action in plan.existing_actions]
            previews += ["", *plan.notes]
            if len(indices) > 1:
                previews.append(f"(same applies to the other {len(indices) - 1} selected chunk(s))")
        except Exception as exc:
            self.notify(str(exc), error=True)
            return

        def apply() -> None:
            for index in indices:
                manage_reset.reset_chunk(self.ctx, pass_name, index, policy=policy)  # type: ignore[arg-type]
            self.notify(f"reset {len(indices)} chunk(s)")
            self.refresh()

        self.confirm(f"Reset chunks — {pass_name}", previews, apply, confirm_label="Reset")

    def on_edit_chunk_params(self, pass_name: str, policy: str) -> None:
        assert self.ctx is not None
        if not self.destructive_allowed():
            return
        indices = self._selected_indices(pass_name)
        if not indices:
            self.notify("select one or more chunks")
            return
        pdir = av1an_state.pass_dir(self.ctx, pass_name)  # type: ignore[arg-type]
        by_index = {chunk.index: chunk for chunk in av1an_state.load_chunks(pdir)}
        selected = [by_index[index] for index in indices if index in by_index]
        if not selected:
            self.notify("selected chunk(s) not found", error=True)
            return
        target_indices = [chunk.index for chunk in selected]
        multi = len(selected) > 1
        if multi:
            template_text = " ".join(av1an_state.merge_video_params([chunk.video_params for chunk in selected]))
        else:
            template_text = " ".join(selected[0].video_params)
        params_field = QPlainTextEdit(template_text)
        params_field.setFont(mono_font())
        params_field.setMinimumSize(560, 80)

        def apply() -> None:
            template_tokens = [token for token in params_field.toPlainText().split() if token]
            tx = self._geometry_tx("edit_chunk_params")
            result = av1an_state.update_chunks_params(
                pdir, target_indices, template_tokens, policy=policy, tx=tx  # type: ignore[arg-type]
            )
            if not result.changed_indices:
                self.notify("no changes")
                return
            if result.reset_indices:
                self._post_geometry_cleanup(pass_name, tx)
            tx.commit(
                message=f"edit video_params on {len(result.changed_indices)} chunk(s) ({pass_name})",
                extra={"changed": result.changed_indices, "reset": result.reset_indices},
            )
            reset_note = (
                f"; {len(result.reset_indices)} done-chunk(s) reset + downstream cleared"
                if result.reset_indices
                else ""
            )
            self.notify(f"params saved for {len(result.changed_indices)} chunk(s){reset_note}")
            self.refresh()

        if multi:
            title = f"Edit video_params — {len(selected)} chunks — {pass_name}"
            note = make_label(
                f"Editing {len(selected)} chunks at once. A param whose value differs across them is shown as "
                f"<code>{av1an_state.VARIABLE_PLACEHOLDER}</code> — leave it to keep each chunk's own value, or replace "
                "it to set one value for every selected chunk. Identical params apply to all; removing a param removes "
                "it from all. Only video_params is editable; changed done-chunks are reset and downstream invalidated.",
                color=COLOR_GREY_400,
                wrap=True,
            )
        else:
            title = f"Edit chunk {selected[0].index} video_params — {pass_name}"
            note = make_label(
                "Only video_params is editable (encoder/passes/target_quality stay in sync with av1an). "
                "If the chunk was done it will be reset and downstream stages invalidated.",
                color=COLOR_GREY_400,
                wrap=True,
            )
        self.show_dialog_form(title, [note, labeled("video_params", params_field)], apply)

    def on_patch_pass_params(self, pass_name: str, policy: str) -> None:
        assert self.ctx is not None
        if not self.destructive_allowed():
            return
        scene_path = zone_patch.pass_source_scene_file(self.ctx, pass_name)  # type: ignore[arg-type]
        commands_field = QPlainTextEdit()
        commands_field.setFont(mono_font())
        commands_field.setMinimumSize(620, 150)
        commands_field.setPlaceholderText("100f700 - --crf 25 --preset 4\n5s12 - --crf -10%\n# @preset and ! / ^ selectors supported")

        def apply() -> None:
            text = commands_field.toPlainText().strip()
            if not text:
                raise RuntimeError("enter at least one command")
            pdir = av1an_state.pass_dir(self.ctx, pass_name)  # type: ignore[arg-type]
            chunks = av1an_state.load_chunks(pdir)
            total_frames = max((chunk.end_frame for chunk in chunks), default=0) or None
            video = zone_patch.build_video_info(self.ctx.source, self._scene_fps())
            commands = zone_patch.parse_patch_commands(text, video=video, total_frames=total_frames)
            tx = self._geometry_tx("patch_pass_params")
            result = zone_patch.apply_param_commands_to_chunks(pdir, commands, policy=policy, tx=tx)  # type: ignore[arg-type]
            if not result.changed_indices:
                self.notify("no chunks matched the command(s) — nothing changed")
                return
            scenes_updated = zone_patch.mirror_params_to_scene_file(scene_path, result.range_params, tx=tx)
            if result.reset_indices:
                self._post_geometry_cleanup(pass_name, tx)
            tx.commit(
                message=f"patch params on {len(result.changed_indices)} chunk(s) ({pass_name})",
                extra={
                    "changed": result.changed_indices,
                    "reset": result.reset_indices,
                    "scene_file": str(scene_path),
                    "scene_entries_updated": scenes_updated,
                },
            )
            reset_note = f"; {len(result.reset_indices)} done-chunk(s) reset + downstream cleared" if result.reset_indices else ""
            scene_note = f"; {scenes_updated} scene entr(ies) in {scene_path.name} updated" if scenes_updated else ""
            self.notify(f"patched {len(result.changed_indices)} chunk(s){reset_note}{scene_note}")
            self.refresh()

        note = make_label(
            "Edit video_params for many chunks at once with the zone-edit command DSL "
            "(one command per line). Selectors: <code>100f700</code> frames, <code>5s12</code> chunk indexes, "
            "<code>1:00t2:00</code> time, or a chapter title; separators <code>-</code> (≥half overlap) / "
            "<code>!</code> (any overlap) / <code>^</code> (fully inside). Actions are <code>--key value</code> "
            "pairs; <code>--crf</code> also takes <code>+5</code> / <code>-10%</code> / <code>-</code> (delete). "
            f"<code>@name</code> presets work. Boundary mode (| … |) is not supported here. Matched done-chunks are "
            f"reset and downstream is invalidated; the same edit is mirrored into {scene_path.name}.",
            color=COLOR_GREY_400,
            wrap=True,
        )
        self.show_dialog_form(
            f"Patch params — {pass_name}",
            [note, labeled("zone-edit commands", commands_field)],
            apply,
            submit_label="Patch",
            width=720,
        )

    def _geometry_warning(self, pass_name: str) -> str:
        if pass_name == "mainpass":
            return (
                "Mainpass geometry edit: untouched chunks keep their encode state via reindexing; "
                "changed chunks are reset, video-final.mkv and Verify/Mux markers are cleared, "
                "split_scenes are synced in mainpass/scenes.json and scenes-final.json."
            )
        return (
            "Fastpass geometry edit: untouched chunks are reindexed with their encode state; "
            "changed chunks are reset and the SSIMU2 chain (zones, mainpass, verify, mux) is invalidated."
        )

    def on_split_chunk(self, pass_name: str, policy: str) -> None:
        assert self.ctx is not None
        if not self.destructive_allowed():
            return
        indices = self._selected_indices(pass_name)
        if len(indices) != 1:
            self.notify("select exactly one chunk to split")
            return
        index = indices[0]
        frame_field = QLineEdit()
        frame_field.setFixedWidth(220)

        def apply() -> None:
            frame = int(frame_field.text().strip())
            pdir = av1an_state.pass_dir(self.ctx, pass_name)  # type: ignore[arg-type]
            tx = self._geometry_tx("split_chunk")
            result = av1an_state.split_chunk(pdir, index, frame, policy=policy, tx=tx)  # type: ignore[arg-type]
            synced = self._sync_scenes_after_geometry(pass_name, tx)
            self._post_geometry_cleanup(pass_name, tx)
            tx.commit(
                message=f"split chunk {index} at frame {frame} ({pass_name})",
                extra={"reindexed": len(result.reindexed), "scenes_synced": synced},
            )
            self.notify(f"chunk {index} split; {len(result.reindexed)} chunks reindexed")
            self.refresh()

        self.show_dialog_form(
            f"Split chunk {index} — {pass_name}",
            [
                make_label(self._geometry_warning(pass_name), color=COLOR_ORANGE, wrap=True),
                labeled("split at frame (absolute)", frame_field),
            ],
            apply,
            submit_label="Split",
        )

    def on_merge_chunks(self, pass_name: str, policy: str) -> None:
        assert self.ctx is not None
        if not self.destructive_allowed():
            return
        indices = self._selected_indices(pass_name)
        if len(indices) < 2:
            self.notify("select two or more consecutive chunks")
            return

        def apply() -> None:
            pdir = av1an_state.pass_dir(self.ctx, pass_name)  # type: ignore[arg-type]
            tx = self._geometry_tx("merge_chunks")
            result = av1an_state.merge_chunks(pdir, indices, policy=policy, tx=tx)  # type: ignore[arg-type]
            synced = self._sync_scenes_after_geometry(pass_name, tx)
            self._post_geometry_cleanup(pass_name, tx)
            tx.commit(
                message=f"merge chunks {indices} ({pass_name})",
                extra={"reindexed": len(result.reindexed), "scenes_synced": synced},
            )
            self.notify(f"merged {len(indices)} chunks into {result.new_indices[0]}")
            self.refresh()

        self.confirm(
            f"Merge chunks — {pass_name}",
            [f"Chunks: {indices}", "", self._geometry_warning(pass_name)],
            apply,
            confirm_label="Merge",
        )

    def on_reshape_range(self, pass_name: str, policy: str) -> None:
        assert self.ctx is not None
        if not self.destructive_allowed():
            return
        fps = self._scene_fps()
        start_field = QLineEdit()
        start_field.setFixedWidth(240)
        end_field = QLineEdit()
        end_field.setFixedWidth(240)

        def apply() -> None:
            start = parse_time_or_frame(start_field.text(), fps)
            end = parse_time_or_frame(end_field.text(), fps)
            if start is None or end is None:
                raise RuntimeError("enter range as frame numbers or hh:mm:ss timestamps")
            pdir = av1an_state.pass_dir(self.ctx, pass_name)  # type: ignore[arg-type]
            tx = self._geometry_tx("reshape_chunk_range")
            result = av1an_state.reshape_chunk_range(pdir, FrameRange(start, end), policy=policy, tx=tx)  # type: ignore[arg-type]
            synced = self._sync_scenes_after_geometry(pass_name, tx)
            self._post_geometry_cleanup(pass_name, tx)
            tx.commit(
                message=f"reshape [{start}, {end}) into one chunk ({pass_name})",
                extra={"reset_indices": result.reset_indices, "scenes_synced": synced},
            )
            self.notify(f"range [{start}, {end}) reshaped; reset chunks: {result.reset_indices}")
            self.refresh()

        self.show_dialog_form(
            f"Reshape frame range — {pass_name}",
            [
                make_label(
                    "Splits chunks at the range edges, then merges everything inside into one chunk. "
                    + self._geometry_warning(pass_name),
                    color=COLOR_ORANGE,
                    wrap=True,
                ),
                widget_row(
                    labeled("range start (frame or hh:mm:ss)", start_field),
                    labeled("range end (frame or hh:mm:ss)", end_field),
                ),
            ],
            apply,
            submit_label="Reshape",
        )

    def on_move_chunks(self, pass_name: str) -> None:
        assert self.ctx is not None
        indices = self._selected_indices(pass_name)
        if not indices:
            self.notify("select chunk(s) to move")
            return
        kind_dd = QComboBox()
        kind_dd.addItems(["start", "end", "before", "after"])
        kind_dd.setFixedWidth(160)
        anchor_field = QLineEdit()
        anchor_field.setFixedWidth(260)
        order_dd = QComboBox()
        order_dd.addItems(["original", "ascending", "descending"])
        order_dd.setFixedWidth(200)

        def apply() -> None:
            kind = kind_dd.currentText()
            anchor = int(anchor_field.text().strip()) if kind in ("before", "after") else None
            pdir = av1an_state.pass_dir(self.ctx, pass_name)  # type: ignore[arg-type]
            tx = self._geometry_tx("move_chunks")
            av1an_state.move_chunks(
                pdir, indices, MoveTarget(kind=kind, anchor_index=anchor), order=order_dd.currentText(), tx=tx  # type: ignore[arg-type]
            )
            tx.commit(message=f"move chunks {indices} {kind} ({pass_name})")
            self.notify("queue order updated (affects only not-done chunks on av1an resume)")
            self.refresh()

        self.show_dialog_form(
            f"Move chunks — {pass_name}",
            [
                make_label(
                    "Only the av1an processing order changes; indices and done state stay intact. "
                    "Order matters only for chunks av1an has not finished yet.",
                    color=COLOR_GREY_400,
                    wrap=True,
                ),
                widget_row(
                    labeled("target", kind_dd),
                    labeled("anchor chunk index (for before/after)", anchor_field),
                    labeled("order of moved set", order_dd),
                ),
            ],
            apply,
            submit_label="Move",
        )

    def on_sort_queue(self, pass_name: str) -> None:
        assert self.ctx is not None
        algo_dd = QComboBox()
        algo_dd.addItems(list(av1an_state.CHUNK_SORT_ALGORITHMS))
        algo_dd.setCurrentText("long-to-short")
        algo_dd.setFixedWidth(240)
        desc_cb = QCheckBox("descending (for bitrate/ssimu2)")
        desc_cb.setChecked(True)

        def apply() -> None:
            algorithm = algo_dd.currentText()
            pdir = av1an_state.pass_dir(self.ctx, pass_name)  # type: ignore[arg-type]
            key_values: Optional[Dict[int, float]] = None
            if algorithm in ("bitrate", "ssimu2"):
                rows = self.pass_states[pass_name].rows
                if pass_name == "mainpass":
                    chunks = av1an_state.load_chunks(pdir)
                    stats = analytics.fastpass_stats_for_mainpass(self.ctx, chunks)
                    key = "bitrate_kbps" if algorithm == "bitrate" else "ssimu2_avg"
                    key_values = {index: values[key] for index, values in stats.items() if key in values}
                else:
                    attr = "bitrate_kbps" if algorithm == "bitrate" else "ssimu2_avg"
                    key_values = {
                        row.index: float(getattr(row, attr))
                        for row in rows
                        if getattr(row, attr) is not None
                    }
                if not key_values:
                    raise RuntimeError(f"no {algorithm} data available to sort by")
            tx = self._geometry_tx("sort_chunks")
            av1an_state.sort_chunks(
                pdir, algorithm, key_values=key_values, descending=desc_cb.isChecked(), tx=tx
            )
            tx.commit(message=f"sort queue {algorithm} ({pass_name})")
            self.notify("queue order updated (affects only not-done chunks on av1an resume)")
            self.refresh()

        self.show_dialog_form(
            f"Sort queue — {pass_name}",
            [
                make_label(
                    "Re-orders the av1an chunk queue. For mainpass, bitrate/ssimu2 come from fastpass stats "
                    "at matching frame ranges.",
                    color=COLOR_GREY_400,
                    wrap=True,
                ),
                widget_row(labeled("algorithm", algo_dd), desc_cb),
            ],
            apply,
            submit_label="Sort",
        )

    # ------------------------------------------------------------------ Logs tab

    def build_logs_tab(self, layout: QVBoxLayout) -> None:
        assert self.ctx is not None
        logs_dir = self.ctx.workdir / "00_logs"
        if not logs_dir.is_dir():
            layout.addWidget(make_label("no 00_logs directory", color=COLOR_GREY_500))
            layout.addStretch(1)
            return
        layout.addWidget(make_label(str(logs_dir), color=COLOR_GREY_500))
        listing = QListWidget()
        listing.setCursor(Qt.CursorShape.PointingHandCursor)
        listing.setStyleSheet(
            "QListWidget::item { border-bottom: 1px solid #3a3b40; padding: 6px; }"
            "QListWidget::item:hover { background-color: #2c2d31; }"
            f"QListWidget::item:selected {{ background-color: {COLOR_GREY_800}; }}"
        )
        try:
            files = sorted(logs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            files = []
        for path in files:
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            item = QListWidgetItem(f"{path.name}\n{fmt_size(stat.st_size)} | {fmt_timestamp(stat.st_mtime)}")
            item.setFont(mono_font())
            item.setToolTip(str(path))
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            listing.addItem(item)
        listing.itemClicked.connect(lambda item: self.show_log_tail(Path(item.data(Qt.ItemDataRole.UserRole))))
        layout.addWidget(listing, 1)

    def show_log_tail(self, path: Path, max_lines: int = 500) -> None:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            self.notify(f"failed to read {path}: {exc}", error=True)
            return
        lines = text.splitlines()
        if len(lines) > max_lines:
            lines = [f"... (showing last {max_lines} of {len(lines)} lines)"] + lines[-max_lines:]
        self.show_text_dialog(str(path), "\n".join(lines))

    # ------------------------------------------------------------------ Raw JSON tab

    def build_raw_tab(self, layout: QVBoxLayout) -> None:
        assert self.ctx is not None
        candidates: List[Path] = []
        for pass_name in ("fastpass", "mainpass"):
            pdir = self.ctx.workdir / "video" / pass_name
            candidates += [pdir / "chunks.json", pdir / "done.json"]
        candidates += [self.ctx.workdir / Path(rel) for rel, _ in manage_scenes.KNOWN_SCENE_FILES]
        candidates += [
            self.ctx.meta_dir / "runner_state.json",
            self.ctx.meta_dir / "source_info.json",
            self.ctx.meta_dir / "manage_state.json",
        ]
        existing = [path for path in candidates if path.is_file()]
        if not existing:
            layout.addWidget(make_label("no JSON files found", color=COLOR_GREY_500))
            layout.addStretch(1)
            return
        layout.addWidget(
            make_label(
                "Read-only JSON view. Manual raw editing is intentionally disabled in this version — "
                "use the safe forms on the other tabs.",
                color=COLOR_GREY_400,
                wrap=True,
            )
        )
        for path in existing:
            layout.addWidget(
                link_button(str(path.relative_to(self.ctx.workdir)), lambda p=path: self.show_json_file(p))
            )
        layout.addStretch(1)


def run_gui(refs: Sequence[WorkdirRef]) -> int:
    # must be set before the QApplication exists; ~115% feels right on desktop
    os.environ.setdefault("QT_SCALE_FACTOR", "1.15")
    app = QApplication.instance()
    if app is None:
        app = QApplication([sys.argv[0] if sys.argv else "manage"])
    apply_dark_theme(app)
    window = ManageWindow(refs)
    window.showMaximized()
    return app.exec()


__all__ = ["ManageWindow", "run_gui"]
