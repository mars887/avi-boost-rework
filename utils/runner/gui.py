from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QEasingCurve, QEvent, QPropertyAnimation, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QCursor, QTextOption
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from utils.discord_config import discord_config_value
from utils.track_gui_shared import (
    APP_ACCENT,
    APP_ACCENT_SOFT,
    APP_BG,
    APP_BORDER,
    APP_CARD,
    APP_INPUT,
    APP_MUTED,
    APP_SUCCESS,
    APP_SURFACE,
    APP_SURFACE_ALT,
    APP_TEXT,
    APP_WARNING,
)
from utils.runner_state import is_cached_stage_message

from .api import RunnerLaunchConfig, RunnerRuntime
from .helpers import build_queue
from .integrations import attach_discord_integrations
from .logs import RunnerLogLine
from .stage_bank import StageBankConfig, load_stage_bank_config
from .terminal import TerminalScreen


class TerminalView(QTextEdit):
    def __init__(self, *, max_lines: int = 2000) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.WidgetWidth)
        self.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setAcceptRichText(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.screen = TerminalScreen(max_lines=max_lines)
        self.setObjectName("TerminalView")

    def feed(self, text: str) -> None:
        if not text:
            return
        self.screen.feed(str(text))
        self._render()

    def append_plain_line(self, text: str) -> None:
        self.feed(str(text or "") + "\n")

    def _render(self) -> None:
        scrollbar = self.verticalScrollBar()
        old_value = scrollbar.value()
        at_bottom = self.verticalScrollBar().value() >= self.verticalScrollBar().maximum() - 4
        body = "".join(self.screen.html_line_divs(APP_TEXT))
        self.setHtml(
            "<html><body "
            f"style=\"background:{APP_INPUT}; color:{APP_TEXT}; font-family:Consolas, monospace; "
            "font-size:10.5pt; white-space:pre-wrap; overflow-wrap:anywhere;\">"
            "<style>"
            ".term-line { padding-left: 4ch; text-indent: -4ch; white-space: pre-wrap; overflow-wrap: anywhere; }"
            "</style>"
            f"{body}</body></html>"
        )
        if at_bottom:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        else:
            self.verticalScrollBar().setValue(min(self.verticalScrollBar().maximum(), max(0, old_value)))


class ConsoleBlock(QGroupBox):
    def __init__(self, title: str, *, created_index: int) -> None:
        super().__init__(title)
        self.setObjectName("ConsoleBlock")
        self.plan_run_id = ""
        self.plan_name = "Session"
        self.stage = ""
        self.stage_order = 1_000_000
        self.status = "started"
        self.active = True
        self.finished_at = 0.0
        self.last_update_at = time.time()
        self.error_message = ""
        self.manual_stop = False
        self.created_index = created_index
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 10)
        self.view = TerminalView(max_lines=4000)
        self.view.setFixedHeight(210)
        layout.addWidget(self.view, 1)
        self.set_status("started")

    def append(self, line: RunnerLogLine) -> None:
        raw = line.raw_text if getattr(line, "raw_text", "") else line.text
        self.view.feed(raw)

    def update_identity(self, line: RunnerLogLine) -> None:
        self.plan_run_id = line.plan_run_id or self.plan_run_id
        self.stage = line.stage or self.stage or line.stream
        self.plan_name = Path(line.plan).stem if line.plan else self.plan_name
        self.last_update_at = line.timestamp or time.time()
        self._refresh_title()

    def set_status(self, status: str, *, message: str = "", manual_stop: bool = False) -> None:
        value = str(status or "started").lower()
        if value in ("pending", "queued"):
            value = "started"
        was_active = self.active
        self.status = value
        if message and value == "failed":
            self.error_message = str(message)
        if manual_stop:
            self.manual_stop = True
        self.active = self.manual_stop or value not in ("completed", "failed", "skipped")
        if was_active and not self.active and not self.finished_at:
            self.finished_at = time.time()
        self.setProperty("consoleStatus", value)
        self.style().unpolish(self)
        self.style().polish(self)
        self._refresh_title()

    def _refresh_title(self) -> None:
        suffix = f" | {self.error_message}" if self.error_message else ""
        self.setTitle(f"{self.plan_name} | {self.stage}{suffix}")


class RunnerMainWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.runtime: Optional[RunnerRuntime] = None
        self.join_thread: Optional[threading.Thread] = None
        self.event_queue: "queue.Queue[tuple[Dict[str, Any], Dict[str, Any]]]" = queue.Queue()
        self.log_queue: "queue.Queue[RunnerLogLine]" = queue.Queue()
        self.join_queue: "queue.Queue[int]" = queue.Queue()
        self.last_snapshot: Dict[str, Any] = {}
        self.base_stage_bank = load_stage_bank_config()
        self.console_blocks: Dict[str, ConsoleBlock] = {}
        self.console_separators: List[QWidget] = []
        self.console_block_counter = 0
        self.console_layout_dirty = False
        self.scroll_animations: Dict[int, QPropertyAnimation] = {}

        self.setWindowTitle("PBBatch Runner")
        self.resize(1480, 920)
        self.setMinimumSize(1120, 720)
        self._build_ui()
        self._apply_style()
        self._load_args()
        self._refresh_queue_preview()
        self._set_running_state(False)

        self.timer = QTimer(self)
        self.timer.setInterval(400)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start()

    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar(self))

        self.tabs = QTabWidget()
        self._build_tab_corner()
        layout.addWidget(self.tabs, 1)
        self._build_plans_tab()
        self._build_status_tab()
        self._build_console_tab()
        self._build_settings_tab()
        self.tabs.setCurrentIndex(0)

    def _build_tab_corner(self) -> None:
        corner = QWidget()
        corner_layout = QHBoxLayout(corner)
        corner_layout.setContentsMargins(0, 0, 0, 0)
        corner_layout.setSpacing(10)
        self.summary_label = QLabel("No runner session.")
        self.summary_label.setObjectName("SummaryLabel")
        self.summary_label.setMinimumWidth(420)
        self.clear_console_button = QPushButton("Clear console")
        self.console_autoscroll_check = QCheckBox("Auto-scroll")
        self.console_autoscroll_check.setChecked(True)
        corner_layout.addWidget(self.summary_label, 1)
        corner_layout.addWidget(self.clear_console_button)
        corner_layout.addWidget(self.console_autoscroll_check)
        self.tabs.setCornerWidget(corner, Qt.TopRightCorner)
        self.clear_console_button.clicked.connect(self._clear_console)

    def _build_plans_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        top = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause after current")
        self.resume_button = QPushButton("Resume")
        self.retry_button = QPushButton("Retry failed")
        self.rerun_button = QPushButton("Rerun current")
        self.exit_idle_button = QPushButton("Exit when idle")
        self.stop_button = QPushButton("Stop")
        top.addWidget(self.start_button)
        top.addSpacing(14)
        for button in (
            self.pause_button,
            self.resume_button,
            self.retry_button,
            self.rerun_button,
            self.exit_idle_button,
        ):
            top.addWidget(button)
        top.addStretch(1)
        top.addWidget(self.stop_button)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        left = QGroupBox("Plans")
        left_layout = QVBoxLayout(left)
        controls = QHBoxLayout()
        self.add_plan_button = QPushButton("Add .plan")
        self.add_folder_button = QPushButton("Add folder")
        self.remove_plan_button = QPushButton("Remove")
        self.move_up_button = QPushButton("Move up")
        self.move_down_button = QPushButton("Move down")
        self.clear_plan_button = QPushButton("Clear")
        for button in (
            self.add_plan_button,
            self.add_folder_button,
            self.remove_plan_button,
            self.move_up_button,
            self.move_down_button,
            self.clear_plan_button,
        ):
            controls.addWidget(button)
        controls.addStretch(1)
        left_layout.addLayout(controls)
        self.plans_table = self._make_table(["#", "Mode", "Name", "Source", "Plan"])
        self.plans_table.setSelectionMode(QTableWidget.ExtendedSelection)
        left_layout.addWidget(self.plans_table, 1)
        splitter.addWidget(left)

        right = QGroupBox("Runner configuration")
        right_layout = QFormLayout(right)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["full", "fastpass"])
        self.add_source_bitrate_check = QCheckBox("Add source bitrate")
        self.exit_when_idle_check = QCheckBox("Exit when idle")
        self.queue_count_label = QLabel("-")
        self.queue_count_label.setObjectName("MutedLabel")
        right_layout.addRow("FULL/FP only", self.mode_combo)
        right_layout.addRow("", self.add_source_bitrate_check)
        right_layout.addRow("", self.exit_when_idle_check)
        right_layout.addRow("Resolved queue", self.queue_count_label)
        splitter.addWidget(right)
        splitter.setSizes([980, 360])

        self.tabs.addTab(tab, "Plans")

        self.add_plan_button.clicked.connect(self._add_plans)
        self.add_folder_button.clicked.connect(self._add_folder)
        self.remove_plan_button.clicked.connect(self._remove_selected_plans)
        self.move_up_button.clicked.connect(lambda: self._move_selected_plan(-1))
        self.move_down_button.clicked.connect(lambda: self._move_selected_plan(1))
        self.clear_plan_button.clicked.connect(self._clear_plans)
        self.start_button.clicked.connect(self._start_runner)
        self.pause_button.clicked.connect(lambda: self._send_command("pause_after_current"))
        self.resume_button.clicked.connect(lambda: self._send_command("resume"))
        self.retry_button.clicked.connect(lambda: self._send_command("retry_failed"))
        self.rerun_button.clicked.connect(lambda: self._send_command("rerun_current_item"))
        self.exit_idle_button.clicked.connect(lambda: self._send_command("exit_when_idle"))
        self.stop_button.clicked.connect(self._stop_runner)
        self.mode_combo.currentTextChanged.connect(lambda _value: self._refresh_queue_preview())

    def _build_status_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        self.status_scroll = QScrollArea()
        self.status_scroll.setWidgetResizable(True)
        self.status_body = QWidget()
        self.status_layout = QVBoxLayout(self.status_body)
        self.status_layout.setContentsMargins(0, 0, 0, 0)
        self.status_layout.setSpacing(10)
        self.status_layout.addStretch(1)
        self.status_scroll.setWidget(self.status_body)
        layout.addWidget(self.status_scroll, 1)
        self.tabs.addTab(tab, "Status")

    def _build_console_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)

        self.console_scroll = QScrollArea()
        self.console_scroll.setWidgetResizable(True)
        self.console_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.console_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.console_body = QWidget()
        self.console_layout = QVBoxLayout(self.console_body)
        self.console_layout.setContentsMargins(0, 0, 0, 0)
        self.console_layout.setSpacing(8)
        self.console_layout.addStretch(1)
        self.console_scroll.setWidget(self.console_body)
        layout.addWidget(self.console_scroll, 1)
        self.tabs.addTab(tab, "Console")
        for widget in (tab, self.console_scroll, self.console_scroll.viewport(), self.console_body):
            widget.installEventFilter(self)

    def _build_settings_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)

        discord = QGroupBox("External integrations")
        discord_layout = QFormLayout(discord)
        self.discord_check = QCheckBox("Discord integration")
        self.discord_url_edit = QLineEdit()
        self.discord_secret_edit = QLineEdit()
        self.discord_secret_edit.setEchoMode(QLineEdit.Password)
        discord_layout.addRow("", self.discord_check)
        discord_layout.addRow("Discord service URL", self.discord_url_edit)
        discord_layout.addRow("Shared secret", self.discord_secret_edit)
        layout.addWidget(discord)

        bank = QGroupBox("Stage bank")
        bank_layout = QFormLayout(bank)
        self.capacity_spin = self._make_spin(1, 128, self.base_stage_bank.capacity)
        self.active_plans_spin = self._make_spin(1, 64, self.base_stage_bank.max_active_plans)
        self.running_stages_spin = self._make_spin(1, 128, self.base_stage_bank.max_running_stages)
        bank_layout.addRow("Bank capacity", self.capacity_spin)
        bank_layout.addRow("Active plans", self.active_plans_spin)
        bank_layout.addRow("Running stages", self.running_stages_spin)
        layout.addWidget(bank)

        layout.addStretch(1)
        self.tabs.addTab(tab, "Settings")

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background: {APP_BG};
                color: {APP_TEXT};
                font-family: Segoe UI;
                font-size: 10.5pt;
            }}
            QGroupBox {{
                border: 1px solid {APP_BORDER};
                border-radius: 6px;
                margin-top: 8px;
                padding: 10px;
                background: {APP_CARD};
                font-weight: 600;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {APP_ACCENT};
            }}
            QPushButton {{
                background: {APP_SURFACE_ALT};
                color: {APP_TEXT};
                border: 1px solid {APP_BORDER};
                border-radius: 5px;
                padding: 7px 11px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {APP_ACCENT_SOFT};
                border-color: {APP_ACCENT};
            }}
            QPushButton:disabled {{
                color: {APP_MUTED};
                background: {APP_SURFACE};
            }}
            QLineEdit, QComboBox, QSpinBox, QTableWidget, QTextEdit, QPlainTextEdit {{
                background: {APP_INPUT};
                color: {APP_TEXT};
                border: 1px solid {APP_BORDER};
                selection-background-color: {APP_ACCENT_SOFT};
            }}
            QHeaderView::section {{
                background: {APP_SURFACE_ALT};
                color: {APP_ACCENT};
                border: 0;
                padding: 6px;
                font-weight: 600;
            }}
            QTabWidget::pane {{
                border: 1px solid {APP_BORDER};
            }}
            QTabBar::tab {{
                background: {APP_SURFACE_ALT};
                color: {APP_MUTED};
                padding: 8px 16px;
                border: 1px solid {APP_BORDER};
            }}
            QTabBar::tab:selected {{
                background: {APP_CARD};
                color: {APP_TEXT};
            }}
            QScrollArea {{
                border: 0;
            }}
            QScrollBar {{
                width: 0px;
                height: 0px;
                background: transparent;
            }}
            QTextEdit#TerminalView {{
                border: 0;
                border-radius: 4px;
                padding: 8px;
            }}
            QGroupBox#ConsoleBlock {{
                border: 1px solid {APP_BORDER};
                border-radius: 6px;
            }}
            QGroupBox#ConsoleBlock[consoleStatus="started"] {{
                border-color: {APP_ACCENT};
            }}
            QGroupBox#ConsoleBlock[consoleStatus="completed"],
            QGroupBox#ConsoleBlock[consoleStatus="skipped"] {{
                border-color: {APP_SUCCESS};
            }}
            QGroupBox#ConsoleBlock[consoleStatus="failed"] {{
                border-color: {APP_WARNING};
            }}
            QLabel#MutedLabel {{
                color: {APP_MUTED};
            }}
            QLabel {{
                padding: 2px;
            }}
            QLabel#ConsoleSection {{
                color: {APP_TEXT};
                background: {APP_SURFACE_ALT};
                border: 1px solid {APP_BORDER};
                border-radius: 4px;
                padding: 7px 10px;
                font-weight: 700;
                qproperty-alignment: AlignCenter;
            }}
            QLabel#ConsolePlan {{
                color: {APP_ACCENT};
                padding: 7px 2px;
                font-weight: 700;
                qproperty-alignment: AlignCenter;
            }}
            QLabel#SummaryLabel {{
                color: {APP_TEXT};
                font-weight: 600;
                padding: 5px 8px;
            }}
            QProgressBar {{
                border: 1px solid {APP_BORDER};
                border-radius: 4px;
                background: {APP_INPUT};
                color: {APP_TEXT};
                text-align: center;
                min-height: 18px;
            }}
            QProgressBar::chunk {{
                background: {APP_ACCENT_SOFT};
            }}
            QStatusBar {{
                background: {APP_SURFACE};
                color: {APP_MUTED};
            }}
            """
        )

    @staticmethod
    def _make_spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    @staticmethod
    def _make_table(columns: List[str]) -> QTableWidget:
        table = QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setAlternatingRowColors(False)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def _load_args(self) -> None:
        mode = str(getattr(self.args, "mode", "") or "full")
        self.mode_combo.setCurrentText("fastpass" if mode == "fastpass" else "full")
        self.add_source_bitrate_check.setChecked(bool(getattr(self.args, "add_source_bitrate", False)))
        self.exit_when_idle_check.setChecked(bool(getattr(self.args, "exit_when_idle", False)))
        self.discord_check.setChecked(bool(getattr(self.args, "discord_enabled", True)))
        self.discord_url_edit.setText(
            str(
                getattr(self.args, "discord_service_url", "")
                or discord_config_value("PBBATCH_DISCORD_SERVICE_URL", "http://127.0.0.1:8794")
            )
        )
        self.discord_secret_edit.setText(str(getattr(self.args, "discord_shared_secret", "") or ""))
        for raw in list(getattr(self.args, "plans", []) or []):
            self._append_plan_path(raw)

    def _append_plan_path(self, raw_path: str) -> None:
        path = str(Path(raw_path).expanduser().resolve())
        for existing in self._plan_paths():
            if existing.lower() == path.lower():
                return
        row = self.plans_table.rowCount()
        self.plans_table.insertRow(row)
        self.plans_table.setItem(row, 4, QTableWidgetItem(path))
        self._refresh_queue_preview()

    def _plan_paths(self) -> List[str]:
        out: List[str] = []
        for row in range(self.plans_table.rowCount()):
            item = self.plans_table.item(row, 4)
            if item is not None and item.text().strip():
                out.append(item.text().strip())
        return out

    def _add_plans(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add plans", "", "Plan files (*.plan);;All files (*.*)")
        for path in paths:
            self._append_plan_path(path)

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add folder")
        if not folder:
            return
        for path in sorted(Path(folder).glob("*.plan"), key=lambda item: item.name.lower()):
            self._append_plan_path(str(path))

    def _remove_selected_plans(self) -> None:
        rows = sorted({item.row() for item in self.plans_table.selectedItems()}, reverse=True)
        for row in rows:
            self.plans_table.removeRow(row)
        self._refresh_queue_preview()

    def _move_selected_plan(self, delta: int) -> None:
        selected = self.plans_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        target = row + delta
        if target < 0 or target >= self.plans_table.rowCount():
            return
        paths = self._plan_paths()
        paths[row], paths[target] = paths[target], paths[row]
        self.plans_table.setRowCount(0)
        for path in paths:
            self._append_plan_path(path)
        self.plans_table.selectRow(target)

    def _clear_plans(self) -> None:
        self.plans_table.setRowCount(0)
        self._refresh_queue_preview()

    def _stage_bank_config(self) -> StageBankConfig:
        return StageBankConfig(
            capacity=int(self.capacity_spin.value()),
            max_active_plans=int(self.active_plans_spin.value()),
            max_running_stages=int(self.running_stages_spin.value()),
            stages=dict(self.base_stage_bank.stages),
        )

    def _refresh_queue_preview(self) -> None:
        if self.runtime is not None:
            return
        paths = self._plan_paths()
        self.start_button.setEnabled(bool(paths))
        if not paths:
            self.plans_table.setRowCount(0)
            self.queue_count_label.setText("0 item(s)")
            self.statusBar().showMessage("Add .plan files to prepare a runner session.")
            return
        try:
            items = build_queue(paths, self.mode_combo.currentText())
        except Exception as exc:
            self.queue_count_label.setText("preview failed")
            self.statusBar().showMessage(f"Queue preview failed: {exc}")
            self._fill_plan_table(paths, [])
            return
        self._fill_plan_table(paths, items)
        self.queue_count_label.setText(f"{len(items)} item(s)")
        self.statusBar().showMessage(f"Queue preview: {len(items)} item(s).")

    def _fill_plan_table(self, paths: List[str], items: List[Any]) -> None:
        selected_rows = {item.row() for item in self.plans_table.selectedItems()}
        item_by_plan = {str(item.plan_path).lower(): item for item in items}
        self.plans_table.setRowCount(len(paths))
        for row, path in enumerate(paths):
            resolved = item_by_plan.get(str(Path(path).resolve()).lower())
            values = [
                str(row + 1),
                self.mode_combo.currentText(),
                resolved.name if resolved is not None else "",
                str(resolved.source) if resolved is not None else "",
                path,
            ]
            for col, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                table_item.setToolTip(value)
                self.plans_table.setItem(row, col, table_item)
            if row in selected_rows:
                self.plans_table.selectRow(row)
        self.plans_table.resizeColumnsToContents()
        self.plans_table.horizontalHeader().setStretchLastSection(True)

    def _start_runner(self) -> None:
        if self.runtime is not None:
            return
        paths = self._plan_paths()
        if not paths:
            QMessageBox.warning(self, "Runner", "Add at least one .plan file first.")
            return
        config = RunnerLaunchConfig(
            plans=paths,
            mode=self.mode_combo.currentText(),
            events_jsonl=str(getattr(self.args, "events_jsonl", "") or ""),
            add_source_bitrate=self.add_source_bitrate_check.isChecked(),
            exit_when_idle=self.exit_when_idle_check.isChecked(),
            no_interactive=True,
            session_id=str(getattr(self.args, "session_id", "") or ""),
            stage_bank_config=self._stage_bank_config(),
        )
        try:
            runtime = RunnerRuntime(config)
            runtime.add_event_sink(lambda event, snapshot: self.event_queue.put((event, snapshot)))
            runtime.add_log_sink(lambda line: self.log_queue.put(line))
            attach_discord_integrations(
                runtime,
                service_url=self.discord_url_edit.text().strip(),
                shared_secret=self.discord_secret_edit.text(),
                enabled=self.discord_check.isChecked(),
                verbose=bool(getattr(self.args, "discord_verbose", False)),
                logger=lambda message: self.log_queue.put(
                    RunnerLogLine(
                        timestamp=time.time(),
                        session_id=runtime.session_id,
                        plan_run_id="",
                        source="",
                        plan="",
                        stage="Discord",
                        stream="stderr",
                        text=message,
                        raw_text=message + "\n",
                    )
                ),
            )
            runtime.start()
        except Exception as exc:
            QMessageBox.critical(self, "Runner start failed", str(exc))
            try:
                runtime.close()  # type: ignore[possibly-undefined]
            except Exception:
                pass
            return
        self.runtime = runtime
        self._set_running_state(True)
        self.last_snapshot = runtime.snapshot()
        self._refresh_from_snapshot(self.last_snapshot)
        self.join_thread = threading.Thread(target=self._join_runtime, name="runner-gui-join", daemon=True)
        self.join_thread.start()
        self.tabs.setCurrentIndex(1)
        self.statusBar().showMessage(f"Runner started: {runtime.session_id}")

    def _join_runtime(self) -> None:
        runtime = self.runtime
        if runtime is None:
            return
        code = runtime.join()
        self.join_queue.put(code)

    def _stop_runner(self) -> None:
        if self.runtime is None:
            return
        self.runtime.stop()
        self.statusBar().showMessage("Stop requested.")

    def _send_command(self, command: str) -> None:
        if self.runtime is None:
            return
        try:
            message = self.runtime.handle_command(command, source_dir=self._selected_source_dir())
        except Exception as exc:
            QMessageBox.warning(self, "Runner command failed", str(exc))
            return
        self.statusBar().showMessage(message)

    def _selected_source_dir(self) -> str:
        active = [dict(item) for item in list(self.last_snapshot.get("active") or [])]
        if active:
            source = str(active[0].get("source") or "")
            if source:
                return str(Path(source).parent.resolve())
        queued = [dict(item) for item in list(self.last_snapshot.get("queue") or [])]
        if queued:
            source = str(queued[0].get("source") or "")
            if source:
                return str(Path(source).parent.resolve())
        return ""

    def _on_timer(self) -> None:
        drained_events = False
        while True:
            try:
                event, snapshot = self.event_queue.get_nowait()
            except queue.Empty:
                break
            self._update_console_from_event(event)
            self.last_snapshot = snapshot
            drained_events = True
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log_line(line)
        while True:
            try:
                code = self.join_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_finished(code)
        if self.runtime is not None:
            try:
                self.last_snapshot = self.runtime.snapshot()
                drained_events = True
            except Exception:
                pass
        if drained_events:
            self._refresh_from_snapshot(self.last_snapshot)

    def _handle_finished(self, code: int) -> None:
        self.statusBar().showMessage(f"Runner finished with exit code {code}.")
        self.runtime = None
        self._set_running_state(False)
        self._refresh_queue_preview()

    def _set_running_state(self, running: bool) -> None:
        self.start_button.setEnabled((not running) and bool(self._plan_paths()))
        for widget in (
            self.add_plan_button,
            self.add_folder_button,
            self.remove_plan_button,
            self.move_up_button,
            self.move_down_button,
            self.clear_plan_button,
            self.mode_combo,
            self.add_source_bitrate_check,
            self.exit_when_idle_check,
            self.discord_check,
            self.discord_url_edit,
            self.discord_secret_edit,
            self.capacity_spin,
            self.active_plans_spin,
            self.running_stages_spin,
            self.plans_table,
        ):
            widget.setEnabled(not running)
        for widget in (
            self.pause_button,
            self.resume_button,
            self.retry_button,
            self.rerun_button,
            self.exit_idle_button,
            self.stop_button,
        ):
            widget.setEnabled(running)

    def _refresh_from_snapshot(self, snapshot: Dict[str, Any]) -> None:
        active = [dict(item) for item in list(snapshot.get("active") or [])]
        queued = [dict(item) for item in list(snapshot.get("queue") or [])]
        completed = [dict(item) for item in list(snapshot.get("completed") or [])]
        failed = [dict(item) for item in list(snapshot.get("failed") or [])]
        counts = dict(snapshot.get("counts") or {})
        self.summary_label.setText(
            f"state={snapshot.get('state', '-')} | "
            f"active={counts.get('active', len(active))} | queued={counts.get('queued', len(queued))} | "
            f"completed={counts.get('completed', len(completed))} | failed={counts.get('failed', len(failed))}"
        )
        self._rebuild_status_blocks(active, queued, completed, failed)
        self._sync_console_from_snapshot(active, completed, failed)

    def _rebuild_status_blocks(
        self,
        active: List[Dict[str, Any]],
        queued: List[Dict[str, Any]],
        completed: List[Dict[str, Any]],
        failed: List[Dict[str, Any]],
    ) -> None:
        while self.status_layout.count() > 1:
            item = self.status_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not active and not queued and not completed and not failed:
            label = QLabel("No plans yet.")
            label.setObjectName("MutedLabel")
            self.status_layout.insertWidget(0, label)
            return
        for plan in active:
            self.status_layout.insertWidget(self.status_layout.count() - 1, self._build_plan_status_block(plan, active=True))
        for plan in failed[-10:]:
            self.status_layout.insertWidget(self.status_layout.count() - 1, self._build_finished_block(plan, failed=True))
        for plan in completed[-10:]:
            self.status_layout.insertWidget(self.status_layout.count() - 1, self._build_finished_block(plan, failed=False))
        if queued:
            queue_block = QGroupBox(f"Queued ({len(queued)})")
            q_layout = QVBoxLayout(queue_block)
            q_layout.setContentsMargins(12, 16, 12, 12)
            q_layout.setSpacing(8)
            table = self._make_table(["Name", "Mode", "Source"])
            self._set_table_rows(table, [[item.get("name", ""), item.get("mode", ""), item.get("source", "")] for item in queued])
            table.setMinimumHeight(min(360, max(150, 34 * len(queued) + 54)))
            queue_block.setMinimumHeight(min(420, max(190, 34 * len(queued) + 92)))
            q_layout.addWidget(table)
            self.status_layout.insertWidget(self.status_layout.count() - 1, queue_block)

    def _build_plan_status_block(self, plan: Dict[str, Any], *, active: bool) -> QWidget:
        title = f"{plan.get('name', '-')}"
        block = QGroupBox(title)
        layout = QVBoxLayout(block)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(8)
        header = QLabel(
            f"{plan.get('mode', '-')} | elapsed {self._fmt_seconds(plan.get('elapsed_seconds'))} | "
            f"{plan.get('source', '-')}"
        )
        header.setObjectName("MutedLabel")
        header.setWordWrap(True)
        layout.addWidget(header)
        stages = [dict(item) for item in list(plan.get("stages") or [])]
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(4, 1)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(7)
        for row, stage in enumerate(stages):
            name = QLabel(str(stage.get("name") or ""))
            name.setMinimumWidth(130)
            status = str(stage.get("status") or "pending")
            status_label = QLabel(status)
            status_label.setStyleSheet(f"color: {self._status_color(status)}; font-weight: 600;")
            status_label.setMinimumWidth(78)
            progress = QProgressBar()
            progress.setRange(0, 1000)
            value = self._progress_value(stage)
            progress.setValue(int(value * 10) if value >= 0 else 0)
            progress.setFormat(f"{value:.1f}%" if value >= 0 else "")
            details = QLabel(self._stage_detail_text(stage))
            details.setObjectName("MutedLabel")
            details.setWordWrap(True)
            grid.addWidget(name, row, 0)
            grid.addWidget(progress, row, 1)
            grid.addWidget(status_label, row, 2)
            grid.addWidget(details, row, 3, 1, 2)
        layout.addLayout(grid)
        return block

    def _build_finished_block(self, plan: Dict[str, Any], *, failed: bool) -> QWidget:
        status = "Failed" if failed else "Completed"
        block = QGroupBox(f"{status}: {plan.get('name', '-')}")
        layout = QVBoxLayout(block)
        layout.setContentsMargins(12, 16, 12, 12)
        color = APP_WARNING if failed else APP_SUCCESS
        label = QLabel(
            f"{plan.get('stage', '') or plan.get('status', '')} | {self._fmt_seconds(plan.get('elapsed_seconds'))} | "
            f"{plan.get('message', '') or plan.get('output', '')}"
        )
        label.setStyleSheet(f"color: {color};")
        label.setWordWrap(True)
        layout.addWidget(label)
        return block

    def _append_log_line(self, line: RunnerLogLine) -> None:
        key = f"{line.plan_run_id or 'session'}:{line.stage or line.stream}"
        page_bar = self.console_scroll.verticalScrollBar()
        page_at_bottom = page_bar.value() >= page_bar.maximum() - 4
        page_value = page_bar.value()
        block = self.console_blocks.get(key)
        if block is None:
            title = self._console_title(line)
            block = ConsoleBlock(title, created_index=self.console_block_counter)
            self.console_block_counter += 1
            block.installEventFilter(self)
            block.view.installEventFilter(self)
            block.view.viewport().installEventFilter(self)
            self.console_blocks[key] = block
            self.console_layout_dirty = True
        block.update_identity(line)
        block.append(line)
        if self.console_layout_dirty:
            self._rebuild_console_layout()
        if self.console_autoscroll_check.isChecked() and page_at_bottom:
            self.console_scroll.verticalScrollBar().setValue(self.console_scroll.verticalScrollBar().maximum())
        else:
            self.console_scroll.verticalScrollBar().setValue(min(self.console_scroll.verticalScrollBar().maximum(), page_value))

    def _console_title(self, line: RunnerLogLine) -> str:
        plan_name = Path(line.plan).stem if line.plan else "Session"
        return f"{plan_name} | {line.stage or line.stream}"

    def _update_console_from_event(self, event: Dict[str, Any]) -> None:
        plan_run_id = str(event.get("plan_run_id") or "")
        stage = str(event.get("stage") or "")
        if not plan_run_id or not stage:
            return
        block = self.console_blocks.get(f"{plan_run_id}:{stage}")
        if block is None:
            return
        status = str(event.get("status") or "")
        if status:
            old_active = block.active
            message = str(event.get("message") or "")
            manual_stop = "stop_requested" in message.lower()
            block.set_status(status, message=message, manual_stop=manual_stop)
            if status.lower() in ("completed", "failed", "skipped") and not block.active:
                try:
                    block.finished_at = max(block.finished_at, float(event.get("timestamp") or time.time()))
                except Exception:
                    block.finished_at = block.finished_at or time.time()
            self.console_layout_dirty = self.console_layout_dirty or old_active != block.active

    def _sync_console_from_snapshot(
        self,
        active: List[Dict[str, Any]],
        completed: List[Dict[str, Any]],
        failed: List[Dict[str, Any]],
    ) -> None:
        changed = False
        for plan in active:
            plan_run_id = str(plan.get("plan_run_id") or "")
            if not plan_run_id:
                continue
            plan_name = str(plan.get("name") or Path(str(plan.get("plan") or "")).stem or "Plan")
            stages = [dict(item) for item in list(plan.get("stages") or [])]
            for index, stage in enumerate(stages):
                stage_name = str(stage.get("name") or "")
                block = self.console_blocks.get(f"{plan_run_id}:{stage_name}")
                if block is None:
                    continue
                if block.plan_name != plan_name or block.stage_order != index:
                    block.plan_name = plan_name
                    block.stage_order = index
                    block._refresh_title()
                    changed = True
                old_active = block.active
                message = str(stage.get("message") or "")
                block.set_status(
                    str(stage.get("status") or "started"),
                    message=message,
                    manual_stop="stop_requested" in message.lower(),
                )
                changed = changed or old_active != block.active

        finished: Dict[str, Dict[str, Any]] = {}
        for plan in completed:
            if plan.get("plan_run_id"):
                finished[str(plan.get("plan_run_id"))] = {**plan, "_final_status": "completed"}
        for plan in failed:
            if plan.get("plan_run_id"):
                finished[str(plan.get("plan_run_id"))] = {**plan, "_final_status": "failed"}
        for block in self.console_blocks.values():
            info = finished.get(block.plan_run_id)
            if not info:
                continue
            old_active = block.active
            final_status = str(info.get("_final_status") or "")
            failed_stage = str(info.get("stage") or "")
            message = str(info.get("message") or "")
            manual_stop = "stop_requested" in message.lower()
            if final_status == "failed" and (not failed_stage or failed_stage == block.stage):
                block.set_status("failed", message=message, manual_stop=manual_stop)
            elif block.status in ("started", "pending", "queued"):
                block.set_status("completed", manual_stop=manual_stop)
            if not block.active:
                try:
                    block.finished_at = max(block.finished_at, float(info.get("ended_at") or time.time()))
                except Exception:
                    block.finished_at = block.finished_at or time.time()
            block.plan_name = str(info.get("name") or block.plan_name)
            block._refresh_title()
            changed = changed or old_active != block.active
        if changed or self.console_layout_dirty:
            self._rebuild_console_layout()

    def _console_sort_key(self, block: ConsoleBlock) -> tuple[str, str, int, int]:
        return (
            block.plan_name.lower(),
            block.plan_run_id,
            int(block.stage_order),
            int(block.created_index),
        )

    def _make_console_separator(self, text: str, *, kind: str) -> QLabel:
        label = QLabel(f"---------- {text} ----------" if kind == "plan" else text)
        label.setObjectName("ConsoleSection" if kind == "section" else "ConsolePlan")
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(30 if kind == "section" else 28)
        self.console_separators.append(label)
        return label

    def _rebuild_console_layout(self) -> None:
        page_bar = self.console_scroll.verticalScrollBar()
        page_value = page_bar.value()
        page_at_bottom = page_bar.value() >= page_bar.maximum() - 4
        while self.console_layout.count():
            self.console_layout.takeAt(0)
        for separator in self.console_separators:
            separator.deleteLater()
        self.console_separators.clear()

        def plan_key(block: ConsoleBlock) -> str:
            return block.plan_name.lower()

        def add_blocks(title: str, blocks: List[ConsoleBlock], *, presorted: bool = False) -> None:
            if not blocks:
                return
            self.console_layout.addWidget(self._make_console_separator(title, kind="section"))
            current_plan = ""
            ordered = list(blocks) if presorted else sorted(blocks, key=self._console_sort_key)
            for block in ordered:
                if block.plan_name != current_plan:
                    current_plan = block.plan_name
                    self.console_layout.addWidget(self._make_console_separator(current_plan, kind="plan"))
                self.console_layout.addWidget(block)

        blocks = list(self.console_blocks.values())
        active_blocks = sorted([block for block in blocks if block.active], key=self._console_sort_key)
        completed_blocks = [block for block in blocks if not block.active]
        active_plan_order: Dict[str, int] = {}
        for index, block in enumerate(active_blocks):
            active_plan_order.setdefault(plan_key(block), index)

        completed_with_active = sorted(
            [block for block in completed_blocks if plan_key(block) in active_plan_order],
            key=lambda block: (
                active_plan_order.get(plan_key(block), 1_000_000),
                block.stage_order,
                block.finished_at or block.last_update_at,
            ),
        )
        rest_by_plan: Dict[str, List[ConsoleBlock]] = {}
        for block in completed_blocks:
            if plan_key(block) in active_plan_order:
                continue
            rest_by_plan.setdefault(plan_key(block), []).append(block)
        recent_groups = sorted(
            rest_by_plan.values(),
            key=lambda group: max((block.finished_at or block.last_update_at or 0.0) for block in group),
            reverse=True,
        )
        completed_recent: List[ConsoleBlock] = []
        for group in recent_groups:
            completed_recent.extend(
                sorted(group, key=lambda block: (block.finished_at or block.last_update_at or 0.0), reverse=True)
            )

        add_blocks("Active processes", active_blocks, presorted=True)
        add_blocks("Completed processes", [*completed_with_active, *completed_recent], presorted=True)
        self.console_layout.addStretch(1)
        self.console_layout_dirty = False
        if self.console_autoscroll_check.isChecked() and page_at_bottom:
            self.console_scroll.verticalScrollBar().setValue(self.console_scroll.verticalScrollBar().maximum())
        else:
            self.console_scroll.verticalScrollBar().setValue(min(self.console_scroll.verticalScrollBar().maximum(), page_value))

    def _clear_console(self) -> None:
        for block in list(self.console_blocks.values()):
            block.deleteLater()
        self.console_blocks.clear()
        for separator in self.console_separators:
            separator.deleteLater()
        self.console_separators.clear()
        while self.console_layout.count():
            item = self.console_layout.takeAt(0)
            widget = item.widget()
            if widget is not None and widget not in self.console_blocks.values():
                widget.deleteLater()
        self.console_layout.addStretch(1)
        self.console_layout_dirty = False

    def _set_table_rows(self, table: QTableWidget, rows: List[List[Any]]) -> None:
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for col_index, value in enumerate(row):
                item = QTableWidgetItem(str(value if value is not None else ""))
                item.setToolTip(item.text())
                table.setItem(row_index, col_index, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)

    @staticmethod
    def _fmt_seconds(value: Any) -> str:
        try:
            seconds = max(0.0, float(value or 0.0))
        except Exception:
            return "-"
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, sec = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{sec:02d}"
        return f"{minutes:d}:{sec:02d}"

    def _stage_detail_text(self, stage: Dict[str, Any]) -> str:
        elapsed = self._fmt_seconds(stage.get("elapsed_seconds"))
        message = str(stage.get("message") or "").strip()
        details = [elapsed]
        if is_cached_stage_message(message):
            details.append("cached")
        elif message:
            details.append(message)
        return " | ".join(item for item in details if item)

    @staticmethod
    def _progress_value(stage: Dict[str, Any]) -> float:
        try:
            value = float(stage.get("progress"))
        except Exception:
            return -1.0
        if value < 0:
            return -1.0
        return max(0.0, min(100.0, value))

    @staticmethod
    def _status_color(status: str) -> str:
        value = str(status or "").lower()
        if value == "completed":
            return APP_SUCCESS
        if value == "started":
            return APP_ACCENT
        if value == "failed":
            return APP_WARNING
        if value == "skipped":
            return APP_MUTED
        return APP_TEXT

    def eventFilter(self, obj: Any, event: Any) -> bool:
        if event.type() == QEvent.Wheel and hasattr(self, "console_scroll"):
            pos = self.console_scroll.viewport().mapFromGlobal(QCursor.pos())
            if self.console_scroll.viewport().rect().contains(pos):
                delta = event.pixelDelta().y() or event.angleDelta().y()
                if not delta:
                    return True
                right_zone = pos.x() >= int(self.console_scroll.viewport().width() * 0.75)
                if right_zone:
                    self._smooth_scroll_hidden_bar(self.console_scroll.verticalScrollBar(), delta)
                    return True
                block = self._console_block_at_cursor()
                if block is not None and block.view.verticalScrollBar().maximum() > 0:
                    self._smooth_scroll_hidden_bar(block.view.verticalScrollBar(), delta)
                    return True
                self._smooth_scroll_hidden_bar(self.console_scroll.verticalScrollBar(), delta)
                return True
        return super().eventFilter(obj, event)

    def _smooth_scroll_hidden_bar(self, scrollbar: Any, delta: int) -> None:
        target = max(scrollbar.minimum(), min(scrollbar.maximum(), scrollbar.value() - int(delta)))
        key = id(scrollbar)
        animation = self.scroll_animations.get(key)
        if animation is None:
            animation = QPropertyAnimation(scrollbar, b"value", self)
            self.scroll_animations[key] = animation
        animation.stop()
        animation.setDuration(150)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.setStartValue(scrollbar.value())
        animation.setEndValue(target)
        animation.start()

    def _console_block_at_cursor(self) -> Optional[ConsoleBlock]:
        widget = QApplication.widgetAt(QCursor.pos())
        while widget is not None:
            if isinstance(widget, ConsoleBlock):
                return widget
            widget = widget.parentWidget()
        return None

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.runtime is not None and not self.runtime.is_finished():
            self.runtime.stop()
        event.accept()


def run_runner_gui(args: argparse.Namespace) -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = RunnerMainWindow(args)
    window.show()
    return int(app.exec())
