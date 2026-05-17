from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


CSI_RE = re.compile(r"\x1b\[([0-9;?]*)([ -/]*)([@-~])")
OSC_RE = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)", re.DOTALL)
VISIBLE_ESC_RE = re.compile(r"(?:\\x1b|\\u001b|\\033)", re.IGNORECASE)
ANSI_STRIP_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
WINDOWS_PROGRESS_GLYPH_TRANSLATION = str.maketrans(
    {
        "▁": "▏",
        "▂": "▎",
        "▃": "▍",
        "▄": "▌",
        "▅": "▋",
        "▆": "▊",
        "▇": "▉",
    }
)

ANSI_HTML_COLORS = {
    30: "#000000",
    31: "#ff5f5f",
    32: "#73ffbe",
    33: "#ffc66d",
    34: "#67a7ff",
    35: "#ff5fd2",
    36: "#35f2ff",
    37: "#f0fbff",
    90: "#89a9bf",
    91: "#ff8f8f",
    92: "#9dffd0",
    93: "#ffdc91",
    94: "#8ebcff",
    95: "#ff8add",
    96: "#8ff7ff",
    97: "#ffffff",
}


@dataclass(frozen=True)
class TerminalStyle:
    fg: Optional[int] = None
    bold: bool = False

    def sgr(self) -> str:
        codes: List[str] = []
        if self.bold:
            codes.append("1")
        if self.fg is not None:
            codes.append(str(self.fg))
        return f"\x1b[{';'.join(codes)}m" if codes else "\x1b[0m"

    def html_style(self, default_color: str) -> str:
        color = ANSI_HTML_COLORS.get(self.fg or 0, default_color)
        style = f"color:{color};"
        if self.bold:
            style += "font-weight:700;"
        return style


@dataclass(frozen=True)
class TerminalCell:
    char: str
    style: TerminalStyle


def decode_visible_ansi_escapes(text: str) -> str:
    return VISIBLE_ESC_RE.sub("\x1b", str(text or ""))


def strip_ansi(text: str) -> str:
    decoded = decode_visible_ansi_escapes(text)
    decoded = OSC_RE.sub("", decoded)
    return ANSI_STRIP_RE.sub("", decoded)


def normalize_terminal_glyphs(text: str) -> str:
    return str(text or "").translate(WINDOWS_PROGRESS_GLYPH_TRANSLATION)


def has_terminal_repaint(text: str) -> bool:
    decoded = decode_visible_ansi_escapes(text)
    if "\r" in decoded:
        return True
    for match in CSI_RE.finditer(decoded):
        if match.group(3) in "ABCDEFGHJKfhl":
            return True
    return False


class TerminalScreen:
    def __init__(self, *, max_lines: int = 2000) -> None:
        self.max_lines = max(20, int(max_lines))
        self.lines: List[List[TerminalCell]] = [[]]
        self.row = 0
        self.col = 0
        self.saved_cursor: Tuple[int, int] = (0, 0)
        self.style = TerminalStyle()

    def feed(self, text: str) -> None:
        raw = normalize_terminal_glyphs(OSC_RE.sub("", decode_visible_ansi_escapes(text)))
        if not raw:
            return
        idx = 0
        while idx < len(raw):
            match = CSI_RE.search(raw, idx)
            end = match.start() if match else len(raw)
            self._feed_plain(raw[idx:end])
            if not match:
                break
            self._apply_csi(match.group(1), match.group(3))
            idx = match.end()
        self._trim()

    def plain_lines(self) -> List[str]:
        return ["".join(cell.char for cell in line).rstrip() for line in self.lines]

    def latest_nonempty_plain(self) -> str:
        for line in reversed(self.plain_lines()):
            if line.strip():
                return line
        return ""

    def latest_nonempty_ansi(self) -> str:
        for line in reversed(self.lines):
            plain = "".join(cell.char for cell in line).rstrip()
            if not plain.strip():
                continue
            return self._cells_to_ansi(line).rstrip()
        return ""

    def current_nonempty_plain(self) -> str:
        for row in range(min(self.row, len(self.lines) - 1), -1, -1):
            line = "".join(cell.char for cell in self.lines[row]).rstrip()
            if line.strip():
                return line
        return self.latest_nonempty_plain()

    def current_nonempty_ansi(self) -> str:
        for row in range(min(self.row, len(self.lines) - 1), -1, -1):
            plain = "".join(cell.char for cell in self.lines[row]).rstrip()
            if plain.strip():
                return self._cells_to_ansi(self.lines[row]).rstrip()
        return self.latest_nonempty_ansi()

    def html_lines(self, default_color: str) -> List[str]:
        out: List[str] = []
        for line in self.lines:
            out.append(self._cells_to_html(line, default_color) or "&nbsp;")
        return out

    def html_line_divs(self, default_color: str) -> List[str]:
        return [f'<div class="term-line">{line}</div>' for line in self.html_lines(default_color)]

    def _feed_plain(self, text: str) -> None:
        for ch in text:
            if ch == "\r":
                self.col = 0
            elif ch == "\n":
                self.row += 1
                self.col = 0
                self._ensure_row()
            elif ch == "\b":
                self.col = max(0, self.col - 1)
            elif ch == "\t":
                for _ in range(4 - (self.col % 4)):
                    self._put_char(" ")
            elif ch >= " ":
                self._put_char(ch)

    def _put_char(self, ch: str) -> None:
        self._ensure_row()
        line = self.lines[self.row]
        while len(line) < self.col:
            line.append(TerminalCell(" ", TerminalStyle()))
        cell = TerminalCell(ch, self.style)
        if self.col < len(line):
            line[self.col] = cell
        else:
            line.append(cell)
        self.col += 1

    def _ensure_row(self) -> None:
        while self.row >= len(self.lines):
            self.lines.append([])
        self._trim()

    def _trim(self) -> None:
        if len(self.lines) <= self.max_lines:
            return
        drop = len(self.lines) - self.max_lines
        del self.lines[:drop]
        self.row = max(0, self.row - drop)
        saved_row, saved_col = self.saved_cursor
        self.saved_cursor = (max(0, saved_row - drop), saved_col)

    def _apply_csi(self, payload: str, final: str) -> None:
        if final == "m":
            self._apply_sgr(payload)
            return
        values = self._int_params(payload)
        first = values[0] if values else 1
        if final == "A":
            self.row = max(0, self.row - max(1, first))
        elif final == "B":
            self.row += max(1, first)
            self._ensure_row()
        elif final == "C":
            self.col += max(1, first)
        elif final == "D":
            self.col = max(0, self.col - max(1, first))
        elif final == "E":
            self.row += max(1, first)
            self.col = 0
            self._ensure_row()
        elif final == "F":
            self.row = max(0, self.row - max(1, first))
            self.col = 0
        elif final == "G":
            self.col = max(0, first - 1)
        elif final in ("H", "f"):
            self.row = max(0, (values[0] if len(values) >= 1 else 1) - 1)
            self.col = max(0, (values[1] if len(values) >= 2 else 1) - 1)
            self._ensure_row()
        elif final == "J":
            self._erase_display(first)
        elif final == "K":
            self._erase_line(first)
        elif final == "s":
            self.saved_cursor = (self.row, self.col)
        elif final == "u":
            self.row, self.col = self.saved_cursor
            self._ensure_row()

    def _apply_sgr(self, payload: str) -> None:
        values = self._int_params(payload, zero_default=True)
        if not values:
            values = [0]
        idx = 0
        fg = self.style.fg
        bold = self.style.bold
        while idx < len(values):
            code = values[idx]
            if code == 0:
                fg = None
                bold = False
            elif code == 1:
                bold = True
            elif code == 22:
                bold = False
            elif code in ANSI_HTML_COLORS:
                fg = code
            elif code == 39:
                fg = None
            elif code == 38 and idx + 2 < len(values) and values[idx + 1] == 5:
                # Keep 256-color output readable without trying to map every code.
                fg = None
                idx += 2
            idx += 1
        self.style = TerminalStyle(fg=fg, bold=bold)

    def _erase_line(self, mode: int) -> None:
        self._ensure_row()
        line = self.lines[self.row]
        if mode == 1:
            for index in range(min(self.col + 1, len(line))):
                line[index] = TerminalCell(" ", TerminalStyle())
        elif mode == 2:
            line.clear()
            self.col = 0
        else:
            del line[self.col :]

    def _erase_display(self, mode: int) -> None:
        self._ensure_row()
        if mode in (2, 3):
            self.lines = [[]]
            self.row = 0
            self.col = 0
        elif mode == 1:
            del self.lines[: self.row]
            self.row = 0
            self._erase_line(1)
        else:
            self._erase_line(0)
            del self.lines[self.row + 1 :]

    @staticmethod
    def _int_params(payload: str, *, zero_default: bool = False) -> List[int]:
        cleaned = payload.replace("?", "")
        if cleaned == "":
            return [0] if zero_default else []
        values: List[int] = []
        for part in cleaned.split(";"):
            if part == "":
                values.append(0 if zero_default else 1)
                continue
            try:
                values.append(int(part))
            except ValueError:
                values.append(0 if zero_default else 1)
        return values

    @staticmethod
    def _cells_to_html(cells: List[TerminalCell], default_color: str) -> str:
        if not cells:
            return ""
        parts: List[str] = []
        current_style: Optional[TerminalStyle] = None
        chunk: List[str] = []
        for cell in cells:
            if current_style is not None and cell.style != current_style:
                parts.append(TerminalScreen._html_span("".join(chunk), current_style, default_color))
                chunk.clear()
            current_style = cell.style
            chunk.append(cell.char)
        if chunk and current_style is not None:
            parts.append(TerminalScreen._html_span("".join(chunk), current_style, default_color))
        return "".join(parts)

    @staticmethod
    def _html_span(text: str, style: TerminalStyle, default_color: str) -> str:
        escaped = html.escape(text).replace(" ", "&nbsp;")
        return f'<span style="{style.html_style(default_color)}">{escaped}</span>'

    @staticmethod
    def _cells_to_ansi(cells: List[TerminalCell]) -> str:
        if not cells:
            return ""
        parts: List[str] = []
        current_style: Optional[TerminalStyle] = None
        for cell in cells:
            if cell.style != current_style:
                parts.append(cell.style.sgr())
                current_style = cell.style
            parts.append(cell.char)
        if current_style != TerminalStyle():
            parts.append("\x1b[0m")
        return "".join(parts)
