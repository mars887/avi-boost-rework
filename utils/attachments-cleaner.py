#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    from fontTools.ttLib import TTFont, TTCollection  # type: ignore
    FONTTOOLS_OK = True
except Exception:
    FONTTOOLS_OK = False


# -----------------------------
# Logging
# -----------------------------

def log(msg: str) -> None:
    print(msg)

def warn(msg: str) -> None:
    print(f"[W] {msg}", file=sys.stderr)

def die(msg: str, code: int = 1) -> int:
    print(f"[E] {msg}", file=sys.stderr)
    return code


# -----------------------------
# Normalization / Matching
# -----------------------------

_STYLE_TOKENS = {
    "regular", "bold", "italic", "oblique", "medium", "semibold", "demibold",
    "black", "heavy", "light", "thin", "ultralight", "extrabold", "extrablack",
    "condensed", "narrow", "expanded", "compressed",
    "roman",
}

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def normalize_font_name(s: str) -> str:
    """
    Soft-normalize names for matching between ASS and font metadata:
    - trim, collapse whitespace
    - remove leading '@' (Windows vertical fonts)
    - casefold
    - remove quotes
    """
    if s is None:
        return ""
    s = _norm_ws(str(s))
    s = s.strip("'\"")
    if s.startswith("@"):
        s = s[1:]
    s = s.casefold()
    return s

def strip_style_suffix(norm_name: str) -> str:
    """
    Remove trailing style tokens: "FontName Bold" -> "FontName"
    Operates on already-normalized string.
    """
    parts = norm_name.split()
    while parts and parts[-1] in _STYLE_TOKENS:
        parts.pop()
    return " ".join(parts)

def name_variants(raw: str) -> Set[str]:
    n = normalize_font_name(raw)
    out = {n}
    s = strip_style_suffix(n)
    if s:
        out.add(s)
    return {x for x in out if x}


# -----------------------------
# ASS parsing (Styles + \fn)
# -----------------------------

RE_SECTION = re.compile(r"^\s*\[(.+?)\]\s*$")
RE_FORMAT = re.compile(r"^\s*Format\s*:\s*(.+?)\s*$", re.IGNORECASE)
RE_STYLE = re.compile(r"^\s*Style\s*:\s*(.+?)\s*$", re.IGNORECASE)

# \fn<name> ends at next '\' or '}' or line end
RE_FN = re.compile(r"\\fn([^\\}\r\n]+)", re.IGNORECASE)

def read_text_best_effort(p: Path) -> str:
    # ASS typically UTF-8; keep robust.
    data = p.read_bytes()
    try:
        return data.decode("utf-8-sig", errors="replace")
    except Exception:
        return data.decode("utf-8", errors="replace")

def parse_ass_used_fonts(text: str) -> Set[str]:
    used: Set[str] = set()

    section = ""
    style_fields: List[str] = []
    font_idx: Optional[int] = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        msec = RE_SECTION.match(line)
        if msec:
            section = (msec.group(1) or "").strip().casefold()
            # reset on section change
            style_fields = []
            font_idx = None
            continue

        # Collect \fn overrides everywhere (not only Events)
        for m in RE_FN.finditer(raw):
            fn = _norm_ws(m.group(1))
            if fn:
                used |= name_variants(fn)

        # Parse styles only inside styles sections
        if section not in ("v4+ styles", "v4 styles"):
            continue

        mfmt = RE_FORMAT.match(raw)
        if mfmt:
            # Example: Name, Fontname, Fontsize, PrimaryColour, ...
            fields = [f.strip() for f in mfmt.group(1).split(",")]
            style_fields = fields
            # find Fontname column
            font_idx = None
            for i, f in enumerate(fields):
                if f.strip().casefold() == "fontname":
                    font_idx = i
                    break
            continue

        mst = RE_STYLE.match(raw)
        if mst and style_fields and font_idx is not None:
            payload = mst.group(1)
            # split into exactly len(style_fields) fields (ASS style lines are comma-separated)
            parts = [p.strip() for p in payload.split(",", maxsplit=len(style_fields) - 1)]
            if len(parts) > font_idx:
                fn = parts[font_idx]
                if fn:
                    used |= name_variants(fn)

    return used


# -----------------------------
# Font attachment inspection
# -----------------------------

FONT_EXTS = {".ttf", ".otf", ".ttc", ".otc", ".woff", ".woff2"}

def get_font_names_from_ttfont(tt: "TTFont") -> Set[str]:
    names: Set[str] = set()
    if "name" not in tt:
        return names
    table = tt["name"]
    for rec in table.names:
        # Prefer family / typographic family / full name
        if rec.nameID not in (1, 4, 16):
            continue
        try:
            s = rec.toUnicode()
        except Exception:
            try:
                s = str(rec.string, errors="replace")
            except Exception:
                continue
        s = _norm_ws(s)
        if s:
            names.add(s)
    return names

def get_font_name_variants(path: Path) -> Set[str]:
    """
    Returns normalized name variants for a font file.
    Includes:
    - family/full names from font tables
    - file stem as fallback
    """
    raw_names: Set[str] = set()
    raw_names.add(path.stem)

    if not FONTTOOLS_OK:
        return {v for n in raw_names for v in name_variants(n)}

    ext = path.suffix.casefold()
    try:
        if ext in (".ttc", ".otc"):
            coll = TTCollection(str(path))
            for tt in coll.fonts:
                raw_names |= get_font_names_from_ttfont(tt)
        else:
            tt = TTFont(str(path), lazy=True)
            raw_names |= get_font_names_from_ttfont(tt)
            tt.close()
    except Exception as ex:
        warn(f"Failed to parse font '{path.name}': {ex}")
        # fall back to stem only

    out: Set[str] = set()
    for n in raw_names:
        out |= name_variants(n)
    return {x for x in out if x}


# -----------------------------
# Main cleaning
# -----------------------------

@dataclass
class Removal:
    path: str
    names: List[str]
    reason: str

def scan_used_fonts(subs_dir: Path) -> Tuple[Set[str], Dict[str, List[str]]]:
    """
    Returns:
      - used fonts (normalized variants)
      - per-file debug mapping: file -> list of raw fonts found (best-effort)
    """
    used: Set[str] = set()
    debug: Dict[str, List[str]] = {}

    ass_files = sorted([p for p in subs_dir.glob("*") if p.is_file() and p.suffix.casefold() in (".ass", ".ssa")])
    for p in ass_files:
        text = read_text_best_effort(p)
        used_here = parse_ass_used_fonts(text)
        used |= used_here
        debug[p.name] = sorted(list(used_here))

    return used, debug

def clean_attachments(
    used_fonts: Set[str],
    attachments_dir: Path,
    dry_run: bool,
    keep_nonfonts: bool = True
) -> Tuple[List[Removal], int, int]:
    """
    Returns:
      removed list, kept_fonts_count, total_fonts_count
    """
    removed: List[Removal] = []
    font_files = [p for p in attachments_dir.iterdir() if p.is_file() and p.suffix.casefold() in FONT_EXTS]
    total_fonts = len(font_files)
    kept_fonts = 0

    for fp in sorted(font_files):
        variants = get_font_name_variants(fp)
        # Match if ANY variant is present in used fonts
        matched = any(v in used_fonts for v in variants)

        if matched:
            kept_fonts += 1
            log(f"[keep] {fp.name}")
            continue

        # Not matched -> remove
        log(f"[rm]   {fp.name}  (unused)")
        removed.append(Removal(
            path=str(fp),
            names=sorted(list(variants))[:20],
            reason="unused_font"
        ))
        if not dry_run:
            try:
                fp.unlink()
            except Exception as ex:
                warn(f"Failed to delete '{fp}': {ex}")

    # Non-font attachments are kept by default
    if not keep_nonfonts:
        # Not used in your current contract; left for future strict mode.
        pass

    return removed, kept_fonts, total_fonts

def main() -> int:
    ap = argparse.ArgumentParser(description="Remove unused font attachments based on ASS subtitles usage.")
    ap.add_argument("--subs", required=True, help="Path to WORKDIR\\sub")
    ap.add_argument("--attachments", required=True, help="Path to WORKDIR\\attachments")
    ap.add_argument("--dry-run", action="store_true", help="Do not delete, only log what would be removed.")
    ap.add_argument("--report", default="", help="Optional path to write JSON report (default: <attachments>/attachments_cleaner_report.json)")
    args = ap.parse_args()

    subs_dir = Path(args.subs)
    att_dir = Path(args.attachments)

    if not subs_dir.exists() or not subs_dir.is_dir():
        return die(f"Subs dir not found: {subs_dir}", 2)
    if not att_dir.exists() or not att_dir.is_dir():
        return die(f"Attachments dir not found: {att_dir}", 3)

    if not FONTTOOLS_OK:
        warn("fontTools is not available. Will not delete fonts (safe mode). Install: pip install fonttools")
        # Still produce a report/log with used fonts for debugging.

    used_fonts, debug_map = scan_used_fonts(subs_dir)

    log("=== attachments-cleaner ===")
    log(f"subs:        {subs_dir}")
    log(f"attachments: {att_dir}")
    log(f"dry_run:     {args.dry_run}")
    log(f"fontTools:   {'OK' if FONTTOOLS_OK else 'MISSING (safe mode)'}")
    log("")

    ass_count = len([p for p in subs_dir.iterdir() if p.is_file() and p.suffix.casefold() in (".ass", ".ssa")])
    log(f"ASS/SSA files scanned: {ass_count}")

    if ass_count == 0:
        log("[info] No ASS/SSA subtitles found -> keep attachments as-is.")
        # Still write report
        report_obj = {
            "subs": str(subs_dir),
            "attachments": str(att_dir),
            "dry_run": bool(args.dry_run),
            "fontTools": FONTTOOLS_OK,
            "used_fonts": [],
            "removed_fonts": [],
            "note": "no_ass_subtitles",
        }
        report_path = Path(args.report) if args.report else (att_dir / "attachments_cleaner_report.json")
        report_path.write_text(json.dumps(report_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    log(f"Used font tokens (normalized): {len(used_fonts)}")
    if len(used_fonts) <= 200:
        for f in sorted(used_fonts):
            log(f"  [use] {f}")
    else:
        log("  [use] (too many to print)")

    log("")

    removed: List[Removal] = []
    kept_fonts = 0
    total_fonts = 0

    if FONTTOOLS_OK:
        removed, kept_fonts, total_fonts = clean_attachments(
            used_fonts=used_fonts,
            attachments_dir=att_dir,
            dry_run=bool(args.dry_run),
            keep_nonfonts=True
        )
    else:
        # Safe mode: do not delete anything
        total_fonts = len([p for p in att_dir.iterdir() if p.is_file() and p.suffix.casefold() in FONT_EXTS])
        kept_fonts = total_fonts

    log("")
    log(f"Fonts total: {total_fonts}")
    log(f"Fonts kept:  {kept_fonts}")
    log(f"Fonts {'would be removed' if args.dry_run else 'removed'}: {len(removed)}")

    report_obj = {
        "subs": str(subs_dir),
        "attachments": str(att_dir),
        "dry_run": bool(args.dry_run),
        "fontTools": FONTTOOLS_OK,
        "ass_debug_used_fonts_by_file": debug_map,
        "used_fonts": sorted(list(used_fonts)),
        "removed_fonts": [r.__dict__ for r in removed],
        "summary": {
            "fonts_total": total_fonts,
            "fonts_kept": kept_fonts,
            "fonts_removed": len(removed),
        }
    }

    report_path = Path(args.report) if args.report else (att_dir / "attachments_cleaner_report.json")
    try:
        report_path.write_text(json.dumps(report_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"Report: {report_path}")
    except Exception as ex:
        warn(f"Failed to write report '{report_path}': {ex}")

    log("[ok]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
