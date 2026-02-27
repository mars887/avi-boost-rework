#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract all Matroska (MKV) attachments to a directory using mkvextract.

Usage examples:
  python extract_mkv_attachments.py "input.mkv"
  python extract_mkv_attachments.py "input.mkv" -o "out_attachments"
  python extract_mkv_attachments.py "input.mkv" --mkvextract "C:\\Program Files\\MKVToolNix\\mkvextract.exe"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")


def sanitize_filename(name: str) -> str:
    # Keep it Windows-safe.
    name = name.strip().replace("\u0000", "")
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
    # Avoid trailing dots/spaces on Windows
    name = name.rstrip(" .")
    return name or "attachment"


def find_tool(explicit: str | None, exe_name: str) -> str:
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return str(p)
        raise FileNotFoundError(f"Tool not found: {explicit}")

    found = shutil.which(exe_name)
    if found:
        return found

    # Common Windows default
    if os.name == "nt":
        candidates = [
            Path(r"C:\Program Files\MKVToolNix") / f"{exe_name}.exe",
            Path(r"C:\Program Files (x86)\MKVToolNix") / f"{exe_name}.exe",
        ]
        for c in candidates:
            if c.exists():
                return str(c)

    raise FileNotFoundError(
        f"'{exe_name}' not found in PATH. Install MKVToolNix or pass --{exe_name} with full path."
    )


def load_attachments(mkvmerge: str, mkv_path: Path) -> list[dict]:
    # mkvmerge -J outputs JSON with attachments metadata
    cp = run([mkvmerge, "-J", str(mkv_path)])
    if cp.returncode != 0:
        raise RuntimeError(f"mkvmerge failed:\n{cp.stderr}")

    data = json.loads(cp.stdout)
    attachments = data.get("attachments") or []
    # Normalize fields we care about
    out: list[dict] = []
    for a in attachments:
        # Expected keys: id, file_name, content_type, size, ...
        out.append(
            {
                "id": a.get("id"),
                "file_name": a.get("file_name") or f"attachment_{a.get('id')}",
                "content_type": a.get("content_type"),
                "size": a.get("size"),
                "description": a.get("description"),
            }
        )
    return out


def unique_path(base_dir: Path, filename: str) -> Path:
    p = base_dir / filename
    if not p.exists():
        return p
    stem = p.stem
    suffix = p.suffix
    i = 1
    while True:
        cand = base_dir / f"{stem}.{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract all attachments from an MKV file (Matroska attachments).")
    ap.add_argument("mkv", type=Path, help="Input .mkv file")
    ap.add_argument("-o", "--outdir", type=Path, default=None, help="Output directory (default: <mkvname>_attachments)")
    ap.add_argument("--mkvextract", default=None, help="Path to mkvextract(.exe)")
    ap.add_argument("--mkvmerge", default=None, help="Path to mkvmerge(.exe)")
    ap.add_argument("--dry-run", action="store_true", help="Only list attachments, do not extract")
    args = ap.parse_args()

    mkv_path: Path = args.mkv
    if not mkv_path.is_file():
        print(f"Input file not found: {mkv_path}", file=sys.stderr)
        return 2

    try:
        mkvextract = find_tool(args.mkvextract, "mkvextract")
        mkvmerge = find_tool(args.mkvmerge, "mkvmerge")
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 2

    outdir = args.outdir or (mkv_path.with_suffix("").name + "_attachments")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        attachments = load_attachments(mkvmerge, mkv_path)
    except Exception as e:
        print(f"Failed to read attachments via mkvmerge: {e}", file=sys.stderr)
        return 1

    if not attachments:
        print("No attachments found.")
        return 0

    print(f"Found {len(attachments)} attachment(s):")
    for a in attachments:
        print(f"  id={a['id']}  name={a['file_name']}  type={a.get('content_type')}  size={a.get('size')}")

    if args.dry_run:
        return 0

    # Build mkvextract args: mkvextract attachments input.mkv id:outpath id:outpath ...
    pairs: list[str] = []
    for a in attachments:
        aid = a["id"]
        if aid is None:
            continue
        safe_name = sanitize_filename(a["file_name"])
        outpath = unique_path(outdir, safe_name)
        pairs.append(f"{aid}:{str(outpath)}")

    if not pairs:
        print("No valid attachment IDs to extract.")
        return 0

    cmd = [mkvextract, "attachments", str(mkv_path), *pairs]
    cp = run(cmd)
    if cp.returncode != 0:
        print("mkvextract failed.", file=sys.stderr)
        print(cp.stderr, file=sys.stderr)
        return cp.returncode

    # mkvextract writes progress/info to stderr in many builds, so show both if present.
    if cp.stdout.strip():
        print(cp.stdout)
    if cp.stderr.strip():
        print(cp.stderr)

    print(f"Done. Output: {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
