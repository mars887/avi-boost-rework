#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


VIDEO_EXTS = (".mkv", ".mp4")


@dataclass(frozen=True)
class SourceGroup:
    source: Path
    base: str
    source_dir: Path
    workdir: Path
    per_file_bat: Path
    result_mkv: Path
    result_mp4: Path


def is_result_name(name: str) -> bool:
    n = name.lower()
    return n.endswith("-av1.mkv") or n.endswith("-av1.mp4")


def strip_av1_suffix(stem: str) -> str:
    return stem[:-4] if stem.lower().endswith("-av1") else stem


def choose_source_path(source_dir: Path, base: str) -> Path:
    for ext in VIDEO_EXTS:
        p = source_dir / f"{base}{ext}"
        if p.exists():
            return p
    return source_dir / f"{base}.mkv"


def is_workdir(p: Path) -> bool:
    return (p / "tracks.json").exists() or (p / "zone_edit_command.txt").exists()


def find_workdir(start: Path) -> Optional[Path]:
    for parent in [start] + list(start.parents):
        if is_workdir(parent):
            return parent
    return None


def resolve_source_from_path(p: Path) -> Optional[Path]:
    p = p.resolve()
    if p.is_dir():
        if is_workdir(p):
            base = p.name
            return choose_source_path(p.parent, base)
        return None

    name = p.name
    stem = p.stem
    lower = name.lower()

    if lower.endswith((".mkv", ".mp4")):
        if is_result_name(name):
            base = strip_av1_suffix(stem)
            return choose_source_path(p.parent, base)
        return p

    if lower.endswith(".bat"):
        base = stem
        return choose_source_path(p.parent, base)

    if name in ("tracks.json", "zone_edit_command.txt"):
        workdir = p.parent
        base = workdir.name
        return choose_source_path(workdir.parent, base)

    workdir = find_workdir(p.parent)
    if workdir:
        base = workdir.name
        return choose_source_path(workdir.parent, base)

    return None


def collect_sources(args: Iterable[str]) -> Tuple[List[SourceGroup], List[str]]:
    seen: set[str] = set()
    groups: List[SourceGroup] = []
    unknown: List[str] = []

    def add_source(source: Path) -> None:
        src = source.resolve()
        key = str(src).lower()
        if key in seen:
            return
        seen.add(key)
        base = src.stem
        source_dir = src.parent
        workdir = source_dir / base
        groups.append(
            SourceGroup(
                source=src,
                base=base,
                source_dir=source_dir,
                workdir=workdir,
                per_file_bat=source_dir / f"{base}.bat",
                result_mkv=source_dir / f"{base}-av1.mkv",
                result_mp4=source_dir / f"{base}-av1.mp4",
            )
        )

    for raw in args:
        p = Path(raw).expanduser()
        if p.is_dir():
            if is_workdir(p):
                src = resolve_source_from_path(p)
                if src is not None:
                    add_source(src)
                else:
                    unknown.append(raw)
                continue
            found = False
            for ext in VIDEO_EXTS:
                for f in p.glob(f"*{ext}"):
                    if not is_result_name(f.name):
                        add_source(f)
                        found = True
            if not found:
                unknown.append(raw)
            continue

        src = resolve_source_from_path(p)
        if src is not None:
            add_source(src)
        else:
            unknown.append(raw)

    return groups, unknown


def remove_path(p: Path) -> None:
    if not p.exists():
        print(f"[skip] missing: {p}")
        return
    try:
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()
        print(f"[del] {p}")
    except Exception as e:
        print(f"[err] failed to remove {p}: {e}")


def move_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            remove_path(dst)
        shutil.move(str(src), str(dst))
        print(f"[move] {src} -> {dst}")
    except Exception as e:
        print(f"[err] failed to move {src} -> {dst}: {e}")


def clear_run(group: SourceGroup) -> None:
    workdir = group.workdir
    if not workdir.exists():
        print(f"[skip] workdir missing: {workdir}")
        return
    keep = {"zone_edit_command.txt", "tracks.json", "00_logs"}
    for entry in workdir.iterdir():
        if entry.name in keep:
            continue
        remove_path(entry)


def make_web_mp4(group: SourceGroup) -> None:
    if not group.result_mkv.exists():
        print(f"[skip] missing result: {group.result_mkv}")
        return
    if group.result_mp4.exists():
        print(f"[skip] mp4 exists: {group.result_mp4}")
        return
    cmd = [
        "ffmpeg",
        "-i", str(group.result_mkv),
        "-c", "copy",
        "-movflags", "+faststart",
        str(group.result_mp4),
    ]
    print("[cmd]", " ".join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        print(f"[err] ffmpeg failed (code={p.returncode})")


def clear_stage_mux(group: SourceGroup) -> None:
    remove_path(group.workdir / ".state" / "MUX_DONE")


def clear_stage_mainpass(group: SourceGroup) -> None:
    clear_stage_mux(group)
    remove_path(group.workdir / "video" / "mainpass")
    remove_path(group.workdir / "video" / "video-final.mkv")
    remove_path(group.workdir / ".state" / "VERIFY_DONE")


def clear_stage_zoning(group: SourceGroup) -> None:
    clear_stage_mainpass(group)
    remove_path(group.workdir / "video" / "scenes-final.json")
    remove_path(group.workdir / "video" / ".state" / "FINAL_SCENES_COMPLETED")
    remove_path(group.workdir / ".state" / "ZONE_EDIT_DONE")


def clear_stage_fastpass(group: SourceGroup) -> None:
    clear_stage_zoning(group)
    remove_path(group.workdir / ".state" / "HDR_PATCH_DONE")
    remove_path(group.workdir / "video" / ".state" / "FASTPASS_COMPLETED")
    remove_path(group.workdir / "video" / ".state" / "SSIMU2_COMPLETED")
    remove_path(group.workdir / "video" / "scenes-hdr.json")
    remove_path(group.workdir / "video" / "scenes.json")
    remove_path(group.workdir / "video" / "fastpass")


def full_clear(groups: List[SourceGroup]) -> None:
    by_dir: Dict[Path, List[SourceGroup]] = {}
    for g in groups:
        by_dir.setdefault(g.source_dir, []).append(g)

    for source_dir, items in by_dir.items():
        meta = source_dir / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        for g in items:
            move_if_exists(g.per_file_bat, meta / f"{g.base}-bat.bat")
            move_if_exists(g.workdir / "tracks.json", meta / f"{g.base}-tracks.json")
            move_if_exists(g.workdir / "zone_edit_command.txt", meta / f"{g.base}-zoning.txt")

        for entry in source_dir.iterdir():
            if entry.name == "meta":
                continue
            if entry.is_file() and is_result_name(entry.name):
                continue
            remove_path(entry)


def print_menu() -> None:
    print()
    print("Menu:")
    print("  1) Clear Run")
    print("  2) Make Web MP4")
    print("  3) Clear Stage")
    print("  4) Full Clear")
    print("  Q) Quit")


def print_clear_stage_menu() -> None:
    print()
    print("Clear Stage:")
    print("  1) Mux")
    print("  2) Mainpass")
    print("  3) Zoning")
    print("  4) Auto Boost Fastpass")
    print("  B) Back")


def main(argv: List[str]) -> int:
    if not argv:
        print("Drag files onto this script or pass paths as arguments.")
        return 1

    groups, unknown = collect_sources(argv)
    if not groups:
        print("No sources found.")
        if unknown:
            print("Unknown inputs:")
            for u in unknown:
                print(f"  {u}")
        return 1

    print("Sources:")
    for idx, g in enumerate(groups, start=1):
        exists = "OK" if g.source.exists() else "missing"
        print(f"  {idx}) {g.source} [{exists}]")
    if unknown:
        print()
        print("Unknown inputs:")
        for u in unknown:
            print(f"  {u}")

    while True:
        print_menu()
        choice = input("Select: ").strip().lower()
        if choice in ("q", "quit", "exit", "0", ""):
            return 0
        if choice == "1":
            for g in groups:
                print()
                print(f"[Clear Run] {g.base}")
                clear_run(g)
        elif choice == "2":
            for g in groups:
                print()
                print(f"[Make Web MP4] {g.base}")
                make_web_mp4(g)
        elif choice == "3":
            while True:
                print_clear_stage_menu()
                c2 = input("Select: ").strip().lower()
                if c2 in ("b", "back", ""):
                    break
                if c2 == "1":
                    for g in groups:
                        print()
                        print(f"[Clear Stage: Mux] {g.base}")
                        clear_stage_mux(g)
                elif c2 == "2":
                    for g in groups:
                        print()
                        print(f"[Clear Stage: Mainpass] {g.base}")
                        clear_stage_mainpass(g)
                elif c2 == "3":
                    for g in groups:
                        print()
                        print(f"[Clear Stage: Zoning] {g.base}")
                        clear_stage_zoning(g)
                elif c2 == "4":
                    for g in groups:
                        print()
                        print(f"[Clear Stage: Fastpass] {g.base}")
                        clear_stage_fastpass(g)
                else:
                    print("Unknown option.")
        elif choice == "4":
            full_clear(groups)
        else:
            print("Unknown option.")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
