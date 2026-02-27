#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


VIDEO_EXTS = (".mkv", ".mp4")
VPY_KEYS = ("MAIN_VPY", "FAST_VPY", "PROXY_VPY")
ADDITIONAL_WORK_EXTS = ("mkv.ffindex", "mkv.lvi")
KNOWN_WORKDIR_TOP_LEVEL = {
    "tracks.json",
    "zone_edit_command.txt",
    ".state",
    "00_logs",
    "audio",
    "video",
    "sub",
    "attachments",
    "chapters",
}
GENERATED_BATCH_FILES = (
    "start-batch.bat",
    "batch-fastpass.bat",
    "Batch Manager.bat",
)


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


def read_text_best_effort(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_bat_vpy_vars(bat_path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not bat_path.exists() or not bat_path.is_file():
        return result
    text = read_text_best_effort(bat_path)
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if not low.startswith("set "):
            continue
        payload = line[4:].strip()
        if payload.startswith('"') and payload.endswith('"') and "=" in payload:
            payload = payload[1:-1]
        if "=" not in payload:
            continue
        key_raw, value = payload.split("=", 1)
        key = key_raw.strip().upper()
        if key in VPY_KEYS:
            result[key] = value.strip()
    return result


def expand_known_bat_vars(value: str, group: SourceGroup) -> str:
    source_dir = str(group.source_dir)
    if not source_dir.endswith("\\"):
        source_dir = source_dir + "\\"
    expanded = value.replace("%~dp0", source_dir)
    known = {
        "workdir": str(group.workdir),
        "src": str(group.source),
    }

    def repl(match: re.Match[str]) -> str:
        key = match.group(1).strip().lower()
        return known.get(key, match.group(0))

    return re.sub(r"%([^%]+)%", repl, expanded)


def resolve_vpy_path_from_bat_value(raw_value: str, group: SourceGroup) -> Optional[Path]:
    value = raw_value.strip().strip('"').strip()
    if not value:
        return None
    expanded = expand_known_bat_vars(value, group)
    p = Path(expanded).expanduser()
    if not p.is_absolute():
        p = group.source_dir / p
    try:
        return p.resolve()
    except Exception:
        return p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def dump_vpy_files_to_meta(items: List[SourceGroup], meta: Path) -> None:
    refs_by_path: Dict[Path, List[Dict[str, str]]] = {}
    for g in items:
        bat_candidates = [
            g.per_file_bat,
            meta / f"{g.base}-bat.bat",
        ]
        bat_path = next((b for b in bat_candidates if b.exists()), None)
        if bat_path is None:
            continue
        vars_map = parse_bat_vpy_vars(bat_path)
        for key in VPY_KEYS:
            raw = vars_map.get(key, "").strip()
            if not raw:
                continue
            resolved = resolve_vpy_path_from_bat_value(raw, g)
            if resolved is None:
                continue
            if resolved.suffix.lower() != ".vpy":
                continue
            if not resolved.exists() or not resolved.is_file():
                print(f"[skip] vpy not found: {resolved} (from {g.base}.bat:{key})")
                continue
            refs_by_path.setdefault(resolved, []).append(
                {
                    "bat": f"{g.base}.bat",
                    "var": key,
                    "raw": raw,
                    "source": str(resolved),
                }
            )

    if not refs_by_path:
        return

    vpy_dir = meta / "vpy-files"
    vpy_dir.mkdir(parents=True, exist_ok=True)

    by_hash: Dict[str, Dict[str, object]] = {}
    for src_path in sorted(refs_by_path.keys(), key=lambda p: str(p).lower()):
        digest = sha256_file(src_path)
        entry = by_hash.setdefault(digest, {"paths": [], "uses": []})
        paths = entry["paths"]
        uses = entry["uses"]
        assert isinstance(paths, list)
        assert isinstance(uses, list)
        paths.append(str(src_path))
        uses.extend(refs_by_path[src_path])

    manifest: List[Dict[str, object]] = []
    for digest in sorted(by_hash.keys()):
        entry = by_hash[digest]
        paths = entry["paths"]
        uses = entry["uses"]
        assert isinstance(paths, list)
        assert isinstance(uses, list)
        src_path = Path(paths[0])
        dst_name = f"{digest[:12]}-{src_path.name}"
        dst = vpy_dir / dst_name
        if not dst.exists():
            shutil.copy2(src_path, dst)
            print(f"[copy] {src_path} -> {dst}")
        else:
            print(f"[skip] vpy exists: {dst}")

        uniq_uses: List[Dict[str, str]] = []
        seen_use: set[tuple[str, str, str, str]] = set()
        for u in uses:
            assert isinstance(u, dict)
            k = (
                str(u.get("bat", "")),
                str(u.get("var", "")),
                str(u.get("raw", "")),
                str(u.get("source", "")),
            )
            if k in seen_use:
                continue
            seen_use.add(k)
            uniq_uses.append(
                {
                    "bat": str(u.get("bat", "")),
                    "var": str(u.get("var", "")),
                    "raw": str(u.get("raw", "")),
                    "source": str(u.get("source", "")),
                }
            )

        manifest.append(
            {
                "copied_as": dst.name,
                "sha256": digest,
                "source_files": paths,
                "used_by": uniq_uses,
            }
        )

    manifest_path = vpy_dir / "usage.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[write] {manifest_path}")


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

def clear_stage_crf_calc(group: SourceGroup) -> None:
    clear_stage_zoning(group)
    remove_path(group.workdir / ".state" / "HDR_PATCH_DONE")
    remove_path(group.workdir / "video" / "scenes-hdr.json")
    remove_path(group.workdir / "video" / "scenes.json")

def clear_stage_fastpass(group: SourceGroup) -> None:
    clear_stage_zoning(group)
    remove_path(group.workdir / ".state" / "HDR_PATCH_DONE")
    remove_path(group.workdir / "video" / ".state" / "FASTPASS_COMPLETED")
    remove_path(group.workdir / "video" / ".state" / "SSIMU2_COMPLETED")
    remove_path(group.workdir / "video" / "scenes-hdr.json")
    remove_path(group.workdir / "video" / "scenes.json")
    remove_path(group.workdir / "video" / "fastpass")


def full_clear_workdir(group: SourceGroup) -> None:
    workdir = group.workdir
    if not workdir.exists():
        return
    remove_path(workdir)


def find_unexpected_workdir_entries(group: SourceGroup) -> List[Path]:
    workdir = group.workdir
    if not workdir.exists() or not workdir.is_dir():
        return []
    extras: List[Path] = []
    for entry in workdir.iterdir():
        if entry.name not in KNOWN_WORKDIR_TOP_LEVEL:
            extras.append(entry)
    return extras


def confirm_full_clear_workdirs(groups: List[SourceGroup]) -> bool:
    suspicious: List[Tuple[SourceGroup, List[Path]]] = []
    for g in groups:
        extras = find_unexpected_workdir_entries(g)
        if extras:
            suspicious.append((g, extras))
    if not suspicious:
        return True

    print("[warn] Found unexpected items in workdirs:")
    for g, extras in suspicious:
        print(f"  {g.workdir}")
        for p in extras:
            print(f"    - {p.name}")
    ans = input("Continue and delete these workdirs? [y/N]: ").strip().lower()
    return ans in ("y", "yes", "д", "да")


def remove_additional_work_files(group: SourceGroup) -> None:
    for ext in ADDITIONAL_WORK_EXTS:
        remove_path(group.source_dir / f"{group.base}.{ext}")


def dump_configs_to_meta(groups: List[SourceGroup]) -> Dict[Path, List[SourceGroup]]:
    by_dir: Dict[Path, List[SourceGroup]] = {}
    for g in groups:
        by_dir.setdefault(g.source_dir, []).append(g)

    for source_dir, items in by_dir.items():
        meta = source_dir / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        dump_vpy_files_to_meta(items, meta)
        for g in items:
            move_if_exists(g.per_file_bat, meta / f"{g.base}-bat.bat")
            move_if_exists(g.workdir / "tracks.json", meta / f"{g.base}-tracks.json")
            move_if_exists(g.workdir / "zone_edit_command.txt", meta / f"{g.base}-zoning.txt")

    return by_dir


def full_clear(groups: List[SourceGroup]) -> None:
    if not confirm_full_clear_workdirs(groups):
        print("[skip] Full Clear cancelled by user.")
        return

    by_dir = dump_configs_to_meta(groups)
    for source_dir, items in by_dir.items():
        for g in items:
            full_clear_workdir(g)
            remove_additional_work_files(g)

        for name in GENERATED_BATCH_FILES:
            remove_path(source_dir / name)


def config_dump(groups: List[SourceGroup]) -> None:
    dump_configs_to_meta(groups)


def verify_config(group: SourceGroup, *, check_filters: bool, check_params: bool) -> None:
    script = Path(__file__).with_name("batch-verify.py")
    if not script.exists():
        print(f"[err] verify script not found: {script}")
        return
    cmd = [
        sys.executable,
        str(script),
        "--source", str(group.source),
        "--workdir", str(group.workdir),
        "--per-file-bat", str(group.per_file_bat),
        "--result-mkv", str(group.result_mkv),
        "--result-mp4", str(group.result_mp4),
    ]
    if check_filters:
        cmd.append("--check-filters")
    if check_params:
        cmd.append("--check-params")
    print("[cmd]", " ".join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        print(f"[err] verify failed (code={p.returncode})")


def print_menu() -> None:
    print()
    print("Menu:")
    print("  1) Clear Run")
    print("  2) Make Web MP4")
    print("  3) Clear Stage")
    print("  4) Full Clear")
    print("  5) Config Dump")
    print("  6) Verify")
    print("  Q) Quit")


def print_clear_stage_menu() -> None:
    print()
    print("Clear Stage:")
    print("  1) Mux")
    print("  2) Mainpass")
    print("  3) Zoning")
    print("  4) Calculation")
    print("  5) Auto Boost Fastpass")
    print("  B) Back")


def main(argv: List[str]) -> int:
    check_filters = True
    check_params = False
    paths: List[str] = []
    for a in argv:
        if a == "--check-filters":
            continue
        elif a == "--check-params":
            continue
        else:
            paths.append(a)
    if not paths:
        print("No input args. Scanning current directory...")
        paths = ["."]

    groups, unknown = collect_sources(paths)
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
                        print(f"[Clear Stage: Calculation] {g.base}")
                        clear_stage_crf_calc(g)
                elif c2 == "5":
                    for g in groups:
                        print()
                        print(f"[Clear Stage: Fastpass] {g.base}")
                        clear_stage_fastpass(g)
                else:
                    print("Unknown option.")
        elif choice == "4":
            full_clear(groups)
        elif choice == "5":
            config_dump(groups)
        elif choice == "6":
            for g in groups:
                print()
                print(f"[Verify] {g.base}")
                verify_config(g, check_filters=check_filters, check_params=check_params)
        else:
            print("Unknown option.")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
