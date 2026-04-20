#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.pipeline_runtime import load_toolchain
from utils.plan_model import (
    FilePlan,
    load_file_plan,
    load_plan,
    resolve_batch_plan,
    resolve_paths,
    save_plan,
)
from utils.plan_support import collect_file_plan_paths, refresh_directory_support_files

VIDEO_EXTS = (".mkv", ".mp4")
ADDITIONAL_WORK_EXTS = ("ffindex", "lvi")
KNOWN_WORKDIR_TOP_LEVEL = {
    "zone_edit_command.txt",
    "crop_resize_command.txt",
    ".state",
    "00_meta",
    "00_logs",
    "audio",
    "video",
    "sub",
    "attachments",
    "chapters",
    "fastpass-analytics",
    "mainpass-analytics",
}
GENERATED_BATCH_FILES = (
    "runner.bat",
    "Batch Manager.bat",
    "full-batch.plan",
    "fastpass-batch.plan",
    "start-batch.bat",
    "batch-fastpass.bat",
)


@dataclass(frozen=True)
class SourceGroup:
    source: Path
    base: str
    source_dir: Path
    workdir: Path
    plan_path: Path
    zone_file: Path
    crop_resize_file: Path
    runner_bat: Path
    manager_bat: Path
    full_batch_plan: Path
    fastpass_batch_plan: Path
    result_mkv: Path
    result_mp4: Path

    @property
    def per_file_bat(self) -> Path:
        return self.plan_path


@dataclass(frozen=True)
class EditMatch:
    index: int
    group: SourceGroup
    file_path: Path
    line_no: int
    line_text: str


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
    return (p / "zone_edit_command.txt").exists() or (p / "crop_resize_command.txt").exists() or (p / "video").exists() or (p / "audio").exists()


def build_group_from_source(source: Path) -> SourceGroup:
    src = source.resolve()
    base = src.stem
    source_dir = src.parent
    plan_path = source_dir / f"{base}.plan"
    workdir = source_dir / base
    zone_file = workdir / "zone_edit_command.txt"
    crop_resize_file = workdir / "crop_resize_command.txt"
    if plan_path.exists():
        try:
            plan = load_file_plan(plan_path)
            resolved = resolve_paths(plan, plan_path)
            workdir = resolved.workdir
            zone_file = resolved.zone_file
            crop_resize_file = resolved.crop_resize_file
        except Exception:
            pass
    return SourceGroup(
        source=src,
        base=base,
        source_dir=source_dir,
        workdir=workdir,
        plan_path=plan_path,
        zone_file=zone_file,
        crop_resize_file=crop_resize_file,
        runner_bat=source_dir / "runner.bat",
        manager_bat=source_dir / "Batch Manager.bat",
        full_batch_plan=source_dir / "full-batch.plan",
        fastpass_batch_plan=source_dir / "fastpass-batch.plan",
        result_mkv=source_dir / f"{base}-av1.mkv",
        result_mp4=source_dir / f"{base}-av1.mp4",
    )


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
        plans = collect_file_plan_paths(p)
        if len(plans) == 1:
            try:
                return resolve_paths(load_file_plan(plans[0]), plans[0]).source
            except Exception:
                return None
        return None

    name = p.name
    stem = p.stem
    lower = name.lower()

    if lower.endswith((".mkv", ".mp4")):
        if is_result_name(name):
            base = strip_av1_suffix(stem)
            return choose_source_path(p.parent, base)
        return p

    if lower.endswith(".plan"):
        try:
            plan = load_plan(p)
        except Exception:
            return None
        if isinstance(plan, FilePlan):
            return resolve_paths(plan, p).source
        resolved = resolve_batch_plan(p)
        if len(resolved) == 1:
            return resolved[0].paths.source
        return None

    if lower.endswith(".bat"):
        if stem.lower() == "runner":
            full = p.parent / "full-batch.plan"
            if full.exists():
                resolved = resolve_batch_plan(full)
                if len(resolved) == 1:
                    return resolved[0].paths.source
        return None

    if name in ("zone_edit_command.txt",):
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
        group = build_group_from_source(source)
        key = str(group.source).lower()
        if key in seen:
            return
        seen.add(key)
        groups.append(group)

    def add_from_plan(plan_path: Path) -> bool:
        try:
            plan = load_plan(plan_path)
        except Exception:
            return False
        if isinstance(plan, FilePlan):
            add_source(resolve_paths(plan, plan_path).source)
            return True
        resolved_items = resolve_batch_plan(plan_path)
        for resolved in resolved_items:
            add_source(resolved.paths.source)
        return bool(resolved_items)

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
            full_batch_plan = p / "full-batch.plan"
            if full_batch_plan.exists():
                if add_from_plan(full_batch_plan):
                    found = True
            plans = collect_file_plan_paths(p)
            if plans:
                for plan_path in plans:
                    try:
                        add_source(resolve_paths(load_file_plan(plan_path), plan_path).source)
                        found = True
                    except Exception:
                        unknown.append(str(plan_path))
            for ext in VIDEO_EXTS:
                for f in p.glob(f"*{ext}"):
                    if not is_result_name(f.name):
                        add_source(f)
                        found = True
            if not found:
                unknown.append(raw)
            continue

        if p.suffix.lower() == ".plan":
            if not add_from_plan(p):
                unknown.append(raw)
            continue

        if p.suffix.lower() == ".bat" and p.name.lower() in ("runner.bat", "batch manager.bat"):
            plan_paths = collect_file_plan_paths(p.parent)
            if not plan_paths:
                unknown.append(raw)
                continue
            for plan_path in plan_paths:
                add_from_plan(plan_path)
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
            with path.open("r", encoding=enc, newline="") as f:
                return f.read()
        except Exception:
            continue
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return f.read()


def normalize_text_newlines(text: str) -> str:
    text = re.sub(r"\r+\n", "\n", text)
    return text.replace("\r", "\n")


def vpy_refs_from_plan(group: SourceGroup) -> List[Tuple[str, Path]]:
    if not group.plan_path.exists():
        return []
    try:
        plan = load_file_plan(group.plan_path)
    except Exception:
        return []

    resolved = resolve_paths(plan, group.plan_path)
    refs: List[Tuple[str, Path]] = []
    for key, raw_value in (
        ("MAIN_VPY", plan.video.details.main_vpy),
        ("FAST_VPY", plan.video.details.fast_vpy),
        ("PROXY_VPY", plan.video.details.proxy_vpy),
    ):
        value = str(raw_value or "").strip()
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (group.plan_path.parent / path).resolve()
        if path.suffix.lower() != ".vpy":
            continue
        refs.append((key, path))
    return refs


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
        for key, resolved in vpy_refs_from_plan(g):
            if not resolved.exists() or not resolved.is_file():
                print(f"[skip] vpy not found: {resolved} (from {g.plan_path.name}:{key})")
                continue
            refs_by_path.setdefault(resolved, []).append(
                {
                    "plan": g.plan_path.name,
                    "var": key,
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
        seen_use: set[tuple[str, str, str]] = set()
        for u in uses:
            assert isinstance(u, dict)
            k = (
                str(u.get("plan", "")),
                str(u.get("var", "")),
                str(u.get("source", "")),
            )
            if k in seen_use:
                continue
            seen_use.add(k)
            uniq_uses.append(
                {
                    "plan": str(u.get("plan", "")),
                    "var": str(u.get("var", "")),
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
    keep = {"00_logs"}
    if group.zone_file.parent.resolve() == workdir.resolve():
        keep.add(group.zone_file.name)
    if group.crop_resize_file.parent.resolve() == workdir.resolve():
        keep.add(group.crop_resize_file.name)
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
    remove_path(group.workdir / "video" / "scenes-zoned.json")
    remove_path(group.workdir / "video" / ".state" / "FINAL_SCENES_COMPLETED")
    remove_path(group.workdir / ".state" / "HDR_PATCH_DONE")
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
        remove_path(group.source_dir / f"{group.source.name}.{ext}")


def dump_configs_to_meta(groups: List[SourceGroup]) -> Dict[Path, List[SourceGroup]]:
    by_dir: Dict[Path, List[SourceGroup]] = {}
    for g in groups:
        by_dir.setdefault(g.source_dir, []).append(g)

    for source_dir, items in by_dir.items():
        meta = source_dir / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        dump_vpy_files_to_meta(items, meta)
        moved_dir_level: set[str] = set()
        for g in items:
            move_if_exists(g.plan_path, meta / f"{g.base}.plan")
            move_if_exists(g.zone_file, meta / f"{g.base}-zoning.txt")
            move_if_exists(g.crop_resize_file, meta / f"{g.base}-crop-resize.txt")
            for src, dst_name in (
                (g.full_batch_plan, "full-batch.plan"),
                (g.fastpass_batch_plan, "fastpass-batch.plan"),
                (g.runner_bat, "runner.bat"),
                (g.manager_bat, "Batch Manager.bat"),
            ):
                key = str(src).lower()
                if key in moved_dir_level:
                    continue
                moved_dir_level.add(key)
                move_if_exists(src, meta / dst_name)

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


def to_windows_eol(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")


def write_text_with_fallback(path: Path, text: str, *, preferred_encoding: str = "cp1251") -> None:
    try:
        path.write_text(text, encoding=preferred_encoding, newline="")
    except UnicodeEncodeError:
        path.write_text(text, encoding="utf-8", newline="")


def enter_numbers(raw: str, min_value: int, max_value: int) -> List[int]:
    selected: List[int] = []
    for token in raw.split():
        try:
            if token == "*":
                selected.extend(range(min_value, max_value + 1))
                continue
            if ".." in token:
                parts = token.split("..", 1)
                a = int(parts[0])
                b = int(parts[1])
                lo = min(a, b)
                hi = max(a, b)
                selected.extend(range(lo, hi + 1))
                continue
            if "-." in token:
                parts = token.split("-.", 1)
                a = int(parts[0])
                b = int(parts[1])
                lo = min(a, b)
                hi = max(a, b)
                selected = [x for x in selected if x < lo or x > hi]
                continue
            if token.startswith("/") and token[1:].isdigit():
                value = int(token[1:])
                selected = [x for x in selected if x != value]
                continue
            if token.isdigit():
                selected.append(int(token))
                continue
            print(f"[warn] invalid selection token skipped: {token}")
        except Exception as e:
            print(f"[warn] failed to parse selection token {token!r}: {e}")

    out: List[int] = []
    seen: set[int] = set()
    for value in selected:
        if value < min_value or value > max_value:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def prompt_edit_target() -> Optional[str]:
    print()
    print("Edit:")
    print("  1) {basename}.plan")
    print("  2) full-batch.plan")
    print("  3) fastpass-batch.plan")
    print("  4) zone_edit_command.txt")
    print("  5) crop_resize_command.txt")
    print("  B) Back")
    while True:
        choice = input("Select file to edit: ").strip().lower()
        if choice in ("b", "back", ""):
            return None
        if choice == "1":
            return "plan"
        if choice == "2":
            return "full-batch"
        if choice == "3":
            return "fastpass-batch"
        if choice == "4":
            return "zone"
        if choice == "5":
            return "crop-resize"
        print("Unknown option.")


def edit_target_path(group: SourceGroup, target: str) -> Path:
    if target == "plan":
        return group.plan_path
    if target == "full-batch":
        return group.full_batch_plan
    if target == "fastpass-batch":
        return group.fastpass_batch_plan
    if target == "zone":
        return group.zone_file
    if target == "crop-resize":
        return group.crop_resize_file
    raise ValueError(f"Unsupported edit target: {target}")


def edit_target_label(target: str) -> str:
    if target == "plan":
        return "{basename}.plan"
    if target == "full-batch":
        return "full-batch.plan"
    if target == "fastpass-batch":
        return "fastpass-batch.plan"
    if target == "zone":
        return "zone_edit_command.txt"
    if target == "crop-resize":
        return "crop_resize_command.txt"
    return target


def find_edit_matches(groups: List[SourceGroup], target: str, needle: str) -> List[EditMatch]:
    matches: List[EditMatch] = []
    idx = 1
    for group in groups:
        file_path = edit_target_path(group, target)
        if not file_path.exists():
            continue
        try:
            text = read_text_best_effort(file_path)
        except Exception as e:
            print(f"[err] failed to read {file_path}: {e}")
            continue
        for line_no, line in enumerate(normalize_text_newlines(text).split("\n"), start=1):
            if needle not in line:
                continue
            matches.append(
                EditMatch(
                    index=idx,
                    group=group,
                    file_path=file_path,
                    line_no=line_no,
                    line_text=line,
                )
            )
            idx += 1
    return matches


def print_edit_matches(matches: List[EditMatch]) -> None:
    print()
    print("Matches:")
    for match in matches:
        print(f"  {match.index}) {match.group.base} | {match.file_path.name}:{match.line_no}")
        print(f"     {match.line_text}")


def replace_selected_lines(matches: List[EditMatch], replacement: str) -> int:
    by_file: Dict[Path, List[EditMatch]] = {}
    for match in matches:
        by_file.setdefault(match.file_path, []).append(match)

    changed = 0
    for file_path, file_matches in by_file.items():
        try:
            raw = read_text_best_effort(file_path)
        except Exception as e:
            print(f"[err] failed to read {file_path}: {e}")
            continue

        normalized = normalize_text_newlines(raw)
        trailing_newline = normalized.endswith("\n")
        lines = normalized.split("\n")
        if trailing_newline and lines and lines[-1] == "":
            lines = lines[:-1]

        changed_here = 0
        for match in file_matches:
            index = match.line_no - 1
            if index < 0 or index >= len(lines):
                print(f"[warn] line out of range, skipped: {file_path}:{match.line_no}")
                continue
            lines[index] = replacement
            changed_here += 1

        if changed_here == 0:
            continue

        out = "\n".join(lines)
        if trailing_newline:
            out += "\n"

        preferred_encoding = "cp1251" if file_path.suffix.lower() == ".bat" else "utf-8"
        try:
            write_text_with_fallback(file_path, to_windows_eol(out) if file_path.suffix.lower() == ".bat" else out, preferred_encoding=preferred_encoding)
            print(f"[write] {file_path} ({changed_here} line(s))")
            changed += changed_here
        except Exception as e:
            print(f"[err] failed to write {file_path}: {e}")
    return changed


def edit_config_lines(groups: List[SourceGroup]) -> None:
    target = prompt_edit_target()
    if target is None:
        return

    needle = input(f"Find string in {edit_target_label(target)}: ")
    if needle == "":
        print("[skip] empty search string.")
        return

    matches = find_edit_matches(groups, target, needle)
    if not matches:
        print(f"[skip] no matches found in {edit_target_label(target)}.")
        return

    print_edit_matches(matches)
    selection_raw = input("Select matches to edit (e.g. 1 2 5..7 * /3): ").strip()
    selected_ids = enter_numbers(selection_raw, 1, len(matches))
    if not selected_ids:
        print("[skip] nothing selected.")
        return

    selected = [m for m in matches if m.index in set(selected_ids)]
    replacement = input("Replacement line: ")
    changed = replace_selected_lines(selected, replacement)
    print(f"[done] Edit completed, replaced {changed} line(s).")


def is_template_source(group: SourceGroup) -> bool:
    return group.plan_path.exists()


def is_new_source_without_config(group: SourceGroup) -> bool:
    return (not group.plan_path.exists()) and (not group.workdir.exists())


def is_partial_source_state(group: SourceGroup) -> bool:
    if is_template_source(group):
        return False
    if is_new_source_without_config(group):
        return False
    return group.plan_path.exists() or group.workdir.exists()


def choose_template_group(candidates: List[SourceGroup], source_dir: Path) -> Optional[SourceGroup]:
    print()
    print(f"[Expand] Sources with existing config in: {source_dir}")
    for idx, g in enumerate(candidates, start=1):
        print(f"  {idx}) {g.base} ({g.plan_path.name})")

    while True:
        raw = input("Select template source [number, Enter=skip]: ").strip().lower()
        if raw in ("", "q", "quit", "b", "back", "s", "skip"):
            return None
        if not raw.isdigit():
            print("Please enter a valid number.")
            continue
        idx = int(raw)
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1]
        print("Index out of range.")


def copy_zone_command(template: SourceGroup, target: SourceGroup) -> bool:
    return copy_sidecar_config(template.zone_file, target.zone_file, "zoning command")


def copy_crop_resize_command(template: SourceGroup, target: SourceGroup) -> bool:
    return copy_sidecar_config(template.crop_resize_file, target.crop_resize_file, "crop/resize command")


def copy_sidecar_config(src_path: Path, dst_path: Path, label: str) -> bool:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
        else:
            dst_path.write_text("", encoding="utf-8", newline="\n")
        print(f"[write] {dst_path}")
        return True
    except Exception as e:
        print(f"[err] failed to write {label} {dst_path}: {e}")
        return False


def clone_source_config(template: SourceGroup, target: SourceGroup) -> bool:
    if target.plan_path.exists():
        print(f"[skip] target already has config: {target.base}")
        return False
    if not template.plan_path.exists():
        print(f"[err] missing template plan: {template.plan_path}")
        return False

    try:
        target.workdir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[err] failed to create workdir {target.workdir}: {e}")
        return False

    try:
        template_plan = load_file_plan(template.plan_path)
        cloned = FilePlan(
            format_version=template_plan.format_version,
            plan_type=template_plan.plan_type,
            meta=template_plan.meta.__class__(
                name=target.base,
                created_by="batch-manager.py:expand",
                mode=template_plan.meta.mode,
            ),
            paths=template_plan.paths.__class__(
                source=target.source.name,
            ),
            video=template_plan.video,
            audio=list(template_plan.audio),
            sub=list(template_plan.sub),
        )
        save_plan(cloned, target.plan_path)
        print(f"[write] {target.plan_path}")
    except Exception as e:
        print(f"[err] failed to write template plan {target.plan_path}: {e}")
        return False

    ok_zone = copy_zone_command(template, target)
    ok_crop = copy_crop_resize_command(template, target)
    return ok_zone and ok_crop


def expand_configs_for_new_sources(groups: List[SourceGroup]) -> None:
    by_dir: Dict[Path, List[SourceGroup]] = {}
    for g in groups:
        by_dir.setdefault(g.source_dir, []).append(g)

    had_new = False
    total_created = 0
    for source_dir, items in sorted(by_dir.items(), key=lambda x: str(x[0]).lower()):
        items_sorted = sorted(items, key=lambda g: g.base.lower())
        new_sources = [g for g in items_sorted if is_new_source_without_config(g)]
        if not new_sources:
            continue
        had_new = True

        print()
        print(f"[Expand] New sources in: {source_dir}")
        for g in new_sources:
            print(f"  - {g.source.name}")

        partial = [g for g in items_sorted if is_partial_source_state(g)]
        if partial:
            print("[warn] Found partial config state:")
            for g in partial:
                plan_state = "yes" if g.plan_path.exists() else "no"
                work_state = "yes" if g.workdir.exists() else "no"
                print(f"  - {g.base}: plan={plan_state}, workdir={work_state}")

        templates = [g for g in items_sorted if is_template_source(g)]
        if not templates:
            print("[warn] No template sources with file .plan in this folder.")
            continue

        template = choose_template_group(templates, source_dir)
        if template is None:
            print("[skip] Expand skipped for this folder.")
            continue

        created_bases: List[str] = []
        for target in new_sources:
            print()
            print(f"[Expand] {target.base} <= {template.base}")
            if clone_source_config(template, target):
                created_bases.append(target.base)

        if not created_bases:
            continue

        total_created += len(created_bases)
        toolchain = load_toolchain()
        refresh_directory_support_files(source_dir=source_dir, python_exe=toolchain.python_exe, project_root=ROOT)

    if not had_new:
        print("[skip] No new sources without .plan and workdir.")
        return
    print(f"[done] Expand completed, created {total_created} config(s).")


def refresh_support_files_for_groups(groups: List[SourceGroup]) -> None:
    source_dirs = sorted({g.source_dir.resolve() for g in groups}, key=lambda item: str(item).lower())
    if not source_dirs:
        return
    toolchain = load_toolchain()
    for source_dir in source_dirs:
        if not collect_file_plan_paths(source_dir):
            continue
        required = (
            source_dir / "runner.bat",
            source_dir / "Batch Manager.bat",
            source_dir / "full-batch.plan",
            source_dir / "fastpass-batch.plan",
        )
        if all(path.exists() for path in required):
            continue
        refresh_directory_support_files(source_dir=source_dir, python_exe=toolchain.python_exe, project_root=ROOT)


def verify_config(group: SourceGroup, *, check_filters: bool, check_params: bool) -> None:
    script = Path(__file__).with_name("batch-verify.py")
    if not script.exists():
        print(f"[err] verify script not found: {script}")
        return
    if not group.plan_path.exists():
        print(f"[err] file plan not found: {group.plan_path}")
        return
    cmd = [
        sys.executable,
        str(script),
        "--plan", str(group.plan_path),
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


def parse_frame_rate_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if v > 0 else None
    s = str(value).strip()
    if not s:
        return None
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            num = float(a)
            den = float(b)
            if den == 0:
                return None
            v = num / den
            return v if v > 0 else None
        except Exception:
            return None
    try:
        v = float(s)
        return v if v > 0 else None
    except Exception:
        return None


def get_source_fps(source: Path) -> Optional[float]:
    if not shutil.which("ffprobe"):
        return None
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate",
        "-of", "csv=p=0",
        str(source),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        return None
    for line in p.stdout.splitlines():
        parts = [x.strip() for x in line.split(",") if x.strip()]
        for part in parts:
            fps = parse_frame_rate_value(part)
            if fps:
                return fps
    return None


def load_pass_chunks(pass_dir: Path) -> Dict[int, Dict[str, Any]]:
    chunks_path = pass_dir / "chunks.json"
    if not chunks_path.exists():
        return {}
    try:
        raw = json.loads(read_text_best_effort(chunks_path))
    except Exception:
        return {}

    chunks_raw: Any
    if isinstance(raw, dict):
        chunks_raw = raw.get("chunks")
    else:
        chunks_raw = raw
    if not isinstance(chunks_raw, list):
        return {}

    by_index: Dict[int, Dict[str, Any]] = {}
    for item in chunks_raw:
        if not isinstance(item, dict):
            continue
        if "index" not in item:
            continue
        try:
            idx = int(item["index"])
        except Exception:
            continue
        by_index[idx] = item
    return by_index


def choose_scenes_path_for_analytics(workdir: Path) -> Path:
    video_dir = workdir / "video"
    p1 = video_dir / "scenes.json"
    if p1.exists():
        return p1
    p2 = video_dir / "scenes-preview.json"
    if p2.exists():
        return p2
    raise RuntimeError(f"Missing scenes file: expected {p1} or {p2}")


def load_scenes_for_analytics(workdir: Path) -> Tuple[Path, List[Dict[str, Any]]]:
    scenes_path = choose_scenes_path_for_analytics(workdir)
    try:
        obj = json.loads(read_text_best_effort(scenes_path))
    except Exception as e:
        raise RuntimeError(f"Failed to parse scenes json: {scenes_path} ({e})") from e

    scenes_raw = obj.get("split_scenes")
    if not isinstance(scenes_raw, list) or not scenes_raw:
        scenes_raw = obj.get("scenes")
    if not isinstance(scenes_raw, list) or not scenes_raw:
        raise RuntimeError(f"Invalid scenes data in {scenes_path}: missing non-empty scenes list")

    out: List[Dict[str, Any]] = []
    for i, s in enumerate(scenes_raw):
        if not isinstance(s, dict):
            raise RuntimeError(f"Invalid scene entry at index {i}: expected object")
        if "start_frame" not in s or "end_frame" not in s:
            raise RuntimeError(f"Invalid scene entry at index {i}: missing start_frame/end_frame")
        out.append(s)
    return scenes_path, out


def parse_crf_from_video_params(tokens: List[str]) -> Optional[float]:
    idx = -1
    for i, tok in enumerate(tokens):
        if tok == "--crf":
            idx = i
    if idx < 0 or idx + 1 >= len(tokens):
        return None
    try:
        return float(str(tokens[idx + 1]).strip())
    except Exception:
        return None


def scene_crf(scene: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(scene, dict):
        return None
    zo = scene.get("zone_overrides")
    if not isinstance(zo, dict):
        return None
    vp = zo.get("video_params")
    if not isinstance(vp, list):
        return None
    tokens = [str(x) for x in vp]
    return parse_crf_from_video_params(tokens)


def collect_ivf_files_by_index(encode_dir: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for p in encode_dir.glob("*.ivf"):
        stem = p.stem.strip()
        if not stem.isdigit():
            continue
        idx = int(stem)
        prev = out.get(idx)
        if prev is None:
            out[idx] = p
            continue
        # Prefer shorter stem representation if duplicated (e.g. 1.ivf over 00001.ivf)
        if len(p.stem) < len(prev.stem):
            out[idx] = p
    return out


def parse_ssimu2_log_file(path: Path) -> Tuple[int, List[float]]:
    scores: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
        m = re.search(r"skip:\s*([0-9]+)", first)
        if not m:
            raise RuntimeError(f"Skip value not detected in SSIMU2 log: {path}")
        skip = int(m.group(1))
        for line in f:
            m2 = re.search(r"([0-9]+):\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)", line.strip())
            if not m2:
                continue
            try:
                scores.append(max(float(m2.group(2)), 0.0))
            except Exception:
                continue
    if not scores:
        raise RuntimeError(f"No SSIMU2 values parsed: {path}")
    return skip, scores


def slice_metric_samples(scores: List[float], st: int, en: int, skip: int) -> List[float]:
    """
    scores[k] corresponds to frame index k*skip.
    Select subset where st <= k*skip < en.
    """
    if not scores:
        return []
    if skip <= 0:
        skip = 1
    if en <= st:
        k = max(0, min((st + skip - 1) // skip, len(scores) - 1))
        return [scores[k]]
    k0 = (st + skip - 1) // skip
    k1 = (en - 1) // skip
    k0 = max(0, min(k0, len(scores) - 1))
    k1 = max(0, min(k1, len(scores) - 1))
    if k1 < k0:
        return [scores[k0]]
    out = scores[k0: k1 + 1]
    return out if out else [scores[k0]]


def slice_metric_samples_strict(scores: List[float], st: int, en: int, skip: int) -> List[float]:
    if not scores:
        return []
    if skip <= 0:
        skip = 1
    if en <= st:
        return []
    k0 = (st + skip - 1) // skip
    k1 = (en - 1) // skip
    k0 = max(0, k0)
    k1 = min(len(scores) - 1, k1)
    if k1 < k0:
        return []
    return scores[k0: k1 + 1]


def calc_avg_p5(values: List[float]) -> Tuple[float, float]:
    if not values:
        raise RuntimeError("Cannot compute stats for empty metric list")
    filtered = [max(float(v), 0.0) for v in values]
    sorted_vals = sorted(filtered)
    avg = sum(filtered) / len(filtered)
    p5 = sorted_vals[max(0, len(filtered) // 20)]
    return float(avg), float(p5)


def format_float(value: Optional[float], digits: int) -> str:
    if value is None:
        return "-"
    if not math.isfinite(value):
        return "-"
    return f"{float(value):.{digits}f}"


def frame_to_timecode(frame: int, fps: float) -> str:
    if fps <= 0:
        return "-"
    total = max(0.0, float(frame) / float(fps))
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    seconds = total % 60.0
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def _xml_local_name(tag: str) -> str:
    if not tag:
        return ""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def parse_chapter_time_sec(raw: str) -> Optional[float]:
    s = str(raw).strip().replace(",", ".")
    m = re.match(r"^(\d+):(\d+):(\d+(?:\.\d+)?)$", s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = float(m.group(3))
    return (hh * 3600.0) + (mm * 60.0) + ss


def load_chapters_xml(chapters_path: Path) -> List[Dict[str, Any]]:
    if not chapters_path.exists():
        return []
    try:
        root = ET.parse(chapters_path).getroot()
    except Exception as e:
        raise RuntimeError(f"Failed to parse chapters XML: {chapters_path} ({e})") from e

    raw: List[Tuple[float, str]] = []
    for atom in root.iter():
        if _xml_local_name(atom.tag) != "ChapterAtom":
            continue
        start_txt: Optional[str] = None
        title_txt: Optional[str] = None
        for node in atom.iter():
            name = _xml_local_name(node.tag)
            if start_txt is None and name == "ChapterTimeStart" and node.text:
                start_txt = node.text.strip()
            elif title_txt is None and name == "ChapterString" and node.text:
                title_txt = node.text.strip()
            if start_txt is not None and title_txt is not None:
                break
        if not start_txt:
            continue
        sec = parse_chapter_time_sec(start_txt)
        if sec is None:
            continue
        raw.append((sec, title_txt or ""))

    if not raw:
        return []
    raw.sort(key=lambda x: x[0])

    dedup: List[Tuple[float, str]] = []
    for sec, title in raw:
        if dedup and abs(sec - dedup[-1][0]) < 1e-6:
            if not dedup[-1][1] and title:
                dedup[-1] = (sec, title)
            continue
        dedup.append((sec, title))

    out: List[Dict[str, Any]] = []
    for i, (sec, title) in enumerate(dedup):
        end_sec = dedup[i + 1][0] if i + 1 < len(dedup) else float("inf")
        out.append(
            {
                "name": title or f"Chapter {i + 1}",
                "start_sec": float(sec),
                "end_sec": float(end_sec),
            }
        )
    return out


def build_chapter_frame_ranges(chapters: List[Dict[str, Any]], fps: float, total_frames: int) -> List[Dict[str, Any]]:
    if fps <= 0:
        return []
    out: List[Dict[str, Any]] = []
    for ch in chapters:
        start_sec = float(ch.get("start_sec", 0.0))
        end_sec = float(ch.get("end_sec", float("inf")))
        st = max(0, int(math.floor(start_sec * fps + 1e-9)))
        if math.isinf(end_sec):
            en = max(total_frames, st + 1)
        else:
            en = int(math.ceil(end_sec * fps - 1e-9))
            if en <= st:
                en = st + 1
            if total_frames > 0:
                en = min(en, total_frames)
        out.append(
            {
                "name": str(ch.get("name") or ""),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_frame": st,
                "end_frame": en,
            }
        )
    return out


def chapter_name_for_frame(frame: int, chapters: List[Dict[str, Any]]) -> str:
    for ch in chapters:
        st = int(ch.get("start_frame", 0))
        en = int(ch.get("end_frame", st + 1))
        if frame >= st and frame < en:
            name = str(ch.get("name") or "").strip()
            return name if name else "-"
    return "-"


def mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not nums:
        return None
    return sum(nums) / len(nums)


def render_aligned_pipe_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    col_count = max(len(r) for r in rows)
    normalized: List[List[str]] = []
    widths = [0] * col_count

    for row in rows:
        rr = [str(x) for x in row] + [""] * (col_count - len(row))
        normalized.append(rr)
        for i, cell in enumerate(rr):
            if len(cell) > widths[i]:
                widths[i] = len(cell)

    lines: List[str] = []
    for rr in normalized:
        lines.append(" | ".join(rr[i].ljust(widths[i]) for i in range(col_count)))
    return "\n".join(lines) + "\n"


def build_step_series(records: List[Dict[str, Any]], key: str) -> Tuple[List[float], List[float]]:
    if not records:
        return [], []
    x: List[float] = []
    y: List[float] = []
    for rec in records:
        x.append(float(rec["start_frame"]))
        val = rec.get(key)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            y.append(float(val))
        else:
            y.append(float("nan"))
    x.append(float(records[-1]["end_frame"]))
    y.append(y[-1])
    return x, y


def has_any_finite(values: List[float]) -> bool:
    for v in values:
        if math.isfinite(v):
            return True
    return False


def draw_chapter_boundaries(ax: Any, chapters: List[Dict[str, Any]]) -> None:
    for ch in chapters[1:]:
        x = int(ch.get("start_frame", 0))
        ax.axvline(x=x, color="0.55", linestyle="--", linewidth=0.8, alpha=0.6)


def write_info_plot(
    *,
    out_path: Path,
    title: str,
    pass_name: str,
    records: List[Dict[str, Any]],
    chapters: List[Dict[str, Any]],
    ssimu_skip: Optional[int],
    ssimu_scores: Optional[List[float]],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise RuntimeError(f"matplotlib is required to generate {out_path.name}: {e}") from e

    fastpass = pass_name == "fastpass"
    if fastpass:
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        ax_crf, ax_kbps, ax_ssim = axes
    else:
        fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
        ax_crf, ax_kbps = axes
        ax_ssim = None

    x_crf, y_crf = build_step_series(records, "crf")
    crf_plotted = False
    if has_any_finite(y_crf):
        ax_crf.step(x_crf, y_crf, where="post", color="tab:red", linewidth=1.5, label=("new_crf" if fastpass else "crf"))
        crf_plotted = True
    ax_crf.set_ylabel("CRF")
    ax_crf.grid(True, alpha=0.25)
    if crf_plotted:
        ax_crf.legend(loc="best")

    x_kbps, y_kbps = build_step_series(records, "kbps")
    kbps_plotted = False
    if has_any_finite(y_kbps):
        ax_kbps.step(x_kbps, y_kbps, where="post", color="tab:blue", linewidth=1.5, label="kbps")
        kbps_plotted = True
    ax_kbps.set_ylabel("kbps")
    ax_kbps.grid(True, alpha=0.25)
    if kbps_plotted:
        ax_kbps.legend(loc="best")

    if fastpass and ax_ssim is not None:
        x_scene_ssim, y_scene_ssim = build_step_series(records, "ssimu_scene")
        ssim_plotted = False
        if has_any_finite(y_scene_ssim):
            ax_ssim.step(
                x_scene_ssim,
                y_scene_ssim,
                where="post",
                color="tab:orange",
                linewidth=1.4,
                label="ssimu_scene",
            )
            ssim_plotted = True
        if ssimu_skip and ssimu_skip > 0 and ssimu_scores:
            x_frame = [i * int(ssimu_skip) for i in range(len(ssimu_scores))]
            ax_ssim.plot(x_frame, ssimu_scores, color="tab:green", linewidth=0.8, alpha=0.8, label="ssimu")
            ssim_plotted = True
        ax_ssim.set_ylabel("ssimu2")
        ax_ssim.grid(True, alpha=0.25)
        if ssim_plotted:
            ax_ssim.legend(loc="best")

    for ax in axes:
        draw_chapter_boundaries(ax, chapters)

    axes[-1].set_xlabel("Frame")
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_pass_analytics(group: SourceGroup, pass_name: str) -> None:
    mode = pass_name.strip().lower()
    if mode not in ("fastpass", "mainpass"):
        raise RuntimeError(f"Unsupported pass: {pass_name}")

    video_dir = group.workdir / "video"
    pass_dir = video_dir / mode
    encode_dir = pass_dir / "encode"
    if not pass_dir.exists():
        raise RuntimeError(f"Pass directory not found: {pass_dir}")
    if not encode_dir.exists():
        raise RuntimeError(f"Encode directory not found: {encode_dir}")

    scenes_path, scenes = load_scenes_for_analytics(group.workdir)
    chunks_by_index = load_pass_chunks(pass_dir)
    ivf_by_index = collect_ivf_files_by_index(encode_dir)
    if not ivf_by_index:
        raise RuntimeError(f"No numeric IVF files found in {encode_dir}")

    fps_default: Optional[float] = None
    for idx in sorted(chunks_by_index.keys()):
        fps_default = parse_frame_rate_value(chunks_by_index[idx].get("frame_rate"))
        if fps_default:
            break
    if fps_default is None:
        fps_default = get_source_fps(group.source)
    if fps_default is None:
        raise RuntimeError("Unable to determine FPS from chunks.json or ffprobe.")

    records: List[Dict[str, Any]] = []
    for idx in sorted(ivf_by_index.keys()):
        scene = scenes[idx] if 0 <= idx < len(scenes) else None
        ch = chunks_by_index.get(idx)

        st: Optional[int] = None
        en: Optional[int] = None
        if isinstance(ch, dict):
            try:
                st = int(ch.get("start_frame"))
                en = int(ch.get("end_frame"))
            except Exception:
                st, en = None, None
        if (st is None or en is None) and isinstance(scene, dict):
            try:
                st = int(scene.get("start_frame"))
                en = int(scene.get("end_frame"))
            except Exception:
                st, en = None, None
        if st is None or en is None or en <= st:
            print(f"[skip] cannot determine frame range for {mode} index={idx}")
            continue

        fps_scene = parse_frame_rate_value(ch.get("frame_rate") if isinstance(ch, dict) else None) or fps_default
        if fps_scene <= 0:
            print(f"[skip] invalid fps for {mode} index={idx}")
            continue

        duration = (en - st) / fps_scene
        if duration <= 0:
            print(f"[skip] invalid duration for {mode} index={idx}")
            continue

        ivf_path = ivf_by_index[idx]
        try:
            size_bytes = ivf_path.stat().st_size
        except Exception:
            print(f"[skip] failed to stat IVF: {ivf_path}")
            continue
        kbps = (size_bytes * 8.0) / duration / 1000.0

        records.append(
            {
                "index": idx,
                "start_frame": st,
                "end_frame": en,
                "starttime": frame_to_timecode(st, fps_scene),
                "crf": scene_crf(scene),
                "kbps": kbps,
                "duration": duration,
                "chapter": "-",
                "ssimu_scene": None,
                "ssimu_p5": None,
            }
        )

    if not records:
        raise RuntimeError(f"No valid IVF scene records collected for {mode}")

    ssimu_skip: Optional[int] = None
    ssimu_scores: Optional[List[float]] = None
    if mode == "fastpass":
        ssimu_log = pass_dir / f"{group.base}_ssimu2.log"
        if not ssimu_log.exists():
            raise RuntimeError(f"Required SSIMU2 log not found: {ssimu_log}")
        ssimu_skip, ssimu_scores = parse_ssimu2_log_file(ssimu_log)
        for rec in records:
            st = int(rec["start_frame"])
            en = int(rec["end_frame"])
            scene_scores = slice_metric_samples(ssimu_scores, st, en, ssimu_skip)
            if not scene_scores:
                continue
            avg, p5 = calc_avg_p5(scene_scores)
            rec["ssimu_scene"] = avg
            rec["ssimu_p5"] = p5

    total_frames = max(int(r["end_frame"]) for r in records)
    chapters_xml = group.workdir / "chapters" / "chapters.xml"
    chapters: List[Dict[str, Any]] = []
    if chapters_xml.exists():
        chapters_raw = load_chapters_xml(chapters_xml)
        chapters = build_chapter_frame_ranges(chapters_raw, fps_default, total_frames)
        if chapters:
            for rec in records:
                rec["chapter"] = chapter_name_for_frame(int(rec["start_frame"]), chapters)

    analytics_dir = group.workdir / f"{mode}-analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)

    ivf_rows: List[List[str]] = []
    for rec in records:
        if mode == "fastpass":
            ivf_rows.append(
                [
                    str(rec["index"]),
                    str(rec["starttime"]),
                    format_float(rec.get("crf"), 2),
                    format_float(rec.get("kbps"), 2),
                    format_float(rec.get("ssimu_scene"), 4),
                    format_float(rec.get("ssimu_p5"), 4),
                    format_float(rec.get("duration"), 3),
                    str(rec.get("chapter") or "-"),
                ]
            )
        else:
            ivf_rows.append(
                [
                    str(rec["index"]),
                    str(rec["starttime"]),
                    format_float(rec.get("crf"), 2),
                    format_float(rec.get("kbps"), 2),
                    format_float(rec.get("duration"), 3),
                    str(rec.get("chapter") or "-"),
                ]
            )

    ivf_info_path = analytics_dir / "ivf-info.csv"
    ivf_info_path.write_text(render_aligned_pipe_table(ivf_rows), encoding="utf-8", newline="\n")
    print(f"[write] {ivf_info_path}")

    chapters_info_path = analytics_dir / "chapters-info.csv"
    if chapters:
        chapter_rows: List[List[str]] = []
        for ch in chapters:
            st = int(ch["start_frame"])
            en = int(ch["end_frame"])
            in_chapter = [r for r in records if int(r["start_frame"]) >= st and int(r["start_frame"]) < en]
            if not in_chapter:
                continue
            first_idx = int(in_chapter[0]["index"])
            last_idx = int(in_chapter[-1]["index"])
            avg_crf = mean_or_none([r.get("crf") for r in in_chapter])
            avg_kbps = mean_or_none([r.get("kbps") for r in in_chapter])

            if mode == "fastpass":
                ch_scores = slice_metric_samples_strict(ssimu_scores or [], st, en, ssimu_skip or 1)
                ch_avg: Optional[float] = None
                ch_p5: Optional[float] = None
                if ch_scores:
                    ch_avg, ch_p5 = calc_avg_p5(ch_scores)
                chapter_rows.append(
                    [
                        str(ch["name"]),
                        f"{first_idx}-{last_idx}",
                        format_float(avg_crf, 2),
                        format_float(avg_kbps, 2),
                        format_float(ch_avg, 4),
                        format_float(ch_p5, 4),
                    ]
                )
            else:
                chapter_rows.append(
                    [
                        str(ch["name"]),
                        f"{first_idx}-{last_idx}",
                        format_float(avg_crf, 2),
                        format_float(avg_kbps, 2),
                    ]
                )

        if chapter_rows:
            chapters_info_path.write_text(render_aligned_pipe_table(chapter_rows), encoding="utf-8", newline="\n")
            print(f"[write] {chapters_info_path}")
    elif chapters_info_path.exists():
        remove_path(chapters_info_path)

    info_png_path = analytics_dir / "info.png"
    plot_title = f"{group.base} | {mode} | scenes={len(records)} | source={scenes_path.name}"
    write_info_plot(
        out_path=info_png_path,
        title=plot_title,
        pass_name=mode,
        records=records,
        chapters=chapters,
        ssimu_skip=ssimu_skip,
        ssimu_scores=ssimu_scores,
    )
    print(f"[write] {info_png_path}")


def print_menu() -> None:
    print()
    print("Menu:")
    print("  1) Clear Run")
    print("  2) Make Web MP4")
    print("  3) Clear Stage")
    print("  4) Full Clear")
    print("  5) Config Dump")
    print("  6) Verify")
    print("  7) Pass Analytics")
    print("  8) Expand")
    print("  9) Edit")
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


def print_pass_analytics_menu() -> None:
    print()
    print("Pass Analytics:")
    print("  1) Fastpass")
    print("  2) Mainpass")
    print("  3) Both")
    print("  B) Back")


def print_help() -> None:
    print(
        "Usage:\n"
        "  python utils/batch-manager.py [--check-filters] [--check-params] [path ...]\n"
        "\n"
        "Accepted paths:\n"
        "  source files, source directories, workdirs, file .plan, batch .plan,\n"
        "  local runner.bat, local Batch Manager.bat"
    )


def main(argv: List[str]) -> int:
    check_filters = True
    check_params = False
    paths: List[str] = []
    for a in argv:
        if a in ("-h", "--help", "help"):
            print_help()
            return 0
        if a == "--check-filters":
            check_filters = True
            continue
        elif a == "--check-params":
            check_params = True
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

    refresh_support_files_for_groups(groups)

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
        elif choice == "7":
            while True:
                print_pass_analytics_menu()
                c3 = input("Select: ").strip().lower()
                if c3 in ("b", "back", ""):
                    break
                if c3 == "1":
                    selected = ["fastpass"]
                elif c3 == "2":
                    selected = ["mainpass"]
                elif c3 == "3":
                    selected = ["fastpass", "mainpass"]
                else:
                    print("Unknown option.")
                    continue

                for g in groups:
                    for mode in selected:
                        print()
                        print(f"[Pass Analytics: {mode}] {g.base}")
                        try:
                            run_pass_analytics(g, mode)
                        except Exception as e:
                            print(f"[err] pass analytics failed for {g.base} ({mode}): {e}")
        elif choice == "8":
            expand_configs_for_new_sources(groups)
        elif choice == "9":
            edit_config_lines(groups)
        else:
            print("Unknown option.")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
