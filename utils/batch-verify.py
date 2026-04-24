#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.param_utils import apply_override
from utils.plan_model import format_value, normalize_track_type, resolve_file_plan
from utils.crop_resize import load_crop_resize_plan, validate_crop_resize_plan
from utils.zoned_commands import ZONED_COMMAND_NAME, project_zone_command_lines


NULL_DEVICE = "NUL" if os.name == "nt" else "/dev/null"
VIDEO_EXTS = (".mkv", ".mp4")


@dataclass
class PipelineConfig:
    fastpass_workers: Optional[int]
    mainpass_workers: Optional[int]
    ab_multiplier: Optional[float]
    ab_pos_dev: Optional[float]
    ab_neg_dev: Optional[float]
    quality: Optional[float]
    fastpass_preset: str
    fastpass: str
    mainpass: str
    encode: str
    fastpass_vf: str
    mainpass_vf: str


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def read_text_guess(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp1251", errors="replace")


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid_json:{path}: {exc}")


def validate_resolved_plan_tracks(tracks: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    if not isinstance(tracks, list):
        return ["plan: invalid track list"]
    for i, t in enumerate(tracks):
        if not isinstance(t, dict):
            errors.append(f"plan: track[{i}] not an object")
            continue
        if "trackId" not in t:
            errors.append(f"plan: track[{i}] missing trackId")
        if "type" not in t:
            errors.append(f"plan: track[{i}] missing type")
        if "trackStatus" not in t:
            errors.append(f"plan: track[{i}] missing trackStatus")
    return errors


def build_plan_cfg(resolved_plan: Any) -> PipelineConfig:
    primary = resolved_plan.plan.video.primary
    details = resolved_plan.plan.video.details
    return PipelineConfig(
        fastpass_workers=int(primary.fastpass_workers),
        mainpass_workers=int(primary.mainpass_workers),
        ab_multiplier=float(primary.ab_multiplier),
        ab_pos_dev=float(primary.ab_pos_dev),
        ab_neg_dev=float(primary.ab_neg_dev),
        quality=float(primary.quality),
        fastpass_preset=str(primary.fastpass_preset or ""),
        fastpass=resolved_plan.build_fastpass_params_text(),
        mainpass=resolved_plan.build_mainpass_params_text(),
        encode=str(resolved_plan.build_encode_params_text() or ""),
        fastpass_vf=str(details.fastpass_filter or ""),
        mainpass_vf=str(details.mainpass_filter or ""),
    )


def check_zone_syntax(zone_path: Path, source: Path) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    if not zone_path.exists():
        warnings.append(f"{ZONED_COMMAND_NAME} missing: {zone_path}")
        return errors, warnings

    zone_editor = Path(__file__).with_name("zone-editor.py")
    if not zone_editor.exists():
        warnings.append(f"zone-editor.py missing: {zone_editor}")
        return errors, warnings

    cmd = [
        sys.executable,
        str(zone_editor),
        "--parse-check",
        "--source", str(source),
        "--command", str(zone_path),
        "--no-vfr-warn",
    ]
    print("[cmd]", " ".join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        errors.append(f"zone syntax check failed (code={p.returncode})")
    return errors, warnings


def check_crop_resize_syntax(crop_path: Path, source: Path, enabled: bool) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    if not crop_path.exists():
        if enabled:
            warnings.append(f"{ZONED_COMMAND_NAME} missing: {crop_path}")
        return errors, warnings
    text = read_text_guess(crop_path)
    if not text.strip():
        return errors, warnings
    try:
        plan = load_crop_resize_plan(crop_path)
        validate_crop_resize_plan(plan, source=source if source.exists() else None)
    except Exception as exc:
        errors.append(f"crop/resize syntax failed: {exc}")
    return errors, warnings


def parse_param_tokens(value: str) -> List[str]:
    v = (value or "").strip()
    if not v:
        return []
    return shlex.split(v, posix=False)


def normalize_param_key(tok: str) -> str:
    t = tok.strip()
    if t.startswith("-"):
        return t
    return "--" + t


def parse_zone_actions(line: str) -> List[str]:
    tokens = shlex.split(line, posix=False)
    if not tokens:
        return []
    sep_idx = None
    if "|" in tokens:
        first_bar = tokens.index("|")
        try:
            second_bar = tokens.index("|", first_bar + 1)
        except ValueError:
            raise ValueError("missing second boundary separator (|)")
        sep_idx = second_bar
    else:
        for i, t in enumerate(tokens):
            if t in ("-", "!", "^"):
                sep_idx = i
                break
    if sep_idx is None:
        raise ValueError("missing separator (- ! ^ |)")
    action_tokens = tokens[sep_idx + 1 :]
    if len(action_tokens) % 2 != 0:
        raise ValueError("actions must be param/value pairs")
    out: List[str] = []
    for i in range(0, len(action_tokens), 2):
        k = normalize_param_key(action_tokens[i])
        v = action_tokens[i + 1]
        out.extend([k, v])
    return out


def load_zone_lines(path: Path) -> List[str]:
    text = read_text_guess(path)
    lines = []
    for raw in project_zone_command_lines(text.splitlines()):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        lines.append(line)
    return lines


def check_filter(source: Path, filter_str: str) -> Optional[str]:
    if not filter_str.strip():
        return None
    if not shutil.which("ffmpeg"):
        return "ffmpeg not found; filter check skipped"
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(source),
        "-frames:v", "5",
        "-vf", filter_str,
        "-f", "null",
        "-",
    ]
    print("[cmd]", " ".join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        return f"filter check failed (code={p.returncode})"
    return None


def run_param_check(source: Path, params: List[str]) -> Tuple[bool, str]:
    if not params:
        return True, "empty params"
    if not shutil.which("ffmpeg"):
        return True, "ffmpeg not found; param check skipped"
    if not shutil.which("SvtAv1EncApp"):
        return True, "SvtAv1EncApp not found; param check skipped"

    ff_cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(source),
        "-frames:v", "5",
        "-f", "yuv4mpegpipe",
        "-",
    ]
    enc_cmd = ["SvtAv1EncApp", "-i", "-"] + params + ["-o", NULL_DEVICE]
    print("[cmd]", " ".join(ff_cmd) + " | " + " ".join(enc_cmd))

    ff = subprocess.Popen(
        ff_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert ff.stdout is not None
    enc = subprocess.Popen(
        enc_cmd,
        stdin=ff.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    ff.stdout.close()

    enc_out, enc_err = enc.communicate()
    ff_err = ""
    if ff.stderr is not None:
        ff_err = ff.stderr.read()
    ff_rc = ff.wait()

    if ff_rc != 0:
        return False, (ff_err.strip() or f"ffmpeg failed rc={ff_rc}")
    if enc.returncode != 0:
        err = enc_err.strip() or enc_out.strip() or f"encoder failed rc={enc.returncode}"
        return False, err
    return True, "ok"


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Batch config verifier")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--result-mkv", default="")
    ap.add_argument("--result-mp4", default="")
    ap.add_argument("--check-filters", action="store_true")
    ap.add_argument("--check-params", action="store_true")
    args = ap.parse_args(argv)

    errors: List[str] = []
    warnings: List[str] = []

    resolved_plan = resolve_file_plan(Path(args.plan))
    source = resolved_plan.paths.source
    workdir = resolved_plan.paths.workdir
    plan_cfg = build_plan_cfg(resolved_plan)
    errors.extend(validate_resolved_plan_tracks(resolved_plan.runtime_tracks()))
    zone_path = resolved_plan.paths.zone_file
    crop_path = resolved_plan.paths.crop_resize_file

    if not source.exists():
        errors.append(f"source missing: {source}")

    if not workdir.exists():
        warnings.append(f"workdir missing: {workdir}")

    if source.exists():
        z_errs, z_warns = check_zone_syntax(zone_path, source)
        errors.extend(z_errs)
        warnings.extend(z_warns)
        if crop_path != zone_path or crop_path.exists():
            c_errs, c_warns = check_crop_resize_syntax(
                crop_path,
                source,
                bool(resolved_plan.plan.video.experimental.crop_resize_enabled),
            )
            errors.extend(c_errs)
            warnings.extend(c_warns)
    else:
        if zone_path.exists():
            warnings.append("zone syntax skipped: source missing")
        if crop_path.exists() and crop_path != zone_path:
            warnings.append("crop/resize syntax skipped: source missing")

    if args.check_filters and source.exists():
        if plan_cfg.fastpass_vf.strip():
            msg = check_filter(source, plan_cfg.fastpass_vf)
            if msg:
                if "skipped" in msg:
                    warnings.append(f"FASTPASS_VF: {msg}")
                else:
                    errors.append(f"FASTPASS_VF: {msg}")
        if plan_cfg.mainpass_vf.strip():
            msg = check_filter(source, plan_cfg.mainpass_vf)
            if msg:
                if "skipped" in msg:
                    warnings.append(f"MAINPASS_VF: {msg}")
                else:
                    errors.append(f"MAINPASS_VF: {msg}")

    if args.check_params and source.exists():
        fast_tokens = parse_param_tokens(plan_cfg.fastpass)
        if plan_cfg.quality is not None:
            fast_tokens.extend(["--crf", format_value(plan_cfg.quality)])
        if plan_cfg.fastpass_preset.strip():
            fast_tokens.extend(["--preset", plan_cfg.fastpass_preset.strip()])

        ok, msg = run_param_check(source, fast_tokens)
        if not ok:
            errors.append(f"FASTPASS params: {msg}")
        elif msg != "ok":
            warnings.append(f"FASTPASS params: {msg}")

        combined = parse_param_tokens(plan_cfg.encode)
        if not combined:
            main_tokens = parse_param_tokens(plan_cfg.mainpass)
            combined = apply_override(list(fast_tokens), main_tokens)
        ok, msg = run_param_check(source, combined)
        if not ok:
            errors.append(f"MAINPASS overlay params: {msg}")
        elif msg != "ok":
            warnings.append(f"MAINPASS overlay params: {msg}")

        if zone_path.exists():
            for line in load_zone_lines(zone_path):
                try:
                    zone_tokens = parse_zone_actions(line)
                except Exception as exc:
                    errors.append(f"zone line parse failed: {line}: {exc}")
                    continue
                merged = apply_override(list(combined), zone_tokens)
                ok, msg = run_param_check(source, merged)
                if not ok:
                    errors.append(f"zone params failed: {line}: {msg}")
                elif msg != "ok":
                    warnings.append(f"zone params skipped: {line}: {msg}")

    if warnings:
        print("[verify] warnings:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        eprint("[verify] errors:")
        for e in errors:
            eprint(f"  - {e}")
        return 2

    print("[verify] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
