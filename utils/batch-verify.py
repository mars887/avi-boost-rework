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


NULL_DEVICE = "NUL" if os.name == "nt" else "/dev/null"
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
VIDEO_EXTS = (".mkv", ".mp4")


@dataclass
class BatConfig:
    fastpass_workers: Optional[int]
    mainpass_workers: Optional[int]
    ab_multiplier: Optional[float]
    ab_pos_dev: Optional[float]
    ab_neg_dev: Optional[float]
    quality: Optional[float]
    fastpass: str
    mainpass: str
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


def merge_bat_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    buf = ""
    for raw in lines:
        line = raw.rstrip()
        if line.endswith("^"):
            buf += line[:-1].strip() + " "
            continue
        if buf:
            line = buf + line.strip()
            buf = ""
        out.append(line)
    if buf:
        out.append(buf)
    return out


def parse_bat_vars(lines: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in lines:
        s = raw.strip()
        if not s.lower().startswith("set "):
            continue
        rest = s[4:].strip()
        if rest.startswith('"'):
            rest = rest[1:]
            if rest.endswith('"'):
                rest = rest[:-1]
        if "=" not in rest:
            continue
        key, val = rest.split("=", 1)
        out[key.strip()] = val
    return out


def extract_quality_from_lines(lines: List[str]) -> Optional[str]:
    for line in merge_bat_lines(lines):
        if "--quality" not in line:
            continue
        try:
            tokens = shlex.split(line, posix=False)
        except Exception:
            continue
        for i, tok in enumerate(tokens):
            if tok.lower() == "--quality" and i + 1 < len(tokens):
                return tokens[i + 1].strip("\"")
    return None


def parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except Exception:
        return None


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def is_multiple_of(value: float, step: float) -> bool:
    if step <= 0:
        return False
    return abs(round(value / step) * step - value) <= 1e-6


def validate_bat(path: Path) -> Tuple[Optional[BatConfig], List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if not path.exists():
        errors.append(f"per-file bat missing: {path}")
        return None, errors, warnings

    text = read_text_guess(path)
    if CYRILLIC_RE.search(text):
        errors.append("per-file bat contains cyrillic characters")

    lines = text.splitlines()
    vars_map = parse_bat_vars(lines)

    workers_fallback = parse_int(vars_map.get("WORKERS"))
    fastpass_workers = parse_int(vars_map.get("FASTPASS_WORKERS")) or workers_fallback
    mainpass_workers = parse_int(vars_map.get("MAINPASS_WORKERS")) or workers_fallback
    cores = os.cpu_count() or 1
    if fastpass_workers is None:
        errors.append("FASTPASS_WORKERS missing or invalid")
    elif fastpass_workers < 1 or fastpass_workers > cores:
        errors.append(f"FASTPASS_WORKERS out of range: {fastpass_workers} (1..{cores})")
    if mainpass_workers is None:
        errors.append("MAINPASS_WORKERS missing or invalid")
    elif mainpass_workers < 1 or mainpass_workers > cores:
        errors.append(f"MAINPASS_WORKERS out of range: {mainpass_workers} (1..{cores})")

    ab_multiplier = parse_float(vars_map.get("AB_MULTIPIER"))
    if ab_multiplier is not None:
        if not (0.0 < ab_multiplier <= 5.0):
            errors.append(f"AB_MULTIPIER out of range: {ab_multiplier} (0..5]")

    ab_pos = parse_float(vars_map.get("AB_POS_DEV"))
    if ab_pos is None:
        errors.append("AB_POS_DEV missing or invalid")
    elif not (0.0 <= ab_pos <= 20.0) or not is_multiple_of(ab_pos, 0.25):
        errors.append(f"AB_POS_DEV out of range or not multiple of 0.25: {ab_pos}")

    ab_neg = parse_float(vars_map.get("AB_NEG_DEV"))
    if ab_neg is None:
        errors.append("AB_NEG_DEV missing or invalid")
    elif not (0.0 <= ab_neg <= 20.0) or not is_multiple_of(ab_neg, 0.25):
        errors.append(f"AB_NEG_DEV out of range or not multiple of 0.25: {ab_neg}")

    ab_pos_mult = parse_float(vars_map.get("AB_POS_MULT"))
    ab_neg_mult = parse_float(vars_map.get("AB_NEG_MULT"))
    if ab_multiplier is None:
        if (ab_pos_mult is None) != (ab_neg_mult is None):
            errors.append("AB_POS_MULT/AB_NEG_MULT must be set together when AB_MULTIPIER is empty")
        else:
            if ab_pos_mult is not None and ab_pos_mult <= 0:
                errors.append(f"AB_POS_MULT out of range: {ab_pos_mult} (must be > 0)")
            if ab_neg_mult is not None and ab_neg_mult <= 0:
                errors.append(f"AB_NEG_MULT out of range: {ab_neg_mult} (must be > 0)")
    else:
        if ab_pos_mult is not None and ab_neg_mult is not None:
            if ab_pos_mult <= 0:
                errors.append(f"AB_POS_MULT out of range: {ab_pos_mult} (must be > 0)")
            if ab_neg_mult <= 0:
                errors.append(f"AB_NEG_MULT out of range: {ab_neg_mult} (must be > 0)")

    quality_str = vars_map.get("QUALITY") or extract_quality_from_lines(lines)
    quality = parse_float(quality_str)
    if quality is None:
        errors.append("QUALITY missing or invalid (not found in bat)")
    elif not (1.0 <= quality <= 70.0) or not is_multiple_of(quality, 0.25):
        errors.append(f"QUALITY out of range or not multiple of 0.25: {quality}")

    fastpass = vars_map.get("FASTPASS", "")
    mainpass = vars_map.get("MAINPASS", "")
    fastpass_vf = vars_map.get("FASTPASS_VF", "")
    mainpass_vf = vars_map.get("MAINPASS_VF", "")

    cfg = BatConfig(
        fastpass_workers=fastpass_workers,
        mainpass_workers=mainpass_workers,
        ab_multiplier=ab_multiplier,
        ab_pos_dev=ab_pos,
        ab_neg_dev=ab_neg,
        quality=quality,
        fastpass=fastpass,
        mainpass=mainpass,
        fastpass_vf=fastpass_vf,
        mainpass_vf=mainpass_vf,
    )
    return cfg, errors, warnings


def validate_tracks_json(obj: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    tracks = obj.get("tracks")
    if not isinstance(tracks, list):
        errors.append("tracks.json: missing or invalid 'tracks' list")
        return errors
    for i, t in enumerate(tracks):
        if not isinstance(t, dict):
            errors.append(f"tracks.json: track[{i}] not an object")
            continue
        if "trackId" not in t:
            errors.append(f"tracks.json: track[{i}] missing trackId")
        if "type" not in t:
            errors.append(f"tracks.json: track[{i}] missing type")
        if "trackStatus" not in t:
            errors.append(f"tracks.json: track[{i}] missing trackStatus")
    return errors


def check_zone_syntax(zone_path: Path, source: Path) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    if not zone_path.exists():
        warnings.append(f"zone_edit_command.txt missing: {zone_path}")
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
    for i, t in enumerate(tokens):
        if t in ("-", "!", "^"):
            sep_idx = i
            break
    if sep_idx is None:
        raise ValueError("missing separator (- ! ^)")
    action_tokens = tokens[sep_idx + 1 :]
    if len(action_tokens) % 2 != 0:
        raise ValueError("actions must be param/value pairs")
    out: List[str] = []
    for i in range(0, len(action_tokens), 2):
        k = normalize_param_key(action_tokens[i])
        v = action_tokens[i + 1]
        out.extend([k, v])
    return out


def is_param_key(tok: str) -> bool:
    t = tok.strip()
    return t.startswith("--") or t.startswith("-")


def apply_override(base_tokens: List[str], override_tokens: List[str]) -> List[str]:
    i = 0
    while i < len(override_tokens):
        tok = override_tokens[i]
        if not is_param_key(tok):
            i += 1
            continue

        key = tok
        has_val = (i + 1 < len(override_tokens)) and (not is_param_key(override_tokens[i + 1]))
        val = override_tokens[i + 1] if has_val else None

        # Find last occurrence in base
        loc = None
        for j in range(len(base_tokens) - 1, -1, -1):
            if base_tokens[j] == key:
                base_has_val = (j + 1 < len(base_tokens)) and (not is_param_key(base_tokens[j + 1]))
                loc = (j, base_has_val)
                break

        if loc is None:
            base_tokens.append(key)
            if val is not None:
                base_tokens.append(val)
        else:
            k_idx, base_has_val = loc
            if val is None:
                if base_has_val:
                    del base_tokens[k_idx + 1]
            else:
                if base_has_val:
                    base_tokens[k_idx + 1] = val
                else:
                    base_tokens.insert(k_idx + 1, val)

        i += 2 if has_val else 1

    return base_tokens


def load_zone_lines(path: Path) -> List[str]:
    text = read_text_guess(path)
    lines = []
    for raw in text.splitlines():
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
    ap.add_argument("--source", required=True)
    ap.add_argument("--workdir", default="")
    ap.add_argument("--per-file-bat", default="")
    ap.add_argument("--result-mkv", default="")
    ap.add_argument("--result-mp4", default="")
    ap.add_argument("--check-filters", action="store_true")
    ap.add_argument("--check-params", action="store_true")
    args = ap.parse_args(argv)

    errors: List[str] = []
    warnings: List[str] = []

    source = Path(args.source)
    if not source.exists():
        errors.append(f"source missing: {source}")

    bat_cfg: Optional[BatConfig] = None
    if args.per_file_bat:
        bat_cfg, errs, warns = validate_bat(Path(args.per_file_bat))
        errors.extend(errs)
        warnings.extend(warns)

    if args.workdir:
        workdir = Path(args.workdir)
        if not workdir.exists():
            warnings.append(f"workdir missing: {workdir}")
        tracks_path = workdir / "tracks.json"
        if tracks_path.exists():
            try:
                obj = load_json(tracks_path)
                errors.extend(validate_tracks_json(obj))
            except Exception as exc:
                errors.append(str(exc))
        else:
            warnings.append(f"tracks.json missing: {tracks_path}")

        zone_path = workdir / "zone_edit_command.txt"
        if source.exists():
            z_errs, z_warns = check_zone_syntax(zone_path, source)
            errors.extend(z_errs)
            warnings.extend(z_warns)
        elif zone_path.exists():
            warnings.append("zone syntax skipped: source missing")

        if args.check_filters and source.exists() and bat_cfg is not None:
            if bat_cfg.fastpass_vf.strip():
                msg = check_filter(source, bat_cfg.fastpass_vf)
                if msg:
                    if "skipped" in msg:
                        warnings.append(f"FASTPASS_VF: {msg}")
                    else:
                        errors.append(f"FASTPASS_VF: {msg}")
            if bat_cfg.mainpass_vf.strip():
                msg = check_filter(source, bat_cfg.mainpass_vf)
                if msg:
                    if "skipped" in msg:
                        warnings.append(f"MAINPASS_VF: {msg}")
                    else:
                        errors.append(f"MAINPASS_VF: {msg}")

        if args.check_params and source.exists() and bat_cfg is not None:
            fast_tokens = parse_param_tokens(bat_cfg.fastpass)
            main_tokens = parse_param_tokens(bat_cfg.mainpass)

            ok, msg = run_param_check(source, fast_tokens)
            if not ok:
                errors.append(f"FASTPASS params: {msg}")
            elif msg != "ok":
                warnings.append(f"FASTPASS params: {msg}")

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
