#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto-boost 3.0 - refactored into modules.
Entry-point CLI that wires the pipeline stages together.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from ab_fastpass import run_fastpass_av1an
from ab_fs import ensure_dir, ensure_exists, load_json, save_json, safe_unlink, touch, which_or_none
from ab_logging import eprint, setup_logging
from ab_nvof import (
    build_nvof_schedule,
    is_nvof_metric,
    load_nvof_csv,
    needed_bases_for_metrics,
    run_nvof_script,
)
from ab_psd import run_psd
from ab_rules import (
    RuleMetricsState,
    apply_rules_to_scenes,
    check_rules_prereqs,
    collect_rule_requirements,
    extract_metric_calls,
)
from ab_scene_ops import apply_avg_func, compute_chunk_5p_single_metric
from ab_scenes_io import (
    apply_crf_adjustments_to_scenes,
    sanitize_scenes_json,
    scenes_to_ranges,
    write_base_scenes_from_av1an,
    write_uniform_scenes,
)
from ab_ssimu2 import calculate_ssimu2
from ab_state import (
    is_valid_base_scenes,
    is_valid_final_scenes,
    is_valid_ssimu2_log,
    marker_paths,
)

def main() -> int:
    """CLI entry point that orchestrates stages and resume logic."""
    parser = argparse.ArgumentParser(description="auto-boost 2.8.json + av1an fastpass + per-scene CRF zones + resume.")
    parser.add_argument("-s", "--stage", type=int, default=0,
                        help="Stage: 0=all, 1=PSD scenes (sdm=psd), 2=fastpass, 3=metrics, 4=write base scenes.json, 5=apply rules.")
    parser.add_argument("-i", "--input", required=True, help="Input video file (source).")
    parser.add_argument("-t", "--temp", default=None,
                        help="Project directory. Default: <input stem>_autoboost next to input.")
    parser.add_argument("--log", default="",
                        help="Optional log file path (relative to --temp if not absolute).")
    parser.add_argument("--force", action="store_true",
                        help="Start from scratch: clear resume markers and remove generated artifacts in the project dir.")
    parser.add_argument("--sdm", choices=["psd", "av1an"], default="psd",
                        help="Scene detection mode: psd (default, run PSD and pass --scenes) or av1an (use av1an internal detection).")
    parser.add_argument("--no-fastpass", action="store_true",
                        help="Skip fast-pass and metrics; write uniform scenes with base CRF.")
    parser.add_argument("--stop-before-stage4", "--fastpass-only", action="store_true",
                        help="Run stages 1-3, then exit before Stage 4 (useful for two-pass batch).")

    # PSD
    parser.add_argument("--psd-script", default="Progressive-Scene-Detection.py",
                        help="Path to Progressive-Scene-Detection.py (PSD). Default: Progressive-Scene-Detection.py (cwd).")
    parser.add_argument("--psd-python", default=None,
                        help="Python executable for PSD. Default: current Python.")
    parser.add_argument("--psd-args", default="",
                        help="Extra arguments appended to PSD invocation (string, shlex-split).")
    parser.add_argument("--base-scenes", default=None,
                        help="Use an existing PSD scenes.json instead of running PSD (zone_overrides must be null; --sdm psd only).")
    parser.add_argument("--out-scenes", default=None,
                        help="Final scenes.json output path. Default: <project>/scenes.final.json")

    # av1an fastpass
    parser.add_argument("--fastpass-out", default=None,
                        help="Fast-pass output file. Default: <project>/<stem>.fastpass.mkv")
    parser.add_argument("--fastpass-vpy", default=None,
                        help="Optional .vpy path for fast-pass input (used for av1an -i). Adds --vspipe-args src=<input>.")
    parser.add_argument("--fastpass-proxy", default=None,
                        help="Optional av1an --proxy path for fast-pass. Adds --proxy and --vspipe-args src=<input>.")
    parser.add_argument("--workers", type=int, default=8,
                        help="av1an workers for fast-pass.")
    parser.add_argument("--lp", type=int, default=3, help="--lp for fast-pass encoding (svt-av1).")
    parser.add_argument("-q", "--quality", type=float, default=30.0, help="Base CRF for targeting (also used for fast-pass).")
    parser.add_argument("--fast-preset", type=int, default=7, help="Fast-pass SVT preset (speed).")
    parser.add_argument("-p", "--preset", type=int, default=2, help="Final SVT preset embedded into zone_overrides.")
    parser.add_argument("-v", "--video-params", default="",
                        help="Extra SVT encoder params to embed into zone_overrides (string). Do NOT include --crf/--preset.")
    parser.add_argument("--final-override", default="",
                        help="Overriding params from video-params for final encoding scenes.")
    parser.add_argument("-f", "--ffmpeg", default="",
                        help="FFmpeg options for av1an. If it starts with '-', passed as-is to av1an -f. Otherwise treated as a filtergraph and wrapped as -vf \"...\".")
    parser.add_argument("--verbose", action="store_true", help="Verbose output (fast-pass + rules logging).")
    parser.add_argument("--keep", action="store_true", help="Pass --keep to av1an (keep temp files).")
    parser.add_argument("--av1an-log-file", default="auto",
                        help="av1an --log-file path. 'auto' => <project>/av1an.log, 'none' => disable.")
    parser.add_argument("--av1an-log-level", default="info",
                        help="av1an --log-level (e.g. info, debug, warn).")

    # Rules
    rules_group = parser.add_mutually_exclusive_group()
    rules_group.add_argument("--rules", default=None, help="Path to python rule script.")
    rules_group.add_argument("--rules-inline", default=None, help="Inline python rules string.")
    parser.add_argument("--rules-required-metrics", nargs="*", default=[],
                        help="Space-separated list of metric names required by rules (fail-fast).")
    parser.add_argument("--rule-test", action="store_true",
                        help="Execute rules without saving results; implies --verbose.")

    # NVOF metrics (external script integration)
    default_nvof_script = Path(__file__).resolve().parent.parent / "utils" / "nvof_noise_est_opt_main.py"
    parser.add_argument("--nvof-script", default=str(default_nvof_script),
                        help="Path to nvof_noise_est_opt_main.py.")
    parser.add_argument("--nvof-args", default="",
                        help="Extra args passed to NVOF metrics script.")
    parser.add_argument("--nvof-csv", default="auto",
                        help="NVOF CSV output path (auto => <project>/metrics/<stem>_nvof.csv).")
    parser.add_argument("--nvof-schedule", default="auto",
                        help="NVOF schedule JSON path (auto => <project>/metrics/nvof_schedule.json).")

    # Metrics (SSIMU2)
    parser.add_argument("-S", "--skip", type=int, default=2, help="SSIMU2 sampling step (VapourSynth SelectEvery).")

    # SSIMU2 backend selection
    parser.add_argument("--ssimu2-backend", default="auto",
                        help="SSIMU2 backend (VapourSynth only): auto (default), vship (preferred), vszip.")
    parser.add_argument("--vs-source", default="ffms2",
                        help="VapourSynth decode provider for SSIMU2 clips: ffms2 (default), bestsource, lsmas, auto.")

    # Zone computation
    parser.add_argument("-d", "--deviation", type=float, default=10.0,
                        help="Max CRF deviation (used for both directions if --max-* not set).")
    parser.add_argument("--max-positive-dev", type=float, default=None,
                        help="Max allowed CRF increase above base (worse quality).")
    parser.add_argument("--max-negative-dev", type=float, default=None,
                        help="Max allowed CRF decrease below base (better quality).")
    parser.add_argument("-a", "--aggressive", type=float, default=None,
                        help="Boosting multiplier (overrides +/- multipliers when specified).")
    parser.add_argument("--pos-dev-multiplier", type=float, default=None,
                        help="Multiplier used when adj is positive (requires --neg-dev-multiplier if set).")
    parser.add_argument("--neg-dev-multiplier", type=float, default=None,
                        help="Multiplier used when adj is negative (requires --pos-dev-multiplier if set).")
    parser.add_argument("--avg-func", default="",
                        help="Adjust metric avg: +N/-N (shift), !N (set), or target%percent (move toward target by percent).")

    args = parser.parse_args()

    if args.rule_test and not args.verbose:
        args.verbose = True
    if args.sdm == "av1an" and args.base_scenes:
        raise RuntimeError("--base-scenes cannot be used with --sdm av1an (av1an generates scenes during stage 2).")

    input_file = Path(args.input).expanduser().resolve()
    ensure_exists(input_file, "Input file")

    project_dir = Path(args.temp).expanduser().resolve() if args.temp else (input_file.parent / f"{input_file.stem}")
    ensure_dir(project_dir)
    setup_logging(args.log, project_dir)

    psd_dir = project_dir / "psd"
    fastpass_dir = project_dir / "fastpass"
    ensure_dir(psd_dir)
    ensure_dir(fastpass_dir)

    av1an_temp = fastpass_dir
    ensure_dir(av1an_temp)

    base_scenes_path = Path(args.base_scenes).expanduser().resolve() if args.base_scenes else (psd_dir / "scenes.psd.json")
    out_scenes_path = Path(args.out_scenes).expanduser().resolve() if args.out_scenes else (project_dir / "scenes.final.json")
    fastpass_out = Path(args.fastpass_out).expanduser().resolve() if args.fastpass_out else (fastpass_dir / f"{input_file.stem}.fastpass.mkv")
    fastpass_vpy = Path(args.fastpass_vpy).expanduser().resolve() if args.fastpass_vpy else None
    fastpass_proxy = Path(args.fastpass_proxy).expanduser().resolve() if args.fastpass_proxy else None

    # Metrics reference: if fast-pass input is a .vpy (may crop/scale), compare against that pipeline, not the original source.
    metrics_ref_src = fastpass_vpy if fastpass_vpy is not None else input_file
    metrics_ref_vpy_src = input_file if fastpass_vpy is not None else None

    ssimu2_log = fastpass_dir / f"{input_file.stem}_ssimu2.log"

    av1an_log_file: Optional[Path]
    av1an_log_arg = str(args.av1an_log_file).strip().lower()
    if av1an_log_arg in ("none", "off", "false", "0"):
        av1an_log_file = None
    elif av1an_log_arg == "auto" or not av1an_log_arg:
        av1an_log_file = fastpass_dir / "av1an.log"
    else:
        av1an_log_file = Path(args.av1an_log_file).expanduser()
        if not av1an_log_file.is_absolute():
            av1an_log_file = project_dir / av1an_log_file

    marks = marker_paths(project_dir)

    rule_name: Optional[str] = None
    compiled_rules: Optional[Any] = None
    required_metrics: List[str] = []
    if args.stage in (0, 5):
        if args.rules or args.rules_inline:
            if args.rules:
                rules_path = Path(args.rules).expanduser().resolve()
                ensure_exists(rules_path, "Rules file")
                rules_text = rules_path.read_text(encoding="utf-8")
                rule_name = str(rules_path)
            else:
                rules_text = str(args.rules_inline)
                rule_name = "<rules-inline>"

            compiled_rules = compile(rules_text, rule_name, "exec")
            required_metrics.extend([m.strip() for m in args.rules_required_metrics if m and m.strip()])
            for m in extract_metric_calls(rules_text):
                if m not in required_metrics:
                    required_metrics.append(m)
        elif args.rules_required_metrics:
            eprint("[warn] --rules-required-metrics specified without --rules/--rules-inline; ignoring.")

    if args.force:
        print("[force] clearing state markers and generated outputs...")
        for mp in marks.values():
            safe_unlink(mp)

        # Remove generated artifacts (do not remove user-provided --base-scenes).
        if not args.base_scenes:
            safe_unlink(base_scenes_path)
        safe_unlink(out_scenes_path)
        safe_unlink(ssimu2_log)
        safe_unlink(fastpass_out)

        # Wipe av1an temp directory unless user wants to keep it explicitly.
        if av1an_temp.exists():
            try:
                shutil.rmtree(av1an_temp)
            except Exception as ex:
                eprint(f"[warn] failed to remove av1an temp dir: {av1an_temp} ({ex})")
        ensure_dir(av1an_temp)

    # Tool sanity checks (warnings only)
    if not which_or_none("av1an"):
        eprint("[warn] 'av1an' not found in PATH. Stages 2+ will fail unless you add it to PATH.")
    # -----------------
    # Stage 1: PSD (scene detection)
    # -----------------
    if args.stage in (0, 1):
        if args.sdm == "av1an":
            if args.stage == 1:
                print("[skip] --sdm av1an uses av1an scene detection during stage 2.")
        else:
            if args.base_scenes:
                print(f"[skip] using existing base scenes: {base_scenes_path}")
                if not is_valid_base_scenes(base_scenes_path):
                    raise RuntimeError("--base-scenes provided but is not a valid scenes.json (or cannot be sanitized).")
                touch(marks["psd"])
            else:
                if marks["psd"].exists() and is_valid_base_scenes(base_scenes_path):
                    print(f"[resume] PSD already completed: {base_scenes_path}")
                else:
                    psd_script = Path(args.psd_script).expanduser()
                    if not psd_script.exists():
                        cand = project_dir / psd_script
                        if cand.exists():
                            psd_script = cand
                    psd_python = Path(args.psd_python).expanduser() if args.psd_python else None
                    run_psd(psd_script=psd_script, psd_python=psd_python, input_file=input_file,
                            base_scenes_path=base_scenes_path, extra_args=args.psd_args)
                    touch(marks["psd"])

    # -----------------
    # Stage 2: fast-pass
    # -----------------
    if args.stage in (0, 2):
        if args.no_fastpass and args.sdm == "psd":
            print("[skip] no-fastpass enabled; fast-pass skipped for sdm=psd.")
        else:
            if args.no_fastpass and args.sdm == "av1an":
                scenes_hint = av1an_temp / "scenes.json"
                if marks["fastpass"].exists() and scenes_hint.exists():
                    print(f"[resume] scene-only already completed: {scenes_hint}")
                else:
                    if fastpass_vpy is not None:
                        ensure_exists(fastpass_vpy, "Fast-pass vpy")
                    if fastpass_proxy is not None:
                        ensure_exists(fastpass_proxy, "Fast-pass proxy")
                    run_fastpass_av1an(
                        input_file=input_file,
                        fastpass_vpy=fastpass_vpy,
                        fastpass_proxy=fastpass_proxy,
                        output_file=fastpass_out,
                        scenes_path=base_scenes_path,
                        av1an_temp=av1an_temp,
                        sdm=str(args.sdm),
                        workers=int(args.workers),
                        lp=int(args.lp),
                        fast_preset=int(args.fast_preset),
                        fast_crf=float(args.quality),
                        video_params=str(args.video_params),
                        ffmpeg_arg=str(args.ffmpeg),
                        verbose=bool(args.verbose),
                        keep=bool(args.keep),
                        sc_only=True,
                        log_file=av1an_log_file,
                        log_level=(
                            None if str(args.av1an_log_level).strip().lower() in ("", "none", "off", "false", "0")
                            else str(args.av1an_log_level).strip()
                        ),
                    )
                    touch(marks["fastpass"])
            else:
                if marks["fastpass"].exists() and fastpass_out.exists() and fastpass_out.stat().st_size > 0:
                    print(f"[resume] fast-pass already completed: {fastpass_out}")
                else:
                    if fastpass_vpy is not None:
                        ensure_exists(fastpass_vpy, "Fast-pass vpy")
                    if fastpass_proxy is not None:
                        ensure_exists(fastpass_proxy, "Fast-pass proxy")
                    run_fastpass_av1an(
                        input_file=input_file,
                        fastpass_vpy=fastpass_vpy,
                        fastpass_proxy=fastpass_proxy,
                        output_file=fastpass_out,
                        scenes_path=base_scenes_path,
                        av1an_temp=av1an_temp,
                        sdm=str(args.sdm),
                        workers=int(args.workers),
                        lp=int(args.lp),
                        fast_preset=int(args.fast_preset),
                        fast_crf=float(args.quality),
                        video_params=str(args.video_params),
                        ffmpeg_arg=str(args.ffmpeg),
                        verbose=bool(args.verbose),
                        keep=bool(args.keep),
                        sc_only=False,
                        log_file=av1an_log_file,
                        log_level=(
                            None if str(args.av1an_log_level).strip().lower() in ("", "none", "off", "false", "0")
                            else str(args.av1an_log_level).strip()
                        ),
                    )
                    touch(marks["fastpass"])

    frames_count = 0
    scene_ranges: List[Tuple[int, int]] = []
    if args.stage in (0, 3, 4, 5):
        if args.sdm == "av1an":
            if base_scenes_path.exists():
                raw = load_json(base_scenes_path)
                norm = sanitize_scenes_json(raw)
                save_json(base_scenes_path, norm)
                print(f"[ok] av1an scenes normalized: {base_scenes_path}")
            else:
                try:
                    write_base_scenes_from_av1an(av1an_temp, base_scenes_path)
                except FileNotFoundError as exc:
                    if not is_valid_base_scenes(base_scenes_path):
                        raise
                    eprint(f"[warn] {exc}. Using existing base scenes: {base_scenes_path}")
        ensure_exists(base_scenes_path, "Base scenes.json")
        base_scenes_obj = sanitize_scenes_json(load_json(base_scenes_path))
        frames_count = int(base_scenes_obj["frames"])
        scene_ranges = scenes_to_ranges(base_scenes_obj)

    # -----------------
    # Stage 3: metrics
    # -----------------
    if args.stage in (0, 3):
        if args.no_fastpass:
            print("[skip] no-fastpass enabled; metrics skipped.")
        else:
            ensure_exists(fastpass_out, "Fast-pass output")

            if marks["ssimu2"].exists() and is_valid_ssimu2_log(ssimu2_log):
                print(f"[resume] SSIMU2 already completed: {ssimu2_log}")
            else:
                calculate_ssimu2(
                    src_file=metrics_ref_src,
                    enc_file=fastpass_out,
                    out_path=ssimu2_log,
                    frames_count=frames_count,
                    skip=int(args.skip),
                    backend=str(args.ssimu2_backend),
                    vs_source=str(args.vs_source),
                    vpy_src=metrics_ref_vpy_src,
                )
                touch(marks["ssimu2"])

    if args.stop_before_stage4:
        print("[stop] requested stop before stage 4; skipping stages 4+.")
        print(f"Base scenes : {base_scenes_path}")
        print(f"Fast-pass   : {fastpass_out}")
        print(f"SSIMU2 log  : {ssimu2_log}")
        return 0

    # -----------------
    # Stage 4: base scenes
    # -----------------
    if args.stage in (0, 4):
        if marks["final"].exists() and is_valid_final_scenes(out_scenes_path):
            print(f"[resume] base scenes already written: {out_scenes_path}")
        else:
            if args.no_fastpass:
                write_uniform_scenes(
                    base_scenes_path=base_scenes_path,
                    out_scenes_path=out_scenes_path,
                    base_crf=float(args.quality),
                    final_preset=int(args.preset),
                    video_params=str(args.video_params),
                    final_override=str(args.final_override),
                )
            else:
                _, per_chunk_5, avg_total = compute_chunk_5p_single_metric(
                    scene_ranges=scene_ranges,
                    ssimu2_path=ssimu2_log,
                )
                
                avg_total_adj = apply_avg_func(avg_total, args.avg_func)
                if str(args.avg_func).strip():
                    print(f"[avg-func] {avg_total} -> {avg_total_adj} ({args.avg_func})")

                aggressive_val = args.aggressive
                pos_val = args.pos_dev_multiplier
                neg_val = args.neg_dev_multiplier
                if aggressive_val is not None:
                    pos_dev_multiplier = float(aggressive_val)
                    neg_dev_multiplier = float(aggressive_val)
                else:
                    if (pos_val is None) != (neg_val is None):
                        raise RuntimeError(
                            "Specify either --aggressive or both --pos-dev-multiplier and --neg-dev-multiplier."
                        )
                    if pos_val is None and neg_val is None:
                        pos_dev_multiplier = 1.0
                        neg_dev_multiplier = 1.0
                    else:
                        pos_dev_multiplier = float(pos_val)
                        neg_dev_multiplier = float(neg_val)

                apply_crf_adjustments_to_scenes(
                    base_scenes_path=base_scenes_path,
                    out_scenes_path=out_scenes_path,
                    scene_ranges=scene_ranges,
                    per_chunk_5=per_chunk_5,
                    avg_total=avg_total_adj,
                    base_crf=float(args.quality),
                    pos_dev_multiplier=pos_dev_multiplier,
                    neg_dev_multiplier=neg_dev_multiplier,
                    deviation=float(args.deviation),
                    max_positive_dev=args.max_positive_dev,
                    max_negative_dev=args.max_negative_dev,
                    final_preset=int(args.preset),
                    video_params=str(args.video_params),
                    final_override=str(args.final_override)
                )
            touch(marks["final"])

    # -----------------
    # Stage 5: apply rules
    # -----------------
    if args.stage in (0, 5):
        if compiled_rules is None:
            if args.stage == 5:
                print("[skip] no rules provided for stage 5.")
        else:
            if marks["rules"].exists() and is_valid_final_scenes(out_scenes_path):
                print(f"[resume] rules already applied: {out_scenes_path}")
            else:
                ensure_exists(out_scenes_path, "Base scenes.json (Stage 4 output)")
                check_rules_prereqs(required_metrics, av1an_temp, fastpass_out, ssimu2_log)

                base_obj = load_json(out_scenes_path)
                rule_scene_ranges = scenes_to_ranges(base_obj)
                metrics_state = RuleMetricsState(
                    src_file=metrics_ref_src,
                    vpy_src=metrics_ref_vpy_src,
                    frames_count=frames_count,
                    skip=int(args.skip),
                    av1an_temp=av1an_temp,
                    fastpass_out=fastpass_out,
                    ssimu2_log=ssimu2_log,
                    vs_source=str(args.vs_source),
                    verbose=bool(args.verbose),
                    scene_ranges=rule_scene_ranges,
                    required_metrics=required_metrics,
                )
                requirements = collect_rule_requirements(
                    base_obj=base_obj,
                    compiled_rules=compiled_rules,
                    rule_name=rule_name or "<rules>",
                    metrics_state=metrics_state,
                    verbose=bool(args.verbose),
                )
                for m in required_metrics:
                    if is_nvof_metric(m) and not requirements.has(m):
                        requirements.add_global(m)
                        eprint(f"[warn] metric '{m}' used in rules but not required in pass1; using global ranges.")

                schedule = build_nvof_schedule(
                    scene_requirements=requirements.scene_metrics,
                    global_requirements=requirements.global_metrics,
                    scene_ranges=rule_scene_ranges,
                )
                if schedule:
                    script_arg = str(args.nvof_script or "").strip().lower()
                    if script_arg in ("", "none", "off", "false", "0"):
                        raise RuntimeError("NVOF metrics required but --nvof-script is disabled.")
                    nvof_script = Path(args.nvof_script).expanduser()
                    if not nvof_script.is_absolute():
                        cand = project_dir / nvof_script
                        if cand.exists():
                            nvof_script = cand

                    def resolve_path(arg: str, default_path: Path) -> Path:
                        key = str(arg or "").strip().lower()
                        if key in ("", "auto"):
                            return default_path
                        p = Path(arg).expanduser()
                        if not p.is_absolute():
                            p = project_dir / p
                        return p

                    nvof_dir = project_dir / "metrics"
                    nvof_csv = resolve_path(args.nvof_csv, nvof_dir / f"{input_file.stem}_nvof.csv")
                    nvof_schedule = resolve_path(args.nvof_schedule, nvof_dir / "nvof_schedule.json")

                    run_nvof_script(
                        input_file=input_file,
                        script_path=nvof_script,
                        schedule=schedule,
                        schedule_path=nvof_schedule,
                        out_csv=nvof_csv,
                        extra_args=str(args.nvof_args),
                        workdir=project_dir,
                    )
                    needed_bases = needed_bases_for_metrics(requirements.all_metrics())
                    if needed_bases:
                        metrics_state.nvof_cache = load_nvof_csv(nvof_csv, needed_bases=needed_bases)

                updated_obj = apply_rules_to_scenes(
                    base_obj=base_obj,
                    compiled_rules=compiled_rules,
                    rule_name=rule_name or "<rules>",
                    metrics_state=metrics_state,
                    verbose=bool(args.verbose),
                    rule_pass=2,
                )
                if args.rule_test:
                    print("[ok] rules executed (test mode): no output written.")
                    return 0

                save_json(out_scenes_path, updated_obj)
                touch(marks["rules"])
                print(f"[ok] rules applied: {out_scenes_path}")

    print("\nDone.")
    print(f"Base scenes : {base_scenes_path}")
    print(f"Fast-pass   : {fastpass_out}")
    print(f"SSIMU2 log  : {ssimu2_log}")
    print(f"Final scenes: {out_scenes_path}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        eprint(f"[error] command failed with exit code {e.returncode}")
        raise
