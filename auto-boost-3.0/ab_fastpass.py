"""av1an fast-pass invocation helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ab_encoder import build_fastpass_hdr10_params, build_fastpass_params, normalize_encoder
from ab_cmd import run_cmd
from ab_fs import ensure_dir, ensure_exists

def build_av1an_filter_arg(ffmpeg_arg: str) -> List[str]:
    """
    av1an expects `-f <ffmpeg options>`.
    Support both:
      - If arg starts with '-', treat as raw ffmpeg args
      - Else treat as filtergraph and wrap as: -vf "<filtergraph>"
    """
    if not ffmpeg_arg:
        return []
    s = ffmpeg_arg.strip()
    if not s:
        return []
    if s.startswith("-"):
        return ["-f", s]
    return ["-f", f'-vf "{s}"']


def query_fastpass_hdr10_payload(*, input_file: Path, hdr_patch_script: Path) -> Dict[str, Any]:
    ensure_exists(hdr_patch_script, "HDR patch script")
    cmd = [
        sys.executable,
        str(hdr_patch_script),
        "--source",
        str(input_file),
        "--print-hdr10-json",
    ]
    cp = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if cp.returncode != 0:
        detail = (cp.stderr or cp.stdout or "").strip() or f"exit code {cp.returncode}"
        raise RuntimeError(f"Failed to query HDR10 metadata for fast-pass: {detail}")

    try:
        payload = json.loads(cp.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse HDR10 metadata response: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("HDR10 metadata response is not a JSON object.")
    return payload


def run_fastpass_av1an(
    *,
    av1an_exe: str,
    input_file: Path,
    fastpass_vpy: Optional[Path],
    fastpass_proxy: Optional[Path],
    output_file: Path,
    scenes_path: Optional[Path],
    av1an_temp: Path,
    sdm: str,
    workers: int,
    lp: int,
    fast_preset: str,
    fast_crf: float,
    encoder: str,
    video_params: str,
    ffmpeg_arg: str,
    verbose: bool,
    keep: bool,
    sc_only: bool,
    log_file: Optional[Path],
    log_level: Optional[str],
    progress_jsonl: Optional[Path] = None,
    fastpass_hdr: bool = False,
    hdr_patch_script: Optional[Path] = None,
    chunk_order: str = "",
    encoder_path: str = "",
    fast_interrupt: bool = False,
    vspipe_args: Optional[List[str]] = None,
) -> None:
    """Build and execute the av1an fast-pass command (or sc-only)."""
    encoder = normalize_encoder(encoder)
    if scenes_path is not None and str(sdm).strip().lower() == "psd":
        ensure_exists(scenes_path, "Base scenes.json")
    ensure_dir(av1an_temp)

    av1an_input = fastpass_vpy if fastpass_vpy is not None else input_file
    cmd: List[str] = [
        str(av1an_exe or "av1an"),
        "-i", str(av1an_input),
        "--temp", str(av1an_temp),
        "-y",
    ]
    if sc_only:
        cmd.append("--sc-only")
    if verbose:
        cmd.append("--verbose")
    if keep:
        cmd.append("--keep")
    if log_level:
        cmd.extend(["--log-level", str(log_level)])
    if log_file is not None:
        cmd.extend(["--log-file", str(log_file)])
    if progress_jsonl is not None:
        cmd.extend(["--progress-jsonl", str(progress_jsonl)])
    if fastpass_proxy is not None:
        cmd.extend(["--proxy", str(fastpass_proxy)])
    vspipe_arg_list = [str(item) for item in (vspipe_args or []) if str(item).strip()]
    if (fastpass_vpy is not None or fastpass_proxy is not None) and not any(item.startswith("src=") for item in vspipe_arg_list):
        vspipe_arg_list.insert(0, f"src={input_file}")
    if vspipe_arg_list:
        cmd.append("--vspipe-args")
        cmd.extend(vspipe_arg_list)

    # Provide scenes file (skip scene detection) or emit scenes when --sc-only
    if scenes_path is not None:
        cmd.extend(["--scenes", str(scenes_path)])
        if str(sdm).strip().lower() == "psd":
            cmd.extend(["--extra-split-sec", "30"])
        else:
            cmd.extend(["--extra-split-sec", "15"])
    else:
        cmd.extend(["--extra-split-sec", "15"])


    # Muxing defaults from old scripts
    cmd.extend(["-m", "lsmash", "-c", "mkvmerge"])
    cmd.extend(["--chunk-order", "random"])
    # Prepared for the fork-only chunk-order option, but fastpass stays on random for now.
    # if str(chunk_order).strip():
    #     cmd.extend(["--chunk-order", str(chunk_order).strip()])
    if str(encoder_path).strip():
        cmd.extend(["--encoder-path", str(encoder_path).strip()])
    if fast_interrupt:
        cmd.append("--fast-interrupt")
    cmd.extend(["--cache-mode", "temp"])

    # Encoder & encode settings (fast pass)
    cmd.extend(["-e", encoder, "--force"])
    cmd.extend(["-a","-an -sn"])
    cmd.extend(["--resume"]) # TODO - resume fastpass if params not changed

    enc_params = build_fastpass_params(
        encoder=encoder,
        preset=str(fast_preset),
        crf=float(fast_crf),
        lp=int(lp),
        video_params=str(video_params),
    )
    if fastpass_hdr:
        script_path = hdr_patch_script or (Path(__file__).resolve().parent.parent / "utils" / "av1an_hdr_metadata_patch_v2.py")
        hdr_payload = query_fastpass_hdr10_payload(input_file=input_file, hdr_patch_script=script_path)
        hdr_tokens = build_fastpass_hdr10_params(hdr_payload, encoder=encoder)
        if hdr_tokens:
            hdr_params = " ".join(hdr_tokens)
            print(f"[hdr] fast-pass static HDR params: {hdr_params}")
            enc_params += f" {hdr_params}"
        else:
            print("[hdr] fast-pass static HDR params: none detected")
    cmd.extend(["-v", enc_params])

    cmd.extend(build_av1an_filter_arg(ffmpeg_arg))
    cmd.extend(["-w", str(int(workers))])
    cmd.extend(["-o", str(output_file)])

    run_cmd(cmd, check=True, inherit_output=True, cwd=av1an_temp.parent)
    if sc_only:
        print("[ok] scene detection (sc-only) completed")
    else:
        print(f"[ok] fast-pass output: {output_file}")
