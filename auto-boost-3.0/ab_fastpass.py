"""av1an fast-pass invocation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

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

def run_fastpass_av1an(
    *,
    input_file: Path,
    fastpass_vpy: Optional[Path],
    fastpass_proxy: Optional[Path],
    output_file: Path,
    scenes_path: Optional[Path],
    av1an_temp: Path,
    sdm: str,
    workers: int,
    lp: int,
    fast_preset: int,
    fast_crf: float,
    video_params: str,
    ffmpeg_arg: str,
    verbose: bool,
    keep: bool,
    sc_only: bool,
    log_file: Optional[Path],
    log_level: Optional[str],
) -> None:
    """Build and execute the av1an fast-pass command (or sc-only)."""
    if scenes_path is not None and str(sdm).strip().lower() == "psd":
        ensure_exists(scenes_path, "Base scenes.json")
    ensure_dir(av1an_temp)

    av1an_input = fastpass_vpy if fastpass_vpy is not None else input_file
    cmd: List[str] = [
        "av1an",
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
    if fastpass_proxy is not None:
        cmd.extend(["--proxy", str(fastpass_proxy)])
    if fastpass_vpy is not None or fastpass_proxy is not None:
        cmd.extend(["--vspipe-args", f"src={input_file}"])

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
    cmd.extend(["--cache-mode", "temp"])

    # Encoder & encode settings (fast pass)
    cmd.extend(["-e", "svt-av1", "--force"])
    cmd.extend(["-a","-an -sn"])

    enc_params = f'--preset {int(fast_preset)} --crf {float(fast_crf):.2f} --lp {int(lp)}'
    if video_params:
        enc_params += f" {video_params.strip()}"
    cmd.extend(["-v", enc_params])

    cmd.extend(build_av1an_filter_arg(ffmpeg_arg))
    cmd.extend(["-w", str(int(workers))])
    cmd.extend(["-o", str(output_file)])

    run_cmd(cmd, check=True, inherit_output=True, cwd=av1an_temp.parent)
    if sc_only:
        print("[ok] scene detection (sc-only) completed")
    else:
        print(f"[ok] fast-pass output: {output_file}")
