#!/usr/bin/env python3

# Progressive Scene Detection
# Copyright (c) Akatsumekusa and contributors
# Scene-detection-only build.
#
# Usage:
#   python Progressive-Scene-Detection.py -i INPUT.mkv -o OUTPUT.scenes.json
#
# Optional:
#   --temp DIR  Temporary directory. By default it is based on the output path.
#   -v          Verbose output. Repeating -v is accepted.
#   --get-config        Print default TOML config and exit.
#   --describe-config   Print detailed config descriptions and exit.
#
# Scene detection settings can be overridden with command line arguments.

import argparse
from datetime import datetime
import json
import math
import numpy as np
import os
from pathlib import Path
import platform
import re
import runpy
import shutil
import subprocess
import sys
import time

if platform.system() == "Windows":
    os.system("")


DEFAULT_SCENE_DETECTION_CONFIG = {
#   VapourSynth luma range used for black/white transition detection.
#   Use "limited" for regular limited-range video, which is common for anime
#   and most encoded sources. Use "full" only when the source really uses
#   full-range luma. A wrong value mostly affects fade/flash detection.
    "scene_detection_vapoursynth_range": "limited",

#   Regular maximum scene length target. When a scene grows beyond this scale,
#   the splitter starts looking for additional cuts even if the source does not
#   contain a strong hard scenecut. Prefer 32*n+1 or 16*n+1 values so scene
#   lengths line up better with common encoder hierarchical structures.
    "scene_detection_extra_split": 289,

#   Maximum length for nearly-still scenes whose frame-to-frame luma difference
#   reaches the 0.0042 stillness band. This can be larger than
#   scene_detection_extra_split because almost-static scenes usually tolerate
#   longer chunks without visible quality instability.
    "scene_detection_0042_still_scene_extra_split": 353,

#   Maximum length for extremely still scenes whose frame-to-frame luma
#   difference stays around the 0.0012 band. This can be the largest split
#   target because there is very little motion to protect.
    "scene_detection_0012_still_scene_extra_split": 481,

#   Hard minimum scene length. The recursive splitter will not accept a split
#   that creates a side shorter than this value. Raising it suppresses tiny
#   scenes and accidental cuts; lowering it allows more aggressive detection.
    "scene_detection_min_scene_len": 17,

#   Small target split. For short ranges, the splitter prefers cuts where both
#   sides fit within this value while still respecting scene_detection_min_scene_len.
#   It acts as the lower target size class for strong scene-change candidates.
    "scene_detection_18_target_split": 33,

#   Medium target split. The splitter uses this as the next size class: both
#   sides should usually be at least scene_detection_18_target_split and at most
#   this value for medium-confidence cuts. Raising it makes medium scenes less
#   likely to be split further; lowering it makes the script split more often.
    "scene_detection_12_target_split": 97,

#   Long-scene early split guard. This is used when a range is very long and a
#   very strong candidate cut appears. Both sides must be at least this long, so
#   the algorithm avoids cutting off a short tail from an otherwise long scene.
    "scene_detection_27_extra_target_split": 161,
}

# values_32n_plus_1 = [33, 65, 97, 129, 161, 193, 225, 257, 289, 321, 353, 385, 417, 449, 481, 513, 545, 577, 609, 641, 673, 705, 737, 769, 801, 833, 865, 897, 929, 961, 993]
# values_16n_plus_1 = [17, 33, 49, 65, 81, 97, 113, 129, 145, 161, 177, 193, 209, 225, 241, 257, 273, 289, 305, 321, 337, 353, 369, 385, 401, 417, 433, 449, 465, 481, 497, 513, 529, 545, 561, 577, 593, 609, 625, 641, 657, 673, 689, 705, 721, 737, 753, 769, 785, 801, 817, 833, 849, 865, 881, 897, 913, 929, 945, 961, 977, 993]



def recommended_values(step: int) -> list[int]:
    return [step * n + 1 for n in range(1, 1000 // step + 1) if step * n + 1 <= 1000]


def toml_string(value) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    return str(value)


def generate_default_config_toml() -> str:
    return f"""
scene_detection_vapoursynth_range = {toml_string(DEFAULT_SCENE_DETECTION_CONFIG["scene_detection_vapoursynth_range"])}
scene_detection_extra_split = {DEFAULT_SCENE_DETECTION_CONFIG["scene_detection_extra_split"]}
scene_detection_0042_still_scene_extra_split = {DEFAULT_SCENE_DETECTION_CONFIG["scene_detection_0042_still_scene_extra_split"]}
scene_detection_0012_still_scene_extra_split = {DEFAULT_SCENE_DETECTION_CONFIG["scene_detection_0012_still_scene_extra_split"]}
scene_detection_min_scene_len = {DEFAULT_SCENE_DETECTION_CONFIG["scene_detection_min_scene_len"]}
scene_detection_18_target_split = {DEFAULT_SCENE_DETECTION_CONFIG["scene_detection_18_target_split"]}
scene_detection_12_target_split = {DEFAULT_SCENE_DETECTION_CONFIG["scene_detection_12_target_split"]}
scene_detection_27_extra_target_split = {DEFAULT_SCENE_DETECTION_CONFIG["scene_detection_27_extra_target_split"]}
"""


class NumpyEncoder(json.JSONEncoder):
    def default(self, object):
        if isinstance(object, np.generic):
            return object.item()
        if isinstance(object, np.ndarray):
            return object.tolist()
        return super(NumpyEncoder, self).default(object)


parser = argparse.ArgumentParser(prog="Progressive Scene Detection")
parser.add_argument("--get-config", action="store_true", help="Print default TOML config and exit")
parser.add_argument("-i", "--input", type=Path, help="Source video file")
parser.add_argument("-o", "--output-scenes", type=Path, help="Output scenes JSON for av1an")
parser.add_argument("--temp", type=Path, help="Temporary folder. Default: output path with .scene-detection.tmp suffix")
parser.add_argument("--vspipe-arg", action="append", default=[],
                    help="Extra key=value argument for .vpy input. Can be repeated.")
parser.add_argument("--vspipe-args", nargs="+", default=[],
                    help="Extra key=value arguments for .vpy input, forwarded to av1an.")
parser.add_argument("-v", "--verbose", action="count", default=0, help="Show more progress details")
parser.add_argument("--scene-detection-vapoursynth-range",
                    choices=["limited", "full"],
                    default=DEFAULT_SCENE_DETECTION_CONFIG["scene_detection_vapoursynth_range"],
                    help="Luma range used for black/white transition detection")
parser.add_argument("--scene-detection-extra-split",
                    type=int,
                    default=DEFAULT_SCENE_DETECTION_CONFIG["scene_detection_extra_split"],
                    help="Regular maximum scene length target")
parser.add_argument("--scene-detection-0042-still-scene-extra-split",
                    type=int,
                    default=DEFAULT_SCENE_DETECTION_CONFIG["scene_detection_0042_still_scene_extra_split"],
                    help="Maximum length for nearly-still scenes")
parser.add_argument("--scene-detection-0012-still-scene-extra-split",
                    type=int,
                    default=DEFAULT_SCENE_DETECTION_CONFIG["scene_detection_0012_still_scene_extra_split"],
                    help="Maximum length for extremely still scenes")
parser.add_argument("--scene-detection-min-scene-len",
                    type=int,
                    default=DEFAULT_SCENE_DETECTION_CONFIG["scene_detection_min_scene_len"],
                    help="Hard minimum scene length")
parser.add_argument("--scene-detection-18-target-split",
                    type=int,
                    default=DEFAULT_SCENE_DETECTION_CONFIG["scene_detection_18_target_split"],
                    help="Small target split size")
parser.add_argument("--scene-detection-12-target-split",
                    type=int,
                    default=DEFAULT_SCENE_DETECTION_CONFIG["scene_detection_12_target_split"],
                    help="Medium target split size")
parser.add_argument("--scene-detection-27-extra-target-split",
                    type=int,
                    default=DEFAULT_SCENE_DETECTION_CONFIG["scene_detection_27_extra_target_split"],
                    help="Long-scene early split guard")
args = parser.parse_args()

if args.get_config:
    print(generate_default_config_toml(), end="")
    raise SystemExit(0)

if args.input is None:
    parser.error("the following arguments are required unless --get-config or --describe-config is used: -i/--input")
if args.output_scenes is None:
    parser.error("the following arguments are required unless --get-config or --describe-config is used: -o/--output-scenes")

if args.scene_detection_min_scene_len < 1:
    parser.error("--scene-detection-min-scene-len must be at least 1")
if args.scene_detection_extra_split < 2 * args.scene_detection_min_scene_len:
    parser.error("--scene-detection-extra-split must be at least 2 times --scene-detection-min-scene-len")
if args.scene_detection_0042_still_scene_extra_split < args.scene_detection_extra_split:
    parser.error("--scene-detection-0042-still-scene-extra-split must be >= --scene-detection-extra-split")
if args.scene_detection_0012_still_scene_extra_split < args.scene_detection_extra_split:
    parser.error("--scene-detection-0012-still-scene-extra-split must be >= --scene-detection-extra-split")
if args.scene_detection_18_target_split * 2 > args.scene_detection_12_target_split:
    parser.error("--scene-detection-18-target-split * 2 must be <= --scene-detection-12-target-split")
if args.scene_detection_18_target_split * 2 > args.scene_detection_extra_split:
    parser.error("--scene-detection-18-target-split * 2 must be <= --scene-detection-extra-split")
if args.scene_detection_12_target_split * 2 > args.scene_detection_extra_split:
    parser.error("--scene-detection-12-target-split * 2 must be <= --scene-detection-extra-split")
if args.scene_detection_27_extra_target_split > args.scene_detection_extra_split:
    parser.error("--scene-detection-27-extra-target-split must be <= --scene-detection-extra-split")

import vapoursynth as vs
from vapoursynth import core

input_file = args.input
scenes_file = args.output_scenes
temp_dir = args.temp
if not temp_dir:
    temp_dir = scenes_file
    if temp_dir.with_suffix("").suffix.lower() == ".scenes":
        temp_dir = temp_dir.with_suffix("")
    temp_dir = temp_dir.with_suffix(".scene-detection.tmp")

scene_detection_temp_dir = temp_dir / "scene-detection"
scene_detection_temp_dir.mkdir(parents=True, exist_ok=True)

verbose = args.verbose
if 1 <= verbose < 3:
    verbose = 3

temp_dir.joinpath("source.ffindex").unlink(missing_ok=True)

vspipe_arg_list = [str(item) for item in (args.vspipe_args or []) if str(item).strip()]
vspipe_arg_list.extend(str(item) for item in (args.vspipe_arg or []) if str(item).strip())


def vspipe_args_to_dict(items):
    out = {}
    for item in items or []:
        text = str(item or "").strip()
        if not text:
            continue
        if "=" not in text:
            out[text] = ""
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        if key:
            out[key] = value
    return out


def unwrap_vapoursynth_output(out):
    try:
        if isinstance(out, vs.VideoNode):
            return out
    except Exception:
        pass

    for attr in ("clip", "node", "video", "output"):
        try:
            value = getattr(out, attr, None)
            if value is not None and isinstance(value, vs.VideoNode):
                return value
        except Exception:
            continue

    if isinstance(out, (tuple, list)):
        for value in out:
            try:
                if isinstance(value, vs.VideoNode):
                    return value
            except Exception:
                continue
    return None


def load_vpy_clip(vpy_path, vspipe_items):
    try:
        if hasattr(vs, "clear_outputs"):
            vs.clear_outputs()
        elif hasattr(core, "clear_outputs"):
            core.clear_outputs()
    except Exception:
        pass

    old_cwd = os.getcwd()
    old_sys_path = list(sys.path)
    vpy_args = vspipe_args_to_dict(vspipe_items)

    try:
        os.chdir(str(vpy_path.parent))
        if str(vpy_path.parent) not in sys.path:
            sys.path.insert(0, str(vpy_path.parent))

        init_globals = {
            "__file__": str(vpy_path),
            "__name__": "__vapoursynth__",
        }
        init_globals.update({str(key): str(value) for key, value in vpy_args.items()})
        ns = runpy.run_path(str(vpy_path), init_globals=init_globals)
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path

    getters = []
    if hasattr(vs, "get_output"):
        getters.append(getattr(vs, "get_output"))
    if hasattr(core, "get_output"):
        getters.append(getattr(core, "get_output"))

    for get_output in getters:
        try:
            node = unwrap_vapoursynth_output(get_output(0))
        except Exception:
            continue
        if node is not None:
            return node

    for key in ("clip", "out", "output", "src_clip"):
        node = unwrap_vapoursynth_output(ns.get(key))
        if node is not None:
            return node

    for value in ns.values():
        node = unwrap_vapoursynth_output(value)
        if node is not None:
            return node

    raise RuntimeError(f"No VapourSynth output found in vpy: {vpy_path}")


class DefaultZone:
    # Source loading. Regular media uses ffms2; .vpy is executed in-process.
    if input_file.suffix.lower() == ".vpy":
        source_clip = load_vpy_clip(input_file.expanduser().resolve(), vspipe_arg_list)
        source_clip_cache = None
        source_provider_av1an = ""
        source_clip_cache_reuse = False
    else:
        source_clip = core.ffms2.Source(input_file.expanduser().resolve(), cachefile=temp_dir.joinpath("source.ffindex").expanduser().resolve())
        source_clip_cache = temp_dir.joinpath("source.ffindex")
        source_provider_av1an = "ffms2"
        source_clip_cache_reuse = True

    # Scene detection is fixed to x264 + WWXD.
    scene_detection_vapoursynth_range = args.scene_detection_vapoursynth_range

    # Maximum scene length. Prefer values shaped as 32*n+1 or 16*n+1.
    scene_detection_extra_split = args.scene_detection_extra_split
    scene_detection_0042_still_scene_extra_split = args.scene_detection_0042_still_scene_extra_split
    scene_detection_0012_still_scene_extra_split = args.scene_detection_0012_still_scene_extra_split

    # Minimum scene length and target split thresholds.
    scene_detection_min_scene_len = args.scene_detection_min_scene_len
    scene_detection_18_target_split = args.scene_detection_18_target_split
    scene_detection_12_target_split = args.scene_detection_12_target_split
    scene_detection_27_extra_target_split = args.scene_detection_27_extra_target_split


zone_default = DefaultZone()
zones = [{"start_frame": 0, "end_frame": zone_default.source_clip.num_frames, "zone": zone_default}]


print(f"\r\033[KTime {datetime.now().time().isoformat(timespec="seconds")} / Progressive Scene Detection started", end="\n", flush=True)



scene_detection_scenes_file = scene_detection_temp_dir.joinpath("scenes.json")
scene_detection_x264_scenes_file = scene_detection_temp_dir.joinpath("x264.scenes.json")
scene_detection_x264_temp_dir = scene_detection_temp_dir.joinpath("x264.tmp")
scene_detection_x264_output_file = scene_detection_temp_dir.joinpath("x264.mkv")
scene_detection_x264_stats_dir = scene_detection_temp_dir.joinpath("x264.logs")
scene_detection_diffs_file = scene_detection_temp_dir.joinpath("luma-diff.txt")
scene_detection_average_file = scene_detection_temp_dir.joinpath("luma-average.txt")
scene_detection_min_file = scene_detection_temp_dir.joinpath("luma-min.txt")
scene_detection_max_file = scene_detection_temp_dir.joinpath("luma-max.txt")

scene_detection_diffs_available = False

frame_rjust_digits = math.floor(np.log10(zone_default.source_clip.num_frames)) + 1
frame_print = lambda frame: f"Frame {frame}"
frame_rjust = lambda frame: str(frame).rjust(frame_rjust_digits)
frame_scene_print = lambda start_frame, end_frame: f"Scene [{frame_rjust(start_frame)}:{frame_rjust(end_frame)}]"




scene_detection_x264_output_file.unlink(missing_ok=True)
scene_detection_x264_stats_dir.mkdir(exist_ok=True)

scene_detection_x264_scenes = {}
scene_detection_x264_scenes["scenes"] = []
scene_detection_x264_total_frames = 0
scene_detection_x264_total_frames_print = 0
for zone_i, zone in enumerate(zones):
    def scene_detection_append_x264_scene(name, start_frame, end_frame):
        scene_detection_x264_scenes["scenes"].append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "zone_overrides": {
                "encoder": "x264",
                "passes": 1,
                "video_params": [
                    "--output-depth", "10",
                    "--preset", "veryfast",
                    "--qp", "80",
                    "--keyint", f"{end_frame - start_frame + 240}",
                    "--min-keyint", "1",
                    "--scenecut", "40",
                    "--rc-lookahead", "120",
                    "--ref", "1",
                    "--aq-mode", "0",
                    "--no-8x8dct",
                    "--partition", "none",
                    "--no-weightb",
                    "--weightp", "0",
                    "--me", "dia",
                    "--subme", "2", # Required for scene detection
                    "--no-psy",
                    "--trellis", "0",
                    "--no-cabac",
                    "--no-deblock",
                    "--slow-firstpass",
                    "--pass", "1",
                    "--stats", f"{scene_detection_x264_stats_dir / f"{name}.log"}"
                ],
                "photon_noise": None,
                "photon_noise_height": None,
                "photon_noise_width": None,
                "chroma_noise": False,
                "extra_splits_len": zone["zone"].scene_detection_extra_split,
                "min_scene_len": zone["zone"].scene_detection_min_scene_len
            }
        })
    scene_detection_x264_total_frames_print += zone["end_frame"] - zone["start_frame"]
    if zone["end_frame"] - zone["start_frame"] < 120:
        scene_detection_x264_total_frames += zone["end_frame"] - zone["start_frame"]
        scene_detection_append_x264_scene(f"{zone_i}", zone["start_frame"], zone["end_frame"])
    else:
        scene_detection_x264_total_frames += zone["end_frame"] - zone["start_frame"] + 4
        scene_detection_append_x264_scene(f"{zone_i}_left", zone["start_frame"], math.floor((zone["start_frame"] + zone["end_frame"]) / 2) + 4)
        scene_detection_append_x264_scene(f"{zone_i}_right", math.floor((zone["start_frame"] + zone["end_frame"]) / 2), zone["end_frame"])
scene_detection_x264_scenes["frames"] = scene_detection_x264_total_frames
scene_detection_x264_scenes["split_scenes"] = scene_detection_x264_scenes["scenes"]

with scene_detection_x264_scenes_file.open("w") as scene_detection_x264_scenes_f:
    json.dump(scene_detection_x264_scenes, scene_detection_x264_scenes_f, cls=NumpyEncoder)

if zone_default.source_clip_cache_reuse and zone_default.source_clip_cache is not None:
    scene_detection_x264_temp_dir_cache = scene_detection_x264_temp_dir / "split" / "cache"
    scene_detection_x264_temp_dir_cache = scene_detection_x264_temp_dir_cache.with_suffix(zone_default.source_clip_cache.suffix)
    
    if not scene_detection_x264_temp_dir_cache.exists():
        scene_detection_x264_temp_dir_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(zone_default.source_clip_cache, scene_detection_x264_temp_dir_cache)

command = [
    "av1an",
    "-y"
]
if verbose < 2:
    command += ["--quiet"]
if verbose >= 3:
    command += ["--verbose"]
command += [
    "--temp", scene_detection_x264_temp_dir,
    "--keep"
]
command += [
    "-i", input_file
]
command += [
    "-o", scene_detection_x264_output_file,
    "--scenes", scene_detection_x264_scenes_file,
    "--encoder", "x264",
    "--pix-format", "yuv420p10le",
    "--cache-mode", "temp",
    "--workers", "2",
    "--force", "--video-params", f"[K[0m[1;3m> Progressive Scene Detection [0m[3mx264-based-scene-detection[0m[1;3m <[0m",
    "--audio-params", "-an",
    "--concat", "mkvmerge"
]
if zone_default.source_provider_av1an:
    command += ["--chunk-method", zone_default.source_provider_av1an]
if vspipe_arg_list:
    command += ["--vspipe-args"]
    command += vspipe_arg_list
scene_detection_x264_process = subprocess.Popen(command, text=True)




if not scene_detection_diffs_available:
    scene_detection_diffs = np.empty((zone_default.source_clip.num_frames,), dtype=np.float32)
    scene_detection_average = np.empty((zone_default.source_clip.num_frames,), dtype=np.float32)
    scene_detection_min = np.empty((zone_default.source_clip.num_frames,), dtype=np.float32)
    scene_detection_max = np.empty((zone_default.source_clip.num_frames,), dtype=np.float32)

scene_detection_clip_base = zone_default.source_clip
scene_detection_bits = scene_detection_clip_base.format.bits_per_sample

if not scene_detection_diffs_available:
    scene_detection_clip_base = scene_detection_clip_base.std.PlaneStats(scene_detection_clip_base[0] + scene_detection_clip_base, plane=0, prop="Luma")

target_width = np.round(np.sqrt(1280 * 720 / scene_detection_clip_base.width / scene_detection_clip_base.height) * scene_detection_clip_base.width / 40) * 40
if target_width < scene_detection_clip_base.width * 0.9:
    target_height = np.ceil(target_width / scene_detection_clip_base.width * scene_detection_clip_base.height / 2) * 2
    src_height = target_height / target_width * scene_detection_clip_base.width
    src_top = (scene_detection_clip_base.height - src_height) / 2
    scene_detection_clip_base = scene_detection_clip_base.resize.Point(width=target_width, height=target_height, src_top=src_top, src_height=src_height,
                                                                       format=vs.YUV420P8, dither_type="none")

zones_diffs = {}
zones_vapoursynth_scenecut = {}
zones_luma_scenecut = {}
for zone_i, zone in enumerate(zones):

    assert zone["zone"].scene_detection_vapoursynth_range in ["limited", "full"], "Invalid `scene_detection_vapoursynth_range`. Please check your config inside `Progressive-Scene-Detection.py`."
    assert zone["zone"].scene_detection_extra_split >= zone["zone"].scene_detection_min_scene_len * 2, "`scene_detection_extra_split` must be at least 2 times `scene_detection_min_scene_len`."

    scene_detection_clip = scene_detection_clip_base[zone["start_frame"]:zone["end_frame"]]
    scene_detection_clip = scene_detection_clip.wwxd.WWXD()

    diffs = np.empty((scene_detection_clip.num_frames,), dtype=float)
    vapoursynth_scenecut = np.zeros((scene_detection_clip.num_frames,), dtype=float)
    luma_scenecut = np.zeros((scene_detection_clip.num_frames,), dtype=bool)
    luma_scenecut_prev = True

    start = time.time() - 0.000001
    for offset_frame, frame in enumerate(scene_detection_clip.frames(backlog=48)):
        current_frame = zone["start_frame"] + offset_frame
        print(f"\r\033[K{frame_print(current_frame)} / Detecting scenes / {offset_frame / (time.time() - start):.2f} fps", end="", flush=True)

        if not scene_detection_diffs_available:
            scene_detection_diffs[current_frame] = frame.props["LumaDiff"]
            scene_detection_average[current_frame] = frame.props["LumaAverage"]
            scene_detection_min[current_frame] = frame.props["LumaMin"]
            scene_detection_max[current_frame] = frame.props["LumaMax"]
        diffs[offset_frame] = scene_detection_diffs[current_frame]

        vapoursynth_scenecut[offset_frame] = frame.props["Scenechange"] == 1

        if zone["zone"].scene_detection_vapoursynth_range == "limited":
            luma_scenecut_current = scene_detection_min[current_frame] > 231.125 * 2 ** (scene_detection_bits - 8) or \
                                    scene_detection_max[current_frame] < 19.875 * 2 ** (scene_detection_bits - 8)
        elif zone["zone"].scene_detection_vapoursynth_range == "full":
            luma_scenecut_current = scene_detection_min[current_frame] > 251.125 * 2 ** (scene_detection_bits - 8) or \
                                    scene_detection_max[current_frame] < 3.875 * 2 ** (scene_detection_bits - 8)
        if luma_scenecut_current or luma_scenecut_prev:
            luma_scenecut[offset_frame] = True
        luma_scenecut_prev = luma_scenecut_current

    zones_diffs[zone_i] = diffs
    zones_vapoursynth_scenecut[zone_i] = vapoursynth_scenecut
    zones_luma_scenecut[zone_i] = luma_scenecut

print(f"\r\033[K{frame_print(current_frame + 1)} / VapourSynth based scene detection complete", end="\n", flush=True)



if scene_detection_x264_process.poll() is None:
    print(f"\r\033[K{frame_print(0)} / Performing x264 based scene detection", end="", flush=True)
scene_detection_x264_process.wait()
print(f"\r\033[K{frame_print(scene_detection_x264_total_frames_print)} / x264 based scene detection finished", end="\n", flush=True)

zones_x264_scenecut = {}
scene_detection_match_x264_I = re.compile(r"^in:(\d+) out:\d+ type:(\w)")
for zone_i, zone in enumerate(zones):
    x264_scenecut = np.zeros((zone["end_frame"] - zone["start_frame"],), dtype=float)
    def scene_detection_write_x264_scenecut(name, start_frame, end_frame, skip_starting_frames=False):
        assert (scene_detection_x264_stats_dir / f"{name}.log").exists(), "Unexpected result from x264"
        with (scene_detection_x264_stats_dir / f"{name}.log").open("r") as x264_stats_f:
            x264_stats = x264_stats_f.read()

        for line in x264_stats.splitlines():
            if match := scene_detection_match_x264_I.match(line):
                try:
                    offset_frame = int(match.group(1))
                except ValueError:
                    raise ValueError("Unexpected result from x264")
                assert offset_frame + start_frame < end_frame, "Unexpected result from x264"

                if offset_frame == 0 and skip_starting_frames:
                    continue

                if match.group(2) == "I":
                    x264_scenecut[offset_frame + start_frame] = 1

    if zone["end_frame"] - zone["start_frame"] < 120:
        scene_detection_write_x264_scenecut(f"{zone_i}", 0, zone["end_frame"] - zone["start_frame"])
    else:
        scene_detection_write_x264_scenecut(f"{zone_i}_left", 0, math.floor((zone["end_frame"] - zone["start_frame"]) / 2) + 4)
        scene_detection_write_x264_scenecut(f"{zone_i}_right", math.floor((zone["end_frame"] - zone["start_frame"]) / 2), zone["end_frame"] - zone["start_frame"],
                                                               skip_starting_frames=True)

    zones_x264_scenecut[zone_i] = x264_scenecut


scenes = {}
scenes["frames"] = zone_default.source_clip.num_frames
scenes["scenes"] = []
for zone_i, zone in enumerate(zones):
    diffs = zones_diffs[zone_i]
    luma_scenecut = zones_luma_scenecut[zone_i]
    vapoursynth_scenecut = zones_vapoursynth_scenecut[zone_i]
    x264_scenecut = zones_x264_scenecut[zone_i]

    diffs_half = diffs / 2
    diffs_0012 = diffs >= 0.0012
    diffs_0042 = diffs >= 0.0042
    diffs[1:] -= diffs[:-1]
    diffs[diffs < diffs_half] = diffs_half[diffs < diffs_half]
    diffs[np.logical_and(diffs_0012, diffs < 0.0012)] = 0.0012
    diffs[np.logical_and(diffs_0042, diffs < 0.0042)] = 0.0042

    diffs[luma_scenecut] *= 1.70
    diffs[luma_scenecut] += 1.24

    vapoursynth_scenecut *= 0.88
    x264_scenecut *= 0.94
    vapoursynth_scenecut += x264_scenecut
    vapoursynth_scenecut[vapoursynth_scenecut > 1.0] = 1.0
    diffs[~luma_scenecut] += vapoursynth_scenecut[~luma_scenecut]
    
    diffs_sort = np.argsort(diffs, stable=True)[::-1]

    def scene_detection_split_scene(start_frame, end_frame):
        assert zone["zone"].scene_detection_0042_still_scene_extra_split >= zone["zone"].scene_detection_extra_split, "Invalid `scene_detection_0042_still_scene_extra_split`. This value must be bigger than or equal to `scene_detection_extra_split`. Please check your config inside `Progressive-Scene-Detection.py`."
        assert zone["zone"].scene_detection_0012_still_scene_extra_split >= zone["zone"].scene_detection_extra_split, "Invalid `scene_detection_0012_still_scene_extra_split`. This value must be bigger than or equal to `scene_detection_extra_split`. Please check your config inside `Progressive-Scene-Detection.py`."
        assert zone["zone"].scene_detection_min_scene_len * 2 <= zone["zone"].scene_detection_extra_split, "Invalid `scene_detection_min_scene_len`. 2 times this value must be smaller than or equal to `scene_detection_extra_split`. Please check your config inside `Progressive-Scene-Detection.py`."
        assert zone["zone"].scene_detection_18_target_split * 2 <= zone["zone"].scene_detection_12_target_split, "Invalid `scene_detection_18_target_split`. 2 times this value must be smaller than or equal to `scene_detection_12_target_split`. Please check your config inside `Progressive-Scene-Detection.py`."
        assert zone["zone"].scene_detection_18_target_split * 2 <= zone["zone"].scene_detection_extra_split, "Invalid `scene_detection_18_target_split`. 2 times this value must be smaller than or equal to `scene_detection_extra_split`. Please check your config inside `Progressive-Scene-Detection.py`."
        assert zone["zone"].scene_detection_12_target_split * 2 <= zone["zone"].scene_detection_extra_split, "Invalid `scene_detection_12_target_split`. 2 times this value must be smaller than or equal to `scene_detection_extra_split`. Please check your config inside `Progressive-Scene-Detection.py`."
        assert zone["zone"].scene_detection_27_extra_target_split <= zone["zone"].scene_detection_extra_split, "Invalid `scene_detection_27_extra_target_split`. This value must be smaller than or equal to `scene_detection_extra_split`. Please check your config inside `Progressive-Scene-Detection.py`."



        print(f"\r\033[K{frame_scene_print(start_frame + zone["start_frame"], end_frame + zone["start_frame"])} / Creating scenes", end="", flush=True)



        if end_frame - start_frame < 2 * zone["zone"].scene_detection_min_scene_len:
            if verbose >= 3:
                print(f" / branch complete", end="\n", flush=True)
            return [start_frame]



        if end_frame - start_frame >= 2 * zone["zone"].scene_detection_extra_split:
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.27:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_27_extra_target_split and end_frame - current_frame >= zone["zone"].scene_detection_27_extra_target_split and \
                   math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
                   math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
                   math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split + 0.40):
                    if verbose >= 3:
                        print(f" / split / extra_split 1.27 mode / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)



        if end_frame - start_frame <= 2 * zone["zone"].scene_detection_18_target_split:
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.18:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len and \
                   current_frame - start_frame <= zone["zone"].scene_detection_18_target_split and end_frame - current_frame <= zone["zone"].scene_detection_18_target_split and \
                   ((current_frame - start_frame) % 32 == 1 or (end_frame - current_frame) % 32 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.18 doubleside mode / 32-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)
                           
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.18:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len and \
                   current_frame - start_frame <= zone["zone"].scene_detection_18_target_split and end_frame - current_frame <= zone["zone"].scene_detection_18_target_split and \
                   ((current_frame - start_frame) % 16 == 1 or (end_frame - current_frame) % 16 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.18 doubleside mode / 16-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)
                           
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.18:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len and \
                   current_frame - start_frame <= zone["zone"].scene_detection_18_target_split and end_frame - current_frame <= zone["zone"].scene_detection_18_target_split and \
                   ((current_frame - start_frame) % 8 == 1 or (end_frame - current_frame) % 8 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.18 doubleside mode / 8-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)
                           
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.18:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len and \
                   current_frame - start_frame <= zone["zone"].scene_detection_18_target_split and end_frame - current_frame <= zone["zone"].scene_detection_18_target_split and \
                   ((current_frame - start_frame) % 4 == 1 or (end_frame - current_frame) % 4 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.18 doubleside mode / 4-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)
                           
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.18:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len and \
                   current_frame - start_frame <= zone["zone"].scene_detection_18_target_split and end_frame - current_frame <= zone["zone"].scene_detection_18_target_split and \
                   ((current_frame - start_frame) % 2 == 1 or (end_frame - current_frame) % 2 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.18 doubleside mode / 2-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)



        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.18:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_18_target_split and (current_frame - start_frame) % 32 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_18_target_split and (end_frame - current_frame) % 32 == 1)):
                if verbose >= 3:
                    print(f" / split / 1.18 singleside mode / 32-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.18:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_18_target_split and (current_frame - start_frame) % 16 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_18_target_split and (end_frame - current_frame) % 16 == 1)):
                if verbose >= 3:
                    print(f" / split / 1.18 singleside mode / 16-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.18:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_18_target_split and (current_frame - start_frame) % 8 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_18_target_split and (end_frame - current_frame) % 8 == 1)):
                if verbose >= 3:
                    print(f" / split / 1.18 singleside mode / 8-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.18:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_18_target_split and (current_frame - start_frame) % 4 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_18_target_split and (end_frame - current_frame) % 4 == 1)):
                if verbose >= 3:
                    print(f" / split / 1.18 singleside mode / 4-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.18:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_18_target_split and (current_frame - start_frame) % 2 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_18_target_split and (end_frame - current_frame) % 2 == 1)):
                if verbose >= 3:
                    print(f" / split / 1.18 singleside mode / 2-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)



        if end_frame - start_frame <= 2 * zone["zone"].scene_detection_18_target_split:
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.18:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len and \
                   ((current_frame - start_frame) % 32 == 1 or (end_frame - current_frame) % 32 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.18 mode / 32-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)
                           
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.18:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len and \
                   ((current_frame - start_frame) % 16 == 1 or (end_frame - current_frame) % 16 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.18 mode / 16-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)
                           
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.18:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len and \
                   ((current_frame - start_frame) % 8 == 1 or (end_frame - current_frame) % 8 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.18 mode / 8-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)
                           
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.18:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len and \
                   ((current_frame - start_frame) % 4 == 1 or (end_frame - current_frame) % 4 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.18 mode / 4-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)
                           
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.18:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len and \
                   ((current_frame - start_frame) % 2 == 1 or (end_frame - current_frame) % 2 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.18 mode / 2-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)

            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.18:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len:
                    if verbose >= 3:
                        print(f" / split / 1.18 mode / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)

            if verbose >= 3:
                print(f" / branch complete", end="\n", flush=True)
            return [start_frame]



        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.18:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 32 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 32 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.18 singleside mode / 32-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.18:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 16 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 16 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.18 singleside mode / 16-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.18:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 8 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 8 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.18 singleside mode / 8-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.18:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 4 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 4 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.18 singleside mode / 4-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.18:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 2 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 2 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.18 singleside mode / 2-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)



        if end_frame - start_frame <= 2 * zone["zone"].scene_detection_12_target_split:
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.12:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
                   current_frame - start_frame <= zone["zone"].scene_detection_12_target_split and end_frame - current_frame <= zone["zone"].scene_detection_12_target_split and \
                   ((current_frame - start_frame) % 32 == 1 or (end_frame - current_frame) % 32 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.12 doubleside mode / 32-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)
                           
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.12:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
                   current_frame - start_frame <= zone["zone"].scene_detection_12_target_split and end_frame - current_frame <= zone["zone"].scene_detection_12_target_split and \
                   ((current_frame - start_frame) % 16 == 1 or (end_frame - current_frame) % 16 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.12 doubleside mode / 16-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)
                           
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.12:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
                   current_frame - start_frame <= zone["zone"].scene_detection_12_target_split and end_frame - current_frame <= zone["zone"].scene_detection_12_target_split and \
                   ((current_frame - start_frame) % 8 == 1 or (end_frame - current_frame) % 8 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.12 doubleside mode / 8-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)
                           
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.12:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
                   current_frame - start_frame <= zone["zone"].scene_detection_12_target_split and end_frame - current_frame <= zone["zone"].scene_detection_12_target_split and \
                   ((current_frame - start_frame) % 4 == 1 or (end_frame - current_frame) % 4 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.12 doubleside mode / 4-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)
                           
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.12:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
                   current_frame - start_frame <= zone["zone"].scene_detection_12_target_split and end_frame - current_frame <= zone["zone"].scene_detection_12_target_split and \
                   ((current_frame - start_frame) % 2 == 1 or (end_frame - current_frame) % 2 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.12 doubleside mode / 2-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)



        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.12:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_12_target_split and (current_frame - start_frame) % 32 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_12_target_split and (end_frame - current_frame) % 32 == 1)):
                if verbose >= 3:
                    print(f" / split / 1.12 singleside mode / 32-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.12:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_12_target_split and (current_frame - start_frame) % 16 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_12_target_split and (end_frame - current_frame) % 16 == 1)):
                if verbose >= 3:
                    print(f" / split / 1.12 singleside mode / 16-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.12:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_12_target_split and (current_frame - start_frame) % 8 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_12_target_split and (end_frame - current_frame) % 8 == 1)):
                if verbose >= 3:
                    print(f" / split / 1.12 singleside mode / 8-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.12:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_12_target_split and (current_frame - start_frame) % 4 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_12_target_split and (end_frame - current_frame) % 4 == 1)):
                if verbose >= 3:
                    print(f" / split / 1.12 singleside mode / 4-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.12:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_12_target_split and (current_frame - start_frame) % 2 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_12_target_split and (end_frame - current_frame) % 2 == 1)):
                if verbose >= 3:
                    print(f" / split / 1.12 singleside mode / 2-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)



        if end_frame - start_frame <= zone["zone"].scene_detection_extra_split:
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.12:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
                   ((current_frame - start_frame) % 32 == 1 or (end_frame - current_frame) % 32 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.12 mode / 32-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)
                           
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.12:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
                   ((current_frame - start_frame) % 16 == 1 or (end_frame - current_frame) % 16 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.12 mode / 16-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)
                           
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.12:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
                   ((current_frame - start_frame) % 8 == 1 or (end_frame - current_frame) % 8 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.12 mode / 8-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)
                           
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.12:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
                   ((current_frame - start_frame) % 4 == 1 or (end_frame - current_frame) % 4 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.12 mode / 4-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)
                           
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.12:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split and \
                   ((current_frame - start_frame) % 2 == 1 or (end_frame - current_frame) % 2 == 1):
                    if verbose >= 3:
                        print(f" / split / 1.12 mode / 2-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)

            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.12:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_18_target_split and end_frame - current_frame >= zone["zone"].scene_detection_18_target_split:
                    if verbose >= 3:
                        print(f" / split / 1.12 mode / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)

            if verbose >= 3:
                print(f" / branch complete", end="\n", flush=True)
            return [start_frame]



        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.12:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 32 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 32 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.12 singleside mode / 32-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.12:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 16 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 16 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.12 singleside mode / 16-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.12:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 8 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 8 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.12 singleside mode / 8-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.12:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 4 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 4 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.12 singleside mode / 4-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.12:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 2 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 2 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.12 singleside mode / 2-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)



        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.12:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               (current_frame - start_frame <= zone["zone"].scene_detection_extra_split or \
                end_frame - current_frame <= zone["zone"].scene_detection_extra_split):
                if verbose >= 3:
                    print(f" / split / extra_split 1.12 singleside mode / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)

        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.12:
                break
            if (current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len) and \
               (current_frame - start_frame <= zone["zone"].scene_detection_extra_split or \
                end_frame - current_frame <= zone["zone"].scene_detection_extra_split) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split):
                if verbose >= 3:
                    print(f" / split / extra_split 1.12 singleside mode / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                        scene_detection_split_scene(current_frame, end_frame)

        if end_frame - start_frame >= 2 * zone["zone"].scene_detection_extra_split:
            for current_frame in diffs_sort:
                if diffs[current_frame] < 1.15:
                    break
                if current_frame - start_frame >= zone["zone"].scene_detection_extra_split and end_frame - current_frame >= zone["zone"].scene_detection_extra_split and \
                   math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
                   math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
                   math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split + 0.50):
                    if verbose >= 3:
                        print(f" / split / extra_split 1.15 mode / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                    return scene_detection_split_scene(start_frame, current_frame) + \
                           scene_detection_split_scene(current_frame, end_frame)



        section_diffs = diffs[start_frame + 1:end_frame]
        section_diffs_0012 = section_diffs >= 0.0012
        section_diffs_0042 = section_diffs >= 0.0042


        if np.all(~section_diffs_0012):
            if end_frame - start_frame <= zone["zone"].scene_detection_0012_still_scene_extra_split:
                if verbose >= 3:
                    print(f" / branch complete / 0.0012 mode", end="\n", flush=True)
                return [start_frame]
            else:
                sections = math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_0012_still_scene_extra_split)
                section_frames = (end_frame - start_frame) / sections
                section_frames = np.min([math.ceil((section_frames - 1) / 16) * 16 + 1, zone["zone"].scene_detection_0012_still_scene_extra_split])
                returning_frames = []
                for frame in range(start_frame, end_frame, section_frames):
                    returning_frames.append(frame)
                if verbose >= 3:
                    print(f" / split / 0.0012 divide mode / frame {" ".join([str(item) for item in returning_frames[1:]])}", end="\n", flush=True)
                return returning_frames

        if np.all(~section_diffs_0042):
            if end_frame - start_frame <= zone["zone"].scene_detection_0042_still_scene_extra_split:
                if verbose >= 3:
                    print(f" / branch complete / 0.0042 mode", end="\n", flush=True)
                return [start_frame]
            else:
                sections = math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_0042_still_scene_extra_split)
                section_frames = (end_frame - start_frame) / sections
                section_frames = np.min([math.ceil((section_frames - 1) / 16) * 16 + 1, zone["zone"].scene_detection_0042_still_scene_extra_split])
                returning_frames = []
                for frame in range(start_frame, end_frame, section_frames):
                    returning_frames.append(frame)
                if verbose >= 3:
                    print(f" / split / 0.0042 divide mode / frame {" ".join([str(item) for item in returning_frames[1:]])}", end="\n", flush=True)
                return returning_frames


        offset_frame = np.argmax(section_diffs_0012) + 1
        reserve_offset_frame = np.argmax(section_diffs_0012[::-1]) + 1

        split_frame = np.max([end_frame - reserve_offset_frame,
                              end_frame - zone["zone"].scene_detection_0012_still_scene_extra_split,
                              start_frame + zone["zone"].scene_detection_min_scene_len])
        if end_frame - split_frame > zone["zone"].scene_detection_12_target_split and \
           math.ceil((split_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
           1 <= \
           math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split):
            if verbose >= 3:
                print(f" / split / 0.0012 rear mode / frame {split_frame}", end="\n", flush=True)
            return scene_detection_split_scene(start_frame, split_frame) + \
                   [split_frame]

        split_frame = np.min([start_frame + offset_frame,
                              start_frame + zone["zone"].scene_detection_0012_still_scene_extra_split,
                              end_frame - zone["zone"].scene_detection_min_scene_len])
        if split_frame - start_frame > zone["zone"].scene_detection_12_target_split and \
           1 + \
           math.ceil((end_frame - split_frame) / zone["zone"].scene_detection_extra_split) <= \
           math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split):
            if verbose >= 3:
                print(f" / split / 0.0012 front mode / frame {split_frame}", end="\n", flush=True)
            return [start_frame] + \
                   scene_detection_split_scene(split_frame, end_frame)


        offset_frame = np.argmax(section_diffs_0042) + 1
        reserve_offset_frame = np.argmax(section_diffs_0042[::-1]) + 1

        split_frame = np.max([end_frame - reserve_offset_frame,
                              end_frame - zone["zone"].scene_detection_0042_still_scene_extra_split,
                              start_frame + zone["zone"].scene_detection_min_scene_len])
        if end_frame - split_frame > zone["zone"].scene_detection_12_target_split and \
           math.ceil((split_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
           1 <= \
           math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split):
            if verbose >= 3:
                print(f" / split / 0.0042 rear mode / frame {split_frame}", end="\n", flush=True)
            return scene_detection_split_scene(start_frame, split_frame) + \
                   [split_frame]

        split_frame = np.min([start_frame + offset_frame,
                              start_frame + zone["zone"].scene_detection_0042_still_scene_extra_split,
                              end_frame - zone["zone"].scene_detection_min_scene_len])
        if split_frame - start_frame > zone["zone"].scene_detection_12_target_split and \
           1 + \
           math.ceil((end_frame - split_frame) / zone["zone"].scene_detection_extra_split) <= \
           math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split):
            if verbose >= 3:
                print(f" / split / 0.0042 front mode / frame {split_frame}", end="\n", flush=True)
            return [start_frame] + \
                   scene_detection_split_scene(split_frame, end_frame)



        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.08:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 32 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 32 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.08 singleside mode / 32-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.08:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 16 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 16 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.08 singleside mode / 16-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.08:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 8 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 8 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.08 singleside mode / 8-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.08:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 4 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 4 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.08 singleside mode / 4-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.08:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 2 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 2 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.08 singleside mode / 2-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)

        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.08:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               (current_frame - start_frame <= zone["zone"].scene_detection_extra_split or \
                end_frame - current_frame <= zone["zone"].scene_detection_extra_split):
                if verbose >= 3:
                    print(f" / split / extra_split 1.08 singleside mode / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)



        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.02:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 32 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 32 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.02 singleside mode / 32-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.02:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 16 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 16 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.02 singleside mode / 16-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.02:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 8 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 8 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.02 singleside mode / 8-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.02:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 4 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 4 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.02 singleside mode / 4-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.02:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 2 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 2 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 1.02 singleside mode / 2-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.02:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               (current_frame - start_frame <= zone["zone"].scene_detection_extra_split or \
                end_frame - current_frame <= zone["zone"].scene_detection_extra_split):
                if verbose >= 3:
                    print(f" / split / extra_split 1.02 singleside mode / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)



        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.08:
                break
            if (current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len) and \
               (current_frame - start_frame <= zone["zone"].scene_detection_extra_split or \
                end_frame - current_frame <= zone["zone"].scene_detection_extra_split) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split):
                if verbose >= 3:
                    print(f" / split / extra_split 1.08 mode / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                        scene_detection_split_scene(current_frame, end_frame)



        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.96:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 32 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 32 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 0.96 singleside mode / 32-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.96:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 16 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 16 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 0.96 singleside mode / 16-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.96:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 8 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 8 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 0.96 singleside mode / 8-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.96:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 4 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 4 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 0.96 singleside mode / 4-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.96:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 2 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 2 == 1)):
                if verbose >= 3:
                    print(f" / split / extra_split 0.96 singleside mode / 2-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.96:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               (current_frame - start_frame <= zone["zone"].scene_detection_extra_split or \
                end_frame - current_frame <= zone["zone"].scene_detection_extra_split):
                if verbose >= 3:
                    print(f" / split / extra_split 0.96 singleside mode / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)


                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 1.02:
                break
            if (current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len) and \
               (current_frame - start_frame <= zone["zone"].scene_detection_extra_split or \
                end_frame - current_frame <= zone["zone"].scene_detection_extra_split) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split):
                if verbose >= 3:
                    print(f" / split / extra_split 1.02 mode / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                        scene_detection_split_scene(current_frame, end_frame)



        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.84:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 32 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 32 == 1)) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split + 0.50):
                if verbose >= 3:
                    print(f" / split / extra_split 0.84 singleside mode / 32-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.84:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 16 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 16 == 1)) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split + 0.50):
                if verbose >= 3:
                    print(f" / split / extra_split 0.84 singleside mode / 16-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.84:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 8 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 8 == 1)) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split + 0.50):
                if verbose >= 3:
                    print(f" / split / extra_split 0.84 singleside mode / 8-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.84:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 4 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 4 == 1)) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split + 0.50):
                if verbose >= 3:
                    print(f" / split / extra_split 0.84 singleside mode / 4-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.84:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               ((current_frame - start_frame <= zone["zone"].scene_detection_extra_split and (current_frame - start_frame) % 2 == 1) or \
                (end_frame - current_frame <= zone["zone"].scene_detection_extra_split and (end_frame - current_frame) % 2 == 1)) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split + 0.50):
                if verbose >= 3:
                    print(f" / split / extra_split 0.84 singleside mode / 2-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)
                       
        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.84:
                break
            if current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split and \
               (current_frame - start_frame <= zone["zone"].scene_detection_extra_split or \
                end_frame - current_frame <= zone["zone"].scene_detection_extra_split) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split + 0.50):
                if verbose >= 3:
                    print(f" / split / extra_split 0.84 singleside mode / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)

        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.84:
                break
            if (current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split + 0.50):
                if verbose >= 3:
                    print(f" / split / extra_split 0.84 mode / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                        scene_detection_split_scene(current_frame, end_frame)

        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.84:
                break
            if (current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len) and \
               (current_frame - start_frame <= zone["zone"].scene_detection_extra_split or \
                end_frame - current_frame <= zone["zone"].scene_detection_extra_split) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split):
                if verbose >= 3:
                    print(f" / split / extra_split 0.84 singleside mode / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                        scene_detection_split_scene(current_frame, end_frame)



        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.96:
                break
            if (current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len) and \
               (current_frame - start_frame <= zone["zone"].scene_detection_extra_split or \
                end_frame - current_frame <= zone["zone"].scene_detection_extra_split) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split):
                if verbose >= 3:
                    print(f" / split / extra_split 0.96 mode / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                        scene_detection_split_scene(current_frame, end_frame)

                        

        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.84:
                break
            if (current_frame - start_frame >= zone["zone"].scene_detection_min_scene_len and end_frame - current_frame >= zone["zone"].scene_detection_min_scene_len) and \
               (current_frame - start_frame <= zone["zone"].scene_detection_extra_split or \
                end_frame - current_frame <= zone["zone"].scene_detection_extra_split):
                if verbose >= 3:
                    print(f" / split / extra_split 0.84 mode / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                        scene_detection_split_scene(current_frame, end_frame)



        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.09:
                break
            if (current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split) and \
               ((current_frame - start_frame) % 32 == 1 or (end_frame - current_frame) % 32 == 1) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split + 0.05):
                if verbose >= 3:
                    print(f" / split / low scenechange / 32-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)

        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.09:
                break
            if (current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split) and \
               ((current_frame - start_frame) % 16 == 1 or (end_frame - current_frame) % 16 == 1) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split + 0.05):
                if verbose >= 3:
                    print(f" / split / low scenechange / 16-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)

        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.09:
                break
            if (current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split) and \
               ((current_frame - start_frame) % 8 == 1 or (end_frame - current_frame) % 8 == 1) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split + 0.05):
                if verbose >= 3:
                    print(f" / split / low scenechange / 8-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)

        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.09:
                break
            if (current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split) and \
               ((current_frame - start_frame) % 4 == 1 or (end_frame - current_frame) % 4 == 1) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split + 0.05):
                if verbose >= 3:
                    print(f" / split / low scenechange / 4-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)

        for current_frame in diffs_sort:
            if diffs[current_frame] < 0.09:
                break
            if (current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split) and \
               ((current_frame - start_frame) % 2 == 1 or (end_frame - current_frame) % 2 == 1) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split + 0.05):
                if verbose >= 3:
                    print(f" / split / low scenechange / 2-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)


        for current_frame in diffs_sort:
            if (current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split) and \
               ((current_frame - start_frame) % 32 == 1 or (end_frame - current_frame) % 32 == 1) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split):
                if verbose >= 3:
                    print(f" / split / low scenechange / 32-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)

        for current_frame in diffs_sort:
            if (current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split) and \
               ((current_frame - start_frame) % 16 == 1 or (end_frame - current_frame) % 16 == 1) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split):
                if verbose >= 3:
                    print(f" / split / low scenechange / 16-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)

        for current_frame in diffs_sort:
            if (current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split) and \
               ((current_frame - start_frame) % 8 == 1 or (end_frame - current_frame) % 8 == 1) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split):
                if verbose >= 3:
                    print(f" / split / low scenechange / 8-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)

        for current_frame in diffs_sort:
            if (current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split) and \
               ((current_frame - start_frame) % 4 == 1 or (end_frame - current_frame) % 4 == 1) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split):
                if verbose >= 3:
                    print(f" / split / low scenechange / 4-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)

        for current_frame in diffs_sort:
            if (current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split) and \
               ((current_frame - start_frame) % 2 == 1 or (end_frame - current_frame) % 2 == 1) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split):
                if verbose >= 3:
                    print(f" / split / low scenechange / 2-frame hierarchical structure flavoured / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)

        for current_frame in diffs_sort:
            if (current_frame - start_frame >= zone["zone"].scene_detection_12_target_split and end_frame - current_frame >= zone["zone"].scene_detection_12_target_split) and \
               math.ceil((current_frame - start_frame) / zone["zone"].scene_detection_extra_split) + \
               math.ceil((end_frame - current_frame) / zone["zone"].scene_detection_extra_split) <= \
               math.ceil((end_frame - start_frame) / zone["zone"].scene_detection_extra_split):
                if verbose >= 3:
                    print(f" / split / low scenechange / frame {current_frame} / diff {np.floor(diffs[current_frame] * 100) / 100:.2f}", end="\n", flush=True)
                return scene_detection_split_scene(start_frame, current_frame) + \
                       scene_detection_split_scene(current_frame, end_frame)


        assert False, "This indicates a bug in the original code. Please report this to the repository including this entire error message."

    start_frames = scene_detection_split_scene(0, len(diffs))

    start_frames += [zone["end_frame"] - zone["start_frame"]]
    for i in range(len(start_frames) - 1):
        scenes["scenes"].append({"start_frame": start_frames[i] + zone["start_frame"],
                                 "end_frame": start_frames[i + 1] + zone["start_frame"],
                                 "zone_overrides": None})

print(f"\r\033[K{frame_scene_print(scenes["scenes"][-1]["start_frame"], scenes["scenes"][-1]["end_frame"])} / Scene creation complete", end="\n", flush=True)

with scene_detection_scenes_file.open("w") as scenes_f:
    json.dump(scenes, scenes_f, cls=NumpyEncoder)

if not scene_detection_diffs_available:
    np.savetxt(scene_detection_diffs_file, scene_detection_diffs, fmt="%.9f")
    np.savetxt(scene_detection_average_file, scene_detection_average, fmt="%.9f")
    np.savetxt(scene_detection_min_file, scene_detection_min, fmt="%.9f")
    np.savetxt(scene_detection_max_file, scene_detection_max, fmt="%.9f")
    scene_detection_diffs_available = True

scenes["split_scenes"] = scenes["scenes"]
with scenes_file.open("w") as scenes_f:
    json.dump(scenes, scenes_f, cls=NumpyEncoder)

print(f"\r\033[KTime {datetime.now().time().isoformat(timespec="seconds")} / Progressive Scene Detection finished", end="\n", flush=True)
