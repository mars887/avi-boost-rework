"""Single source of truth for the on-disk layout of a pipeline workdir.

Every relative directory name, scene-file name, manifest name, state-marker
directory and final-output naming convention used by the pipeline is defined
here. Producers (``runner.session`` builds the av1an / zone-editor / hdr-patch
commands), consumers (``runner_state`` resume/validation checks,
``manage.reset`` / ``manage.scenes`` / ``mux``) and the GUIs must all go through
this module instead of hardcoding a path string, so renaming a file is a
one-line change here.

This module is a leaf: it imports only :mod:`pathlib` so anything may import it
without risking a cycle.

Workdir layout (relative to ``<workdir>``)::

    00_meta/                      metadata + manifests (source_info, demux, ...)
    00_logs/                      per-stage log files
    .state/                       stage-completion markers (demux..mux)
    attachments/                  attachment-cleanup output
    audio/                        audio-tool output
    video/                        all video-pipeline artifacts
        .state/                   auto-boost stage markers (psd/fastpass/ssimu2)
        psd/scenes.psd.json       PSD scene-detection base scenes
        fastpass/scenes.json      av1an scene-detection scenes
        fastpass/<stem>.fastpass.mkv
        fastpass/<stem>_ssimu2.log
        scenes.json               stage-4 scenes (full mode)
        scenes-preview.json       stage-4 scenes (fastpass mode)
        scenes-boundaries.json    Zone Boundaries output
        scenes-recalc.json        Zone CRF Recalc output
        scenes-zoned.json         Zone Edit output
        scenes-final.json         HDR Patch output (mainpass input)
        video-final.mkv           mainpass encode output
        mainpass/ , hdr_tmp/      av1an temp dirs

The user-facing result lives next to the source as ``<stem>-av1.mkv``.
"""

from __future__ import annotations

from pathlib import Path

# --- top-level workdir subdirectories -------------------------------------
META_DIRNAME = "00_meta"
LOGS_DIRNAME = "00_logs"
STATE_DIRNAME = ".state"
VIDEO_DIRNAME = "video"
ATTACHMENTS_DIRNAME = "attachments"
AUDIO_DIRNAME = "audio"
SUB_DIRNAME = "sub"
CHAPTERS_DIRNAME = "chapters"

# --- video/ pipeline subdirectories ---------------------------------------
PSD_DIRNAME = "psd"
FASTPASS_DIRNAME = "fastpass"
MAINPASS_DIRNAME = "mainpass"
HDR_TMP_DIRNAME = "hdr_tmp"
VPY_TEMP_DIRNAME = "vpy_temp"                      # video/vpy_temp/<pass>/ scratch for .vpy scripts
VPY_TEMP_PASSES = ("proxy", "fast", "main")        # per-pass scratch subdirs

# --- av1an per-pass state files (inside video/<pass>/) --------------------
CHUNKS_JSON_NAME = "chunks.json"                   # av1an encode queue (array)
DONE_JSON_NAME = "done.json"                       # av1an completed-chunk state (object)
ENCODE_DIRNAME = "encode"                          # video/<pass>/encode/<index>.<ext>

# --- scene / output file names (inside video/, unless noted) --------------
PSD_BASE_SCENES_NAME = "scenes.psd.json"          # video/psd/
AV1AN_SCENES_NAME = "scenes.json"                 # video/fastpass/ (av1an SD)
STAGE4_SCENES_FULL_NAME = "scenes.json"           # video/ (full mode)
STAGE4_SCENES_PREVIEW_NAME = "scenes-preview.json"  # video/ (fastpass mode)
BOUNDARY_SCENES_NAME = "scenes-boundaries.json"
RECALC_SCENES_NAME = "scenes-recalc.json"
ZONED_SCENES_NAME = "scenes-zoned.json"
FINAL_SCENES_NAME = "scenes-final.json"
LEGACY_HDR_SCENES_NAME = "scenes-hdr.json"        # legacy; kept only for cleanup
VIDEO_FINAL_OUTPUT_NAME = "video-final.mkv"

# --- meta manifests (inside 00_meta/) -------------------------------------
SOURCE_INFO_NAME = "source_info.json"
DEMUX_MANIFEST_NAME = "demux_manifest.json"
AUDIO_MANIFEST_NAME = "audio_manifest.json"
MUX_MANIFEST_NAME = "mux_manifest.json"
RUNNER_STATE_NAME = "runner_state.json"
RUNNER_EVENTS_NAME = "runner_events.jsonl"

# --- other artifacts ------------------------------------------------------
ATTACHMENTS_REPORT_NAME = "attachments_cleaner_report.json"  # attachments/

# --- final output naming (next to the source file) ------------------------
FINAL_OUTPUT_SUFFIX = "-av1"   # <source_stem>-av1.mkv
FINAL_OUTPUT_EXT = ".mkv"      # the encode result
# Extensions a pipeline final output may carry: .mkv (encode) or .mp4 (optional
# web export). Used to recognise "already encoded" sources during discovery.
FINAL_OUTPUT_EXTS = (".mkv", ".mp4")

# Ordered newest-first so consumers that just want "the best scenes available"
# (e.g. mux bitrate reporting) can pick the first that exists.
SCENE_FILE_FALLBACK_ORDER = (
    FINAL_SCENES_NAME,
    ZONED_SCENES_NAME,
    RECALC_SCENES_NAME,
    BOUNDARY_SCENES_NAME,
    LEGACY_HDR_SCENES_NAME,
    STAGE4_SCENES_FULL_NAME,
)


# --- directory helpers ----------------------------------------------------
def meta_dir(workdir: Path) -> Path:
    return Path(workdir) / META_DIRNAME


def logs_dir(workdir: Path) -> Path:
    return Path(workdir) / LOGS_DIRNAME


def state_dir(workdir: Path) -> Path:
    return Path(workdir) / STATE_DIRNAME


def attachments_dir(workdir: Path) -> Path:
    return Path(workdir) / ATTACHMENTS_DIRNAME


def audio_dir(workdir: Path) -> Path:
    return Path(workdir) / AUDIO_DIRNAME


def sub_dir(workdir: Path) -> Path:
    return Path(workdir) / SUB_DIRNAME


def chapters_dir(workdir: Path) -> Path:
    return Path(workdir) / CHAPTERS_DIRNAME


def video_dir(workdir: Path) -> Path:
    return Path(workdir) / VIDEO_DIRNAME


def video_state_dir(workdir: Path) -> Path:
    """Auto-boost stage markers live under ``video/.state``."""
    return video_dir(workdir) / STATE_DIRNAME


def psd_dir(workdir: Path) -> Path:
    return video_dir(workdir) / PSD_DIRNAME


def fastpass_dir(workdir: Path) -> Path:
    return video_dir(workdir) / FASTPASS_DIRNAME


def mainpass_dir(workdir: Path) -> Path:
    return video_dir(workdir) / MAINPASS_DIRNAME


def encode_dir(pass_dir: Path) -> Path:
    """av1an encode-output directory of a pass temp dir (video/<pass>/encode)."""
    return Path(pass_dir) / ENCODE_DIRNAME


def chunks_json_path(pass_dir: Path) -> Path:
    return Path(pass_dir) / CHUNKS_JSON_NAME


def done_json_path(pass_dir: Path) -> Path:
    return Path(pass_dir) / DONE_JSON_NAME


def hdr_tmp_dir(workdir: Path) -> Path:
    return video_dir(workdir) / HDR_TMP_DIRNAME


def vpy_temp_dir(workdir: Path, pass_name: str, *, split: bool = False) -> Path:
    """Scratch directory a .vpy script may use for temp files. Forwarded to
    scripts as the ``temp=`` vspipe arg.

    ``split`` toggles (plan ``video.experimental.vpy_temp_split``):
      - False (default): one shared ``<workdir>/video/vpy_temp/`` for all passes.
      - True: per-pass ``<workdir>/video/vpy_temp/<pass>/`` (pass in VPY_TEMP_PASSES,
        i.e. proxy/fast/main).
    """
    base = video_dir(workdir) / VPY_TEMP_DIRNAME
    if not split:
        return base
    name = str(pass_name or "").strip() or "fast"
    return base / name


# --- scene / output file helpers ------------------------------------------
def psd_base_scenes(workdir: Path) -> Path:
    return psd_dir(workdir) / PSD_BASE_SCENES_NAME


def av1an_scenes(workdir: Path) -> Path:
    return fastpass_dir(workdir) / AV1AN_SCENES_NAME


def stage4_scenes(workdir: Path, mode: str) -> Path:
    name = STAGE4_SCENES_PREVIEW_NAME if str(mode).strip().lower() == "fastpass" else STAGE4_SCENES_FULL_NAME
    return video_dir(workdir) / name


def boundary_scenes(workdir: Path) -> Path:
    return video_dir(workdir) / BOUNDARY_SCENES_NAME


def recalc_scenes(workdir: Path) -> Path:
    return video_dir(workdir) / RECALC_SCENES_NAME


def zoned_scenes(workdir: Path) -> Path:
    return video_dir(workdir) / ZONED_SCENES_NAME


def final_scenes(workdir: Path) -> Path:
    return video_dir(workdir) / FINAL_SCENES_NAME


def legacy_hdr_scenes(workdir: Path) -> Path:
    return video_dir(workdir) / LEGACY_HDR_SCENES_NAME


def video_final_output(workdir: Path) -> Path:
    return video_dir(workdir) / VIDEO_FINAL_OUTPUT_NAME


def fastpass_output(workdir: Path, source_stem: str) -> Path:
    return fastpass_dir(workdir) / f"{source_stem}.fastpass.mkv"


def ssimu2_log(workdir: Path, source_stem: str) -> Path:
    return fastpass_dir(workdir) / f"{source_stem}_ssimu2.log"


# --- meta manifest helpers ------------------------------------------------
def source_info_path(workdir: Path) -> Path:
    return meta_dir(workdir) / SOURCE_INFO_NAME


def demux_manifest_path(workdir: Path) -> Path:
    return meta_dir(workdir) / DEMUX_MANIFEST_NAME


def audio_manifest_path(workdir: Path) -> Path:
    return meta_dir(workdir) / AUDIO_MANIFEST_NAME


def mux_manifest_path(workdir: Path) -> Path:
    return meta_dir(workdir) / MUX_MANIFEST_NAME


def runner_state_path(workdir: Path) -> Path:
    return meta_dir(workdir) / RUNNER_STATE_NAME


def runner_events_path(workdir: Path) -> Path:
    return meta_dir(workdir) / RUNNER_EVENTS_NAME


def attachments_report_path(workdir: Path) -> Path:
    return attachments_dir(workdir) / ATTACHMENTS_REPORT_NAME


# --- final result naming --------------------------------------------------
def final_output_path_for_source(source: Path, *, ext: str = FINAL_OUTPUT_EXT) -> Path:
    source = Path(source)
    return source.parent / f"{source.stem}{FINAL_OUTPUT_SUFFIX}{ext}"


def strip_final_output_suffix(stem: str) -> str:
    """Inverse of the ``-av1`` naming: ``episode-av1`` -> ``episode``."""
    return stem[: -len(FINAL_OUTPUT_SUFFIX)] if str(stem).lower().endswith(FINAL_OUTPUT_SUFFIX) else stem


def is_final_output_name(name: str) -> bool:
    """True if ``name`` is a pipeline final output (``<stem>-av1.mkv`` / ``.mp4``)."""
    low = str(name).lower()
    return any(low.endswith(f"{FINAL_OUTPUT_SUFFIX}{ext}") for ext in FINAL_OUTPUT_EXTS)
