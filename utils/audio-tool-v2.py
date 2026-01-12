#!/usr/bin/env python3
# audio-tool.py
#
# Audio processing utility for a single source MKV based on tracks.json.
#
# Key behaviors (updated per January 2026 clarifications):
# - Source is always MKV.
# - COPY: extract/remux without transcoding; container/extension chosen adaptively (not necessarily .mka).
# - EDIT: encode ONLY via opusenc (opus-tools). Decode via ffmpeg, optionally via pipe (ffmpeg -> opusenc).
#         For complex inputs, create intermediate FLAC/WAV before opusenc (policy-controlled).
# - Optional preservation of "special" original tracks (TrueHD / DTS-HD / Atmos-ish) as an extra .mka alongside .opus.
#
# Outputs:
#   %WORKDIR%\audio\        (final artifacts)
#   %WORKDIR%\audio\tmp\    (intermediate)
#   %WORKDIR%\00_meta\audio_manifest.json
#   %WORKDIR%\00_logs\audio-tool-{error}.txt on failure
#
# Notes:
# - Re-encoding to Opus necessarily decodes the input first; codec-specific bitstream features
#   (e.g. object-based metadata) cannot be carried inside Opus. The optional "preserve-special"
#   sidecar keeps the original bitstream in an .mka fallback.

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

TOOL_NAME = "audio-tool"
TOOL_VERSION = "1.1"

MIN_OUT_BYTES = 1024  # sanity check
DEFAULT_TMP_CODEC = "flac"  # preferred intermediate for "complex" EDIT
STATE_DIR_NAME = ".state"
AUDIO_MARKER = "AUDIO_DONE"


class AudioToolError(RuntimeError):
    def __init__(self, err_id: str, message: str):
        super().__init__(message)
        self.err_id = err_id


def eprint(*a: Any) -> None:
    print(*a, file=sys.stderr)


class TeeStream:
    def __init__(self, stream: TextIO, log_file: TextIO) -> None:
        self._stream = stream
        self._log: Optional[TextIO] = log_file

    def write(self, s: str) -> int:
        try:
            self._stream.write(s)
            self._stream.flush()
        except Exception:
            pass
        if self._log is not None:
            try:
                self._log.write(s)
                self._log.flush()
            except Exception:
                self._log = None
        return len(s)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception:
            pass
        if self._log is not None:
            try:
                self._log.flush()
            except Exception:
                self._log = None

    def close_log(self) -> None:
        if self._log is None:
            return
        try:
            self._log.flush()
        except Exception:
            pass
        try:
            self._log.close()
        except Exception:
            pass
        self._log = None

    def isatty(self) -> bool:
        return bool(getattr(self._stream, "isatty", lambda: False)())

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8")


def setup_logging(log_path: str, workdir: Optional[Path] = None) -> None:
    if not log_path:
        return
    p = Path(log_path)
    if not p.is_absolute() and workdir is not None:
        p = workdir / p
    p.parent.mkdir(parents=True, exist_ok=True)
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    log_fh = p.open("w", encoding=enc, errors="replace")
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    tee_out = TeeStream(orig_stdout, log_fh)
    tee_err = TeeStream(orig_stderr, log_fh)
    sys.stdout = tee_out
    sys.stderr = tee_err

    def _cleanup() -> None:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        tee_out.close_log()
        tee_err.close_log()

    atexit.register(_cleanup)


def marker_path(workdir: Path) -> Path:
    return workdir / STATE_DIR_NAME / AUDIO_MARKER


def write_marker(workdir: Path) -> None:
    p = marker_path(workdir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("ok\n", encoding="utf-8")


def sanitize_error_id(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip())
    if not s:
        s = "unknown_error"
    return s[:80]


def write_error_marker(workdir: Path, err_id: str) -> None:
    logs = workdir / "00_logs"
    logs.mkdir(parents=True, exist_ok=True)
    p = logs / f"{TOOL_NAME}-{sanitize_error_id(err_id)}.txt"
    try:
        p.write_text(sanitize_error_id(err_id) + "\n", encoding="utf-8")
    except Exception:
        pass


def which_or_path(p: str) -> str:
    # Accept explicit path or search PATH.
    if not p:
        return p
    if os.path.isabs(p) or (len(p) > 2 and p[1] == ":" and (p[2] == "\\" or p[2] == "/")):
        return p
    found = shutil.which(p)
    return found or p


def run_cmd(
    cmd: List[str],
    *,
    check: bool,
    capture: bool,
    cwd: Optional[Path] = None,
    text_mode: bool = True,
) -> subprocess.CompletedProcess:
    kwargs: Dict[str, Any] = {}
    if cwd is not None:
        kwargs["cwd"] = str(cwd)
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    # For binary pipes (ffmpeg -> opusenc) we use Popen directly, not this helper.
    return subprocess.run(cmd, check=check, text=text_mode, **kwargs)


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def ffprobe_audio_info(ffprobe: str, path: Path) -> Dict[str, Any]:
    """
    Return minimal audio stream info for a single-audio file (or first audio stream):
    codec, container (best-effort), channels, sample_rate, bitrate_kbps, duration_ms, profile, tags(title/lang).
    """
    cmd = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    r = run_cmd(cmd, check=True, capture=True)
    js = json.loads((r.stdout or "") if isinstance(r.stdout, str) else (r.stdout or b"").decode("utf-8", "ignore"))

    streams = js.get("streams") or []
    a = None
    for s in streams:
        if str(s.get("codec_type") or "") == "audio":
            a = s
            break
    if a is None:
        raise AudioToolError("ffprobe_no_audio_stream", f"No audio stream in: {path}")

    fmt = js.get("format") or {}
    # container: use format_name first component
    container = ""
    try:
        container = str(fmt.get("format_name") or "").split(",")[0]
    except Exception:
        container = ""

    codec = str(a.get("codec_name") or "")
    profile = str(a.get("profile") or "")
    channels = a.get("channels")
    sr = a.get("sample_rate")
    duration = fmt.get("duration") or a.get("duration")
    bit_rate = a.get("bit_rate") or fmt.get("bit_rate")

    # tags
    tags = a.get("tags") or {}
    title = str(tags.get("title") or "")
    lang = str(tags.get("language") or "")

    out: Dict[str, Any] = {
        "container": container,
        "codec": codec.lower(),
        "profile": profile,
        "channels": int(channels) if channels is not None else None,
        "sample_rate": int(sr) if sr is not None else None,
        "duration_ms": int(float(duration) * 1000.0) if duration is not None else 0,
        "bitrate_kbps": int(int(bit_rate) / 1000) if bit_rate is not None and str(bit_rate).isdigit() else None,
        "title": title,
        "lang": lang,
    }
    return out


def mkvmerge_copy_audio_track(mkvmerge: str, source: Path, track_id: int, out_mka: Path) -> None:
    """
    Extract one audio track by Matroska track ID into an audio-only Matroska (.mka) without transcoding.
    """
    cmd = [
        mkvmerge,
        "-o",
        str(out_mka),
        "--audio-tracks",
        str(track_id),
        "--no-video",
        "--no-subtitles",
        "--no-buttons",
        "--no-chapters",
        "--no-attachments",
        "--no-global-tags",
        str(source),
    ]
    run_cmd(cmd, check=True, capture=True)


def ffmpeg_copy_remux(ffmpeg: str, in_mka: Path, out_path: Path, *, threads: Optional[int]) -> None:
    """
    Remux/copy the first audio stream from in_mka into out_path without transcoding.
    """
    cmd = [ffmpeg, "-hide_banner", "-v", "error"]
    if threads and threads > 0:
        cmd += ["-threads", str(threads)]
    cmd += ["-i", str(in_mka), "-map", "0:a:0", "-vn", "-sn", "-dn", "-c:a", "copy"]

    # For raw bitstream containers, be explicit where reasonable.
    suf = out_path.suffix.lower()
    if suf == ".ac3":
        cmd += ["-f", "ac3"]
    elif suf == ".eac3":
        cmd += ["-f", "eac3"]
    elif suf == ".dts":
        cmd += ["-f", "dts"]
    elif suf in (".thd", ".truehd"):
        cmd += ["-f", "truehd"]
    elif suf == ".opus":
        # Ogg Opus
        cmd += ["-f", "opus"]
    # m4a/mp3/flac/ogg/mka inferred
    cmd += [str(out_path)]
    run_cmd(cmd, check=True, capture=True)


def choose_copy_ext(codec: str, profile: str, policy: str) -> str:
    """
    Decide output extension for COPY.
    policy:
      - mka: always .mka
      - native: prefer codec-native container/bitstream extension
      - auto: pick a reasonable container (native for many, mka fallback)
    """
    codec = (codec or "").lower()
    prof = (profile or "").lower()

    if policy == "mka":
        return ".mka"

    # "native"/"auto" mapping
    if codec == "aac":
        # AAC is best carried as .m4a for broad tool support
        return ".m4a"
    if codec == "alac":
        return ".m4a"
    if codec == "mp3":
        return ".mp3"
    if codec == "flac":
        return ".flac"
    if codec == "opus":
        return ".opus"
    if codec == "vorbis":
        return ".ogg"
    if codec == "ac3":
        return ".ac3"
    if codec == "eac3":
        return ".eac3"
    if codec == "truehd":
        # Keep TrueHD bitstream as raw; Matroska also works, but raw is a clean "native" artifact.
        return ".thd"
    if codec == "dts":
        return ".dts"
    if codec.startswith("pcm_"):
        return ".wav"

    # DTS-HD profiles still show codec "dts"; keep .dts above.
    # Unknown / exotic: fall back to Matroska audio container.
    return ".mka"


def relpath_under(workdir: Path, p: Path) -> str:
    try:
        return str(p.relative_to(workdir))
    except Exception:
        return str(p)


def parse_profile(track: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (mode, params)
    mode: always "opus" (COPY vs EDIT is driven by trackStatus)
    params: {"bitrate_kbps": int, "target_channels": Optional[int]}
    """
    tp = track.get("trackParam") or {}
    if not isinstance(tp, dict):
        tp = {}

    def as_int(val: Any) -> Optional[int]:
        try:
            return int(str(val))
        except Exception:
            return None

    # Prefer explicit fields.
    bitrate_kbps: Optional[int] = None
    channels: Optional[int] = None
    bitrate_kbps = as_int(tp.get("bitrate"))
    channels = as_int(tp.get("channels"))

    if bitrate_kbps is None or bitrate_kbps <= 0:
        bitrate_kbps = 128  # contract default

    # Downmix only if explicitly 1 or 2; else preserve.
    target_channels = channels if channels in (1, 2) else None

    return "opus", {"bitrate_kbps": bitrate_kbps, "target_channels": target_channels}


@dataclass
class OutputEntry:
    srcTrackId: int
    status: str              # COPY | EDIT (from tracks.json)
    role: str                # primary | fallback_original (optional)
    outPath: str             # relative to workdir
    container: str
    codec: str
    channels: Optional[int]
    sample_rate: Optional[int]
    bitrate_kbps: Optional[int]
    duration_ms: int
    notes: str = ""


def is_special_source(info: Dict[str, Any], track_meta: Dict[str, Any]) -> bool:
    """
    Heuristic: mark sources where an Opus encode will inevitably lose meaningful bitstream features.
    """
    codec = (info.get("codec") or "").lower()
    profile = (info.get("profile") or "")
    title = (info.get("title") or "") + " " + str(track_meta.get("origName") or "") + " " + str((track_meta.get("trackMux") or {}).get("name") or "")
    tl = title.lower()

    if codec == "truehd":
        return True
    if codec == "dts":
        # DTS-HD MA / HRA typically report codec "dts" + profile
        if "hd" in str(profile).lower():
            return True
        return True  # treat all DTS as special for preservation purposes
    if codec == "eac3" and "atmos" in tl:
        return True
    if "atmos" in tl:
        return True
    return False


def build_ffmpeg_decode_args(
    in_mka: Path,
    *,
    target_channels: Optional[int],
    force_ar_48000_on_downmix: bool,
    codec: str,
    threads: Optional[int],
) -> List[str]:
    """
    Build ffmpeg args for decoding from in_mka to a lossless intermediate (FLAC/WAV),
    applying only *minimal* transformations required by profile.
    """
    args: List[str] = ["-hide_banner", "-v", "error"]
    if threads and threads > 0:
        args += ["-threads", str(threads)]

    # Contract guidance: disable DRC for AC3/EAC3 if possible.
    if codec in ("ac3", "eac3"):
        args += ["-drc_scale", "0"]

    args += ["-i", str(in_mka), "-map", "0:a:0", "-vn", "-sn", "-dn"]

    if target_channels in (1, 2):
        args += ["-ac", str(target_channels)]
        if force_ar_48000_on_downmix:
            args += ["-ar", "48000"]

    return args


def encode_opus_with_opusenc(
    *,
    ffmpeg: str,
    opusenc: str,
    in_mka: Path,
    out_opus: Path,
    bitrate_kbps: int,
    target_channels: Optional[int],
    threads: Optional[int],
    temp_policy: str,
    src_info: Dict[str, Any],
    track_meta: Dict[str, Any],
) -> None:
    """
    Encode in_mka to out_opus via opusenc.
    temp_policy:
      - none: always pipe ffmpeg -> opusenc (no intermediate file)
      - auto: pipe for "simple"; intermediate FLAC for "complex"
      - flac: intermediate FLAC
      - wav: intermediate WAV
    """

    # Determine "complexity" (heuristic)
    codec = (src_info.get("codec") or "").lower()
    src_ch = int(src_info.get("channels") or 0) if src_info.get("channels") is not None else 0
    needs_downmix = target_channels in (1, 2) and (src_ch != target_channels)
    is_simple = (
        codec in ("aac", "mp3", "flac", "vorbis", "opus", "ac3", "eac3")
        and src_ch in (1, 2)
        and not needs_downmix
    )
    # Special formats are "complex" by definition for this policy.
    if is_special_source(src_info, track_meta):
        is_simple = False

    # Decide intermediate mode
    use_pipe = False
    intermediate_kind: Optional[str] = None
    if temp_policy == "none":
        use_pipe = True
    elif temp_policy == "auto":
        use_pipe = is_simple
        intermediate_kind = None if use_pipe else DEFAULT_TMP_CODEC
    elif temp_policy in ("flac", "wav"):
        use_pipe = False
        intermediate_kind = temp_policy
    else:
        # defensive
        use_pipe = is_simple
        intermediate_kind = None if use_pipe else DEFAULT_TMP_CODEC

    ff_args = build_ffmpeg_decode_args(
        in_mka,
        target_channels=target_channels,
        force_ar_48000_on_downmix=True,
        codec=codec,
        threads=threads,
    )

    # Metadata for opusenc
    title = str((track_meta.get("trackMux") or {}).get("name") or track_meta.get("origName") or "").strip()
    lang = str((track_meta.get("trackMux") or {}).get("lang") or track_meta.get("origLang") or "").strip()

    opus_cmd: List[str] = [
        opusenc,
        "--bitrate",
        str(int(bitrate_kbps)),
        "--vbr",
        "--comp",
        "10",
    ]
    if title:
        opus_cmd += ["--title", title]
    if lang:
        opus_cmd += ["--comment", f"LANGUAGE={lang}"]
    # Preserve provenance as tags for later debugging
    if codec:
        opus_cmd += ["--comment", f"SRC_CODEC={codec}"]
    prof = str(src_info.get("profile") or "").strip()
    if prof:
        opus_cmd += ["--comment", f"SRC_PROFILE={prof}"]

    if use_pipe:
        # Use lossless FLAC stream as the pipe payload: smaller than WAV, and opusenc supports FLAC on stdin.
        # opusenc can read from stdin when input is "-".
        ff_cmd = [ffmpeg] + ff_args + ["-f", "flac", "-compression_level", "0", "-"]
        # For stdin pipelines, opusenc sometimes needs --ignorelength (especially for WAV),
        # but it's harmless and improves robustness with pipes.
        opus_cmd2 = opus_cmd + ["--ignorelength", "-", str(out_opus)]

        # Run pipeline
        with subprocess.Popen(ff_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p_ff:
            with subprocess.Popen(opus_cmd2, stdin=p_ff.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p_op:
                assert p_ff.stdout is not None
                p_ff.stdout.close()  # allow ffmpeg to receive SIGPIPE
                # Prevent communicate() from trying to read a closed stdout pipe.
                p_ff.stdout = None
                out_op, err_op = p_op.communicate()
                _, err_ff = p_ff.communicate()

                if p_ff.returncode != 0:
                    raise AudioToolError(
                        f"ffmpeg_failed_decode_pipe",
                        f"ffmpeg decode (pipe) failed: rc={p_ff.returncode} stderr={err_ff.decode('utf-8','ignore')[:4000]}",
                    )
                if p_op.returncode != 0:
                    raise AudioToolError(
                        f"opusenc_failed_encode",
                        f"opusenc failed: rc={p_op.returncode} stderr={err_op.decode('utf-8','ignore')[:4000]}",
                    )
        return

    # Intermediate file mode
    tmp_dir = out_opus.parent / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_in = tmp_dir / f"{out_opus.stem}.pcm.{intermediate_kind}"

    try:
        if intermediate_kind == "wav":
            # 24-bit PCM WAV is a decent compromise; opusenc reads WAV input.
            ff_cmd = [ffmpeg] + ff_args + ["-c:a", "pcm_s24le", "-f", "wav", str(tmp_in)]
        else:
            # FLAC lossless; opusenc reads FLAC input.
            ff_cmd = [ffmpeg] + ff_args + ["-f", "flac", "-compression_level", "0", str(tmp_in)]
        run_cmd(ff_cmd, check=True, capture=True, text_mode=True)

        op_cmd = opus_cmd + [str(tmp_in), str(out_opus)]
        r = run_cmd(op_cmd, check=False, capture=True, text_mode=True)
        if r.returncode != 0:
            raise AudioToolError(
                "opusenc_failed_encode",
                f"opusenc failed: rc={r.returncode} stderr={(r.stderr or '')[:4000]}",
            )
    finally:
        try:
            if tmp_in.exists():
                tmp_in.unlink()
        except Exception:
            pass


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog=TOOL_NAME)
    ap.add_argument("--source", required=True, help="Path to source MKV")
    ap.add_argument("--workdir", required=True, help="Episode workdir")
    ap.add_argument("--tracksData", required=True, help="Path to tracks.json (relative is relative to --workdir)")

    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--mkvmerge", default="mkvmerge")
    ap.add_argument("--opusenc", default="opusenc")

    ap.add_argument("--temp-policy", default="auto", choices=["auto", "flac", "wav", "none"])
    ap.add_argument("--copy-container", default="auto", choices=["auto", "mka", "native"])
    ap.add_argument("--no-preserve-special", action="store_true", help="Disable extra .mka copy for special formats in EDIT")

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--log-json", default="", help="Optional machine log path")
    ap.add_argument("--log", default="", help="Optional text log file path (relative to --workdir if not absolute)")

    args = ap.parse_args(argv)

    source = Path(args.source)
    workdir = Path(args.workdir)
    setup_logging(args.log, workdir)
    tracks_data = Path(args.tracksData)
    marker = marker_path(workdir)
    if marker.exists() and not args.overwrite:
        print(f"[{TOOL_NAME}] skip: marker exists: {marker}")
        return 0

    if not source.exists():
        write_error_marker(workdir, "audio_missing_source")
        eprint(f"[{TOOL_NAME}] ERROR: source not found: {source}")
        return 2

    if source.suffix.lower() != ".mkv":
        # user clarified: always mkv
        write_error_marker(workdir, "audio_invalid_source_container")
        eprint(f"[{TOOL_NAME}] ERROR: source must be MKV: {source}")
        return 2

    if not tracks_data.is_absolute():
        tracks_data = workdir / tracks_data
    if not tracks_data.exists():
        write_error_marker(workdir, "audio_missing_tracksjson")
        eprint(f"[{TOOL_NAME}] ERROR: tracks.json not found: {tracks_data}")
        return 2

    ffmpeg = which_or_path(args.ffmpeg)
    ffprobe = which_or_path(args.ffprobe)
    mkvmerge = which_or_path(args.mkvmerge)
    opusenc = which_or_path(args.opusenc)

    # tool presence checks (best-effort)
    for (tool, err_id) in [(ffmpeg, "ffmpeg_not_found"), (ffprobe, "ffprobe_not_found"), (mkvmerge, "mkvmerge_not_found"), (opusenc, "opusenc_not_found")]:
        if shutil.which(tool) is None and not Path(tool).exists():
            write_error_marker(workdir, err_id)
            eprint(f"[{TOOL_NAME}] ERROR: tool not found: {tool}")
            return 2

    audio_dir = workdir / "audio"
    tmp_dir = audio_dir / "tmp"
    meta_dir = workdir / "00_meta"
    audio_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    machine_log: Dict[str, Any] = {
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "source": str(source),
        "workdir": str(workdir),
        "events": [],
        "started_at": time.time(),
    }

    def log_event(ev: Dict[str, Any]) -> None:
        machine_log["events"].append({"t": time.time(), **ev})

    try:
        data = read_json(tracks_data)
        tracks = data.get("tracks") if isinstance(data, dict) else None
        if not isinstance(tracks, list):
            raise AudioToolError("invalid_tracksjson", "tracks.json has no 'tracks' array")

        outputs: List[OutputEntry] = []

        for t in tracks:
            if not isinstance(t, dict):
                continue
            t_type = str(t.get("type") or "").lower().strip()
            if t_type not in ("audio", "aud"):
                continue

            status = str(t.get("trackStatus") or "").upper().strip()
            track_id = t.get("trackId")
            file_base = str(t.get("fileBase") or "").strip()
            if not file_base:
                raise AudioToolError("invalid_track_filebase", f"Track missing fileBase: {t}")

            try:
                track_id_int = int(track_id)
            except Exception:
                raise AudioToolError("invalid_track_id", f"Invalid trackId: {track_id}")

            if status == "SKIP":
                print(f"[{TOOL_NAME}] trackId={track_id_int} SKIP fileBase={file_base}")
                log_event({"trackId": track_id_int, "status": "SKIP"})
                continue
            if status not in ("COPY", "EDIT"):
                raise AudioToolError("invalid_track_status", f"Invalid trackStatus={status} for trackId={track_id_int}")

            mode, p = parse_profile(t)

            # Step 1: produce a stable per-track .mka (source bitstream, no transcode)
            tmp_in = tmp_dir / f"{file_base}.src.mka"
            if not tmp_in.exists() or args.overwrite:
                print(f"[{TOOL_NAME}] trackId={track_id_int} extract-> {tmp_in.name}")
                log_event({"trackId": track_id_int, "action": "extract_mka", "tmp": str(tmp_in)})
                mkvmerge_copy_audio_track(mkvmerge, source, track_id_int, tmp_in)

            # Probe source track info from tmp_in
            src_info = ffprobe_audio_info(ffprobe, tmp_in)

            if status == "COPY":
                ext = choose_copy_ext(src_info.get("codec") or "", src_info.get("profile") or "", args.copy_container)
                out_path = audio_dir / f"{file_base}{ext}"
                print(f"[{TOOL_NAME}] trackId={track_id_int} COPY -> {out_path.name}")
                log_event({"trackId": track_id_int, "status": "COPY", "out": str(out_path), "ext": ext})

                if out_path.exists() and not args.overwrite and out_path.stat().st_size >= MIN_OUT_BYTES:
                    # ok
                    info = ffprobe_audio_info(ffprobe, out_path)
                    outputs.append(OutputEntry(
                        srcTrackId=track_id_int,
                        status="COPY",
                        role="primary",
                        outPath=relpath_under(workdir, out_path),
                        container=info.get("container") or out_path.suffix.lstrip("."),
                        codec=info.get("codec") or "",
                        channels=info.get("channels"),
                        sample_rate=info.get("sample_rate"),
                        bitrate_kbps=info.get("bitrate_kbps"),
                        duration_ms=int(info.get("duration_ms") or 0),
                        notes="skip_exists",
                    ))
                    continue

                # If ext is .mka, simply copy the tmp_in
                if out_path.suffix.lower() == ".mka":
                    shutil.copyfile(tmp_in, out_path)
                else:
                    ffmpeg_copy_remux(ffmpeg, tmp_in, out_path, threads=(args.threads or None))

                if not out_path.exists() or out_path.stat().st_size < MIN_OUT_BYTES:
                    raise AudioToolError(f"copy_failed_track_{track_id_int}", f"COPY output not created or too small: {out_path}")

                info = ffprobe_audio_info(ffprobe, out_path)
                outputs.append(OutputEntry(
                    srcTrackId=track_id_int,
                    status="COPY",
                    role="primary",
                    outPath=relpath_under(workdir, out_path),
                    container=info.get("container") or out_path.suffix.lstrip("."),
                    codec=info.get("codec") or "",
                    channels=info.get("channels"),
                    sample_rate=info.get("sample_rate"),
                    bitrate_kbps=info.get("bitrate_kbps"),
                    duration_ms=int(info.get("duration_ms") or 0),
                ))
                continue

            # EDIT
            out_opus = audio_dir / f"{file_base}.opus"
            bitrate_kbps = int(p.get("bitrate_kbps") or 128)
            target_channels = p.get("target_channels")

            print(f"[{TOOL_NAME}] trackId={track_id_int} EDIT -> {out_opus.name} (br={bitrate_kbps}k ch={'keep' if target_channels is None else target_channels})")
            log_event({"trackId": track_id_int, "status": "EDIT", "out": str(out_opus), "bitrate": bitrate_kbps, "target_channels": target_channels})

            # Optional preservation of special formats as extra .mka:
            preserve_special = not bool(args.no_preserve_special)
            is_special = is_special_source(src_info, t)
            if preserve_special and is_special:
                out_orig = audio_dir / f"{file_base}.mka"
                if not out_orig.exists() or args.overwrite:
                    print(f"[{TOOL_NAME}] trackId={track_id_int} preserve-special -> {out_orig.name}")
                    shutil.copyfile(tmp_in, out_orig)
                # Add to manifest as fallback_original
                if out_orig.exists() and out_orig.stat().st_size >= MIN_OUT_BYTES:
                    info_o = ffprobe_audio_info(ffprobe, out_orig)
                    outputs.append(OutputEntry(
                        srcTrackId=track_id_int,
                        status="EDIT",
                        role="fallback_original",
                        outPath=relpath_under(workdir, out_orig),
                        container=info_o.get("container") or "matroska",
                        codec=info_o.get("codec") or "",
                        channels=info_o.get("channels"),
                        sample_rate=info_o.get("sample_rate"),
                        bitrate_kbps=info_o.get("bitrate_kbps"),
                        duration_ms=int(info_o.get("duration_ms") or 0),
                        notes="preserve_special_original_bitstream",
                    ))

            # Encode opus via opusenc (optionally skip if exists)
            if out_opus.exists() and not args.overwrite and out_opus.stat().st_size >= MIN_OUT_BYTES:
                info = ffprobe_audio_info(ffprobe, out_opus)
                outputs.append(OutputEntry(
                    srcTrackId=track_id_int,
                    status="EDIT",
                    role="primary",
                    outPath=relpath_under(workdir, out_opus),
                    container=info.get("container") or "opus",
                    codec=info.get("codec") or "opus",
                    channels=info.get("channels"),
                    sample_rate=info.get("sample_rate"),
                    bitrate_kbps=info.get("bitrate_kbps") or bitrate_kbps,
                    duration_ms=int(info.get("duration_ms") or 0),
                    notes="skip_exists",
                ))
                continue

            encode_opus_with_opusenc(
                ffmpeg=ffmpeg,
                opusenc=opusenc,
                in_mka=tmp_in,
                out_opus=out_opus,
                bitrate_kbps=bitrate_kbps,
                target_channels=target_channels,
                threads=(args.threads or None),
                temp_policy=args.temp_policy,
                src_info=src_info,
                track_meta=t,
            )

            if not out_opus.exists() or out_opus.stat().st_size < MIN_OUT_BYTES:
                raise AudioToolError(
                    f"opus_failed_track_{track_id_int}",
                    f"EDIT output not created or too small: {out_opus}",
                )

            info = ffprobe_audio_info(ffprobe, out_opus)
            outputs.append(OutputEntry(
                srcTrackId=track_id_int,
                status="EDIT",
                role="primary",
                outPath=relpath_under(workdir, out_opus),
                container=info.get("container") or "opus",
                codec=info.get("codec") or "opus",
                channels=info.get("channels"),
                sample_rate=info.get("sample_rate"),
                bitrate_kbps=info.get("bitrate_kbps") or bitrate_kbps,
                duration_ms=int(info.get("duration_ms") or 0),
            ))

        manifest = {
            "source": str(source),
            "workdir": str(workdir),
            "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
            "outputs": [asdict(o) for o in outputs],
        }
        write_json(meta_dir / "audio_manifest.json", manifest)
        machine_log["finished_at"] = time.time()

        if args.log_json:
            log_path = Path(args.log_json)
            if not log_path.is_absolute():
                log_path = workdir / log_path
            try:
                write_json(log_path, machine_log)
            except Exception:
                pass

        print(f"[{TOOL_NAME}] done. outputs={len(outputs)} manifest=00_meta/audio_manifest.json")
        write_marker(workdir)
        return 0

    except AudioToolError as ex:
        err_id = ex.err_id
        write_error_marker(workdir, err_id)
        eprint(f"[{TOOL_NAME}] ERROR: {err_id}: {ex}")
        # machine log best-effort
        machine_log["finished_at"] = time.time()
        machine_log["error_id"] = err_id
        machine_log["error_message"] = str(ex)
        if args.log_json:
            log_path = Path(args.log_json)
            if not log_path.is_absolute():
                log_path = workdir / log_path
            try:
                write_json(log_path, machine_log)
            except Exception:
                pass
        return 1

    except subprocess.CalledProcessError as ex:
        err_id = "tool_failed"
        write_error_marker(workdir, err_id)
        eprint(f"[{TOOL_NAME}] ERROR: {err_id}: rc={ex.returncode} cmd={ex.cmd}")
        if getattr(ex, "stderr", None):
            try:
                eprint(str(ex.stderr)[:4000])
            except Exception:
                pass
        return 1

    except Exception as ex:
        err_id = "unhandled_exception"
        write_error_marker(workdir, err_id)
        eprint(f"[{TOOL_NAME}] ERROR: {err_id}: {repr(ex)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
