#!/usr/bin/env python3
# verify.py
#
# Integrity check before final mux.
# On failure:
#   - write single-line sanitized errorInfo to: %WORKDIR%\00_logs\verify_error.txt
#   - exit with non-zero code

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple


MIN_BYTES_SUB = 0          # subtitles can be tiny
MIN_BYTES_ATTACHMENT = 16
MIN_BYTES_AUDIO = 1024
MIN_BYTES_VIDEO = 1024 * 256  # 256 KiB; adjust if you want stricter
STATE_DIR_NAME = ".state"
VERIFY_MARKER = "VERIFY_DONE"


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
    log_fh = p.open("a", encoding=enc, errors="replace")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        log_fh.write(f"=== START verify {ts} ===\n")
        log_fh.flush()
    except Exception:
        pass
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    tee_out = TeeStream(orig_stdout, log_fh)
    tee_err = TeeStream(orig_stderr, log_fh)
    sys.stdout = tee_out
    sys.stderr = tee_err

    def _cleanup() -> None:
        ts_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            log_fh.write(f"=== END verify {ts_end} ===\n")
            log_fh.flush()
        except Exception:
            pass
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        tee_out.close_log()
        tee_err.close_log()

    atexit.register(_cleanup)


def marker_path(workdir: Path) -> Path:
    return workdir / STATE_DIR_NAME / VERIFY_MARKER


def write_marker(workdir: Path) -> None:
    p = marker_path(workdir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("ok\n", encoding="utf-8")


def sanitize_error_id(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s).strip())
    if not s:
        s = "verify_failed"
    return s[:80]


def write_verify_error(workdir: Path, err: str) -> None:
    logs = workdir / "00_logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "verify_error.txt").write_text(sanitize_error_id(err) + "\n", encoding="utf-8")


def resolve_rel_to_workdir(workdir: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (workdir / pp)


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def which_or(path_or_name: str) -> Optional[str]:
    if not path_or_name:
        return None
    if Path(path_or_name).exists():
        return path_or_name
    return shutil.which(path_or_name)


def run_ffprobe_json(ffprobe: str, path: Path) -> Dict[str, Any]:
    cmd = [
        ffprobe,
        "-hide_banner",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe_failed rc={p.returncode} out={p.stdout[:2000]}")
    try:
        return json.loads(p.stdout)
    except Exception as ex:
        raise RuntimeError(f"ffprobe_json_parse_failed: {ex}")


def ffprobe_has_stream(js: Dict[str, Any], codec_type: str) -> bool:
    for s in js.get("streams") or []:
        if str(s.get("codec_type") or "") == codec_type:
            return True
    return False


def ffprobe_duration_ms(js: Dict[str, Any]) -> int:
    fmt = js.get("format") or {}
    dur = fmt.get("duration")
    if dur is None:
        # fallback: first stream duration
        for s in js.get("streams") or []:
            if s.get("duration") is not None:
                dur = s.get("duration")
                break
    try:
        return int(float(dur) * 1000.0) if dur is not None else 0
    except Exception:
        return 0


def source_has_chapters(source: Path, ffprobe: Optional[str]) -> Optional[bool]:
    if not ffprobe:
        return None
    cmd = [
        ffprobe,
        "-hide_banner",
        "-v", "error",
        "-print_format", "json",
        "-show_chapters",
        str(source),
    ]
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if p.returncode != 0:
        return None
    try:
        js = json.loads(p.stdout)
    except Exception:
        return None
    chapters = js.get("chapters") or []
    return len(chapters) > 0


def check_file_exists(p: Path, min_bytes: int, err_id: str) -> None:
    if not p.exists():
        raise RuntimeError(err_id)
    try:
        if p.stat().st_size < min_bytes:
            raise RuntimeError(err_id)
    except FileNotFoundError:
        raise RuntimeError(err_id)


def parse_tracks(tracks_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    tracks = tracks_json.get("tracks")
    if not isinstance(tracks, list):
        raise RuntimeError("invalid_tracksjson_no_tracks")
    out: List[Dict[str, Any]] = []
    for t in tracks:
        if isinstance(t, dict):
            out.append(t)
    return out


def norm_type(t: str) -> str:
    v = (t or "").strip().lower()
    if v.startswith("sub") or v == "subtitle":
        return "sub"
    if v.startswith("aud") or v == "audio":
        return "audio"
    if v.startswith("vid") or v == "video":
        return "video"
    return v


def is_skip(status: str) -> bool:
    return (status or "").strip().upper() == "SKIP"


def verify_video(workdir: Path, tracks: List[Dict[str, Any]], ffprobe: Optional[str]) -> Tuple[Optional[Path], int]:
    # If any video track is not SKIP -> expect video-final.mkv
    need_video = any(norm_type(str(t.get("type") or "")) == "video" and not is_skip(str(t.get("trackStatus") or "")) for t in tracks)
    if not need_video:
        print("[verify] video: no active video tracks => skip video-final check")
        return None, 0

    vpath = workdir / "video" / "video-final.mkv"
    print(f"[verify] video: checking {vpath}")
    check_file_exists(vpath, MIN_BYTES_VIDEO, "missing_or_too_small_video_final")

    if ffprobe:
        js = run_ffprobe_json(ffprobe, vpath)
        if not ffprobe_has_stream(js, "video"):
            raise RuntimeError("video_final_no_video_stream")
        dms = ffprobe_duration_ms(js)
        if dms <= 0:
            raise RuntimeError("video_final_bad_duration")
        return vpath, dms

    return vpath, 0


def verify_audio(workdir: Path, tracks: List[Dict[str, Any]], ffprobe: Optional[str]) -> List[Tuple[Path, int]]:
    audio_tracks = [
        t for t in tracks
        if norm_type(str(t.get("type") or "")) == "audio" and not is_skip(str(t.get("trackStatus") or ""))
    ]
    if not audio_tracks:
        print("[verify] audio: no active audio tracks")
        return []

    manifest_path = workdir / "00_meta" / "audio_manifest.json"
    if not manifest_path.exists():
        # Backward-compatible fallback: check that audio dir has something, but do not hard-fail.
        print("[verify] audio: audio_manifest.json missing; fallback to directory sanity only")
        adir = workdir / "audio"
        if not adir.exists():
            raise RuntimeError("missing_audio_dir_no_manifest")
        any_file = any(p.is_file() and p.stat().st_size >= MIN_BYTES_AUDIO for p in adir.glob("*.*"))
        if not any_file:
            raise RuntimeError("audio_dir_empty_no_manifest")
        return []

    man = read_json(manifest_path)
    outs = man.get("outputs")
    if not isinstance(outs, list):
        raise RuntimeError("invalid_audio_manifest_outputs")

    # Build: trackId -> list[output]
    by_id: Dict[int, List[Dict[str, Any]]] = {}
    for o in outs:
        if not isinstance(o, dict):
            continue
        try:
            tid = int(o.get("srcTrackId"))
        except Exception:
            continue
        by_id.setdefault(tid, []).append(o)

    results: List[Tuple[Path, int]] = []

    for t in audio_tracks:
        tid = int(t.get("trackId"))
        lst = by_id.get(tid) or []
        # Require primary
        prim = [o for o in lst if str(o.get("role") or "") == "primary"]
        if not prim:
            raise RuntimeError(f"audio_missing_primary_track_{tid}")

        # Verify each referenced file exists; at least primary must be good
        for o in prim:
            rel = str(o.get("outPath") or "")
            if not rel:
                raise RuntimeError(f"audio_manifest_bad_outPath_track_{tid}")
            p = resolve_rel_to_workdir(workdir, rel)
            print(f"[verify] audio: trackId={tid} primary -> {p.name}")
            check_file_exists(p, MIN_BYTES_AUDIO, f"audio_missing_file_track_{tid}")

            dms = 0
            if ffprobe:
                js = run_ffprobe_json(ffprobe, p)
                if not ffprobe_has_stream(js, "audio"):
                    raise RuntimeError(f"audio_file_no_audio_stream_track_{tid}")
                dms = ffprobe_duration_ms(js)
                if dms <= 0:
                    raise RuntimeError(f"audio_bad_duration_track_{tid}")
            results.append((p, dms))

    return results


def verify_demux_outputs(workdir: Path, source: Path, ffprobe: Optional[str]) -> None:
    demux_manifest = workdir / "00_meta" / "demux_manifest.json"
    if not demux_manifest.exists():
        print("[verify] demux: demux_manifest.json missing => skip demux checks")
        return

    man = read_json(demux_manifest)

    # subs
    subs = man.get("subs") or []
    if isinstance(subs, list):
        for s in subs:
            if not isinstance(s, dict):
                continue
            p = s.get("path")
            if not p:
                continue
            pp = Path(p)
            # manifest might store absolute; but accept both
            if not pp.is_absolute():
                pp = workdir / pp
            print(f"[verify] sub: {pp.name}")
            check_file_exists(pp, MIN_BYTES_SUB, "sub_missing_or_too_small")

    # attachments
    atts = man.get("attachments") or []
    if isinstance(atts, list):
        for a in atts:
            if not isinstance(a, dict):
                continue
            p = a.get("path")
            if not p:
                continue
            pp = Path(p)
            if not pp.is_absolute():
                pp = workdir / pp
            print(f"[verify] att: {pp.name}")
            check_file_exists(pp, MIN_BYTES_ATTACHMENT, "attachment_missing_or_too_small")

    # chapters
    ch = man.get("chapters") or {}
    if isinstance(ch, dict):
        has_chapters = source_has_chapters(source, ffprobe)
        if has_chapters is False:
            print("[verify] chapters: none in source => skip")
            return
        if has_chapters is None:
            print("[verify] chapters: ffprobe unavailable => skip")
            return
        p = ch.get("path")
        if p:
            pp = Path(p)
            if not pp.is_absolute():
                pp = workdir / pp
            print(f"[verify] chapters: {pp.name}")
            check_file_exists(pp, 16, "chapters_missing_or_too_small")
        else:
            raise RuntimeError("chapters_missing_or_too_small")


def verify_duration_consistency(
    video_dms: int,
    audio_list: List[Tuple[Path, int]],
    tolerance_ms: int = 5000,
) -> None:
    if video_dms <= 0:
        return
    audio_durations = [d for (_, d) in audio_list if d > 0]
    if not audio_durations:
        return
    amin, amax = min(audio_durations), max(audio_durations)
    # compare against video
    if abs(video_dms - amin) > tolerance_ms or abs(video_dms - amax) > tolerance_ms:
        raise RuntimeError("duration_mismatch_audio_vs_video")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="verify")
    ap.add_argument("--source", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--tracksData", required=True, help="Relative is relative to --workdir")
    ap.add_argument("--ffprobe", default="ffprobe", help="Optional; used if available")
    ap.add_argument("--log", default="", help="Optional log file path (relative to --workdir if not absolute)")
    args = ap.parse_args(argv)

    source = Path(args.source)
    workdir = Path(args.workdir)
    setup_logging(args.log, workdir)
    tracks_path = resolve_rel_to_workdir(workdir, args.tracksData)
    marker = marker_path(workdir)
    if marker.exists():
        print(f"[verify] skip: marker exists: {marker}")
        return 0

    try:
        if not source.exists():
            write_verify_error(workdir, "missing_source")
            eprint("[verify] ERROR: source missing")
            return 2
        if source.suffix.lower() != ".mkv":
            write_verify_error(workdir, "source_not_mkv")
            eprint("[verify] ERROR: source is not mkv")
            return 2
        if not workdir.exists():
            write_verify_error(workdir, "missing_workdir")
            eprint("[verify] ERROR: workdir missing")
            return 2
        if not tracks_path.exists():
            write_verify_error(workdir, "missing_tracksjson")
            eprint(f"[verify] ERROR: tracks.json missing: {tracks_path}")
            return 2

        ffprobe = which_or(args.ffprobe)
        if ffprobe:
            print(f"[verify] ffprobe: {ffprobe}")
        else:
            print("[verify] ffprobe: not found => media-structure checks will be skipped")

        tracks_json = read_json(tracks_path)
        tracks = parse_tracks(tracks_json)

        # 1) demux artifacts if manifest exists
        verify_demux_outputs(workdir, source, ffprobe)

        # 2) video
        vpath, v_dms = verify_video(workdir, tracks, ffprobe)

        # 3) audio
        audio_list = verify_audio(workdir, tracks, ffprobe)

        # 4) optional duration consistency
        if ffprobe and v_dms > 0 and audio_list:
            verify_duration_consistency(v_dms, audio_list, tolerance_ms=5000)

        # If we got here => OK
        # Clean previous verify_error.txt if present (optional)
        try:
            p = workdir / "00_logs" / "verify_error.txt"
            if p.exists():
                p.unlink()
        except Exception:
            pass

        print("[verify] OK")
        write_marker(workdir)
        return 0

    except Exception as ex:
        err = sanitize_error_id(str(ex))
        write_verify_error(workdir, err)
        eprint(f"[verify] FAIL: {err} ({ex})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
