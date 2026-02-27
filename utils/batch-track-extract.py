#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


WIN_BAD = r'<>:"/\|?*'
WIN_BAD_RE = re.compile(rf"[{re.escape(WIN_BAD)}]")


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def run_cmd(cmd: List[str]) -> Tuple[bool, str]:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return (proc.returncode == 0, (proc.stdout or "").strip())


def run_mkvmerge_json(mkvmerge: str, source: Path) -> Dict[str, Any]:
    ok, output = run_cmd([mkvmerge, "-J", str(source)])
    if not ok:
        raise RuntimeError(f"mkvmerge -J failed for {source}:\n{output}")
    try:
        return json.loads(output)
    except json.JSONDecodeError as ex:
        raise RuntimeError(f"mkvmerge JSON parse failed for {source}: {ex}")


def normalize_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sub"):
        return "sub"
    if text.startswith("aud"):
        return "audio"
    if text.startswith("vid"):
        return "video"
    return text


def sanitize_component(name: str, *, default: str, max_len: int = 120) -> str:
    text = (name or "").strip()
    text = WIN_BAD_RE.sub("_", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip(". ")
    if not text:
        text = default
    if len(text) > max_len:
        text = text[:max_len].rstrip(". ")
    if not text:
        text = default
    return text


def pick_track_name(track: Dict[str, Any]) -> str:
    mux = track.get("trackMux") if isinstance(track.get("trackMux"), dict) else {}
    candidate = str((mux or {}).get("name") or "").strip()
    if candidate and candidate != "-":
        return candidate
    original = str(track.get("origName") or "").strip()
    if original and original != "-":
        return original
    track_type = normalize_type(track.get("type"))
    track_id = track.get("trackId")
    return f"{track_type}_{track_id}"


def pick_track_lang(track: Dict[str, Any]) -> str:
    mux = track.get("trackMux") if isinstance(track.get("trackMux"), dict) else {}
    lang = str((mux or {}).get("lang") or "").strip()
    if lang:
        return lang
    original = str(track.get("origLang") or "").strip()
    if original:
        return original
    return "und"


def default_track_name(track_type: str, track_id: Any) -> str:
    return f"{track_type}_{track_id}"


def find_track_info(mkv_json: Dict[str, Any], track_id: int) -> Optional[Dict[str, Any]]:
    for track in mkv_json.get("tracks", []) or []:
        try:
            tid = int(track.get("id"))
        except Exception:
            continue
        if tid == track_id:
            return track
    return None


def sub_ext_from_codec(codec_id: str) -> Optional[str]:
    c = (codec_id or "").upper()
    if "S_TEXT/ASS" in c:
        return ".ass"
    if "S_TEXT/SSA" in c:
        return ".ssa"
    if "S_TEXT/UTF8" in c:
        return ".srt"
    if "S_TEXT/WEBVTT" in c:
        return ".vtt"
    if "S_TEXT/USF" in c:
        return ".usf"
    if "S_TEXT/TIMEDTEXT" in c or "S_TEXT/TTML" in c:
        return ".ttml"
    if "S_HDMV/PGS" in c:
        return ".sup"
    if "S_VOBSUB" in c or "S_DVBSUB" in c:
        return ".sub"
    return None


def audio_ext_from_codec(codec_id: str) -> Optional[str]:
    c = (codec_id or "").upper()
    if "A_AAC" in c:
        return ".aac"
    if "A_EAC3" in c:
        return ".eac3"
    if "A_AC3" in c:
        return ".ac3"
    if "A_DTS" in c:
        return ".dts"
    if "A_TRUEHD" in c:
        return ".thd"
    if "A_FLAC" in c:
        return ".flac"
    if "A_OPUS" in c:
        return ".opus"
    if "A_VORBIS" in c:
        return ".ogg"
    if "A_MPEG/L3" in c:
        return ".mp3"
    if "A_MPEG/L2" in c:
        return ".mp2"
    if "A_PCM" in c:
        return ".wav"
    return None


def video_ext_from_codec(codec_id: str) -> Optional[str]:
    c = (codec_id or "").upper()
    if "V_MPEG4/ISO/AVC" in c:
        return ".h264"
    if "V_MPEGH/ISO/HEVC" in c:
        return ".h265"
    if "V_AV1" in c:
        return ".av1"
    if "V_VP9" in c:
        return ".vp9"
    if "V_VP8" in c:
        return ".vp8"
    if "V_MPEG2" in c:
        return ".m2v"
    return None


def native_ext(track_type: str, codec_id: str) -> Optional[str]:
    if track_type == "sub":
        return sub_ext_from_codec(codec_id)
    if track_type == "audio":
        return audio_ext_from_codec(codec_id)
    if track_type == "video":
        return video_ext_from_codec(codec_id)
    return None


def fallback_container_ext(track_type: str) -> str:
    if track_type == "audio":
        return ".mka"
    if track_type == "sub":
        return ".mks"
    if track_type == "video":
        return ".mkv"
    return ".mkv"


def mkvmerge_extract_cmd(mkvmerge: str, source: Path, track_id: int, track_type: str, out_path: Path) -> List[str]:
    base = [
        mkvmerge,
        "-o",
        str(out_path),
        "--no-chapters",
        "--no-attachments",
        "--no-global-tags",
        "--no-track-tags",
    ]
    if track_type == "audio":
        base += ["--audio-tracks", str(track_id), "--no-video", "--no-subtitles"]
    elif track_type == "sub":
        base += ["--subtitle-tracks", str(track_id), "--no-video", "--no-audio"]
    elif track_type == "video":
        base += ["--video-tracks", str(track_id), "--no-audio", "--no-subtitles"]
    else:
        raise RuntimeError(f"Unsupported track type for mkvmerge extract: {track_type}")
    base.append(str(source))
    return base


def choose_output_path(
    folder: Path,
    stem: str,
    ext: str,
    used_paths: set[str],
    *,
    overwrite: bool,
) -> Path:
    stem_safe = sanitize_component(stem, default="source", max_len=150)
    candidate = folder / f"{stem_safe}{ext}"

    def available(path: Path) -> bool:
        key = str(path).lower()
        if key in used_paths:
            return False
        if overwrite:
            return True
        return not path.exists()

    if available(candidate):
        used_paths.add(str(candidate).lower())
        return candidate

    index = 2
    while True:
        alt = folder / f"{stem_safe}__{index}{ext}"
        if available(alt):
            used_paths.add(str(alt).lower())
            return alt
        index += 1


def extract_single_track(
    *,
    source: Path,
    track_id: int,
    track_type: str,
    codec_id: str,
    dst_folder: Path,
    source_stem: str,
    overwrite: bool,
    mkvextract: Optional[str],
    mkvmerge: str,
    used_paths: set[str],
) -> Tuple[bool, str, Optional[Path]]:
    use_native = bool(mkvextract) and source.suffix.lower() == ".mkv"
    native = native_ext(track_type, codec_id) if use_native else None

    if native:
        out_path = choose_output_path(dst_folder, source_stem, native, used_paths, overwrite=overwrite)
        cmd = [mkvextract, "tracks", str(source), f"{track_id}:{out_path}"]
        ok, output = run_cmd(cmd)
        if ok and out_path.exists():
            return True, "", out_path
        eprint(f"[warn] mkvextract failed, fallback to mkvmerge: {source.name} track={track_id}")
        if output:
            eprint(output)

    ext = fallback_container_ext(track_type)
    out_path = choose_output_path(dst_folder, source_stem, ext, used_paths, overwrite=overwrite)
    cmd = mkvmerge_extract_cmd(mkvmerge, source, track_id, track_type, out_path)
    ok, output = run_cmd(cmd)
    if ok and out_path.exists():
        return True, "", out_path
    return False, output or "unknown extraction error", out_path


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        eprint("No extraction plan received on stdin.")
        return 1

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as ex:
        eprint(f"Invalid extraction plan JSON: {ex}")
        return 1

    items = payload.get("items")
    if not isinstance(items, list) or not items:
        eprint("Extraction plan has no items.")
        return 1

    overwrite = bool(payload.get("overwrite"))
    output_root_raw = str(payload.get("outputRoot") or "").strip()
    output_root = Path(output_root_raw) if output_root_raw else Path.cwd()
    output_root.mkdir(parents=True, exist_ok=True)
    eprint(f"[extract] output root: {output_root}")
    eprint(f"[extract] overwrite: {'yes' if overwrite else 'no'}")

    mkvmerge = shutil.which("mkvmerge") or "mkvmerge"
    if shutil.which(mkvmerge) is None and not Path(mkvmerge).exists():
        eprint("mkvmerge not found in PATH.")
        return 1
    mkvextract = shutil.which("mkvextract")

    probe_cache: Dict[str, Dict[str, Any]] = {}
    grouped: "OrderedDict[Tuple[str, int], Dict[str, Any]]" = OrderedDict()

    for item in items:
        if not isinstance(item, dict):
            continue
        source = Path(str(item.get("source") or "")).absolute()
        tracks = item.get("tracks")
        if not source.exists() or not isinstance(tracks, list):
            continue

        for track in tracks:
            if not isinstance(track, dict):
                continue
            status = str(track.get("trackStatus") or "").upper()
            if status == "SKIP":
                continue
            track_type = normalize_type(track.get("type"))
            if track_type not in ("audio", "sub", "video"):
                continue

            try:
                track_id = int(track.get("trackId"))
            except Exception:
                continue

            name = pick_track_name(track)
            lang = pick_track_lang(track)
            fallback_name = default_track_name(track_type, track_id)
            key = (track_type, track_id)
            if key not in grouped:
                grouped[key] = {
                    "type": track_type,
                    "track_id": track_id,
                    "name": name,
                    "lang": lang,
                    "members": [],
                }
            else:
                current_name = grouped[key]["name"]
                if current_name == fallback_name and name != fallback_name:
                    grouped[key]["name"] = name
                current_lang = str(grouped[key].get("lang") or "")
                if (not current_lang or current_lang == "und") and lang and lang != "und":
                    grouped[key]["lang"] = lang
            grouped[key]["members"].append((source, track))

    if not grouped:
        eprint("[extract] no tracks selected for extraction.")
        out = {
            "status": "ok",
            "outputRoot": str(output_root),
            "processed": 0,
            "extracted": 0,
            "failed": 0,
            "groups": [],
        }
        sys.stdout.write(json.dumps(out, ensure_ascii=False))
        sys.stdout.flush()
        return 0

    used_folder_names: set[str] = set()
    used_output_paths: set[str] = set()
    groups_out: List[Dict[str, Any]] = []
    errors: List[str] = []
    processed = 0
    extracted = 0
    failed = 0
    total_groups = len(grouped)
    eprint(f"[extract] groups: {total_groups}")

    for group_idx, group in enumerate(grouped.values(), start=1):
        folder_base = sanitize_component(group["name"], default=f"{group['type']}_track")
        folder_name = folder_base
        index = 2
        while folder_name.lower() in used_folder_names:
            folder_name = f"{folder_base}__{index}"
            index += 1
        used_folder_names.add(folder_name.lower())

        folder_path = output_root / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        eprint(
            f"[group {group_idx}/{total_groups}] "
            f"type={group['type']} track={group.get('track_id')} "
            f"folder={folder_name} items={len(group['members'])}"
        )

        group_extracted = 0
        for source, track in group["members"]:
            processed += 1
            try:
                track_id = int(track.get("trackId"))
            except Exception:
                failed += 1
                errors.append(f"{source.name}: invalid trackId={track.get('trackId')}")
                continue

            if not source.exists():
                failed += 1
                errors.append(f"{source}: source file missing")
                continue

            source_key = str(source).lower()
            if source_key not in probe_cache:
                try:
                    probe_cache[source_key] = run_mkvmerge_json(mkvmerge, source)
                except Exception as ex:
                    failed += 1
                    errors.append(f"{source.name}: {ex}")
                    continue

            probe = probe_cache[source_key]
            track_info = find_track_info(probe, track_id)
            track_type = normalize_type(track.get("type"))
            if track_info and (track_type not in ("audio", "sub", "video")):
                track_type = normalize_type(track_info.get("type"))
            if track_type not in ("audio", "sub", "video"):
                failed += 1
                errors.append(f"{source.name}: unsupported track type for track {track_id}")
                continue

            codec_id = ""
            if track_info and isinstance(track_info.get("properties"), dict):
                codec_id = str(track_info["properties"].get("codec_id") or "")

            ok, err, out_path = extract_single_track(
                source=source,
                track_id=track_id,
                track_type=track_type,
                codec_id=codec_id,
                dst_folder=folder_path,
                source_stem=source.stem,
                overwrite=overwrite,
                mkvextract=mkvextract,
                mkvmerge=mkvmerge,
                used_paths=used_output_paths,
            )
            if ok:
                extracted += 1
                group_extracted += 1
                eprint(f"[ok] {source.name} track={track_id} -> {out_path}")
            else:
                failed += 1
                errors.append(f"{source.name}: track={track_id}: {err}")
                eprint(f"[fail] {source.name} track={track_id}: {err}")

        groups_out.append({"folder": folder_name, "tracks": group_extracted})
        eprint(f"[group done] {folder_name}: extracted={group_extracted}/{len(group['members'])}")

    status = "ok" if failed == 0 else "partial"
    out = {
        "status": status,
        "outputRoot": str(output_root),
        "processed": processed,
        "extracted": extracted,
        "failed": failed,
        "groups": groups_out,
    }
    if errors:
        out["errors"] = errors

    eprint(
        f"[extract done] processed={processed} extracted={extracted} failed={failed} status={status}"
    )

    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
