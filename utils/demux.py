#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import atexit
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple


WIN_BAD = r'<>:"/\|?*'
WIN_BAD_RE = re.compile(rf"[{re.escape(WIN_BAD)}]")


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


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


def setup_logging(log_path: Optional[str], workdir: Optional[Path] = None) -> None:
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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def which_or(name: str, fallback: str) -> str:
    return shutil.which(name) or fallback


def safe_filename(name: str, max_len: int = 180) -> str:
    name = (name or "").strip()
    if not name:
        return "unnamed"
    name = WIN_BAD_RE.sub("_", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = name.rstrip(". ")
    if not name:
        name = "unnamed"
    if len(name) > max_len:
        name = name[:max_len].rstrip(". ")
    return name


PROGRESS_RE = re.compile(r"^(progress|processed)\s*[:\-]?\s*\d+%\s*$", re.IGNORECASE)


def is_progress_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if PROGRESS_RE.match(s):
        return True
    if s.endswith("%") and any(ch.isdigit() for ch in s) and len(s) <= 40:
        return True
    return False


def run_cmd(cmd: List[str]) -> None:
    # Важно: encoding/errors фиксируем, иначе Windows "charmap" может упасть.
    print("[cmd]", " ".join(cmd))
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    progress_width = 0
    had_progress = False
    if p.stdout is not None:
        for line in p.stdout:
            if is_progress_line(line):
                s = line.strip()
                if len(s) < progress_width:
                    s = s + (" " * (progress_width - len(s)))
                progress_width = max(progress_width, len(s))
                sys.stdout.write(s + "\r")
                sys.stdout.flush()
                had_progress = True
                continue

            if had_progress:
                sys.stdout.write("\n")
                sys.stdout.flush()
                had_progress = False
                progress_width = 0

            if line:
                sys.stdout.write(line)
                if not line.endswith("\n"):
                    sys.stdout.write("\n")
                sys.stdout.flush()

    rc = p.wait()
    if had_progress:
        sys.stdout.write("\n")
        sys.stdout.flush()
    if rc != 0:
        raise RuntimeError(f"Command failed with code {rc}")


def load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_tracks_path(workdir: Path, tracks_data_arg: str) -> Path:
    td = Path(tracks_data_arg)
    return td if td.is_absolute() else (workdir / td)


@dataclass
class TrackEntry:
    trackId: int
    type: str
    trackStatus: str
    fileBase: str


def normalize_type(raw: str) -> str:
    v = (raw or "").strip().lower()
    if v.startswith("sub") or v == "subtitle":
        return "sub"
    if v.startswith("aud") or v == "audio":
        return "audio"
    if v.startswith("vid") or v == "video":
        return "video"
    return v


def is_skip(status: str) -> bool:
    return (status or "").strip().upper() == "SKIP"


def get_mkvmerge_json(mkvmerge: str, source: Path) -> Dict[str, Any]:
    cmd = [mkvmerge, "-J", str(source)]
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
        raise RuntimeError(f"mkvmerge -J failed ({p.returncode}). Output:\n{p.stdout}")

    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError as ex:
        raise RuntimeError(f"Failed to parse mkvmerge JSON: {ex}\nOutput:\n{p.stdout}")


def find_track_info(mkvj: Dict[str, Any], track_id: int) -> Optional[Dict[str, Any]]:
    for t in mkvj.get("tracks", []) or []:
        if int(t.get("id", -1)) == int(track_id):
            return t
    return None


def sub_ext_from_codec(codec_id: str) -> str:
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
    if "S_VOBSUB" in c:
        # может быть пара idx+sub; оставляем .sub как базовый
        return ".sub"
    if "S_DVBSUB" in c:
        return ".sub"
    return ".sub"


def extract_subtitles(
    mkvextract: str,
    source: Path,
    sub_dir: Path,
    track_entries: List[TrackEntry],
    mkvj: Dict[str, Any],
    overwrite: bool,
) -> List[Dict[str, Any]]:
    extracted: List[Dict[str, Any]] = []
    if not track_entries:
        print("[demux] No subtitle tracks to extract.")
        return extracted

    specs: List[str] = []
    for te in track_entries:
        ti = find_track_info(mkvj, te.trackId)
        codec_id = ""
        if ti:
            codec_id = ((ti.get("properties") or {}).get("codec_id") or "")
        ext = sub_ext_from_codec(codec_id)

        out_name = safe_filename(te.fileBase) + ext
        out_path = sub_dir / out_name

        if out_path.exists() and not overwrite:
            print(f"[demux] SUB skip exists: {out_path.name}")
            extracted.append({"trackId": te.trackId, "path": str(out_path), "skipped": True, "codec_id": codec_id})
            continue

        ensure_dir(sub_dir)
        specs.append(f"{te.trackId}:{out_path}")
        extracted.append({"trackId": te.trackId, "path": str(out_path), "skipped": False, "codec_id": codec_id})

    if specs:
        cmd = [mkvextract, "tracks", str(source)] + specs
        run_cmd(cmd)

    return extracted


def ext_from_mime(content_type: str) -> str:
    ct = (content_type or "").lower().strip()
    if ct in ("image/jpeg", "image/jpg"):
        return ".jpg"
    if ct == "image/png":
        return ".png"
    if ct == "image/webp":
        return ".webp"
    if ct == "image/bmp":
        return ".bmp"
    if ct == "image/gif":
        return ".gif"
    if "opentype" in ct:
        return ".otf"
    if "truetype" in ct or "x-font-ttf" in ct:
        return ".ttf"
    return ""


def guess_name_from_attachment_dict(att: dict) -> str:
    """
    Best-effort: пытаемся найти строку типа 'cover.jpg' в самом dict, т.к. иногда она лежит не в file_name.
    """
    candidates = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            s = x.strip()
            # кандидат — "похоже на имя файла"
            if 3 <= len(s) <= 128 and "." in s and not any(ch in s for ch in "\\/"):
                # простая фильтрация по расширению
                if s.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".ttf", ".otf", ".ass", ".srt", ".sup", ".vtt", ".xml")):
                    candidates.append(s)

    walk(att)
    # если нашли — берём первый (обычно он один)
    return candidates[0] if candidates else ""


def extract_attachments(
    mkvextract: str,
    source: Path,
    att_dir: Path,
    mkvj: Dict[str, Any],
    overwrite: bool,
) -> List[Dict[str, Any]]:
    extracted: List[Dict[str, Any]] = []
    atts = (mkvj.get("attachments", []) or [])
    if not atts:
        print("[demux] No attachments found.")
        return extracted

    ensure_dir(att_dir)

    for a in atts:
        att_id = a.get("id")
        props = a.get("properties") or {}
        ctype = props.get("content_type") or props.get("content-type") or ""

        # 1) Базовое имя: file_name если есть, иначе attachment_<id>
        base_name = (
            props.get("file_name")
            or props.get("file-name")
            or a.get("file_name")
            or a.get("file-name")
            or f"attachment_{att_id}"
        )
        base_name = safe_filename(str(base_name))

        # 2) Попытка найти "cover.jpg" из Description/Title/etc (best-effort)
        guessed = guess_name_from_attachment_dict(a)
        if guessed:
            guessed = safe_filename(guessed)

        # 3) Расширение по MIME (если нет расширения)
        mime_ext = ext_from_mime(str(ctype))
        def with_ext(name: str) -> str:
            if "." not in Path(name).name and mime_ext:
                return name + mime_ext
            return name

        final_name = with_ext(guessed) if guessed else with_ext(base_name)
        final_path = att_dir / final_name

        # коллизии
        if final_path.exists():
            final_path = att_dir / f"{final_path.stem}__att{att_id}{final_path.suffix}"

        if final_path.exists() and not overwrite:
            print(f"[demux] ATT skip exists: {final_path.name}")
            extracted.append({
                "id": att_id,
                "path": str(final_path),
                "skipped": True,
                "content_type": ctype,
            })
            continue

        # Пишем во временный файл (чтобы потом можно было rename)
        tmp_name = f"__att{att_id}__tmp"
        tmp_name = with_ext(tmp_name)
        tmp_path = att_dir / tmp_name
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

        cmd = [mkvextract, "attachments", str(source), f"{att_id}:{tmp_path}"]
        run_cmd(cmd)

        # rename -> финальное имя
        if tmp_path.exists():
            # если финальный уже есть и overwrite — удалим
            if final_path.exists() and overwrite:
                try:
                    final_path.unlink()
                except Exception:
                    pass
            try:
                tmp_path.replace(final_path)
            except Exception:
                # fallback: оставим tmp
                final_path = tmp_path

        extracted.append({
            "id": att_id,
            "path": str(final_path),
            "skipped": False,
            "content_type": ctype,
        })

    return extracted




def extract_chapters(mkvextract: str, source: Path, chapters_dir: Path, overwrite: bool) -> Dict[str, Any]:
    ensure_dir(chapters_dir)
    out_path = chapters_dir / "chapters.xml"

    if out_path.exists() and not overwrite:
        print(f"[demux] CHAPTERS skip exists: {out_path.name}")
        return {"path": str(out_path), "skipped": True}

    # ВАЖНО: правильный порядок аргументов:
    # mkvextract <source> chapters <output.xml>
    cmd = [mkvextract, str(source), "chapters", str(out_path)]

    print("[cmd]", " ".join(cmd))
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if p.stdout:
        print(p.stdout, end="" if p.stdout.endswith("\n") else "\n")

    if p.returncode != 0:
        # если глав нет — mkvextract может вернуть non-zero
        if out_path.exists() and out_path.stat().st_size > 0:
            return {"path": str(out_path), "skipped": False, "note": "nonzero_exit_but_file_exists"}
        if out_path.exists() and out_path.stat().st_size == 0:
            try:
                out_path.unlink()
            except Exception:
                pass
        print("[demux] No chapters (or mkvextract returned non-zero). Continuing.")
        return {"path": None, "skipped": False, "note": "no_chapters"}

    return {"path": str(out_path), "skipped": False}



def parse_tracks_json(tracks_json: Dict[str, Any]) -> List[TrackEntry]:
    tracks = tracks_json.get("tracks", []) or []
    parsed: List[TrackEntry] = []
    for t in tracks:
        try:
            tid = int(t.get("trackId"))
        except Exception:
            continue
        ty = normalize_type(str(t.get("type", "")))
        st = str(t.get("trackStatus", ""))
        fb = str(t.get("fileBase", f"track{tid}"))
        parsed.append(TrackEntry(trackId=tid, type=ty, trackStatus=st, fileBase=fb))
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser(description="Demux subtitles, attachments, chapters to WORKDIR")
    ap.add_argument("--source", required=True, help="Input source MKV")
    ap.add_argument("--workdir", required=True, help="Per-file workdir")
    ap.add_argument("--tracksData", required=True, help="tracks.json path (relative to workdir is allowed)")
    ap.add_argument("--mkvmerge", default="mkvmerge", help="Path to mkvmerge")
    ap.add_argument("--mkvextract", default="mkvextract", help="Path to mkvextract")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted files")
    ap.add_argument("--log", default="", help="Optional log file path (relative to --workdir if not absolute)")
    args = ap.parse_args()

    source = Path(args.source)
    workdir = Path(args.workdir)
    setup_logging(args.log, workdir)

    if not source.exists():
        eprint(f"[demux] Source not found: {source}")
        return 2

    ensure_dir(workdir)

    tracks_path = resolve_tracks_path(workdir, args.tracksData)
    if not tracks_path.exists():
        eprint(f"[demux] tracksData not found: {tracks_path}")
        return 3

    mkvmerge = which_or(args.mkvmerge, args.mkvmerge)
    mkvextract = which_or(args.mkvextract, args.mkvextract)

    if not shutil.which(mkvmerge) and not Path(mkvmerge).exists():
        eprint(f"[demux] mkvmerge not found: {mkvmerge}")
        return 4
    if not shutil.which(mkvextract) and not Path(mkvextract).exists():
        eprint(f"[demux] mkvextract not found: {mkvextract}")
        return 4

    try:
        tracks_json = load_json(tracks_path)
        entries = parse_tracks_json(tracks_json)

        # subs only (status != SKIP)
        subs = [t for t in entries if t.type == "sub" and not is_skip(t.trackStatus)]

        sub_dir = workdir / "sub"
        att_dir = workdir / "attachments"
        chapters_dir = workdir / "chapters"
        ensure_dir(sub_dir)
        ensure_dir(att_dir)
        ensure_dir(chapters_dir)

        mkvj = get_mkvmerge_json(mkvmerge, source)

        print(f"[demux] Source:  {source}")
        print(f"[demux] Workdir: {workdir}")
        print(f"[demux] Subs selected: {len(subs)}")
        print(f"[demux] Attachments listed: {len(mkvj.get('attachments', []) or [])}")

        subs_manifest = extract_subtitles(
            mkvextract=mkvextract,
            source=source,
            sub_dir=sub_dir,
            track_entries=subs,
            mkvj=mkvj,
            overwrite=bool(args.overwrite),
        )

        att_manifest = extract_attachments(
            mkvextract=mkvextract,
            source=source,
            att_dir=att_dir,
            mkvj=mkvj,
            overwrite=bool(args.overwrite),
        )

        chapters_info = extract_chapters(
            mkvextract=mkvextract,
            source=source,
            chapters_dir=chapters_dir,
            overwrite=bool(args.overwrite),
        )

        manifest = {
            "source": str(source),
            "workdir": str(workdir),
            "subs": subs_manifest,
            "attachments": att_manifest,
            "chapters": chapters_info,
        }
        write_json(workdir / "00_meta" / "demux_manifest.json", manifest)

        print("[demux] OK")
        return 0

    except Exception as ex:
        eprint(f"[demux] ERROR: {ex}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
