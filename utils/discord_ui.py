from __future__ import annotations

import asyncio
import json
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils.zoned_commands import ZONED_COMMAND_NAME

MAX_EMBED_FIELD = 1024
MAX_ACTIVE_PLAN_DISPLAY = 3

DONE_SQUARE = "\U0001F7E9"
RUNNING_SQUARE = "\U0001F7E6"
PENDING_SQUARE = "\u2B1B"
FAILED_SQUARE = "\U0001F7E5"
PAUSED_SQUARE = "\U0001F7E7"
IDLE_SQUARE = "\u2B1C"
STAGE_DISPLAY_NAMES = {
    "Attachments cleanup": "Fonts clean",
    "Auto-Boost: Scene Detection": "Auto-Boost: SCD",
    "Auto-Boost: PSD Scene Detection": "PSD Scene Detect",
}
AUTOBOOST_PARENT_STAGES = {"Auto-Boost: Scene Detection", "Auto-Boost: PSD Scene Detection"}
AUTOBOOST_CHILD_STAGES = {"Fastpass", "SSIMU2 Metrics"}


def sanitize_channel_component(value: str, *, fallback: str = "folder", limit: int = 70) -> str:
    base = str(value or "").strip().lower() or fallback
    base = re.sub(r"[^\w-]+", "-", base, flags=re.IGNORECASE | re.UNICODE).strip("-").lower()
    return base[:limit] or fallback


def channel_base_name_for_source(source_dir: str, alias: str = "") -> str:
    return sanitize_channel_component(alias or Path(source_dir).name, fallback="folder", limit=90)


def channel_status_key(snapshot: Dict[str, Any], *, updates_paused: bool = False) -> str:
    if str(snapshot.get("state") or "").strip().lower() == "offline":
        return "idle"
    if list(snapshot.get("failed") or []):
        return "error"
    if updates_paused or bool(snapshot.get("paused")) or bool(snapshot.get("pause_after_current")):
        return "paused"
    if list(snapshot.get("active") or []) or str(snapshot.get("state") or "") == "running":
        return "running"
    if str(snapshot.get("state") or "") == "finished":
        return "completed"
    return "idle"


def channel_name_for_source(source_dir: str, *, alias: str = "", status: str = "idle") -> str:
    prefixes = {
        "error": FAILED_SQUARE,
        "running": RUNNING_SQUARE,
        "completed": DONE_SQUARE,
        "paused": PAUSED_SQUARE,
        "idle": IDLE_SQUARE,
    }
    return f"{prefixes.get(status, IDLE_SQUARE)}-{channel_base_name_for_source(source_dir, alias)}"


def truncate(text: Any, limit: int = MAX_EMBED_FIELD) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def fmt_seconds(value: Any) -> str:
    try:
        total = int(float(value))
    except Exception:
        total = 0
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def fmt_duration(value: Any) -> str:
    try:
        total = float(value)
    except Exception:
        total = 0.0
    return fmt_seconds(total) if total > 0 else "-"


def fmt_size(value: Any) -> str:
    try:
        size = float(value)
    except Exception:
        size = 0.0
    units = ("B", "KB", "MB", "GB", "TB")
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.1f} {units[idx]}" if idx else f"{int(size)} {units[idx]}"


def fmt_size_pair(source_size: Any, output_size: Any) -> str:
    try:
        output = float(output_size)
    except Exception:
        output = 0.0
    return f"{fmt_size(source_size)} -> {fmt_size(output)}" if output > 0 else f"{fmt_size(source_size)} -> -"


def fastpass_output_size(item: Dict[str, Any]) -> int:
    try:
        size = int(float(item.get("fastpass_output_size") or 0))
        if size > 0:
            return size
    except Exception:
        pass
    raw_path = str(item.get("fastpass_output") or "").strip()
    candidates: List[Path] = []
    if raw_path:
        candidates.append(Path(raw_path))
    workdir = str(item.get("workdir") or "").strip()
    source = str(item.get("source") or "").strip()
    if workdir and source:
        candidates.append(Path(workdir) / "video" / "fastpass" / f"{Path(source).stem}.fastpass.mkv")
    for path in candidates:
        try:
            if path.exists() and path.is_file():
                return int(path.stat().st_size)
        except Exception:
            continue
    return 0


def fmt_plan_size_summary(item: Dict[str, Any]) -> str:
    base = fmt_size_pair(item.get("source_size"), item.get("output_size"))
    if str(item.get("mode") or "").strip().lower() != "fastpass":
        return base
    fp_size = fastpass_output_size(item)
    return f"{base}  {fmt_size(fp_size) if fp_size > 0 else '-'}"


def mode_label(value: Any) -> str:
    return "FP ONLY" if str(value or "").strip().lower() == "fastpass" else "FULL"


def display_filename(item: Dict[str, Any]) -> str:
    source = str(item.get("source") or "")
    if source:
        return Path(source).name
    plan = str(item.get("plan") or "")
    if plan:
        return Path(plan).name
    return str(item.get("name") or "plan")


def clip_field(value: Any, width: int) -> str:
    text = str(value or "")
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def display_stage_name(stage: Dict[str, Any]) -> str:
    name = str(stage.get("name") or "")
    return STAGE_DISPLAY_NAMES.get(name, name)


def is_cached_stage(stage: Dict[str, Any]) -> bool:
    message = str(stage.get("message") or "").strip().lower()
    if message in ("cached", "resume", "using_existing_base_scenes"):
        return True
    status = str(stage.get("status") or "").strip().lower()
    if status not in ("completed", "skipped"):
        return False
    try:
        elapsed = float(stage.get("elapsed_seconds") or 0.0)
        started_at = float(stage.get("started_at") or 0.0)
        ended_at = float(stage.get("ended_at") or 0.0)
    except Exception:
        return False
    return elapsed <= 0.0 and started_at <= 0.0 and ended_at <= 0.0


def progress_bar(progress: Any, width: int = 23) -> str:
    try:
        value = float(progress)
    except Exception:
        value = -1.0
    percent_text = "--.-%" if value < 0 else f"  {max(0.0, min(100.0, value)):.1f}%"
    inner_width = max(4, int(width) - 3 - len(percent_text))
    if value < 0:
        return "[" + ("-" * inner_width) + "]" + percent_text
    value = max(0.0, min(100.0, value))
    filled = int(round((value / 100.0) * inner_width))
    return "[" + ("#" * filled) + ("-" * (inner_width - filled)) + "]" + percent_text


def render_progress_line(stage: Dict[str, Any], width: int) -> str:
    details = dict(stage.get("details") or {})
    progress_parts = [progress_bar(stage.get("progress"), width=max(8, width))]
    try:
        progress_parts.append(f"{float(details.get('fps')):.2f} FPS")
    except Exception:
        try:
            progress_parts.append(f"{float(details.get('spf')):.2f} s/fr")
        except Exception:
            pass
    try:
        progress_parts.append(f"{float(details.get('kbps')):.0f} Kbps")
    except Exception:
        pass
    return " | ".join(progress_parts)


def render_stage_table(stages: List[Dict[str, Any]]) -> str:
    if not stages:
        return ""
    name_width = max(10, min(max(len(display_stage_name(stage)) for stage in stages), 36))
    autoboost_child_started = any(
        str(stage.get("name") or "") in AUTOBOOST_CHILD_STAGES
        and str(stage.get("status") or "").lower() == "started"
        for stage in stages
    )
    lines: List[str] = []
    for stage in stages:
        raw_name = str(stage.get("name") or "")
        name = clip_field(display_stage_name(stage), name_width)
        status = str(stage.get("status") or "pending").lower()
        if raw_name in AUTOBOOST_PARENT_STAGES and status == "started" and autoboost_child_started:
            status = "completed"
        cached = is_cached_stage(stage)
        icon = PENDING_SQUARE
        if status == "failed":
            icon = FAILED_SQUARE
        elif cached or status in ("completed", "skipped"):
            icon = DONE_SQUARE
        elif status == "started":
            icon = RUNNING_SQUARE
        detail = ""
        if cached:
            detail = "cached"
        elif status == "started":
            detail = fmt_seconds(stage.get("elapsed_seconds"))
        elif status == "completed":
            detail = fmt_seconds(stage.get("elapsed_seconds"))
        elif status == "skipped":
            detail = "skipped"
        elif status == "failed":
            detail = truncate(str(stage.get("message") or "failed"), 44)
        line = f"{icon} {name:<{name_width}}"
        details = dict(stage.get("details") or {})
        eta = str(details.get("eta") or "").strip()
        if status == "started" and eta:
            detail = f"{detail} | eta {eta}" if detail else f"eta {eta}"
        if detail:
            line += f" | {detail}"
        lines.append(line.rstrip())
        if status == "started" and stage.get("progress") is not None:
            lines.append(render_progress_line(stage, name_width + 4))
    return "\n".join(lines)


def snapshot_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    try:
        return json.loads(str(row["snapshot_json"] or "{}"))
    except Exception:
        return {}


def collect_plan_candidates(snapshot: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    for kind in ("completed", "failed", "active", "queue"):
        for item in list(snapshot.get(kind) or []):
            candidates.append((kind, dict(item)))
    return candidates


def select_plan(snapshot: Dict[str, Any], selector: str) -> Tuple[str, Dict[str, Any]]:
    raw = str(selector or "").strip()
    candidates = collect_plan_candidates(snapshot)
    if not candidates:
        raise ValueError("No .plan entries are known in this folder.")
    if raw.lower() in ("active", "current"):
        active = [(kind, item) for kind, item in candidates if kind == "active"]
        if not active:
            raise ValueError("No active .plan in this folder.")
        return active[0]
    if raw.lower() in ("last", "latest"):
        done = [(kind, item) for kind, item in candidates if kind in ("completed", "failed")]
        if done:
            return done[-1]
        return candidates[0]
    if raw.isdigit():
        index = int(raw) - 1
        if index < 0 or index >= len(candidates):
            raise ValueError(f".plan selector is out of range: {raw}")
        return candidates[index]
    needle = raw.lower()
    matches = [
        (kind, item)
        for kind, item in candidates
        if needle
        and (
            needle in display_filename(item).lower()
            or needle in Path(str(item.get("plan") or "")).name.lower()
            or needle in str(item.get("name") or "").lower()
        )
    ]
    if not matches:
        raise ValueError(f".plan selector did not match anything: {selector}")
    if len(matches) > 1:
        names = ", ".join(display_filename(item) for _, item in matches[:5])
        raise ValueError(f".plan selector is ambiguous: {names}")
    return matches[0]


def safe_relative_path(base: Path, relative: str) -> Path:
    raw = Path(str(relative or "").strip())
    if raw.is_absolute():
        raise ValueError("Absolute paths are not allowed.")
    target = (base / raw).resolve()
    base_resolved = base.resolve()
    if target != base_resolved and base_resolved not in target.parents:
        raise ValueError("Path escapes the folder.")
    return target


def plan_named_file(plan: Dict[str, Any], name: str) -> Path:
    key = str(name or "").strip().replace("\\", "/")
    normalized = key.lower()
    workdir = Path(str(plan.get("workdir") or "")).resolve()
    plan_path = Path(str(plan.get("plan") or "")).resolve()
    mapping = {
        "plan": plan_path,
        ".plan": plan_path,
        "state": workdir / "00_meta" / "runner_state.json",
        "runner_state": workdir / "00_meta" / "runner_state.json",
        "events": workdir / "00_meta" / "runner_events.jsonl",
        "zone": workdir / ZONED_COMMAND_NAME,
        "zone_edit": workdir / ZONED_COMMAND_NAME,
        "zoned": workdir / ZONED_COMMAND_NAME,
        "crop": workdir / ZONED_COMMAND_NAME,
        "crop_resize": workdir / ZONED_COMMAND_NAME,
    }
    if normalized in mapping:
        return mapping[normalized]
    return safe_relative_path(workdir, key)


def ensure_uploadable(path: Path, *, max_upload_mb: int) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(str(path))
    max_bytes = max_upload_mb * 1024 * 1024
    if path.stat().st_size > max_bytes:
        raise RuntimeError(f"File is too large: {fmt_size(path.stat().st_size)}")


def tree_listing(path: Path, *, limit: int = 80) -> str:
    if not path.exists():
        return "missing"
    if path.is_file():
        return f"{path.name} | {fmt_size(path.stat().st_size)}"
    rows: List[str] = []
    for entry in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:limit]:
        suffix = "/" if entry.is_dir() else ""
        size = "-" if entry.is_dir() else fmt_size(entry.stat().st_size)
        rows.append(f"{entry.name}{suffix:<1} | {size}")
    if not rows:
        return "empty"
    return "\n".join(rows)


def probe_media_duration(path: Path) -> str:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return "-"
    try:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=8,
            check=False,
        )
        if proc.returncode == 0:
            return fmt_duration(float(str(proc.stdout or "0").strip() or "0"))
    except Exception:
        pass
    return "-"


async def probe_media_duration_async(path: Path) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, probe_media_duration, path)


def list_ivf_files(plan: Dict[str, Any], pass_name: str, *, count: int, sort_mode: str) -> List[Path]:
    workdir = Path(str(plan.get("workdir") or "")).resolve()
    folder = workdir / "video" / f"{pass_name}pass" / "encode"
    files = list(folder.glob("*.ivf")) if folder.exists() else []
    if sort_mode == "size":
        files.sort(key=lambda p: (p.stat().st_size, p.name.lower()))
    elif sort_mode == "mtime":
        files.sort(key=lambda p: (p.stat().st_mtime, p.name.lower()))
    else:
        def key(path: Path) -> Tuple[int, str]:
            try:
                return int(path.stem), path.name.lower()
            except Exception:
                return 10**9, path.name.lower()
        files.sort(key=key)
    return files[: max(1, min(int(count), 25))]


def render_plan_detail(plan: Dict[str, Any]) -> str:
    header = (
        f"{mode_label(plan.get('mode'))} | "
        f"{fmt_duration(plan.get('duration_seconds'))} | "
        f"{fmt_size_pair(plan.get('source_size'), plan.get('output_size'))}"
    )
    elapsed = f"Elapsed: {fmt_seconds(plan.get('elapsed_seconds'))}"
    table = render_stage_table(list(plan.get("stages") or [])) or "-"
    return f"{header}\n\n{elapsed}\n```text\n{table}\n```"


def plan_overview_elapsed_label(item: Dict[str, Any]) -> str:
    status = str(item.get("status") or "").lower()
    if status == "skipped":
        return "skipped"
    try:
        elapsed = float(item.get("elapsed_seconds") or 0.0)
    except Exception:
        elapsed = 0.0
    return "cached" if status == "completed" and elapsed < 1.0 else fmt_seconds(elapsed)


def render_overview_embed(discord_module: Any, snapshot: Dict[str, Any]) -> Any:
    embed = discord_module.Embed(title="Plans", color=0x2B2D31)
    lines: List[str] = []

    def row(icon: str, item: Dict[str, Any], *fields: str) -> None:
        suffix = " | ".join(fields)
        lines.append(display_filename(item))
        lines.append(f"{icon} | {suffix}".rstrip())

    for item in list(snapshot.get("completed") or [])[-8:]:
        skipped = str(item.get("status") or "").lower() == "skipped"
        row(
            IDLE_SQUARE if skipped else DONE_SQUARE,
            item,
            f"{fmt_duration(item.get('duration_seconds')):<7}",
            plan_size_overview_field(item),
            plan_overview_elapsed_label(item),
        )
    for item in list(snapshot.get("failed") or [])[-6:]:
        row(
            FAILED_SQUARE,
            item,
            f"{fmt_duration(item.get('duration_seconds')):<7}",
            clip_field(str(item.get("stage") or "-"), 22),
            truncate(str(item.get("message") or "failed"), 80),
        )
    for item in list(snapshot.get("active") or [])[:MAX_ACTIVE_PLAN_DISPLAY]:
        row(
            RUNNING_SQUARE,
            item,
            fmt_duration(item.get("duration_seconds")),
        )
    queue_items = list(snapshot.get("queue") or [])
    for item in queue_items[:20]:
        row(PENDING_SQUARE, item, fmt_duration(item.get("duration_seconds")))
    if len(queue_items) > 20:
        lines.append(f"... and {len(queue_items) - 20} more queued")

    body = "\n".join(lines) if lines else "No plans yet."
    embed.description = "```text\n" + truncate(body, 3900) + "\n```"
    return embed


def plan_size_overview_field(item: Dict[str, Any]) -> str:
    summary = fmt_plan_size_summary(item)
    width = 29 if str(item.get("mode") or "").strip().lower() == "fastpass" else 20
    return f"{summary:<{width}}"


def render_current_embed(discord_module: Any, snapshot: Dict[str, Any]) -> Any:
    active = list(snapshot.get("active") or [])
    if not active:
        state = str(snapshot.get("state") or "idle")
        return discord_module.Embed(title="Current process", description=f"State: `{state}`", color=0x5865F2)

    title = "Active plans" if len(active) > 1 else display_filename(active[0])
    embed = discord_module.Embed(title=title, color=0x5865F2)
    visible = active[:MAX_ACTIVE_PLAN_DISPLAY]
    for index, plan in enumerate(visible):
        value = truncate(render_plan_detail(plan), 4000 if len(active) == 1 else 1000)
        if len(active) == 1:
            embed.description = value
        else:
            embed.add_field(name=display_filename(plan) or f"plan {index + 1}", value=value, inline=False)
    if len(active) > len(visible):
        embed.set_footer(text=f"{len(active) - len(visible)} more active plan(s) not shown")
    return embed


def dashboard_major_signature(snapshot: Dict[str, Any]) -> str:
    def plan_signature(plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(plan.get("plan_run_id") or ""),
            "name": display_filename(plan),
            "status": str(plan.get("status") or ""),
            "mode": str(plan.get("mode") or ""),
            "stages": [
                {
                    "name": str(stage.get("name") or ""),
                    "status": str(stage.get("status") or ""),
                    "message": str(stage.get("message") or ""),
                }
                for stage in list(plan.get("stages") or [])
            ],
        }

    data = {
        "state": str(snapshot.get("state") or ""),
        "paused": bool(snapshot.get("paused")),
        "pause_after_current": bool(snapshot.get("pause_after_current")),
        "exit_when_idle": bool(snapshot.get("exit_when_idle")),
        "active": [plan_signature(plan) for plan in list(snapshot.get("active") or [])],
        "queue": [
            {
                "name": display_filename(item),
                "mode": str(item.get("mode") or ""),
                "plan": str(item.get("plan") or ""),
            }
            for item in list(snapshot.get("queue") or [])
        ],
        "completed": [
            {
                "id": str(item.get("plan_run_id") or ""),
                "status": str(item.get("status") or ""),
                "name": display_filename(item),
            }
            for item in list(snapshot.get("completed") or [])
        ],
        "failed": [
            {
                "id": str(item.get("plan_run_id") or ""),
                "status": str(item.get("status") or ""),
                "stage": str(item.get("stage") or ""),
                "message": str(item.get("message") or ""),
            }
            for item in list(snapshot.get("failed") or [])
        ],
    }
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
