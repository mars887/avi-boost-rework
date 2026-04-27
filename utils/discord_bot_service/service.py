from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.discord_store import (
    DEFAULT_DASHBOARD_MAJOR_INTERVAL_SECONDS,
    DEFAULT_DASHBOARD_PROGRESS_INTERVAL_SECONDS,
    DEFAULT_RUNNER_STARTUP_DELAY_SECONDS,
    StateStore,
)
from utils.discord_ui import (
    channel_name_for_source,
    channel_status_key,
    dashboard_major_signature,
    display_filename,
    display_stage_name,
    ensure_uploadable,
    fmt_duration,
    fmt_seconds,
    fmt_size,
    list_ivf_files,
    plan_named_file,
    probe_media_duration_async,
    render_current_embed,
    render_overview_embed,
    safe_relative_path,
    sanitize_channel_component,
    select_plan,
    snapshot_from_row,
    tree_listing,
    truncate,
)
from utils.zoned_commands import ZONED_COMMAND_NAME

from .batch_tools import run_batch_edit_text, run_batch_tool
from .settings import (
    CHANNEL_RENAME_FAILURE_BACKOFF_SECONDS,
    CHANNEL_RENAME_MIN_INTERVAL_SECONDS,
    CHANNEL_RENAME_TIMEOUT_SECONDS,
    COMMANDS,
    DASHBOARD_LOCK_TIMEOUT_SECONDS,
    DASHBOARD_UPDATE_TIMEOUT_SECONDS,
    DISCORD_API_TIMEOUT_SECONDS,
    INACTIVE_MESSAGE_TEXT,
    LOADER_FILE_TTL_SECONDS,
    SESSION_STALE_SECONDS,
    BotConfig,
)

try:
    import discord  # type: ignore[import-not-found]
    from aiohttp import web  # type: ignore[import-not-found]
    from discord import app_commands  # type: ignore[import-not-found]
except ImportError:
    discord = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]
    app_commands = None  # type: ignore[assignment]


async def run_bot(config: BotConfig) -> None:
    if discord is None or web is None or app_commands is None:
        raise RuntimeError("discord.py and aiohttp are required. Install requirements-discord.txt first.")
    if not config.token or not config.guild_id or not config.category_id:
        raise RuntimeError("PBBATCH_DISCORD_TOKEN, PBBATCH_DISCORD_GUILD_ID and PBBATCH_DISCORD_CATEGORY_ID are required.")
    if config.host not in ("127.0.0.1", "localhost", "::1") and not config.shared_secret:
        raise RuntimeError("PBBATCH_DISCORD_SHARED_SECRET is required when the bot service is not bound to localhost.")

    store = StateStore(config.db_path)
    intents = discord.Intents.default()
    client = discord.Client(intents=intents)
    tree = app_commands.CommandTree(client)
    guild_object = discord.Object(id=config.guild_id)
    last_render_at: Dict[str, float] = {}
    last_major_signature: Dict[str, str] = {}
    pending_snapshots: Dict[str, Dict[str, Any]] = {}
    pending_tasks: Dict[str, Any] = {}
    pending_due_at: Dict[str, float] = {}
    pending_generation: Dict[str, int] = {}
    startup_render_block_until: Dict[str, float] = {}
    latest_session_by_source: Dict[str, str] = {}
    last_message_signature: Dict[Tuple[str, str], str] = {}
    channel_locks: Dict[str, Any] = {}
    dashboard_locks: Dict[str, Any] = {}
    loader_files: Dict[str, Tuple[Path, float]] = {}
    last_channel_rename_at: Dict[str, float] = {}
    last_channel_rename_failed_at: Dict[str, float] = {}
    channel_rename_tasks: Dict[str, Any] = {}
    pending_channel_rename_requests: Dict[str, Tuple[Any, Dict[str, Any], bool, str]] = {}
    sent_failed_event_keys: Dict[str, float] = {}
    startup_reconciled = False
    stale_monitor_started = False
    stale_sessions_marked: set[str] = set()

    def log(message: str, *, debug: bool = False) -> None:
        if debug and not config.debug:
            return
        prefix = "[discord-bot:debug]" if debug else "[discord-bot]"
        print(f"{prefix} {message}", flush=True)

    def fmt_age(timestamp: Any) -> str:
        try:
            value = float(timestamp or 0.0)
        except Exception:
            value = 0.0
        if value <= 0:
            return "-"
        return f"{max(0.0, time.time() - value):.1f}s"

    def fmt_elapsed(seconds: Any) -> str:
        try:
            value = float(seconds)
        except Exception:
            value = 0.0
        return f"{max(0.0, value):.1f}s"

    def fmt_monotonic_age(timestamp: Any) -> str:
        try:
            value = float(timestamp or 0.0)
        except Exception:
            value = 0.0
        if value <= 0:
            return "-"
        return fmt_elapsed(time.monotonic() - value)

    def snapshot_counts(snapshot: Dict[str, Any]) -> Dict[str, int]:
        counts = dict(snapshot.get("counts") or {})

        def value(key: str, fallback: int) -> int:
            try:
                return int(counts.get(key) if counts.get(key) is not None else fallback)
            except Exception:
                return fallback

        return {
            "active": value("active", len(list(snapshot.get("active") or []))),
            "queued": value("queued", len(list(snapshot.get("queue") or []))),
            "completed": value("completed", len(list(snapshot.get("completed") or []))),
            "failed": value("failed", len(list(snapshot.get("failed") or []))),
        }

    def snapshot_summary(snapshot: Dict[str, Any]) -> str:
        counts = snapshot_counts(snapshot)
        session_id = str(snapshot.get("session_id") or "-")
        return (
            f"session={session_id} state={snapshot.get('state', '-')} "
            f"active={counts['active']} queued={counts['queued']} "
            f"completed={counts['completed']} failed={counts['failed']} "
            f"snapshot_age={fmt_age(snapshot.get('snapshot_at'))}"
        )

    def stored_snapshot_is_terminal(snapshot: Dict[str, Any]) -> bool:
        state = str(snapshot.get("state") or "").strip().lower()
        if state in ("finished", "offline"):
            return True
        counts = snapshot_counts(snapshot)
        return (
            counts["active"] <= 0
            and counts["queued"] <= 0
            and bool(list(snapshot.get("completed") or []) or list(snapshot.get("failed") or []))
        )

    def live_session_diagnostics(row: Optional[sqlite3.Row]) -> Tuple[bool, str]:
        if row is None:
            return False, "missing_session_row"
        source_dir = str(row["source_dir"] or "")
        session_id = str(row["session_id"] or "")
        if not source_dir:
            return False, "missing_source_dir"
        if not session_id:
            return False, "missing_session_id"
        snapshot = snapshot_from_row(row)
        heartbeat_at = 0.0
        try:
            heartbeat_at = float(snapshot.get("snapshot_at") or row["updated_at"] or 0.0)
        except Exception:
            return False, "invalid_snapshot_at"
        age = time.time() - heartbeat_at if heartbeat_at > 0 else 10**9
        if age > SESSION_STALE_SECONDS:
            return False, f"stale heartbeat_age={age:.1f}s limit={SESSION_STALE_SECONDS:.1f}s"
        state = str(snapshot.get("state") or "").strip().lower()
        if state in ("finished", "offline"):
            return False, f"terminal_state={state or '-'}"
        owner = latest_session_by_source.get(source_dir)
        if owner is None:
            latest_session_by_source[source_dir] = session_id
            log(f"live owner restored from db: source={source_dir} session={session_id}", debug=True)
        elif owner != session_id:
            return False, f"owner_mismatch owner={owner or '-'} row={session_id}"
        return True, f"live heartbeat_age={max(0.0, age):.1f}s state={state or '-'}"

    log(
        f"starting service bind=http://{config.host}:{config.port} guild={config.guild_id} "
        f"category={config.category_id} db={config.db_path} debug={config.debug}"
    )
    log(
        f"auth admins_only={config.admins_only} admin_role={config.admin_role_id or '-'} "
        f"operator_role={config.operator_role_id or '-'} shared_secret={'set' if config.shared_secret else 'not_set'}",
        debug=True,
    )

    async def discord_api_call(label: str, awaitable: Any, *, timeout: float = DISCORD_API_TIMEOUT_SECONDS) -> Any:
        started = time.monotonic()
        log(f"discord api start: {label} timeout={fmt_elapsed(timeout)}", debug=True)
        try:
            result = await asyncio.wait_for(awaitable, timeout=timeout)
        except asyncio.TimeoutError:
            log(f"discord api timeout: {label} after={fmt_elapsed(timeout)}")
            raise
        except Exception as exc:
            log(f"discord api failed: {label} after={fmt_monotonic_age(started)} error={truncate(exc, 300)}", debug=True)
            raise
        log(f"discord api ok: {label} elapsed={fmt_monotonic_age(started)}", debug=True)
        return result

    def is_authorized(interaction: Any) -> bool:
        user = getattr(interaction, "user", None)
        permissions = getattr(user, "guild_permissions", None)
        roles = getattr(user, "roles", []) or []
        role_ids = {int(getattr(role, "id", 0)) for role in roles}
        is_admin = bool(getattr(permissions, "administrator", False)) or (
            bool(config.admin_role_id) and config.admin_role_id in role_ids
        )
        if config.admins_only:
            return is_admin
        if config.admin_role_id == 0 and config.operator_role_id == 0:
            return True
        return bool(role_ids.intersection({config.admin_role_id, config.operator_role_id}))

    def active_stage_label(plan: Dict[str, Any]) -> str:
        stages = list(plan.get("stages") or [])
        running = [stage for stage in stages if str(stage.get("status") or "").lower() == "started"]
        if running:
            return ", ".join(display_stage_name(dict(stage)) for stage in running)
        completed = [stage for stage in stages if str(stage.get("status") or "").lower() in ("completed", "skipped")]
        if completed:
            return display_stage_name(dict(completed[-1]))
        return "-"

    def is_live_session_row(row: Optional[sqlite3.Row]) -> bool:
        return live_session_diagnostics(row)[0]

    def offline_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(snapshot)
        out["state"] = "offline"
        out["paused"] = False
        out["pause_after_current"] = False
        out["active"] = []
        out["queue"] = []
        counts = dict(out.get("counts") or {})
        counts["active"] = 0
        counts["queued"] = 0
        out["counts"] = counts
        return out

    def latest_failed_stage_from_events(workdir: Path, plan_run_id: str) -> Tuple[str, str]:
        path = workdir / "00_meta" / "runner_events.jsonl"
        if not path.exists() or not path.is_file():
            return "", ""
        try:
            with path.open("rb") as fh:
                size = fh.seek(0, os.SEEK_END)
                fh.seek(max(0, size - 256 * 1024))
                text = fh.read().decode("utf-8", errors="replace")
        except Exception:
            return "", ""
        for line in reversed(text.splitlines()):
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if str(payload.get("plan_run_id") or "") != plan_run_id:
                continue
            if str(payload.get("status") or "").strip().lower() != "failed":
                continue
            stage = str(payload.get("stage") or "")
            if not stage or stage == "Item":
                continue
            return stage, str(payload.get("message") or "")
        return "", ""

    def recover_terminal_snapshot_from_runner_state(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        active = [dict(item) for item in list(snapshot.get("active") or [])]
        if not active:
            return None
        completed = [dict(item) for item in list(snapshot.get("completed") or [])]
        failed = [dict(item) for item in list(snapshot.get("failed") or [])]
        queue_items = list(snapshot.get("queue") or [])
        remaining_active: List[Dict[str, Any]] = []
        recovered = False
        seen_finished = {
            str(item.get("plan_run_id") or "")
            for item in completed + failed
            if str(item.get("plan_run_id") or "")
        }

        for plan in active:
            workdir = Path(str(plan.get("workdir") or ""))
            plan_run_id = str(plan.get("plan_run_id") or "")
            state_path = workdir / "00_meta" / "runner_state.json"
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                remaining_active.append(plan)
                continue
            state_plan_run_id = str(state.get("plan_run_id") or "")
            if plan_run_id and state_plan_run_id and plan_run_id != state_plan_run_id:
                remaining_active.append(plan)
                continue
            status = str(state.get("status") or "").strip().lower()
            if str(state.get("stage") or "") != "Item" or status not in ("completed", "failed", "skipped"):
                remaining_active.append(plan)
                continue
            finished = dict(plan)
            finished["status"] = status
            finished["message"] = str(state.get("message") or "")
            try:
                ended_at = float(state.get("timestamp") or time.time())
            except Exception:
                ended_at = time.time()
            finished["ended_at"] = ended_at
            try:
                started_at = float(finished.get("started_at") or ended_at)
            except Exception:
                started_at = ended_at
            finished["elapsed_seconds"] = round(max(0.0, ended_at - started_at), 3)
            finished_plan_run_id = str(finished.get("plan_run_id") or state_plan_run_id)
            if finished_plan_run_id in seen_finished:
                recovered = True
                continue
            if status == "failed":
                failed_stage, failed_message = latest_failed_stage_from_events(workdir, finished_plan_run_id)
                finished["stage"] = failed_stage or str(finished.get("stage") or "Item")
                if failed_message and not finished["message"]:
                    finished["message"] = failed_message
                failed.append(finished)
            else:
                completed.append(finished)
            if finished_plan_run_id:
                seen_finished.add(finished_plan_run_id)
            recovered = True

        if not recovered:
            return None
        out = dict(snapshot)
        out["active"] = remaining_active
        out["completed"] = completed
        out["failed"] = failed
        out["paused"] = False if not remaining_active else bool(out.get("paused"))
        out["pause_after_current"] = False if not remaining_active else bool(out.get("pause_after_current"))
        out["state"] = "finished" if not remaining_active and not queue_items else ("running" if remaining_active else "idle")
        out["snapshot_at"] = time.time()
        counts = dict(out.get("counts") or {})
        counts["active"] = len(remaining_active)
        counts["queued"] = len(queue_items)
        counts["completed"] = len(completed)
        counts["failed"] = len(failed)
        out["counts"] = counts
        return out

    def snapshot_for_row(row: sqlite3.Row) -> Dict[str, Any]:
        snapshot = snapshot_from_row(row)
        if is_live_session_row(row) or stored_snapshot_is_terminal(snapshot):
            return snapshot
        recovered = recover_terminal_snapshot_from_runner_state(snapshot)
        if recovered is not None:
            return recovered
        return offline_snapshot(snapshot)

    async def print_console_status() -> None:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        rows = store.folder_rows()
        print(f"[discord-bot] status at {now}", flush=True)
        print(f"[discord-bot] folders: {len(rows)} | debug={config.debug}", flush=True)
        if not rows:
            return
        for folder in rows:
            source_dir = str(folder["source_dir"] or "")
            channel_id = int(folder["channel_id"] or 0)
            stored_name = str(folder["channel_name"] or "")
            settings = store.folder_settings(source_dir)
            row = store.latest_session_for_source_dir(source_dir)
            snapshot = snapshot_from_row(row) if row else {}
            live, live_reason = live_session_diagnostics(row)
            display_snapshot = snapshot if live else offline_snapshot(snapshot)
            status = channel_status_key(display_snapshot, updates_paused=bool(settings.get("updates_paused")))
            channel = client.get_channel(channel_id) if channel_id else None
            channel_name = str(getattr(channel, "name", "") or stored_name or "-")
            session_id = str(row["session_id"] or "-") if row else "-"
            counts = dict(display_snapshot.get("counts") or {})
            row_updated_at = row["updated_at"] if row else 0.0
            inactive_message_id = int(folder["inactive_message_id"] or 0)
            owner_session_id = latest_session_by_source.get(source_dir, "-")
            pending_due = pending_due_at.get(session_id, 0.0) if session_id != "-" else 0.0
            pending_delay = max(0.0, pending_due - time.monotonic()) if pending_due else 0.0
            last_render_age = fmt_monotonic_age(last_render_at.get(session_id, 0.0)) if session_id != "-" else "-"
            print("", flush=True)
            print(f"[folder] {source_dir}", flush=True)
            print(f"  channel: {channel_name} ({channel_id or '-'})", flush=True)
            print(f"  status: {status}", flush=True)
            print(f"  session: {session_id}{'' if live else ' (offline/stale)'}", flush=True)
            print(f"  live_reason: {live_reason}", flush=True)
            print(f"  owner_session: {owner_session_id}", flush=True)
            print(f"  bot_updates_paused: {bool(settings.get('updates_paused'))}", flush=True)
            print(
                "  timestamps: "
                f"row_age={fmt_age(row_updated_at)}, "
                f"snapshot_age={fmt_age(snapshot.get('snapshot_at')) if snapshot else '-'}, "
                f"last_render_age={last_render_age}, "
                f"pending_render_in={pending_delay:.1f}s",
                flush=True,
            )
            print(
                "  messages: "
                f"history={int(row['history_message_id'] or 0) if row else 0}, "
                f"current={int(row['current_message_id'] or 0) if row else 0}, "
                f"queue={int(row['queue_message_id'] or 0) if row else 0}, "
                f"inactive={inactive_message_id}",
                flush=True,
            )
            print(
                "  intervals: "
                f"major={int(settings.get('major_interval_seconds') or 0)}s, "
                f"progress={int(settings.get('progress_interval_seconds') or 0)}s, "
                f"startup_delay={int(settings.get('startup_delay_seconds') or 0)}s",
                flush=True,
            )
            print(
                "  runner: "
                f"state={display_snapshot.get('state', '-')}, "
                f"paused={bool(display_snapshot.get('paused'))}, "
                f"pause_after_current={bool(display_snapshot.get('pause_after_current'))}, "
                f"active={counts.get('active', 0)}, "
                f"queued={counts.get('queued', 0)}, "
                f"completed={counts.get('completed', 0)}, "
                f"failed={counts.get('failed', 0)}",
                flush=True,
            )
            for index, plan in enumerate(list(display_snapshot.get("active") or []), start=1):
                print(
                    f"  active {index}: {display_filename(dict(plan))} | "
                    f"{active_stage_label(dict(plan))} | elapsed {fmt_seconds(dict(plan).get('elapsed_seconds'))}",
                    flush=True,
                )
            queue = list(display_snapshot.get("queue") or [])
            for index, item in enumerate(queue[:5], start=1):
                print(f"  queue {index}: {display_filename(dict(item))} | {fmt_duration(dict(item).get('duration_seconds'))}", flush=True)
            if len(queue) > 5:
                print(f"  queue: ... and {len(queue) - 5} more", flush=True)
            if config.debug and snapshot:
                print(f"  raw_snapshot: {snapshot_summary(snapshot)}", flush=True)
                print(
                    "  render_state: "
                    f"pending_task={bool(pending_tasks.get(session_id))}, "
                    f"stale_marked={session_id in stale_sessions_marked}, "
                    f"startup_block_remaining={max(0.0, startup_render_block_until.get(session_id, 0.0) - time.monotonic()):.1f}s",
                    flush=True,
                )

    def console_command_loop(loop: asyncio.AbstractEventLoop) -> None:
        for raw in sys.stdin:
            command = raw.strip().lower()
            if not command:
                continue
            if command == "status":
                future = asyncio.run_coroutine_threadsafe(print_console_status(), loop)
                try:
                    future.result(timeout=10)
                except Exception as exc:
                    print(f"[discord-bot] status command failed: {exc}", flush=True)
                continue
            if command in ("help", "?"):
                print("[discord-bot] commands: status", flush=True)
                continue
            print(f"[discord-bot] unknown command: {command}", flush=True)

    loop = asyncio.get_running_loop()
    threading.Thread(target=console_command_loop, args=(loop,), name="discord-bot-console", daemon=True).start()

    async def ensure_channel(source_dir: str) -> Any:
        lock = channel_locks.setdefault(source_dir, asyncio.Lock())
        log(f"channel lock waiting: source={source_dir}", debug=True)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=DASHBOARD_LOCK_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            log(f"channel lock timeout: source={source_dir} after={fmt_elapsed(DASHBOARD_LOCK_TIMEOUT_SECONDS)}")
            raise
        log(f"channel lock acquired: source={source_dir}", debug=True)
        try:
            await discord_api_call("client.wait_until_ready ensure_channel", client.wait_until_ready(), timeout=DISCORD_API_TIMEOUT_SECONDS)
            existing_id = store.get_channel_id(source_dir)
            settings = store.folder_settings(source_dir)
            channel_name = channel_name_for_source(source_dir, alias=str(settings.get("alias") or ""), status="idle")
            guild = client.get_guild(config.guild_id)
            if guild is None:
                raise RuntimeError(f"guild not found: {config.guild_id}")
            log(f"ensure_channel start: source={source_dir} stored_id={existing_id or '-'}", debug=True)
            if existing_id:
                channel = client.get_channel(existing_id)
                if channel is None:
                    try:
                        channel = await discord_api_call(
                            f"guild.fetch_channel stored source={source_dir} id={existing_id}",
                            guild.fetch_channel(existing_id),
                        )
                    except Exception as exc:
                        log(f"stored channel fetch failed for {source_dir}: id={existing_id} error={truncate(exc, 180)}", debug=True)
                        channel = None
                if channel is not None:
                    log(f"using stored channel for {source_dir}: id={existing_id}", debug=True)
                    return channel
            category = guild.get_channel(config.category_id)
            if category is None:
                try:
                    category = await discord_api_call(
                        f"guild.fetch_channel category id={config.category_id}",
                        guild.fetch_channel(config.category_id),
                    )
                except Exception as exc:
                    log(f"category fetch failed: id={config.category_id} error={truncate(exc, 180)}", debug=True)
                    category = None
            if category is None:
                raise RuntimeError(f"category not found: {config.category_id}")
            candidates = list(getattr(category, "channels", []) or [])
            try:
                fetched_channels = await discord_api_call(
                    f"guild.fetch_channels scan source={source_dir}",
                    guild.fetch_channels(),
                )
                for channel in fetched_channels:
                    parent_id = int(getattr(channel, "category_id", 0) or getattr(channel, "parent_id", 0) or 0)
                    if parent_id == int(config.category_id):
                        candidates.append(channel)
            except Exception as exc:
                log(f"guild channel scan failed for {source_dir}: {truncate(exc, 180)}", debug=True)
                pass
            seen_ids: set[int] = set()
            for channel in candidates:
                channel_id = int(getattr(channel, "id", 0) or 0)
                if not channel_id or channel_id in seen_ids:
                    continue
                seen_ids.add(channel_id)
                topic = str(getattr(channel, "topic", "") or "")
                existing_name = str(getattr(channel, "name", "") or "")
                if topic == f"PBBatch source: {source_dir}":
                    store.set_channel(source_dir, channel_id, existing_name)
                    log(f"matched existing channel by topic for {source_dir}: id={channel_id} name={existing_name}", debug=True)
                    return channel
            channel = await discord_api_call(
                f"guild.create_text_channel source={source_dir} name={channel_name}",
                guild.create_text_channel(
                    name=channel_name,
                    category=category,
                    topic=f"PBBatch source: {source_dir}",
                    reason="PBBatch runner session",
                ),
            )
            store.set_channel(source_dir, int(channel.id), channel_name)
            log(f"created channel for {source_dir}: id={int(channel.id)} name={channel_name}")
            return channel
        finally:
            lock.release()
            log(f"channel lock released: source={source_dir}", debug=True)

    async def find_existing_channel(source_dir: str, channel_id: int) -> Any:
        await discord_api_call("client.wait_until_ready find_existing_channel", client.wait_until_ready(), timeout=DISCORD_API_TIMEOUT_SECONDS)
        guild = client.get_guild(config.guild_id)
        if guild is None:
            return None
        if channel_id:
            channel = client.get_channel(channel_id)
            if channel is None:
                try:
                    channel = await discord_api_call(
                        f"guild.fetch_channel reconcile source={source_dir} id={channel_id}",
                        guild.fetch_channel(channel_id),
                    )
                except Exception as exc:
                    log(f"reconcile fetch channel failed for {source_dir}: id={channel_id} error={truncate(exc, 180)}", debug=True)
                    channel = None
            if channel is not None:
                log(f"reconcile found stored channel for {source_dir}: id={channel_id}", debug=True)
                return channel
        category = guild.get_channel(config.category_id)
        if category is None:
            try:
                category = await discord_api_call(
                    f"guild.fetch_channel reconcile category id={config.category_id}",
                    guild.fetch_channel(config.category_id),
                )
            except Exception as exc:
                log(f"reconcile category fetch failed: id={config.category_id} error={truncate(exc, 180)}", debug=True)
                category = None
        candidates = list(getattr(category, "channels", []) or []) if category is not None else []
        try:
            fetched_channels = await discord_api_call(
                f"guild.fetch_channels reconcile source={source_dir}",
                guild.fetch_channels(),
            )
            for channel in fetched_channels:
                parent_id = int(getattr(channel, "category_id", 0) or getattr(channel, "parent_id", 0) or 0)
                if parent_id == int(config.category_id):
                    candidates.append(channel)
        except Exception as exc:
            log(f"reconcile channel scan failed for {source_dir}: {truncate(exc, 180)}", debug=True)
            pass
        seen_ids: set[int] = set()
        for channel in candidates:
            candidate_id = int(getattr(channel, "id", 0) or 0)
            if not candidate_id or candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            if str(getattr(channel, "topic", "") or "") == f"PBBatch source: {source_dir}":
                store.set_channel(source_dir, candidate_id, str(getattr(channel, "name", "") or ""))
                log(f"reconcile matched channel by topic for {source_dir}: id={candidate_id}", debug=True)
                return channel
        return None

    def channel_rename_wait_seconds(source_dir: str) -> float:
        now = time.monotonic()
        waits: List[float] = []
        last_at = last_channel_rename_at.get(source_dir, 0.0)
        if last_at:
            waits.append(CHANNEL_RENAME_MIN_INTERVAL_SECONDS - (now - last_at))
        failed_at = last_channel_rename_failed_at.get(source_dir, 0.0)
        if failed_at:
            waits.append(CHANNEL_RENAME_FAILURE_BACKOFF_SECONDS - (now - failed_at))
        return max([0.0, *waits])

    async def update_channel_name(channel: Any, source_dir: str, snapshot: Dict[str, Any], *, force: bool = False) -> bool:
        settings = store.folder_settings(source_dir)
        status = channel_status_key(snapshot, updates_paused=bool(settings.get("updates_paused")))
        desired = channel_name_for_source(source_dir, alias=str(settings.get("alias") or ""), status=status)
        current = str(getattr(channel, "name", "") or "")
        if current == desired:
            last_channel_rename_failed_at.pop(source_dir, None)
            log(f"channel already named {desired} for {source_dir}", debug=True)
            return True
        now = time.monotonic()
        wait_seconds = channel_rename_wait_seconds(source_dir)
        if wait_seconds > 0:
            log(
                f"channel rename deferred for {source_dir}: current={current} desired={desired} "
                f"retry_in={wait_seconds:.1f}s",
                debug=True,
            )
            return False
        last_channel_rename_failed_at.pop(source_dir, None)
        last_at = last_channel_rename_at.get(source_dir, 0.0)
        if not force and last_at and now - last_at < 30.0:
            log(
                f"channel rename throttled for {source_dir}: current={current} desired={desired} "
                f"wait={30.0 - (now - last_at):.1f}s",
                debug=True,
            )
            return False
        try:
            await discord_api_call(
                f"channel.edit name source={source_dir} channel={int(getattr(channel, 'id', 0) or 0)} desired={desired}",
                channel.edit(name=desired, reason="PBBatch folder status"),
                timeout=CHANNEL_RENAME_TIMEOUT_SECONDS,
            )
            last_channel_rename_at[source_dir] = time.monotonic()
            last_channel_rename_failed_at.pop(source_dir, None)
            store.set_channel(source_dir, int(channel.id), desired)
            log(f"channel renamed for {source_dir}: {current} -> {desired} | status={status}")
            return True
        except Exception as exc:
            last_channel_rename_failed_at[source_dir] = time.monotonic()
            detail = truncate(str(exc) or type(exc).__name__, 180)
            print(f"[discord-bot] channel rename failed for {source_dir}: {detail}", flush=True)
            return False

    async def ensure_inactive_message(channel: Any, source_dir: str) -> None:
        row = store.get_folder(source_dir)
        message_id = int(row["inactive_message_id"] or 0) if row is not None else 0
        if message_id:
            try:
                message = await discord_api_call(
                    f"channel.fetch_message inactive source={source_dir} message={message_id}",
                    channel.fetch_message(message_id),
                )
                if str(getattr(message, "content", "") or "") != INACTIVE_MESSAGE_TEXT:
                    await discord_api_call(
                        f"message.edit inactive source={source_dir} message={message_id}",
                        message.edit(content=INACTIVE_MESSAGE_TEXT),
                    )
                log(f"inactive message already present for {source_dir}: {message_id}", debug=True)
                return
            except Exception:
                log(f"inactive message fetch failed for {source_dir}: {message_id}", debug=True)
                pass
        try:
            message = await discord_api_call(
                f"channel.send inactive source={source_dir}",
                channel.send(INACTIVE_MESSAGE_TEXT),
            )
            store.set_inactive_message(source_dir, int(message.id))
            log(f"inactive message created for {source_dir}: {int(message.id)}")
        except Exception as exc:
            print(f"[discord-bot] inactive message failed for {source_dir}: {exc}", flush=True)

    async def clear_inactive_message(channel: Any, source_dir: str) -> None:
        row = store.get_folder(source_dir)
        message_id = int(row["inactive_message_id"] or 0) if row is not None else 0
        if not message_id:
            return
        try:
            message = await discord_api_call(
                f"channel.fetch_message clear_inactive source={source_dir} message={message_id}",
                channel.fetch_message(message_id),
            )
            await discord_api_call(
                f"message.delete inactive source={source_dir} message={message_id}",
                message.delete(),
            )
            log(f"inactive message cleared for {source_dir}: {message_id}")
        except Exception:
            log(f"inactive message delete/fetch failed for {source_dir}: {message_id}", debug=True)
            pass
        store.set_inactive_message(source_dir, 0)

    async def reconcile_existing_channels() -> None:
        rows = store.folder_rows()
        if not rows:
            log("reconcile: no known folder channels", debug=True)
            return
        log(f"reconciling {len(rows)} known folder channel(s)")
        for folder in rows:
            source_dir = str(folder["source_dir"] or "")
            if not source_dir:
                continue
            channel = await find_existing_channel(source_dir, int(folder["channel_id"] or 0))
            if channel is None:
                log(f"reconcile skipped {source_dir}: channel not found", debug=True)
                continue
            row = store.latest_session_for_source_dir(source_dir)
            snapshot = snapshot_for_row(row) if row is not None else {"source_dir": source_dir, "state": "offline"}
            live, reason = live_session_diagnostics(row)
            log(
                f"reconcile source={source_dir} channel={int(getattr(channel, 'id', 0) or 0)} "
                f"session={str(row['session_id'] or '-') if row else '-'} live={live} reason={reason}",
                debug=True,
            )
            schedule_channel_name_update(channel, source_dir, snapshot, force=True, label="reconcile channel rename")
            await ensure_inactive_message(channel, source_dir)
            if row is not None:
                await update_dashboard_timed(
                    f"reconcile source={source_dir}",
                    snapshot,
                    force=True,
                )

    async def mark_stale_sessions_offline_once() -> None:
        for folder in store.folder_rows():
            source_dir = str(folder["source_dir"] or "")
            if not source_dir:
                continue
            row = store.latest_session_for_source_dir(source_dir)
            if row is None:
                log(f"stale check skipped {source_dir}: no session row", debug=True)
                continue
            session_id = str(row["session_id"] or "")
            live, reason = live_session_diagnostics(row)
            if not session_id:
                log(f"stale check skipped {source_dir}: missing session id", debug=True)
                continue
            if session_id in stale_sessions_marked:
                log(f"stale check skipped {source_dir}: session already marked offline {session_id}", debug=True)
                continue
            if live:
                log(f"stale check ok {source_dir}: session={session_id} {reason}", debug=True)
                continue
            snapshot = snapshot_from_row(row)
            if str(snapshot.get("state") or "").strip().lower() in ("finished", "offline"):
                log(f"stale check skipped {source_dir}: session={session_id} already terminal state={snapshot.get('state')}", debug=True)
                continue
            recovered = recover_terminal_snapshot_from_runner_state(snapshot)
            if recovered is not None and stored_snapshot_is_terminal(recovered):
                log(f"runner session stale, recovered terminal state: source={source_dir} session={session_id}")
                if await update_dashboard_timed(
                    f"stale recovered terminal source={source_dir} session={session_id}",
                    recovered,
                    force=True,
                ):
                    stale_sessions_marked.add(session_id)
                continue
            log(f"runner session stale, marking offline: source={source_dir} session={session_id} reason={reason}")
            if not await update_dashboard_timed(
                f"stale offline source={source_dir} session={session_id}",
                offline_snapshot(snapshot),
                force=True,
            ):
                continue
            stale_sessions_marked.add(session_id)

    async def stale_session_monitor() -> None:
        while True:
            await asyncio.sleep(max(5.0, min(15.0, SESSION_STALE_SECONDS / 3.0)))
            try:
                await mark_stale_sessions_offline_once()
            except Exception as exc:
                print(f"[discord-bot] stale monitor failed: {exc}", flush=True)

    def create_logged_task(coro: Any, label: str) -> Any:
        task = asyncio.create_task(coro)

        def on_done(done_task: Any) -> None:
            try:
                exc = done_task.exception()
            except asyncio.CancelledError:
                return
            except Exception as callback_exc:
                log(f"background task status check failed: {label}: {truncate(callback_exc, 300)}")
                return
            if exc is not None:
                log(f"background task failed: {label}: {truncate(exc, 300)}")

        task.add_done_callback(on_done)
        return task

    async def scheduled_channel_name_update(source_dir: str) -> None:
        while True:
            request = pending_channel_rename_requests.get(source_dir)
            if request is None:
                return
            wait_seconds = channel_rename_wait_seconds(source_dir)
            if wait_seconds > 0:
                log(f"channel rename delayed: source={source_dir} wait={wait_seconds:.1f}s", debug=True)
                await asyncio.sleep(wait_seconds + 1.0)
                continue
            request = pending_channel_rename_requests.pop(source_dir, None)
            if request is None:
                return
            channel, snapshot, force, label = request
            row = store.latest_session_for_source_dir(source_dir)
            if row is not None:
                try:
                    snapshot = snapshot_for_row(row)
                except Exception as exc:
                    log(f"channel rename snapshot refresh failed: source={source_dir} error={truncate(exc, 180)}", debug=True)
            ok = await update_channel_name(channel, source_dir, snapshot, force=force)
            if ok and source_dir not in pending_channel_rename_requests:
                return
            if not ok:
                pending_channel_rename_requests[source_dir] = (channel, snapshot, force, label)

    def schedule_channel_name_update(
        channel: Any,
        source_dir: str,
        snapshot: Dict[str, Any],
        *,
        force: bool = False,
        label: str = "channel rename",
    ) -> bool:
        if channel is None or not source_dir:
            return False
        pending_channel_rename_requests[source_dir] = (channel, dict(snapshot), force, label)
        existing = channel_rename_tasks.get(source_dir)
        if existing is not None and not existing.done():
            log(f"channel rename already pending; desired state refreshed: source={source_dir} label={label}", debug=True)
            return False
        task = create_logged_task(
            scheduled_channel_name_update(source_dir),
            f"{label} source={source_dir}",
        )
        channel_rename_tasks[source_dir] = task

        def clear_task(done_task: Any) -> None:
            if channel_rename_tasks.get(source_dir) is done_task:
                channel_rename_tasks.pop(source_dir, None)

        task.add_done_callback(clear_task)
        return True

    async def delayed_dashboard_update(session_id: str, generation: int, delay: float) -> None:
        try:
            await asyncio.sleep(max(0.0, delay))
        except asyncio.CancelledError:
            return
        if pending_generation.get(session_id) != generation:
            log(
                f"delayed render skipped: session={session_id} generation={generation} "
                f"current_generation={pending_generation.get(session_id)}",
                debug=True,
            )
            return
        latest = pending_snapshots.pop(session_id, None)
        pending_tasks.pop(session_id, None)
        pending_due_at.pop(session_id, None)
        if latest is not None:
            log(f"delayed render firing: session={session_id} delay={fmt_elapsed(delay)}", debug=True)
            await update_dashboard_timed(f"delayed render session={session_id}", latest)

    def schedule_dashboard_update(session_id: str, snapshot: Dict[str, Any], delay: float) -> None:
        if not session_id:
            log(f"render schedule skipped: missing session id | {snapshot_summary(snapshot)}", debug=True)
            return
        source_dir = str(snapshot.get("source_dir") or "")
        source_lock = dashboard_locks.get(source_dir) if source_dir else None
        if source_lock is not None and source_lock.locked() and delay < 1.0:
            log(
                f"render schedule delayed because dashboard lock is busy: session={session_id} "
                f"source={source_dir} requested_due_in={fmt_elapsed(delay)}",
                debug=True,
            )
            delay = 1.0
        now = time.monotonic()
        due_at = now + max(0.0, delay)
        pending_snapshots[session_id] = snapshot
        existing_task = pending_tasks.get(session_id)
        existing_due_at = pending_due_at.get(session_id, 0.0)
        if existing_task is not None and not existing_task.done() and existing_due_at <= due_at + 0.5:
            log(
                f"render schedule coalesced: session={session_id} existing_due_in={fmt_elapsed(existing_due_at - now)} "
                f"new_due_in={fmt_elapsed(due_at - now)}",
                debug=True,
            )
            return
        if existing_task is not None and not existing_task.done():
            log(f"render schedule replacing pending task: session={session_id}", debug=True)
            existing_task.cancel()
        generation = pending_generation.get(session_id, 0) + 1
        pending_generation[session_id] = generation
        pending_due_at[session_id] = due_at
        pending_tasks[session_id] = create_logged_task(
            delayed_dashboard_update(session_id, generation, delay),
            f"delayed dashboard render session={session_id} generation={generation}",
        )
        log(
            f"render scheduled: session={session_id} generation={generation} due_in={fmt_elapsed(delay)} "
            f"{snapshot_summary(snapshot)}",
            debug=True,
        )

    def terminal_snapshot(snapshot: Dict[str, Any]) -> bool:
        return stored_snapshot_is_terminal(snapshot)

    async def update_dashboard(
        snapshot: Dict[str, Any],
        *,
        force: bool = False,
        takeover: bool = False,
        live: bool = False,
        render_messages: bool = True,
    ) -> None:
        session_id = str(snapshot.get("session_id") or "")
        source_dir = str(snapshot.get("source_dir") or "")
        if not session_id or not source_dir:
            log(
                f"dashboard update skipped: missing source/session source={source_dir or '-'} "
                f"session={session_id or '-'} keys={','.join(sorted(str(key) for key in snapshot.keys()))}",
                debug=True,
            )
            return
        if live and not snapshot.get("snapshot_at"):
            snapshot["snapshot_at"] = time.time()
        log(
            f"dashboard update requested: force={force} takeover={takeover} live={live} "
            f"render_messages={render_messages} {snapshot_summary(snapshot)}",
            debug=True,
        )
        lock = dashboard_locks.setdefault(source_dir, asyncio.Lock())
        if lock.locked() and not force and not takeover:
            log(
                f"dashboard update deferred: lock busy source={source_dir} session={session_id} "
                f"force={force} takeover={takeover}",
                debug=True,
            )
            schedule_dashboard_update(session_id, snapshot, 1.0)
            return
        log(f"dashboard lock waiting: source={source_dir} session={session_id}", debug=True)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=DASHBOARD_LOCK_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            log(
                f"dashboard lock timeout: source={source_dir} session={session_id} "
                f"after={fmt_elapsed(DASHBOARD_LOCK_TIMEOUT_SECONDS)}"
            )
            raise
        log(f"dashboard lock acquired: source={source_dir} session={session_id}", debug=True)
        try:
            owner_session_id = latest_session_by_source.get(source_dir)
            if takeover or (live and not owner_session_id):
                latest_session_by_source[source_dir] = session_id
                log(
                    f"dashboard owner set: source={source_dir} session={session_id} "
                    f"reason={'takeover' if takeover else 'live_without_owner'}",
                    debug=True,
                )
            elif owner_session_id and owner_session_id != session_id:
                log(
                    f"dashboard update skipped: source={source_dir} session={session_id} "
                    f"owner={owner_session_id} {snapshot_summary(snapshot)}",
                    debug=True,
                )
                return
            channel = await ensure_channel(source_dir)
            store.upsert_session(snapshot, int(channel.id))
            log(
                f"dashboard snapshot stored: source={source_dir} session={session_id} "
                f"channel={int(channel.id)} {snapshot_summary(snapshot)}",
                debug=True,
            )
            schedule_channel_name_update(
                channel,
                source_dir,
                snapshot,
                force=force or takeover,
                label="dashboard channel rename",
            )
            if live:
                await clear_inactive_message(channel, source_dir)
            elif str(snapshot.get("state") or "").strip().lower() == "offline":
                await ensure_inactive_message(channel, source_dir)

            settings = store.folder_settings(source_dir)
            if not render_messages:
                log(f"dashboard message render skipped: source={source_dir} session={session_id} render_messages=False", debug=True)
                return
            if bool(settings.get("updates_paused")) and not force and not takeover:
                log(f"dashboard message render skipped: updates paused source={source_dir} session={session_id}")
                return

            block_until = startup_render_block_until.get(session_id, 0.0)
            if block_until and not force:
                remaining = block_until - time.monotonic()
                if remaining > 0:
                    log(f"dashboard startup render delayed: session={session_id} remaining={fmt_elapsed(remaining)}", debug=True)
                    schedule_dashboard_update(session_id, snapshot, remaining)
                    return
                log(f"dashboard startup render block expired: session={session_id}", debug=True)
                startup_render_block_until.pop(session_id, None)

            signature = dashboard_major_signature(snapshot)
            last_at = last_render_at.get(session_id, 0.0)
            signature_changed = signature != last_major_signature.get(session_id)
            interval = float(
                settings.get("major_interval_seconds")
                if signature_changed
                else settings.get("progress_interval_seconds")
            )
            elapsed = time.monotonic() - last_at if last_at else interval
            if not force and elapsed < interval:
                log(
                    f"dashboard render throttled: session={session_id} elapsed={fmt_elapsed(elapsed)} "
                    f"interval={fmt_elapsed(interval)} signature_changed={signature_changed}",
                    debug=True,
                )
                schedule_dashboard_update(session_id, snapshot, interval - elapsed)
                return

            current_task = asyncio.current_task()
            task = pending_tasks.pop(session_id, None)
            pending_snapshots.pop(session_id, None)
            pending_due_at.pop(session_id, None)
            pending_generation[session_id] = pending_generation.get(session_id, 0) + 1
            if task is not None and task is not current_task and not task.done():
                log(f"dashboard render canceling superseded pending task: session={session_id}", debug=True)
                task.cancel()

            row = store.get_session(session_id)
            history_id = int(row["history_message_id"] or 0) if row else 0
            current_id = int(row["current_message_id"] or 0) if row else 0
            queue_id = int(row["queue_message_id"] or 0) if row else 0

            def message_signature(label: str, embed: Any, view: Any = None) -> str:
                def view_signature(value: Any) -> List[Dict[str, str]]:
                    if value is None:
                        return []
                    children = list(getattr(value, "children", []) or [])
                    return [
                        {
                            "type": type(child).__name__,
                            "custom_id": str(getattr(child, "custom_id", "") or ""),
                            "label": str(getattr(child, "label", "") or ""),
                            "style": str(getattr(child, "style", "") or ""),
                            "disabled": str(bool(getattr(child, "disabled", False))),
                        }
                        for child in children
                    ]

                embed_payload = embed.to_dict() if hasattr(embed, "to_dict") else {"repr": repr(embed)}
                return json.dumps(
                    {
                        "label": label,
                        "embed": embed_payload,
                        "view": view_signature(view),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )

            async def upsert_message(message_id: int, *, label: str, embed: Any, view: Any = None) -> int:
                signature_key = (session_id, label)
                signature = message_signature(label, embed, view)
                if message_id:
                    if not force and last_message_signature.get(signature_key) == signature:
                        log(
                            f"dashboard message unchanged: session={session_id} label={label} id={message_id}",
                            debug=True,
                        )
                        return int(message_id)
                    try:
                        partial_factory = getattr(channel, "get_partial_message", None)
                        if callable(partial_factory):
                            message = partial_factory(message_id)
                        else:
                            message = await discord_api_call(
                                f"channel.fetch_message dashboard label={label} session={session_id} message={message_id}",
                                channel.fetch_message(message_id),
                            )
                        await discord_api_call(
                            f"message.edit dashboard label={label} session={session_id} message={message_id}",
                            message.edit(embed=embed, view=view),
                        )
                        last_message_signature[signature_key] = signature
                        log(f"dashboard message edited: session={session_id} label={label} id={message_id}", debug=True)
                        return int(message_id)
                    except Exception as exc:
                        log(
                            f"dashboard message edit failed, sending replacement: session={session_id} "
                            f"label={label} id={message_id} error={truncate(exc, 180)}",
                            debug=True,
                        )
                        pass
                message = await discord_api_call(
                    f"channel.send dashboard label={label} session={session_id}",
                    channel.send(embed=embed, view=view),
                )
                last_message_signature[signature_key] = signature
                log(f"dashboard message sent: session={session_id} label={label} id={int(message.id)}", debug=True)
                return int(message.id)

            history_message_id = await upsert_message(
                history_id,
                label="overview",
                embed=render_overview_embed(discord, snapshot),
            )
            current_message_id = await upsert_message(
                current_id,
                label="current",
                embed=render_current_embed(discord, snapshot),
                view=build_control_view(session_id, snapshot),
            )
            if queue_id and queue_id not in (history_message_id, current_message_id):
                try:
                    partial_factory = getattr(channel, "get_partial_message", None)
                    if callable(partial_factory):
                        queue_msg = partial_factory(queue_id)
                    else:
                        queue_msg = await discord_api_call(
                            f"channel.fetch_message old_queue session={session_id} message={queue_id}",
                            channel.fetch_message(queue_id),
                        )
                    await discord_api_call(
                        f"message.delete old_queue session={session_id} message={queue_id}",
                        queue_msg.delete(),
                    )
                    log(f"dashboard old queue message deleted: session={session_id} id={queue_id}", debug=True)
                except Exception as exc:
                    log(f"dashboard old queue message delete failed: session={session_id} id={queue_id} error={truncate(exc, 180)}", debug=True)
                    pass
            store.set_session_messages(
                session_id,
                history=history_message_id,
                current=current_message_id,
                queue_message=0,
            )
            last_render_at[session_id] = time.monotonic()
            last_major_signature[session_id] = signature
            log(
                f"dashboard rendered: source={source_dir} session={session_id} "
                f"history={history_message_id} current={current_message_id} {snapshot_summary(snapshot)}"
            )
        finally:
            lock.release()
            log(f"dashboard lock released: source={source_dir} session={session_id}", debug=True)

    async def update_dashboard_timed(label: str, snapshot: Dict[str, Any], **kwargs: Any) -> bool:
        try:
            await asyncio.wait_for(update_dashboard(snapshot, **kwargs), timeout=DASHBOARD_UPDATE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            detail = f" detail={truncate(exc, 220)}" if str(exc) else ""
            log(f"dashboard update timed out: {label} after={fmt_elapsed(DASHBOARD_UPDATE_TIMEOUT_SECONDS)}{detail}")
            return False
        except Exception as exc:
            log(f"dashboard update failed: {label} error={truncate(exc, 300)}")
            return False
        return True

    def ingest_live_snapshot(snapshot: Dict[str, Any]) -> bool:
        session_id = str(snapshot.get("session_id") or "")
        source_dir = str(snapshot.get("source_dir") or "")
        if not session_id or not source_dir:
            log(
                f"live snapshot ingest skipped: missing source/session source={source_dir or '-'} "
                f"session={session_id or '-'}",
                debug=True,
            )
            return False
        if not snapshot.get("snapshot_at"):
            snapshot["snapshot_at"] = time.time()
        previous_owner = latest_session_by_source.get(source_dir)
        if previous_owner and previous_owner != session_id:
            owner_row = store.get_session(previous_owner)
            owner_live, owner_reason = live_session_diagnostics(owner_row)
            if owner_live:
                log(
                    f"live snapshot ignored for non-owner: source={source_dir} "
                    f"session={session_id} owner={previous_owner} owner_reason={owner_reason}",
                    debug=True,
                )
                return False
        latest_session_by_source[source_dir] = session_id
        was_stale = session_id in stale_sessions_marked
        stale_sessions_marked.discard(session_id)
        if previous_owner and previous_owner != session_id:
            log(f"live snapshot changed owner: source={source_dir} old_session={previous_owner} new_session={session_id}")
        if was_stale:
            log(f"live snapshot revived stale session: source={source_dir} session={session_id}")
        row = store.get_session(session_id)
        if row is None:
            log(f"live snapshot ingest needs full dashboard update: source={source_dir} session={session_id}", debug=True)
            return False
        store.upsert_session(snapshot, int(row["channel_id"] or 0))
        log(f"live snapshot ingested: source={source_dir} session={session_id} {snapshot_summary(snapshot)}", debug=True)
        return True

    async def enqueue_from_interaction(interaction: Any, session_id: str, command: str) -> None:
        if not is_authorized(interaction):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        row = store.get_session(session_id)
        if row is None:
            await interaction.response.send_message("Session not found.", ephemeral=True)
            return
        if not is_live_session_row(row):
            await interaction.response.send_message("No active runner is connected for this folder.", ephemeral=True)
            return
        if str(command or "").strip().lower().replace("-", "_").replace(" ", "_") == "resume":
            store.set_updates_paused(str(row["source_dir"]), False)
        store.enqueue_command(session_id, command)
        await interaction.response.send_message(f"Command queued: `{command}`", ephemeral=True)

    class CommandButton(discord.ui.Button):
        def __init__(self, *, session_id: str, command: str, label: str, style: Any) -> None:
            super().__init__(label=label, style=style, custom_id=f"pbbatch:cmd:{session_id}:{command}")
            self.session_id = session_id
            self.command = command

        async def callback(self, interaction: Any) -> None:
            await enqueue_from_interaction(interaction, self.session_id, self.command)

    def build_control_view(session_id: str, snapshot: Dict[str, Any]) -> Any:
        state = str(snapshot.get("state") or "").strip().lower()
        if state == "offline" or (state == "finished" and not list(snapshot.get("active") or [])):
            return None
        view = discord.ui.View(timeout=None)
        if bool(snapshot.get("paused")) or bool(snapshot.get("pause_after_current")):
            view.add_item(CommandButton(session_id=session_id, command="resume", label="Resume", style=discord.ButtonStyle.success))
        else:
            view.add_item(CommandButton(session_id=session_id, command="pause_after_current", label="Pause", style=discord.ButtonStyle.secondary))
        return view

    def latest_row_for_interaction(interaction: Any) -> Optional[sqlite3.Row]:
        return store.latest_session_for_channel(int(getattr(interaction, "channel_id", 0) or 0))

    def ensure_inactive_folder(snapshot: Dict[str, Any]) -> None:
        if list(snapshot.get("active") or []):
            raise RuntimeError("This command is disabled while a .plan is active in this folder.")

    async def send_path_response(interaction: Any, target: Path, *, ephemeral: bool = True) -> None:
        ensure_uploadable(target, max_upload_mb=config.max_upload_mb)
        await interaction.response.send_message(file=discord.File(str(target)), ephemeral=ephemeral)

    class LoaderFileButton(discord.ui.Button):
        def __init__(self, *, token: str, label: str) -> None:
            super().__init__(label=label, style=discord.ButtonStyle.secondary, custom_id=f"pbbatch:load:{token}")
            self.token = token

        async def callback(self, interaction: Any) -> None:
            cleanup_loader_files()
            entry = loader_files.get(self.token)
            if entry is None:
                await interaction.response.send_message("File handle expired.", ephemeral=True)
                return
            target, expires_at = entry
            if time.monotonic() > expires_at:
                loader_files.pop(self.token, None)
                await interaction.response.send_message("File handle expired.", ephemeral=True)
                return
            try:
                await send_path_response(interaction, target, ephemeral=True)
            except Exception as exc:
                await interaction.response.send_message(f"Upload failed: `{truncate(exc, 180)}`", ephemeral=True)

    def cleanup_loader_files() -> None:
        now = time.monotonic()
        for token, (_, expires_at) in list(loader_files.items()):
            if expires_at <= now:
                loader_files.pop(token, None)

    def build_loader_view(paths: List[Path]) -> Any:
        cleanup_loader_files()
        view = discord.ui.View(timeout=600)
        expires_at = time.monotonic() + LOADER_FILE_TTL_SECONDS
        for index, path in enumerate(paths[:20], start=1):
            token = uuid.uuid4().hex[:18]
            loader_files[token] = (path, expires_at)
            view.add_item(LoaderFileButton(token=token, label=str(index)))
        return view

    files_group = app_commands.Group(name="files", description="Folder-scoped file operations.")
    bot_group = app_commands.Group(name="bot", description="PBBatch Discord bot controls.")
    workdir_group = app_commands.Group(name="workdir", description="Configure this folder channel.")
    batch_group = app_commands.Group(name="batch", description="Folder-scoped batch utilities.")
    loader_group = app_commands.Group(name="loader", description="Inspect and upload generated content.")

    @files_group.command(name="get", description="Upload a file belonging to a selected .plan.")
    async def files_get(interaction: Any, selector: str, file: str) -> None:
        if not is_authorized(interaction):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        row = latest_row_for_interaction(interaction)
        if row is None:
            await interaction.response.send_message("No session in this channel.", ephemeral=True)
            return
        snapshot = snapshot_for_row(row)
        try:
            ensure_inactive_folder(snapshot)
            kind, plan = select_plan(snapshot, selector)
            if kind == "active":
                raise RuntimeError("Cannot read files for an active .plan.")
            await send_path_response(interaction, plan_named_file(plan, file), ephemeral=True)
        except Exception as exc:
            await interaction.response.send_message(f"File get failed: `{truncate(exc, 180)}`", ephemeral=True)

    @files_group.command(name="replace", description="Replace a file belonging to a selected .plan.")
    async def files_replace(interaction: Any, selector: str, file: str, upload: discord.Attachment) -> None:
        if not is_authorized(interaction):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        row = latest_row_for_interaction(interaction)
        if row is None:
            await interaction.response.send_message("No session in this channel.", ephemeral=True)
            return
        snapshot = snapshot_for_row(row)
        try:
            ensure_inactive_folder(snapshot)
            kind, plan = select_plan(snapshot, selector)
            if kind == "active":
                raise RuntimeError("Cannot replace files for an active .plan.")
            target = plan_named_file(plan, file)
            target.parent.mkdir(parents=True, exist_ok=True)
            await interaction.response.defer(ephemeral=True)
            await upload.save(str(target))
            await interaction.followup.send(f"Replaced `{target.name}` ({fmt_size(target.stat().st_size)}).", ephemeral=True)
        except Exception as exc:
            if not interaction.response.is_done():
                await interaction.response.send_message(f"File replace failed: `{truncate(exc, 180)}`", ephemeral=True)
            else:
                await interaction.followup.send(f"File replace failed: `{truncate(exc, 180)}`", ephemeral=True)

    @files_group.command(name="load", description="Upload a file by path relative to this folder.")
    async def files_load(interaction: Any, path: str) -> None:
        if not is_authorized(interaction):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        row = latest_row_for_interaction(interaction)
        if row is None:
            await interaction.response.send_message("No session in this channel.", ephemeral=True)
            return
        snapshot = snapshot_for_row(row)
        try:
            ensure_inactive_folder(snapshot)
            await send_path_response(interaction, safe_relative_path(Path(str(row["source_dir"])), path), ephemeral=True)
        except Exception as exc:
            await interaction.response.send_message(f"File load failed: `{truncate(exc, 180)}`", ephemeral=True)

    @bot_group.command(name="settings", description="View or update bot settings for this folder.")
    async def bot_settings(
        interaction: Any,
        major_interval_seconds: Optional[int] = None,
        progress_interval_seconds: Optional[int] = None,
        startup_delay_seconds: Optional[int] = None,
        updates_paused: Optional[bool] = None,
    ) -> None:
        if not is_authorized(interaction):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        row = latest_row_for_interaction(interaction)
        if row is None:
            await interaction.response.send_message("No session in this channel.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        source_dir = str(row["source_dir"])
        settings = store.folder_settings(source_dir)
        major = int(settings.get("major_interval_seconds") or DEFAULT_DASHBOARD_MAJOR_INTERVAL_SECONDS)
        progress = int(settings.get("progress_interval_seconds") or DEFAULT_DASHBOARD_PROGRESS_INTERVAL_SECONDS)
        startup_delay = int(settings.get("startup_delay_seconds", DEFAULT_RUNNER_STARTUP_DELAY_SECONDS))
        changed = False
        if major_interval_seconds is not None:
            major = max(5, min(int(major_interval_seconds), 3600))
            changed = True
        if progress_interval_seconds is not None:
            progress = max(15, min(int(progress_interval_seconds), 7200))
            changed = True
        if startup_delay_seconds is not None:
            startup_delay = max(0, min(int(startup_delay_seconds), 120))
            changed = True
        if changed:
            store.set_update_settings(source_dir, major=major, progress=progress, startup_delay=startup_delay)
        if updates_paused is not None:
            store.set_updates_paused(source_dir, bool(updates_paused))
            settings["updates_paused"] = bool(updates_paused)
            changed = True
        settings = store.folder_settings(source_dir)
        body = (
            f"updates_paused: {bool(settings.get('updates_paused'))}\n"
            f"major_interval_seconds: {int(settings.get('major_interval_seconds'))}\n"
            f"progress_interval_seconds: {int(settings.get('progress_interval_seconds'))}\n"
            f"startup_delay_seconds: {int(settings.get('startup_delay_seconds'))}"
        )
        if changed:
            snapshot = snapshot_for_row(row)
            schedule_channel_name_update(
                interaction.channel,
                source_dir,
                snapshot,
                force=True,
                label="bot settings channel rename",
            )
        await interaction.followup.send(f"```text\n{body}\n```", ephemeral=True)

    @bot_group.command(name="status", description="Show runner and bot status for this folder.")
    async def bot_status(interaction: Any) -> None:
        if not is_authorized(interaction):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        row = latest_row_for_interaction(interaction)
        if row is None:
            await interaction.response.send_message("No session in this channel.", ephemeral=True)
            return
        live = is_live_session_row(row)
        display_snapshot = snapshot_for_row(row)
        settings = store.folder_settings(str(row["source_dir"]))
        counts = dict(display_snapshot.get("counts") or {})
        body = (
            f"state: {display_snapshot.get('state')}\n"
            f"runner_connected: {live}\n"
            f"runner_paused: {bool(display_snapshot.get('paused'))}\n"
            f"pause_after_current: {bool(display_snapshot.get('pause_after_current'))}\n"
            f"bot_updates_paused: {bool(settings.get('updates_paused'))}\n"
            f"startup_delay_seconds: {int(settings.get('startup_delay_seconds') or 0)}\n"
            f"active: {counts.get('active', 0)}\n"
            f"queued: {counts.get('queued', 0)}\n"
            f"completed: {counts.get('completed', 0)}\n"
            f"failed: {counts.get('failed', 0)}"
        )
        await interaction.response.send_message(f"```text\n{body}\n```", ephemeral=True)

    @bot_group.command(name="pause", description="Pause dashboard message updates in this folder.")
    async def bot_pause(interaction: Any) -> None:
        if not is_authorized(interaction):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        row = latest_row_for_interaction(interaction)
        if row is None:
            await interaction.response.send_message("No session in this channel.", ephemeral=True)
            return
        source_dir = str(row["source_dir"])
        store.set_updates_paused(source_dir, True)
        await interaction.response.send_message("Dashboard updates paused for this folder.", ephemeral=True)
        schedule_channel_name_update(
            interaction.channel,
            source_dir,
            snapshot_for_row(row),
            force=True,
            label="bot pause channel rename",
        )

    @bot_group.command(name="resume", description="Resume dashboard message updates in this folder.")
    async def bot_resume(interaction: Any) -> None:
        if not is_authorized(interaction):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        row = latest_row_for_interaction(interaction)
        if row is None:
            await interaction.response.send_message("No session in this channel.", ephemeral=True)
            return
        source_dir = str(row["source_dir"])
        store.set_updates_paused(source_dir, False)
        snapshot = snapshot_for_row(row)
        await interaction.response.send_message("Dashboard updates resumed for this folder.", ephemeral=True)
        create_logged_task(
            update_dashboard_timed("bot resume command", snapshot, force=True),
            f"bot resume dashboard source={source_dir}",
        )

    @workdir_group.command(name="alias", description="Set channel alias for this folder.")
    async def workdir_alias(interaction: Any, name: str) -> None:
        if not is_authorized(interaction):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        row = latest_row_for_interaction(interaction)
        if row is None:
            await interaction.response.send_message("No session in this channel.", ephemeral=True)
            return
        source_dir = str(row["source_dir"])
        alias = sanitize_channel_component(name, fallback=Path(source_dir).name, limit=70)
        store.set_folder_alias(source_dir, alias)
        scheduled = schedule_channel_name_update(
            interaction.channel,
            source_dir,
            snapshot_for_row(row),
            force=True,
            label="workdir alias channel rename",
        )
        suffix = " Channel rename is running in background." if scheduled else " Channel rename is already pending."
        await interaction.response.send_message(f"Alias set to `{alias}`.{suffix}", ephemeral=True)

    @batch_group.command(name="edit-text", description="Replace matching text lines through batch-manager edit logic.")
    @app_commands.choices(
        target=[
            app_commands.Choice(name="{basename}.plan", value="plan"),
            app_commands.Choice(name="full-batch.plan", value="full-batch"),
            app_commands.Choice(name="fastpass-batch.plan", value="fastpass-batch"),
            app_commands.Choice(name=ZONED_COMMAND_NAME, value="zone"),
        ]
    )
    async def batch_edit_text(
        interaction: Any,
        target: app_commands.Choice[str],
        find: str,
        replacement: str,
        selection: str = "*",
    ) -> None:
        if not is_authorized(interaction):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        row = latest_row_for_interaction(interaction)
        if row is None:
            await interaction.response.send_message("No session in this channel.", ephemeral=True)
            return
        snapshot = snapshot_for_row(row)
        try:
            ensure_inactive_folder(snapshot)
        except Exception as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        output = await run_batch_edit_text(
            str(row["source_dir"]),
            target=str(target.value),
            find_text=find,
            replacement=replacement,
            selection=selection,
        )
        await interaction.followup.send(f"```text\n{truncate(output, 1900)}\n```", ephemeral=True)

    @loader_group.command(name="show", description="Show selected .plan workdir contents.")
    async def loader_show(interaction: Any, selector: str) -> None:
        if not is_authorized(interaction):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        row = latest_row_for_interaction(interaction)
        if row is None:
            await interaction.response.send_message("No session in this channel.", ephemeral=True)
            return
        try:
            _, plan = select_plan(snapshot_for_row(row), selector)
            workdir = Path(str(plan.get("workdir") or ""))
            body = tree_listing(workdir)
            await interaction.response.send_message(f"```text\n{truncate(body, 1900)}\n```", ephemeral=True)
        except Exception as exc:
            await interaction.response.send_message(f"Loader show failed: `{truncate(exc, 180)}`", ephemeral=True)

    async def loader_pass(interaction: Any, pass_name: str, selector: str, count: int, sort_mode: str) -> None:
        if not is_authorized(interaction):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        row = latest_row_for_interaction(interaction)
        if row is None:
            await interaction.response.send_message("No session in this channel.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        try:
            _, plan = select_plan(snapshot_for_row(row), selector or "active")
            files = list_ivf_files(plan, pass_name, count=count, sort_mode=sort_mode)
            if not files:
                await interaction.followup.send(f"No `{pass_name}pass` ivf files found.", ephemeral=True)
                return
            durations = await asyncio.gather(*(probe_media_duration_async(path) for path in files))
            rows = []
            for index, (path, duration) in enumerate(zip(files, durations), start=1):
                rows.append(f"{index:>2} | {duration:>9} | {fmt_size(path.stat().st_size):>9} | {path.name}")
            await interaction.followup.send(
                f"```text\n{truncate(chr(10).join(rows), 1900)}\n```",
                view=build_loader_view(files),
                ephemeral=True,
            )
        except Exception as exc:
            await interaction.followup.send(f"Loader failed: `{truncate(exc, 180)}`", ephemeral=True)

    @loader_group.command(name="mainpass", description="List mainpass ivf chunks and provide upload buttons.")
    @app_commands.choices(
        sort_mode=[
            app_commands.Choice(name="Index/name", value="name"),
            app_commands.Choice(name="Size", value="size"),
            app_commands.Choice(name="Modified time", value="mtime"),
        ]
    )
    async def loader_mainpass(
        interaction: Any,
        selector: str = "active",
        count: int = 10,
        sort_mode: str = "name",
    ) -> None:
        await loader_pass(interaction, "main", selector, count, str(sort_mode))

    @loader_group.command(name="fastpass", description="List fastpass ivf chunks and provide upload buttons.")
    @app_commands.choices(
        sort_mode=[
            app_commands.Choice(name="Index/name", value="name"),
            app_commands.Choice(name="Size", value="size"),
            app_commands.Choice(name="Modified time", value="mtime"),
        ]
    )
    async def loader_fastpass(
        interaction: Any,
        selector: str = "active",
        count: int = 10,
        sort_mode: str = "name",
    ) -> None:
        await loader_pass(interaction, "fast", selector, count, str(sort_mode))

    tree.add_command(files_group, guild=guild_object)
    tree.add_command(bot_group, guild=guild_object)
    tree.add_command(workdir_group, guild=guild_object)
    tree.add_command(batch_group, guild=guild_object)
    tree.add_command(loader_group, guild=guild_object)

    @client.event
    async def on_ready() -> None:
        nonlocal startup_reconciled, stale_monitor_started
        await tree.sync(guild=guild_object)
        if not startup_reconciled:
            startup_reconciled = True
            await reconcile_existing_channels()
        if not stale_monitor_started:
            stale_monitor_started = True
            create_logged_task(stale_session_monitor(), "stale session monitor")
        print(f"[discord-bot] logged in as {client.user}", flush=True)

    @tree.command(name="panel", description="Refresh the PBBatch dashboard in this channel.", guild=guild_object)
    async def panel(interaction: Any) -> None:
        if not is_authorized(interaction):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        row = store.latest_session_for_channel(int(interaction.channel_id))
        if row is None:
            await interaction.response.send_message("No PBBatch session is known for this channel.", ephemeral=True)
            return
        snapshot = snapshot_for_row(row)
        await update_dashboard_timed("panel command", snapshot, force=True)
        await interaction.response.send_message("Panel refreshed.", ephemeral=True)

    @tree.command(name="pbbatch_command", description="Queue a command for the latest runner session in this channel.", guild=guild_object)
    @app_commands.choices(command=[app_commands.Choice(name=label, value=name) for name, label in COMMANDS.items()])
    async def pbbatch_command(interaction: Any, command: app_commands.Choice[str]) -> None:
        row = store.latest_session_for_channel(int(interaction.channel_id))
        if row is None:
            await interaction.response.send_message("No active session in this channel.", ephemeral=True)
            return
        await enqueue_from_interaction(interaction, str(row["session_id"]), str(command.value))

    @tree.command(name="pbbatch_file", description="Upload a small runner file from the latest session.", guild=guild_object)
    @app_commands.choices(
        kind=[
            app_commands.Choice(name="Current .plan", value="plan"),
            app_commands.Choice(name="Runner state", value="state"),
            app_commands.Choice(name="Recent logs zip hint", value="logs"),
            app_commands.Choice(name="Zoned command", value="zone"),
        ]
    )
    async def pbbatch_file(interaction: Any, kind: app_commands.Choice[str]) -> None:
        if not is_authorized(interaction):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        row = store.latest_session_for_channel(int(interaction.channel_id))
        if row is None:
            await interaction.response.send_message("No session in this channel.", ephemeral=True)
            return
        snapshot = snapshot_for_row(row)
        active = list(snapshot.get("active") or [])
        source = active[0] if active else None
        if source is None:
            completed = list(snapshot.get("completed") or [])
            failed = list(snapshot.get("failed") or [])
            source = (completed + failed)[-1] if completed or failed else None
        if source is None:
            await interaction.response.send_message("No plan file is available yet.", ephemeral=True)
            return
        workdir = Path(str(source.get("workdir") or ""))
        plan = Path(str(source.get("plan") or ""))
        paths = {
            "plan": plan,
            "state": workdir / "00_meta" / "runner_state.json",
            "zone": workdir / ZONED_COMMAND_NAME,
            "crop": workdir / ZONED_COMMAND_NAME,
        }
        if str(kind.value) == "logs":
            await interaction.response.send_message(f"Logs directory: `{workdir / '00_logs'}`", ephemeral=True)
            return
        target = paths.get(str(kind.value))
        if target is None or not target.exists() or not target.is_file():
            await interaction.response.send_message("Requested file is missing.", ephemeral=True)
            return
        max_bytes = config.max_upload_mb * 1024 * 1024
        if target.stat().st_size > max_bytes:
            await interaction.response.send_message(f"File is too large for upload: `{target}` ({fmt_size(target.stat().st_size)})", ephemeral=True)
            return
        await interaction.response.send_message(file=discord.File(str(target)), ephemeral=True)

    @tree.command(name="pbbatch_batch_tool", description="Run a safe batch-manager tool for this folder.", guild=guild_object)
    @app_commands.choices(
        tool=[
            app_commands.Choice(name="Verify", value="verify"),
            app_commands.Choice(name="Pass Analytics", value="analytics"),
            app_commands.Choice(name="Make Web MP4", value="make_web_mp4"),
            app_commands.Choice(name="Config Dump", value="config_dump"),
        ],
        pass_name=[
            app_commands.Choice(name="Fastpass", value="fastpass"),
            app_commands.Choice(name="Mainpass", value="mainpass"),
            app_commands.Choice(name="Both", value="both"),
        ],
    )
    async def pbbatch_batch_tool(
        interaction: Any,
        tool: app_commands.Choice[str],
        pass_name: app_commands.Choice[str],
        confirm: bool = False,
    ) -> None:
        if not is_authorized(interaction):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        row = store.latest_session_for_channel(int(interaction.channel_id))
        if row is None:
            await interaction.response.send_message("No session in this channel.", ephemeral=True)
            return
        snapshot = snapshot_for_row(row)
        if snapshot.get("active"):
            await interaction.response.send_message("Batch tools are disabled while runner is active in this folder.", ephemeral=True)
            return
        if str(tool.value) == "config_dump" and not confirm:
            await interaction.response.send_message("Config Dump moves config files into `meta`. Re-run with `confirm: true`.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        output = await run_batch_tool(str(row["source_dir"]), str(tool.value), str(pass_name.value))
        await interaction.followup.send(f"```text\n{truncate(output, 1900)}\n```", ephemeral=True)

    async def http_register(request: Any) -> Any:
        payload = await request.json()
        snapshot = dict(payload.get("snapshot") or {})
        session_id = str(snapshot.get("session_id") or "")
        source_dir = str(snapshot.get("source_dir") or "")
        log(f"http register: source={source_dir or '-'} session={session_id or '-'} {snapshot_summary(snapshot)}")
        if session_id and source_dir:
            previous_owner = latest_session_by_source.get(source_dir)
            latest_session_by_source[source_dir] = session_id
            stale_sessions_marked.discard(session_id)
            if previous_owner and previous_owner != session_id:
                log(f"http register replaced owner: source={source_dir} old_session={previous_owner} new_session={session_id}")
        settings = store.folder_settings(source_dir) if source_dir else {}
        startup_delay = max(0.0, float(settings.get("startup_delay_seconds", DEFAULT_RUNNER_STARTUP_DELAY_SECONDS)))
        log(f"http register startup delay: source={source_dir or '-'} session={session_id or '-'} delay={fmt_elapsed(startup_delay)}", debug=True)
        if session_id and startup_delay > 0:
            startup_render_block_until[session_id] = time.monotonic() + startup_delay
        initial_ok = await update_dashboard_timed(
            f"http register initial source={source_dir or '-'} session={session_id or '-'}",
            snapshot,
            force=True,
            takeover=True,
            live=True,
            render_messages=False,
        )
        if not initial_ok:
            return web.json_response(
                {"status": "error", "message": "dashboard_register_failed"},
                status=503,
            )
        if session_id and startup_delay > 0:
            schedule_dashboard_update(session_id, snapshot, startup_delay)
        else:
            render_ok = await update_dashboard_timed(
                f"http register render source={source_dir or '-'} session={session_id or '-'}",
                snapshot,
                force=True,
                takeover=True,
                live=True,
            )
            if not render_ok:
                return web.json_response(
                    {"status": "error", "message": "dashboard_register_render_failed"},
                    status=503,
                )
        return web.json_response({"status": "ok"})

    async def http_snapshot(request: Any) -> Any:
        payload = await request.json()
        snapshot = dict(payload.get("snapshot") or {})
        ingested = ingest_live_snapshot(snapshot)
        terminal = terminal_snapshot(snapshot)
        log(
            f"http snapshot: ingested={ingested} terminal={terminal} {snapshot_summary(snapshot)}",
            debug=not terminal,
        )
        if ingested:
            if terminal:
                log(f"http terminal snapshot: {snapshot_summary(snapshot)}")
                create_logged_task(
                    update_dashboard_timed(
                        f"http terminal snapshot session={str(snapshot.get('session_id') or '-')}",
                        snapshot,
                        force=True,
                        live=True,
                    ),
                    f"terminal snapshot dashboard session={str(snapshot.get('session_id') or '-')}",
                )
            else:
                schedule_dashboard_update(str(snapshot.get("session_id") or ""), snapshot, 0.0)
        else:
            create_logged_task(
                update_dashboard_timed(
                    f"http snapshot session={str(snapshot.get('session_id') or '-')}",
                    snapshot,
                    live=True,
                ),
                f"snapshot dashboard session={str(snapshot.get('session_id') or '-')}",
            )
        return web.json_response({"status": "ok"})

    async def http_event(request: Any) -> Any:
        session_id = str(request.match_info["session_id"])
        payload = await request.json()
        event = dict(payload.get("event") or {})
        snapshot = dict(payload.get("snapshot") or {})
        event_name = str(event.get("event") or "")
        stage = str(event.get("stage") or "")
        status = str(event.get("status") or "")
        plan_run_id = str(event.get("plan_run_id") or "")
        if event_name != "runner_heartbeat":
            store.add_event(session_id, event)
            log(
                f"http event: session={session_id} event={event_name or '-'} "
                f"stage={stage or '-'} status={status or '-'} plan={plan_run_id or '-'} {snapshot_summary(snapshot)}"
            )
        else:
            log(f"http heartbeat: session={session_id} {snapshot_summary(snapshot)}", debug=True)
        ingested = ingest_live_snapshot(snapshot)
        terminal = terminal_snapshot(snapshot)
        if ingested:
            if terminal:
                log(f"http terminal event snapshot: session={session_id} event={event_name or '-'} {snapshot_summary(snapshot)}")
                create_logged_task(
                    update_dashboard_timed(
                        f"http terminal event session={str(snapshot.get('session_id') or session_id)} event={event_name or '-'}",
                        snapshot,
                        force=True,
                        live=True,
                    ),
                    f"terminal event dashboard session={str(snapshot.get('session_id') or session_id)} event={event_name or '-'}",
                )
            else:
                schedule_dashboard_update(str(snapshot.get("session_id") or session_id), snapshot, 0.0)
        else:
            create_logged_task(
                update_dashboard_timed(
                    f"http event session={str(snapshot.get('session_id') or session_id)} event={event_name or '-'}",
                    snapshot,
                    live=True,
                ),
                f"event dashboard session={str(snapshot.get('session_id') or session_id)} event={event_name or '-'}",
            )
        snapshot_source_dir = str(snapshot.get("source_dir") or "")
        owner_session_id = latest_session_by_source.get(snapshot_source_dir) if snapshot_source_dir else ""
        if owner_session_id and owner_session_id != session_id:
            log(
                f"http event ignored for non-owner notification: source={snapshot_source_dir} "
                f"session={session_id} owner={owner_session_id} event={event_name or '-'}",
                debug=True,
            )
            return web.json_response({"status": "ok"})
        if event.get("status") == "failed":
            if stage == "Item":
                return web.json_response({"status": "ok"})
            now = time.monotonic()
            for key, sent_at in list(sent_failed_event_keys.items()):
                if now - sent_at > 3600.0:
                    sent_failed_event_keys.pop(key, None)
            event_key = f"{session_id}:{plan_run_id}:{stage}"
            if event_key in sent_failed_event_keys:
                return web.json_response({"status": "ok"})
            sent_failed_event_keys[event_key] = now
            row = store.get_session(session_id)
            if row:
                channel = client.get_channel(int(row["channel_id"]))
                if channel is not None:
                    embed = discord.Embed(title="Important event", color=0xED4245)
                    embed.description = truncate(f"{event.get('stage')} failed: {event.get('message')}", 4000)
                    await discord_api_call(
                        f"channel.send failed_event session={session_id} stage={stage or '-'}",
                        channel.send(embed=embed),
                    )
        return web.json_response({"status": "ok"})

    async def http_commands(request: Any) -> Any:
        session_id = str(request.match_info["session_id"])
        return web.json_response(store.pending_commands(session_id))

    async def http_ack(request: Any) -> Any:
        command_id = str(request.match_info["command_id"])
        payload = await request.json()
        store.ack_command(command_id, str(payload.get("status") or ""), str(payload.get("message") or ""))
        log(
            f"http command ack: command={command_id} status={str(payload.get('status') or '-')} "
            f"message={truncate(payload.get('message'), 180)}",
            debug=True,
        )
        snapshot = dict(payload.get("snapshot") or {})
        if snapshot:
            ingest_live_snapshot(snapshot)
            create_logged_task(
                update_dashboard_timed(
                    f"http command ack command={command_id}",
                    snapshot,
                    force=True,
                    live=True,
                ),
                f"command ack dashboard command={command_id}",
            )
        return web.json_response({"status": "ok"})

    @web.middleware
    async def shared_secret_middleware(request: Any, handler: Any) -> Any:
        if config.shared_secret:
            received = str(request.headers.get("X-PBBATCH-Discord-Secret") or "")
            if received != config.shared_secret:
                return web.json_response({"status": "error", "message": "unauthorized"}, status=401)
        return await handler(request)

    app = web.Application(middlewares=[shared_secret_middleware])
    app.router.add_post("/api/sessions/register", http_register)
    app.router.add_post("/api/sessions/{session_id}/snapshot", http_snapshot)
    app.router.add_post("/api/sessions/{session_id}/events", http_event)
    app.router.add_get("/api/sessions/{session_id}/commands", http_commands)
    app.router.add_post("/api/commands/{command_id}/ack", http_ack)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, config.host, config.port)
    await site.start()
    log(f"local service listening on http://{config.host}:{config.port}")
    log("cmd commands: status")
    try:
        await client.start(config.token)
    finally:
        await runner.cleanup()
