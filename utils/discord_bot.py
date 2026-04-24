from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import io
import os
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.discord_config import discord_config_int, discord_config_value, read_discord_config
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
    select_plan,
    snapshot_from_row,
    tree_listing,
    truncate,
)
from utils.zoned_commands import ZONED_COMMAND_NAME

try:
    import discord  # type: ignore[import-not-found]
    from aiohttp import web  # type: ignore[import-not-found]
    from discord import app_commands  # type: ignore[import-not-found]
except ImportError:
    discord = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]
    app_commands = None  # type: ignore[assignment]



DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8794
MAX_UPLOAD_MB_DEFAULT = 25
LOADER_FILE_TTL_SECONDS = 600.0
ADMINS_ONLY = False
INACTIVE_MESSAGE_TEXT = "Runner is not connected for this folder. This message will be removed when a runner registers again."
SESSION_STALE_SECONDS = 45.0

COMMANDS = {
    "pause_after_current": "Pause",
    "resume": "Resume",
}


@dataclass(frozen=True)
class BotConfig:
    token: str
    guild_id: int
    category_id: int
    host: str
    port: int
    db_path: Path
    max_upload_mb: int
    admin_role_id: int = 0
    operator_role_id: int = 0
    admins_only: bool = False
    shared_secret: str = ""


def load_config(args: argparse.Namespace) -> BotConfig:
    file_values = read_discord_config()
    appdata = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    default_db = appdata / "PBBatchProcessUtil" / "discord" / "state.sqlite3"
    return BotConfig(
        token=args.token or discord_config_value("PBBATCH_DISCORD_TOKEN", "", file_values=file_values),
        guild_id=int(args.guild_id or discord_config_int("PBBATCH_DISCORD_GUILD_ID", 0, file_values=file_values)),
        category_id=int(args.category_id or discord_config_int("PBBATCH_DISCORD_CATEGORY_ID", 0, file_values=file_values)),
        host=args.host or discord_config_value("PBBATCH_DISCORD_HOST", DEFAULT_HOST, file_values=file_values),
        port=int(args.port or discord_config_int("PBBATCH_DISCORD_PORT", DEFAULT_PORT, file_values=file_values)),
        db_path=Path(args.db or discord_config_value("PBBATCH_DISCORD_STATE_DB", str(default_db), file_values=file_values)).expanduser(),
        max_upload_mb=int(
            args.max_upload_mb
            or discord_config_int("PBBATCH_DISCORD_MAX_UPLOAD_MB", MAX_UPLOAD_MB_DEFAULT, file_values=file_values)
        ),
        admin_role_id=discord_config_int("PBBATCH_DISCORD_ADMIN_ROLE_ID", 0, file_values=file_values),
        operator_role_id=discord_config_int("PBBATCH_DISCORD_OPERATOR_ROLE_ID", 0, file_values=file_values),
        admins_only=bool(discord_config_int("PBBATCH_DISCORD_ADMINS_ONLY", 1 if ADMINS_ONLY else 0, file_values=file_values)),
        shared_secret=discord_config_value("PBBATCH_DISCORD_SHARED_SECRET", "", file_values=file_values),
    )


def load_batch_manager_module() -> Any:
    path = ROOT / "utils" / "batch-manager.py"
    spec = importlib.util.spec_from_file_location("pbbatch_batch_manager", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["pbbatch_batch_manager"] = module
    spec.loader.exec_module(module)
    return module


async def run_batch_tool(source_dir: str, tool: str, pass_name: str = "") -> str:
    def work() -> str:
        module = load_batch_manager_module()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            groups, unknown = module.collect_sources([source_dir])
            if unknown:
                print("Unknown inputs:")
                for item in unknown:
                    print(f"  {item}")
            if not groups:
                print("No sources found.")
                return buf.getvalue()
            for group in groups:
                print()
                print(f"[{tool}] {group.base}")
                if tool == "verify":
                    module.verify_config(group, check_filters=True, check_params=False)
                elif tool == "make_web_mp4":
                    module.make_web_mp4(group)
                elif tool == "analytics":
                    selected = [pass_name] if pass_name in ("fastpass", "mainpass") else ["fastpass", "mainpass"]
                    for selected_pass in selected:
                        module.run_pass_analytics(group, selected_pass)
                elif tool == "config_dump":
                    module.config_dump(groups)
                    break
                else:
                    print(f"Unsupported tool: {tool}")
        return buf.getvalue()

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, work)


async def run_batch_edit_text(
    source_dir: str,
    *,
    target: str,
    find_text: str,
    replacement: str,
    selection: str,
) -> str:
    def work() -> str:
        module = load_batch_manager_module()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            groups, unknown = module.collect_sources([source_dir])
            if unknown:
                print("Unknown inputs:")
                for item in unknown:
                    print(f"  {item}")
            if not groups:
                print("No sources found.")
                return buf.getvalue()
            matches = module.find_edit_matches(groups, target, find_text)
            if not matches:
                print(f"[skip] no matches found in {module.edit_target_label(target)}.")
                return buf.getvalue()
            module.print_edit_matches(matches)
            selected_ids = module.enter_numbers(selection or "*", 1, len(matches))
            selected = [m for m in matches if m.index in set(selected_ids)]
            if not selected:
                print("[skip] nothing selected.")
                return buf.getvalue()
            changed = module.replace_selected_lines(selected, replacement)
            print(f"[done] Edit completed, replaced {changed} line(s).")
        return buf.getvalue()

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, work)


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
    channel_locks: Dict[str, Any] = {}
    dashboard_locks: Dict[str, Any] = {}
    loader_files: Dict[str, Tuple[Path, float]] = {}
    last_channel_rename_at: Dict[str, float] = {}
    sent_failed_event_keys: Dict[str, float] = {}
    startup_reconciled = False
    stale_monitor_started = False
    stale_sessions_marked: set[str] = set()

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
            return display_stage_name(dict(running[-1]))
        completed = [stage for stage in stages if str(stage.get("status") or "").lower() in ("completed", "skipped")]
        if completed:
            return display_stage_name(dict(completed[-1]))
        return "-"

    def is_live_session_row(row: Optional[sqlite3.Row]) -> bool:
        if row is None:
            return False
        source_dir = str(row["source_dir"] or "")
        session_id = str(row["session_id"] or "")
        if not source_dir or not session_id or latest_session_by_source.get(source_dir) != session_id:
            return False
        snapshot = snapshot_from_row(row)
        try:
            heartbeat_at = float(snapshot.get("snapshot_at") or row["updated_at"] or 0.0)
            if time.time() - heartbeat_at > SESSION_STALE_SECONDS:
                return False
        except Exception:
            return False
        state = str(snapshot.get("state") or "").strip().lower()
        return state not in ("finished", "offline")

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

    def snapshot_for_row(row: sqlite3.Row) -> Dict[str, Any]:
        snapshot = snapshot_from_row(row)
        return snapshot if is_live_session_row(row) else offline_snapshot(snapshot)

    async def print_console_status() -> None:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        rows = store.folder_rows()
        print(f"[discord-bot] status at {now}", flush=True)
        print(f"[discord-bot] folders: {len(rows)}", flush=True)
        if not rows:
            return
        for folder in rows:
            source_dir = str(folder["source_dir"] or "")
            channel_id = int(folder["channel_id"] or 0)
            stored_name = str(folder["channel_name"] or "")
            settings = store.folder_settings(source_dir)
            row = store.latest_session_for_source_dir(source_dir)
            snapshot = snapshot_from_row(row) if row else {}
            live = is_live_session_row(row)
            display_snapshot = snapshot if live else offline_snapshot(snapshot)
            status = channel_status_key(display_snapshot, updates_paused=bool(settings.get("updates_paused")))
            channel = client.get_channel(channel_id) if channel_id else None
            channel_name = str(getattr(channel, "name", "") or stored_name or "-")
            session_id = str(row["session_id"] or "-") if row else "-"
            counts = dict(display_snapshot.get("counts") or {})
            print("", flush=True)
            print(f"[folder] {source_dir}", flush=True)
            print(f"  channel: {channel_name} ({channel_id or '-'})", flush=True)
            print(f"  status: {status}", flush=True)
            print(f"  session: {session_id}{'' if live else ' (offline/stale)'}", flush=True)
            print(f"  bot_updates_paused: {bool(settings.get('updates_paused'))}", flush=True)
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
        async with lock:
            await client.wait_until_ready()
            existing_id = store.get_channel_id(source_dir)
            settings = store.folder_settings(source_dir)
            channel_name = channel_name_for_source(source_dir, alias=str(settings.get("alias") or ""), status="idle")
            guild = client.get_guild(config.guild_id)
            if guild is None:
                raise RuntimeError(f"guild not found: {config.guild_id}")
            if existing_id:
                channel = client.get_channel(existing_id)
                if channel is None:
                    try:
                        channel = await guild.fetch_channel(existing_id)
                    except Exception:
                        channel = None
                if channel is not None:
                    return channel
            category = guild.get_channel(config.category_id)
            if category is None:
                try:
                    category = await guild.fetch_channel(config.category_id)
                except Exception:
                    category = None
            if category is None:
                raise RuntimeError(f"category not found: {config.category_id}")
            candidates = list(getattr(category, "channels", []) or [])
            try:
                fetched_channels = await guild.fetch_channels()
                for channel in fetched_channels:
                    parent_id = int(getattr(channel, "category_id", 0) or getattr(channel, "parent_id", 0) or 0)
                    if parent_id == int(config.category_id):
                        candidates.append(channel)
            except Exception:
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
                    return channel
            channel = await guild.create_text_channel(
                name=channel_name,
                category=category,
                topic=f"PBBatch source: {source_dir}",
                reason="PBBatch runner session",
            )
            store.set_channel(source_dir, int(channel.id), channel_name)
            return channel

    async def find_existing_channel(source_dir: str, channel_id: int) -> Any:
        await client.wait_until_ready()
        guild = client.get_guild(config.guild_id)
        if guild is None:
            return None
        if channel_id:
            channel = client.get_channel(channel_id)
            if channel is None:
                try:
                    channel = await guild.fetch_channel(channel_id)
                except Exception:
                    channel = None
            if channel is not None:
                return channel
        category = guild.get_channel(config.category_id)
        if category is None:
            try:
                category = await guild.fetch_channel(config.category_id)
            except Exception:
                category = None
        candidates = list(getattr(category, "channels", []) or []) if category is not None else []
        try:
            fetched_channels = await guild.fetch_channels()
            for channel in fetched_channels:
                parent_id = int(getattr(channel, "category_id", 0) or getattr(channel, "parent_id", 0) or 0)
                if parent_id == int(config.category_id):
                    candidates.append(channel)
        except Exception:
            pass
        seen_ids: set[int] = set()
        for channel in candidates:
            candidate_id = int(getattr(channel, "id", 0) or 0)
            if not candidate_id or candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            if str(getattr(channel, "topic", "") or "") == f"PBBatch source: {source_dir}":
                store.set_channel(source_dir, candidate_id, str(getattr(channel, "name", "") or ""))
                return channel
        return None

    async def update_channel_name(channel: Any, source_dir: str, snapshot: Dict[str, Any], *, force: bool = False) -> None:
        settings = store.folder_settings(source_dir)
        status = channel_status_key(snapshot, updates_paused=bool(settings.get("updates_paused")))
        desired = channel_name_for_source(source_dir, alias=str(settings.get("alias") or ""), status=status)
        current = str(getattr(channel, "name", "") or "")
        if current == desired:
            return
        now = time.monotonic()
        last_at = last_channel_rename_at.get(source_dir, 0.0)
        if not force and last_at and now - last_at < 30.0:
            return
        try:
            await channel.edit(name=desired, reason="PBBatch folder status")
            last_channel_rename_at[source_dir] = now
            store.set_channel(source_dir, int(channel.id), desired)
        except Exception as exc:
            print(f"[discord-bot] channel rename failed for {source_dir}: {exc}", flush=True)

    async def ensure_inactive_message(channel: Any, source_dir: str) -> None:
        row = store.get_folder(source_dir)
        message_id = int(row["inactive_message_id"] or 0) if row is not None else 0
        if message_id:
            try:
                message = await channel.fetch_message(message_id)
                if str(getattr(message, "content", "") or "") != INACTIVE_MESSAGE_TEXT:
                    await message.edit(content=INACTIVE_MESSAGE_TEXT)
                return
            except Exception:
                pass
        try:
            message = await channel.send(INACTIVE_MESSAGE_TEXT)
            store.set_inactive_message(source_dir, int(message.id))
        except Exception as exc:
            print(f"[discord-bot] inactive message failed for {source_dir}: {exc}", flush=True)

    async def clear_inactive_message(channel: Any, source_dir: str) -> None:
        row = store.get_folder(source_dir)
        message_id = int(row["inactive_message_id"] or 0) if row is not None else 0
        if not message_id:
            return
        try:
            message = await channel.fetch_message(message_id)
            await message.delete()
        except Exception:
            pass
        store.set_inactive_message(source_dir, 0)

    async def reconcile_existing_channels() -> None:
        rows = store.folder_rows()
        if not rows:
            return
        print(f"[discord-bot] reconciling {len(rows)} known folder channel(s)", flush=True)
        for folder in rows:
            source_dir = str(folder["source_dir"] or "")
            if not source_dir:
                continue
            channel = await find_existing_channel(source_dir, int(folder["channel_id"] or 0))
            if channel is None:
                continue
            row = store.latest_session_for_source_dir(source_dir)
            snapshot = snapshot_for_row(row) if row is not None else {"source_dir": source_dir, "state": "offline"}
            await update_channel_name(channel, source_dir, snapshot, force=True)
            await ensure_inactive_message(channel, source_dir)
            if row is not None:
                await update_dashboard(snapshot, force=True)

    async def mark_stale_sessions_offline_once() -> None:
        for folder in store.folder_rows():
            source_dir = str(folder["source_dir"] or "")
            if not source_dir:
                continue
            row = store.latest_session_for_source_dir(source_dir)
            if row is None:
                continue
            session_id = str(row["session_id"] or "")
            if not session_id or session_id in stale_sessions_marked or is_live_session_row(row):
                continue
            snapshot = snapshot_from_row(row)
            if str(snapshot.get("state") or "").strip().lower() in ("finished", "offline"):
                continue
            stale_sessions_marked.add(session_id)
            print(f"[discord-bot] runner session stale, marking offline: {session_id}", flush=True)
            await update_dashboard(offline_snapshot(snapshot), force=True)

    async def stale_session_monitor() -> None:
        while True:
            await asyncio.sleep(max(5.0, min(15.0, SESSION_STALE_SECONDS / 3.0)))
            try:
                await mark_stale_sessions_offline_once()
            except Exception as exc:
                print(f"[discord-bot] stale monitor failed: {exc}", flush=True)

    async def delayed_dashboard_update(session_id: str, generation: int, delay: float) -> None:
        try:
            await asyncio.sleep(max(0.0, delay))
        except asyncio.CancelledError:
            return
        if pending_generation.get(session_id) != generation:
            return
        latest = pending_snapshots.pop(session_id, None)
        pending_tasks.pop(session_id, None)
        pending_due_at.pop(session_id, None)
        if latest is not None:
            await update_dashboard(latest)

    def schedule_dashboard_update(session_id: str, snapshot: Dict[str, Any], delay: float) -> None:
        now = time.monotonic()
        due_at = now + max(0.0, delay)
        pending_snapshots[session_id] = snapshot
        existing_task = pending_tasks.get(session_id)
        existing_due_at = pending_due_at.get(session_id, 0.0)
        if existing_task is not None and not existing_task.done() and existing_due_at <= due_at + 0.5:
            return
        if existing_task is not None and not existing_task.done():
            existing_task.cancel()
        generation = pending_generation.get(session_id, 0) + 1
        pending_generation[session_id] = generation
        pending_due_at[session_id] = due_at
        pending_tasks[session_id] = asyncio.create_task(delayed_dashboard_update(session_id, generation, delay))

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
            return
        if live and not snapshot.get("snapshot_at"):
            snapshot["snapshot_at"] = time.time()
        lock = dashboard_locks.setdefault(source_dir, asyncio.Lock())
        async with lock:
            owner_session_id = latest_session_by_source.get(source_dir)
            if takeover or (live and not owner_session_id):
                latest_session_by_source[source_dir] = session_id
            elif owner_session_id and owner_session_id != session_id:
                return
            channel = await ensure_channel(source_dir)
            store.upsert_session(snapshot, int(channel.id))
            await update_channel_name(channel, source_dir, snapshot, force=force or takeover)
            if live:
                await clear_inactive_message(channel, source_dir)
            elif str(snapshot.get("state") or "").strip().lower() == "offline":
                await ensure_inactive_message(channel, source_dir)

            settings = store.folder_settings(source_dir)
            if not render_messages:
                return
            if bool(settings.get("updates_paused")) and not force and not takeover:
                return

            block_until = startup_render_block_until.get(session_id, 0.0)
            if block_until and not force:
                remaining = block_until - time.monotonic()
                if remaining > 0:
                    schedule_dashboard_update(session_id, snapshot, remaining)
                    return
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
                schedule_dashboard_update(session_id, snapshot, interval - elapsed)
                return

            current_task = asyncio.current_task()
            task = pending_tasks.pop(session_id, None)
            pending_snapshots.pop(session_id, None)
            pending_due_at.pop(session_id, None)
            pending_generation[session_id] = pending_generation.get(session_id, 0) + 1
            if task is not None and task is not current_task and not task.done():
                task.cancel()

            row = store.get_session(session_id)
            history_id = int(row["history_message_id"] or 0) if row else 0
            current_id = int(row["current_message_id"] or 0) if row else 0
            queue_id = int(row["queue_message_id"] or 0) if row else 0

            async def upsert_message(message_id: int, *, embed: Any, view: Any = None) -> Any:
                if message_id:
                    try:
                        message = await channel.fetch_message(message_id)
                        await message.edit(embed=embed, view=view)
                        return message
                    except Exception:
                        pass
                return await channel.send(embed=embed, view=view)

            history_msg = await upsert_message(history_id, embed=render_overview_embed(discord, snapshot))
            current_msg = await upsert_message(
                current_id,
                embed=render_current_embed(discord, snapshot),
                view=build_control_view(session_id, snapshot),
            )
            if queue_id and queue_id not in (int(history_msg.id), int(current_msg.id)):
                try:
                    queue_msg = await channel.fetch_message(queue_id)
                    await queue_msg.delete()
                except Exception:
                    pass
            store.set_session_messages(
                session_id,
                history=int(history_msg.id),
                current=int(current_msg.id),
                queue_message=0,
            )
            last_render_at[session_id] = time.monotonic()
            last_major_signature[session_id] = signature

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
        if str(snapshot.get("state") or "").strip().lower() == "offline":
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
            await update_channel_name(interaction.channel, source_dir, snapshot, force=True)
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
        await interaction.response.defer(ephemeral=True)
        store.set_updates_paused(str(row["source_dir"]), True)
        await update_channel_name(interaction.channel, str(row["source_dir"]), snapshot_for_row(row), force=True)
        await interaction.followup.send("Dashboard updates paused for this folder.", ephemeral=True)

    @bot_group.command(name="resume", description="Resume dashboard message updates in this folder.")
    async def bot_resume(interaction: Any) -> None:
        if not is_authorized(interaction):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        row = latest_row_for_interaction(interaction)
        if row is None:
            await interaction.response.send_message("No session in this channel.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        source_dir = str(row["source_dir"])
        store.set_updates_paused(source_dir, False)
        snapshot = snapshot_for_row(row)
        await update_dashboard(snapshot, force=True)
        await interaction.followup.send("Dashboard updates resumed for this folder.", ephemeral=True)

    @workdir_group.command(name="alias", description="Set channel alias for this folder.")
    async def workdir_alias(interaction: Any, name: str) -> None:
        if not is_authorized(interaction):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        row = latest_row_for_interaction(interaction)
        if row is None:
            await interaction.response.send_message("No session in this channel.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        source_dir = str(row["source_dir"])
        alias = sanitize_channel_component(name, fallback=Path(source_dir).name, limit=70)
        store.set_folder_alias(source_dir, alias)
        await update_channel_name(interaction.channel, source_dir, snapshot_for_row(row), force=True)
        await interaction.followup.send(f"Alias set to `{alias}`.", ephemeral=True)

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
            asyncio.create_task(stale_session_monitor())
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
        await update_dashboard(snapshot, force=True)
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
        if session_id and source_dir:
            latest_session_by_source[source_dir] = session_id
            stale_sessions_marked.discard(session_id)
        settings = store.folder_settings(source_dir) if source_dir else {}
        startup_delay = max(0.0, float(settings.get("startup_delay_seconds", DEFAULT_RUNNER_STARTUP_DELAY_SECONDS)))
        if session_id and startup_delay > 0:
            startup_render_block_until[session_id] = time.monotonic() + startup_delay
        await update_dashboard(snapshot, force=True, takeover=True, live=True, render_messages=False)
        if session_id and startup_delay > 0:
            schedule_dashboard_update(session_id, snapshot, startup_delay)
        else:
            await update_dashboard(snapshot, force=True, takeover=True, live=True)
        return web.json_response({"status": "ok"})

    async def http_snapshot(request: Any) -> Any:
        payload = await request.json()
        snapshot = dict(payload.get("snapshot") or {})
        await update_dashboard(snapshot, live=True)
        return web.json_response({"status": "ok"})

    async def http_event(request: Any) -> Any:
        session_id = str(request.match_info["session_id"])
        payload = await request.json()
        event = dict(payload.get("event") or {})
        snapshot = dict(payload.get("snapshot") or {})
        store.add_event(session_id, event)
        await update_dashboard(snapshot, live=True)
        if event.get("status") == "failed":
            stage = str(event.get("stage") or "")
            plan_run_id = str(event.get("plan_run_id") or "")
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
                    await channel.send(embed=embed)
        return web.json_response({"status": "ok"})

    async def http_commands(request: Any) -> Any:
        session_id = str(request.match_info["session_id"])
        return web.json_response(store.pending_commands(session_id))

    async def http_ack(request: Any) -> Any:
        command_id = str(request.match_info["command_id"])
        payload = await request.json()
        store.ack_command(command_id, str(payload.get("status") or ""), str(payload.get("message") or ""))
        snapshot = dict(payload.get("snapshot") or {})
        if snapshot:
            await update_dashboard(snapshot, force=True, live=True)
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
    print(f"[discord-bot] local service listening on http://{config.host}:{config.port}", flush=True)
    print("[discord-bot] cmd commands: status", flush=True)
    try:
        await client.start(config.token)
    finally:
        await runner.cleanup()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PBBatchProcessUtil Discord bot service.")
    parser.add_argument("--token", default="")
    parser.add_argument("--guild-id", default="")
    parser.add_argument("--category-id", default="")
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--db", default="")
    parser.add_argument("--max-upload-mb", type=int, default=0)
    args = parser.parse_args(argv)
    config = load_config(args)
    asyncio.run(run_bot(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
