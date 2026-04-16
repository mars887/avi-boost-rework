from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.discord_config import discord_config_int, discord_config_value, read_discord_config

try:
    import discord  # type: ignore[import-not-found]
    from aiohttp import web  # type: ignore[import-not-found]
    from discord import app_commands  # type: ignore[import-not-found]
except Exception:
    discord = None
    web = None
    app_commands = None


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8794
MAX_EMBED_FIELD = 1024
MAX_UPLOAD_MB_DEFAULT = 25

COMMANDS = {
    "pause_after_current": "Pause",
    "resume": "Resume",
    "retry_failed": "Retry failed",
    "rerun_current": "Rerun current",
    "exit_when_idle": "Exit when idle",
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


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        if not str(self.path):
            raise RuntimeError("Discord state database path is empty.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        except sqlite3.OperationalError as exc:
            raise RuntimeError(f"Unable to open Discord state database: {self.path} ({exc})") from exc
        self.db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript(
            """
            create table if not exists folders (
                source_dir text primary key,
                channel_id integer not null,
                channel_name text not null,
                updated_at real not null
            );
            create table if not exists sessions (
                session_id text primary key,
                source_dir text not null,
                channel_id integer not null,
                snapshot_json text not null,
                history_message_id integer,
                current_message_id integer,
                queue_message_id integer,
                updated_at real not null
            );
            create table if not exists commands (
                command_id text primary key,
                session_id text not null,
                name text not null,
                status text not null,
                message text not null,
                created_at real not null,
                acked_at real
            );
            create table if not exists events (
                id integer primary key autoincrement,
                session_id text not null,
                payload_json text not null,
                created_at real not null
            );
            """
        )
        self.db.commit()

    def get_channel_id(self, source_dir: str) -> int:
        row = self.db.execute("select channel_id from folders where source_dir = ?", (source_dir,)).fetchone()
        return int(row["channel_id"]) if row else 0

    def set_channel(self, source_dir: str, channel_id: int, channel_name: str) -> None:
        self.db.execute(
            """
            insert into folders(source_dir, channel_id, channel_name, updated_at)
            values(?, ?, ?, ?)
            on conflict(source_dir) do update set
                channel_id=excluded.channel_id,
                channel_name=excluded.channel_name,
                updated_at=excluded.updated_at
            """,
            (source_dir, int(channel_id), channel_name, time.time()),
        )
        self.db.commit()

    def upsert_session(self, snapshot: Dict[str, Any], channel_id: int) -> None:
        session_id = str(snapshot.get("session_id") or "")
        source_dir = str(snapshot.get("source_dir") or "")
        row = self.db.execute("select * from sessions where session_id = ?", (session_id,)).fetchone()
        if row:
            self.db.execute(
                """
                update sessions set source_dir=?, channel_id=?, snapshot_json=?, updated_at=?
                where session_id=?
                """,
                (source_dir, int(channel_id), json.dumps(snapshot, ensure_ascii=False), time.time(), session_id),
            )
        else:
            self.db.execute(
                """
                insert into sessions(session_id, source_dir, channel_id, snapshot_json, updated_at)
                values(?, ?, ?, ?, ?)
                """,
                (session_id, source_dir, int(channel_id), json.dumps(snapshot, ensure_ascii=False), time.time()),
            )
        self.db.commit()

    def set_session_messages(self, session_id: str, *, history: int, current: int, queue_message: int) -> None:
        self.db.execute(
            """
            update sessions
            set history_message_id=?, current_message_id=?, queue_message_id=?, updated_at=?
            where session_id=?
            """,
            (int(history), int(current), int(queue_message), time.time(), session_id),
        )
        self.db.commit()

    def get_session(self, session_id: str) -> Optional[sqlite3.Row]:
        return self.db.execute("select * from sessions where session_id = ?", (session_id,)).fetchone()

    def latest_session_for_channel(self, channel_id: int) -> Optional[sqlite3.Row]:
        return self.db.execute(
            "select * from sessions where channel_id = ? order by updated_at desc limit 1",
            (int(channel_id),),
        ).fetchone()

    def latest_session_for_source_dir(self, source_dir: str) -> Optional[sqlite3.Row]:
        return self.db.execute(
            "select * from sessions where source_dir = ? order by updated_at desc limit 1",
            (source_dir,),
        ).fetchone()

    def add_event(self, session_id: str, payload: Dict[str, Any]) -> None:
        self.db.execute(
            "insert into events(session_id, payload_json, created_at) values(?, ?, ?)",
            (session_id, json.dumps(payload, ensure_ascii=False), time.time()),
        )
        self.db.commit()

    def enqueue_command(self, session_id: str, name: str) -> str:
        command_id = uuid.uuid4().hex
        self.db.execute(
            """
            insert into commands(command_id, session_id, name, status, message, created_at)
            values(?, ?, ?, 'pending', '', ?)
            """,
            (command_id, session_id, name, time.time()),
        )
        self.db.commit()
        return command_id

    def pending_commands(self, session_id: str) -> List[Dict[str, str]]:
        rows = self.db.execute(
            """
            select command_id, name from commands
            where session_id = ? and status = 'pending'
            order by created_at asc
            """,
            (session_id,),
        ).fetchall()
        self.db.executemany(
            "update commands set status='sent' where command_id=?",
            [(str(row["command_id"]),) for row in rows],
        )
        self.db.commit()
        return [{"command_id": str(row["command_id"]), "name": str(row["name"])} for row in rows]

    def ack_command(self, command_id: str, status: str, message: str) -> None:
        self.db.execute(
            "update commands set status=?, message=?, acked_at=? where command_id=?",
            (status, message, time.time(), command_id),
        )
        self.db.commit()


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
    )


def channel_name_for_source(source_dir: str) -> str:
    base = Path(source_dir).name.strip().lower() or "folder"
    base = re.sub(r"[^a-z0-9_-]+", "-", base, flags=re.IGNORECASE).strip("-").lower()
    base = base[:70] or "folder"
    suffix = hashlib.sha1(source_dir.lower().encode("utf-8", errors="ignore")).hexdigest()[:6]
    return f"{base}-{suffix}"


def truncate(text: str, limit: int = MAX_EMBED_FIELD) -> str:
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


def status_icon(status: str) -> str:
    normalized = str(status or "").lower()
    if normalized == "completed":
        return "OK"
    if normalized == "skipped":
        return "SKP"
    if normalized == "started":
        return "RUN"
    if normalized == "failed":
        return "ERR"
    return "..."


def is_cached_stage(stage: Dict[str, Any]) -> bool:
    message = str(stage.get("message") or "").strip().lower()
    return message in ("cached", "resume", "using_existing_base_scenes")


def progress_bar(progress: Any, width: int = 24) -> str:
    try:
        value = float(progress)
    except Exception:
        value = -1.0
    if value < 0:
        return "[" + ("-" * width) + "]"
    value = max(0.0, min(100.0, value))
    filled = int(round((value / 100.0) * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {value:.1f}%"


def render_stage_table(stages: List[Dict[str, Any]]) -> str:
    if not stages:
        return ""
    name_width = max(len(str(stage.get("name") or "")) for stage in stages)
    name_width = max(10, min(name_width, 34))
    lines: List[str] = []
    for stage in stages:
        name = truncate(str(stage.get("name") or ""), name_width)
        status = str(stage.get("status") or "pending").lower()
        cached = is_cached_stage(stage)
        icon = "OK" if cached else status_icon(status)
        detail = ""
        if cached:
            detail = "cached"
        elif status == "started":
            detail = f"{fmt_seconds(stage.get('elapsed_seconds'))} {progress_bar(stage.get('progress'))}"
        elif status == "completed":
            detail = fmt_seconds(stage.get("elapsed_seconds"))
        elif status == "skipped":
            detail = "skipped"
        elif status == "failed":
            detail = truncate(str(stage.get("message") or "failed"), 44)
        lines.append(f"{icon:<3} {name:<{name_width}} {detail}".rstrip())
    return "```text\n" + "\n".join(lines) + "\n```"


def snapshot_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    try:
        return json.loads(str(row["snapshot_json"] or "{}"))
    except Exception:
        return {}


def render_history_embed(snapshot: Dict[str, Any]) -> Any:
    embed = discord.Embed(title="History", color=0x2B2D31)
    completed = list(snapshot.get("completed") or [])[-12:]
    failed = list(snapshot.get("failed") or [])[-8:]
    lines: List[str] = []
    for index, item in enumerate(completed, start=1):
        lines.append(
            f"✅ {index} | {Path(str(item.get('plan') or '')).name} | {fmt_seconds(item.get('elapsed_seconds'))} | "
            f"{fmt_size(item.get('source_size'))} -> {fmt_size(item.get('output_size'))}"
        )
    for index, item in enumerate(failed, start=1):
        lines.append(
            f"❌ {index} | {Path(str(item.get('plan') or '')).name} | {item.get('stage') or '-'} | "
            f"{truncate(str(item.get('message') or ''), 160)}"
        )
    embed.description = truncate("\n".join(lines) if lines else "No finished plans yet.", 4000)
    return embed


def render_current_embed(snapshot: Dict[str, Any]) -> Any:
    active = list(snapshot.get("active") or [])
    if not active:
        state = str(snapshot.get("state") or "idle")
        embed = discord.Embed(title="Current process", description=f"State: `{state}`", color=0x5865F2)
        return embed

    title = "Active plans" if len(active) > 1 else str(active[0].get("name") or "Current plan")
    embed = discord.Embed(title=title, color=0x5865F2)
    for index, plan in enumerate(active[:6]):
        header = f"mode `{plan.get('mode')}` | elapsed `{fmt_seconds(plan.get('elapsed_seconds'))}` | source `{fmt_size(plan.get('source_size'))}`"
        table = render_stage_table(list(plan.get("stages") or []))
        value = truncate(f"{header}\n{table}", 4000)
        if len(active) == 1:
            embed.description = value
        else:
            embed.add_field(name=str(plan.get("name") or f"plan {index + 1}"), value=value, inline=False)
    return embed


def render_queue_embed(snapshot: Dict[str, Any]) -> Any:
    embed = discord.Embed(title="Queue", color=0x2B2D31)
    queue_items = list(snapshot.get("queue") or [])
    if not queue_items:
        embed.description = "Queue is empty."
        return embed
    lines = []
    for index, item in enumerate(queue_items[:30], start=1):
        lines.append(f"... {index}. `{Path(str(item.get('plan') or '')).name}` | `{item.get('mode')}` | {fmt_size(item.get('source_size'))}")
    if len(queue_items) > 30:
        lines.append(f"... and {len(queue_items) - 30} more")
    embed.description = truncate("\n".join(lines), 4000)
    return embed


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


async def run_bot(config: BotConfig) -> None:
    if discord is None or web is None or app_commands is None:
        raise RuntimeError("discord.py and aiohttp are required. Install requirements-discord.txt first.")
    if not config.token or not config.guild_id or not config.category_id:
        raise RuntimeError("PBBATCH_DISCORD_TOKEN, PBBATCH_DISCORD_GUILD_ID and PBBATCH_DISCORD_CATEGORY_ID are required.")

    store = StateStore(config.db_path)
    intents = discord.Intents.default()
    client = discord.Client(intents=intents)
    tree = app_commands.CommandTree(client)
    guild_object = discord.Object(id=config.guild_id)

    def is_authorized(interaction: Any) -> bool:
        if config.admin_role_id == 0 and config.operator_role_id == 0:
            return True
        roles = getattr(getattr(interaction, "user", None), "roles", []) or []
        role_ids = {int(getattr(role, "id", 0)) for role in roles}
        return bool(role_ids.intersection({config.admin_role_id, config.operator_role_id}))

    async def ensure_channel(source_dir: str) -> Any:
        await client.wait_until_ready()
        existing_id = store.get_channel_id(source_dir)
        if existing_id:
            channel = client.get_channel(existing_id)
            if channel is not None:
                return channel
        guild = client.get_guild(config.guild_id)
        if guild is None:
            raise RuntimeError(f"guild not found: {config.guild_id}")
        category = guild.get_channel(config.category_id)
        if category is None:
            raise RuntimeError(f"category not found: {config.category_id}")
        channel_name = channel_name_for_source(source_dir)
        for channel in category.channels:
            if getattr(channel, "name", "") == channel_name:
                store.set_channel(source_dir, int(channel.id), channel_name)
                return channel
        channel = await guild.create_text_channel(
            name=channel_name,
            category=category,
            topic=f"PBBatch source: {source_dir}",
            reason="PBBatch runner session",
        )
        store.set_channel(source_dir, int(channel.id), channel_name)
        return channel

    async def update_dashboard(snapshot: Dict[str, Any]) -> None:
        session_id = str(snapshot.get("session_id") or "")
        source_dir = str(snapshot.get("source_dir") or "")
        if not session_id or not source_dir:
            return
        channel = await ensure_channel(source_dir)
        store.upsert_session(snapshot, int(channel.id))
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

        history_msg = await upsert_message(history_id, embed=render_history_embed(snapshot))
        current_msg = await upsert_message(current_id, embed=render_current_embed(snapshot), view=build_control_view(session_id))
        queue_msg = await upsert_message(queue_id, embed=render_queue_embed(snapshot))
        store.set_session_messages(
            session_id,
            history=int(history_msg.id),
            current=int(current_msg.id),
            queue_message=int(queue_msg.id),
        )

    async def enqueue_from_interaction(interaction: Any, session_id: str, command: str) -> None:
        if not is_authorized(interaction):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        row = store.get_session(session_id)
        if row is None:
            await interaction.response.send_message("Session not found.", ephemeral=True)
            return
        store.enqueue_command(session_id, command)
        await interaction.response.send_message(f"Command queued: `{command}`", ephemeral=True)

    class CommandButton(discord.ui.Button):
        def __init__(self, *, session_id: str, command: str, label: str, style: Any) -> None:
            super().__init__(label=label, style=style, custom_id=f"pbbatch:cmd:{session_id}:{command}")
            self.session_id = session_id
            self.command = command

        async def callback(self, interaction: Any) -> None:
            await enqueue_from_interaction(interaction, self.session_id, self.command)

    def build_control_view(session_id: str) -> Any:
        view = discord.ui.View(timeout=None)
        for name, label in COMMANDS.items():
            style = discord.ButtonStyle.secondary
            if name == "resume":
                style = discord.ButtonStyle.success
            if name == "exit_when_idle":
                style = discord.ButtonStyle.danger
            view.add_item(CommandButton(session_id=session_id, command=name, label=label, style=style))
        return view

    @client.event
    async def on_ready() -> None:
        await tree.sync(guild=guild_object)
        print(f"[discord-bot] logged in as {client.user}", flush=True)

    @tree.command(name="panel", description="Refresh the PBBatch dashboard in this channel.", guild=guild_object)
    async def panel(interaction: Any) -> None:
        row = store.latest_session_for_channel(int(interaction.channel_id))
        if row is None:
            await interaction.response.send_message("No PBBatch session is known for this channel.", ephemeral=True)
            return
        await update_dashboard(snapshot_from_row(row))
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
            app_commands.Choice(name="Zone edit command", value="zone"),
            app_commands.Choice(name="Crop/resize command", value="crop"),
        ]
    )
    async def pbbatch_file(interaction: Any, kind: app_commands.Choice[str]) -> None:
        row = store.latest_session_for_channel(int(interaction.channel_id))
        if row is None:
            await interaction.response.send_message("No session in this channel.", ephemeral=True)
            return
        snapshot = snapshot_from_row(row)
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
            "zone": workdir / "zone_edit_command.txt",
            "crop": workdir / "crop_resize_command.txt",
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
        snapshot = snapshot_from_row(row)
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
        await update_dashboard(snapshot)
        return web.json_response({"status": "ok"})

    async def http_snapshot(request: Any) -> Any:
        payload = await request.json()
        snapshot = dict(payload.get("snapshot") or {})
        await update_dashboard(snapshot)
        return web.json_response({"status": "ok"})

    async def http_event(request: Any) -> Any:
        session_id = str(request.match_info["session_id"])
        payload = await request.json()
        event = dict(payload.get("event") or {})
        snapshot = dict(payload.get("snapshot") or {})
        store.add_event(session_id, event)
        await update_dashboard(snapshot)
        if event.get("status") == "failed":
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
            await update_dashboard(snapshot)
        return web.json_response({"status": "ok"})

    app = web.Application()
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
