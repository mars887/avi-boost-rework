from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from utils.discord_config import discord_config_int, discord_config_value, read_discord_config

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8794
MAX_UPLOAD_MB_DEFAULT = 25
LOADER_FILE_TTL_SECONDS = 600.0
ADMINS_ONLY = False
INACTIVE_MESSAGE_TEXT = "Runner is not connected for this folder. This message will be removed when a runner registers again."
SESSION_STALE_SECONDS = 45.0
DISCORD_API_TIMEOUT_SECONDS = 25.0
CHANNEL_RENAME_TIMEOUT_SECONDS = 5.0
CHANNEL_RENAME_MIN_INTERVAL_SECONDS = 10.0 * 60.0
CHANNEL_RENAME_FAILURE_BACKOFF_SECONDS = CHANNEL_RENAME_MIN_INTERVAL_SECONDS + 60.0
DASHBOARD_LOCK_TIMEOUT_SECONDS = 20.0
DASHBOARD_UPDATE_TIMEOUT_SECONDS = 90.0

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
    debug: bool = False


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
        debug=bool(args.debug or discord_config_int("PBBATCH_DISCORD_DEBUG", 0, file_values=file_values)),
    )
