from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DASHBOARD_MAJOR_INTERVAL_SECONDS = 15.0
DEFAULT_DASHBOARD_PROGRESS_INTERVAL_SECONDS = 60.0
DEFAULT_RUNNER_STARTUP_DELAY_SECONDS = 5.0
COMMAND_LEASE_SECONDS_DEFAULT = 30.0


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
                alias text not null default '',
                updates_paused integer not null default 0,
                major_interval_seconds integer not null default 15,
                progress_interval_seconds integer not null default 60,
                startup_delay_seconds integer not null default 5,
                inactive_message_id integer not null default 0,
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
                sent_at real,
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
        self._ensure_column("folders", "alias", "text not null default ''")
        self._ensure_column("folders", "updates_paused", "integer not null default 0")
        self._ensure_column("folders", "major_interval_seconds", "integer not null default 15")
        self._ensure_column("folders", "progress_interval_seconds", "integer not null default 60")
        self._ensure_column("folders", "startup_delay_seconds", "integer not null default 5")
        self._ensure_column("folders", "inactive_message_id", "integer not null default 0")
        self._ensure_column("commands", "sent_at", "real")
        self.db.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        rows = self.db.execute(f"pragma table_info({table})").fetchall()
        if any(str(row["name"]) == column for row in rows):
            return
        self.db.execute(f"alter table {table} add column {column} {definition}")

    def get_channel_id(self, source_dir: str) -> int:
        row = self.db.execute("select channel_id from folders where source_dir = ?", (source_dir,)).fetchone()
        return int(row["channel_id"]) if row else 0

    def get_folder(self, source_dir: str) -> Optional[sqlite3.Row]:
        return self.db.execute("select * from folders where source_dir = ?", (source_dir,)).fetchone()

    def folder_settings(self, source_dir: str) -> Dict[str, Any]:
        row = self.get_folder(source_dir)
        if row is None:
            return {
                "alias": "",
                "updates_paused": False,
                "major_interval_seconds": int(DEFAULT_DASHBOARD_MAJOR_INTERVAL_SECONDS),
                "progress_interval_seconds": int(DEFAULT_DASHBOARD_PROGRESS_INTERVAL_SECONDS),
                "startup_delay_seconds": int(DEFAULT_RUNNER_STARTUP_DELAY_SECONDS),
            }
        def int_setting(name: str, default: float) -> int:
            value = row[name]
            return int(default) if value is None else int(value)
        return {
            "alias": str(row["alias"] or ""),
            "updates_paused": bool(int(row["updates_paused"] or 0)),
            "major_interval_seconds": int_setting("major_interval_seconds", DEFAULT_DASHBOARD_MAJOR_INTERVAL_SECONDS),
            "progress_interval_seconds": int_setting("progress_interval_seconds", DEFAULT_DASHBOARD_PROGRESS_INTERVAL_SECONDS),
            "startup_delay_seconds": int_setting("startup_delay_seconds", DEFAULT_RUNNER_STARTUP_DELAY_SECONDS),
        }

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

    def set_folder_alias(self, source_dir: str, alias: str) -> None:
        cur = self.db.execute(
            """
            update folders
            set alias=?, updated_at=?
            where source_dir=?
            """,
            (alias, time.time(), source_dir),
        )
        if cur.rowcount == 0:
            self.db.execute(
                """
                insert into folders(source_dir, channel_id, channel_name, alias, updated_at)
                values(?, 0, '', ?, ?)
                """,
                (source_dir, alias, time.time()),
            )
        self.db.commit()

    def set_updates_paused(self, source_dir: str, paused: bool) -> None:
        self.db.execute(
            """
            update folders
            set updates_paused=?, updated_at=?
            where source_dir=?
            """,
            (1 if paused else 0, time.time(), source_dir),
        )
        self.db.commit()

    def set_inactive_message(self, source_dir: str, message_id: int) -> None:
        self.db.execute(
            """
            update folders
            set inactive_message_id=?, updated_at=?
            where source_dir=?
            """,
            (int(message_id), time.time(), source_dir),
        )
        self.db.commit()

    def set_update_settings(self, source_dir: str, *, major: int, progress: int, startup_delay: int) -> None:
        self.db.execute(
            """
            update folders
            set major_interval_seconds=?, progress_interval_seconds=?, startup_delay_seconds=?, updated_at=?
            where source_dir=?
            """,
            (int(major), int(progress), int(startup_delay), time.time(), source_dir),
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
            previous = self.latest_session_for_source_dir(source_dir)
            self.db.execute(
                """
                insert into sessions(
                    session_id,
                    source_dir,
                    channel_id,
                    snapshot_json,
                    history_message_id,
                    current_message_id,
                    queue_message_id,
                    updated_at
                )
                values(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    source_dir,
                    int(channel_id),
                    json.dumps(snapshot, ensure_ascii=False),
                    int(previous["history_message_id"] or 0) if previous else 0,
                    int(previous["current_message_id"] or 0) if previous else 0,
                    int(previous["queue_message_id"] or 0) if previous else 0,
                    time.time(),
                ),
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

    def pending_commands(self, session_id: str, *, lease_seconds: float = COMMAND_LEASE_SECONDS_DEFAULT) -> List[Dict[str, str]]:
        now = time.time()
        lease_before = now - max(1.0, float(lease_seconds))
        rows = self.db.execute(
            """
            select command_id, name from commands
            where session_id = ?
              and (
                status = 'pending'
                or (status = 'sent' and coalesce(sent_at, 0) <= ?)
              )
            order by created_at asc
            """,
            (session_id, lease_before),
        ).fetchall()
        self.db.executemany(
            "update commands set status='sent', sent_at=? where command_id=?",
            [(now, str(row["command_id"])) for row in rows],
        )
        self.db.commit()
        return [{"command_id": str(row["command_id"]), "name": str(row["name"])} for row in rows]

    def ack_command(self, command_id: str, status: str, message: str) -> None:
        self.db.execute(
            "update commands set status=?, message=?, acked_at=? where command_id=?",
            (status, message, time.time(), command_id),
        )
        self.db.commit()

    def folder_rows(self) -> List[sqlite3.Row]:
        return list(self.db.execute("select * from folders order by updated_at desc").fetchall())
