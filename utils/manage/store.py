from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

STORE_FILE_NAME = "manage_store.sqlite"
SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 5000

COLLECTION_STAGE_STATUS = "stage_status"
COLLECTION_STAGE_EVENTS = "stage_events"
COLLECTION_PROGRESS = "progress"
COLLECTION_MANIFESTS = "manifests"
COLLECTION_CONFIG_FINGERPRINTS = "config_fingerprints"
COLLECTION_PASS_CACHE = "pass_cache"
COLLECTION_UI_STATE = "ui_state"
COLLECTION_MANAGE_EVENTS = "manage_events"


def store_path_for_workdir(workdir: Path) -> Path:
    return Path(workdir) / "00_meta" / STORE_FILE_NAME


class WorkdirStore:
    """Document/event store backed by one SQLite file per workdir.

    The store is a derived cache: source of truth stays in the workdir files
    (markers, JSONL, manifests). Deleting the database is always safe — it is
    rebuilt by re-importing those files.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._tx_depth = 0

    # -- lifecycle -----------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is not None:
                return self._conn
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
            self._conn = conn
            self._ensure_schema(conn)
            return conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    def __enter__(self) -> "WorkdirStore":
        self.connect()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS docs (
                    collection TEXT NOT NULL,
                    key TEXT NOT NULL,
                    payload TEXT NOT NULL CHECK (json_valid(payload)),
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_docs_collection_key ON docs(collection, key)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    payload TEXT NOT NULL CHECK (json_valid(payload))
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_collection_ts ON events(collection, timestamp)")
            row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )

    def _execute_write(self, sql: str, params: tuple) -> sqlite3.Cursor:
        """Run a write statement; commit immediately unless inside transaction()."""
        conn = self.connect()
        with self._lock:
            if self._tx_depth > 0:
                return conn.execute(sql, params)
            with conn:
                return conn.execute(sql, params)

    # -- documents -----------------------------------------------------

    def put_doc(self, collection: str, key: str, payload: Dict[str, Any]) -> None:
        text = json.dumps(payload, ensure_ascii=False)
        self._execute_write(
            """
            INSERT INTO docs(collection, key, payload, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(collection, key) DO UPDATE SET
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (str(collection), str(key), text, time.time()),
        )

    def get_doc(self, collection: str, key: str) -> Optional[Dict[str, Any]]:
        conn = self.connect()
        row = conn.execute(
            "SELECT payload FROM docs WHERE collection=? AND key=?",
            (str(collection), str(key)),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload"])
        return payload if isinstance(payload, dict) else None

    def delete_doc(self, collection: str, key: str) -> None:
        self._execute_write("DELETE FROM docs WHERE collection=? AND key=?", (str(collection), str(key)))

    def list_docs(self, collection: str) -> Dict[str, Dict[str, Any]]:
        conn = self.connect()
        out: Dict[str, Dict[str, Any]] = {}
        for row in conn.execute(
            "SELECT key, payload FROM docs WHERE collection=? ORDER BY key",
            (str(collection),),
        ):
            payload = json.loads(row["payload"])
            if isinstance(payload, dict):
                out[str(row["key"])] = payload
        return out

    # -- events ----------------------------------------------------------

    def append_event(self, collection: str, payload: Dict[str, Any]) -> int:
        timestamp = float(payload.get("timestamp") or time.time())
        cursor = self._execute_write(
            "INSERT INTO events(collection, timestamp, payload) VALUES(?, ?, ?)",
            (str(collection), timestamp, json.dumps(payload, ensure_ascii=False)),
        )
        return int(cursor.lastrowid or 0)

    def latest_event(self, collection: str, **filters: Any) -> Optional[Dict[str, Any]]:
        for payload in self.iter_events(collection, descending=True):
            if all(payload.get(key) == value for key, value in filters.items()):
                return payload
        return None

    def iter_events(
        self,
        collection: str,
        *,
        since: float = 0.0,
        descending: bool = False,
        limit: int = 0,
    ) -> Iterator[Dict[str, Any]]:
        conn = self.connect()
        sql = "SELECT payload FROM events WHERE collection=? AND timestamp>=? ORDER BY id"
        if descending:
            sql += " DESC"
        if limit > 0:
            sql += f" LIMIT {int(limit)}"
        for row in conn.execute(sql, (str(collection), float(since))):
            payload = json.loads(row["payload"])
            if isinstance(payload, dict):
                yield payload

    def events(self, collection: str, **kwargs: Any) -> List[Dict[str, Any]]:
        return list(self.iter_events(collection, **kwargs))

    # -- transactions ----------------------------------------------------

    @contextmanager
    def transaction(self) -> Iterator[None]:
        conn = self.connect()
        with self._lock:
            self._tx_depth += 1
            outer = self._tx_depth == 1
            try:
                if outer:
                    conn.execute("BEGIN IMMEDIATE")
                yield
                if outer:
                    conn.commit()
            except Exception:
                if outer:
                    conn.rollback()
                raise
            finally:
                self._tx_depth -= 1

    # -- import helpers ----------------------------------------------------

    def _get_meta(self, key: str) -> str:
        conn = self.connect()
        row = conn.execute("SELECT value FROM meta WHERE key=?", (str(key),)).fetchone()
        return str(row["value"]) if row is not None else ""

    def _set_meta(self, key: str, value: str) -> None:
        self._execute_write(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(key), str(value)),
        )

    def import_jsonl_tail(self, path: Path, *, collection: str) -> int:
        """Read-through import of an append-only JSONL file.

        Stores the byte offset of the last import per file and only reads the
        tail on subsequent calls. Returns the number of imported events.
        """
        path = Path(path)
        offset_key = f"jsonl_offset:{str(path).lower()}"
        try:
            size = path.stat().st_size
        except OSError:
            return 0
        try:
            offset = int(self._get_meta(offset_key) or 0)
        except ValueError:
            offset = 0
        if offset > size:
            offset = 0  # file was truncated/recreated; re-import from scratch
        if offset == size:
            return 0
        imported = 0
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(offset)
            with self.transaction():
                for line in fh:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        payload = json.loads(text)
                    except Exception:
                        continue
                    if isinstance(payload, dict):
                        self.append_event(collection, payload)
                        imported += 1
                new_offset = fh.tell()
                self._set_meta(offset_key, str(new_offset))
        return imported

    def import_json_doc(self, path: Path, *, collection: str, key: str) -> bool:
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        self.put_doc(collection, key, payload)
        return True


def open_store(workdir: Path) -> WorkdirStore:
    return WorkdirStore(store_path_for_workdir(workdir))


def import_workdir_files(store: WorkdirStore, workdir: Path) -> Dict[str, int]:
    """Read-through import of runner files into the store cache."""
    workdir = Path(workdir)
    meta_dir = workdir / "00_meta"
    stats: Dict[str, int] = {}

    stats["runner_events"] = store.import_jsonl_tail(
        meta_dir / "runner_events.jsonl", collection=COLLECTION_STAGE_EVENTS
    )
    child_imported = 0
    try:
        child_files = sorted(meta_dir.glob("runner_child_*.jsonl"), key=lambda p: p.stat().st_mtime)
    except OSError:
        child_files = []
    for child in child_files:
        child_imported += store.import_jsonl_tail(child, collection=COLLECTION_PROGRESS)
    stats["child_events"] = child_imported

    if store.import_json_doc(meta_dir / "runner_state.json", collection=COLLECTION_PROGRESS, key="runner_state"):
        stats["runner_state"] = 1
    for name in ("demux_manifest", "audio_manifest", "mux_manifest", "source_info"):
        if store.import_json_doc(meta_dir / f"{name}.json", collection=COLLECTION_MANIFESTS, key=name):
            stats[name] = 1
    return stats


__all__ = [
    "BUSY_TIMEOUT_MS",
    "COLLECTION_CONFIG_FINGERPRINTS",
    "COLLECTION_MANAGE_EVENTS",
    "COLLECTION_MANIFESTS",
    "COLLECTION_PASS_CACHE",
    "COLLECTION_PROGRESS",
    "COLLECTION_STAGE_EVENTS",
    "COLLECTION_STAGE_STATUS",
    "COLLECTION_UI_STATE",
    "SCHEMA_VERSION",
    "STORE_FILE_NAME",
    "WorkdirStore",
    "import_workdir_files",
    "open_store",
    "store_path_for_workdir",
]
