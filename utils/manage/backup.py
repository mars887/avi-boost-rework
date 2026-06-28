from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import workdir_layout as layout

MANAGE_BACKUPS_DIR = "manage_backups"
MANAGE_EVENTS_FILE = "manage_events.jsonl"
MAX_BACKUP_FILE_BYTES = 64 * 1024 * 1024  # never copy encode outputs into backups


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Write JSON via temp file + os.replace (atomic as far as Windows allows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=indent) + "\n"
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding=encoding, newline="\n")
    os.replace(tmp, path)


def append_manage_event(workdir: Path, payload: Dict[str, Any]) -> None:
    meta_dir = layout.meta_dir(workdir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": 1, "event": "manage", "timestamp": time.time()}
    record.update(payload)
    with (meta_dir / MANAGE_EVENTS_FILE).open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class BackupEntry:
    path: str
    backup: str
    sha256: str
    size_bytes: int


@dataclass
class ManageTransaction:
    """File-level transaction: backup originals, stage writes, audit on commit.

    Usage::

        tx = ManageTransaction(workdir, operation="reset_chunk")
        tx.backup(chunks_path)
        ... perform writes/deletes through tx ...
        tx.commit(message="chunk 12 reset")
    """

    workdir: Path
    operation: str
    source: Optional[Path] = None
    plan: Optional[Path] = None
    notes: List[str] = field(default_factory=list)
    changed_paths: List[str] = field(default_factory=list)
    _entries: List[BackupEntry] = field(default_factory=list)
    _backed_up: set = field(default_factory=set)
    _backup_dir: Optional[Path] = None
    _committed: bool = False
    _renames_done: List[Tuple[Path, Path]] = field(default_factory=list)
    _created_paths: List[Path] = field(default_factory=list)
    _rolled_back: bool = False

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir)
        self.created_at = datetime.now().astimezone()

    def __enter__(self) -> "ManageTransaction":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None and not self._committed:
            self.rollback()
        return False  # never suppress the exception

    @property
    def backup_dir(self) -> Path:
        if self._backup_dir is None:
            stamp = self.created_at.strftime("%Y%m%d-%H%M%S")
            base = layout.meta_dir(self.workdir) / MANAGE_BACKUPS_DIR
            candidate = base / f"{stamp}-{self.operation}"
            counter = 1
            while candidate.exists():
                counter += 1
                candidate = base / f"{stamp}-{self.operation}-{counter}"
            candidate.mkdir(parents=True, exist_ok=True)
            self._backup_dir = candidate
        return self._backup_dir

    def _relative(self, path: Path) -> str:
        try:
            return str(Path(path).absolute().relative_to(self.workdir.absolute()))
        except ValueError:
            return str(Path(path).absolute())

    def record_change(self, path: Path) -> None:
        rel = self._relative(path)
        if rel not in self.changed_paths:
            self.changed_paths.append(rel)

    def backup(self, path: Path) -> Optional[Path]:
        """Copy a small file into the backup dir; returns the backup path."""
        path = Path(path)
        key = str(path.absolute()).lower()
        if key in self._backed_up:
            return None
        if not path.is_file():
            return None
        try:
            size = path.stat().st_size
        except OSError:
            return None
        if size > MAX_BACKUP_FILE_BYTES:
            self.notes.append(f"skipped backup of large file ({size} bytes): {self._relative(path)}")
            self._backed_up.add(key)
            return None
        rel = self._relative(path)
        target = self.backup_dir / "files" / rel.replace(":", "").replace("\\", os.sep)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        self._backed_up.add(key)
        self._entries.append(
            BackupEntry(
                path=rel,
                backup=str(target.relative_to(self.backup_dir)),
                sha256=_sha256_file(path),
                size_bytes=size,
            )
        )
        return target

    # -- write/delete helpers -------------------------------------------

    def write_json(self, path: Path, payload: Any, *, indent: int = 2) -> None:
        existed = Path(path).exists()
        self.backup(path)
        write_json_atomic(path, payload, indent=indent)
        if not existed:
            self._created_paths.append(Path(path))
        self.record_change(path)

    def write_text(self, path: Path, text: str, *, encoding: str = "utf-8") -> None:
        existed = Path(path).exists()
        self.backup(path)
        write_text_atomic(path, text, encoding=encoding)
        if not existed:
            self._created_paths.append(Path(path))
        self.record_change(path)

    def delete_file(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        self.backup(path)
        try:
            path.unlink()
        except OSError as exc:
            raise RuntimeError(f"failed to delete {path}: {exc}") from exc
        self.record_change(path)
        return True

    def delete_tree(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        if path.is_file():
            return self.delete_file(path)
        shutil.rmtree(path)
        self.record_change(path)
        return True

    def rename(self, src: Path, dst: Path) -> None:
        src, dst = Path(src), Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)
        self._renames_done.append((src, dst))
        self.record_change(src)
        self.record_change(dst)

    def rollback(self) -> None:
        if self._committed or self._rolled_back:
            return
        self._rolled_back = True
        for src, dst in reversed(self._renames_done):
            try:
                if Path(dst).exists():
                    Path(src).parent.mkdir(parents=True, exist_ok=True)
                    os.replace(dst, src)
            except OSError:
                pass
        restored: set = set()
        for entry in self._entries:
            try:
                original = Path(entry.path)
                if not original.is_absolute():
                    original = self.workdir / original
                backup_path = self.backup_dir / entry.backup
                if backup_path.is_file():
                    original.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_path, original)
                    restored.add(str(original.absolute()).lower())
            except OSError:
                pass
        for path in self._created_paths:
            try:
                if str(Path(path).absolute()).lower() in restored:
                    continue  # had an original; already restored above
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass

    # -- finalize ----------------------------------------------------------

    def manifest_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "operation": self.operation,
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "workdir": str(self.workdir),
            "source": str(self.source) if self.source else "",
            "plan": str(self.plan) if self.plan else "",
            "files": [entry.__dict__ for entry in self._entries],
            "notes": list(self.notes),
        }

    def commit(self, *, message: str = "", extra: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """Write the backup manifest and the audit event. Returns manifest path."""
        if self._committed:
            return None
        self._committed = True
        manifest_path: Optional[Path] = None
        if self._entries or self._backup_dir is not None:
            manifest_path = self.backup_dir / "manifest.json"
            write_json_atomic(manifest_path, self.manifest_payload())
        payload: Dict[str, Any] = {
            "operation": self.operation,
            "workdir": str(self.workdir),
            "plan": str(self.plan) if self.plan else "",
            "changed_paths": list(self.changed_paths),
            "backup_manifest": self._relative(manifest_path) if manifest_path else "",
            "message": message,
        }
        if extra:
            payload.update(extra)
        append_manage_event(self.workdir, payload)
        return manifest_path


def list_backups(workdir: Path) -> List[Path]:
    base = layout.meta_dir(workdir) / MANAGE_BACKUPS_DIR
    if not base.is_dir():
        return []
    return sorted((entry for entry in base.iterdir() if entry.is_dir()), key=lambda p: p.name, reverse=True)


__all__ = [
    "BackupEntry",
    "MANAGE_BACKUPS_DIR",
    "MANAGE_EVENTS_FILE",
    "ManageTransaction",
    "append_manage_event",
    "list_backups",
    "write_json_atomic",
    "write_text_atomic",
]
