from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.plan import BatchPlan, FilePlan, dump_plan, load_plan
from utils.zoned_commands import (
    parse_zoned_payload,
    read_text_guess,
    split_zoned_command_line,
    zoned_command_path,
)

from utils.manage.backup import ManageTransaction, write_json_atomic, write_text_atomic
from utils.manage.context import WorkdirContext, load_json_file
from utils.manage.scenes import ValidationIssue

MANAGE_STATE_FILE = "manage_state.json"
STAGE_REGISTRY_VERSION = 1  # bump when utils.manage.reset registry semantics change


@dataclass(frozen=True)
class ConfigFileRef:
    path: Path
    kind: str  # plan | batch_plan | zone | manifest | scenes_bank | other
    description: str
    exists: bool
    editable: bool


@dataclass
class ConfigFingerprint:
    plan_sha256: str = ""
    zone_sha256: str = ""
    source_path: str = ""
    source_size: int = 0
    source_mtime_ns: int = 0
    stage_registry_version: int = STAGE_REGISTRY_VERSION

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ConfigFingerprint":
        out = cls()
        for key in out.__dict__:
            if key in payload:
                setattr(out, key, payload[key])
        return out

    def differs_from(self, other: "ConfigFingerprint") -> List[str]:
        changed: List[str] = []
        if self.plan_sha256 != other.plan_sha256:
            changed.append("plan")
        if self.zone_sha256 != other.zone_sha256:
            changed.append("zone_file")
        if (
            self.source_path.lower() != other.source_path.lower()
            or self.source_size != other.source_size
            or self.source_mtime_ns != other.source_mtime_ns
        ):
            changed.append("source")
        if self.stage_registry_version != other.stage_registry_version:
            changed.append("stage_registry")
        return changed


def _sha256_file(path: Optional[Path]) -> str:
    if path is None:
        return ""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def list_config_files(ctx: WorkdirContext) -> List[ConfigFileRef]:
    out: List[ConfigFileRef] = []

    def add(path: Optional[Path], kind: str, description: str, editable: bool) -> None:
        if path is None:
            return
        path = Path(path)
        out.append(
            ConfigFileRef(path=path, kind=kind, description=description, exists=path.is_file(), editable=editable)
        )

    add(ctx.plan_path, "plan", "file plan (.plan)", True)
    add(ctx.batch_plan_path, "batch_plan", "batch plan", False)
    add(ctx.zone_file or zoned_command_path(ctx.workdir), "zone", "zoned_command.txt", True)
    add(ctx.workdir / "zone_edit_command.txt", "zone", "legacy zone command file", False)
    add(ctx.workdir / "crop_resize_command.txt", "zone", "legacy crop/resize command file", False)
    meta = ctx.meta_dir
    add(meta / "source_info.json", "manifest", "source signature snapshot", False)
    add(meta / "demux_manifest.json", "manifest", "demux manifest", False)
    add(meta / "audio_manifest.json", "manifest", "audio manifest", False)
    add(meta / "mux_manifest.json", "manifest", "mux manifest", False)
    return out


def load_plan_config(path: Path) -> FilePlan | BatchPlan:
    return load_plan(Path(path))


def save_plan_config(
    plan: FilePlan | BatchPlan,
    path: Path,
    *,
    tx: Optional[ManageTransaction] = None,
) -> None:
    text = dump_plan(plan)
    # parse-before-save: dump and re-load to guarantee the file stays readable
    reparsed = _reparse_plan_text(text, Path(path))
    if type(reparsed) is not type(plan):
        raise RuntimeError("plan round-trip produced a different plan type; aborting save")
    if tx is not None:
        tx.write_text(Path(path), text)
    else:
        write_text_atomic(Path(path), text)


def _reparse_plan_text(text: str, path: Path) -> FilePlan | BatchPlan:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / path.name
        probe.write_text(text, encoding="utf-8", newline="\n")
        return load_plan(probe)


def load_zone_text(path: Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    return read_text_guess(path)


def validate_zone_text(ctx: WorkdirContext, text: str) -> List[ValidationIssue]:
    """Line-by-line parse validation via utils.zoned_commands."""
    issues: List[ValidationIssue] = []
    for line_no, raw in enumerate(str(text or "").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        try:
            split = split_zoned_command_line(line)
            parse_zoned_payload(split.payload_tokens)
        except ValueError as exc:
            issues.append(ValidationIssue("error", f"line {line_no}: {exc}"))
    return issues


def save_zone_text(
    ctx: WorkdirContext,
    path: Path,
    text: str,
    *,
    tx: Optional[ManageTransaction] = None,
) -> List[ValidationIssue]:
    issues = validate_zone_text(ctx, text)
    if any(issue.is_error for issue in issues):
        raise RuntimeError(
            "zone text has parse errors: " + "; ".join(issue.message for issue in issues if issue.is_error)
        )
    normalized = str(text or "").replace("\r\n", "\n").rstrip() + "\n"
    if tx is not None:
        tx.write_text(Path(path), normalized)
    else:
        write_text_atomic(Path(path), normalized)
    return issues


def compute_config_fingerprint(ctx: WorkdirContext) -> ConfigFingerprint:
    fingerprint = ConfigFingerprint(
        plan_sha256=_sha256_file(ctx.plan_path),
        zone_sha256=_sha256_file(ctx.zone_file),
    )
    if ctx.source is not None:
        fingerprint.source_path = str(ctx.source)
        try:
            stat = ctx.source.stat()
            fingerprint.source_size = int(stat.st_size)
            fingerprint.source_mtime_ns = int(getattr(stat, "st_mtime_ns", 0))
        except OSError:
            pass
    return fingerprint


def manage_state_path(ctx: WorkdirContext) -> Path:
    return ctx.meta_dir / MANAGE_STATE_FILE


def load_manage_state(ctx: WorkdirContext) -> Dict[str, Any]:
    payload = load_json_file(manage_state_path(ctx))
    return payload if isinstance(payload, dict) else {}


def save_manage_state(ctx: WorkdirContext, state: Dict[str, Any]) -> None:
    payload = {"schema_version": 1, "workdir": str(ctx.workdir)}
    payload.update(state)
    payload["last_saved_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_json_atomic(manage_state_path(ctx), payload)


def confirmed_fingerprint(ctx: WorkdirContext) -> Optional[ConfigFingerprint]:
    state = load_manage_state(ctx)
    payload = state.get("config_fingerprint")
    if isinstance(payload, dict):
        return ConfigFingerprint.from_dict(payload)
    return None


def confirm_fingerprint(ctx: WorkdirContext, fingerprint: Optional[ConfigFingerprint] = None) -> ConfigFingerprint:
    """The «обновить hash» button: persist the current fingerprint as confirmed."""
    fingerprint = fingerprint or compute_config_fingerprint(ctx)
    state = load_manage_state(ctx)
    state["config_fingerprint"] = fingerprint.as_dict()
    save_manage_state(ctx, state)
    return fingerprint


@dataclass
class StaleConfigWarning:
    kind: str
    message: str


def stale_config_warnings(ctx: WorkdirContext) -> List[StaleConfigWarning]:
    """Warnings when configs changed after the confirmed fingerprint or markers."""
    warnings: List[StaleConfigWarning] = []
    current = compute_config_fingerprint(ctx)
    confirmed = confirmed_fingerprint(ctx)
    if confirmed is not None:
        for kind in current.differs_from(confirmed):
            warnings.append(
                StaleConfigWarning(
                    kind=kind,
                    message=f"{kind} changed since the last confirmed fingerprint",
                )
            )

    def mtime(path: Optional[Path]) -> float:
        try:
            return Path(path).stat().st_mtime if path else 0.0
        except OSError:
            return 0.0

    plan_mtime = mtime(ctx.plan_path)
    zone_mtime = mtime(ctx.zone_file)
    state_dir = ctx.workdir / ".state"
    markers = []
    if state_dir.is_dir():
        try:
            markers = list(state_dir.iterdir())
        except OSError:
            markers = []
    newest_marker = max((mtime(marker) for marker in markers), default=0.0)
    if plan_mtime and newest_marker and plan_mtime > newest_marker:
        warnings.append(
            StaleConfigWarning(
                kind="plan_vs_markers",
                message=".plan is newer than completed stage markers; results may be stale",
            )
        )
    zone_markers = [marker for marker in markers if marker.name.startswith("ZONE_")]
    newest_zone_marker = max((mtime(marker) for marker in zone_markers), default=0.0)
    if zone_mtime and newest_zone_marker and zone_mtime > newest_zone_marker:
        warnings.append(
            StaleConfigWarning(
                kind="zone_vs_markers",
                message="zone file is newer than ZONE_* markers; zone stages may be stale",
            )
        )
    return warnings


__all__ = [
    "ConfigFileRef",
    "ConfigFingerprint",
    "MANAGE_STATE_FILE",
    "STAGE_REGISTRY_VERSION",
    "StaleConfigWarning",
    "compute_config_fingerprint",
    "confirm_fingerprint",
    "confirmed_fingerprint",
    "list_config_files",
    "load_manage_state",
    "load_plan_config",
    "load_zone_text",
    "save_manage_state",
    "save_plan_config",
    "save_zone_text",
    "stale_config_warnings",
    "validate_zone_text",
]
