from __future__ import annotations

import copy
import json
import random
import re
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from utils import workdir_layout as layout
from utils.manage.backup import ManageTransaction, write_json_atomic
from utils.manage.context import WorkdirContext

PassName = Literal["fastpass", "mainpass"]
EncodeOutputPolicy = Literal["delete", "quarantine"]

PASS_NAMES = ("fastpass", "mainpass")
DEFAULT_POLICY_BY_PASS: Dict[str, EncodeOutputPolicy] = {
    "fastpass": "delete",  # cheap to recompute
    "mainpass": "quarantine",  # expensive encode, keep a recoverable copy
}

CHUNK_SORT_ALGORITHMS = (
    "sequential",
    "long-to-short",
    "short-to-long",
    "random",
    "long-biased-random",
    "bitrate",
    "ssimu2",
)


class Av1anStateError(RuntimeError):
    pass


@dataclass
class ChunkState:
    """Thin wrapper over a raw av1an chunk dict; unknown fields are preserved."""

    data: Dict[str, Any]

    @property
    def index(self) -> int:
        return int(self.data.get("index") or 0)

    @index.setter
    def index(self, value: int) -> None:
        self.data["index"] = int(value)

    @property
    def start_frame(self) -> int:
        return int(self.data.get("start_frame") or 0)

    @property
    def end_frame(self) -> int:
        return int(self.data.get("end_frame") or 0)

    @property
    def frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)

    @property
    def frame_rate(self) -> float:
        try:
            return float(self.data.get("frame_rate") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def duration_seconds(self) -> float:
        rate = self.frame_rate
        return self.frames / rate if rate > 0 else 0.0

    @property
    def output_ext(self) -> str:
        return str(self.data.get("output_ext") or "ivf")

    @property
    def video_params(self) -> List[str]:
        params = self.data.get("video_params")
        return [str(token) for token in params] if isinstance(params, list) else []

    @property
    def name(self) -> str:
        return chunk_name(self.index)


@dataclass
class DoneState:
    """Wrapper over av1an ``done.json``; unknown top-level fields preserved."""

    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def frames(self) -> int:
        try:
            return int(self.data.get("frames") or 0)
        except (TypeError, ValueError):
            return 0

    @frames.setter
    def frames(self, value: int) -> None:
        self.data["frames"] = int(value)

    @property
    def done(self) -> Dict[str, Any]:
        done = self.data.get("done")
        if not isinstance(done, dict):
            done = {}
            self.data["done"] = done
        return done

    def is_done(self, index: int) -> bool:
        return chunk_name(index) in self.done

    def remove(self, index: int) -> bool:
        return self.done.pop(chunk_name(index), None) is not None


def chunk_name(index: int) -> str:
    return f"{int(index):05d}"


def pass_dir(ctx: WorkdirContext, pass_name: PassName) -> Path:
    if pass_name not in PASS_NAMES:
        raise Av1anStateError(f"unknown pass name: {pass_name}")
    return layout.video_dir(ctx.workdir) / pass_name


def chunks_path(pass_dir: Path) -> Path:
    return Path(pass_dir) / "chunks.json"


def done_path(pass_dir: Path) -> Path:
    return Path(pass_dir) / "done.json"


def load_chunks(pass_dir: Path) -> List[ChunkState]:
    path = chunks_path(pass_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise Av1anStateError(f"chunks.json not found: {path}") from None
    except Exception as exc:
        raise Av1anStateError(f"failed to parse {path}: {exc}") from exc
    if isinstance(raw, dict):
        raw = raw.get("chunks")
    if not isinstance(raw, list):
        raise Av1anStateError(f"chunks.json must be an array: {path}")
    chunks: List[ChunkState] = []
    for item in raw:
        if not isinstance(item, dict) or "index" not in item:
            raise Av1anStateError(f"invalid chunk entry in {path}")
        chunks.append(ChunkState(data=item))
    return chunks


def save_chunks(pass_dir: Path, chunks: List[ChunkState], *, tx: Optional[ManageTransaction] = None) -> None:
    validate_chunks(chunks)
    payload = [chunk.data for chunk in chunks]
    path = chunks_path(pass_dir)
    if tx is not None:
        tx.write_json(path, payload)
    else:
        write_json_atomic(path, payload)


def load_done(pass_dir: Path) -> DoneState:
    path = done_path(pass_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return DoneState(data={"frames": 0, "done": {}, "audio_done": False})
    except Exception as exc:
        raise Av1anStateError(f"failed to parse {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise Av1anStateError(f"done.json must be an object: {path}")
    return DoneState(data=raw)


def save_done(pass_dir: Path, done: DoneState, *, tx: Optional[ManageTransaction] = None) -> None:
    path = done_path(pass_dir)
    if tx is not None:
        tx.write_json(path, done.data)
    else:
        write_json_atomic(path, done.data)


def chunk_output_path(chunk: ChunkState, pass_dir: Optional[Path] = None) -> Path:
    base = Path(pass_dir) if pass_dir is not None else Path(str(chunk.data.get("temp") or "."))
    return base / "encode" / f"{chunk.name}.{chunk.output_ext}"


def sidecar_paths(pass_dir: Path, index: int) -> List[Path]:
    """Probe/sidecar files for a chunk (first-pass logs, analysis, probes)."""
    split_dir = Path(pass_dir) / "split"
    name = chunk_name(index)
    out: List[Path] = []
    if split_dir.is_dir():
        patterns = (f"{name}_fpf.*", f"{name}_fpf_analysis.dat", f"v_{name}_*.*", f"{index}.json")
        for pattern in patterns:
            out.extend(sorted(split_dir.glob(pattern)))
    return out


def validate_chunks(chunks: Sequence[ChunkState]) -> List[str]:
    """Raise on fatal issues; return non-fatal warnings."""
    warnings: List[str] = []
    seen: Dict[int, int] = {}
    ranges: List[Tuple[int, int, int]] = []
    for chunk in chunks:
        index = chunk.index
        if index < 0:
            raise Av1anStateError(f"chunk index must be >= 0: {index}")
        if index in seen:
            raise Av1anStateError(f"duplicate chunk index: {index}")
        seen[index] = 1
        if chunk.start_frame < 0:
            raise Av1anStateError(f"chunk {index}: start_frame must be >= 0")
        if chunk.end_frame <= chunk.start_frame:
            raise Av1anStateError(f"chunk {index}: end_frame must be > start_frame")
        if not chunk.output_ext:
            raise Av1anStateError(f"chunk {index}: output_ext is empty")
        params = chunk.data.get("video_params")
        if params is not None and not isinstance(params, list):
            raise Av1anStateError(f"chunk {index}: video_params must be an array")
        ranges.append((chunk.start_frame, chunk.end_frame, index))
    ranges.sort()
    for left, right in zip(ranges, ranges[1:]):
        if right[0] < left[1]:
            warnings.append(f"chunks {left[2]} and {right[2]} have overlapping frame ranges")
    return warnings


def validate_done(done: DoneState, chunks: Sequence[ChunkState]) -> List[str]:
    warnings: List[str] = []
    by_name = {chunk.name: chunk for chunk in chunks}
    for key, value in done.done.items():
        if not (isinstance(key, str) and key.isdigit() and len(key) == 5):
            warnings.append(f"done key has unexpected format: {key!r}")
            continue
        chunk = by_name.get(key)
        if chunk is None:
            warnings.append(f"done entry {key} has no matching chunk")
            continue
        try:
            frames = int(dict(value).get("frames") or 0)
        except Exception:
            frames = 0
        if frames and frames != chunk.frames:
            warnings.append(f"done entry {key} frames={frames} does not match chunk frames={chunk.frames}")
    if done.frames < 0:
        warnings.append("done.frames is negative")
    return warnings


def quarantine_or_delete_encode_output(
    chunk: ChunkState,
    policy: EncodeOutputPolicy,
    *,
    pass_dir: Optional[Path] = None,
    tx: Optional[ManageTransaction] = None,
) -> List[Path]:
    """Apply the policy to the chunk encode output. Returns affected paths."""
    output = chunk_output_path(chunk, pass_dir)
    if not output.exists():
        return []
    if policy == "quarantine":
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = output.with_name(
            f"{chunk.name}_deleted-{stamp}-{chunk.start_frame}-{chunk.end_frame}.{chunk.output_ext}"
        )
        counter = 1
        while target.exists():
            counter += 1
            target = output.with_name(
                f"{chunk.name}_deleted-{stamp}-{chunk.start_frame}-{chunk.end_frame}-{counter}.{chunk.output_ext}"
            )
        if tx is not None:
            tx.rename(output, target)
        else:
            output.rename(target)
        return [output, target]
    if tx is not None:
        tx.record_change(output)
    output.unlink()
    return [output]


def mark_chunk_not_done(
    pass_dir: Path,
    index: int,
    *,
    policy: EncodeOutputPolicy = "delete",
    tx: Optional[ManageTransaction] = None,
) -> List[Path]:
    """Remove the done entry, encode output (per policy) and sidecars of a chunk.

    Downstream cleanup (fastpass output, video-final, stage markers) is the
    caller's job — see ``utils.manage.reset.reset_chunk``.
    """
    pass_dir = Path(pass_dir)
    chunks = load_chunks(pass_dir)
    chunk = next((item for item in chunks if item.index == int(index)), None)
    if chunk is None:
        raise Av1anStateError(f"chunk index {index} not found in {chunks_path(pass_dir)}")
    done = load_done(pass_dir)
    changed: List[Path] = []
    if done.remove(chunk.index):
        save_done(pass_dir, done, tx=tx)
        changed.append(done_path(pass_dir))
    changed.extend(quarantine_or_delete_encode_output(chunk, policy, pass_dir=pass_dir, tx=tx))
    for sidecar in sidecar_paths(pass_dir, chunk.index):
        # probe/sidecar files are always deleted, the quarantine policy only
        # protects the (expensive) encode output
        if tx is not None:
            tx.record_change(sidecar)
        sidecar.unlink(missing_ok=True)
        changed.append(sidecar)
    return changed


# -- queue order operations ----------------------------------------------


def _position_of(chunks: Sequence[ChunkState], index: int) -> int:
    for position, chunk in enumerate(chunks):
        if chunk.index == int(index):
            return position
    raise Av1anStateError(f"chunk index {index} not found")


def swap_chunks(
    pass_dir: Path,
    left_index: int,
    right_index: int,
    *,
    tx: Optional[ManageTransaction] = None,
) -> None:
    chunks = load_chunks(pass_dir)
    left = _position_of(chunks, left_index)
    right = _position_of(chunks, right_index)
    chunks[left], chunks[right] = chunks[right], chunks[left]
    save_chunks(pass_dir, chunks, tx=tx)


@dataclass(frozen=True)
class MoveTarget:
    kind: Literal["start", "end", "before", "after"]
    anchor_index: Optional[int] = None


MoveOrder = Literal["original", "ascending", "descending"]


def move_chunks(
    pass_dir: Path,
    selector: Sequence[int],
    target: MoveTarget,
    order: MoveOrder = "original",
    *,
    tx: Optional[ManageTransaction] = None,
) -> None:
    chunks = load_chunks(pass_dir)
    selected_set = {int(index) for index in selector}
    if not selected_set:
        raise Av1anStateError("no chunks selected to move")
    missing = selected_set - {chunk.index for chunk in chunks}
    if missing:
        raise Av1anStateError(f"chunk indices not found: {sorted(missing)}")
    if target.kind in ("before", "after"):
        if target.anchor_index is None:
            raise Av1anStateError(f"move target {target.kind!r} requires anchor_index")
        if target.anchor_index in selected_set:
            raise Av1anStateError("anchor chunk cannot be part of the moved selection")

    moved = [chunk for chunk in chunks if chunk.index in selected_set]
    if order == "ascending":
        moved.sort(key=lambda chunk: chunk.index)
    elif order == "descending":
        moved.sort(key=lambda chunk: chunk.index, reverse=True)
    rest = [chunk for chunk in chunks if chunk.index not in selected_set]

    if target.kind == "start":
        result = moved + rest
    elif target.kind == "end":
        result = rest + moved
    else:
        anchor_pos = _position_of(rest, int(target.anchor_index))
        insert_at = anchor_pos if target.kind == "before" else anchor_pos + 1
        result = rest[:insert_at] + moved + rest[insert_at:]
    save_chunks(pass_dir, result, tx=tx)


def sort_chunks(
    pass_dir: Path,
    algorithm: str,
    *,
    key_values: Optional[Dict[int, float]] = None,
    descending: bool = False,
    seed: Optional[int] = None,
    tx: Optional[ManageTransaction] = None,
) -> None:
    """Reorder the av1an queue. Only array order changes, never indices.

    ``bitrate``/``ssimu2`` need per-chunk values via ``key_values`` (collected
    by :mod:`utils.manage.analytics`); chunks without a value keep relative
    order at the end of the queue.
    """
    algorithm = str(algorithm or "").strip().lower()
    chunks = load_chunks(pass_dir)
    rng = random.Random(seed)
    if algorithm == "sequential":
        chunks.sort(key=lambda chunk: chunk.start_frame)
    elif algorithm == "long-to-short":
        chunks.sort(key=lambda chunk: chunk.frames, reverse=True)
    elif algorithm == "short-to-long":
        chunks.sort(key=lambda chunk: chunk.frames)
    elif algorithm == "random":
        rng.shuffle(chunks)
    elif algorithm == "long-biased-random":
        chunks = _long_biased_random(chunks, rng)
    elif algorithm in ("bitrate", "ssimu2", "length", "duration"):
        if algorithm in ("length", "duration"):
            values = {chunk.index: float(chunk.frames) for chunk in chunks}
        else:
            values = dict(key_values or {})
            if not values:
                raise Av1anStateError(f"sort algorithm {algorithm!r} requires key_values")
        known = [chunk for chunk in chunks if chunk.index in values]
        unknown = [chunk for chunk in chunks if chunk.index not in values]
        known.sort(key=lambda chunk: values[chunk.index], reverse=descending)
        chunks = known + unknown
    else:
        raise Av1anStateError(f"unknown chunk sort algorithm: {algorithm!r}")
    save_chunks(pass_dir, chunks, tx=tx)


def _long_biased_random(chunks: List[ChunkState], rng: random.Random) -> List[ChunkState]:
    """Weighted sampling without replacement, longer chunks earlier on average."""
    pool = list(chunks)
    out: List[ChunkState] = []
    while pool:
        weights = [max(1, chunk.frames) for chunk in pool]
        total = sum(weights)
        pick = rng.uniform(0, total)
        cursor = 0.0
        chosen = len(pool) - 1
        for position, weight in enumerate(weights):
            cursor += weight
            if pick <= cursor:
                chosen = position
                break
        out.append(pool.pop(chosen))
    return out


# -- chunk params ----------------------------------------------------------


@dataclass(frozen=True)
class ChunkPatch:
    video_params: Optional[List[str]] = None


def update_chunk_params(
    pass_dir: Path,
    index: int,
    patch: ChunkPatch,
    *,
    policy: EncodeOutputPolicy = "delete",
    tx: Optional[ManageTransaction] = None,
) -> bool:
    """Apply a chunk patch; resets the chunk done state if it was done.

    Only ``video_params`` is editable in the first version: other fields
    (encoder, passes, target_quality, noise) feed the av1an fork command
    builder and target-quality probes — partial sync breaks resume.
    Returns True if the chunk had been done and was reset.
    """
    pass_dir = Path(pass_dir)
    chunks = load_chunks(pass_dir)
    chunk = next((item for item in chunks if item.index == int(index)), None)
    if chunk is None:
        raise Av1anStateError(f"chunk index {index} not found in {chunks_path(pass_dir)}")
    if patch.video_params is None:
        return False
    chunk.data["video_params"] = [str(token) for token in patch.video_params]
    done = load_done(pass_dir)
    was_done = done.is_done(chunk.index)
    save_chunks(pass_dir, chunks, tx=tx)
    if was_done:
        mark_chunk_not_done(pass_dir, chunk.index, policy=policy, tx=tx)
    return was_done


# -- multi-chunk param editing ----------------------------------------------
#
# "Edit params" can target several chunks at once. The dialog shows a merged
# view of their ``video_params`` where any key whose value is not identical
# across the selection is rendered as the ``{variable}`` placeholder. Leaving
# the placeholder keeps each chunk's own value; replacing it sets one value for
# the whole selection. Editing matches single-chunk behaviour for identical
# values.

VARIABLE_PLACEHOLDER = "{variable}"


def parse_video_param_pairs(tokens: Sequence[str]) -> List[Tuple[str, Optional[str]]]:
    """Tokens -> ordered ``(--key, value | None)`` pairs (value None for flags).

    Keys are ``--`` prefixed (so negative numeric values are kept as values);
    a stray non-key token is preserved as a valueless pseudo-flag.
    """
    items = [str(token) for token in tokens]
    pairs: List[Tuple[str, Optional[str]]] = []
    index = 0
    while index < len(items):
        key = items[index]
        if key.startswith("--"):
            if index + 1 < len(items) and not items[index + 1].startswith("--"):
                pairs.append((key, items[index + 1]))
                index += 2
            else:
                pairs.append((key, None))
                index += 1
        else:
            pairs.append((key, None))
            index += 1
    return pairs


def format_video_param_pairs(pairs: Sequence[Tuple[str, Optional[str]]]) -> List[str]:
    tokens: List[str] = []
    for key, value in pairs:
        tokens.append(key)
        if value is not None:
            tokens.append(value)
    return tokens


def _param_pairs_to_map(pairs: Sequence[Tuple[str, Optional[str]]]) -> Dict[str, Optional[str]]:
    return {key: value for key, value in pairs}  # last occurrence wins


def merge_video_params(param_lists: Sequence[Sequence[str]]) -> List[str]:
    """Merge several ``video_params`` lists into one editable template.

    A key present with the same value in every list keeps that value; any key
    that differs (different value, or missing from some lists) becomes
    ``{variable}``. Key order follows first appearance across the lists.
    """
    parsed = [parse_video_param_pairs(params) for params in param_lists]
    maps = [_param_pairs_to_map(pairs) for pairs in parsed]
    order: List[str] = []
    seen = set()
    for pairs in parsed:
        for key, _value in pairs:
            if key not in seen:
                seen.add(key)
                order.append(key)
    tokens: List[str] = []
    for key in order:
        present = [mapping for mapping in maps if key in mapping]
        values = [mapping[key] for mapping in present]
        uniform = len(present) == len(maps) and all(value == values[0] for value in values)
        tokens.append(key)
        if uniform:
            if values[0] is not None:
                tokens.append(str(values[0]))
        else:
            tokens.append(VARIABLE_PLACEHOLDER)
    return tokens


def apply_param_template(template_tokens: Sequence[str], current_tokens: Sequence[str]) -> List[str]:
    """Apply a merged-view template to one chunk's current ``video_params``.

    ``{variable}`` keeps the chunk's own value for that key (or stays absent if
    the chunk lacks it); any other value overrides; keys absent from the
    template are dropped. Output order follows the template.
    """
    current_map = _param_pairs_to_map(parse_video_param_pairs(current_tokens))
    out: List[Tuple[str, Optional[str]]] = []
    for key, value in parse_video_param_pairs(template_tokens):
        if value == VARIABLE_PLACEHOLDER:
            if key in current_map:
                out.append((key, current_map[key]))
        else:
            out.append((key, value))
    return format_video_param_pairs(out)


@dataclass
class MultiParamResult:
    changed_indices: List[int] = field(default_factory=list)
    reset_indices: List[int] = field(default_factory=list)


def update_chunks_params(
    pass_dir: Path,
    indices: Sequence[int],
    template_tokens: Sequence[str],
    *,
    policy: EncodeOutputPolicy = "delete",
    tx: Optional[ManageTransaction] = None,
) -> MultiParamResult:
    """Apply an Edit-Params template to several chunks at once.

    Only chunks whose resulting ``video_params`` actually change (compared as a
    key->value map, so a pure reorder is not a change) are rewritten; changed
    chunks that were done are reset via ``policy``.
    """
    pass_dir = Path(pass_dir)
    chunks = load_chunks(pass_dir)
    by_index = {chunk.index: chunk for chunk in chunks}
    targets: List[ChunkState] = []
    for index in indices:
        chunk = by_index.get(int(index))
        if chunk is None:
            raise Av1anStateError(f"chunk index {index} not found in {chunks_path(pass_dir)}")
        targets.append(chunk)
    if not targets:
        raise Av1anStateError("no chunks selected")

    done = load_done(pass_dir)
    result = MultiParamResult()
    for chunk in targets:
        new_tokens = apply_param_template(template_tokens, chunk.video_params)
        new_map = _param_pairs_to_map(parse_video_param_pairs(new_tokens))
        old_map = _param_pairs_to_map(parse_video_param_pairs(chunk.video_params))
        if new_map == old_map:
            continue  # no semantic change -> leave the chunk untouched
        chunk.data["video_params"] = [str(token) for token in new_tokens]
        result.changed_indices.append(chunk.index)
        if done.is_done(chunk.index):
            result.reset_indices.append(chunk.index)

    if not result.changed_indices:
        return result
    result.changed_indices.sort()
    result.reset_indices.sort()
    save_chunks(pass_dir, chunks, tx=tx)
    for index in result.reset_indices:
        mark_chunk_not_done(pass_dir, index, policy=policy, tx=tx)
    return result


# -- reindex -----------------------------------------------------------------


@dataclass
class ReindexResult:
    mapping: Dict[int, int]
    renamed_files: List[Tuple[Path, Path]] = field(default_factory=list)
    remapped_done_keys: List[Tuple[str, str]] = field(default_factory=list)


def reindex_chunks_transactionally(
    pass_dir: Path,
    mapping: Dict[int, int],
    *,
    tx: Optional[ManageTransaction] = None,
    save: bool = True,
    removed_indices: Optional[set] = None,
) -> ReindexResult:
    """Renumber chunks: chunk dicts, done keys, encode outputs and sidecars.

    Renames go through unique temp names first so overlapping mappings
    (e.g. every index shifted by +1) cannot collide. ``removed_indices``
    marks chunks the caller is about to drop (merge) so their indices do not
    count as occupied.
    """
    pass_dir = Path(pass_dir)
    mapping = {int(old): int(new) for old, new in mapping.items() if int(old) != int(new)}
    result = ReindexResult(mapping=dict(mapping))
    if not mapping:
        return result
    targets = list(mapping.values())
    if len(set(targets)) != len(targets):
        raise Av1anStateError("reindex mapping has duplicate target indices")

    chunks = load_chunks(pass_dir)
    by_index = {chunk.index: chunk for chunk in chunks}
    for old in mapping:
        if old not in by_index:
            raise Av1anStateError(f"reindex source chunk not found: {old}")
    stable_indices = set(by_index) - set(mapping) - set(removed_indices or ())
    collisions = stable_indices & set(targets)
    if collisions:
        raise Av1anStateError(f"reindex targets collide with untouched chunks: {sorted(collisions)}")

    # 1. plan file renames (encode output + sidecars)
    renames: List[Tuple[Path, Path]] = []
    for old, new in mapping.items():
        chunk = by_index[old]
        old_output = chunk_output_path(chunk, pass_dir)
        if old_output.exists():
            new_output = old_output.with_name(f"{chunk_name(new)}.{chunk.output_ext}")
            renames.append((old_output, new_output))
        for sidecar in sidecar_paths(pass_dir, old):
            new_name = sidecar.name
            old_long, new_long = chunk_name(old), chunk_name(new)
            if new_name.startswith(f"{old_long}_"):
                new_name = f"{new_long}_" + new_name[len(old_long) + 1 :]
            elif new_name.startswith(f"v_{old_long}_"):
                new_name = f"v_{new_long}_" + new_name[len(old_long) + 3 :]
            elif new_name == f"{old}.json":
                new_name = f"{new}.json"
            else:
                continue
            renames.append((sidecar, sidecar.with_name(new_name)))

    # 2. two-phase rename through temp names
    temp_pairs: List[Tuple[Path, Path]] = []
    for ordinal, (src, dst) in enumerate(renames):
        temp = src.with_name(f"__reindex_{ordinal}__{dst.name}")
        if tx is not None:
            tx.rename(src, temp)
        else:
            src.rename(temp)
        temp_pairs.append((temp, dst))
    for temp, dst in temp_pairs:
        if dst.exists():
            raise Av1anStateError(f"reindex target already exists: {dst}")
        if tx is not None:
            tx.rename(temp, dst)
        else:
            temp.rename(dst)
    result.renamed_files = renames

    # 3. remap chunk indices and done keys
    done = load_done(pass_dir)
    old_done = dict(done.done)
    new_done: Dict[str, Any] = {}
    for key, value in old_done.items():
        try:
            key_index = int(key)
        except ValueError:
            new_done[key] = value
            continue
        if key_index in mapping:
            new_key = chunk_name(mapping[key_index])
            new_done[new_key] = value
            result.remapped_done_keys.append((key, new_key))
        else:
            new_done[key] = value
    done.data["done"] = new_done
    for old, new in mapping.items():
        by_index[old].index = new

    if save:
        save_chunks(pass_dir, chunks, tx=tx)
        save_done(pass_dir, done, tx=tx)
    return result


# -- chunk frame range / source command --------------------------------------
#
# av1an stores the *real* decoded frame range of a chunk inside ``source_cmd``
# (and ``proxy_cmd``), not only in ``start_frame``/``end_frame``. On --resume
# av1an runs ``source_cmd`` verbatim and rejects the chunk with a FRAME MISMATCH
# if the piped frame count differs from ``end_frame - start_frame``. So every
# geometry edit that moves a chunk boundary must rewrite the command too, or the
# encode loops forever. Two command shapes are produced by av1an:
#   * vspipe:         vspipe <script> -c y4m - -s <start> -e <end-1> [-a ...]
#   * ffmpeg Select:  ffmpeg ... -vf select=between(n\,<start>\,<end-1>) ...
# Tokens may be plain strings (``--chunks-cmd-format text``) or native OsStrings
# (the default "safe" format: ``{"Windows": [u16, ...]}`` / ``{"Unix": [u8...]}``).

OsToken = Union[str, Dict[str, Any]]

_SELECT_BETWEEN_RE = re.compile(
    r"(select\s*=\s*between\(\s*n\s*\\?,\s*)\d+(\s*\\?,\s*)\d+(\s*\))",
    re.IGNORECASE,
)


def _os_token_text(token: OsToken) -> Optional[str]:
    """Decode an av1an command token (str or native OsString) to text."""
    if isinstance(token, str):
        return token
    if isinstance(token, dict):
        windows = token.get("Windows")
        if isinstance(windows, list):
            try:
                units = [int(unit) & 0xFFFF for unit in windows]
                return struct.pack(f"<{len(units)}H", *units).decode("utf-16-le", "surrogatepass")
            except Exception:
                return None
        unix = token.get("Unix")
        if isinstance(unix, list):
            try:
                return bytes(int(byte) & 0xFF for byte in unix).decode("utf-8", "surrogateescape")
            except Exception:
                return None
    return None


def _os_token_like(template: OsToken, text: str) -> OsToken:
    """Encode ``text`` in the same JSON representation as ``template``."""
    if isinstance(template, dict):
        if isinstance(template.get("Windows"), list):
            data = text.encode("utf-16-le", "surrogatepass")
            return {"Windows": list(struct.unpack(f"<{len(data) // 2}H", data))}
        if isinstance(template.get("Unix"), list):
            return {"Unix": list(text.encode("utf-8", "surrogateescape"))}
    return text


def _rewrite_cmd_frame_range(cmd: List[OsToken], start_frame: int, end_frame: int) -> List[OsToken]:
    """Return a copy of ``cmd`` with the embedded frame range updated.

    av1an's ``-e`` / select upper bound is the *last inclusive* frame, i.e.
    ``end_frame - 1``. Commands with no recognizable frame range (e.g. ffmpeg
    Segment, which reads a pre-split file) are returned unchanged.
    """
    end_inclusive = end_frame - 1
    out: List[OsToken] = list(cmd)
    index = 0
    count = len(out)
    while index < count:
        text = _os_token_text(out[index])
        # vspipe script args follow ``-a``; never interpret them as seek flags
        if text in ("-a", "--arg") and index + 1 < count:
            index += 2
            continue
        if text in ("-s", "--start") and index + 1 < count:
            out[index + 1] = _os_token_like(out[index + 1], str(start_frame))
            index += 2
            continue
        if text in ("-e", "--end") and index + 1 < count:
            out[index + 1] = _os_token_like(out[index + 1], str(end_inclusive))
            index += 2
            continue
        if text:
            rewritten = _SELECT_BETWEEN_RE.sub(
                lambda match: f"{match.group(1)}{start_frame}{match.group(2)}{end_inclusive}{match.group(3)}",
                text,
            )
            if rewritten != text:
                out[index] = _os_token_like(out[index], rewritten)
        index += 1
    return out


def set_chunk_frame_range(chunk: ChunkState, start_frame: int, end_frame: int) -> None:
    """Set a chunk's frame range and keep ``source_cmd``/``proxy_cmd`` in sync.

    This is the single place that mutates chunk geometry so the decode command
    av1an actually runs can never drift from ``start_frame``/``end_frame``.
    """
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    chunk.data["start_frame"] = start_frame
    chunk.data["end_frame"] = end_frame
    for key in ("source_cmd", "proxy_cmd"):
        cmd = chunk.data.get(key)
        if isinstance(cmd, list) and cmd:
            chunk.data[key] = _rewrite_cmd_frame_range(cmd, start_frame, end_frame)


# -- geometry ----------------------------------------------------------------


@dataclass
class GeometryResult:
    """Outcome of a split/merge/reshape edit."""

    reset_indices: List[int] = field(default_factory=list)
    new_indices: List[int] = field(default_factory=list)
    reindexed: Dict[int, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def split_chunk(
    pass_dir: Path,
    index: int,
    frame: int,
    *,
    policy: EncodeOutputPolicy = "delete",
    tx: Optional[ManageTransaction] = None,
) -> GeometryResult:
    """Split a chunk at ``frame`` (absolute, exclusive end of the left part)."""
    pass_dir = Path(pass_dir)
    chunks = load_chunks(pass_dir)
    position = _position_of(chunks, index)
    chunk = chunks[position]
    if not (chunk.start_frame < int(frame) < chunk.end_frame):
        raise Av1anStateError(
            f"split frame {frame} must be inside ({chunk.start_frame}, {chunk.end_frame}) of chunk {index}"
        )

    old_index = chunk.index
    mapping = {item.index: item.index + 1 for item in chunks if item.index > old_index}
    # reset the changed source chunk before files get renumbered
    reset_paths = mark_chunk_not_done(pass_dir, old_index, policy=policy, tx=tx)
    reindex = reindex_chunks_transactionally(pass_dir, mapping, tx=tx, save=False)
    for item in chunks:
        if item.index in mapping:
            item.index = mapping[item.index]

    original_end = chunk.end_frame
    right = ChunkState(data=copy.deepcopy(chunk.data))
    right.index = old_index + 1
    set_chunk_frame_range(right, int(frame), original_end)
    set_chunk_frame_range(chunk, chunk.start_frame, int(frame))
    chunks.insert(position + 1, right)

    save_chunks(pass_dir, chunks, tx=tx)
    done = load_done(pass_dir)
    done.remove(old_index)  # safety: reset above already dropped it
    _remap_done_after_reindex(done, reindex.mapping)
    save_done(pass_dir, done, tx=tx)

    result = GeometryResult(
        reset_indices=[old_index],
        new_indices=[old_index + 1],
        reindexed=reindex.mapping,
    )
    if reset_paths:
        result.warnings.append(f"chunk {old_index} encode state was reset ({policy})")
    return result


def _remap_done_after_reindex(done: DoneState, mapping: Dict[int, int]) -> None:
    """Apply an index mapping to done keys (for flows that re-save done.json)."""
    if not mapping:
        return
    old_done = dict(done.done)
    new_done: Dict[str, Any] = {}
    for key, value in old_done.items():
        try:
            key_index = int(key)
        except ValueError:
            new_done[key] = value
            continue
        new_done[chunk_name(mapping.get(key_index, key_index))] = value
    done.data["done"] = new_done


def merge_chunks(
    pass_dir: Path,
    indices: Sequence[int],
    *,
    policy: EncodeOutputPolicy = "delete",
    tx: Optional[ManageTransaction] = None,
) -> GeometryResult:
    """Merge a run of frame-adjacent chunks into the first one."""
    pass_dir = Path(pass_dir)
    ordered = sorted({int(value) for value in indices})
    if len(ordered) < 2:
        raise Av1anStateError("merge requires at least two chunk indices")
    chunks = load_chunks(pass_dir)
    by_index = {chunk.index: chunk for chunk in chunks}
    missing = [index for index in ordered if index not in by_index]
    if missing:
        raise Av1anStateError(f"chunk indices not found: {missing}")
    for left_idx, right_idx in zip(ordered, ordered[1:]):
        if right_idx != left_idx + 1:
            raise Av1anStateError(f"merge indices must be consecutive: {ordered}")
        left, right = by_index[left_idx], by_index[right_idx]
        if left.end_frame != right.start_frame:
            raise Av1anStateError(
                f"chunks {left_idx} and {right_idx} are not adjacent by frames "
                f"({left.end_frame} != {right.start_frame})"
            )

    first = by_index[ordered[0]]
    last = by_index[ordered[-1]]
    removed = ordered[1:]
    shift = len(removed)

    # all merged chunks lose their encode state
    for merge_index in ordered:
        mark_chunk_not_done(pass_dir, merge_index, policy=policy, tx=tx)
    mapping = {chunk.index: chunk.index - shift for chunk in chunks if chunk.index > ordered[-1]}
    reindex = reindex_chunks_transactionally(
        pass_dir, mapping, tx=tx, save=False, removed_indices=set(removed)
    )

    removed_ids = {id(by_index[index]) for index in removed}
    merged_chunks = [chunk for chunk in chunks if id(chunk) not in removed_ids]
    for chunk in merged_chunks:
        if chunk.index in mapping:
            chunk.index = mapping[chunk.index]
    set_chunk_frame_range(first, first.start_frame, last.end_frame)

    save_chunks(pass_dir, merged_chunks, tx=tx)
    done = load_done(pass_dir)
    for merge_index in ordered:
        done.remove(merge_index)
    _remap_done_after_reindex(done, reindex.mapping)
    save_done(pass_dir, done, tx=tx)

    return GeometryResult(
        reset_indices=list(ordered),
        new_indices=[ordered[0]],
        reindexed=reindex.mapping,
    )


@dataclass(frozen=True)
class FrameRange:
    start_frame: int
    end_frame: int


def reshape_chunk_range(
    pass_dir: Path,
    selector: FrameRange,
    *,
    policy: EncodeOutputPolicy = "delete",
    tx: Optional[ManageTransaction] = None,
) -> GeometryResult:
    """Make ``[start_frame, end_frame)`` a single chunk: split edges, merge inside."""
    start, end = int(selector.start_frame), int(selector.end_frame)
    if end <= start:
        raise Av1anStateError("reshape range must have end_frame > start_frame")
    combined = GeometryResult()

    def locate(frame: int) -> Optional[ChunkState]:
        for chunk in load_chunks(pass_dir):
            if chunk.start_frame < frame < chunk.end_frame:
                return chunk
        return None

    boundary_chunk = locate(start)
    if boundary_chunk is not None:
        step = split_chunk(pass_dir, boundary_chunk.index, start, policy=policy, tx=tx)
        combined.reset_indices.extend(step.reset_indices)
    boundary_chunk = locate(end)
    if boundary_chunk is not None:
        step = split_chunk(pass_dir, boundary_chunk.index, end, policy=policy, tx=tx)
        combined.reset_indices.extend(step.reset_indices)

    inside = [
        chunk.index
        for chunk in load_chunks(pass_dir)
        if chunk.start_frame >= start and chunk.end_frame <= end
    ]
    if not inside:
        raise Av1anStateError(f"no chunks inside range [{start}, {end})")
    if len(inside) > 1:
        step = merge_chunks(pass_dir, inside, policy=policy, tx=tx)
        combined.reset_indices.extend(step.reset_indices)
        combined.new_indices = step.new_indices
        combined.reindexed = step.reindexed
    else:
        combined.new_indices = inside
    combined.reset_indices = sorted(set(combined.reset_indices))
    return combined


__all__ = [
    "Av1anStateError",
    "CHUNK_SORT_ALGORITHMS",
    "ChunkPatch",
    "ChunkState",
    "DEFAULT_POLICY_BY_PASS",
    "DoneState",
    "EncodeOutputPolicy",
    "FrameRange",
    "GeometryResult",
    "MoveOrder",
    "MoveTarget",
    "MultiParamResult",
    "PASS_NAMES",
    "PassName",
    "ReindexResult",
    "VARIABLE_PLACEHOLDER",
    "apply_param_template",
    "chunk_name",
    "chunk_output_path",
    "chunks_path",
    "done_path",
    "format_video_param_pairs",
    "load_chunks",
    "load_done",
    "mark_chunk_not_done",
    "merge_chunks",
    "merge_video_params",
    "move_chunks",
    "parse_video_param_pairs",
    "pass_dir",
    "quarantine_or_delete_encode_output",
    "reindex_chunks_transactionally",
    "reshape_chunk_range",
    "save_chunks",
    "save_done",
    "set_chunk_frame_range",
    "sidecar_paths",
    "sort_chunks",
    "split_chunk",
    "swap_chunks",
    "update_chunk_params",
    "update_chunks_params",
    "validate_chunks",
    "validate_done",
]
