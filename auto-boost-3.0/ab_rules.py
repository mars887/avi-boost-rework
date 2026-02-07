"""Rule engine and per-scene metrics helpers."""

from __future__ import annotations

import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ab_fs import ensure_exists, load_json
from ab_logging import eprint
from ab_nvof import normalize_metric_name, requires_global
from ab_params import (
    coerce_param_value,
    find_last_option,
    format_param_value,
    normalize_param_key,
    normalize_video_params,
    parse_numeric_value,
)
from ab_registry import registry
import ab_metrics_builtin  # noqa: F401
import ab_metrics_nvof  # noqa: F401
from ab_ssimu2 import calc_stats, parse_ssimu2_log, slice_samples_for_scene
from ab_state import is_valid_ssimu2_log

_METRIC_CALL_RE = re.compile(r"metric\(\s*['\"]([^'\"]+)['\"]\s*\)")


def extract_metric_calls(rule_text: str) -> List[str]:
    if not rule_text:
        return []
    return [m.strip().lower() for m in _METRIC_CALL_RE.findall(rule_text)]


def parse_frame_rate(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if "/" in s:
            return float(Fraction(s))
        return float(s)
    raise ValueError(f"Unsupported frame_rate value: {value!r}")


def load_av1an_chunks(av1an_temp: Path) -> List[Dict[str, Any]]:
    chunks_path = av1an_temp / "chunks.json"
    ensure_exists(chunks_path, "av1an chunks.json")
    raw = load_json(chunks_path)
    chunks = raw.get("chunks") if isinstance(raw, dict) else raw
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"Invalid chunks.json: {chunks_path}")
    return chunks


PASS1_METRICS = {
    "scene_bitrate",
    "scene_bitrate_ratio",
    "ssimu2",
    "ssimu2_p5",
    "ssimu2_avg",
}


class MetricRequirements:
    """Track which metrics are required per-scene or globally."""

    def __init__(self) -> None:
        self.scene_metrics: Dict[str, set[int]] = {}
        self.global_metrics: set[str] = set()

    def add(self, name: str, scene_index: int) -> None:
        key = normalize_metric_name(name)
        if key.startswith("global_") or requires_global(key):
            self.global_metrics.add(key)
            return
        self.scene_metrics.setdefault(key, set()).add(int(scene_index))

    def add_global(self, name: str) -> None:
        key = normalize_metric_name(name)
        self.global_metrics.add(key)

    def has(self, name: str) -> bool:
        key = normalize_metric_name(name)
        return key in self.global_metrics or key in self.scene_metrics

    def all_metrics(self) -> set[str]:
        return set(self.global_metrics) | set(self.scene_metrics.keys())


def collect_rule_requirements(
    *,
    base_obj: Dict[str, Any],
    compiled_rules: Any,
    rule_name: str,
    metrics_state: "RuleMetricsState",
    verbose: bool,
    pass1_metrics: Optional[Sequence[str]] = None,
) -> MetricRequirements:
    """Execute rules in pass-1 mode to collect required metric ranges."""
    scenes = validate_scenes_with_overrides(base_obj)
    scene_count = len(scenes)
    fps = metrics_state._get_fps()
    requirements = MetricRequirements()
    allowed = set(m.strip().lower() for m in (pass1_metrics or PASS1_METRICS))
    warned: set[str] = set()

    for i, s in enumerate(scenes):
        st = int(s["start_frame"])
        en = int(s["end_frame"])
        zone = dict(s["zone_overrides"])
        video_params = list(zone["video_params"])

        def rule_log(*args: Any) -> None:
            if not verbose:
                return
            msg = " ".join(str(a) for a in args)
            print(f"[rules] pass1 scene {i} [{st}:{en}] {msg}")

        editor = VideoParamEditor(
            video_params,
            scene_index=i,
            start_frame=st,
            end_frame=en,
            rule_name=rule_name,
            verbose=verbose,
        )
        metrics = SceneMetrics(metrics_state, i, st, en)

        def metric_pass1(
            name: str,
            *,
            scene_index_override: Optional[int] = None,
            st_override: Optional[int] = None,
            en_override: Optional[int] = None,
        ) -> float:
            idx = i if scene_index_override is None else int(scene_index_override)
            if idx < 0 or idx >= scene_count:
                raise IndexError(f"Scene index out of range: {idx}")
            if st_override is None or en_override is None:
                st_use = int(scenes[idx]["start_frame"])
                en_use = int(scenes[idx]["end_frame"])
            else:
                st_use = int(st_override)
                en_use = int(en_override)
            key = normalize_metric_name(name)
            if key in allowed:
                if idx == i and st_use == st and en_use == en:
                    return metrics.metric(key)
                return SceneMetrics(metrics_state, idx, st_use, en_use).metric(key)
            requirements.add(key, idx)
            if key not in warned:
                warned.add(key)
                eprint(f"[warn] metric '{key}' requested in pass1; use require() for better scheduling.")
            return float("nan")

        def metric_at(scene_index: int, name: str) -> float:
            return metric_pass1(name, scene_index_override=scene_index)

        def require(name: str) -> None:
            requirements.add(name, i)

        scene_len = max(0, en - st)
        scene_seconds = (scene_len / fps) if fps > 0 else 0.0

        ctx_globals = {
            "__builtins__": __builtins__,
            "metric": metric_pass1,
            "metric_at": metric_at,
            "require": require,
            "param": editor.param,
            "sparam": editor.sparam,
            "cparam": editor.cparam,
            "log": rule_log,
            "rule_pass": 1,
            "RULE_PASS": 1,
            "scene_index": i,
            "scene_start": st,
            "scene_end": en,
            "scene_len": scene_len,
            "scene_seconds": scene_seconds,
            "scene_fps": fps,
            "scene_count": scene_count,
        }
        ctx_locals: Dict[str, Any] = ctx_globals

        try:
            exec(compiled_rules, ctx_globals, ctx_locals)
        except Exception as exc:
            eprint(f"[error] rules pass1 failed for scene {i} [{st}:{en}]: {exc}")
            raise

    return requirements


class RuleMetricsState:
    """Cache and compute metrics required by rule scripts."""

    def __init__(
        self,
        *,
        src_file: Path,
        vpy_src: Optional[Path] = None,
        frames_count: int,
        skip: int,
        av1an_temp: Path,
        fastpass_out: Path,
        ssimu2_log: Path,
        vs_source: str,
        verbose: bool,
        scene_ranges: Optional[List[Tuple[int, int]]] = None,
        required_metrics: Optional[Sequence[str]] = None,
    ) -> None:
        self.src_file = src_file
        self.vpy_src = vpy_src
        self.frames_count = int(frames_count)
        self.skip = int(skip)
        self.av1an_temp = av1an_temp
        self.fastpass_out = fastpass_out
        self.ssimu2_log = ssimu2_log
        self.vs_source = vs_source
        self.verbose = verbose
        self.scene_ranges = scene_ranges or []
        self._chunks_by_index: Optional[Dict[int, Dict[str, Any]]] = None
        self._scene_bitrate_cache: Dict[int, float] = {}
        self._video_bitrate_avg: Optional[float] = None
        self._fps: Optional[float] = None
        self._ssimu2_skip: Optional[int] = None
        self._ssimu2_scores: Optional[List[float]] = None
        self._ssimu2_avg_all: Optional[float] = None
        self._ssimu2_scene_avg_cache: Dict[int, float] = {}
        self._ssimu2_scene_p5_cache: Dict[int, float] = {}
        self.nvof_cache = None

    def _load_chunks(self) -> Dict[int, Dict[str, Any]]:
        if self._chunks_by_index is None:
            chunks = load_av1an_chunks(self.av1an_temp)
            by_index: Dict[int, Dict[str, Any]] = {}
            for ch in chunks:
                if not isinstance(ch, dict) or "index" not in ch:
                    raise ValueError("Invalid chunks.json entry (missing index).")
                by_index[int(ch["index"])] = ch
            if not by_index:
                raise ValueError("No chunks found in chunks.json.")
            if self._fps is None:
                first = by_index[min(by_index.keys())]
                if "frame_rate" not in first:
                    raise ValueError("chunks.json missing frame_rate field.")
                self._fps = parse_frame_rate(first["frame_rate"])
            self._chunks_by_index = by_index
        return self._chunks_by_index

    def _get_fps(self) -> float:
        if self._fps is None:
            self._load_chunks()
        if not self._fps or self._fps <= 0:
            raise ValueError("Invalid fps derived from chunks.json.")
        return float(self._fps)

    def _load_ssimu2(self) -> Tuple[int, List[float]]:
        if self._ssimu2_skip is None or self._ssimu2_scores is None:
            ensure_exists(self.ssimu2_log, "SSIMU2 log")
            skip, scores = parse_ssimu2_log(self.ssimu2_log)
            self._ssimu2_skip = int(skip)
            self._ssimu2_scores = list(scores)
        return int(self._ssimu2_skip), self._ssimu2_scores

    def ssimu2_scene_avg(self, scene_index: int, st: int, en: int) -> float:
        if scene_index in self._ssimu2_scene_avg_cache:
            return float(self._ssimu2_scene_avg_cache[scene_index])
        skip, scores = self._load_ssimu2()
        scene_scores = slice_samples_for_scene(scores, st, en, skip)
        if not scene_scores:
            raise ValueError(f"No SSIMU2 samples for scene {scene_index}.")
        avg = sum(scene_scores) / len(scene_scores)
        self._ssimu2_scene_avg_cache[scene_index] = float(avg)
        return float(avg)

    def ssimu2_scene_p5(self, scene_index: int, st: int, en: int) -> float:
        if scene_index in self._ssimu2_scene_p5_cache:
            return float(self._ssimu2_scene_p5_cache[scene_index])
        skip, scores = self._load_ssimu2()
        scene_scores = slice_samples_for_scene(scores, st, en, skip)
        if not scene_scores:
            raise ValueError(f"No SSIMU2 samples for scene {scene_index}.")
        _avg, p5, _p95 = calc_stats(scene_scores)
        self._ssimu2_scene_p5_cache[scene_index] = float(p5)
        return float(p5)

    def ssimu2_avg_all(self) -> float:
        if self._ssimu2_avg_all is None:
            _skip, scores = self._load_ssimu2()
            if not scores:
                raise ValueError("No SSIMU2 samples for global average.")
            self._ssimu2_avg_all = sum(scores) / len(scores)
        return float(self._ssimu2_avg_all)

    def scene_bitrate(self, scene_index: int) -> float:
        if scene_index in self._scene_bitrate_cache:
            return self._scene_bitrate_cache[scene_index]

        chunks = self._load_chunks()
        if scene_index not in chunks:
            raise KeyError(f"Chunk index {scene_index} not found in chunks.json.")
        ch = chunks[scene_index]

        try:
            st = int(ch["start_frame"])
            en = int(ch["end_frame"])
        except Exception as e:
            raise ValueError(f"Invalid chunk frame range for index {scene_index}: {e}") from e

        fps = parse_frame_rate(ch.get("frame_rate", self._get_fps()))
        duration = (en - st) / fps if fps else 0.0
        if duration <= 0:
            raise ValueError(f"Invalid duration for chunk {scene_index}: {st}..{en}, fps={fps}")

        ivf_path = self.av1an_temp / "encode" / f"{scene_index:05d}.ivf"
        ensure_exists(ivf_path, "av1an encode chunk")
        bytes_size = ivf_path.stat().st_size
        bitrate_kbps = (bytes_size * 8.0) / duration / 1000.0
        self._scene_bitrate_cache[scene_index] = bitrate_kbps
        return bitrate_kbps

    def video_bitrate_avg(self) -> float:
        if self._video_bitrate_avg is None:
            ensure_exists(self.fastpass_out, "Fast-pass output")
            bytes_size = self.fastpass_out.stat().st_size
            fps = self._get_fps()
            duration = self.frames_count / fps if fps else 0.0
            if duration <= 0:
                raise ValueError(f"Invalid video duration: frames={self.frames_count}, fps={fps}")
            self._video_bitrate_avg = (bytes_size * 8.0) / duration / 1000.0
        return float(self._video_bitrate_avg)

    def scene_bitrate_ratio(self, scene_index: int) -> float:
        avg = self.video_bitrate_avg()
        if avg <= 0:
            raise ValueError("Invalid average bitrate for ratio calculation.")
        return self.scene_bitrate(scene_index) / avg

class SceneMetrics:
    """Scene-scoped metric accessors for rules."""

    def __init__(self, state: RuleMetricsState, scene_index: int, st: int, en: int) -> None:
        self.state = state
        self.scene_index = int(scene_index)
        self.st = int(st)
        self.en = int(en)

    def metric(self, name: str) -> float:
        key = str(name).strip().lower()
        fn = registry.get(key)
        return fn(self)


class VideoParamEditor:
    """Helper to read and change encoder params for a scene."""

    def __init__(
        self,
        tokens: List[str],
        *,
        scene_index: int,
        start_frame: int,
        end_frame: int,
        rule_name: str,
        verbose: bool,
    ) -> None:
        self.tokens = tokens
        self.scene_index = int(scene_index)
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)
        self.rule_name = rule_name
        self.verbose = verbose

    def _format_log_value(self, value: Any) -> str:
        if value is None:
            return "<missing>"
        if value is True:
            return "<flag>"
        return str(value)

    def _rule_origin(self) -> Optional[str]:
        try:
            frame = sys._getframe()
        except Exception:
            return None
        while frame:
            if frame.f_code.co_filename == self.rule_name:
                return f"{self.rule_name}:{frame.f_lineno}"
            frame = frame.f_back
        return None

    def _log_change(self, action: str, key: str, old: Any, new: Any) -> None:
        if not self.verbose:
            return
        origin = self._rule_origin()
        origin_msg = f" ({origin})" if origin else ""
        print(
            f"[rules] scene {self.scene_index} [{self.start_frame}:{self.end_frame}] "
            f"{action} {key} {self._format_log_value(old)} -> {self._format_log_value(new)}{origin_msg}"
        )

    def param(self, name: str) -> Any:
        key = normalize_param_key(name)
        loc = find_last_option(self.tokens, key)
        if loc is None:
            return None
        k_idx, has_val = loc
        if not has_val:
            return True
        return coerce_param_value(self.tokens[k_idx + 1])

    def sparam(self, name: str, value: Any = None) -> None:
        key = normalize_param_key(name)
        old_val = self.param(key)
        val_token: Optional[str]
        if value is None or (isinstance(value, bool) and value is True):
            val_token = None
        else:
            val_token = format_param_value(value)

        loc = find_last_option(self.tokens, key)
        if loc is None:
            self.tokens.append(key)
            if val_token is not None:
                self.tokens.append(val_token)
        else:
            k_idx, has_val = loc
            if val_token is None:
                if has_val:
                    del self.tokens[k_idx + 1]
            else:
                if has_val:
                    self.tokens[k_idx + 1] = val_token
                else:
                    self.tokens.insert(k_idx + 1, val_token)

        new_val = self.param(key)
        self._log_change("set", key, old_val, new_val)

    def cparam(self, name: str, delta: Any) -> None:
        key = normalize_param_key(name)
        old_val = self.param(key)
        cur_num = parse_numeric_value(old_val)
        delta_num = parse_numeric_value(delta)
        if cur_num is None or delta_num is None:
            eprint(f"[warn] cparam skipped for {key}: non-numeric or missing value ({old_val}).")
            return

        new_val = cur_num + delta_num
        val_token = format_param_value(new_val)
        loc = find_last_option(self.tokens, key)
        if loc is None:
            self.tokens.append(key)
            self.tokens.append(val_token)
        else:
            k_idx, has_val = loc
            if has_val:
                self.tokens[k_idx + 1] = val_token
            else:
                self.tokens.insert(k_idx + 1, val_token)

        self._log_change("change", key, old_val, new_val)


def validate_scenes_with_overrides(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    scenes = obj.get("scenes") or obj.get("split_scenes") or []
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Invalid scenes.json: expected non-empty 'scenes' or 'split_scenes' list")

    for i, s in enumerate(scenes):
        if not isinstance(s, dict):
            raise ValueError(f"Invalid scene entry at index {i}: expected object")
        if "start_frame" not in s or "end_frame" not in s:
            raise ValueError(f"Invalid scene entry at index {i}: missing start_frame/end_frame")
        zo = s.get("zone_overrides")
        if not isinstance(zo, dict) or "video_params" not in zo:
            raise ValueError(f"Scene {i} missing zone_overrides/video_params (rules require Stage 4 output).")
        if not isinstance(zo["video_params"], list):
            raise ValueError(f"Scene {i} zone_overrides.video_params must be a list.")

    return scenes


def check_rules_prereqs(required_metrics: Sequence[str], av1an_temp: Path, fastpass_out: Path, ssimu2_log: Path) -> None:
    """Validate that required rule metrics artifacts are available."""
    if not required_metrics:
        return
    normalized = [m.strip().lower().rstrip("*") for m in required_metrics if m and m.strip()]

    if any(m.startswith("scene_bitrate") for m in normalized):
        chunks_path = av1an_temp / "chunks.json"
        encode_dir = av1an_temp / "encode"
        if not chunks_path.exists() or not encode_dir.exists():
            raise RuntimeError(
                "Rules require scene_bitrate*, but av1an temp artifacts are missing. "
                "Run fast-pass with --keep and keep access to av1an_temp."
            )
        ensure_exists(fastpass_out, "Fast-pass output")

    if any(m.startswith("ssimu2") for m in normalized):
        ensure_exists(ssimu2_log, "SSIMU2 log")
        if not is_valid_ssimu2_log(ssimu2_log):
            raise RuntimeError(f"Rules require ssimu2 metrics, but SSIMU2 log is invalid: {ssimu2_log}")


def apply_rules_to_scenes(
    *,
    base_obj: Dict[str, Any],
    compiled_rules: Any,
    rule_name: str,
    metrics_state: RuleMetricsState,
    verbose: bool,
    rule_pass: int = 2,
) -> Dict[str, Any]:
    scenes = validate_scenes_with_overrides(base_obj)
    frames = int(base_obj.get("frames", scenes[-1]["end_frame"] if scenes else 0))
    scene_count = len(scenes)
    fps = metrics_state._get_fps()

    updated: List[Dict[str, Any]] = []
    for i, s in enumerate(scenes):
        st = int(s["start_frame"])
        en = int(s["end_frame"])
        zone = dict(s["zone_overrides"])
        video_params = list(zone["video_params"])

        def rule_log(*args: Any) -> None:
            if not verbose:
                return
            try:
                frame = sys._getframe()
            except Exception:
                frame = None
            origin = None
            while frame:
                if frame.f_code.co_filename == rule_name:
                    origin = f"{rule_name}:{frame.f_lineno}"
                    break
                frame = frame.f_back
            origin_msg = f" ({origin})" if origin else ""
            msg = " ".join(str(a) for a in args)
            print(f"[rules] scene {i} [{st}:{en}] {msg}{origin_msg}")

        editor = VideoParamEditor(
            video_params,
            scene_index=i,
            start_frame=st,
            end_frame=en,
            rule_name=rule_name,
            verbose=verbose,
        )
        metrics = SceneMetrics(metrics_state, i, st, en)

        def metric_at(scene_index: int, name: str) -> float:
            idx = int(scene_index)
            if idx < 0 or idx >= scene_count:
                raise IndexError(f"Scene index out of range: {idx}")
            st_use = int(scenes[idx]["start_frame"])
            en_use = int(scenes[idx]["end_frame"])
            return SceneMetrics(metrics_state, idx, st_use, en_use).metric(name)

        def require(_name: str) -> None:
            return

        ctx_globals = {
            "__builtins__": __builtins__,
            "metric": metrics.metric,
            "metric_at": metric_at,
            "param": editor.param,
            "sparam": editor.sparam,
            "cparam": editor.cparam,
            "log": rule_log,
            "require": require,
            "rule_pass": int(rule_pass),
            "RULE_PASS": int(rule_pass),
            "scene_index": i,
            "scene_start": st,
            "scene_end": en,
            "scene_len": max(0, en - st),
            "scene_seconds": ((en - st) / fps) if fps > 0 else 0.0,
            "scene_fps": fps,
            "scene_count": scene_count,
        }
        ctx_locals: Dict[str, Any] = ctx_globals

        try:
            exec(compiled_rules, ctx_globals, ctx_locals)
        except Exception as exc:
            eprint(f"[error] rules failed for scene {i} [{st}:{en}]: {exc}")
            raise

        normalize_video_params(video_params)
        zone["video_params"] = video_params
        updated.append({
            "start_frame": st,
            "end_frame": en,
            "zone_overrides": zone,
        })

    return {"frames": frames, "scenes": updated, "split_scenes": [dict(x) for x in updated]}
