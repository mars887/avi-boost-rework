from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from utils.pipeline_runtime import ROOT_DIR
from utils.runner_state import (
    STAGE_ATTACHMENTS,
    STAGE_AUDIO,
    STAGE_AUTOBOOST_PSD_SCENE,
    STAGE_AUTOBOOST_SCENE,
    STAGE_DEMUX,
    STAGE_FASTPASS,
    STAGE_HDR_PATCH,
    STAGE_ITEM,
    STAGE_MAINPASS,
    STAGE_MUX,
    STAGE_SSIMU2,
    STAGE_VERIFY,
    STAGE_ZONE_EDIT,
)

STAGE_BANK_CONFIG_FILE = ROOT_DIR / "StagesBankTree.toml"


@dataclass(frozen=True)
class StageBankStage:
    cost: int = 1
    priority: int = 2
    requires: Tuple[str, ...] = ()


@dataclass(frozen=True)
class StageBankConfig:
    capacity: int
    max_active_plans: int
    max_running_stages: int
    stages: Dict[str, StageBankStage]

    def stage_cost(self, stage: str) -> int:
        return max(1, int(self.stages.get(stage, StageBankStage()).cost))

    def stage_priority(self, stage: str) -> int:
        return max(1, int(self.stages.get(stage, StageBankStage()).priority))

    def stage_requires(self, stage: str) -> Tuple[str, ...]:
        return tuple(self.stages.get(stage, StageBankStage()).requires)


KNOWN_STAGE_NAMES = {
    STAGE_DEMUX,
    STAGE_ATTACHMENTS,
    STAGE_AUTOBOOST_SCENE,
    STAGE_AUTOBOOST_PSD_SCENE,
    STAGE_FASTPASS,
    STAGE_SSIMU2,
    STAGE_ZONE_EDIT,
    STAGE_HDR_PATCH,
    STAGE_MAINPASS,
    STAGE_AUDIO,
    STAGE_VERIFY,
    STAGE_MUX,
}


DEFAULT_STAGE_BANK = StageBankConfig(
    capacity=10,
    max_active_plans=3,
    max_running_stages=5,
    stages={
        STAGE_DEMUX: StageBankStage(cost=2),
        STAGE_ATTACHMENTS: StageBankStage(cost=1, requires=(STAGE_DEMUX,)),
        STAGE_AUTOBOOST_SCENE: StageBankStage(cost=3, requires=(STAGE_DEMUX,)),
        STAGE_AUTOBOOST_PSD_SCENE: StageBankStage(cost=4, requires=(STAGE_DEMUX,)),
        STAGE_FASTPASS: StageBankStage(
            cost=10,
            priority=1,
            requires=(STAGE_AUTOBOOST_SCENE, STAGE_AUTOBOOST_PSD_SCENE),
        ),
        STAGE_SSIMU2: StageBankStage(cost=5, requires=(STAGE_FASTPASS,)),
        STAGE_ZONE_EDIT: StageBankStage(
            cost=2,
            requires=(STAGE_SSIMU2, STAGE_AUTOBOOST_SCENE, STAGE_AUTOBOOST_PSD_SCENE),
        ),
        STAGE_HDR_PATCH: StageBankStage(cost=2, requires=(STAGE_ZONE_EDIT,)),
        STAGE_MAINPASS: StageBankStage(cost=10, priority=1, requires=(STAGE_HDR_PATCH,)),
        STAGE_AUDIO: StageBankStage(cost=2, requires=(STAGE_DEMUX,)),
        STAGE_VERIFY: StageBankStage(
            cost=1,
            priority=3,
            requires=(STAGE_ATTACHMENTS, STAGE_AUDIO, STAGE_MAINPASS),
        ),
        STAGE_MUX: StageBankStage(cost=3, priority=3, requires=(STAGE_VERIFY,)),
    },
)


def _validated_stage_bank_config(
    *,
    capacity: int,
    max_active_plans: int,
    max_running_stages: int,
    stages: Dict[str, StageBankStage],
) -> StageBankConfig:
    for stage_name, stage in stages.items():
        if int(stage.cost) > capacity:
            raise RuntimeError(f"stage cost exceeds bank capacity: {stage_name}")
    _validate_stage_bank_acyclic(stages)
    return StageBankConfig(
        capacity=capacity,
        max_active_plans=max_active_plans,
        max_running_stages=max_running_stages,
        stages=stages,
    )


def _validate_stage_bank_acyclic(stages: Dict[str, StageBankStage]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(stage_name: str, path: List[str]) -> None:
        if stage_name in visited:
            return
        if stage_name in visiting:
            try:
                start = path.index(stage_name)
            except ValueError:
                start = 0
            cycle = path[start:] + [stage_name]
            raise RuntimeError(f"stage bank dependency cycle: {' -> '.join(cycle)}")
        visiting.add(stage_name)
        path.append(stage_name)
        for dependency in stages.get(stage_name, StageBankStage()).requires:
            if dependency in stages:
                visit(dependency, path)
        path.pop()
        visiting.remove(stage_name)
        visited.add(stage_name)

    for stage_name in stages:
        visit(stage_name, [])


def load_stage_bank_config(path: Path = STAGE_BANK_CONFIG_FILE) -> StageBankConfig:
    if not path.exists():
        return DEFAULT_STAGE_BANK
    try:
        with path.open("rb") as fh:
            payload = tomllib.load(fh)
    except Exception as exc:
        raise RuntimeError(f"failed to read stage bank config {path}: {exc}") from exc

    bank = dict(payload.get("bank") or {})
    try:
        capacity = int(bank.get("capacity", DEFAULT_STAGE_BANK.capacity))
    except Exception as exc:
        raise RuntimeError("stage bank capacity must be an integer") from exc
    if capacity <= 0:
        raise RuntimeError("stage bank capacity must be greater than zero")
    try:
        max_active_plans = int(bank.get("max_active_plans", DEFAULT_STAGE_BANK.max_active_plans))
    except Exception as exc:
        raise RuntimeError("stage bank max_active_plans must be an integer") from exc
    if max_active_plans <= 0:
        raise RuntimeError("stage bank max_active_plans must be greater than zero")
    try:
        max_running_stages = int(bank.get("max_running_stages", DEFAULT_STAGE_BANK.max_running_stages))
    except Exception as exc:
        raise RuntimeError("stage bank max_running_stages must be an integer") from exc
    if max_running_stages <= 0:
        raise RuntimeError("stage bank max_running_stages must be greater than zero")

    raw_stages = payload.get("stages")
    if raw_stages is None:
        return _validated_stage_bank_config(
            capacity=capacity,
            max_active_plans=max_active_plans,
            max_running_stages=max_running_stages,
            stages=dict(DEFAULT_STAGE_BANK.stages),
        )
    if not isinstance(raw_stages, dict):
        raise RuntimeError("stage bank [stages] section must be a table")

    stages: Dict[str, StageBankStage] = {}
    for name, raw_stage in raw_stages.items():
        stage_name = str(name or "").strip()
        if stage_name not in KNOWN_STAGE_NAMES:
            raise RuntimeError(f"unknown stage in stage bank config: {stage_name}")
        if not isinstance(raw_stage, dict):
            raise RuntimeError(f"stage config must be a table: {stage_name}")
        try:
            cost = int(raw_stage.get("cost", DEFAULT_STAGE_BANK.stage_cost(stage_name)))
        except Exception as exc:
            raise RuntimeError(f"stage cost must be an integer: {stage_name}") from exc
        if cost <= 0:
            raise RuntimeError(f"stage cost must be greater than zero: {stage_name}")
        if cost > capacity:
            raise RuntimeError(f"stage cost exceeds bank capacity: {stage_name}")
        try:
            priority = int(raw_stage.get("priority", DEFAULT_STAGE_BANK.stage_priority(stage_name)))
        except Exception as exc:
            raise RuntimeError(f"stage priority must be an integer: {stage_name}") from exc
        if priority <= 0:
            raise RuntimeError(f"stage priority must be greater than zero: {stage_name}")
        raw_requires = raw_stage.get("requires", DEFAULT_STAGE_BANK.stage_requires(stage_name))
        if not isinstance(raw_requires, (list, tuple)):
            raise RuntimeError(f"stage requires must be an array: {stage_name}")
        requires = tuple(str(item or "").strip() for item in raw_requires if str(item or "").strip())
        unknown_requires = [item for item in requires if item not in KNOWN_STAGE_NAMES]
        if unknown_requires:
            raise RuntimeError(f"unknown dependency for {stage_name}: {', '.join(unknown_requires)}")
        stages[stage_name] = StageBankStage(cost=cost, priority=priority, requires=requires)

    for stage_name, default_stage in DEFAULT_STAGE_BANK.stages.items():
        stages.setdefault(stage_name, default_stage)
    return _validated_stage_bank_config(
        capacity=capacity,
        max_active_plans=max_active_plans,
        max_running_stages=max_running_stages,
        stages=stages,
    )


def effective_stage_dependencies(stage: str, stage_names: List[str], config: StageBankConfig) -> List[str]:
    available = set(stage_names)
    return [dependency for dependency in config.stage_requires(stage) if dependency in available]


def downstream_stage_names(stage: str, stage_names: List[str], config: StageBankConfig) -> List[str]:
    available = list(stage_names)
    downstream: List[str] = []
    queue = [stage]
    seen = {stage}
    while queue:
        dependency = queue.pop(0)
        for candidate in available:
            if candidate in seen:
                continue
            if dependency not in effective_stage_dependencies(candidate, available, config):
                continue
            seen.add(candidate)
            downstream.append(candidate)
            queue.append(candidate)
    return downstream
