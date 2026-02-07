"""Built-in metric implementations registered in the registry."""

from __future__ import annotations

from ab_registry import metric


@metric("scene_bitrate")
def metric_scene_bitrate(metrics) -> float:
    return metrics.state.scene_bitrate(metrics.scene_index)


@metric("scene_bitrate_ratio")
def metric_scene_bitrate_ratio(metrics) -> float:
    return metrics.state.scene_bitrate_ratio(metrics.scene_index)


@metric("ssimu2")
def metric_ssimu2(metrics) -> float:
    return metrics.state.ssimu2_scene_avg(metrics.scene_index, metrics.st, metrics.en)


@metric("ssimu2_p5")
def metric_ssimu2_p5(metrics) -> float:
    return metrics.state.ssimu2_scene_p5(metrics.scene_index, metrics.st, metrics.en)


@metric("ssimu2_avg")
def metric_ssimu2_avg(metrics) -> float:
    return metrics.state.ssimu2_avg_all()
