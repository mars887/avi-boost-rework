"""NVOF/noise/luma metric implementations registered in the registry."""

from __future__ import annotations

from ab_registry import metric


def _cache(metrics):
    cache = getattr(metrics.state, "nvof_cache", None)
    if cache is None:
        raise RuntimeError("NVOF metrics requested but not computed. Run rules pass with require().")
    return cache


@metric("scene_nvof_avg")
def metric_scene_nvof_avg(metrics) -> float:
    cache = _cache(metrics)
    return cache.scene_avg("nvof_mean", metrics.st, metrics.en)


@metric("global_nvof_avg")
def metric_global_nvof_avg(metrics) -> float:
    cache = _cache(metrics)
    return cache.global_avg("nvof_mean")


@metric("scene_luma_avg")
def metric_scene_luma_avg(metrics) -> float:
    cache = _cache(metrics)
    return cache.scene_avg("luma_mean", metrics.st, metrics.en)


@metric("global_luma_avg")
def metric_global_luma_avg(metrics) -> float:
    cache = _cache(metrics)
    return cache.global_avg("luma_mean")


@metric("scene_luma_p25")
def metric_scene_luma_p25(metrics) -> float:
    cache = _cache(metrics)
    return cache.scene_avg("luma_p25", metrics.st, metrics.en)


@metric("global_luma_p25")
def metric_global_luma_p25(metrics) -> float:
    cache = _cache(metrics)
    return cache.global_avg("luma_p25")


@metric("scene_luma_p75")
def metric_scene_luma_p75(metrics) -> float:
    cache = _cache(metrics)
    return cache.scene_avg("luma_p75", metrics.st, metrics.en)


@metric("global_luma_p75")
def metric_global_luma_p75(metrics) -> float:
    cache = _cache(metrics)
    return cache.global_avg("luma_p75")


@metric("scene_noise_sigma")
def metric_scene_noise_sigma(metrics) -> float:
    cache = _cache(metrics)
    return cache.scene_avg("noise_sigma", metrics.st, metrics.en)


@metric("global_noise_sigma")
def metric_global_noise_sigma(metrics) -> float:
    cache = _cache(metrics)
    return cache.global_avg("noise_sigma")


@metric("scene_grain_ratio")
def metric_scene_grain_ratio(metrics) -> float:
    cache = _cache(metrics)
    return cache.scene_avg("grain_ratio", metrics.st, metrics.en)


@metric("global_grain_ratio")
def metric_global_grain_ratio(metrics) -> float:
    cache = _cache(metrics)
    return cache.global_avg("grain_ratio")


@metric("luma_avg")
def metric_luma_avg_alias(metrics) -> float:
    return metric_scene_luma_avg(metrics)


@metric("luma_mean")
def metric_luma_mean_alias(metrics) -> float:
    return metric_scene_luma_avg(metrics)


@metric("luma_p25")
def metric_luma_p25_alias(metrics) -> float:
    return metric_scene_luma_p25(metrics)


@metric("luma_p75")
def metric_luma_p75_alias(metrics) -> float:
    return metric_scene_luma_p75(metrics)


@metric("luma_ratio")
def metric_luma_ratio(metrics) -> float:
    cache = _cache(metrics)
    scene_avg = cache.scene_avg("luma_mean", metrics.st, metrics.en)
    global_avg = cache.global_avg("luma_mean")
    if global_avg <= 0:
        raise ValueError("Invalid global luma average for ratio.")
    return scene_avg / global_avg


@metric("nvof_mean")
def metric_nvof_mean_alias(metrics) -> float:
    return metric_scene_nvof_avg(metrics)


@metric("noise_sigma")
def metric_noise_sigma_alias(metrics) -> float:
    return metric_scene_noise_sigma(metrics)


@metric("grain_ratio")
def metric_grain_ratio_alias(metrics) -> float:
    return metric_scene_grain_ratio(metrics)
