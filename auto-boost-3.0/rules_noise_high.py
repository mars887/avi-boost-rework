from __future__ import annotations


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def round_step(value: float, step: float = 0.25) -> float:
    return round(value / step) * step


def safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def get_metric(name: str, idx=None) -> float:
    if idx is None:
        return metric(name)
    return metric_at(int(idx), name)


def stage1_metrics(idx=None):
    br_kbps = get_metric("scene_bitrate", idx)
    br_mbit = br_kbps / 1000.0
    br_ratio = get_metric("scene_bitrate_ratio", idx)

    ssim = get_metric("ssimu2_p5", idx)
    ssim_avg = metric("ssimu2_avg")
    if ssim_avg is None or ssim_avg <= 0:
        ssim_avg = ssim

    # ssim_gap = ssim_avg - ssim
    # if br_mbit <= 6.0 and ssim_gap > 0:
    #     deadzone = 5.0
    #     effective_avg = ssim + max(0.0, ssim_gap - deadzone)
    #     ssim_drop = (effective_avg - ssim) / effective_avg if effective_avg > 0 else 0.0
    #     ssim_score = clamp((ssim_drop - 0.08) / 0.15)
    # else:
    #     ssim_drop = (ssim_avg - ssim) / ssim_avg if ssim_avg > 0 else 0.0
    #     ssim_score = clamp((ssim_drop - 0.08) / 0.15)

    ssim_gap = ssim_avg - ssim
    deadzone = 5.0
    if br_mbit <= 6.0 and ssim_gap > deadzone:
        effective_avg = ssim + max(0.0, ssim_gap - deadzone / 2)
        ssim_drop = (effective_avg - ssim) / effective_avg if effective_avg > 0 else 0.0
        ssim_score = clamp((ssim_drop - 0.08) / 0.12)
    else:
        ssim_drop = (ssim_avg - ssim) / ssim_avg if ssim_avg > 0 else 0.0
        ssim_score = clamp((ssim_drop - 0.08) / 0.15)


    br_score_abs = clamp((br_mbit - 6.0) / 6.0)
    br_score_ratio = clamp((br_ratio - 2.0) / 4.0)
    br_score = max(br_score_abs, br_score_ratio)

    return {
        "br_kbps": br_kbps,
        "br_mbit": br_mbit,
        "br_ratio": br_ratio,
        "ssim": ssim,
        "ssim_avg": ssim_avg,
        "ssim_drop": ssim_drop,
        "ssim_score": ssim_score,
        "br_score_abs": br_score_abs,
        "br_score_ratio": br_score_ratio,
        "br_score": br_score,
    }


def is_false_positive(stage) -> bool:
    return stage["br_ratio"] < 1.1 and stage["ssim"] >= stage["ssim_avg"] * 1.02


def is_candidate(stage) -> bool:
    return stage["br_score"] >= 0.25 and stage["ssim_score"] >= 0.20


def short_profile(duration_s: float):
    if duration_s < 2.0:
        return 27.0, 0.6
    if duration_s < 3.0:
        return 30.0, 0.75
    if duration_s < 4.0:
        return 32.0, 0.85
    return None


def compute_conf(idx=None, stage=None):
    st = stage if stage is not None else stage1_metrics(idx)
    if is_false_positive(st):
        return None

    noise = get_metric("scene_noise_sigma", idx)
    noise_avg = metric("global_noise_sigma")
    if noise_avg is None or noise_avg <= 0:
        noise_avg = noise
    noise_ratio = noise / noise_avg if noise_avg > 0 else 1.0
    noise_score = clamp((noise_ratio - 1.2) / 0.6)

    nvof = get_metric("scene_nvof_avg", idx)
    nvof_avg = metric("global_nvof_avg")
    if nvof_avg is None or nvof_avg <= 0:
        nvof_avg = nvof
    nvof_ratio = nvof / nvof_avg if nvof_avg > 0 else 1.0
    # nvof_score = clamp((nvof_ratio - 1.1) / 0.5)
    nvof_score = 0.0

    base_conf = clamp(0.4 * st["br_score"] + 0.6 * st["ssim_score"])
    conf = base_conf * (0.6 + 0.4 * noise_score)
    # conf = clamp(conf + 0.1 * nvof_score)
    conf = clamp(conf)

    return {
        "stage": st,
        "conf": conf,
        "base_conf": base_conf,
        "noise_ratio": noise_ratio,
        "noise_score": noise_score,
        "nvof_ratio": nvof_ratio,
        "nvof_score": nvof_score,
    }


st = stage1_metrics()

if rule_pass == 1:
    require("scene_noise_sigma")
    require("global_noise_sigma")
    require("scene_nvof_avg")
    require("global_nvof_avg")
else:
    cur = compute_conf(stage=st)

    if not cur:
        globals()["_prev_noise_conf"] = 0.0
        globals()["_prev_noise_strong"] = False
        globals()["_prev_noise_apply"] = False
    else:
        conf_raw = cur["conf"]
        log(
            f"pre br={st['br_mbit']:.2f}mbit ratio={st['br_ratio']:.2f} "
            f"ssim_p5={st['ssim']:.2f} drop={st['ssim_drop']:.3f} "
            f"ssim_score={st['ssim_score']:.2f} br_score={st['br_score']:.2f}"
        )
        log(
            f"conf={conf_raw:.3f} base={cur['base_conf']:.3f} "
            f"noise_ratio={cur['noise_ratio']:.2f} noise_score={cur['noise_score']:.2f} "
            f"nvof_ratio={cur['nvof_ratio']:.2f} nvof_score={cur['nvof_score']:.2f}"
        )

        prev_strong = bool(globals().get("_prev_noise_strong"))
        prev_applied = bool(globals().get("_prev_noise_apply"))

        next_applied = False
        if scene_index + 1 < scene_count:
            try:
                nxt = compute_conf(scene_index + 1)
                next_applied = bool(nxt and nxt["conf"] >= 0.15)
            except Exception:
                next_applied = False

        duration_s = float(scene_seconds) if scene_seconds is not None else 0.0
        profile = short_profile(duration_s)
        apply_short = profile is not None and not prev_applied and not next_applied

        conf = conf_raw
        target_full = 20.0 + 0.6 * st["br_mbit"]
        if target_full < 22.0:
            target_full = 22.0
        elif target_full > 33.0:
            target_full = 33.0

        if apply_short:
            limit, mult = profile
            conf = conf_raw * mult
            if target_full > limit:
                target_full = limit
            log(f"short_scene {duration_s:.2f}s cap={limit} mult={mult}")

        cur_crf = safe_float(param("--crf"), default=None)
        if cur_crf is None:
            cur_crf = 22.0

        base_ref = max(22.0, cur_crf)
        if prev_strong:
            target_full = max(target_full, 24.0)
            neighbor_bonus = 2.0
            proposed = base_ref + conf * (target_full - base_ref)
            proposed += neighbor_bonus * (1.0 - conf)
        else:
            proposed = base_ref + conf * (target_full - base_ref)

        hard_cap = target_full if apply_short else 33.0
        if proposed > hard_cap:
            proposed = hard_cap

        proposed = round_step(proposed, 0.25)

        delta = proposed - cur_crf
        if conf_raw < 0.15:
            log(f"skip conf={conf_raw:.3f} < 0.15")
        elif delta >= 0.25:
            log(f"apply crf {cur_crf:.2f} -> {proposed:.2f} (d={delta:+.2f})")
            cparam("--crf", delta)
        else:
            log(f"skip delta {delta:+.2f}")

        globals()["_prev_noise_conf"] = conf_raw
        globals()["_prev_noise_strong"] = bool(conf_raw >= 0.9)
        globals()["_prev_noise_apply"] = bool(conf_raw >= 0.15)
