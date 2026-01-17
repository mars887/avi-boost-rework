#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined NVOF + noise-tiles CUDA metric (single NVDEC decode).
Design goal: avoid killing NVDEC by running two separate processes.

Core idea:
- Decode once with cv2.cudacodec.VideoReader (NVDEC).
- Run NVOF every frame at low-res (e.g. 960x540).
- Run noise-tiles every Nth frame at >=2K.
- Synchronize/harvest results only once per --sync-every frames (default 1).

Tested assumptions (based on your build probe):
- cv2.cuda_GpuMat.cudaPtr() exists and returns a device pointer.
- OpenCV built with cudacodec + CUDA + cuDNN.

Requirements:
- opencv-python built with CUDA (you already have)
- cupy (same CUDA major as driver/toolkit; you already use it in noise_tiles_cuda_v2.py)
"""

from __future__ import annotations

import os
import math
import sys
import json
# For safest OpenCV<->CuPy interop, prefer legacy null stream.
# Must be set before importing cupy.
os.environ.setdefault("CUPY_CUDA_PER_THREAD_DEFAULT_STREAM", "0")

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import cv2

import cupy as cp


# ---------------------------------------------------------------------------
# CUDA interop: cv2.cuda_GpuMat -> CuPy view (zero-copy)
# ---------------------------------------------------------------------------

def _cvtype_to_dtype(cv_type: int):
    depth = cv_type & 7
    if depth == cv2.CV_8U:
        return np.uint8
    if depth == cv2.CV_8S:
        return np.int8
    if depth == cv2.CV_16U:
        return np.uint16
    if depth == cv2.CV_16S:
        return np.int16
    if depth == cv2.CV_32S:
        return np.int32
    if depth == cv2.CV_32F:
        return np.float32
    if depth == cv2.CV_64F:
        return np.float64
    raise ValueError(f"Unsupported cv depth={depth} for cv_type={cv_type}")


def gpumat_to_cupy_2d(gm: cv2.cuda_GpuMat, *, dtype=None) -> cp.ndarray:
    """
    Create a 2D CuPy view over a 1-channel cv2.cuda_GpuMat without copying.

    WARNING:
      - Only safe while gm is alive and its underlying allocation is not reallocated.
      - This function assumes gm is 1-channel.
    """
    if gm.empty():
        raise ValueError("gpumat_to_cupy_2d: gm is empty")
    if gm.channels() != 1:
        raise ValueError(f"gpumat_to_cupy_2d: expected 1-channel, got {gm.channels()}")

    w, h = gm.size()
    step_bytes = int(gm.step)  # bytes per row
    ptr = int(gm.cudaPtr())

    if dtype is None:
        dtype = _cvtype_to_dtype(gm.type())
    np_dtype = np.dtype(dtype)
    itemsize = int(np_dtype.itemsize)

    # total bytes in allocation slice we need
    mem_nbytes = step_bytes * int(h)

    umem = cp.cuda.UnownedMemory(ptr, mem_nbytes, gm)
    mptr = cp.cuda.MemoryPointer(umem, 0)
    arr = cp.ndarray(
        shape=(int(h), int(w)),
        dtype=np_dtype,
        memptr=mptr,
        strides=(step_bytes, itemsize),
        order="C",
    )
    return arr


# ---------------------------------------------------------------------------
# OpenCV CUDA helpers (mostly mirrored from your farne-nvof.py / farne-gpu.py)
# ---------------------------------------------------------------------------

def gpu_ensure(dst: cv2.cuda_GpuMat, size_wh: Tuple[int, int], cv_type: int):
    w, h = size_wh
    if dst.empty() or dst.size() != (w, h) or dst.type() != cv_type:
        dst.create(h, w, cv_type)


def gpu_copy(src: cv2.cuda_GpuMat, dst: cv2.cuda_GpuMat, stream=None):
    gpu_ensure(dst, src.size(), src.type())
    if stream is None:
        src.copyTo(dst)
    else:
        # Python bindings are most consistent with stream-first overload
        src.copyTo(stream, dst)


def gpu_convert_to(src: cv2.cuda_GpuMat,
                   dst: cv2.cuda_GpuMat,
                   rtype: int,
                   alpha: float | None = None,
                   beta: float = 0.0,
                   stream=None):
    gpu_ensure(dst, src.size(), rtype)
    if alpha is None:
        if stream is None:
            src.convertTo(rtype, dst)
        else:
            src.convertTo(rtype, stream, dst)
        return

    a = float(alpha)
    b = float(beta)
    if stream is None:
        src.convertTo(rtype, dst, a, b)
    else:
        src.convertTo(rtype, a, b, stream, dst)


def gpu_cvtColor(src: cv2.cuda_GpuMat, dst: cv2.cuda_GpuMat, code: int, dcn=0, stream=None):
    # dst must be preallocated with correct type/size
    if stream is None:
        return cv2.cuda.cvtColor(src, code, dst, dcn)
    else:
        return cv2.cuda.cvtColor(src, code, dst, dcn, stream)


def try_set_reader_format(reader, want_gray=True, want_8bit=True) -> bool:
    """
    Best-effort attempt to set cudacodec.VideoReader output ColorFormat/BitDepth.
    """
    cc = cv2.cudacodec

    def pick(*names):
        for n in names:
            if hasattr(cc, n):
                return getattr(cc, n)
        return None

    if want_gray:
        color = pick("ColorFormat_GRAY", "GRAY")
    else:
        color = pick("ColorFormat_BGR", "BGR")

    depth = None
    if want_8bit:
        depth = pick("BitDepth_DEPTH_8U", "BitDepth_8U", "DEPTH_8U")
        if depth is None and hasattr(cc, "BitDepth"):
            bd = cc.BitDepth
            for n in ("DEPTH_8U", "DEPTH_8"):
                if hasattr(bd, n):
                    depth = getattr(bd, n)
                    break

    try:
        if color is None:
            return False
        if depth is None:
            return bool(reader.set(color))
        return bool(reader.set(color, depth, False))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# NVOF creation (compat across OpenCV Python binding variants)
# ---------------------------------------------------------------------------

def _pick_attr(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def _make_nvof(size_wh: Tuple[int, int], *, perf: str, grid: int, temporal_hints: bool, gpu_id: int, stream=None):
    """Create NVIDIA Optical Flow session.

    Prefers the 2.0 API when available, falls back to 1.0. This matches the
    proven logic from your working farne-nvof.py.

    Returns: (nvof_object, version_str)
    """
    perf = perf.lower().strip()
    if grid not in (1, 2, 4):
        raise ValueError("--grid must be 1, 2 or 4")

    def _create_from_class(c):
        # perfPreset ints used by NVIDIA Optical Flow SDK:
        # SLOW=5, MEDIUM=10, FAST=20
        perf_map = {"slow": 5, "medium": 10, "fast": 20}
        perf_val = perf_map.get(perf, 20)

        def _enum(prefix: str, val: int):
            for name in (
                f"{prefix}_{val}",
                f"{prefix}_SIZE_{val}",
                f"{prefix}{val}",
            ):
                if hasattr(c, name):
                    return getattr(c, name)
            return val

        output_grid = _enum("NV_OF_OUTPUT_VECTOR_GRID_SIZE", grid)
        hint_grid = _enum("NV_OF_HINT_VECTOR_GRID_SIZE", grid)
        perf_preset = _enum("NV_OF_PERF_LEVEL", perf_val)

        # Binding variants:
        #  1) class with .create(size, **kwargs)
        #  2) function-style creator (callable)
        if hasattr(c, "create"):
            kwargs = dict(
                perfPreset=perf_preset,
                outputGridSize=output_grid,
                hintGridSize=hint_grid,
                enableTemporalHints=bool(temporal_hints),
                enableExternalHints=False,
                enableCostBuffer=False,
                gpuId=int(gpu_id),
            )
            if stream is not None:
                # Some builds accept streams at construction time.
                kwargs["inputStream"] = stream
                kwargs["outputStream"] = stream
            return c.create(size_wh, **kwargs)

        # Function-style creator: keep positional args minimal.
        return c(size_wh, perf_preset, output_grid, hint_grid, bool(temporal_hints), False, False, int(gpu_id))

    # Try 2.0
    cls = _pick_attr(cv2, "cuda_NvidiaOpticalFlow_2_0", "cuda_NvidiaOpticalFlow_2_0_create")
    if cls is None and hasattr(cv2, "cuda"):
        cls = _pick_attr(cv2.cuda, "NvidiaOpticalFlow_2_0", "NvidiaOpticalFlow_2_0_create")
    if cls is not None:
        try:
            return _create_from_class(cls), "2.0"
        except Exception:
            pass

    # Fall back to 1.0
    cls = _pick_attr(cv2, "cuda_NvidiaOpticalFlow_1_0", "cuda_NvidiaOpticalFlow_1_0_create")
    if cls is None and hasattr(cv2, "cuda"):
        cls = _pick_attr(cv2.cuda, "NvidiaOpticalFlow_1_0", "NvidiaOpticalFlow_1_0_create")
    if cls is None:
        raise RuntimeError(
            "NVIDIA Optical Flow is not available in your cv2 build. "
            "You need opencv_contrib cudaoptflow built with NVIDIA Optical Flow SDK enabled."
        )

    return _create_from_class(cls), "1.0"

# ---------------------------------------------------------------------------
# Noise tiles kernel (copied from noise_tiles_cuda_v2.py with minimal changes)
# ---------------------------------------------------------------------------

KERNEL = r"""
extern "C" __global__
void tile_noise_v2_u8(
    const unsigned char* __restrict__ img,
    int H, int W, int stride_bytes,
    int tile, int pix_step, int d2,
    float tile_sample_p,
    unsigned int seed, unsigned int frame_idx,
    float* __restrict__ out_sigma,
    float* __restrict__ out_grain,
    float* __restrict__ out_struct,
    float* __restrict__ out_mean,
    float* __restrict__ out_var,
    int* __restrict__ out_count
){
    // One CUDA block per tile
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bdx = blockDim.x;
    const int bdy = blockDim.y;
    const int tid = ty * bdx + tx;
    const int nthreads = bdx * bdy;

    const int tile_x = (int)blockIdx.x * tile;
    const int tile_y = (int)blockIdx.y * tile;

    const int tiles_x = gridDim.x;
    const int tile_id = (int)blockIdx.y * tiles_x + (int)blockIdx.x;

    // --- RNG: xorshift32 on (seed, frame_idx, tile_id)
    unsigned int x = seed ^ (frame_idx * 747796405u) ^ (tile_id * 2891336453u);
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    float u = (float)(x & 0x00FFFFFFu) * (1.0f / 16777216.0f);
    if (u > tile_sample_p) {
        if (tid == 0) {
            out_sigma[tile_id]  = -1.0f;
            out_grain[tile_id]  = -1.0f;
            out_struct[tile_id] = -1.0f;
            out_mean[tile_id]   = -1.0f;
            out_var[tile_id]    = -1.0f;
            out_count[tile_id]  = 0;
        }
        return;
    }

    // Need 3x3 for Laplacian AND +-d2 for scale-2 residual.
    const int r = d2 > 1 ? d2 : 1;

    float sumAbsLap = 0.0f;
    float sumR1_2   = 0.0f;
    float sumR2_2   = 0.0f;
    float sumStruct = 0.0f;
    unsigned long long sumI  = 0ULL;
    unsigned long long sumI2 = 0ULL;
    int count = 0;

    // Iterate samples inside tile with stride pix_step.
    for (int oy = ty * pix_step; oy < tile; oy += bdy * pix_step) {
        int y0 = tile_y + oy;
        if (y0 < r || y0 >= (H - r)) continue;

        const unsigned char* rowm1 = (const unsigned char*)((const char*)img + (y0 - 1) * stride_bytes);
        const unsigned char* row0  = (const unsigned char*)((const char*)img + (y0)     * stride_bytes);
        const unsigned char* rowp1 = (const unsigned char*)((const char*)img + (y0 + 1) * stride_bytes);

        const unsigned char* row_u2 = (const unsigned char*)((const char*)img + (y0 - d2) * stride_bytes);
        const unsigned char* row_d2 = (const unsigned char*)((const char*)img + (y0 + d2) * stride_bytes);

        for (int ox = tx * pix_step; ox < tile; ox += bdx * pix_step) {
            int x0 = tile_x + ox;
            if (x0 < r || x0 >= (W - r)) continue;

            // 3x3 neighborhood for Immerkaer Laplacian mask:
            // [ 1 -2  1
            //  -2  4 -2
            //   1 -2  1 ]
            int tl = (int)rowm1[x0 - 1];
            int tc = (int)rowm1[x0    ];
            int tr = (int)rowm1[x0 + 1];
            int ml = (int)row0 [x0 - 1];
            int mc = (int)row0 [x0    ];
            int mr = (int)row0 [x0 + 1];
            int bl = (int)rowp1[x0 - 1];
            int bc = (int)rowp1[x0    ];
            int br = (int)rowp1[x0 + 1];

            int lap = (tl - 2*tc + tr) + (-2*ml + 4*mc - 2*mr) + (bl - 2*bc + br);
            float absLap = (float)(lap < 0 ? -lap : lap);
            sumAbsLap += absLap;

            // Residual at distance 1: r1 = mc - mean4(neighbors at 1)
            float mean1 = 0.25f * (float)(ml + mr + tc + bc);
            float r1 = (float)mc - mean1;
            sumR1_2 += r1 * r1;

            // Residual at distance d2: r2 = mc - mean4(neighbors at d2)
            int l2 = (int)row0[x0 - d2];
            int r2p= (int)row0[x0 + d2];
            int u2 = (int)row_u2[x0];
            int d2v= (int)row_d2[x0];
            float mean2 = 0.25f * (float)(l2 + r2p + u2 + d2v);
            float r2 = (float)mc - mean2;
            sumR2_2 += r2 * r2;

            // Structure score (first-order gradients)
            float gx = (float)(mr - mc); if (gx < 0) gx = -gx;
            float gy = (float)(bc - mc); if (gy < 0) gy = -gy;
            sumStruct += gx + gy;

            sumI  += (unsigned long long)mc;
            sumI2 += (unsigned long long)(mc * mc);
            count += 1;
        }
    }

    __shared__ float sh_f0[256];
    __shared__ float sh_f1[256];
    __shared__ float sh_f2[256];
    __shared__ float sh_f3[256];
    __shared__ unsigned long long sh_u0[256];
    __shared__ unsigned long long sh_u1[256];
    __shared__ int sh_i0[256];

    sh_f0[tid] = sumAbsLap;
    sh_f1[tid] = sumR1_2;
    sh_f2[tid] = sumR2_2;
    sh_f3[tid] = sumStruct;
    sh_u0[tid] = sumI;
    sh_u1[tid] = sumI2;
    sh_i0[tid] = count;
    __syncthreads();

    for (int offset = nthreads >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sh_f0[tid] += sh_f0[tid + offset];
            sh_f1[tid] += sh_f1[tid + offset];
            sh_f2[tid] += sh_f2[tid + offset];
            sh_f3[tid] += sh_f3[tid + offset];
            sh_u0[tid] += sh_u0[tid + offset];
            sh_u1[tid] += sh_u1[tid + offset];
            sh_i0[tid] += sh_i0[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int n = sh_i0[0];
        if (n <= 0) {
            out_sigma[tile_id]  = -1.0f;
            out_grain[tile_id]  = -1.0f;
            out_struct[tile_id] = -1.0f;
            out_mean[tile_id]   = -1.0f;
            out_var[tile_id]    = -1.0f;
            out_count[tile_id]  = 0;
            return;
        }

        float meanAbsLap = sh_f0[0] / (float)n;
        // sigma = (sqrt(pi/2)/6) * mean(|L|)  (Immerkaer 1996)
        float sigma = meanAbsLap * 0.20888568955258338f;

        float e1 = sh_f1[0] / (float)n;
        float e2 = sh_f2[0] / (float)n;
        float grain = sqrtf(e2 / (e1 + 1e-12f));

        float structScore = sh_f3[0] / (float)n;

        float meanI = (float)sh_u0[0] / (float)n;
        float varI  = (float)sh_u1[0] / (float)n - meanI * meanI;

        out_sigma[tile_id]  = sigma;
        out_grain[tile_id]  = grain;
        out_struct[tile_id] = structScore;
        out_mean[tile_id]   = meanI;
        out_var[tile_id]    = varI;
        out_count[tile_id]  = n;
    }
}
"""


def pool_values(vals: np.ndarray, mode: str, trim_low: float, trim_high: float) -> float:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    if mode == "median":
        return float(np.median(vals))
    vals = np.sort(vals)
    lo = int(vals.size * trim_low)
    hi = int(vals.size * (1.0 - trim_high))
    hi = max(hi, lo + 1)
    return float(np.mean(vals[lo:hi]))


@dataclass
class PendingNoise:
    frame_idx: int
    out_h: int
    out_w: int
    tiles_used: int = 0  # filled on harvest
    sigma_frame: float = float("nan")
    grain_frame: float = float("nan")
    # device arrays
    sigma: cp.ndarray = None
    grain: cp.ndarray = None
    struct: cp.ndarray = None
    meanI: cp.ndarray = None
    varI: cp.ndarray = None
    cnt: cp.ndarray = None


    bufset: object = None  # return device buffers to pool


@dataclass
class NvofRun:
    id: str
    ranges: Optional[List[Tuple[int, int]]]
    W: int
    H: int
    grid: int
    perf: str
    temporal_hints: bool
    use_magsqr: bool
    nvof: object = None
    ver: str = ""
    hw_grid: int = 0
    flow_w: int = 0
    flow_h: int = 0
    small: List[cv2.cuda_GpuMat] = None
    gray8: List[cv2.cuda_GpuMat] = None
    flow: cv2.cuda_GpuMat = None
    float_flow: cv2.cuda_GpuMat = None
    mag: cv2.cuda_GpuMat = None
    prev_i: int = 0
    has_prev: bool = False
    pending_means: cp.ndarray = None
    pending_frames: List[int] = None
    pending_count: int = 0
    row_sum_buf: cp.ndarray = None
    host_buf: np.ndarray = None
    metrics: Dict[int, float] = None
    last_value: float = float("nan")


def _merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted((int(a), int(b)) for a, b in ranges)
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _parse_ranges_spec(spec: object) -> Optional[List[Tuple[int, int]]]:
    if spec is None:
        return None
    if isinstance(spec, str):
        s = spec.strip()
        if not s:
            return None
        parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
        ranges = []
        for part in parts:
            if "-" in part:
                a, b = part.split("-", 1)
                a = a.strip()
                b = b.strip()
                if not a or not b:
                    raise ValueError(f"Open-ended range not supported: {part}")
                start = int(a)
                end = int(b)
            else:
                start = end = int(part)
            if start > end:
                raise ValueError(f"Invalid range {start}-{end}")
            ranges.append((start, end))
        return _merge_ranges(ranges)
    if isinstance(spec, (list, tuple)):
        ranges = []
        for item in spec:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                start = int(item[0])
                end = int(item[1])
            else:
                start = end = int(item)
            if start > end:
                raise ValueError(f"Invalid range {start}-{end}")
            ranges.append((start, end))
        return _merge_ranges(ranges)
    raise ValueError(f"Unsupported ranges spec: {type(spec)}")


def _in_ranges(idx: int, ranges: Optional[List[Tuple[int, int]]]) -> bool:
    if ranges is None:
        return True
    for start, end in ranges:
        if idx < start:
            return False
        if start <= idx <= end:
            return True
    return False


def _unique_id(base: str, used: set[str]) -> str:
    base = str(base).strip() or "run"
    if base not in used:
        used.add(base)
        return base
    i = 2
    while f"{base}_{i}" in used:
        i += 1
    name = f"{base}_{i}"
    used.add(name)
    return name


def _load_schedule(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Schedule must be a JSON object.")
    return data


def parse_metrics_arg(raw: str) -> set[str]:
    tokens = {t.strip().lower() for t in str(raw).split(",") if t.strip()}
    if not tokens:
        return set()
    if "all" in tokens:
        return {"nvof", "noise", "luma"}
    if "none" in tokens:
        return set()
    allowed = {"nvof", "noise", "luma"}
    unknown = sorted(t for t in tokens if t not in allowed)
    if unknown:
        raise ValueError(f"Unknown metrics: {', '.join(unknown)}. Use: nvof, noise, luma, all, none")
    return tokens
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)

    ap.add_argument("--gpu", type=int, default=0)

    # NVOF path
    ap.add_argument("--nvof-w", type=int, default=800)
    ap.add_argument("--nvof-h", type=int, default=450)
    ap.add_argument("--grid", type=int, default=4, choices=(1, 2, 4))
    ap.add_argument("--perf", default="slow", choices=("fast", "medium", "slow"))
    ap.add_argument("--temporal-hints", type=int, default=1)
    ap.add_argument("--use-magsqr", type=int, default=1)
    ap.add_argument("--metrics", default="all", help="Comma list: nvof,noise,luma (luma=mean/p25/p75), or all/none. Ignored if --schedule is set.")
    ap.add_argument("--schedule", default="", help="JSON schedule with per-metric ranges and multiple NVOF runs")

    # Noise path (>=2K)
    ap.add_argument("--noise-step", type=int, default=1, help="Run noise metric every Nth frame")
    ap.add_argument("--downscale", type=float, default=1, help="Downscale factor (4K->2K ~ 0.5)")
    ap.add_argument("--min-width", type=int, default=2560)
    ap.add_argument("--min-height", type=int, default=1440)

    ap.add_argument("--tile", type=int, default=64)
    ap.add_argument("--pix-step", type=int, default=2)
    ap.add_argument("--tile-sample", type=float, default=0.5)
    ap.add_argument("--tile-keep", type=float, default=0.30)
    ap.add_argument("--d2", type=int, default=2)

    ap.add_argument("--seed", type=int, default=12345)

    ap.add_argument("--min-luma", type=float, default=4.0)
    ap.add_argument("--max-luma", type=float, default=251.0)
    ap.add_argument("--min-var", type=float, default=1.0)
    ap.add_argument("--min-samples", type=int, default=16)

    ap.add_argument("--pool", choices=["median", "trimmed_mean"], default="median")
    ap.add_argument("--trim-low", type=float, default=0.025)
    ap.add_argument("--trim-high", type=float, default=0.30)

    # Sync / logging
    ap.add_argument("--sync-every", type=int, default=1, help="Synchronize+harvest results once per N frames (WDDM often prefers 1)")
    ap.add_argument("--wddm-flush-every", type=int, default=0, help="Windows/WDDM helper: call cudaStreamQuery(0) every N frames to encourage command buffer flush (0=off)")
    ap.add_argument("--luma-subsample", type=int, default=16, help="Subsample luma stats by stride N (1=full res, 2=every other pixel, etc.)")
    ap.add_argument("--print-every", type=int, default=1, help="Print once per N frames (defaults to sync-every)")
    ap.add_argument("--max-frames", type=int, default=0, help="0=all")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--out-csv", default="", help="Optional CSV output path (use '-' for stdout)")
    ap.add_argument("--reader-gray8", type=int, default=1, help="Try to force VideoReader output to GRAY8 (1/0)")
    ap.add_argument("--surfaces", type=int, default=24, help="Override VideoReaderInitParams.minNumDecodeSurfaces (0=default). Higher can increase throughput but uses more GPU memory.")

    return ap.parse_args()


def main():
    args = parse_args()
    if args.print_every <= 0:
        args.print_every = args.sync_every

    schedule = None
    schedule_used = False
    if getattr(args, "schedule", ""):
        try:
            schedule = _load_schedule(args.schedule)
            schedule_used = True
        except Exception as exc:
            raise SystemExit(f"Bad --schedule: {exc}")

    nvof_specs: List[dict] = []
    noise_ranges = None
    luma_ranges = None
    noise_id = "noise"
    luma_id = "luma"

    if schedule_used:
        nvof_specs = schedule.get("nvof", [])
        if not isinstance(nvof_specs, list):
            raise SystemExit("Schedule 'nvof' must be a list.")
        noise_spec = schedule.get("noise")
        if noise_spec is not None and not isinstance(noise_spec, dict):
            raise SystemExit("Schedule 'noise' must be an object.")
        luma_spec = schedule.get("luma")
        if luma_spec is not None and not isinstance(luma_spec, dict):
            raise SystemExit("Schedule 'luma' must be an object.")
        if noise_spec is not None:
            noise_ranges = _parse_ranges_spec(noise_spec.get("ranges"))
            noise_id = str(noise_spec.get("id", noise_id) or noise_id).strip() or "noise"
        if luma_spec is not None:
            luma_ranges = _parse_ranges_spec(luma_spec.get("ranges"))
            luma_id = str(luma_spec.get("id", luma_id) or luma_id).strip() or "luma"
        nvof_enabled = bool(nvof_specs)
        noise_requested = noise_spec is not None
        luma_enabled = luma_spec is not None
    else:
        try:
            metrics = parse_metrics_arg(args.metrics)
        except ValueError as exc:
            raise SystemExit(str(exc))
        nvof_enabled = "nvof" in metrics
        noise_requested = "noise" in metrics
        luma_enabled = "luma" in metrics
        if nvof_enabled:
            nvof_specs = [{
                "id": "nvof",
                "w": int(args.nvof_w),
                "h": int(args.nvof_h),
                "grid": int(args.grid),
                "perf": str(args.perf),
                "temporal_hints": int(args.temporal_hints),
                "use_magsqr": int(args.use_magsqr),
                "ranges": None,
            }]

    noise_enabled = noise_requested and int(args.noise_step) > 0

    csv_stdout = False
    if args.out_csv:
        out_csv_norm = str(args.out_csv).strip().lower()
        if out_csv_norm in ("-", "stdout"):
            csv_stdout = True
            args.out_csv = "-"

    def log(*args, **kwargs):
        if "file" not in kwargs:
            kwargs["file"] = sys.stderr if csv_stdout else sys.stdout
        print(*args, **kwargs)

    if not hasattr(cv2, "cudacodec"):
        raise SystemExit("cv2.cudacodec not available.")
    if cv2.cuda.getCudaEnabledDeviceCount() <= 0:
        raise SystemExit("OpenCV CUDA reports 0 devices.")
    if not hasattr(cv2.cuda_GpuMat, "cudaPtr"):
        raise SystemExit("cv2.cuda_GpuMat.cudaPtr() not available (needed for zero-copy interop).")

    cv2.cuda.setDevice(args.gpu)
    cp.cuda.Device(args.gpu).use()

    # Compile kernel once (only if noise metrics enabled)
    ker = cp.RawKernel(KERNEL, "tile_noise_v2_u8") if noise_enabled else None
    # Create reader (optionally override minNumDecodeSurfaces for higher decode throughput)
    init_params = None
    if int(getattr(args, "surfaces", 0)) > 0:
        try:
            init_params = cv2.cudacodec.VideoReaderInitParams()
            init_params.minNumDecodeSurfaces = int(args.surfaces)
        except Exception as e:
            print(f"[warn] Cannot set minNumDecodeSurfaces={args.surfaces}: {e}", file=sys.stderr)
            init_params = None

    def _create_reader(path: str, params):
        """Best-effort createVideoReader with VideoReaderInitParams.
        Tries newer Python binding signatures first; falls back to createVideoReader(path).
        """
        cc = cv2.cudacodec
        if params is not None:
            last_err = None
            # Newer bindings: createVideoReader(filename, sourceParams=[], params=VideoReaderInitParams())
            try:
                vr_ = cc.createVideoReader(path, sourceParams=[], params=params)
                if isinstance(vr_, (tuple, list)):
                    vr_ = vr_[0]
                return vr_
            except Exception as e:
                last_err = e

            # Some builds accept keyword-only params without sourceParams
            try:
                vr_ = cc.createVideoReader(path, params=params)
                if isinstance(vr_, (tuple, list)):
                    vr_ = vr_[0]
                return vr_
            except Exception as e:
                last_err = e

            print(f"[warn] createVideoReader(params=...) unsupported, falling back to default reader. ({last_err})", file=sys.stderr)

        vr_ = cc.createVideoReader(path)
        if isinstance(vr_, (tuple, list)):
            vr_ = vr_[0]
        return vr_

    vr = _create_reader(args.input, init_params)

    if args.reader_gray8:
        try_set_reader_format(vr, want_gray=True, want_8bit=True)

    ok, g0 = vr.nextFrame()
    if not ok:
        raise SystemExit("Cannot read first frame via cudacodec.VideoReader.")
    ch0 = g0.channels()
    depth0 = g0.depth()
    w0, h0 = g0.size()
    log(f"Input (decoded) frame: {w0}x{h0}, channels={ch0}, depth={depth0} (CV depth enum)")

    # Decide noise working resolution (>=2K constraint)
    noise_w = 0
    noise_h = 0
    if noise_enabled:
        noise_h = int(round(h0 * float(args.downscale)))
        noise_w = int(round(w0 * float(args.downscale)))
        noise_w = max(int(args.min_width), noise_w)
        noise_h = max(int(args.min_height), noise_h)
        noise_w = min(noise_w, w0)
        noise_h = min(noise_h, h0)

    # Create NVOF sessions (one or many) and enforce W/H multiples of hw grid
    nvof_runs: List[NvofRun] = []
    if nvof_enabled:
        used_ids: set[str] = set()
        for spec in nvof_specs:
            if not isinstance(spec, dict):
                raise SystemExit("Each schedule.nvof entry must be an object.")
            W = int(spec.get("w", spec.get("nvof_w", args.nvof_w)))
            H = int(spec.get("h", spec.get("nvof_h", args.nvof_h)))
            grid = int(spec.get("grid", args.grid))
            perf = str(spec.get("perf", args.perf))
            temporal_hints = bool(int(spec.get("temporal_hints", args.temporal_hints)))
            use_magsqr = bool(int(spec.get("use_magsqr", args.use_magsqr)))
            ranges = _parse_ranges_spec(spec.get("ranges"))
            if schedule_used:
                run_id = _unique_id(spec.get("id", f"{W}x{H}"), used_ids)
            else:
                run_id = "nvof"

            nvof, ver = _make_nvof((W, H), perf=perf, grid=grid, temporal_hints=temporal_hints, gpu_id=args.gpu, stream=None)
            hw_grid = int(nvof.getGridSize())
            W2 = W - (W % hw_grid)
            H2 = H - (H % hw_grid)
            if (W2, H2) != (W, H):
                W, H = W2, H2
                nvof, ver = _make_nvof((W, H), perf=perf, grid=grid, temporal_hints=temporal_hints, gpu_id=args.gpu, stream=None)
                hw_grid = int(nvof.getGridSize())

            flow_w, flow_h = W // hw_grid, H // hw_grid
            run = NvofRun(
                id=run_id,
                ranges=ranges,
                W=W,
                H=H,
                grid=grid,
                perf=perf,
                temporal_hints=temporal_hints,
                use_magsqr=use_magsqr,
                nvof=nvof,
                ver=ver,
                hw_grid=hw_grid,
                flow_w=flow_w,
                flow_h=flow_h,
                small=[cv2.cuda_GpuMat(), cv2.cuda_GpuMat()],
                gray8=[cv2.cuda_GpuMat(), cv2.cuda_GpuMat()],
                flow=cv2.cuda_GpuMat(),
                float_flow=cv2.cuda_GpuMat(),
                mag=cv2.cuda_GpuMat(),
                metrics={},
                pending_frames=[],
            )
            run.flow.create(flow_h, flow_w, cv2.CV_16SC2)
            run.float_flow.create(flow_h, flow_w, cv2.CV_32FC2)
            run.mag.create(flow_h, flow_w, cv2.CV_32FC1)
            nvof_runs.append(run)
            range_desc = "all" if ranges is None else str(ranges)
            log(f"NVOF[{run.id}]: version={ver}, perf={perf}, req_grid={grid}, hw_grid={hw_grid}, ranges={range_desc}")
            log(f"NVOF[{run.id}] input: {W}x{H}  -> flow grid: {flow_w}x{flow_h}")

    if noise_enabled:
        log(f"Noise path: every {args.noise_step} frames @ {noise_w}x{noise_h}  (downscale={args.downscale}, min={args.min_width}x{args.min_height})")
    elif noise_requested:
        log("Noise metrics disabled (--noise-step <= 0)")

    log(f"Sync/harvest every {args.sync_every} frames")

    # Reusable GPU buffers
    gray_full = cv2.cuda_GpuMat()   # 1ch (8U or 16U) at full res
    graytmp_full = cv2.cuda_GpuMat()
    luma_u8 = cv2.cuda_GpuMat() if luma_enabled else None

    if noise_enabled:
        # Noise buffers at >=2K
        noise_small = cv2.cuda_GpuMat()
        noise_u8 = cv2.cuda_GpuMat()

    # Helpers
    def ensure_gray_full(src: cv2.cuda_GpuMat) -> cv2.cuda_GpuMat:
        """
        Convert src (1/3/4-ch, 8U/16U) into 1-ch gray_full (8U or 16U).
        """
        nonlocal gray_full, graytmp_full

        if src.channels() == 1:
            # No conversion, but keep reference (do not copy)
            return src

        if src.channels() == 3:
            code = cv2.COLOR_BGR2GRAY
        elif src.channels() == 4:
            code = cv2.COLOR_BGRA2GRAY
        else:
            raise RuntimeError(f"Unsupported channels={src.channels()}")

        tmp_type = cv2.CV_8UC1 if src.depth() == cv2.CV_8U else cv2.CV_16UC1
        gpu_ensure(graytmp_full, src.size(), tmp_type)
        graytmp_full = gpu_cvtColor(src, graytmp_full, code, dcn=0, stream=None)
        return graytmp_full

    def to_u8(src1ch: cv2.cuda_GpuMat, dst1ch_u8: cv2.cuda_GpuMat):
        if src1ch.depth() == cv2.CV_8U:
            gpu_copy(src1ch, dst1ch_u8, stream=None)
            return
        if src1ch.depth() == cv2.CV_16U:
            alpha = 1.0 / 256.0
        elif src1ch.depth() == cv2.CV_32F:
            alpha = 255.0
        else:
            raise RuntimeError(f"to_u8: unsupported depth {src1ch.depth()}")
        gpu_convert_to(src1ch, dst1ch_u8, cv2.CV_8U, alpha=alpha, beta=0.0, stream=None)

    def prep_nvof(gray_src_full: cv2.cuda_GpuMat, run: NvofRun, idx: int):
        # resize full-gray -> small[idx]
        run.small[idx] = cv2.cuda.resize(gray_src_full, (run.W, run.H), dst=run.small[idx], interpolation=cv2.INTER_LINEAR)
        s = run.small[idx]
        if s.channels() != 1:
            raise RuntimeError(f"prep_nvof: expected 1ch after resize, got {s.channels()}")
        gpu_ensure(run.gray8[idx], s.size(), cv2.CV_8UC1)
        to_u8(s, run.gray8[idx])

    def nvof_calc(run: NvofRun, prev_u8: cv2.cuda_GpuMat, cur_u8: cv2.cuda_GpuMat) -> cv2.cuda_GpuMat:
        out = run.nvof.calc(prev_u8, cur_u8, run.flow)
        return out[0] if isinstance(out, (tuple, list)) else out

    def do_convert_to_float(run: NvofRun, flow_in: cv2.cuda_GpuMat) -> cv2.cuda_GpuMat:
        out = run.nvof.convertToFloat(flow_in, run.float_flow)
        return out if isinstance(out, cv2.cuda_GpuMat) else run.float_flow

    def do_magnitude(run: NvofRun, float_flow_in: cv2.cuda_GpuMat):
        if run.use_magsqr:
            cv2.cuda.magnitudeSqr(float_flow_in, run.mag)
        else:
            cv2.cuda.magnitude(float_flow_in, run.mag)

    warm = int(args.warmup)
    if nvof_enabled:
        # Prime with first frame
        gray0 = ensure_gray_full(g0)
        for run in nvof_runs:
            prep_nvof(gray0, run, 0)
            run.prev_i = 0
            run.has_prev = True

        # Warmup frames (optional)
        for _ in range(max(0, warm)):
            ok, g = vr.nextFrame()
            if not ok:
                break
            gray = ensure_gray_full(g)
            for run in nvof_runs:
                cur_i = run.prev_i ^ 1
                prep_nvof(gray, run, cur_i)
                f = nvof_calc(run, run.gray8[run.prev_i], run.gray8[cur_i])
                ff = do_convert_to_float(run, f)
                do_magnitude(run, ff)
                run.prev_i = cur_i
                run.has_prev = True
        # hard sync once after warmup to start clean timings
        cp.cuda.runtime.deviceSynchronize()
    else:
        for _ in range(max(0, warm)):
            ok, _ = vr.nextFrame()
            if not ok:
                break

    # Measured run continues on the same reader (we want real output order)
    frame_idx = 0
    n_total = 0

    # Host-side collected metrics
    metrics_luma_mean: Dict[int, float] = {}
    metrics_luma_p25: Dict[int, float] = {}
    metrics_luma_p75: Dict[int, float] = {}
    metrics_noise_sigma: Dict[int, float] = {}
    metrics_noise_grain: Dict[int, float] = {}

    # Pending device-side results (to harvest per sync interval)
    pending_noise: List[PendingNoise] = []

    metrics_buf_cap = int(args.sync_every) if int(args.sync_every) > 0 else 1024
    if nvof_enabled:
        # NVOF: keep per-frame mean magnitude on GPU and harvest in batches.
        # - No per-frame cv2.cuda.sum() (which forces host sync)
        # - No mag.clone() allocations
        for run in nvof_runs:
            run.pending_means = cp.empty((metrics_buf_cap,), dtype=cp.float32)
            run.pending_frames = []
            run.pending_count = 0
            # Reused temp buffer for fast 2-step reduction (sum over contiguous axis, then final sum)
            run.row_sum_buf = cp.empty((run.flow_h,), dtype=cp.float32)
            run.host_buf = np.empty((metrics_buf_cap,), dtype=np.float32)

    if luma_enabled:
        pending_luma_mean = cp.empty((metrics_buf_cap,), dtype=cp.float32)
        pending_luma_p25 = cp.empty((metrics_buf_cap,), dtype=cp.float32)
        pending_luma_p75 = cp.empty((metrics_buf_cap,), dtype=cp.float32)
        pending_luma_count = 0
        pending_luma_frames: List[int] = []
        # Precomputed lookup for fast histogram-based mean
        luma_lut = cp.arange(256, dtype=cp.float32)


    # Host-side reusable buffers for small D2H copies (avoid per-harvest allocations)
    host_luma_mean_buf = np.empty((metrics_buf_cap,), dtype=np.float32) if luma_enabled else None
    host_luma_p25_buf = np.empty((metrics_buf_cap,), dtype=np.float32) if luma_enabled else None
    host_luma_p75_buf = np.empty((metrics_buf_cap,), dtype=np.float32) if luma_enabled else None

    def _d2h_into(dev_arr, host_view):
        """Best-effort copy from CuPy -> NumPy without allocating new host arrays."""
        if dev_arr is None:
            return None
        try:
            # CuPy ndarray.get(out=...) (preferred)
            dev_arr.get(out=host_view)
            return host_view
        except TypeError:
            # Older CuPy: no out=
            return dev_arr.get()
        except Exception:
            return dev_arr.get()

    # WDDM command batching: encourage timely command-buffer submission without blocking.
    # On Windows/WDDM, the driver may batch small launches; querying a recorded event can flush the software queue.
    _wddm_flush_evt = cp.cuda.Event(disable_timing=True)

    def _maybe_wddm_flush(frame_idx: int) -> None:
        fe = int(getattr(args, "wddm_flush_every", 0) or 0)
        if fe <= 0:
            return
        if os.name != "nt":
            return
        if (frame_idx % fe) != 0:
            return
        try:
            _wddm_flush_evt.record(cp.cuda.Stream.null)
            _wddm_flush_evt.query()  # non-blocking
        except Exception:
            pass

    def _should_prep_nvof(frame_idx: int, ranges: Optional[List[Tuple[int, int]]]) -> bool:
        if ranges is None:
            return True
        return _in_ranges(frame_idx, ranges) or _in_ranges(frame_idx + 1, ranges)

    # Noise: reuse output buffers to avoid allocating cp.empty() every --noise-step frame
    def _alloc_noise_bufset(nt_: int) -> dict:
        return {
            "nt": int(nt_),
            "sigma":  cp.empty((nt_,), dtype=cp.float32),
            "grain":  cp.empty((nt_,), dtype=cp.float32),
            "struct": cp.empty((nt_,), dtype=cp.float32),
            "meanI":  cp.empty((nt_,), dtype=cp.float32),
            "varI":   cp.empty((nt_,), dtype=cp.float32),
            "cnt":    cp.empty((nt_,), dtype=cp.int32),
        }

    noise_buf_pool: List[dict] = []
    if noise_enabled:
        _tiles_x = (int(noise_w) + int(args.tile) - 1) // int(args.tile)
        _tiles_y = (int(noise_h) + int(args.tile) - 1) // int(args.tile)
        _nt = int(_tiles_x * _tiles_y)

        if int(args.sync_every) > 0:
            _max_pending = (int(args.sync_every) + int(args.noise_step) - 1) // int(args.noise_step)
        else:
            _max_pending = 8  # conservative default for "no periodic harvest" mode

        # +2 safety margin (off-by-one + tail flush)
        for _ in range(max(1, _max_pending + 2)):
            noise_buf_pool.append(_alloc_noise_bufset(_nt))

    # Host scratch buffers for noise harvest (reused per record)
    noise_host_sigma = None
    noise_host_grain = None
    noise_host_struct = None
    noise_host_meanI = None
    noise_host_varI = None
    noise_host_cnt = None
    if noise_enabled:
        noise_host_sigma = np.empty((_nt,), dtype=np.float32)
        noise_host_grain = np.empty((_nt,), dtype=np.float32)
        noise_host_struct = np.empty((_nt,), dtype=np.float32)
        noise_host_meanI = np.empty((_nt,), dtype=np.float32)
        noise_host_varI = np.empty((_nt,), dtype=np.float32)
        noise_host_cnt = np.empty((_nt,), dtype=np.int32)
    t0 = time.perf_counter()
    last_print = time.perf_counter()
    header_cols = ["frame", "fps"]
    if nvof_enabled:
        if schedule_used or len(nvof_runs) > 1:
            for run in nvof_runs:
                header_cols.append(f"nvof_mean_{run.id}")
        else:
            header_cols.append("mean_flow")
    if luma_enabled:
        luma_suffix = f"_{luma_id}" if (schedule_used and luma_id and luma_id != "luma") else ""
        header_cols.extend([f"luma_mean{luma_suffix}", f"luma_p25{luma_suffix}", f"luma_p75{luma_suffix}"])
    if noise_enabled:
        noise_suffix = f"_{noise_id}" if (schedule_used and noise_id and noise_id != "noise") else ""
        header_cols.extend([f"noise_sigma{noise_suffix}", f"grain_ratio{noise_suffix}", f"tiles_used{noise_suffix}", f"noise_res{noise_suffix}"])
    log("\t".join(header_cols))
    last_luma_mean = float("nan")
    last_luma_p25 = float("nan")
    last_luma_p75 = float("nan")

    # Processing loop
    while True:
        if args.max_frames and n_total >= int(args.max_frames):
            break

        ok, g = vr.nextFrame()
        if not ok:
            break

        gray_full_src = ensure_gray_full(g)

        if nvof_enabled:
            for run in nvof_runs:
                if not _should_prep_nvof(frame_idx, run.ranges):
                    continue
                cur_i = run.prev_i ^ 1 if run.has_prev else 0
                prep_nvof(gray_full_src, run, cur_i)

                if run.has_prev and _in_ranges(frame_idx, run.ranges):
                    # NVOF: reduce mean magnitude on GPU into pending buffer (no host sync here)
                    if run.pending_count >= int(run.pending_means.size):
                        # Safety flush (relevant if --sync-every <= 0 and long runs)
                        cp.cuda.runtime.deviceSynchronize()
                        n = int(run.pending_count)
                        _vals = _d2h_into(run.pending_means[:n], run.host_buf[:n])
                        for i, fidx in enumerate(run.pending_frames):
                            run.metrics[fidx] = float(_vals[i])
                        run.pending_frames.clear()
                        run.pending_count = 0
                        if n > 0:
                            run.last_value = float(_vals[n - 1])

                    f = nvof_calc(run, run.gray8[run.prev_i], run.gray8[cur_i])
                    ff = do_convert_to_float(run, f)
                    do_magnitude(run, ff)

                    with cp.cuda.Stream.null:
                        mag_cp = gpumat_to_cupy_2d(run.mag, dtype=np.float32)
                        if int(run.row_sum_buf.size) != int(mag_cp.shape[0]):
                            run.row_sum_buf = cp.empty((int(mag_cp.shape[0]),), dtype=cp.float32)

                        # Two-step reduction keeps the reduced axis contiguous (fast CUB path) and avoids a full contiguous copy.
                        cp.sum(mag_cp, axis=1, dtype=cp.float32, out=run.row_sum_buf)
                        total = cp.sum(run.row_sum_buf, dtype=cp.float32)
                        run.pending_means[run.pending_count] = total / cp.float32(int(mag_cp.shape[0]) * int(mag_cp.shape[1]))

                    run.pending_frames.append(frame_idx)
                    run.pending_count += 1

                run.prev_i = cur_i
                run.has_prev = True

        if luma_enabled and _in_ranges(frame_idx, luma_ranges):
            if pending_luma_count >= int(pending_luma_mean.size):
                cp.cuda.runtime.deviceSynchronize()
                n = int(pending_luma_count)
                _means = _d2h_into(pending_luma_mean[:n], host_luma_mean_buf[:n])
                _p25 = _d2h_into(pending_luma_p25[:n], host_luma_p25_buf[:n])
                _p75 = _d2h_into(pending_luma_p75[:n], host_luma_p75_buf[:n])
                for i, fidx in enumerate(pending_luma_frames):
                    metrics_luma_mean[fidx] = float(_means[i])
                    metrics_luma_p25[fidx] = float(_p25[i])
                    metrics_luma_p75[fidx] = float(_p75[i])
                if n > 0:
                    last_luma_mean = float(_means[n - 1])
                    last_luma_p25 = float(_p25[n - 1])
                    last_luma_p75 = float(_p75[n - 1])
                pending_luma_frames.clear()
                pending_luma_count = 0

            luma_src = gray_full_src
            if luma_src.depth() != cv2.CV_8U:
                gpu_ensure(luma_u8, luma_src.size(), cv2.CV_8UC1)
                to_u8(luma_src, luma_u8)
                luma_src = luma_u8

            with cp.cuda.Stream.null:
                luma_cp = gpumat_to_cupy_2d(luma_src, dtype=np.uint8)
                ls = max(1, int(getattr(args, 'luma_subsample', 1) or 1))
                if ls > 1:
                    luma_cp = luma_cp[::ls, ::ls]

                # Histogram-based stats for uint8 luma: much cheaper than percentile/sort.
                flat = luma_cp.ravel()
                hist = cp.bincount(flat, minlength=256)
                total = int(flat.size)
                if total <= 0:
                    pending_luma_mean[pending_luma_count] = cp.float32(0.0)
                    pending_luma_p25[pending_luma_count] = cp.float32(0.0)
                    pending_luma_p75[pending_luma_count] = cp.float32(0.0)
                else:
                    luma_mean = cp.sum(hist.astype(cp.float32) * luma_lut, dtype=cp.float32) / cp.float32(total)

                    # cupy.searchsorted() accepts only Python int or CuPy ndarray for `v` (NumPy scalars are rejected).
                    # Use 0-d CuPy arrays for targets to stay on GPU and avoid type issues.
                    cdf = cp.cumsum(hist, dtype=cp.int64)
                    q25c = cp.asarray(int(math.ceil(0.25 * total)), dtype=cdf.dtype)
                    q75c = cp.asarray(int(math.ceil(0.75 * total)), dtype=cdf.dtype)
                    p25 = cp.searchsorted(cdf, q25c, side='left')
                    p75 = cp.searchsorted(cdf, q75c, side='left')

                    pending_luma_mean[pending_luma_count] = luma_mean
                    pending_luma_p25[pending_luma_count] = p25.astype(cp.float32)
                    pending_luma_p75[pending_luma_count] = p75.astype(cp.float32)

            pending_luma_frames.append(frame_idx)
            pending_luma_count += 1

        # Noise every Nth frame
        if noise_enabled and _in_ranges(frame_idx, noise_ranges) and (frame_idx % int(args.noise_step) == 0):
            # Resize full-res gray -> noise_small (keep depth), then convert to u8
            noise_small = cv2.cuda.resize(gray_full_src, (noise_w, noise_h), dst=noise_small, interpolation=cv2.INTER_LINEAR)
            gpu_ensure(noise_u8, noise_small.size(), cv2.CV_8UC1)
            to_u8(noise_small, noise_u8)

            with cp.cuda.Stream.null:
                img = gpumat_to_cupy_2d(noise_u8, dtype=np.uint8)
                out_h, out_w = int(noise_h), int(noise_w)
                stride = int(img.strides[0])

                tiles_x = (out_w + int(args.tile) - 1) // int(args.tile)
                tiles_y = (out_h + int(args.tile) - 1) // int(args.tile)
                nt = int(tiles_x * tiles_y)
                
                # Reuse preallocated output buffers (pool) to avoid per-call allocations
                if noise_buf_pool:
                    _buf = noise_buf_pool.pop()
                    if int(_buf.get('nt', -1)) != int(nt):
                        _buf = _alloc_noise_bufset(int(nt))
                else:
                    _buf = _alloc_noise_bufset(int(nt))
                
                out_sigma  = _buf['sigma']
                out_grain  = _buf['grain']
                out_struct = _buf['struct']
                out_mean   = _buf['meanI']
                out_var    = _buf['varI']
                out_count  = _buf['cnt']
                block = (16, 16, 1)
                grid  = (int(tiles_x), int(tiles_y), 1)

                ker(
                    grid, block,
                    (
                        img,
                        np.int32(out_h), np.int32(out_w), np.int32(stride),
                        np.int32(args.tile), np.int32(args.pix_step), np.int32(args.d2),
                        np.float32(args.tile_sample),
                        np.uint32(args.seed), np.uint32(frame_idx),
                        out_sigma, out_grain, out_struct, out_mean, out_var, out_count
                    )
                )

                pending_noise.append(PendingNoise(
                    frame_idx=frame_idx,
                    out_h=out_h,
                    out_w=out_w,
                    sigma=out_sigma, grain=out_grain, struct=out_struct,
                    meanI=out_mean, varI=out_var, cnt=out_count,
                    bufset=_buf
                ))

        _maybe_wddm_flush(frame_idx)

        frame_idx += 1
        n_total += 1

        # Harvest once per sync interval
        if args.sync_every > 0 and (n_total % int(args.sync_every) == 0):
            if nvof_enabled or luma_enabled or noise_enabled:
                # One sync point for both OpenCV + CuPy work.
                cp.cuda.runtime.deviceSynchronize()

            if nvof_enabled:
                # NVOF metrics batch: copy the per-frame means once per interval
                for run in nvof_runs:
                    if int(run.pending_count) > 0:
                        n = int(run.pending_count)
                        nvof_vals = _d2h_into(run.pending_means[:n], run.host_buf[:n])
                        for i, fidx in enumerate(run.pending_frames):
                            run.metrics[fidx] = float(nvof_vals[i])
                        run.pending_frames.clear()
                        run.pending_count = 0
                        if n > 0:
                            run.last_value = float(nvof_vals[n - 1])

            if luma_enabled:
                if int(pending_luma_count) > 0:
                    n = int(pending_luma_count)
                    luma_vals = _d2h_into(pending_luma_mean[:n], host_luma_mean_buf[:n])
                    luma_p25 = _d2h_into(pending_luma_p25[:n], host_luma_p25_buf[:n])
                    luma_p75 = _d2h_into(pending_luma_p75[:n], host_luma_p75_buf[:n])
                    for i, fidx in enumerate(pending_luma_frames):
                        metrics_luma_mean[fidx] = float(luma_vals[i])
                        metrics_luma_p25[fidx] = float(luma_p25[i])
                        metrics_luma_p75[fidx] = float(luma_p75[i])
                    if n > 0:
                        last_luma_mean = float(luma_vals[n - 1])
                        last_luma_p25 = float(luma_p25[n - 1])
                        last_luma_p75 = float(luma_p75[n - 1])
                    pending_luma_frames.clear()
                    pending_luma_count = 0

            # Noise harvest (usually 0 or 1 record per interval)
            if noise_enabled:
                last_noise_sigma = float("nan")
                last_noise_grain = float("nan")
                last_tiles_used = 0
                last_noise_res = ""

                for rec in pending_noise:
                    sigma  = _d2h_into(rec.sigma, noise_host_sigma)
                    grain  = _d2h_into(rec.grain, noise_host_grain)
                    struct = _d2h_into(rec.struct, noise_host_struct)
                    meanI  = _d2h_into(rec.meanI, noise_host_meanI)
                    varI   = _d2h_into(rec.varI, noise_host_varI)
                    cnt    = _d2h_into(rec.cnt, noise_host_cnt)

                    valid = (
                        (cnt >= int(args.min_samples))
                        & np.isfinite(sigma) & (sigma >= 0.0)
                        & np.isfinite(grain) & (grain > 0.0)
                        & np.isfinite(struct) & (struct >= 0.0)
                        & np.isfinite(meanI) & (meanI >= float(args.min_luma)) & (meanI <= float(args.max_luma))
                        & np.isfinite(varI) & (varI >= float(args.min_var))
                    )

                    idx = np.where(valid)[0]
                    if idx.size == 0:
                        sigma_frame = float("nan")
                        grain_frame = float("nan")
                        tiles_used = 0
                    else:
                        k = max(1, int(round(idx.size * float(args.tile_keep))))
                        if k >= idx.size:
                            keep = idx
                        else:
                            # Faster than full argsort when we only need the lowest-k tiles
                            part = np.argpartition(struct[idx], k - 1)[:k]
                            keep = idx[part]

                        sigma_frame = pool_values(sigma[keep], args.pool, float(args.trim_low), float(args.trim_high))
                        grain_frame = pool_values(grain[keep], args.pool, float(args.trim_low), float(args.trim_high))
                        tiles_used = int(keep.size)

                    metrics_noise_sigma[rec.frame_idx] = float(sigma_frame)
                    metrics_noise_grain[rec.frame_idx] = float(grain_frame)
                    if rec.bufset is not None:
                        noise_buf_pool.append(rec.bufset)

                    # For printing: last computed noise in this harvest
                    last_noise_sigma = float(sigma_frame)
                    last_noise_grain = float(grain_frame)
                    last_tiles_used = tiles_used
                    last_noise_res = f"{rec.out_w}x{rec.out_h}"

                pending_noise.clear()

            # Print periodically (default aligned with sync)
            if (n_total % int(args.print_every)) == 0:
                dt = time.perf_counter() - t0
                fps = n_total / dt if dt > 0 else 0.0
                row_parts = [str(n_total), f"{fps:.1f}"]
                if nvof_enabled:
                    if schedule_used or len(nvof_runs) > 1:
                        for run in nvof_runs:
                            row_parts.append(f"{run.last_value:.6f}")
                    else:
                        row_parts.append(f"{nvof_runs[0].last_value:.6f}")
                if luma_enabled:
                    row_parts.extend([f"{last_luma_mean:.2f}", f"{last_luma_p25:.2f}", f"{last_luma_p75:.2f}"])
                if noise_enabled:
                    row_parts.extend([f"{last_noise_sigma:.4f}", f"{last_noise_grain:.4f}", str(last_tiles_used), last_noise_res])
                log("\t".join(row_parts))

    # Final sync + harvest any remainder
    if nvof_enabled or luma_enabled or noise_enabled:
        cp.cuda.runtime.deviceSynchronize()

    if nvof_enabled:
        for run in nvof_runs:
            if int(run.pending_count) > 0:
                n = int(run.pending_count)
                nvof_vals = _d2h_into(run.pending_means[:n], run.host_buf[:n])
                for i, fidx in enumerate(run.pending_frames):
                    run.metrics[fidx] = float(nvof_vals[i])
                run.pending_frames.clear()
                run.pending_count = 0
                if n > 0:
                    run.last_value = float(nvof_vals[n - 1])

    if luma_enabled and int(pending_luma_count) > 0:
        n = int(pending_luma_count)
        luma_vals = _d2h_into(pending_luma_mean[:n], host_luma_mean_buf[:n])
        luma_p25 = _d2h_into(pending_luma_p25[:n], host_luma_p25_buf[:n])
        luma_p75 = _d2h_into(pending_luma_p75[:n], host_luma_p75_buf[:n])
        for i, fidx in enumerate(pending_luma_frames):
            metrics_luma_mean[fidx] = float(luma_vals[i])
            metrics_luma_p25[fidx] = float(luma_p25[i])
            metrics_luma_p75[fidx] = float(luma_p75[i])
        pending_luma_frames.clear()
        pending_luma_count = 0

    if noise_enabled:
        # Flush pending noise remainders
        for rec in pending_noise:
            sigma  = _d2h_into(rec.sigma, noise_host_sigma)
            grain  = _d2h_into(rec.grain, noise_host_grain)
            struct = _d2h_into(rec.struct, noise_host_struct)
            meanI  = _d2h_into(rec.meanI, noise_host_meanI)
            varI   = _d2h_into(rec.varI, noise_host_varI)
            cnt    = _d2h_into(rec.cnt, noise_host_cnt)

            valid = (
                (cnt >= int(args.min_samples))
                & np.isfinite(sigma) & (sigma >= 0.0)
                & np.isfinite(grain) & (grain > 0.0)
                & np.isfinite(struct) & (struct >= 0.0)
                & np.isfinite(meanI) & (meanI >= float(args.min_luma)) & (meanI <= float(args.max_luma))
                & np.isfinite(varI) & (varI >= float(args.min_var))
            )

            idx = np.where(valid)[0]
            if idx.size == 0:
                sigma_frame = float("nan")
                grain_frame = float("nan")
            else:
                order = np.argsort(struct[idx])
                k = max(1, int(round(idx.size * float(args.tile_keep))))
                keep = idx[order[:k]]
                sigma_frame = pool_values(sigma[keep], args.pool, float(args.trim_low), float(args.trim_high))
                grain_frame = pool_values(grain[keep], args.pool, float(args.trim_low), float(args.trim_high))

            metrics_noise_sigma[rec.frame_idx] = float(sigma_frame)
            metrics_noise_grain[rec.frame_idx] = float(grain_frame)
            if rec.bufset is not None:
                noise_buf_pool.append(rec.bufset)
    pending_noise.clear()

    t1 = time.perf_counter()
    dt = t1 - t0
    fps = n_total / dt if dt > 0 else 0.0

    log("\nDone.")
    log(f"Processed frames: {n_total}  time: {dt:.2f}s  avg fps: {fps:.2f}")
    if nvof_enabled:
        if schedule_used or len(nvof_runs) > 1:
            for run in nvof_runs:
                log(f"NVOF[{run.id}] metrics collected: {len(run.metrics)}")
        else:
            log(f"NVOF metrics collected: {len(nvof_runs[0].metrics)}")
    if luma_enabled:
        log(f"Luma metrics collected: {len(metrics_luma_mean)}")
    if noise_enabled:
        log(f"Noise frames collected: {len(metrics_noise_sigma)} (every {args.noise_step} frames)")

    if args.out_csv:
        path = args.out_csv
        csv_cols = ["frame"]
        if nvof_enabled:
            if schedule_used or len(nvof_runs) > 1:
                for run in nvof_runs:
                    csv_cols.append(f"nvof_mean_{run.id}")
            else:
                csv_cols.append("nvof_mean")
        if luma_enabled:
            luma_suffix = f"_{luma_id}" if (schedule_used and luma_id and luma_id != "luma") else ""
            csv_cols.extend([f"luma_mean{luma_suffix}", f"luma_p25{luma_suffix}", f"luma_p75{luma_suffix}"])
        if noise_enabled:
            noise_suffix = f"_{noise_id}" if (schedule_used and noise_id and noise_id != "noise") else ""
            csv_cols.extend([f"noise_sigma{noise_suffix}", f"grain_ratio{noise_suffix}"])

        if nvof_enabled or luma_enabled:
            frame_indices = range(n_total)
        elif noise_enabled:
            frame_indices = sorted(metrics_noise_sigma.keys())
        else:
            frame_indices = []

        rows = []
        for fidx in frame_indices:
            row = [fidx]
            if nvof_enabled:
                if schedule_used or len(nvof_runs) > 1:
                    for run in nvof_runs:
                        row.append(run.metrics.get(fidx, float("nan")))
                else:
                    row.append(nvof_runs[0].metrics.get(fidx, float("nan")))
            if luma_enabled:
                row.append(metrics_luma_mean.get(fidx, float("nan")))
                row.append(metrics_luma_p25.get(fidx, float("nan")))
                row.append(metrics_luma_p75.get(fidx, float("nan")))
            if noise_enabled:
                row.append(metrics_noise_sigma.get(fidx, float("nan")))
                row.append(metrics_noise_grain.get(fidx, float("nan")))
            rows.append(row)

        if rows:
            arr = np.array(rows, dtype=np.float64)
        else:
            arr = np.empty((0, len(csv_cols)), dtype=np.float64)
        header = ",".join(csv_cols)
        if csv_stdout:
            np.savetxt(sys.stdout, arr, delimiter=",", header=header, comments="")
        else:
            np.savetxt(path, arr, delimiter=",", header=header, comments="")
            log(f"Saved CSV: {path}")


if __name__ == "__main__":
    main()
