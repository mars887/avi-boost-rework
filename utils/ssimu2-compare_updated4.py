#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ssimu2-compare.py

Сравнение 2 видео/картинок по метрикам:
- SSIMULACRA2 (vship или vszip)
- BUTTERAUGLI (vship или vapoursynth-butteraugli)

Фичи:
- --crop и --scale (scale — это переименованный resize) — применяются к обоим входам
- --left/--right "crop=...,scale=..." — раздельные цепочки для каждого входа (в стиле ffmpeg)
- порядок операций учитывается (как в цепочке ffmpeg crop,scale)
- --step (каждый N-й кадр) + ускоренная выборка через VideoNode.frames(prefetch=..., backlog=...)

Примеры:
  python ssimu2-compare.py ref.mkv enc.mkv --metric ssimulacra2 --backend auto --step 5 --plot ssimu2.png

  python ssimu2-compare.py ref.mkv enc.mkv --metric butteraugli --backend vship --butter-norm 3 \
      --crop 3840x2020 --scale -1x1440 --plot butter.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass
from itertools import islice
from collections import deque
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import vapoursynth as vs
except ImportError as e:
    raise SystemExit(
        "Не найден модуль vapoursynth. Установи VapourSynth и запусти из окружения, где доступен vapoursynth."
    ) from e

# tqdm/matplotlib — опционально
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

core = vs.core

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif"}

# ------------------------------
# Parsing helpers
# ------------------------------

_DIM_RE = None


def _parse_wh(s: str, allow_minus_one: bool) -> Tuple[int, int]:
    """
    Parse 'WxH' (or 'W:H') where W/H are ints.
    If allow_minus_one: accept -1 for one side (auto aspect).
    """
    s = s.strip().lower().replace(":", "x")
    if "x" not in s:
        raise ValueError(f"Ожидался формат WxH, получено: {s}")
    a, b = s.split("x", 1)
    try:
        w = int(a)
        h = int(b)
    except ValueError as e:
        raise ValueError(f"Ожидались целые числа WxH, получено: {s}") from e

    if allow_minus_one:
        if (w == -1 and h == -1) or (w == 0 or h == 0):
            raise ValueError(f"Некорректный scale: {s} (нельзя 0 и нельзя -1x-1)")
        if w < -1 or h < -1:
            raise ValueError(f"Некорректный scale: {s}")
    else:
        if w <= 0 or h <= 0:
            raise ValueError(f"Некорректный crop: {s} (ожидается >0)")
    return w, h


def _parse_crop_or_scale_expr(expr: str) -> Op:
    """
    Parse single expression like:
      - crop=3840:1608
      - crop=3840x1608+0+216
      - crop=3840:1608:0:216
      - scale=-1:1440
      - resize=1920x-1   (alias for scale)
    Returns Op(kind, w, h, x, y).
    """
    s = expr.strip()
    if not s:
        raise ValueError("Пустая операция в фильтр-цепочке.")

    if "=" not in s:
        raise ValueError(f"Ожидалась операция вида name=value, получено: {s}")

    name, val = s.split("=", 1)
    name = name.strip().lower()
    val = val.strip()

    if name in ("scale", "resize"):
        w, h = _parse_wh(val, allow_minus_one=True)
        return Op("scale", w, h, raw=s)

    if name != "crop":
        raise ValueError(f"Неизвестная операция '{name}' в цепочке: {s}")

    # crop: allow WxH, WxH+X+Y, WxH:X:Y
    v = val.strip().lower().replace("x", ":")
    # WxH+X+Y
    m = re.fullmatch(r"(\d+):(\d+)\+(\d+)\+(\d+)", v)
    if m:
        w = int(m.group(1))
        h = int(m.group(2))
        x = int(m.group(3))
        y = int(m.group(4))
        return Op("crop", w, h, raw=s, x=x, y=y)

    # WxH:X:Y
    m = re.fullmatch(r"(\d+):(\d+):(\d+):(\d+)", v)
    if m:
        w = int(m.group(1))
        h = int(m.group(2))
        x = int(m.group(3))
        y = int(m.group(4))
        return Op("crop", w, h, raw=s, x=x, y=y)

    # center crop WxH
    w, h = _parse_wh(val, allow_minus_one=False)
    return Op("crop", w, h, raw=s)


def parse_chain(chain: Optional[str]) -> List[Op]:
    """
    Parse comma-separated chain string (ffmpeg-like) into Op list.
    Example: "crop=3840:1608,scale=-1:1440"
    """
    if chain is None:
        return []
    chain = chain.strip()
    if chain == "":
        return []
    ops: List[Op] = []
    for part in chain.split(","):
        part = part.strip()
        if not part:
            continue
        ops.append(_parse_crop_or_scale_expr(part))
    return ops

def _align_to_subsampling(x: int, sub: int) -> int:
    """Align dimension to subsampling requirement (2**sub)."""
    m = 1 << sub
    if m <= 1:
        return x
    return x - (x % m)


def _validate_crop_dims(clip: vs.VideoNode, w: int, h: int) -> None:
    if w > clip.width or h > clip.height:
        raise ValueError(
            f"Crop {w}x{h} больше исходных размеров {clip.width}x{clip.height}."
        )
    if clip.format is not None:
        mw = 1 << clip.format.subsampling_w
        mh = 1 << clip.format.subsampling_h
        if w % mw != 0 or h % mh != 0:
            raise ValueError(
                f"Crop {w}x{h} должен быть кратен субдискретизации: "
                f"ширина кратна {mw}, высота кратна {mh} для формата {clip.format.name}."
            )


def _resolve_scale_dims(clip: vs.VideoNode, w: int, h: int) -> Tuple[int, int]:
    """
    Resolve -1 in WxH keeping aspect ratio based on *current* clip dimensions.
    Also aligns to subsampling.
    """
    if w == -1:
        # keep AR
        w = int(round(clip.width * (h / clip.height)))
    elif h == -1:
        h = int(round(clip.height * (w / clip.width)))

    if w <= 0 or h <= 0:
        raise ValueError(f"Некорректный scale после вычисления: {w}x{h}")

    if clip.format is not None:
        w2 = _align_to_subsampling(w, clip.format.subsampling_w)
        h2 = _align_to_subsampling(h, clip.format.subsampling_h)
        if w2 <= 0 or h2 <= 0:
            raise ValueError(f"Некорректный scale после выравнивания: {w2}x{h2}")
        w, h = w2, h2

    return w, h


@dataclass(frozen=True)
class Op:
    kind: str  # "crop" | "scale"
    w: int
    h: int
    raw: str = ""
    x: Optional[int] = None  # for crop: left offset (absolute)
    y: Optional[int] = None  # for crop: top offset (absolute)

class _AppendOp(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        ops: List[Op] = getattr(namespace, self.dest, None)
        if ops is None:
            ops = []

        assert option_string is not None
        opt = option_string.lower()

        if opt in ("--crop",):
            w, h = _parse_wh(values, allow_minus_one=False)
            ops.append(Op("crop", w, h, values))
        elif opt in ("--scale", "--resize"):
            # --resize оставлен как алиас для совместимости
            w, h = _parse_wh(values, allow_minus_one=True)
            ops.append(Op("scale", w, h, values))
            if opt == "--resize":
                # Мягкое предупреждение (без зависимостей)
                print(
                    "Предупреждение: --resize переименован в --scale. "
                    "Алиас --resize пока поддерживается, но лучше перейти на --scale.",
                    file=sys.stderr,
                )
        else:
            parser.error(f"Неизвестная операция: {option_string}")

        setattr(namespace, self.dest, ops)


# ------------------------------
# CLI
# ------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Сравнение видео/картинок по SSIMULACRA2 или BUTTERAUGLI (VapourSynth)."
    )
    p.add_argument("reference", help="Референсное видео/картинка (высокое качество)")
    p.add_argument("distorted", help="Искажённое (сжатое) видео/картинка")

    p.add_argument(
        "--metric",
        default="ssimulacra2",
        choices=["ssimulacra2", "butteraugli"],
        help="Какая метрика: ssimulacra2 (больше = лучше) или butteraugli (меньше = лучше).",
    )
    p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "vship", "vszip", "cpu"],
        help=(
            "Бэкенд вычисления метрик. "
            "auto: vship если доступен, иначе vszip (для ssimulacra2) / butteraugli-плагин (для butteraugli). "
            "vship: GPU (если установлен). "
            "vszip: только для ssimulacra2. "
            "cpu: vszip для ssimulacra2 или vapoursynth-butteraugli для butteraugli."
        ),
    )

    # Трансформации (порядок важен)
    p.add_argument(
        "--crop",
        dest="ops",
        action=_AppendOp,
        metavar="WxH",
        help="Центральный crop до WxH. Можно указывать несколько раз; порядок с --scale учитывается.",
    )
    p.add_argument(
        "--scale",
        dest="ops",
        action=_AppendOp,
        metavar="WxH",
        help="Scale до WxH (алиас старого --resize). Можно -1 для одной стороны: -1x1440 или 1920x-1.",
    )
    p.add_argument(
        "--resize",
        dest="ops",
        action=_AppendOp,
        metavar="WxH",
        help=argparse.SUPPRESS,  # алиас
    )

    # Раздельные трансформации для левого/правого входа (если задано, применяется перед общими --crop/--scale)
    p.add_argument(
        "--left",
        default=None,
        metavar="CHAIN",
        help=(
            "Фильтр-цепочка только для reference (левый вход). "
            "Синтаксис: 'crop=WxH[,scale=WxH][,crop=WxH+X+Y]...' (разделитель запятая; WxH можно и через ':'). "
            "Пример: --left 'crop=3840:1608,scale=-1:1440'."
        ),
    )
    p.add_argument(
        "--right",
        default=None,
        metavar="CHAIN",
        help=(
            "Фильтр-цепочка только для distorted (правый вход). "
            "Синтаксис такой же, как у --left. Пример: --right '' (без трансформаций)."
        ),
    )


    # Диапазон/шаг кадров
    p.add_argument("--start", type=int, default=0, help="Начальный кадр (включительно).")
    p.add_argument(
        "--end",
        type=int,
        default=-1,
        help="Конечный кадр (включительно). -1 = до конца (по минимальной длине).",
    )
    p.add_argument(
        "--step",
        type=int,
        default=1,
        help="Брать каждый N-й кадр (например 5 = 0,5,10,...).",
    )

    p.add_argument(
        "--shift",
        type=int,
        default=0,
        help=(
            "Сдвиг между видео в кадрах. "
            ">0: отбросить N кадров с начала distorted. "
            "<0: отбросить abs(N) кадров с начала reference."
        ),
    )
    p.add_argument(
        "--stats",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Промежуточная статистика: среднее значение метрики за последние N обработанных кадров. "
            "0 = выключено."
        ),
    )


    # Параметры vship
    p.add_argument(
        "--streams",
        type=int,
        default=4,
        help="vship numStream (компромисс скорость/VRAM). Имеет смысл только при backend=vship.",
    )

    # Параметры butteraugli (vship)
    p.add_argument(
        "--butter-norm",
        default="3",
        choices=["2", "3", "inf"],
        help="Какой butteraugli score брать из vship: 2, 3 или inf-норма.",
    )
    p.add_argument(
        "--butter-intensity",
        type=float,
        default=1.0,
        help="vship intensity_multiplier (чувствительность) для Butteraugli. Используется только при backend=vship.",
    )

    # Настройки декодера BestSource (если выбран/доступен)
    p.add_argument(
        "--source",
        default="auto",
        choices=["auto", "bs", "ffms2"],
        help="Источник для загрузки видео: auto (ffms2 если есть, иначе bs), bs (BestSource), ffms2.",
    )
    p.add_argument("--hwdevice", default=None, help="bs.VideoSource hwdevice (например 'd3d11va', 'cuda').")
    p.add_argument("--bs-threads", type=int, default=None, help="bs.VideoSource threads (0 = autodetect).")
    p.add_argument("--bs-cachemode", type=int, default=None, help="bs.VideoSource cachemode (см. README BestSource).")
    p.add_argument("--bs-cachepath", default=None, help="bs.VideoSource cachepath.")
    p.add_argument("--bs-cachesize", type=int, default=None, help="bs.VideoSource cachesize (MB).")
    p.add_argument("--bs-seekpreroll", type=int, default=None, help="bs.VideoSource seekpreroll.")
    p.add_argument("--bs-hwfallback", dest="bs_hwfallback", action="store_true", default=None, help="BestSource hwfallback=True.")
    p.add_argument("--no-bs-hwfallback", dest="bs_hwfallback", action="store_false", help="BestSource hwfallback=False (ошибка если HW декод не может использоваться).")
    p.add_argument("--ffms2-threads", type=int, default=None, help="ffms2.Source threads (если используешь ffms2).")

    # Потоки VapourSynth (рендер/фильтры)
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Количество потоков VapourSynth (0 = авто).",
    )

    # Ускорение Python-цикла чтения props
    p.add_argument(
        "--prefetch",
        type=int,
        default=None,
        help="VideoNode.frames(prefetch=...) — сколько кадров рендерить параллельно (опционально).",
    )
    p.add_argument(
        "--backlog",
        type=int,
        default=None,
        help="VideoNode.frames(backlog=...) — размер очереди результатов (опционально).",
    )

    # Output
    p.add_argument("--csv", default=None, help="Сохранить пер-кадровые значения в CSV.")
    p.add_argument("--plot", default=None, help="Сохранить график в PNG по указанному пути (требуется matplotlib).")

    return p.parse_args()


# ------------------------------
# Loading
# ------------------------------

def _load_image(path: str) -> vs.VideoNode:
    if hasattr(core, "vszip") and hasattr(core.vszip, "ImageRead"):
        return core.vszip.ImageRead(path)
    if hasattr(core, "imwri"):
        return core.imwri.Read(path)
    raise RuntimeError("Для загрузки изображений нужен vszip.ImageRead или imwri.Read.")


def _load_video_ffms2(path: str, ffms2_threads: Optional[int]) -> vs.VideoNode:
    if not hasattr(core, "ffms2"):
        raise RuntimeError("ffms2 не найден.")
    kwargs = {}
    if ffms2_threads is not None:
        kwargs["threads"] = ffms2_threads
    return core.ffms2.Source(path, **kwargs)


def _load_video_bs(path: str, args: argparse.Namespace) -> vs.VideoNode:
    if not hasattr(core, "bs"):
        raise RuntimeError("BestSource (bs) не найден.")

    kwargs: dict[str, Any] = {}
    # Sig/параметры см. README BestSource; передаём только если задано. 
    if args.hwdevice is not None:
        kwargs["hwdevice"] = args.hwdevice
    if args.bs_threads is not None:
        kwargs["threads"] = args.bs_threads
    if args.bs_cachemode is not None:
        kwargs["cachemode"] = args.bs_cachemode
    if args.bs_cachepath is not None:
        kwargs["cachepath"] = args.bs_cachepath
    if args.bs_cachesize is not None:
        kwargs["cachesize"] = args.bs_cachesize
    if args.bs_seekpreroll is not None:
        kwargs["seekpreroll"] = args.bs_seekpreroll
    if args.bs_hwfallback is not None:
        kwargs["hwfallback"] = args.bs_hwfallback

    return core.bs.VideoSource(path, **kwargs)


def load_clip(path: str, args: argparse.Namespace) -> vs.VideoNode:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in IMAGE_EXTS:
        return _load_image(path)

    # Видео: по умолчанию bs (если есть), иначе ffms2 (можно переопределить)
    source = args.source
    if source == "auto":
        # Совместимость со старой версией: если есть ffms2 — используем его; иначе пробуем BestSource (bs)
        if hasattr(core, "ffms2"):
            return _load_video_ffms2(path, args.ffms2_threads)
        if hasattr(core, "bs"):
            return _load_video_bs(path, args)
        raise RuntimeError("Не найден ни ffms2, ни bs (BestSource) для загрузки видео.")
    if source == "bs":
        return _load_video_bs(path, args)
    if source == "ffms2":
        return _load_video_ffms2(path, args.ffms2_threads)

    raise RuntimeError(f"Неизвестный source: {source}")


# ------------------------------
# Transforms
# ------------------------------

def apply_ops(clip: vs.VideoNode, ops: Sequence[Op]) -> vs.VideoNode:
    for op in ops:
        if op.kind == "crop":
            _validate_crop_dims(clip, op.w, op.h)
            if op.x is None or op.y is None:
                # center crop
                left = max(0, (clip.width - op.w) // 2)
                top = max(0, (clip.height - op.h) // 2)
            else:
                left = max(0, min(op.x, max(0, clip.width - op.w)))
                top = max(0, min(op.y, max(0, clip.height - op.h)))
            clip = core.std.CropAbs(clip, width=op.w, height=op.h, left=left, top=top)
        elif op.kind == "scale":
            w, h = _resolve_scale_dims(clip, op.w, op.h)
            clip = core.resize.Lanczos(clip, width=w, height=h)
        else:
            raise RuntimeError(f"Неизвестная op.kind: {op.kind}")
    return clip


# ------------------------------
# Metric building
# ------------------------------

def prepare_pair(ref: vs.VideoNode, dist: vs.VideoNode, args: argparse.Namespace) -> Tuple[vs.VideoNode, vs.VideoNode, int, int]:
    """
    Возвращает (ref_clip, dist_clip, start_frame_original, step).
    start_frame_original нужен для корректного отображения индексов в исходном таймлайне.
    """
    # Выровнять таймлайны: сдвиг между клипами в кадрах.
    # >0: отбросить N кадров с начала distorted; <0: отбросить abs(N) кадров с начала reference.
    shift = int(getattr(args, "shift", 0) or 0)
    ref_offset = 0
    if shift > 0:
        if shift >= dist.num_frames:
            raise ValueError(f"--shift={shift} больше/равен длине distorted ({dist.num_frames}).")
        dist = dist.std.Trim(shift, dist.num_frames - 1)
    elif shift < 0:
        ref_offset = -shift
        if ref_offset >= ref.num_frames:
            raise ValueError(f"--shift={shift} по модулю больше/равен длине reference ({ref.num_frames}).")
        ref = ref.std.Trim(ref_offset, ref.num_frames - 1)

    # Ограничиваем общую длину (после сдвига)
    n = min(ref.num_frames, dist.num_frames)
    if n <= 0:
        raise RuntimeError("Нулевая длина клипов после --shift/обрезки.")

    start = max(0, args.start)
    end = (n - 1) if args.end < 0 else min(args.end, n - 1)
    if start > end:
        raise ValueError(f"--start ({start}) больше --end ({end}).")

    # Обрезаем одинаковый диапазон кадров в уже выровненных клипах
    ref = ref.std.Trim(start, end)
    dist = dist.std.Trim(start, end)

    # Индексы на исходном таймлайне reference
    start0 = ref_offset + start

    # Применяем шаг до пространственных фильтров, чтобы не считать лишние кадры (ускоряет --step N)
    step = max(1, args.step)
    if step > 1:
        ref = ref.std.SelectEvery(step, 0)
        dist = dist.std.SelectEvery(step, 0)

    # Раздельные цепочки (если заданы): сначала --left/--right, затем общие --crop/--scale
    common_ops = args.ops or []

    # parse_chain(None) -> [] ; parse_chain("") -> []
    left_ops = parse_chain(args.left) if args.left is not None else None
    right_ops = parse_chain(args.right) if args.right is not None else None

    if left_ops is None and right_ops is None:
        # Совместимость со старым поведением: общий --crop/--scale применяется к обоим
        left_ops = list(common_ops)
        right_ops = list(common_ops)
    else:
        # Если указана хотя бы одна из цепочек, применяем их отдельно,
        # а common_ops — к обоим (после), чтобы можно было нормализовать итоговый размер.
        left_ops = list(left_ops or []) + list(common_ops)
        right_ops = list(right_ops or []) + list(common_ops)

    if left_ops:
        ref = apply_ops(ref, left_ops)
    if right_ops:
        dist = apply_ops(dist, right_ops)

    if ref.width != dist.width or ref.height != dist.height:
        raise RuntimeError(
            f"После crop/scale размеры не совпадают: ref={ref.width}x{ref.height}, dist={dist.width}x{dist.height}."
        )

    # Приводим формат dist к формату ref, чтобы метрики не ругались
    if ref.format is not None and dist.format is not None and ref.format.id != dist.format.id:
        dist = core.resize.Bicubic(dist, format=ref.format.id)

    return ref, dist, start0, step


def _pick_backend(metric: str, backend: str) -> str:
    """
    Resolve 'auto' backend based on available plugins.
    """
    if backend != "auto":
        return backend

    if metric == "ssimulacra2":
        if hasattr(core, "vship"):
            return "vship"
        if hasattr(core, "vszip") and hasattr(core.vszip, "SSIMULACRA2"):
            return "vszip"
        return "cpu"

    # butteraugli
    if hasattr(core, "vship"):
        return "vship"
    if hasattr(core, "butteraugli"):
        return "cpu"
    return "cpu"


def build_metric_clip(ref: vs.VideoNode, dist: vs.VideoNode, args: argparse.Namespace) -> Tuple[vs.VideoNode, str, bool]:
    """
    Returns: (metric_clip, frameprop_key, higher_is_better)
    """
    metric = args.metric
    backend = _pick_backend(metric, args.backend)

    if metric == "ssimulacra2":
        higher_is_better = True

        if backend == "vship":
            if not hasattr(core, "vship"):
                raise RuntimeError("backend=vship выбран, но плагин vship не найден.")
            # vship: frame props обычно содержит '_SSIMULACRA2'
            mclip = core.vship.SSIMULACRA2(ref, dist, numStream=args.streams)
            return mclip, "_SSIMULACRA2", higher_is_better

        if backend in ("vszip", "cpu"):
            if not (hasattr(core, "vszip") and hasattr(core.vszip, "SSIMULACRA2")):
                raise RuntimeError("Нужен vapoursynth-zip (vszip) с фильтром SSIMULACRA2.")
            mclip = core.vszip.SSIMULACRA2(ref, dist)
            # ключ подберём динамически (разные версии могут отличаться)
            return mclip, "", higher_is_better

        raise RuntimeError(f"Неподдерживаемый backend для ssimulacra2: {backend}")

    # BUTTERAUGLI
    higher_is_better = False

    if backend == "vship":
        if not hasattr(core, "vship"):
            raise RuntimeError("backend=vship выбран, но плагин vship не найден.")

        # vship: пишет 2/3/INF нормы в props
        # (В некоторых версиях intensity_multiplier может отсутствовать — сделаем мягкий fallback.)
        try:
            mclip = core.vship.BUTTERAUGLI(
                ref, dist, distmap=0, numStream=args.streams, intensity_multiplier=args.butter_intensity
            )
        except TypeError:
            mclip = core.vship.BUTTERAUGLI(ref, dist, distmap=0, numStream=args.streams)

        norm = args.butter_norm
        key = {"2": "_BUTTERAUGLI_2Norm", "3": "_BUTTERAUGLI_3Norm", "inf": "_BUTTERAUGLI_INFNorm"}[norm]
        return mclip, key, higher_is_better

    # CPU Butteraugli plugin (vapoursynth-butteraugli)
    if not hasattr(core, "butteraugli"):
        raise RuntimeError(
            "CPU Butteraugli требует vapoursynth-butteraugli (core.butteraugli). "
            "Установи плагин или используй backend=vship."
        )

    # Плагин требует RGB24 и кладёт score в props['_Diff']
    ref_rgb = ref if (ref.format is not None and ref.format.id == vs.RGB24) else core.resize.Bicubic(ref, format=vs.RGB24)
    dist_rgb = dist if (dist.format is not None and dist.format.id == vs.RGB24) else core.resize.Bicubic(dist, format=vs.RGB24)

    mclip = core.butteraugli.butteraugli(ref_rgb, dist_rgb, heatmap=False)
    return mclip, "_Diff", higher_is_better


def find_metric_key(frame_props: Any) -> str:
    """
    Heuristic for vszip outputs (different versions use different keys).
    """
    keys = set(frame_props.keys())
    # Наиболее распространённые варианты для vszip.Metrics SSIMULACRA2
    for k in ("_SSIMULACRA2", "SSIMULACRA2", "_Metric", "Metric"):
        if k in keys:
            return k
    # last resort: pick first float-like prop
    for k in keys:
        v = frame_props.get(k)
        if isinstance(v, (float, int, np.floating, np.integer)):
            return k
    raise RuntimeError(f"Не удалось найти ключ метрики в frame props. Доступные ключи: {sorted(keys)}")


# ------------------------------
# Stats / output
# ------------------------------

def compute_stats(values: Sequence[float], higher_is_better: bool) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    stdev = float(arr.std(ddof=0))
    p1 = float(np.percentile(arr, 1))
    p5 = float(np.percentile(arr, 5))
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    worst = float(arr.min() if higher_is_better else arr.max())
    best = float(arr.max() if higher_is_better else arr.min())
    return {
        "mean": mean,
        "stdev": stdev,
        "p1": p1,
        "p5": p5,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "best": best,
        "worst": worst,
    }


def save_csv(path: str, indices: Sequence[int], values: Sequence[float], metric_name: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", metric_name])
        for i, v in zip(indices, values):
            w.writerow([i, v])


def save_plot(path: str, indices: Sequence[int], values: Sequence[float], metric_name: str) -> None:
    if plt is None:
        print("matplotlib не установлен — --plot недоступен.", file=sys.stderr)
        return
    plt.figure()
    plt.plot(indices, values)
    plt.xlabel("Frame")
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ------------------------------
# Main
# ------------------------------

def main() -> None:
    args = parse_args()

    # VapourSynth threads
    if args.threads and args.threads > 0:
        core.num_threads = args.threads

    ref = load_clip(args.reference, args)
    dist = load_clip(args.distorted, args)

    ref, dist, start0, step = prepare_pair(ref, dist, args)

    metric_clip, metric_key, higher_is_better = build_metric_clip(ref, dist, args)

    # Если vszip — подберём ключ по первому кадру
    if not metric_key:
        first = metric_clip.get_frame(0)
        metric_key = find_metric_key(first.props)

    total = metric_clip.num_frames

    # Индексы на исходном таймлайне
    indices = [start0 + (i * step) for i in range(total)]

    metric_name = args.metric
    if metric_name == "butteraugli" and _pick_backend("butteraugli", args.backend) == "vship":
        metric_name = f"butteraugli_{args.butter_norm}norm"

    # Быстрое чтение props: VideoNode.frames() рендерит кадры асинхронно; prefetch/backlog опциональны 
    frames_kwargs = {}
    if args.prefetch is not None:
        frames_kwargs["prefetch"] = args.prefetch
    if args.backlog is not None:
        frames_kwargs["backlog"] = args.backlog

    values: List[float] = []

    # Быстрое чтение props: VideoNode.frames() рендерит кадры асинхронно; prefetch/backlog опциональны.
    use_frames_api = hasattr(metric_clip, "frames") and callable(getattr(metric_clip, "frames"))
    if use_frames_api:
        it = metric_clip.frames(**frames_kwargs) if frames_kwargs else metric_clip.frames()
        it = islice(it, total)
        frame_iter: Iterable[Any] = it
    else:
        frame_iter = (metric_clip.get_frame(i) for i in range(total))
    if tqdm is not None:
        bar = tqdm(total=total, desc=f"Вычисление {metric_name}", unit="f")
    else:
        bar = None

    # Промежуточная статистика (скользящее среднее за последние N кадров)
    window = int(getattr(args, "stats", 0) or 0)
    rolling = deque(maxlen=window) if window > 0 else None
    postfix_every = 1
    if rolling is not None:
        # чтобы set_postfix не стал бутылочным горлышком
        postfix_every = max(1, rolling.maxlen // 10)

    for i, f in enumerate(frame_iter, start=1):
        try:
            v = f.props[metric_key]
        except Exception:
            # На случай, если ключ внезапно не найден — покажем props
            raise RuntimeError(
                f"Ключ метрики '{metric_key}' не найден в props. Доступно: {sorted(f.props.keys())}"
            )
        val = float(v)
        values.append(val)
        if rolling is not None:
            rolling.append(val)
            if len(rolling) == rolling.maxlen and (i % postfix_every == 0):
                rmean = sum(rolling) / len(rolling)
                if bar:
                    bar.set_postfix({f"avg{rolling.maxlen}": f"{rmean:.6f}"})
                else:
                    # Печатать реже, чтобы не засорять консоль
                    if i % rolling.maxlen == 0:
                        print(f"[{i}/{total}] avg(last {rolling.maxlen}) = {rmean:.6f}")
        if bar:
            bar.update(1)

    if bar:
        bar.close()

    stats = compute_stats(values, higher_is_better)

    print(f"Metric: {metric_name} (key={metric_key})")
    print(f"Frames used: {total} (start={args.start}, end={args.end}, step={args.step}, shift={getattr(args, 'shift', 0)})")
    print(f"Reference timeline start (after shift+start): {start0}")
    print(f"Mean:  {stats['mean']:.6f}")
    print(f"Stdev: {stats['stdev']:.6f}")
    print(f"P1:    {stats['p1']:.6f}")
    print(f"P5:    {stats['p5']:.6f}")
    print(f"P50:   {stats['p50']:.6f}")
    print(f"P95:   {stats['p95']:.6f}")
    print(f"P99:   {stats['p99']:.6f}")
    print(f"Best:  {stats['best']:.6f}")
    print(f"Worst: {stats['worst']:.6f}")

    if args.csv:
        save_csv(args.csv, indices, values, metric_name)
        print(f"CSV сохранён: {args.csv}")

    if args.plot:
        save_plot(args.plot, indices, values, metric_name)
        print(f"График сохранён: {args.plot}")


if __name__ == "__main__":
    main()
