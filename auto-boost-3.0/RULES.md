# Rules File Guide (auto-boost 3.0)

Этот документ описывает формат файла правил, доступные функции, метрики и двухпроходную модель вычислений.

## Где и как подключается

Запуск:
- `--rules <path>` — путь к Python‑файлу правил.
- `--rules-inline "<code>"` — правила строкой.

Правила выполняются **по каждой сцене** (scene) из `scenes.json`.

## Два прохода

Правила выполняются в 2 прохода:

1) **Pass 1** — сбор требований:
   - `metric()` доступен только для "дешёвых" метрик (bitrate, ssimu2).
   - Для "дорогих" метрик нужно использовать `require()`.
   - Переменная `rule_pass == 1`.

2) **Pass 2** — применение:
   - Все требуемые метрики уже вычислены.
   - `metric()` возвращает реальные значения.
   - `require()` существует, но ничего не делает.
   - Переменная `rule_pass == 2`.

В коде правил можно проверять `rule_pass`/`RULE_PASS`.

## Доступные функции

В каждом правиле доступны:

- `metric(name: str) -> float`  
  Получить значение метрики для текущей сцены.

- `metric_at(scene_index: int, name: str) -> float`  
  Получить значение метрики для указанной сцены (по индексу).

- `require(name: str) -> None`  
  Заявить, что метрика нужна. В pass1 это создаёт расписание вычислений (NVOF/Noise/Luma).

- `param(name: str) -> Any`  
  Прочитать текущий параметр кодера (из `video_params` текущей сцены).

- `sparam(name: str, value: Any = None) -> None`  
  Установить параметр (или флаг, если value is None/True).

- `cparam(name: str, delta: Any) -> None`  
  Изменить числовой параметр на дельту (например, `cparam("--crf", +0.5)`).

- `log(*args)`  
  Лог в stdout, полезно для отладки.

## Данные текущей сцены

В правилах доступны переменные:

- `scene_index` — индекс текущей сцены (0..N-1)
- `scene_start` — стартовый кадр
- `scene_end` — конечный кадр (exclusive)
- `scene_len` — длительность сцены в кадрах
- `scene_seconds` — длительность сцены в секундах
- `scene_fps` — FPS, использованный для расчёта секунд
- `scene_count` — всего сцен

## Дешёвые метрики (доступны в pass1)

Эти метрики не требуют внешних вычислений:

- `scene_bitrate`  
- `scene_bitrate_ratio`  
- `ssimu2`  
- `ssimu2_p5`  
- `ssimu2_avg`  

## Метрики NVOF/Noise/Luma (нужен `require()`)

Эти метрики вычисляются внешним скриптом `nvof_noise_est_opt_main.py` и требуют **pass1** + `require()`.

### NVOF
- `scene_nvof_avg` — среднее по сцене
- `global_nvof_avg` — среднее по всему видео

### Luma (из NVOF‑скрипта)
- `scene_luma_avg`
- `global_luma_avg`
- `scene_luma_p25`
- `global_luma_p25`
- `scene_luma_p75`
- `global_luma_p75`
- `luma_ratio` — scene_luma_avg / global_luma_avg

### Noise
- `scene_noise_sigma`
- `global_noise_sigma`
- `scene_grain_ratio`
- `global_grain_ratio`

## Алиасы метрик

Для удобства зарегистрированы алиасы:

- `luma_avg` → `scene_luma_avg`
- `luma_mean` → `scene_luma_avg`
- `luma_p25` → `scene_luma_p25`
- `luma_p75` → `scene_luma_p75`
- `nvof_mean` → `scene_nvof_avg`
- `noise_sigma` → `scene_noise_sigma`
- `grain_ratio` → `scene_grain_ratio`

## Примеры

### 1) Простая корректировка CRF по SSIMU2
```python
if metric("ssimu2") < 0.6:
    cparam("--crf", -1.0)
```

### 2) NVOF метрики: требуем и используем
```python
if rule_pass == 1:
    require("scene_nvof_avg")
    require("global_nvof_avg")
else:
    if metric("scene_nvof_avg") > metric("global_nvof_avg") * 1.2:
        cparam("--crf", -0.5)
```

### 3) Luma ratio
```python
if rule_pass == 1:
    require("luma_ratio")
else:
    if metric("luma_ratio") < 0.9:
        cparam("--crf", -0.3)
```

### 4) Проверка pass‑1 (не обязательна)
```python
if rule_pass == 1:
    # только сбор требований
    require("scene_noise_sigma")
else:
    if metric("scene_noise_sigma") > 2.0:
        cparam("--crf", -0.25)
```

## Как определяется диапазон вычислений

- Для `scene_*` метрик вычисляются только нужные сцены.
- Для `global_*` метрик вычисляется всё видео.
- Если метрика не указана через `require()` в pass1, скрипт может посчитать **весь** ролик (fallback).

## Отладка

Полезно использовать `--verbose` для логов.

---
Если нужны дополнительные метрики — добавляй через `ab_metrics_nvof.py` или отдельный модуль + регистрацию в реестре.
