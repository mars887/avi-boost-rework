from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DISCORD_CONFIG = ROOT_DIR / "discord-config.txt"


def read_discord_config(path: Path | str = DEFAULT_DISCORD_CONFIG) -> Dict[str, str]:
    config_path = Path(path).expanduser()
    if not config_path.exists() or not config_path.is_file():
        return {}

    values: Dict[str, str] = {}
    for raw_line in config_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("set "):
            line = line[4:].strip()
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def discord_config_value(key: str, default: str = "", *, file_values: Dict[str, str] | None = None) -> str:
    env_value = os.environ.get(key)
    if env_value is not None and str(env_value).strip():
        return env_value
    values = file_values if file_values is not None else read_discord_config()
    file_value = values.get(key)
    if file_value is not None and str(file_value).strip():
        return file_value
    return default


def discord_config_int(key: str, default: int = 0, *, file_values: Dict[str, str] | None = None) -> int:
    raw = discord_config_value(key, str(default), file_values=file_values)
    try:
        return int(str(raw).strip())
    except Exception:
        return default
