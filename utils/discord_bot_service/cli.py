from __future__ import annotations

import argparse
import asyncio
from typing import List, Optional

from .service import run_bot
from .settings import load_config

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PBBatchProcessUtil Discord bot service.")
    parser.add_argument("--token", default="")
    parser.add_argument("--guild-id", default="")
    parser.add_argument("--category-id", default="")
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--db", default="")
    parser.add_argument("--max-upload-mb", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="Enable verbose Discord bot diagnostics in stdout.")
    args = parser.parse_args(argv)
    config = load_config(args)
    asyncio.run(run_bot(config))
    return 0
