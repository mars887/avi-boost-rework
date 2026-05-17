#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.web_mp4 import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], program="python additional/make-web-mp4.py"))
