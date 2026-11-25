#!/usr/bin/env python3
"""Convenience root launcher so users can run `python3 run.py ...` without setting PYTHONPATH."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_daily.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
