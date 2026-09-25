#!/usr/bin/env python3
"""Native Unsloth launcher for openbmb/MiniCPM5-2B."""
import sys
from pathlib import Path

STAGE1 = Path(__file__).resolve().parents[1]
SRC = STAGE1 / "src"
for path in (str(STAGE1), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from unsloth_native.cli import run  # noqa: E402

if __name__ == "__main__":
    run("minicpm5-2b")

