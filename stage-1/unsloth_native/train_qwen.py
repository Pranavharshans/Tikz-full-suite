#!/usr/bin/env python3
"""Native text-only Unsloth launcher for Qwen/Qwen3.5-4B."""
import sys
from pathlib import Path

STAGE1 = Path(__file__).resolve().parents[1]
SRC = STAGE1 / "src"
for path in (str(STAGE1), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from unsloth_native.cli import run  # noqa: E402

if __name__ == "__main__":
    run("qwen3.5-4b")

