#!/usr/bin/env python3
"""Audited Transformers/PEFT launcher for inclusionAI/Ling-3.0-tiny."""
import sys
from pathlib import Path

STAGE1 = Path(__file__).resolve().parents[1]
SRC = STAGE1 / "src"
for path in (str(STAGE1), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from unsloth_native.cli import run  # noqa: E402

if __name__ == "__main__":
    run("ling3-tiny")

