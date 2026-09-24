"""Add ``stage-1/src`` to ``sys.path`` for every test module.

Importing any ``tests.test_*`` module is therefore enough to make the
``stage1`` package importable, whether discovery runs from the repository root
or from ``stage-1/``.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
