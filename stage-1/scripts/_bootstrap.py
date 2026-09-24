"""Add ``stage-1/src`` to ``sys.path`` so scripts run from a checkout.

This keeps every entry point usable without installing the package, matching
the repository's "clone and run" convention.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
