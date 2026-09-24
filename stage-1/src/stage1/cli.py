"""Shared CLI plumbing: exit codes and error rendering.

Scripts call ``stage1.cli.run(main)`` where ``main(argv) -> int``. The helper
maps Stage 1 errors onto the documented exit codes and never swallows
tracebacks for unexpected programming defects.
"""
from __future__ import annotations

import sys

from .errors import ConfigError, GateError, GateFailed, Stage1Error


def run(main) -> int:
    try:
        return int(main(sys.argv[1:]))
    except GateFailed as exc:
        print(f"gate failed: {exc}", file=sys.stderr)
        return 1
    except GateError as exc:
        print(f"gate blocked: {exc}", file=sys.stderr)
        return 3
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except Stage1Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
