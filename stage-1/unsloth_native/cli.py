"""CLI shared by the model-specific native launchers."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stage1 import cli as stage1_cli

from .models import get_spec
from .runner import run_native


def parser(model_key: str) -> argparse.ArgumentParser:
    spec = get_spec(model_key)
    result = argparse.ArgumentParser(
        description=f"Native Unsloth + TRL SFT for {model_key}")
    result.add_argument("--method", choices=("lora", "full"), default="lora")
    result.add_argument("--config", default=None,
                        help="optional config override; method must still match")
    result.add_argument("--export", required=True)
    result.add_argument("--prepared", required=True)
    result.add_argument("--run-dir", required=True)
    result.add_argument("--gate", required=True,
                        choices=("overfit-100", "smoke-1000", "full"))
    result.add_argument("--resume", default="auto",
                        help="auto, none, or an explicit native checkpoint")
    result.add_argument("--cache-dir", default=None)
    result.add_argument("--local-files-only", action="store_true")
    result.set_defaults(_spec=spec)
    return result


def main(model_key: str, argv=None) -> int:
    args = parser(model_key).parse_args(argv)
    config_path = Path(args.config).resolve() if args.config else args._spec.config_path(
        args.method)
    metrics = run_native(
        model_key,
        config_path=config_path,
        expected_method=args.method,
        export=args.export,
        prepared=args.prepared,
        run_dir=args.run_dir,
        gate_name=args.gate,
        resume=args.resume,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    print(f"native gate {metrics['gate']} complete: {metrics['final_dir']}")
    return 0


def run(model_key: str) -> None:
    raise SystemExit(stage1_cli.run(lambda argv: main(model_key, argv)))
