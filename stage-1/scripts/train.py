#!/usr/bin/env python3
"""Run one Stage 1 training gate: overfit-100, smoke-1000 or full.

Gates never advance automatically. Each gate refuses to start without
preflight evidence for the same run identity, and later gates additionally
require the previous gate's evidence. A passed gate is not silently repeated:
pass --rerun-gate to do that deliberately.

Examples:
  # 100-example overfit gate
  python3 stage-1/scripts/train.py \
    --config stage-1/configs/minicpm5-2b-full.yaml \
    --export /shared/$USER/tikz-production/export \
    --prepared /shared/$USER/tikz-stage1/prepared \
    --run-dir /shared/$USER/tikz-stage1/runs/minicpm5-2b \
    --gate overfit-100

  # resume an interrupted full run (same command)
  python3 stage-1/scripts/train.py ... --gate full
"""
import argparse
import sys

import _bootstrap  # noqa: F401

from stage1 import cli, config as config_module, train as train_module

TRAINING_GATES = ("overfit-100", "smoke-1000", "full")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-parameter BF16 SFT with Unsloth (one gate per run).")
    parser.add_argument("--config", required=True, help="model config YAML")
    parser.add_argument("--export", default=None,
                        help="absolute cleaning export directory (or data.export_dir)")
    parser.add_argument("--prepared", default=None,
                        help="absolute prepared directory (or data.prepared_dir)")
    parser.add_argument("--run-dir", required=True,
                        help="absolute isolated run directory")
    parser.add_argument("--gate", required=True, choices=TRAINING_GATES,
                        help="training gate to run (data and preflight are "
                             "separate commands and are never run implicitly)")
    parser.add_argument("--local-files-only", action="store_true",
                        help="refuse to download models; use local snapshots only")
    parser.add_argument("--cache-dir", default=None,
                        help="optional Hugging Face cache directory (absolute)")
    parser.add_argument("--rerun-gate", action="store_true",
                        help="deliberately repeat a gate that already passed")
    return parser


def main(argv) -> int:
    args = build_parser().parse_args(argv)
    config = config_module.load_config(args.config)
    paths = config_module.resolve_run_paths(
        config, export=args.export, prepared=args.prepared, run_dir=args.run_dir)
    paths.validate_inputs()
    if args.cache_dir:
        config_module.require_absolute(args.cache_dir, "--cache-dir")
    metrics = train_module.run_training(
        config, paths, args.gate, local_files_only=args.local_files_only,
        cache_dir=args.cache_dir, allow_rerun=args.rerun_gate,
        progress=lambda message: print(message, file=sys.stderr))
    print(f"gate {args.gate}: {metrics['optimizer_steps']} steps, "
          f"{metrics['supervised_tokens_seen']} supervised tokens, "
          f"eval loss {metrics['final_eval_loss']}")
    print(f"metrics: {metrics['metrics_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(cli.run(main))
