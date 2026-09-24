#!/usr/bin/env python3
"""Compare Stage 1 runs and evaluations.

Accepts training metrics files (run_dir/metrics/<gate>.json) and evaluation
metrics files (run_dir/evaluations/<name>/metrics.json or the directory).
Writes comparison.json plus a readable comparison.md. Runs are compared by
supervised tokens as well as quality metrics; missing fields stay explicit.

Example:
  python3 stage-1/scripts/compare_models.py \
    --metrics /runs/qwen/metrics/smoke-1000.json \
    --metrics /runs/minicpm/metrics/smoke-1000.json \
    --metrics /runs/minicpm/evaluations/sft-full-test \
    --labels qwen-smoke minicpm-smoke minicpm-eval \
    --out /runs/comparison
"""
import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from stage1 import cli, report
from stage1.errors import ConfigError
from stage1.util import require_absolute


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a machine-readable and readable run comparison.")
    parser.add_argument("--metrics", action="append", required=True,
                        metavar="PATH_OR_DIR",
                        help="metrics JSON file or a directory containing metrics.json")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="one label per --metrics entry, in the same order")
    parser.add_argument("--out", required=True,
                        help="absolute output directory for comparison.json/md")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="exit 0 even when base/trained artifacts are not "
                             "yet comparable")
    parser.add_argument("--allow-non-comparable", action="store_true",
                        help="exit 0 even when candidates disagree on training "
                             "method, dataset identity, sequence limit or "
                             "evaluation set")
    return parser


def _resolve_metrics(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "metrics.json"
    if not candidate.is_file():
        raise ConfigError(f"Metrics file not found: {candidate}")
    return candidate


def main(argv) -> int:
    args = build_parser().parse_args(argv)
    if len(args.labels) != len(args.metrics):
        raise ConfigError(
            f"--labels has {len(args.labels)} entries but --metrics has "
            f"{len(args.metrics)}; they must match")
    entries = []
    for label, path in zip(args.labels, args.metrics):
        entries.append((label, report.load_metrics_file(_resolve_metrics(path))))
    comparison = report.compare_metrics(
        entries, allow_non_comparable=args.allow_non_comparable)
    written = report.write_comparison(
        require_absolute(args.out, "--out"), comparison)
    print(report.render_comparison(comparison))
    print(f"comparison written to {written['json']} and {written['markdown']}")
    readiness = comparison["experiment_readiness"]
    exit_code = 0
    if readiness["problems"] and not args.allow_incomplete:
        exit_code = 1
    if readiness["comparability_problems"] and not args.allow_non_comparable:
        exit_code = 1
    if exit_code:
        print("comparison is NOT READY: " + "; ".join(
            list(readiness["problems"])
            + list(readiness["comparability_problems"])), file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(cli.run(main))
