#!/usr/bin/env python3
"""RTX PRO 6000 preflight: prove the environment before any training gate.

Checks the pinned environment, GPU identity and VRAM, BF16, torch/CUDA, fused
AdamW, Unsloth loading of the exact checkpoint, dataset identity, disk space, a
real forward/backward step with finite loss and gradients, a checkpoint
save/reload round trip, and peak VRAM. It never starts training.

Exit codes: 0 = all checks passed (no skips), 1 = a check failed or a check was
skipped (a skipped preflight is not a passing preflight), 2 = handled error.

Example:
  python3 stage-1/scripts/preflight.py \
    --config stage-1/configs/qwen3.5-4b-full.yaml \
    --export /shared/$USER/tikz-production/export \
    --prepared /shared/$USER/tikz-stage1/prepared \
    --run-dir /shared/$USER/tikz-stage1/runs/qwen3.5-4b
"""
import argparse
import sys

import _bootstrap  # noqa: F401

from stage1 import (adapters, cli, config as config_module, data, formatting,
                    identity, preflight)
from stage1.errors import DataError
from stage1.util import dependency_versions, detect_repo_commit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify that this machine can train the configured model.")
    parser.add_argument("--config", required=True, help="model config YAML")
    parser.add_argument("--export", default=None,
                        help="absolute cleaning export directory (or data.export_dir)")
    parser.add_argument("--prepared", default=None,
                        help="absolute prepared directory (or data.prepared_dir)")
    parser.add_argument("--run-dir", required=True,
                        help="absolute run directory; run.json is written here")
    parser.add_argument("--local-files-only", action="store_true",
                        help="refuse to download models; use local snapshots only")
    parser.add_argument("--cache-dir", default=None,
                        help="optional Hugging Face cache directory (absolute)")
    parser.add_argument("--skip-check", action="append", default=[],
                        metavar="NAME",
                        help="explicitly skip a check (recorded; blocks gates)")
    return parser


def main(argv) -> int:
    args = build_parser().parse_args(argv)
    # Unsloth must patch Transformers before AutoTokenizer imports it below.
    # This stays after argument parsing so ``--help`` remains usable on a
    # CPU/login node where Unsloth deliberately refuses to initialize.
    try:
        import unsloth  # noqa: F401
    except Exception as exc:
        raise DataError(
            f"Unable to initialize Unsloth before Transformers: "
            f"{type(exc).__name__}: {exc}") from exc
    config = config_module.load_config(args.config)
    paths = config_module.resolve_run_paths(
        config, export=args.export, prepared=args.prepared, run_dir=args.run_dir)
    paths.validate_inputs()
    if args.cache_dir:
        config_module.require_absolute(args.cache_dir, "--cache-dir")

    prepared = data.verify_prepared(paths.prepared_dir,
                                    adapter_slug=config.model.adapter)

    # The run identity needs the tokenizer fingerprint, so load it once here;
    # the check reuses it instead of loading twice.
    tokenizer = adapters.load_tokenizer(
        config, local_files_only=args.local_files_only, cache_dir=args.cache_dir)
    template = formatting.resolve_chat_template(tokenizer, config.tokenizer)
    fingerprint = adapters.tokenizer_fingerprint(tokenizer, template, config)
    problems = adapters.compare_fingerprints(prepared["model"], fingerprint)
    if problems:
        raise data.DataError(
            "Tokenizer fingerprint differs from preparation:\n" +
            "\n".join(f"  - {line}" for line in problems))

    run_identity = identity.build_run_identity(
        data_identity=prepared["manifest"]["data_identity"],
        config=config, model_fingerprint=fingerprint,
        dataset_report_sha256=prepared["report_sha256"],
        dependencies=dependency_versions(),
        code_sha256=identity.stage1_code_sha256(),
        repo_commit=detect_repo_commit(identity.repository_root()))
    identity.ensure_run_record(paths.run_dir, run_identity)

    context = preflight.PreflightContext(
        config=config, run_dir=paths.run_dir, identity=run_identity,
        prepared=prepared, export_dir=paths.export_dir,
        prepared_dir=paths.prepared_dir,
        tokenizer=tokenizer, local_files_only=args.local_files_only,
        cache_dir=args.cache_dir)
    context.scratch["template"] = template
    context.scratch["fingerprint"] = fingerprint
    report = preflight.run_preflight(context, skip=args.skip_check)
    identity.write_preflight_report(
        paths.run_dir, identity_sha256=run_identity["sha256"],
        checks=report["checks"], status=report["status"],
        complete=report["complete"],
        payload={
            "failed": report["failed"],
            "skipped": report["skipped"],
            "peak_vram_bytes": report["peak_vram_bytes"],
            "load_report": report["load_report"],
            "data_identity_sha256": prepared["data_identity_sha256"],
            "started_training": False,
        })

    for check in report["checks"]:
        marker = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}[check["status"]]
        print(f"[{marker}] {check['name']}: {check['detail']}")
    print(f"\npreflight {report['status']} "
          f"({len(report['failed'])} failed, {len(report['skipped'])} skipped); "
          f"report: {identity.preflight_report_path(paths.run_dir)}")
    if report["status"] == "passed" and report["complete"]:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(cli.run(main))
