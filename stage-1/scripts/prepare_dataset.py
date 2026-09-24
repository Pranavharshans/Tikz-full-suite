#!/usr/bin/env python3
"""Prepare the Stage 1 dataset from the audited cleaning export.

Verifies the export, groups duplicate TikZ/image content, creates deterministic
train/validation/test splits, and writes per-model token-length reports and
overlength quarantine files. Run once per dataset; the output directory is
shared by every model and by preflight/training/evaluation.

Example:
  python3 stage-1/scripts/prepare_dataset.py \
    --config stage-1/configs/qwen3.5-4b-full.yaml \
    --config stage-1/configs/minicpm5-2b-full.yaml \
    --export /shared/$USER/tikz-production/export \
    --prepared /shared/$USER/tikz-stage1/prepared
"""
import argparse
import dataclasses
import json
import sys

import _bootstrap  # noqa: F401

from stage1 import adapters, cli, config as config_module, data
from stage1.util import canonical_digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the cleaning export and build deterministic Stage 1 splits.")
    parser.add_argument("--config", action="append", required=True, metavar="PATH",
                        help="model config YAML; repeat for both tokenizers")
    parser.add_argument("--export", default=None,
                        help="absolute path to the cleaning export directory")
    parser.add_argument("--prepared", default=None,
                        help="absolute output directory for prepared artifacts")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the configs and print the resolved plan; "
                             "reads nothing and writes nothing")
    parser.add_argument("--local-files-only", action="store_true",
                        help="refuse to download tokenizers; use local snapshots only")
    parser.add_argument("--cache-dir", default=None,
                        help="optional Hugging Face cache directory (absolute)")
    return parser


def main(argv) -> int:
    args = build_parser().parse_args(argv)
    configs = [config_module.load_config(path) for path in args.config]
    for config in configs:
        adapters.validate_config_against_adapter(config)
    if args.dry_run:
        for path, config in zip(args.config, configs):
            print(json.dumps({
                "config": str(path),
                "model": config.model.id,
                "revision": config.model.revision,
                "adapter": config.model.adapter,
                "loader": config.model.loader,
                "training_method": config.training.method,
                "lora": config.lora.to_jsonable() if config.lora else None,
                "effective_batch_size": config.training.effective_batch_size,
                "hardware_profile": config.hardware.profile,
                "min_vram_gib": config.hardware.min_vram_gib,
                "max_seq_len": config.data.max_seq_len,
                "splits": dataclasses.asdict(config.data.splits),
                "gates": {name: config.gate(name).to_jsonable()
                          for name in ("overfit-100", "smoke-1000", "full")},
                "lock_file": str(config.environment.lock_path),
                "lock_sha256": config.environment.lock_sha256,
                "identity_payload_sha256": canonical_digest(config.identity_payload()),
            }, indent=2, sort_keys=True))
        return 0
    export_dir = config_module.require_absolute(args.export, "--export")
    prepared_dir = config_module.require_absolute(args.prepared, "--prepared")
    if args.cache_dir:
        config_module.require_absolute(args.cache_dir, "--cache-dir")
    report = data.prepare_dataset(
        configs, export_dir=export_dir, prepared_dir=prepared_dir,
        local_files_only=args.local_files_only, cache_dir=args.cache_dir,
        progress=lambda message: print(message, file=sys.stderr))
    identity = report["data_identity"]
    print(f"Prepared {identity['row_count']} rows from "
          f"{report['export']['rows']} exported rows in "
          f"{report['export']['shards']} shards")
    for slug, summary in report["models"].items():
        print(f"  {slug}: {summary['quarantined']} quarantined, "
              f"max_seq_len {summary['max_seq_len']}")
    print(f"  dataset logical sha256: {identity['dataset_logical_sha256']}")
    print(f"  split manifest sha256:  {identity['split_manifest_sha256']}")
    print(f"  data identity sha256:   {identity['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(cli.run(main))
