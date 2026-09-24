#!/usr/bin/env python3
"""Merge a saved Stage 1 LoRA adapter into its pinned BF16 base model.

This command is always separate from training: nothing in the training path
calls it, and it never runs automatically. It:

1. verifies the adapter artifact (identity, model, revision, base provenance),
2. loads the exact pinned BF16 base model **base-only** (no fresh adapter),
3. attaches the saved adapter exactly once,
4. records base-plus-adapter logits on a fixed input, merges the adapter
   (``merge_and_unload``) and checks the merged logits match within
   ``--tolerance`` (LoRA dropout is disabled in eval mode),
5. writes the merged model, tokenizer and a ``merge-metadata.json`` record.

Example:
  python3 stage-1/scripts/merge_adapter.py \
    --config stage-1/configs/minicpm5-2b-lora.yaml \
    --export /shared/$USER/tikz-production/export \
    --prepared /shared/$USER/tikz-stage1/prepared \
    --run-dir /shared/$USER/tikz-stage1/runs/minicpm5-2b-lora \
    --adapter /shared/$USER/tikz-stage1/runs/minicpm5-2b-lora/final/smoke-1000 \
    --out /shared/$USER/tikz-stage1/merged/minicpm5-2b-smoke
"""
import argparse
import sys

import _bootstrap  # noqa: F401

from stage1 import (adapters, checkpointing, cli, config as config_module, data,
                    formatting, identity, merge)
from stage1.errors import DataError
from stage1.util import (dependency_versions, detect_repo_commit,
                         require_absolute, utc_now_iso)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge a saved LoRA adapter into the pinned base model "
                    "(never automatic).")
    parser.add_argument("--config", required=True, help="LoRA model config YAML")
    parser.add_argument("--export", default=None,
                        help="absolute cleaning export directory (or data.export_dir)")
    parser.add_argument("--prepared", default=None,
                        help="absolute prepared directory (or data.prepared_dir)")
    parser.add_argument("--run-dir", required=True,
                        help="absolute run directory (identity must already exist)")
    parser.add_argument("--adapter", required=True,
                        help="absolute path to a Stage 1 LoRA artifact")
    parser.add_argument("--out", required=True,
                        help="absolute output directory for the merged model")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="maximum allowed |base+adapter - merged| logit "
                             "difference (default 1e-3)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip the logit-equivalence check (not recommended)")
    parser.add_argument("--local-files-only", action="store_true",
                        help="refuse to download models; use local snapshots only")
    parser.add_argument("--cache-dir", default=None,
                        help="optional Hugging Face cache directory (absolute)")
    return parser


def main(argv) -> int:
    args = build_parser().parse_args(argv)
    if args.tolerance <= 0:
        raise DataError("--tolerance must be positive")
    config = config_module.load_config(args.config)
    if config.training.method != "lora":
        raise DataError(
            f"{args.config} is a {config.training.method!r} configuration; "
            "adapter merging requires training.method: lora")
    paths = config_module.resolve_run_paths(
        config, export=args.export, prepared=args.prepared, run_dir=args.run_dir)
    paths.validate_inputs()
    adapter_dir = require_absolute(args.adapter, "--adapter")
    out_dir = require_absolute(args.out, "--out")
    if args.cache_dir:
        config_module.require_absolute(args.cache_dir, "--cache-dir")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataError("torch is required to merge an adapter") from exc

    prepared = data.verify_prepared(paths.prepared_dir,
                                    adapter_slug=config.model.adapter)
    eligibility = data.load_eligibility(paths.prepared_dir, config.model.adapter)
    export_info = data.verify_export(
        paths.export_dir, require_complete=config.data.require_complete_export,
        quick=False)
    tokenizer = adapters.load_tokenizer(
        config, local_files_only=args.local_files_only, cache_dir=args.cache_dir)
    template = formatting.resolve_chat_template(tokenizer, config.tokenizer)
    fingerprint = adapters.tokenizer_fingerprint(tokenizer, template, config)
    problems = adapters.compare_fingerprints(prepared["model"], fingerprint)
    if problems:
        raise DataError("Tokenizer fingerprint differs from preparation:\n" +
                        "\n".join(f"  - {line}" for line in problems))
    run_identity = identity.build_run_identity(
        data_identity=prepared["manifest"]["data_identity"],
        config=config, model_fingerprint=fingerprint,
        dataset_report_sha256=prepared["report_sha256"],
        dependencies=dependency_versions(),
        code_sha256=identity.stage1_code_sha256(),
        repo_commit=detect_repo_commit(identity.repository_root()))
    identity.ensure_run_record(paths.run_dir, run_identity)

    artifact_meta = checkpointing.verify_evaluation_artifact(
        adapter_dir, identity_sha256=run_identity["sha256"],
        model_id=config.model.id, model_revision=config.model.revision,
        adapter=config.model.adapter)
    if artifact_meta["training_method"] != "lora":
        raise DataError(
            f"{adapter_dir} is a {artifact_meta['training_method']!r} artifact; "
            "only LoRA adapters can be merged")
    if (artifact_meta.get("base_model_id") != config.model.id
            or artifact_meta.get("base_model_revision") != config.model.revision):
        raise DataError(
            f"Adapter {adapter_dir} was saved for base "
            f"{artifact_meta.get('base_model_id')}@"
            f"{artifact_meta.get('base_model_revision')}, but this config pins "
            f"{config.model.id}@{config.model.revision}")

    split = "validation" if eligibility["by_split"]["validation"]["eligible"] else "train"
    ids = data.select_split_ids(prepared["manifest"], split, limit=1,
                                seed=config.seed,
                                exclude=eligibility["quarantined"])
    rows = data.load_rows_by_ids(export_info, ids)
    if not rows:
        raise DataError("no eligible row is available for the merge check")
    example = formatting.format_example(
        tokenizer, row_id=rows[0]["id"], instruction=rows[0]["instruction"],
        tikz=rows[0]["tikz_code"], template=template,
        kwargs=config.tokenizer.chat_template_kwargs)
    batch = {
        "input_ids": torch.tensor([example.input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([[1] * len(example.input_ids)],
                                       dtype=torch.long),
        "position_ids": torch.tensor([list(range(len(example.input_ids)))],
                                     dtype=torch.long),
    }

    print(f"loading the pinned base model {config.model.id} base-only and "
          "attaching the saved adapter exactly once", file=sys.stderr)
    result = merge.merge_adapter(
        config, adapter_dir, out_dir, batch=batch,
        tolerance=args.tolerance, verify=not args.skip_verify,
        local_files_only=args.local_files_only, cache_dir=args.cache_dir,
        metadata={"run_identity_sha256": run_identity["sha256"],
                  "sample_row_id": rows[0]["id"],
                  "sample_tokens": len(example.input_ids),
                  "created_at": utc_now_iso()})
    print(f"merged model written to {result['out_dir']}")
    print(f"verification: {result['verification']}")
    if not result["verification"].get("checked"):
        print("warning: logit equivalence was skipped for this merge",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(cli.run(main))
