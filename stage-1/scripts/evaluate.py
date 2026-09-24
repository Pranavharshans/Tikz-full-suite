#!/usr/bin/env python3
"""Evaluate the base model or a trained checkpoint on a held-out split.

Generation, extraction, safe TeX compilation, duplicate/memorization checks and
(where Pillow and pdftoppm are available) rendered-image similarity. Writes a
machine-readable metrics file plus per-row details into the run directory.

Examples:
  # base model on 50 test rows
  python3 stage-1/scripts/evaluate.py \
    --config stage-1/configs/minicpm5-2b-full.yaml \
    --export /shared/$USER/tikz-production/export \
    --prepared /shared/$USER/tikz-stage1/prepared \
    --run-dir /shared/$USER/tikz-stage1/runs/minicpm5-2b \
    --name base-test-50 --base --max-examples 50

  # final checkpoint on the full test split
  python3 stage-1/scripts/evaluate.py ... --name sft-full-test \
    --checkpoint /shared/$USER/tikz-stage1/runs/minicpm5-2b/final --all
"""
import argparse
import dataclasses
import json
import sys

import _bootstrap  # noqa: F401

from stage1 import (adapters, checkpointing, cli, config as config_module, data,
                    evaluate, formatting, generate, identity)
from stage1.util import (dependency_versions, detect_repo_commit, utc_now_iso,
                         write_json_atomic, write_text_atomic)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and score model outputs on a held-out split.")
    parser.add_argument("--config", required=True, help="model config YAML")
    parser.add_argument("--export", default=None,
                        help="absolute cleaning export directory (or data.export_dir)")
    parser.add_argument("--prepared", default=None,
                        help="absolute prepared directory (or data.prepared_dir)")
    parser.add_argument("--run-dir", required=True,
                        help="absolute run directory (run.json must already exist)")
    parser.add_argument("--name", required=True,
                        help="evaluation name; output goes to evaluations/<name>/")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--base", action="store_true",
                        help="evaluate the pinned base checkpoint (default)")
    source.add_argument("--checkpoint", default=None,
                        help="absolute path to a Stage 1 checkpoint or final artifact")
    parser.add_argument("--expect-gate", default=None,
                        choices=("overfit-100", "smoke-1000", "full"),
                        help="require the artifact to belong to this gate")
    parser.add_argument("--split", default="test",
                        choices=("validation", "test"),
                        help="held-out split to evaluate (default: test)")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="cap the number of examples (default: evaluation.max_examples)")
    parser.add_argument("--all", action="store_true",
                        help="evaluate the whole split (overrides --max-examples)")
    parser.add_argument("--no-compile", action="store_true",
                        help="skip TeX compilation (metrics mark it disabled)")
    parser.add_argument("--local-files-only", action="store_true",
                        help="refuse to download models; use local snapshots only")
    parser.add_argument("--cache-dir", default=None,
                        help="optional Hugging Face cache directory (absolute)")
    return parser


def main(argv) -> int:
    args = build_parser().parse_args(argv)
    config = config_module.load_config(args.config)
    paths = config_module.resolve_run_paths(
        config, export=args.export, prepared=args.prepared, run_dir=args.run_dir)
    paths.validate_inputs()
    if args.checkpoint:
        config_module.require_absolute(args.checkpoint, "--checkpoint")
    if args.cache_dir:
        config_module.require_absolute(args.cache_dir, "--cache-dir")

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

    if args.all:
        limit = None
    elif args.max_examples is not None:
        limit = args.max_examples
    else:
        limit = config.evaluation.max_examples
    quarantined = eligibility["quarantined"]
    ids = evaluate.select_evaluation_ids(
        prepared["manifest"], args.split, limit=limit, seed=config.seed,
        eligibility=eligibility)
    rows = data.load_rows_by_ids(export_info, ids, include_image=True)
    if not rows:
        raise data.DataError(f"No eligible rows selected from split {args.split!r}")

    train_sample_ids = evaluate.select_memorization_ids(
        prepared["manifest"], limit=config.evaluation.memorization_sample,
        seed=config.seed, eligibility=eligibility)
    train_rows = data.load_rows_by_ids(export_info, train_sample_ids)
    train_targets = [(row["id"], row["tikz_code"]) for row in train_rows]

    artifact = {"kind": "base", "verified": True,
                "identity_sha256": run_identity["sha256"],
                "model_id": config.model.id,
                "model_revision": config.model.revision,
                "adapter": config.model.adapter, "gate": None}
    if args.checkpoint:
        artifact = checkpointing.verify_evaluation_artifact(
            args.checkpoint, identity_sha256=run_identity["sha256"],
            model_id=config.model.id, model_revision=config.model.revision,
            adapter=config.model.adapter, gate=args.expect_gate)
        artifact = {"kind": artifact["kind"], "verified": True,
                    "identity_sha256": artifact["identity_sha256"],
                    "model_id": artifact["model_id"],
                    "model_revision": artifact["model_revision"],
                    "adapter": artifact["adapter"], "gate": artifact["gate"],
                    "global_step": artifact["global_step"],
                    "path": str(args.checkpoint)}

    print(f"loading {'checkpoint ' + args.checkpoint if args.checkpoint else 'base model'}",
          file=sys.stderr)
    model, tokenizer, load_report = adapters.load_model_for_evaluation(
        config, checkpoint_dir=args.checkpoint,
        local_files_only=args.local_files_only, cache_dir=args.cache_dir)
    records = generate.generate_records(
        model, tokenizer, rows, template=template,
        kwargs=config.tokenizer.chat_template_kwargs,
        evaluation=config.evaluation, max_examples=None,
        progress=lambda message: print(message, file=sys.stderr))

    reference_images = {row["id"]: row["png_image"] for row in rows
                        if row.get("png_image")}
    out_dir = identity.evaluation_dir(paths.run_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    compile_callable = (None if args.no_compile
                        else _make_compile_callable(config, out_dir / "tex"))
    result = evaluate.evaluate_records(
        records, evaluation=config.evaluation, compile_fn=compile_callable,
        compile_enabled=not args.no_compile, reference_images=reference_images,
        train_targets=train_targets)

    metrics = result["metrics"]
    metrics.update({
        "name": args.name,
        "run_dir": str(paths.run_dir),
        "identity_sha256": run_identity["sha256"],
        "data_identity_sha256": prepared["data_identity_sha256"],
        "source_kind": load_report.get("source_kind"),
        "checkpoint_dir": load_report.get("checkpoint_dir"),
        "artifact": artifact,
        "eligible": {
            "quarantined_excluded": len(quarantined),
            "by_split": eligibility["by_split"],
        },
        "model_id": config.model.id,
        "model_revision": config.model.revision,
        "split": args.split,
        "requested_examples": limit,
        "load_report": load_report,
        "evaluation_settings": dataclasses.asdict(config.evaluation),
    })
    write_json_atomic(out_dir / "metrics.json", metrics)
    with open(out_dir / "rows.jsonl", "w", encoding="utf-8") as handle:
        for row in result["rows"]:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    write_json_atomic(out_dir / "config.json", {
        "created_at": utc_now_iso(),
        "config_path": str(config.config_path),
        "checkpoint": args.checkpoint,
        "split": args.split,
        "examples": len(records),
        "compile": not args.no_compile,
    })
    write_text_atomic(out_dir / "summary.md", _summary_markdown(metrics))
    print(f"evaluation written to {out_dir}")
    print(_summary_markdown(metrics))
    return 0


def _make_compile_callable(config, tex_root):
    """Compile with a persistent workdir so TeX evidence survives the run."""
    import itertools

    from stage1 import compile_tikz as compile_module

    tex_root.mkdir(parents=True, exist_ok=True)
    counter = itertools.count()

    def compile_callable(text):
        return compile_module.compile_tikz(
            text, engine=config.evaluation.compile.engine,
            timeout_seconds=config.evaluation.compile.timeout_seconds,
            render=config.evaluation.compile.render,
            workdir=tex_root / f"row-{next(counter):05d}")

    return compile_callable


def _summary_markdown(metrics) -> str:
    generation = metrics["generation"]
    compilation = metrics["compilation"]
    latency = metrics["latency"]
    duplicates = metrics["duplicates"]
    memorization = metrics["memorization"]
    similarity = metrics["render_similarity"]
    return "\n".join([
        f"# Evaluation: {metrics['name']} ({metrics['source_kind']})",
        "",
        f"- examples: {generation['examples']}",
        f"- completed (stop): {generation['completed']}",
        f"- truncated (length): {generation['truncated']}",
        f"- empty outputs: {generation['empty']}",
        f"- extraction: {metrics['extraction']}",
        f"- compile success: {compilation['success']}/{compilation['attempted']} "
        f"({compilation['success_rate']})",
        f"- compile categories: {compilation['categories']}",
        f"- mean latency: {latency['mean']} s; tokens/s: {latency['tokens_per_second']}",
        f"- duplicate rows: {duplicates['duplicate_rows']}",
        f"- memorized: exact {memorization['exact_matches']}, near "
        f"{memorization['near_matches']} (threshold "
        f"{memorization['overlap_threshold']})",
        f"- render similarity: {similarity['status']} "
        f"(compared {similarity['compared']}, mean {similarity['mean']})",
        "",
    ])


if __name__ == "__main__":
    sys.exit(cli.run(main))
