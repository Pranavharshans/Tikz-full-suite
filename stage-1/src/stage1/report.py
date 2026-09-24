"""Machine-readable comparison plus a readable Markdown report.

Runs are compared by supervised tokens (not only example counts or optimizer
steps), and evaluation quality is compared by compile success, completion,
latency, duplicates and memorization. Missing fields stay explicitly ``None``;
nothing is silently zero-filled.
"""
from __future__ import annotations

from pathlib import Path

from .errors import DataError
from .util import read_json, utc_now_iso, write_json_atomic, write_text_atomic

COMPARISON_SCHEMA_VERSION = "stage1-comparison-v1"

COLUMNS = (
    ("label", "run"),
    ("model_id", "model"),
    ("training_method", "method"),
    ("source_kind", "source"),
    ("gate", "gate"),
    ("supervised_tokens", "supervised tokens"),
    ("examples", "examples"),
    ("optimizer_steps", "steps"),
    ("eval_loss", "eval loss"),
    ("completion_rate", "completion"),
    ("compile_success_rate", "compile"),
    ("mean_latency_s", "latency s"),
    ("tokens_per_second", "tok/s"),
    ("duplicate_rows", "dup rows"),
    ("memorized", "memorized"),
)


def _ratio(numerator, denominator):
    if numerator is None or not denominator:
        return None
    return round(numerator / denominator, 6)


def summarize_metrics(label: str, metrics: dict) -> dict:
    """Flatten one metrics document (training or evaluation) into one row."""
    evaluation = metrics.get("evaluation") or metrics
    generation = evaluation.get("generation") or {}
    compilation = evaluation.get("compilation") or {}
    latency = evaluation.get("latency") or {}
    duplicates = evaluation.get("duplicates") or {}
    memorization = evaluation.get("memorization") or {}
    training = metrics.get("training") or {}
    supervised = metrics.get("supervised_tokens") or {}
    artifact = metrics.get("artifact") or {}

    supervised_tokens = metrics.get("supervised_tokens_seen")
    if supervised_tokens is None:
        supervised_tokens = supervised.get("supervised_tokens_committed")
    if supervised_tokens is None:
        supervised_tokens = training.get("supervised_tokens_seen")
    examples = generation.get("examples")
    if examples is None:
        examples = metrics.get("examples")
    source_kind = artifact.get("kind") or metrics.get("source_kind")
    training_method = (metrics.get("training_method")
                       or (metrics.get("model_summary") or {}).get("training_method"))
    if training_method is None:
        # Evaluations record the method inside the verified artifact block.
        training_method = artifact.get("training_method")
    return {
        "label": label,
        "model_id": metrics.get("model_id") or training.get("model_id"),
        "training_method": training_method or ("base" if source_kind == "base" else None),
        "source_kind": source_kind,
        "gate": metrics.get("gate") or artifact.get("gate"),
        "supervised_tokens": supervised_tokens,
        "supervised_tokens_exact": supervised.get("supervised_tokens_exact"),
        "examples": examples,
        "optimizer_steps": metrics.get("optimizer_steps") or training.get("global_step"),
        "eval_loss": metrics.get("eval_loss") or training.get("final_eval_loss"),
        "completion_rate": _ratio(generation.get("completed"), generation.get("examples")),
        "compile_success_rate": compilation.get("success_rate"),
        "mean_latency_s": latency.get("mean"),
        "tokens_per_second": latency.get("tokens_per_second"),
        "duplicate_rows": duplicates.get("duplicate_rows"),
        "memorized": (memorization.get("exact_matches") or 0)
        + (memorization.get("near_matches") or 0) if memorization else None,
        "data_identity_sha256": metrics.get("data_identity_sha256"),
        "split": metrics.get("split"),
        "max_seq_len": metrics.get("max_seq_len")
        or (metrics.get("model_summary") or {}).get("max_seq_len"),
        "evaluation_set_sha256": metrics.get("evaluation_set_sha256"),
    }


def comparison_problems(entries) -> list:
    """Comparability rules for a comparison set (requirement: no mixing).

    Candidates must agree on dataset identity, sequence limit and evaluation
    set, and a comparison group must not mix full and LoRA candidates. Base
    evaluations are compatible with either single training method.
    """
    problems = []
    records = []
    for label, metrics in entries:
        method = (metrics.get("training_method")
                  or (metrics.get("model_summary") or {}).get("training_method")
                  or (metrics.get("artifact") or {}).get("training_method"))
        records.append({
            "label": label,
            "method": method or ("base" if (metrics.get("artifact") or {}).get("kind") == "base"
                                 or metrics.get("source_kind") == "base" else None),
            "data_identity_sha256": metrics.get("data_identity_sha256"),
            "max_seq_len": metrics.get("max_seq_len")
            or (metrics.get("model_summary") or {}).get("max_seq_len"),
            "evaluation_set_sha256": metrics.get("evaluation_set_sha256"),
        })
    for field_name, description in (
            ("data_identity_sha256", "dataset identity"),
            ("max_seq_len", "sequence limit"),
            ("evaluation_set_sha256", "evaluation set")):
        values = {record[field_name] for record in records if record[field_name]}
        if len(values) > 1:
            problems.append(
                f"candidates disagree on {description}: "
                + ", ".join(f"{record['label']}={record[field_name]}"
                            for record in records if record[field_name]))
    methods = {record["method"] for record in records if record["method"]}
    if "full" in methods and "lora" in methods:
        problems.append(
            "the comparison mixes training methods (full and lora); compare "
            "full with full or lora with lora, or pass --allow-non-comparable")
    unknown = [record["label"] for record in records if not record["method"]]
    if unknown:
        problems.append(
            f"candidates without a recorded training method: {unknown}")
    return problems


def experiment_readiness(entries, *, allow_non_comparable: bool = False) -> dict:
    """Require comparable base and trained evaluation artifacts.

    The experiment is only complete when at least one base-model evaluation and
    one trained-artifact evaluation exist for the same data identity and split,
    and the comparison set is comparable (same dataset identity, sequence limit
    and evaluation set, without mixing full and LoRA candidates) unless
    ``allow_non_comparable`` is set. Compile improvement is deliberately not
    required.
    """
    base, trained = [], []
    for label, metrics in entries:
        kind = (metrics.get("artifact") or {}).get("kind") or metrics.get("source_kind")
        record = {"label": label,
                  "data_identity_sha256": metrics.get("data_identity_sha256"),
                  "split": metrics.get("split")}
        if kind == "base":
            base.append(record)
        elif kind in ("checkpoint", "final"):
            trained.append(record)
    problems = []
    if not base:
        problems.append("no base-model evaluation artifact")
    if not trained:
        problems.append("no trained checkpoint/final evaluation artifact")
    comparable = []
    for base_entry in base:
        for trained_entry in trained:
            if (base_entry["data_identity_sha256"]
                    and base_entry["data_identity_sha256"]
                    == trained_entry["data_identity_sha256"]
                    and base_entry["split"] == trained_entry["split"]):
                comparable.append([base_entry["label"], trained_entry["label"]])
    if base and trained and not comparable:
        problems.append(
            "base and trained artifacts do not share a data identity and split")
    comparability = comparison_problems(entries)
    return {"ready": not problems and (allow_non_comparable or not comparability),
            "problems": problems,
            "comparability_problems": comparability,
            "comparability_allowed": bool(allow_non_comparable),
            "comparable_pairs": comparable}


def compare_metrics(entries, *, allow_non_comparable: bool = False) -> dict:
    rows = [summarize_metrics(label, metrics) for label, metrics in entries]
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "rows": rows,
        "experiment_readiness": experiment_readiness(
            entries, allow_non_comparable=allow_non_comparable),
    }


def render_comparison(comparison: dict) -> str:
    rows = comparison["rows"]
    headers = [header for _, header in COLUMNS]
    lines = [
        "# Stage 1 run comparison",
        "",
        f"Generated {comparison['created_at']}.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = []
        for key, _ in COLUMNS:
            value = row.get(key)
            if isinstance(value, float):
                cells.append(f"{value:.4g}")
            elif value is None:
                cells.append("-")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Experiment readiness: " + (
            "READY (comparable base and trained artifacts present)"
            if comparison.get("experiment_readiness", {}).get("ready")
            else "NOT READY"),
    ]
    readiness = comparison.get("experiment_readiness") or {}
    lines += [
        "",
        "Experiment readiness: " + (
            "READY (comparable base and trained artifacts present)"
            if readiness.get("ready")
            else "NOT READY"),
        "Comparability: " + (
            "OK" if not readiness.get("comparability_problems")
            else "NOT COMPARABLE"),
    ]
    for problem in readiness.get("problems", []):
        lines.append(f"- {problem}")
    for problem in readiness.get("comparability_problems", []):
        lines.append(f"- {problem}")
    if readiness.get("comparability_allowed"):
        lines.append("- non-comparable report explicitly allowed")
    for pair in readiness.get("comparable_pairs", []):
        lines.append(f"- comparable pair: {pair[0]} <-> {pair[1]}")
    lines += [
        "",
        "Notes:",
        "",
        "- Supervised tokens are the assistant/TikZ tokens that actually carried "
        "loss, committed only on completed optimizer steps; compare runs on this "
        "column, not on example counts or optimizer steps alone.",
        "- `-` means the field was not present in that run's metrics document; it "
        "is never silently treated as zero.",
        "- Compile success is a syntax check against the pinned TeX installation, "
        "not a quality judgement, and is not required to improve before a "
        "base-model baseline exists.",
        "",
    ]
    return "\n".join(lines)


def write_comparison(out_dir, comparison: dict) -> dict:
    out_dir = Path(out_dir)
    json_path = out_dir / "comparison.json"
    markdown_path = out_dir / "comparison.md"
    write_json_atomic(json_path, comparison)
    write_text_atomic(markdown_path, render_comparison(comparison))
    return {"json": str(json_path), "markdown": str(markdown_path)}


def load_metrics_file(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise DataError(f"Metrics file not found: {path}")
    return read_json(path)
