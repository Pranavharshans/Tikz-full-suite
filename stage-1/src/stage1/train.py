"""Full-parameter BF16 supervised fine-tuning with Unsloth.

The trainer is shared by both models; all model-specific behavior comes from
the adapter and configuration. Heavy imports (torch, transformers, datasets,
unsloth) happen inside functions so this module imports cleanly for CLI help
and CPU unit tests.

Gates are isolated: each gate owns ``checkpoints/<gate>/`` and ``final/<gate>/``
and can only resume its own interrupted execution.

Supervised-token accounting counts ``label != -100`` tokens only from
micro-batches actually passed to ``training_step`` and commits them when an
optimizer step completes. Evaluation batches and dataloader-prefetched batches
never contribute; resumed runs continue from the committed count recorded in
the checkpoint metadata.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from . import checkpointing, data, identity
from .errors import DataError, GateFailed
from .util import (canonical_digest, dependency_versions, detect_repo_commit,
                   utc_now_iso, write_json_atomic)


# ---------------------------------------------------------------------------
# Supervised-token accounting and training monitors
# ---------------------------------------------------------------------------


def count_supervised_labels(labels) -> int:
    """Count supervised tokens in a flat/nested list or a torch tensor.

    Never iterates a multi-dimensional tensor: iterating a 2-D tensor yields
    rows, which would count examples instead of tokens (and raising on a
    single ``if tensor`` test). Tensors are counted with vectorised
    comparisons; nested Python sequences are counted row by row.
    """
    if labels is None:
        return 0
    if callable(getattr(labels, "numel", None)):
        return int((labels != -100).sum().item())
    total = 0
    for item in labels:
        if callable(getattr(item, "numel", None)):
            total += int((item != -100).sum().item())
        elif isinstance(item, (list, tuple)):
            total += sum(1 for label in item if label != -100)
        elif item != -100:
            total += 1
    return total


@dataclass
class TrainingMonitor:
    """Exact supervised-token accounting and training health counters.

    Accounting is by construction limited to completed optimizer steps:

    - ``record_micro_batch`` is called from ``Trainer.training_step`` with the
      labels of the micro-batch actually being trained on, so evaluation
      batches (``prediction_step``) and dataloader-prefetched batches that are
      never consumed cannot contribute.
    - ``commit_step`` is called from ``on_step_end``, which fires after an
      optimizer step, so only completed steps are committed.
    - ``pending`` holds micro-batches of an accumulation cycle that has not
      completed (for example after an interruption); they are reported as
      in-flight and are not committed.
    """

    planned_per_epoch: int
    epochs: int
    gradient_accumulation_steps: int
    check_gradients: bool
    committed_supervised_tokens: int = 0
    committed_steps: int = 0
    pending_supervised_tokens: int = 0
    micro_batches: int = 0
    gradient_checks: int = 0
    gradient_failures: int = 0
    non_finite_losses: int = 0
    max_gradient_norm: float | None = None
    method: str = ("trainer.training_step labels, committed only at completed "
                   "on_step_end steps; evaluation and prefetched batches excluded")

    @property
    def planned_total(self) -> int:
        return self.planned_per_epoch * self.epochs

    @property
    def exact(self) -> bool:
        return True

    def resume_from(self, meta: dict) -> None:
        """Seed committed totals from a same-gate checkpoint; never double-count."""
        self.committed_supervised_tokens = int(meta.get("supervised_tokens_seen", 0))
        self.committed_steps = int(meta.get("global_step", 0))
        self.pending_supervised_tokens = 0

    def record_micro_batch(self, labels) -> bool:
        """Count a consumed micro-batch; return True at an accumulation boundary."""
        self.pending_supervised_tokens += count_supervised_labels(labels)
        self.micro_batches += 1
        return (self.micro_batches % self.gradient_accumulation_steps) == 0

    def record_loss(self, finite: bool) -> None:
        if not finite:
            self.non_finite_losses += 1

    def record_gradients(self, norm: float | None, finite: bool) -> None:
        self.gradient_checks += 1
        if not finite or norm is None:
            self.gradient_failures += 1
            return
        if self.max_gradient_norm is None or norm > self.max_gradient_norm:
            self.max_gradient_norm = norm

    def commit_step(self) -> int:
        """Commit the pending tokens of one completed optimizer step."""
        committed = self.pending_supervised_tokens
        self.committed_supervised_tokens += committed
        self.pending_supervised_tokens = 0
        self.committed_steps += 1
        return committed

    def finalize(self, *, global_step: int) -> bool:
        """Commit a trailing cycle only if the Trainer reports a completed step.

        Returns True when a commit happened. This covers the final partial
        gradient-accumulation cycle without inventing steps that never ran.
        """
        if self.pending_supervised_tokens > 0 and global_step > self.committed_steps:
            self.commit_step()
            return True
        return False

    def assert_within_plan(self) -> None:
        """Refuse an accounting result that exceeds the deterministic plan.

        Committed tokens can be lower than the plan (interruption, max_steps),
        but they can never exceed epochs * per-epoch supervised tokens. A
        violation means evaluation or prefetched batches leaked into the
        accounting, so the run is refused instead of reporting wrong numbers.
        """
        if self.committed_supervised_tokens > self.planned_total:
            raise DataError(
                f"Committed supervised tokens {self.committed_supervised_tokens} "
                f"exceed the deterministic plan {self.planned_total} "
                f"({self.planned_per_epoch} per epoch * {self.epochs} epochs); "
                "token accounting is wrong, refusing to report it as exact")

    def snapshot(self) -> dict:
        return {
            "planned_supervised_tokens": self.planned_total,
            "planned_supervised_tokens_per_epoch": self.planned_per_epoch,
            "supervised_tokens_committed": self.committed_supervised_tokens,
            "supervised_tokens_in_flight": self.pending_supervised_tokens,
            "supervised_tokens_exact": self.exact,
            "accounting_method": self.method,
            "committed_steps": self.committed_steps,
            "micro_batches": self.micro_batches,
            "gradient_checks": self.gradient_checks,
            "gradient_failures": self.gradient_failures,
            "non_finite_losses": self.non_finite_losses,
            "max_gradient_norm": self.max_gradient_norm,
        }


def _default_grad_norm(model) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().float().pow(2).sum().item())
    return total ** 0.5


def make_trainer_class(transformers, monitor: TrainingMonitor, *, isfinite=None,
                       grad_norm=None):
    """Build a Trainer subclass that feeds the monitor from real micro-batches.

    Only ``training_step`` is overridden; evaluation uses ``prediction_step``
    and is therefore invisible to the monitor. ``isfinite`` and ``grad_norm``
    are injection seams for unit tests.
    """
    if isfinite is None:
        import torch

        def isfinite(value):
            return bool(torch.isfinite(value))

    if grad_norm is None:
        grad_norm = _default_grad_norm

    class Stage1Trainer(transformers.Trainer):
        def training_step(self, model, inputs, *args, **kwargs):
            labels = inputs.get("labels") if hasattr(inputs, "get") else None
            boundary = False
            if labels is not None:
                boundary = monitor.record_micro_batch(labels)
            loss = super().training_step(model, inputs, *args, **kwargs)
            monitor.record_loss(isfinite(loss))
            if boundary and monitor.check_gradients:
                try:
                    norm = float(grad_norm(model))
                    finite = math.isfinite(norm) and norm > 0.0
                except Exception:
                    norm, finite = None, False
                monitor.record_gradients(norm, finite)
            return loss

    return Stage1Trainer


# ---------------------------------------------------------------------------
# Training schedule bounds
# ---------------------------------------------------------------------------


def optimizer_step_bounds(*, examples: int, per_device_batch_size: int,
                          gradient_accumulation_steps: int, epochs: int,
                          max_steps: int = -1) -> dict:
    """Deterministic bounds on the optimizer steps a run will perform.

    Trainer versions differ in whether the final partial accumulation cycle
    counts as an extra step, so the lower bound uses floor division and the
    upper bound ceiling division. Interval validation uses the lower bound: a
    save/eval interval that is larger than the lower bound may never fire.
    """
    if examples <= 0:
        raise DataError("Cannot compute a schedule for an empty dataset")
    if per_device_batch_size < 1 or gradient_accumulation_steps < 1:
        raise DataError("Batch size and gradient accumulation must be positive")
    micro_batches = math.ceil(examples / per_device_batch_size)
    steps_per_epoch_lower = max(1, micro_batches // gradient_accumulation_steps)
    steps_per_epoch_upper = max(1, math.ceil(micro_batches / gradient_accumulation_steps))
    expected_lower = steps_per_epoch_lower * epochs
    expected_upper = steps_per_epoch_upper * epochs
    if max_steps and max_steps > 0:
        expected_lower = min(expected_lower, max_steps)
        expected_upper = min(expected_upper, max_steps)
    return {
        "examples": examples,
        "micro_batches_per_epoch": micro_batches,
        "steps_per_epoch_lower": steps_per_epoch_lower,
        "steps_per_epoch_upper": steps_per_epoch_upper,
        "expected_steps_lower": expected_lower,
        "expected_steps_upper": expected_upper,
    }


def validate_gate_intervals(gate, bounds: dict, *, has_eval: bool) -> dict:
    """Refuse a checkpoint interval that the schedule cannot reach.

    The smoke gate requires a same-gate checkpoint to exist, so ``save_steps``
    must fit inside the lower step bound. Periodic evaluation is optional (the
    final evaluation always runs), so an eval interval beyond the schedule is
    recorded rather than refused.
    """
    lower = bounds["expected_steps_lower"]
    if gate.save_steps > lower:
        raise DataError(
            f"gate {gate.name!r} would run at least {lower} optimizer step(s) "
            f"but training.save_steps={gate.save_steps}: no checkpoint would "
            "ever be written, so the gate could not verify resume. Lower "
            "save_steps or increase the dataset size.")
    return {
        "save_steps": gate.save_steps,
        "eval_steps": gate.eval_steps,
        "expected_steps_lower": lower,
        "expected_steps_upper": bounds["expected_steps_upper"],
        "save_within_schedule": True,
        "eval_within_schedule": bool(has_eval) and gate.eval_steps <= lower,
    }


# ---------------------------------------------------------------------------
# Artifact weights
# ---------------------------------------------------------------------------


def load_artifact_weights(model, directory, *, method: str = "full",
                          loader=None) -> dict:
    """Load an artifact's weights into the model, exactly and method-aware.

    Thin wrapper over :func:`checkpointing.load_artifact_weights` used by the
    gate-time verification flow.
    """
    return checkpointing.load_artifact_weights(
        model, directory, method=method, loader=loader)


def _eval_forward_loss(model, batch_plan, torch, device="cuda"):
    """Run one deterministic eval-mode forward pass; return ``(loss, shape)``."""
    batch = {
        "input_ids": torch.tensor(batch_plan.input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(batch_plan.labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(batch_plan.attention_mask, dtype=torch.long,
                                       device=device),
        "position_ids": torch.tensor(batch_plan.position_ids, dtype=torch.long,
                                     device=device),
    }
    model.eval()
    with torch.no_grad():
        outputs = model(**batch, use_cache=False)
    loss = float(outputs.loss)
    if not math.isfinite(loss):
        raise DataError(f"Forward pass produced a non-finite loss: {loss}")
    shape = getattr(getattr(outputs, "logits", None), "shape", None)
    if shape is None or callable(shape):
        shape = None
    else:
        shape = tuple(shape)
    return loss, shape


def restore_final_weights(model, final_directory, *, method: str,
                          batch_plan, torch, device="cuda", loader=None) -> dict:
    """Reload the final artifact's weights after checkpoint verification.

    Checkpoint verification loads an older checkpoint into the live model;
    this restores the final weights (full model or adapter, matching the
    training method) and proves the restore with a forward pass, so generation
    and metrics describe the final model.
    """
    restore = checkpointing.load_artifact_weights(
        model, final_directory, method=method, loader=loader)
    loss, shape = _eval_forward_loss(model, batch_plan, torch, device)
    return {
        "restored_after_checkpoint_verification": True,
        "restore_method": method,
        "restore_tensors": restore["tensors"],
        "restore_loss": loss,
        "restore_logits_shape": list(shape) if shape is not None else None,
        "restore_logits_materialized": shape is not None,
    }


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def check_emission_row(row_id: str, total_tokens: int, *, max_seq_len: int,
                       quarantined, wanted) -> None:
    """Refuse an ineligible or overlength row immediately before emission."""
    if row_id in quarantined:
        raise DataError(
            f"Quarantined row {row_id} reached dataset emission; refusing to "
            "train on an ineligible row")
    if row_id not in wanted:
        raise DataError(f"Row {row_id} was emitted but never selected")
    if total_tokens > max_seq_len:
        raise DataError(
            f"Row {row_id} tokenizes to {total_tokens} tokens, above "
            f"data.max_seq_len={max_seq_len}, at dataset emission. Preparation "
            "should have quarantined it; re-run preparation with this sequence "
            "length.")


def check_emitted_ids(wanted, emitted, quarantined) -> None:
    """Refuse a dataset whose emitted ids do not match the eligible selection."""
    missing = set(wanted) - set(emitted)
    if missing:
        raise DataError(
            f"{len(missing)} expected eligible row(s) were not emitted; first: "
            f"{sorted(missing)[0]}")
    extra = set(emitted) - set(wanted)
    if extra:
        raise DataError(
            f"{len(extra)} unexpected row(s) were emitted; first: {sorted(extra)[0]}")
    if set(emitted) & set(quarantined):
        raise DataError("Quarantined rows were emitted into the dataset")


def build_tokenized_dataset(config, tokenizer, template, export_info, manifest,
                            split: str, eligibility: dict, *, limit=None,
                            seed: int = 0, progress=None):
    """Tokenize an eligible split into an Arrow dataset; return ``(dataset, stats)``.

    Rows are selected from the split manifest minus the model's quarantine set,
    and every emitted example is re-checked against ``max_seq_len`` and against
    the quarantine set. A missing expected row or a quarantined row reaching
    emission is a hard error.
    """
    try:
        from datasets import Dataset, Features, Sequence, Value
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataError("datasets is required for training; install the pinned "
                        "environment from environment.lock_file") from exc
    from .formatting import format_example

    quarantined = set(eligibility["quarantined"])
    ids = data.select_split_ids(manifest, split, limit=limit, seed=seed,
                                exclude=quarantined)
    if not ids:
        raise DataError(
            f"No eligible rows for split {split!r} after excluding "
            f"{len(quarantined)} quarantined row(s); refusing to build an "
            "empty dataset")
    wanted = set(ids)
    if len(wanted) != len(ids):
        raise DataError("Eligible id selection contains duplicates")
    stats = {"examples": 0, "prompt_tokens": 0, "supervised_tokens": 0,
             "total_tokens": 0}

    def generate():
        for row in data.iter_rows(export_info, columns="text"):
            if row["id"] not in wanted:
                continue
            example = format_example(
                tokenizer, row_id=row["id"], instruction=row["instruction"],
                tikz=row["tikz_code"], template=template,
                kwargs=config.tokenizer.chat_template_kwargs)
            check_emission_row(row["id"], example.total_tokens,
                               max_seq_len=config.data.max_seq_len,
                               quarantined=quarantined, wanted=wanted)
            stats["examples"] += 1
            stats["prompt_tokens"] += example.prompt_tokens
            stats["supervised_tokens"] += example.supervised_tokens
            stats["total_tokens"] += example.total_tokens
            if progress and stats["examples"] % 5000 == 0:
                progress(f"tokenized {stats['examples']}/{len(ids)}")
            yield {
                "input_ids": list(example.input_ids),
                "labels": list(example.labels),
                "row_id": example.row_id,
            }

    features = Features({
        "input_ids": Sequence(Value("int32")),
        "labels": Sequence(Value("int32")),
        "row_id": Value("string"),
    })
    dataset = Dataset.from_generator(generate, features=features)
    if stats["examples"] != len(ids):
        raise DataError(
            f"Tokenized {stats['examples']} rows but selected {len(ids)}; "
            "refusing to train on an incomplete dataset")
    check_emitted_ids(wanted, dataset["row_id"], quarantined)
    stats["eligible_selected"] = len(ids)
    stats["quarantined_excluded"] = len(quarantined)
    return dataset, stats


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def make_callbacks(transformers, *, monitor: TrainingMonitor,
                   checkpointing_module, identity_sha256, gate_name: str,
                   model_id: str, model_revision: str, adapter: str,
                   run_id: str | None, run_dir=None,
                   training_method: str = "full",
                   base_model_id: str | None = None,
                   base_model_revision: str | None = None,
                   lora: dict | None = None):
    """Trainer callbacks: step commits, JSONL logging, checkpoint metadata."""
    log_path = None
    if run_dir is not None:
        log_directory = Path(run_dir) / "logs"
        log_directory.mkdir(parents=True, exist_ok=True)
        log_path = log_directory / f"{gate_name}.jsonl"

    class MonitorCallback(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            monitor.commit_step()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            logs["supervised_tokens_committed"] = monitor.committed_supervised_tokens
            logs["supervised_tokens_in_flight"] = monitor.pending_supervised_tokens
            if log_path is not None:
                record = {
                    "time": utc_now_iso(),
                    "step": getattr(state, "global_step", None),
                    "epoch": getattr(state, "epoch", None),
                    "logs": dict(logs),
                }
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True,
                                            ensure_ascii=False) + "\n")

        def on_train_end(self, args, state, control, **kwargs):
            monitor.finalize(global_step=int(getattr(state, "global_step", 0)))
            if monitor.committed_supervised_tokens <= 0:
                raise DataError(
                    "Training ended with zero committed supervised tokens; the "
                    "loss masks, the dataset or the accounting are wrong")

    class CheckpointMetaCallback(transformers.TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            directory = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
            if directory.is_dir():
                checkpointing_module.write_checkpoint_meta(
                    directory, identity_sha256=identity_sha256,
                    global_step=int(state.global_step),
                    supervised_tokens_seen=monitor.committed_supervised_tokens,
                    epochs_completed=float(state.epoch or 0.0),
                    gate=gate_name, run_id=run_id, model_id=model_id,
                    model_revision=model_revision, adapter=adapter,
                    training_method=training_method,
                    base_model_id=base_model_id,
                    base_model_revision=base_model_revision, lora=lora)

    return [MonitorCallback(), CheckpointMetaCallback()]


# ---------------------------------------------------------------------------
# Gate-time verification helpers
# ---------------------------------------------------------------------------


def _losses_summary(log_history) -> dict:
    losses = [entry["loss"] for entry in log_history
              if isinstance(entry.get("loss"), (int, float))]
    finite = [value for value in losses if math.isfinite(value)]
    return {
        "count": len(losses),
        "finite": len(finite),
        "non_finite": len(losses) - len(finite),
        "first": losses[0] if losses else None,
        "last": losses[-1] if losses else None,
    }


def run_gate_generation(model, tokenizer, template, config, manifest, export_info,
                        eligibility, split: str, samples: int) -> tuple:
    """Generate a fixed sample from the held-out split; return ``(summary, records)``."""
    from . import generate

    ids = data.select_split_ids(manifest, split, limit=samples,
                                seed=config.seed,
                                exclude=eligibility["quarantined"])
    if not ids:
        return {"examples": 0, "completed": 0, "truncated": 0, "empty": 0,
                "errors": 0, "split": split, "reason": "no eligible rows"}, []
    rows = data.load_rows_by_ids(export_info, ids)
    records = generate.generate_records(
        model, tokenizer, rows, template=template,
        kwargs=config.tokenizer.chat_template_kwargs,
        evaluation=config.evaluation, max_examples=None)
    summary = {
        "split": split,
        "examples": len(records),
        "completed": sum(1 for item in records if item.finish_reason == "stop"),
        "truncated": sum(1 for item in records if item.finish_reason == "length"),
        "empty": sum(1 for item in records if not item.raw_output.strip()),
        "errors": sum(1 for item in records if item.error),
    }
    return summary, records


def run_gate_compilation(records, config) -> dict:
    """Compile the generated sample; record statuses without requiring improvement."""
    from .compile_tikz import compile_tikz

    categories = {}
    statuses = []
    for record in records:
        result = compile_tikz(
            record.raw_output, engine=config.evaluation.compile.engine,
            timeout_seconds=config.evaluation.compile.timeout_seconds,
            render=False)
        statuses.append({"row_id": record.row_id, "status": result.status,
                         "category": result.category})
        categories[result.category] = categories.get(result.category, 0) + 1
    return {
        "attempted": len(statuses),
        "success": sum(1 for item in statuses if item["status"] == "success"),
        "categories": categories,
        "statuses": statuses,
    }


def warmup_steps_for(max_steps: int, warmup_ratio: float) -> int:
    """Warmup steps the way TrainingArguments computes them (ratio path)."""
    if max_steps < 1:
        raise DataError(f"max_steps must be positive, got {max_steps!r}")
    if not 0.0 <= float(warmup_ratio) <= 1.0:
        raise DataError(f"warmup_ratio must be in [0, 1], got {warmup_ratio!r}")
    return math.ceil(max_steps * float(warmup_ratio))


def verify_gate_checkpoint(checkpoint_dir, *, identity_sha256, gate, config,
                           model, torch, batch_plan, num_training_steps,
                           num_warmup_steps, method: str = "full",
                           device="cuda") -> dict:
    """Verify the latest same-gate checkpoint and that it can actually resume.

    Performs, on the real artifact: metadata/completeness verification (full
    model or PEFT adapter, matching the training method), a weight reload into
    the live model with an eval-mode forward, and a restore of the
    optimizer/scheduler/trainer state into fresh objects using the production
    ``get_scheduler("cosine", ...)`` reconstruction.

    This **mutates the live model** to the checkpoint's weights; the caller must
    restore the final artifact weights afterwards (``run_training`` does this
    explicitly and records it) before using the model for anything else.
    """
    meta = checkpointing.verify_checkpoint(
        checkpoint_dir, identity_sha256, gate=gate,
        model_id=config.model.id, model_revision=config.model.revision,
        adapter=config.model.adapter, method=method)
    weights = load_artifact_weights(model, checkpoint_dir, method=method)
    reload_loss, logits_shape = _eval_forward_loss(model, batch_plan, torch, device)
    resume = checkpointing.verify_resume_state(
        checkpoint_dir, torch=torch, model=model,
        learning_rate=config.training.learning_rate,
        fused=config.training.optim == "adamw_torch_fused",
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps, device=device)
    return {
        "path": str(checkpoint_dir),
        "gate": gate,
        "training_method": method,
        "global_step": meta["global_step"],
        "verified": True,
        "reload_verified": True,
        "reload_loss": reload_loss,
        "logits_shape": list(logits_shape),
        "weights_tensors": weights["tensors"],
        "live_model_mutated": True,
        "resume_verified": resume["verified"],
        "resume": resume,
    }


def _gate_generation_split(gate_name: str, eligibility: dict) -> str:
    validation = eligibility["by_split"].get("validation", {}).get("eligible", 0)
    if gate_name == "overfit-100" or validation == 0:
        return "train"
    return "validation"


# ---------------------------------------------------------------------------
# Gate criteria
# ---------------------------------------------------------------------------


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def gate_criteria(gate, metrics: dict) -> dict:
    """Return ``{criterion: (passed, detail)}`` for the gate's acceptance rules."""
    losses = metrics.get("losses") or {}
    gradients = metrics.get("gradients") or {}
    checkpoint = metrics.get("checkpoint") or {}
    final_artifact = metrics.get("final_artifact") or {}
    generation = metrics.get("generation") or {}
    compilation = metrics.get("compilation") or {}
    eval_loss = metrics.get("final_eval_loss")
    steps = metrics.get("optimizer_steps")
    expected = metrics.get("expected_optimizer_steps")
    epochs_completed = metrics.get("epochs_completed")
    checks = {}

    def add(name, passed, detail):
        checks[name] = (bool(passed), detail)

    training_finite = (losses.get("count", 0) > 0
                       and losses.get("non_finite", 0) == 0)
    add("training_losses_finite", training_finite,
        f"{losses.get('finite', 0)}/{losses.get('count', 0)} logged losses finite")

    if gate.name == "overfit-100":
        configured = gate.success_eval_loss is not None or gate.min_loss_reduction is not None
        add("loss_criterion_configured", configured,
            "success_eval_loss and/or min_loss_reduction must be configured")
        if gate.success_eval_loss is not None:
            add("eval_loss_threshold",
                _finite(eval_loss) and eval_loss <= gate.success_eval_loss,
                f"final eval loss {eval_loss} <= {gate.success_eval_loss}")
        if gate.min_loss_reduction is not None:
            first = losses.get("first")
            reduction = None
            if _finite(first) and first > 0 and _finite(eval_loss):
                reduction = 1.0 - eval_loss / first
            add("loss_reduction",
                reduction is not None and reduction >= gate.min_loss_reduction,
                f"loss reduction {reduction} >= {gate.min_loss_reduction}")
        add("checkpoint_reload_verified", checkpoint.get("reload_verified") is True,
            "saved checkpoint was reloaded and produced a finite loss")
        return checks

    if gate.name == "smoke-1000":
        add("gradients_checked", gradients.get("gradient_checks", 0) > 0,
            f"{gradients.get('gradient_checks', 0)} gradient check(s)")
        add("gradients_finite", gradients.get("gradient_failures", 1) == 0,
            f"{gradients.get('gradient_failures', '?')} non-finite gradient check(s)")
        add("no_nan_or_inf",
            losses.get("non_finite", 1) == 0 and gradients.get("non_finite_losses", 0) == 0,
            "no non-finite losses or gradients")
        add("optimizer_steps_completed",
            isinstance(steps, int) and isinstance(expected, int)
            and expected > 0 and steps == expected,
            f"{steps} of {expected} expected optimizer steps")
        add("same_gate_resume_verified", checkpoint.get("resume_verified") is True,
            "checkpoint state (optimizer/scheduler/step) restored")
        add("generation_completed",
            generation.get("examples", 0) > 0 and generation.get("errors", 1) == 0
            and generation.get("completed", 0) + generation.get("truncated", 0)
            == generation.get("examples", 0),
            f"{generation.get('examples', 0)} generated, "
            f"{generation.get('errors', '?')} errors")
        add("compilation_recorded", compilation.get("attempted", 0) > 0,
            f"{compilation.get('attempted', 0)} compile result(s) recorded")
        return checks

    if gate.name == "full":
        steps_ok = (isinstance(steps, int) and isinstance(expected, int)
                    and expected > 0 and steps == expected)
        epochs_ok = (_finite(epochs_completed)
                     and epochs_completed >= gate.epochs - 1e-6)
        add("steps_or_epochs_completed", steps_ok or epochs_ok,
            f"{steps}/{expected} steps, {epochs_completed} of {gate.epochs} epochs")
        add("final_checkpoint_verified", final_artifact.get("verified") is True,
            "final artifact metadata, weights and tokenizer verified")
        add("validation_loss_finite", _finite(eval_loss),
            f"final validation loss {eval_loss}")
        add("generation_completed",
            generation.get("examples", 0) > 0 and generation.get("errors", 1) == 0
            and generation.get("completed", 0) + generation.get("truncated", 0)
            == generation.get("examples", 0),
            f"{generation.get('examples', 0)} generated, "
            f"{generation.get('errors', '?')} errors")
        add("compilation_recorded", compilation.get("attempted", 0) > 0,
            f"{compilation.get('attempted', 0)} compile result(s) recorded")
        return checks

    raise DataError(f"No acceptance criteria defined for gate {gate.name!r}")


def gate_outcome(gate, metrics: dict) -> tuple:
    checks = gate_criteria(gate, metrics)
    failed = [name for name, (passed, _) in checks.items() if not passed]
    details = {name: detail for name, (_, detail) in checks.items()}
    if failed:
        return "failed", {
            "failed_criteria": failed,
            "checks": {name: passed for name, (passed, _) in checks.items()},
            "details": details,
        }
    return "passed", {
        "failed_criteria": [],
        "checks": {name: True for name in checks},
        "details": details,
    }


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def run_training(config, paths, gate_name: str, *, local_files_only: bool = False,
                 cache_dir=None, allow_rerun: bool = False, progress=None) -> dict:
    """Execute one gate. Refuses identity mismatches and missing prerequisites."""
    def say(message):
        if progress:
            progress(message)

    try:
        import torch
        # Unsloth explicitly requires import before Transformers/PEFT so its
        # performance and memory patches are active for the production run.
        import unsloth  # noqa: F401
        import transformers
        from transformers import TrainingArguments
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataError("torch/transformers are required for training; install "
                        "the pinned environment from environment.lock_file") from exc

    from . import adapters, collator, formatting

    gate = config.gate(gate_name)
    say(f"gate {gate_name}: verifying prepared data")
    prepared = data.verify_prepared(paths.prepared_dir,
                                    adapter_slug=config.model.adapter)
    eligibility = data.load_eligibility(paths.prepared_dir, config.model.adapter)
    export_info = data.verify_export(
        paths.export_dir, require_complete=config.data.require_complete_export,
        quick=False)

    say("loading tokenizer and checking its fingerprint")
    tokenizer = adapters.load_tokenizer(
        config, local_files_only=local_files_only, cache_dir=cache_dir)
    template = formatting.resolve_chat_template(tokenizer, config.tokenizer)
    fingerprint = adapters.tokenizer_fingerprint(tokenizer, template, config)
    problems = adapters.compare_fingerprints(prepared["model"], fingerprint)
    if problems:
        raise DataError(
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
    identity.require_prerequisites(paths.run_dir, gate_name,
                                   run_identity["sha256"])
    existing = identity.require_gate_start(
        paths.run_dir, gate_name, run_identity["sha256"],
        allow_rerun=allow_rerun)
    write_json_atomic(paths.run_dir / "resolved-config.json",
                      config.to_jsonable())

    # Only this gate's namespace is visible; metadata must agree with the gate
    # and with the training method (full and LoRA never share checkpoints).
    resume_path, resume_meta = checkpointing.find_latest_checkpoint(
        paths.run_dir, gate_name, run_identity["sha256"],
        model_id=config.model.id, model_revision=config.model.revision,
        adapter=config.model.adapter, method=config.training.method)
    if resume_path is not None:
        say(f"resuming from {resume_path.name} (gate {gate_name}, "
            f"method {config.training.method})")

    say(f"loading the model for BF16 training (method={config.training.method})")
    model, tokenizer, load_report = adapters.load_model_and_tokenizer(
        config, local_files_only=local_files_only, cache_dir=cache_dir,
        for_training=True)
    model_template = formatting.resolve_chat_template(tokenizer, config.tokenizer)
    model_fingerprint = adapters.tokenizer_fingerprint(
        tokenizer, model_template, config)
    model_tokenizer_problems = adapters.compare_fingerprints(
        prepared["model"], model_fingerprint)
    if model_tokenizer_problems:
        raise DataError(
            "Model loader returned a tokenizer that differs from preparation:\n" +
            "\n".join(f"  - {line}" for line in model_tokenizer_problems))
    template = model_template
    if config.training.gradient_checkpointing:
        # The resume lifecycle and the preflight reconstruction must both reach
        # a training forward with a complete checkpointing state.
        load_report["gradient_checkpointing"] = (
            adapters.require_gradient_checkpointing_ready(
                model, context="the production training forward"))
    write_json_atomic(paths.run_dir / "environment.json", {
        "created_at": utc_now_iso(),
        "gate": gate_name,
        "python": dependency_versions()["python"],
        "packages": dependency_versions(),
        "load_report": load_report,
    })

    say("tokenizing the train split")
    train_dataset, train_stats = build_tokenized_dataset(
        config, tokenizer, template, export_info, prepared["manifest"], "train",
        eligibility, limit=gate.max_rows, seed=config.seed, progress=progress)
    if gate.name == "overfit-100":
        eval_dataset, eval_stats = train_dataset, dict(train_stats)
    else:
        eval_dataset, eval_stats = build_tokenized_dataset(
            config, tokenizer, template, export_info, prepared["manifest"],
            "validation", eligibility, limit=config.training.eval_max_rows,
            seed=config.seed, progress=progress)
    say(f"train examples {train_stats['examples']} "
        f"({train_stats['supervised_tokens']} supervised tokens, "
        f"{train_stats['quarantined_excluded']} quarantined excluded), "
        f"eval examples {eval_stats['examples']}")

    step_bounds = optimizer_step_bounds(
        examples=train_stats["examples"],
        per_device_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        epochs=gate.epochs, max_steps=gate.max_steps)
    intervals = validate_gate_intervals(
        gate, step_bounds, has_eval=eval_stats["examples"] > 0)
    say(f"schedule: ~{step_bounds['expected_steps_lower']}-"
        f"{step_bounds['expected_steps_upper']} optimizer steps, "
        f"save every {gate.save_steps}")

    monitor = TrainingMonitor(
        planned_per_epoch=train_stats["supervised_tokens"],
        epochs=gate.epochs,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        check_gradients=gate.check_gradients)
    if resume_meta:
        monitor.resume_from(resume_meta)

    collate = collator.make_torch_collator(
        torch, pad_token_id=tokenizer.pad_token_id,
        validate=config.training.validate_batches)

    arguments = TrainingArguments(**config_module_training_kwargs(
        config, gate, paths.run_dir, has_eval=eval_stats["examples"] > 0))
    callbacks = make_callbacks(
        transformers, monitor=monitor, checkpointing_module=checkpointing,
        identity_sha256=run_identity["sha256"], gate_name=gate_name,
        model_id=config.model.id, model_revision=config.model.revision,
        adapter=config.model.adapter, run_id=run_identity["sha256"][:16],
        run_dir=paths.run_dir, training_method=config.training.method,
        base_model_id=config.model.id,
        base_model_revision=config.model.revision,
        lora=config.lora.to_jsonable() if config.lora is not None else None)
    trainer_class = make_trainer_class(transformers, monitor)
    trainer_kwargs = {
        "model": model, "args": arguments, "train_dataset": train_dataset,
        "eval_dataset": eval_dataset, "data_collator": collate,
        "callbacks": callbacks,
    }
    import inspect
    trainer_parameters = inspect.signature(transformers.Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        raise DataError(
            "This transformers version accepts neither processing_class nor "
            "tokenizer on Trainer; checkpoints would not contain tokenizer "
            "files and evaluation could not rebuild the chat template")
    trainer = trainer_class(**trainer_kwargs)

    torch.cuda.reset_peak_memory_stats()
    started = utc_now_iso()
    try:
        trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
    except Exception as exc:
        monitor.finalize(global_step=int(getattr(trainer.state, "global_step", 0)))
        _record_failure(config, paths, gate_name, run_identity, monitor,
                        f"{type(exc).__name__}: {exc}", train_stats, started)
        raise
    monitor.finalize(global_step=int(getattr(trainer.state, "global_step", 0)))
    monitor.assert_within_plan()

    say("saving the final artifact for this gate")
    final_directory = checkpointing.final_dir(paths.run_dir, gate_name)
    final_directory.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_directory))
    tokenizer.save_pretrained(str(final_directory))
    checkpointing.write_final_meta(
        final_directory, identity_sha256=run_identity["sha256"],
        global_step=int(trainer.state.global_step),
        supervised_tokens_seen=monitor.committed_supervised_tokens,
        epochs_completed=float(trainer.state.epoch or 0.0),
        gate=gate_name, run_id=run_identity["sha256"][:16],
        model_id=config.model.id, model_revision=config.model.revision,
        adapter=config.model.adapter,
        training_method=config.training.method,
        base_model_id=config.model.id,
        base_model_revision=config.model.revision,
        lora=config.lora.to_jsonable() if config.lora is not None else None)
    final_artifact = checkpointing.verify_final_artifact(
        final_directory, identity_sha256=run_identity["sha256"],
        model_id=config.model.id, model_revision=config.model.revision,
        adapter=config.model.adapter, gate=gate_name,
        method=config.training.method)
    final_artifact = {
        "path": str(final_directory),
        "gate": final_artifact["gate"],
        "global_step": final_artifact["global_step"],
        "training_method": config.training.method,
        "kind": "lora_adapter" if config.training.method == "lora" else "model",
        "verified": True,
    }

    eval_loss = None
    if eval_stats["examples"] > 0:
        evaluation = trainer.evaluate()
        eval_loss = evaluation.get("eval_loss")
        if eval_loss is not None and not math.isfinite(float(eval_loss)):
            eval_loss = None
            say("warning: final validation loss was not finite")

    say("verifying the latest same-gate checkpoint")
    checkpoints = checkpointing.list_checkpoints(paths.run_dir, gate_name)
    checkpoint_report = {"path": None, "verified": False,
                         "reload_verified": False, "resume_verified": False}
    batch_plan = collator.build_batch(
        [train_dataset[0]], pad_token_id=tokenizer.pad_token_id)
    if checkpoints:
        trainer_max_steps = int(getattr(trainer.state, "max_steps", -1))
        if trainer_max_steps < 1:
            raise DataError(
                f"trainer.state.max_steps is {trainer_max_steps}; cannot "
                "reconstruct the production cosine schedule for verification")
        scheduler_warmup_steps = warmup_steps_for(
            trainer_max_steps, config.training.warmup_ratio)
        checkpoint_report = verify_gate_checkpoint(
            checkpoints[-1], identity_sha256=run_identity["sha256"],
            gate=gate_name, config=config, model=model, torch=torch,
            batch_plan=batch_plan, num_training_steps=trainer_max_steps,
            num_warmup_steps=scheduler_warmup_steps,
            method=config.training.method)
        # Checkpoint verification loads the checkpoint's weights into the live
        # model. Restore the final artifact weights and prove the restore with
        # a forward pass before anything else uses the model, so generation and
        # metrics describe the final model, not an older checkpoint.
        final_artifact.update(restore_final_weights(
            model, final_directory, method=config.training.method,
            batch_plan=batch_plan, torch=torch))
        say(f"restored final weights after checkpoint verification "
            f"(loss {final_artifact['restore_loss']:.6f})")
    else:
        checkpoint_report["reason"] = (
            "no checkpoint was written; lower training.save_steps so at least "
            "one checkpoint lands inside the gate")

    say("running the fixed evaluation sample for this gate")
    generation_split = _gate_generation_split(gate_name, eligibility)
    generation_summary, generation_records = run_gate_generation(
        model, tokenizer, template, config, prepared["manifest"], export_info,
        eligibility, generation_split, gate.generation_samples)
    compilation = run_gate_compilation(generation_records, config)

    peak_vram = torch.cuda.max_memory_allocated()
    finished = utc_now_iso()
    log_history = list(trainer.state.log_history)
    metrics = {
        "schema_version": "stage1-training-metrics-v1",
        "gate": gate_name,
        "model_id": config.model.id,
        "model_revision": config.model.revision,
        "adapter": config.model.adapter,
        "training_method": config.training.method,
        "lora": config.lora.to_jsonable() if config.lora is not None else None,
        "max_seq_len": config.data.max_seq_len,
        "effective_batch_size": config.training.effective_batch_size,
        "model_summary": {
            "training_method": config.training.method,
            "base_model_id": config.model.id,
            "base_model_revision": config.model.revision,
            "adapter": config.model.adapter,
            "parameter_count": load_report.get("parameter_count"),
            "trainable_parameter_count": load_report.get("trainable_parameter_count"),
            "trainable_percentage": load_report.get("trainable_percentage"),
            "quantization": load_report.get("quantization"),
            "lora": load_report.get("lora"),
            "target_modules": load_report.get("target_modules"),
            "tokenizer_fingerprint_sha256": canonical_digest(
                adapters.fingerprint_core(fingerprint)),
            "data_identity_sha256": prepared["data_identity_sha256"],
            "dataset_report_sha256": prepared["report_sha256"],
            "max_seq_len": config.data.max_seq_len,
            "effective_batch_size": config.training.effective_batch_size,
        },
        "identity_sha256": run_identity["sha256"],
        "data_identity_sha256": prepared["data_identity_sha256"],
        "gate_settings": gate.to_jsonable(),
        "resumed_from": str(resume_path) if resume_path else None,
        "started_at": started,
        "finished_at": finished,
        "examples": train_stats["examples"],
        "eval_examples": eval_stats["examples"],
        "eligible": {
            "train_selected": train_stats["examples"],
            "quarantined_excluded": train_stats["quarantined_excluded"],
            "by_split": eligibility["by_split"],
        },
        "supervised_tokens_seen": monitor.committed_supervised_tokens,
        "supervised_tokens": monitor.snapshot(),
        "optimizer_steps": int(trainer.state.global_step),
        "expected_optimizer_steps": int(getattr(trainer.state, "max_steps", -1)),
        "schedule_bounds": step_bounds,
        "intervals": intervals,
        "epochs_completed": float(trainer.state.epoch or 0.0),
        "num_input_tokens_seen": getattr(trainer.state, "num_input_tokens_seen", None),
        "final_eval_loss": eval_loss,
        "losses": _losses_summary(log_history),
        "gradients": monitor.snapshot(),
        "checkpoint": checkpoint_report,
        "final_artifact": final_artifact,
        "generation": generation_summary,
        "compilation": {key: value for key, value in compilation.items()
                        if key != "statuses"},
        "compilation_statuses": compilation["statuses"],
        "peak_vram_bytes": peak_vram,
        "train_seconds": _duration_seconds(started, finished),
        "supervised_tokens_per_second": _rate(
            monitor.committed_supervised_tokens, started, finished),
        "load_report": load_report,
        "log_history": log_history,
    }
    metrics["training"] = {
        "global_step": metrics["optimizer_steps"],
        "supervised_tokens_seen": monitor.committed_supervised_tokens,
        "supervised_tokens_exact": monitor.exact,
        "final_eval_loss": eval_loss,
        "model_id": config.model.id,
        "peak_vram_bytes": peak_vram,
    }
    metrics_path = identity.metrics_path(paths.run_dir, gate_name)
    metrics["metrics_path"] = str(metrics_path)
    write_json_atomic(metrics_path, metrics)

    status, outcome = gate_outcome(gate, metrics)
    identity.write_gate_evidence(
        paths.run_dir, gate_name, status=status,
        identity_sha256=run_identity["sha256"],
        payload={
            "metrics_path": str(metrics_path),
            "reason": outcome["details"],
            "failed_criteria": outcome["failed_criteria"],
            "criteria": outcome["checks"],
            "gate_settings": gate.to_jsonable(),
            "supervised_tokens_seen": monitor.committed_supervised_tokens,
            "supervised_tokens_exact": monitor.exact,
            "optimizer_steps": metrics["optimizer_steps"],
            "expected_optimizer_steps": metrics["expected_optimizer_steps"],
            "final_eval_loss": eval_loss,
            "peak_vram_bytes": peak_vram,
            "checkpoint": {
                "path": checkpoint_report.get("path"),
                "reload_verified": checkpoint_report.get("reload_verified"),
                "resume_verified": checkpoint_report.get("resume_verified"),
            },
            "final_artifact": final_artifact,
            "generation": generation_summary,
            "compilation": metrics["compilation"],
            "previous_evidence": existing is not None,
            "rerun": bool(existing),
        })
    if status != "passed":
        raise GateFailed(
            f"gate {gate_name!r} failed its criteria: "
            f"{outcome['failed_criteria']}. Evidence and metrics were written; "
            f"inspect {metrics_path}.")
    say(f"gate {gate_name} passed: {outcome['details']}")
    return metrics


def config_module_training_kwargs(config, gate, run_dir, *, has_eval: bool) -> dict:
    """Indirection so tests can monkeypatch the pure mapping if needed."""
    from .config import training_arguments_kwargs
    return training_arguments_kwargs(config, gate, run_dir=run_dir,
                                     has_eval=has_eval)


def _duration_seconds(started_iso: str, finished_iso: str):
    import calendar
    import time as _time

    def parse(value):
        return calendar.timegm(_time.strptime(value, "%Y-%m-%dT%H:%M:%SZ"))

    return parse(finished_iso) - parse(started_iso)


def _rate(tokens, started_iso, finished_iso):
    seconds = _duration_seconds(started_iso, finished_iso)
    if seconds <= 0:
        return None
    return round(tokens / seconds, 3)


def _record_failure(config, paths, gate_name, run_identity, monitor, failure,
                    train_stats, started) -> None:
    metrics_path = identity.metrics_path(paths.run_dir, gate_name)
    write_json_atomic(metrics_path, {
        "schema_version": "stage1-training-metrics-v1",
        "gate": gate_name,
        "status": "failed",
        "error": failure,
        "started_at": started,
        "finished_at": utc_now_iso(),
        "identity_sha256": run_identity["sha256"],
        "supervised_tokens": monitor.snapshot(),
        "supervised_tokens_seen": monitor.committed_supervised_tokens,
        "examples": train_stats.get("examples"),
    })
    identity.write_gate_evidence(
        paths.run_dir, gate_name, status="failed",
        identity_sha256=run_identity["sha256"],
        payload={"metrics_path": str(metrics_path), "reason": failure})
