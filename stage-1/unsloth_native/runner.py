"""Shared native Unsloth + TRL SFTTrainer lifecycle."""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

from stage1 import collator, config as config_module, util
from stage1.errors import DataError, GateFailed

from . import NATIVE_SCHEMA_VERSION
from .dataset import load_native_datasets
from .models import get_spec, load_native_model


def _native_identity(spec, config, gate, datasets, fingerprint) -> dict:
    payload = {
        "schema_version": NATIVE_SCHEMA_VERSION,
        "model_key": spec.key,
        "model_id": config.model.id,
        "model_revision": config.model.revision,
        "adapter": config.model.adapter,
        "method": config.training.method,
        "gate": gate.name,
        # Hash the complete resolved configuration so changing LR, batch size,
        # LoRA rank, scheduler, or any future knob cannot reuse a checkpoint.
        "config": config.to_jsonable(),
        "data_identity_sha256": datasets.prepared["data_identity_sha256"],
        "prepared_report_sha256": datasets.prepared["report_sha256"],
        "dataset_logical_sha256": datasets.export.dataset_logical_sha256,
        "tokenizer": adapters_fingerprint_core(fingerprint),
    }
    return {**payload, "sha256": util.canonical_digest(payload)}


def adapters_fingerprint_core(fingerprint):
    from stage1.adapters import fingerprint_core
    return fingerprint_core(fingerprint)


def _resolve_gate(config, gate_name: str):
    """Resolve raw YAML gate overrides into the typed Stage 1 gate config."""
    if gate_name not in config.gates:
        raise DataError(f"Config has no gate {gate_name!r}")
    return config.gate(gate_name)


def _ensure_identity(run_dir: Path, identity: dict) -> None:
    path = run_dir / "native-run.json"
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing != identity:
            raise DataError(
                f"Native run identity mismatch in {path}; use a fresh --run-dir")
    else:
        util.write_json_atomic(path, identity)

    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("passed") is True:
            raise DataError(
                f"Native run is already complete in {run_dir}; use a fresh --run-dir")


def _sft_arguments(SFTConfig, config, gate, run_dir: Path, *, has_eval: bool):
    values = config_module.training_arguments_kwargs(
        config, gate, run_dir=run_dir, has_eval=has_eval,
        run_name=f"native-{config.model.adapter}-{gate.name}")
    values["output_dir"] = str(run_dir / "checkpoints")
    values.update({
        "packing": False,
        "dataset_kwargs": {"skip_prepare_dataset": True},
    })
    parameters = inspect.signature(SFTConfig).parameters
    length_key = "max_length" if "max_length" in parameters else "max_seq_length"
    values[length_key] = config.data.max_seq_len
    if "assistant_only_loss" in parameters:
        values["assistant_only_loss"] = False
    unsupported = sorted(set(values) - set(parameters))
    for key in unsupported:
        # Refuse version drift instead of silently losing an optimization,
        # checkpoint, dataset, or masking setting.
        raise DataError(f"Installed TRL SFTConfig does not accept {key!r}")
    return SFTConfig(**values)


def _resume_checkpoint(run_dir: Path, request: str | None):
    checkpoint_root = (run_dir / "checkpoints").resolve()
    existing = sorted(checkpoint_root.glob("checkpoint-*")) \
        if checkpoint_root.is_dir() else []
    if request in (None, "none"):
        if existing:
            raise DataError(
                f"{checkpoint_root} already contains checkpoints; use --resume auto "
                "or a fresh --run-dir")
        return None
    if request != "auto":
        path = Path(request).resolve()
        if not path.is_dir():
            raise DataError(f"Resume checkpoint does not exist: {path}")
        if path.parent != checkpoint_root:
            raise DataError(
                f"Resume checkpoint must belong to this native run: {checkpoint_root}")
        return str(path)
    try:
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as exc:  # pragma: no cover
        raise DataError("transformers is required for native resume") from exc
    return get_last_checkpoint(str(checkpoint_root)) if checkpoint_root.is_dir() else None


def run_native(model_key: str, *, config_path, expected_method: str, export, prepared, run_dir,
               gate_name: str, resume: str | None, local_files_only: bool,
               cache_dir=None, progress=print):
    """Run one explicitly selected native gate; never advances automatically."""
    # Import Unsloth before Transformers/TRL so its patches are installed.
    try:
        import unsloth  # noqa: F401
        import torch
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover - GPU environment only
        raise DataError("The pinned Unsloth/TRL training environment is required") from exc

    spec = get_spec(model_key)
    config = config_module.load_config(config_path)
    if config.model.adapter != spec.adapter:
        raise DataError(
            f"Config adapter {config.model.adapter!r} does not match {spec.key!r}")
    if config.training.method != expected_method:
        raise DataError(
            f"--method {expected_method!r} does not match config training.method "
            f"{config.training.method!r}")
    gate = _resolve_gate(config, gate_name)
    paths = config_module.resolve_run_paths(
        config, export=export, prepared=prepared, run_dir=run_dir)
    paths.validate_inputs()
    paths.ensure_run_dir()

    prepared_info = data_verify(paths.prepared_dir, config.model.adapter)
    progress(f"loading {config.model.id} with {spec.loader_name}")
    model, tokenizer, template, fingerprint = load_native_model(
        spec, config, prepared_info["model"],
        local_files_only=local_files_only, cache_dir=cache_dir)
    progress("building audited assistant-only datasets")
    datasets = load_native_datasets(
        config, paths, gate, tokenizer, template, progress=progress)
    identity = _native_identity(spec, config, gate, datasets, fingerprint)
    _ensure_identity(paths.run_dir, identity)
    util.write_json_atomic(paths.run_dir / "resolved-config.json", config.to_jsonable())

    collate = collator.make_torch_collator(
        torch, pad_token_id=tokenizer.pad_token_id,
        validate=config.training.validate_batches)
    arguments = _sft_arguments(
        SFTConfig, config, gate, paths.run_dir,
        has_eval=len(datasets.validation) > 0)
    trainer_kwargs = {
        "model": model,
        "args": arguments,
        "train_dataset": datasets.train,
        "eval_dataset": datasets.validation,
        "data_collator": collate,
    }
    trainer_parameters = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    checkpoint = _resume_checkpoint(paths.run_dir, resume)
    progress(f"native training gate={gate.name}, resume={checkpoint or 'none'}")
    result = trainer.train(resume_from_checkpoint=checkpoint)

    final_dir = paths.run_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    evaluation = trainer.evaluate() if len(datasets.validation) else {}
    log_history = list(getattr(trainer.state, "log_history", []) or [])
    logged_losses = [float(item["loss"]) for item in log_history
                     if isinstance(item, dict) and item.get("loss") is not None]
    eval_loss = evaluation.get("eval_loss")
    artifact_files = {path.name for path in final_dir.iterdir() if path.is_file()}
    if config.training.method == "lora":
        artifact_ok = ("adapter_config.json" in artifact_files and
                       any(name.startswith("adapter_model") for name in artifact_files))
    else:
        artifact_ok = any(
            name.endswith((".safetensors", ".bin")) or name.endswith(".index.json")
            for name in artifact_files)
    acceptance = {
        "training_completed": int(getattr(trainer.state, "global_step", 0)) > 0,
        "logged_losses_finite": bool(logged_losses) and all(
            math.isfinite(value) for value in logged_losses),
        "evaluation_loss_finite": eval_loss is not None and math.isfinite(float(eval_loss)),
        "artifact_complete": artifact_ok,
    }
    if gate.name == "overfit-100":
        threshold_ok = (gate.success_eval_loss is not None and eval_loss is not None
                        and float(eval_loss) <= gate.success_eval_loss)
        reduction = None
        if logged_losses and eval_loss is not None and logged_losses[0] > 0:
            reduction = 1.0 - float(eval_loss) / logged_losses[0]
        reduction_ok = (gate.min_loss_reduction is not None and reduction is not None
                        and reduction >= gate.min_loss_reduction)
        acceptance["overfit_loss_target"] = threshold_ok or reduction_ok
    passed = all(acceptance.values())
    metrics = {
        "schema_version": NATIVE_SCHEMA_VERSION,
        "identity_sha256": identity["sha256"],
        "model_key": spec.key,
        "method": config.training.method,
        "gate": gate.name,
        "resume_from_checkpoint": checkpoint,
        "train": dict(result.metrics),
        "evaluation": dict(evaluation),
        "logged_losses": logged_losses,
        "acceptance": acceptance,
        "passed": passed,
        "train_stats": datasets.train_stats,
        "validation_stats": datasets.validation_stats,
        "final_dir": str(final_dir),
    }
    util.write_json_atomic(paths.run_dir / "metrics.json", metrics)
    if not passed:
        failed = [name for name, value in acceptance.items() if not value]
        raise GateFailed(
            f"native gate {gate.name} failed {failed}; see "
            f"{paths.run_dir / 'metrics.json'}")
    return metrics


def data_verify(prepared_dir, adapter):
    from stage1.data import verify_prepared
    return verify_prepared(prepared_dir, adapter_slug=adapter)
