"""RTX PRO 6000 preflight.

This module is the only authority on GPU compatibility. It checks the pinned
environment, the GPU, BF16, Unsloth/Transformers loading of the exact
checkpoint, dataset identity, disk space, a real forward/backward step with
finite loss and gradients, a checkpoint save/reload round trip and peak VRAM.

It never starts training and never imports the trainer. Checks are ordinary
functions over a context object, so the orchestration (prerequisites, skip
accounting, failure aggregation) is unit-testable with fakes.
"""
from __future__ import annotations

import shutil
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from .errors import DataError
from .util import (human_bytes, parse_lock_file, python_version_pin,
                   verify_lock_versions, write_json_atomic)

CHECK_SPECS = (
    ("environment.lock", "check_environment_lock", ()),
    ("environment.python", "check_python_version", ()),
    ("disk.space", "check_disk_space", ()),
    ("gpu.identity", "check_gpu_identity", ()),
    ("gpu.bf16", "check_bf16", ("gpu.identity",)),
    ("torch.cuda", "check_torch_cuda", ("gpu.identity",)),
    ("optim.fused", "check_fused_optimizer", ("gpu.identity",)),
    ("unsloth.import", "check_unsloth_import", ()),
    ("attention.backend", "check_attention_backend", ("unsloth.import",)),
    ("data.identity", "check_dataset_identity", ()),
    ("tokenizer.load", "check_tokenizer_load", ("data.identity",)),
    ("model.load", "check_model_load",
     ("tokenizer.load", "unsloth.import", "gpu.bf16", "torch.cuda")),
    ("model.one_step", "check_one_step", ("model.load",)),
    ("checkpoint.weight_serialization", "check_weight_serialization",
     ("model.load",)),
    ("checkpoint.model_reload", "check_model_reload",
     ("checkpoint.weight_serialization",)),
    ("checkpoint.trainer_resume", "check_trainer_resume",
     ("checkpoint.model_reload",)),
)
CHECK_NAMES = tuple(name for name, _, _ in CHECK_SPECS)


def _pass(detail: str, **data) -> dict:
    return {"status": "pass", "detail": detail, "data": data}


def _fail(detail: str, **data) -> dict:
    return {"status": "fail", "detail": detail, "data": data}


def _skip(detail: str, **data) -> dict:
    return {"status": "skip", "detail": detail, "data": data}


def _forward_no_cache(model, batch):
    """Direct loss forwards must not enter generation-cache code paths.

    The Trainer disables the KV cache when gradient checkpointing is active.
    Mirror that behavior in standalone preflight/reload forwards as well;
    some Unsloth-patched model families expose generation helpers as callables
    that are valid for generation but not iterable cache objects in eval mode.
    """
    return model(**batch, use_cache=False)


def _materialized_logits_shape(outputs):
    """Return a logits shape only when the training forward materialized one.

    Unsloth may deliberately replace logits with a lazy sentinel while still
    returning a valid loss. SFT training consumes that loss, so absence of a
    materialized logits tensor is not a failed forward.
    """
    logits = getattr(outputs, "logits", None)
    shape = getattr(logits, "shape", None)
    if shape is None or callable(shape):
        return None
    try:
        return tuple(shape)
    except TypeError:
        return None


@dataclass
class PreflightContext:
    config: object
    run_dir: Path
    identity: dict
    prepared: dict
    export_dir: Path | None = None
    prepared_dir: Path | None = None
    tokenizer: object = None
    model: object = None
    load_report: dict = field(default_factory=dict)
    torch: object = None
    scratch: dict = field(default_factory=dict)
    local_files_only: bool = False
    cache_dir: object = None


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------


def check_environment_lock(ctx: PreflightContext) -> dict:
    config = ctx.config
    lock_path = config.environment.lock_path
    if lock_path is None:
        return _fail("environment.lock_file did not resolve to a file")
    pins = parse_lock_file(lock_path)
    installed = ctx.scratch.get("installed_versions")
    differences = verify_lock_versions(pins, installed)
    if differences:
        return _fail(
            f"{len(differences)} version difference(s) against {lock_path.name}",
            lock_file=str(lock_path), differences=differences)
    return _pass(f"{len(pins)} pins match {lock_path.name}",
                 lock_file=str(lock_path), pinned=len(pins))


def check_python_version(ctx: PreflightContext) -> dict:
    import sys
    actual = ".".join(str(part) for part in sys.version_info[:2])
    expected = ctx.config.environment.python_version
    if actual != expected:
        return _fail(f"python {actual} does not match pinned {expected}",
                     actual=actual, expected=expected)
    pinned = python_version_pin()
    if pinned and pinned != expected:
        return _fail(
            f"the repository .python-version pin ({pinned}) does not match "
            f"environment.python_version ({expected})",
            actual=actual, expected=expected, repo_pin=pinned)
    return _pass(f"python {actual} (repo pin {pinned or 'absent'})",
                 actual=actual, repo_pin=pinned)


def check_disk_space(ctx: PreflightContext) -> dict:
    target = ctx.run_dir
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(target)
    minimum = ctx.config.hardware.min_free_disk_gib * 1024 ** 3
    parameter_count = ctx.scratch.get("parameter_count")
    estimate = None
    if parameter_count:
        # bf16 weights (2 bytes/param) + fp32 AdamW moments (8 bytes/param),
        # times the checkpoint retention limit.
        per_checkpoint = parameter_count * 10
        retention = ctx.config.training.save_total_limit or 1
        estimate = per_checkpoint * retention
    detail = (f"free {human_bytes(usage.free)} on {target} "
              f"(minimum {ctx.config.hardware.min_free_disk_gib} GiB)")
    if estimate is not None:
        detail += f"; estimated checkpoint need {human_bytes(estimate)}"
    if usage.free < minimum:
        return _fail(detail + " - below minimum", free_bytes=usage.free,
                     minimum_bytes=minimum, estimate_bytes=estimate)
    if estimate is not None and usage.free < estimate + minimum:
        return _fail(
            detail + " - free space is below minimum + estimated checkpoints",
            free_bytes=usage.free, minimum_bytes=minimum,
            estimate_bytes=estimate)
    return _pass(detail, free_bytes=usage.free, minimum_bytes=minimum,
                 estimate_bytes=estimate)


# ---------------------------------------------------------------------------
# GPU and framework checks
# ---------------------------------------------------------------------------


def _torch(ctx: PreflightContext):
    if ctx.torch is not None:
        return ctx.torch
    import torch
    ctx.torch = torch
    return torch


def check_gpu_identity(ctx: PreflightContext) -> dict:
    import re
    torch = _torch(ctx)
    if not torch.cuda.is_available():
        return _fail("torch.cuda is not available")
    count = torch.cuda.device_count()
    if count < 1:
        return _fail("no CUDA devices visible")
    index = 0
    if ctx.config.hardware.device.startswith("cuda:"):
        index = int(ctx.config.hardware.device.split(":", 1)[1])
    if index >= count:
        return _fail(f"configured device {ctx.config.hardware.device} but only "
                     f"{count} device(s) visible")
    properties = torch.cuda.get_device_properties(index)
    name = properties.name
    pattern = ctx.config.hardware.expected_gpu_name_regex
    if not re.search(pattern, name):
        return _fail(f"GPU {index} is {name!r}, which does not match "
                     f"{pattern!r}", name=name)
    vram_gib = properties.total_memory / 1024 ** 3
    if vram_gib < ctx.config.hardware.min_vram_gib:
        return _fail(f"GPU {index} has {vram_gib:.1f} GiB, below "
                     f"min_vram_gib={ctx.config.hardware.min_vram_gib}",
                     name=name, vram_gib=vram_gib)
    return _pass(f"GPU {index}: {name}, {vram_gib:.1f} GiB",
                 name=name, vram_gib=round(vram_gib, 2), index=index,
                 device_count=count)


def check_bf16(ctx: PreflightContext) -> dict:
    torch = _torch(ctx)
    if not torch.cuda.is_bf16_supported():
        return _fail("CUDA reports no BF16 support on this device")
    return _pass("torch.cuda.is_bf16_supported() is true")


def check_torch_cuda(ctx: PreflightContext) -> dict:
    torch = _torch(ctx)
    properties = torch.cuda.get_device_properties(0)
    capability = f"{properties.major}.{properties.minor}"
    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "capability": capability,
    }
    detail = f"torch {info['torch']}, CUDA {info['cuda']}, cudnn {info['cudnn']}, sm_{capability.replace('.', '')}"
    if not torch.cuda.is_available():
        return _fail("torch.cuda is not available", **info)
    try:
        left = torch.ones(8, device="cuda")
        right = torch.ones(8, device="cuda") * 2
        value = float((left @ right).item())
    except Exception as exc:
        return _fail(f"CUDA kernel smoke test failed: {exc}", **info)
    if value != 16.0:
        return _fail(f"CUDA kernel smoke test returned {value}, expected 16.0", **info)
    return _pass(detail, **info)


def check_fused_optimizer(ctx: PreflightContext) -> dict:
    torch = _torch(ctx)
    parameter = torch.nn.Parameter(torch.zeros(4, device="cuda"))
    try:
        optimizer = torch.optim.AdamW([parameter], lr=1e-5, fused=True)
        optimizer.zero_grad(set_to_none=True)
    except Exception as exc:
        return _fail(f"fused AdamW unavailable: {type(exc).__name__}: {exc}")
    return _pass("torch.optim.AdamW(fused=True) constructs on this device")


def check_unsloth_import(ctx: PreflightContext) -> dict:
    try:
        import unsloth
    except Exception as exc:
        return _fail(f"import unsloth failed: {type(exc).__name__}: {exc}")
    version = getattr(unsloth, "__version__", "unknown")
    loader_name = ("FastVisionModel"
                   if ctx.config.model.loader == "unsloth-vision-model"
                   else "FastLanguageModel")
    loader = getattr(unsloth, loader_name, None)
    if loader is None:
        return _fail(f"unsloth {version} has no {loader_name}", version=version)
    from .adapters import _callable_parameters
    import inspect as _inspect
    method = ctx.config.training.method
    if method == "lora":
        get_peft_model = getattr(loader, "get_peft_model", None)
        if get_peft_model is None:
            return _fail(
                f"unsloth {version} {loader_name} has no get_peft_model; LoRA "
                "mode requires the native PEFT integration", version=version)
        parameters = _callable_parameters(get_peft_model)
        required = {"r", "lora_alpha", "lora_dropout", "bias", "target_modules"}
        accepts_any = any(
            parameter.kind is _inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values())
        missing = sorted(required - set(parameters)) if not accepts_any else []
        if missing:
            return _fail(
                f"unsloth {version} {loader_name}.get_peft_model does not "
                f"accept {missing}", version=version)
        return _pass(
            f"unsloth {version}, {loader_name}.get_peft_model with the "
            "required adapter parameters", version=version, loader=loader_name,
            method=method)
    if "full_finetuning" not in _callable_parameters(loader.from_pretrained):
        return _fail(
            f"unsloth {version} {loader_name}.from_pretrained has no "
            "full_finetuning parameter; Stage 1 requires full-parameter training",
            version=version)
    return _pass(f"unsloth {version}, {loader_name} with full_finetuning",
                 version=version, loader=loader_name, method=method)


def check_attention_backend(ctx: PreflightContext) -> dict:
    """Report the attention backends present; Stage 1 needs neither for padding.

    This is informational: batches are right-padded, so no varlen backend is
    required. The versions are recorded in the preflight evidence because they
    still describe the environment the run will use.
    """
    info = {}
    try:
        import flash_attn  # noqa: F401
        info["flash_attn"] = getattr(flash_attn, "__version__", "unknown")
    except ImportError:
        info["flash_attn"] = None
    try:
        import xformers  # noqa: F401
        info["xformers"] = getattr(xformers, "__version__", "unknown")
    except ImportError:
        info["xformers"] = None
    return _pass(
        f"flash_attn={info['flash_attn']}, xformers={info['xformers']}",
        **info)


# ---------------------------------------------------------------------------
# Data, tokenizer, model
# ---------------------------------------------------------------------------


def check_dataset_identity(ctx: PreflightContext) -> dict:
    from . import data as data_module
    prepared = ctx.prepared
    if prepared["data_identity_sha256"] != ctx.identity["data"]["data_identity_sha256"]:
        return _fail("prepared data identity does not match the run identity")
    export_dir = ctx.export_dir or ctx.config.data.export_dir
    if export_dir is None:
        return _fail("export_dir is not set; cannot verify dataset identity")
    export_info = data_module.verify_export(
        export_dir, require_complete=ctx.config.data.require_complete_export,
        quick=True)
    if export_info.dataset_logical_sha256 != prepared["manifest"]["data_identity"][
            "dataset_logical_sha256"]:
        return _fail("export dataset_logical_sha256 does not match the split manifest")
    ctx.scratch["export_info"] = export_info
    return _pass(
        f"export {export_info.rows} rows, dataset logical "
        f"{export_info.dataset_logical_sha256[:16]}..., data identity "
        f"{prepared['data_identity_sha256'][:16]}...",
        rows=export_info.rows,
        dataset_logical_sha256=export_info.dataset_logical_sha256,
        data_identity_sha256=prepared["data_identity_sha256"])


def check_tokenizer_load(ctx: PreflightContext) -> dict:
    from . import adapters, formatting
    tokenizer = ctx.tokenizer
    if tokenizer is None:
        tokenizer = adapters.load_tokenizer(
            ctx.config, local_files_only=ctx.local_files_only,
            cache_dir=ctx.cache_dir)
    template = formatting.resolve_chat_template(tokenizer, ctx.config.tokenizer)
    fingerprint = adapters.tokenizer_fingerprint(tokenizer, template, ctx.config)
    problems = adapters.compare_fingerprints(ctx.prepared["model"], fingerprint)
    if problems:
        return _fail(
            "tokenizer fingerprint differs from the prepared fingerprint:\n"
            + "\n".join(f"  - {line}" for line in problems))
    ctx.tokenizer = tokenizer
    ctx.scratch["template"] = template
    return _pass(
        f"{fingerprint['tokenizer_class']}, vocab {fingerprint['vocab_size']}, "
        f"template {template.sha256[:16]}...",
        fingerprint=fingerprint)


def check_model_load(ctx: PreflightContext) -> dict:
    from . import adapters
    model, tokenizer, report = adapters.load_model_and_tokenizer(
        ctx.config, local_files_only=ctx.local_files_only, cache_dir=ctx.cache_dir,
        for_training=True)
    if ctx.tokenizer is None:
        ctx.tokenizer = tokenizer
    ctx.model = model
    ctx.load_report = report
    ctx.scratch["parameter_count"] = report["parameter_count"]
    detail = (f"{report['loader']} loaded {report['parameter_count'] / 1e9:.2f}B params "
              f"({report['dtype']}), method={report['training_method']}, "
              f"attn={report['attn_implementation']}")
    if report["training_method"] == "lora":
        detail += (f", trainable {report['trainable_parameter_count'] / 1e6:.1f}M "
                   f"({report['trainable_percentage']:.4f}%)")
    return _pass(detail, **report)


def check_one_step(ctx: PreflightContext) -> dict:
    import time
    from . import collator, data as data_module, formatting
    torch = _torch(ctx)
    if ctx.model is None or ctx.tokenizer is None:
        return _fail("model or tokenizer not loaded")
    template = ctx.scratch.get("template")
    export_dir = ctx.export_dir or ctx.config.data.export_dir
    if export_dir is None:
        return _fail("export_dir is not set; cannot load one-step rows")
    export_info = ctx.scratch.get("export_info") or data_module.verify_export(
        export_dir, require_complete=ctx.config.data.require_complete_export,
        quick=True)
    eligibility = ctx.scratch.get("eligibility")
    if eligibility is None:
        if ctx.prepared_dir is None:
            return _fail("prepared_dir is not set; cannot load eligibility")
        eligibility = data_module.load_eligibility(
            ctx.prepared_dir, ctx.config.model.adapter)
        ctx.scratch["eligibility"] = eligibility
    ids = data_module.select_split_ids(
        ctx.prepared["manifest"], "train", limit=2, seed=ctx.config.seed,
        exclude=eligibility["quarantined"])
    rows = data_module.load_rows_by_ids(export_info, ids)
    if not rows:
        return _fail("no train rows available for the one-step test")
    examples = [formatting.format_example(
        ctx.tokenizer, row_id=row["id"], instruction=row["instruction"],
        tikz=row["tikz_code"], template=template,
        kwargs=ctx.config.tokenizer.chat_template_kwargs) for row in rows]
    overlength = [example for example in examples
                  if example.total_tokens > ctx.config.data.max_seq_len]
    if overlength:
        return _fail(
            f"{len(overlength)} example(s) exceed data.max_seq_len="
            f"{ctx.config.data.max_seq_len} before emission; the eligible-row "
            "manifest and the configured sequence length disagree")
    plan = collator.build_batch(
        examples, pad_token_id=ctx.tokenizer.pad_token_id)
    collator.assert_valid_batch(plan, pad_token_id=ctx.tokenizer.pad_token_id)
    device = ctx.scratch.get("device", "cuda")
    batch = {
        "input_ids": torch.tensor(plan.input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(plan.labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(plan.attention_mask, dtype=torch.long,
                                       device=device),
        "position_ids": torch.tensor(plan.position_ids, dtype=torch.long,
                                     device=device),
    }
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = ctx.model
    model.train()
    started = time.monotonic()
    try:
        outputs = _forward_no_cache(model, batch)
        loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            return _fail(f"loss is not finite: {loss}")
        loss.backward()
    except Exception as exc:
        return _fail(f"forward/backward failed: {type(exc).__name__}: {exc}")
    elapsed = time.monotonic() - started
    norm_squared = 0.0
    gradient_tensors = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient_tensors += 1
        norm_squared += float(parameter.grad.detach().float().pow(2).sum().item())
    grad_norm = norm_squared ** 0.5
    peak_vram = (torch.cuda.max_memory_allocated()
                 if torch.cuda.is_available() else None)
    model.zero_grad(set_to_none=True)
    if gradient_tensors == 0:
        return _fail("backward produced no gradients")
    if not (grad_norm > 0) or grad_norm != grad_norm or grad_norm == float("inf"):
        return _fail(f"gradient norm is not finite and positive: {grad_norm}")

    # Record an eval-mode forward on the same batch so the later reload check
    # can compare like with like.
    model.eval()
    try:
        with torch.no_grad():
            eval_outputs = _forward_no_cache(model, batch)
        eval_loss = float(eval_outputs.loss)
        logits_shape = _materialized_logits_shape(eval_outputs)
    except Exception as exc:
        return _fail(
            f"eval-mode forward failed: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}")
    finally:
        model.train()
    if eval_loss != eval_loss or eval_loss == float("inf"):
        return _fail(f"eval-mode loss is not finite: {eval_loss}")

    ctx.scratch["batch_plan"] = plan
    ctx.scratch["eval_loss"] = eval_loss
    ctx.scratch["logits_shape"] = logits_shape
    ctx.scratch["peak_vram_bytes"] = peak_vram
    peak_detail = human_bytes(peak_vram) if peak_vram is not None else "unmeasured"
    return _pass(
        f"loss {float(loss):.4f}, grad norm {grad_norm:.4f}, "
        f"{elapsed:.2f}s, peak VRAM {peak_detail}",
        loss=float(loss), eval_loss=eval_loss,
        logits_shape=(list(logits_shape) if logits_shape is not None else None),
        logits_materialized=logits_shape is not None,
        grad_norm=grad_norm, seconds=round(elapsed, 4),
        peak_vram_bytes=peak_vram, supervised_tokens=plan.supervised_tokens)


def check_weight_serialization(ctx: PreflightContext, *,
                               adapter_state_getter=None) -> dict:
    """Weight serialization only: save, read back, compare sampled tensors.

    This deliberately does not claim that the model can be reloaded or resumed;
    ``check_model_reload`` and ``check_trainer_resume`` do that.
    """
    import tempfile
    torch = _torch(ctx)
    if ctx.model is None:
        return _fail("model not loaded")
    directory = Path(tempfile.mkdtemp(prefix="stage1-preflight-weights-"))
    try:
        ctx.model.save_pretrained(str(directory), safe_serialization=True)
        try:
            from safetensors.torch import load_file
        except ImportError:
            return _fail("safetensors is not importable")
        weight_files = sorted(directory.glob("*.safetensors"))
        if not weight_files:
            return _fail("save_pretrained produced no safetensors files")
        loaded = {}
        for path in weight_files:
            loaded.update(load_file(str(path)))
        keys = sorted(loaded)
        if not keys:
            return _fail("saved checkpoint contains no tensors")
        method = ctx.config.training.method
        if method == "lora":
            if adapter_state_getter is None:
                try:
                    from peft import get_peft_model_state_dict
                except ImportError:
                    return _fail("peft is not importable")
                adapter_state_getter = get_peft_model_state_dict
            from .checkpointing import (compare_adapter_states,
                                        normalize_adapter_key)
            try:
                live_adapter = adapter_state_getter(ctx.model)
            except Exception as exc:
                return _fail(
                    "PEFT could not read the live adapter state: "
                    f"{type(exc).__name__}: {exc}")
            comparison = compare_adapter_states(loaded, live_adapter)
            if comparison["problems"]:
                preview = comparison["problems"][:20]
                remaining = len(comparison["problems"]) - len(preview)
                suffix = (f"\n  - ... {remaining} additional mismatch(es)"
                          if remaining else "")
                loaded_norm = {
                    normalize_adapter_key(key): value
                    for key, value in loaded.items()}
                live_norm = {
                    normalize_adapter_key(key): value
                    for key, value in live_adapter.items()}
                deltas = []
                for key in sorted(set(loaded_norm) & set(live_norm)):
                    left, right = loaded_norm[key], live_norm[key]
                    if not hasattr(left, "detach") or not hasattr(right, "detach"):
                        continue
                    try:
                        difference = (left.detach().float().cpu()
                                      - right.detach().float().cpu()).abs()
                        if difference.numel():
                            deltas.append({
                                "key": key,
                                "max_abs": float(difference.max().item()),
                                "mean_abs": float(difference.mean().item()),
                                "saved_dtype": str(left.dtype),
                                "live_dtype": str(right.dtype),
                            })
                    except Exception:
                        continue
                    if len(deltas) == 3:
                        break
                delta_text = "".join(
                    f"\n  - diagnostic {item['key']}: "
                    f"max_abs={item['max_abs']:.8g}, "
                    f"mean_abs={item['mean_abs']:.8g}, "
                    f"saved={item['saved_dtype']}, live={item['live_dtype']}"
                    for item in deltas)
                return _fail(
                    "adapter serialization failed:\n" + "\n".join(
                        f"  - {line}" for line in preview) + suffix + delta_text,
                    mismatch_count=len(comparison["problems"]),
                    missing_from_artifact=comparison["missing_from_artifact"],
                    not_part_of_adapter=comparison["not_part_of_adapter"],
                    value_delta_preview=deltas)
            return _pass(
                f"saved {len(weight_files)} adapter weight file(s), compared "
                f"all {comparison['compared']} PEFT tensors, read-back matches "
                "(serialization only)",
                weight_files=len(weight_files),
                sampled=comparison["compared"],
                adapter_tensors=comparison["adapter_parameters"],
                bytes_on_disk=sum(path.stat().st_size for path in weight_files),
                reload_verified=False)

        state = ctx.model.state_dict()
        sample = keys[:2] + keys[len(keys) // 2:len(keys) // 2 + 2] + keys[-2:]
        mismatches = []
        for key in sample:
            if key not in state:
                mismatches.append(f"{key}: missing from the live model")
                continue
            if loaded[key].shape != state[key].shape:
                mismatches.append(f"{key}: shape mismatch")
                continue
            if not torch.equal(loaded[key], state[key]):
                mismatches.append(f"{key}: values differ after read-back")
        if mismatches:
            return _fail("weight serialization failed:\n" +
                         "\n".join(f"  - {line}" for line in mismatches))
        return _pass(
            f"saved {len(weight_files)} weight file(s), sampled {len(sample)} "
            "tensors, read-back matches (serialization only)",
            weight_files=len(weight_files), sampled=len(sample),
            bytes_on_disk=sum(path.stat().st_size for path in weight_files),
            reload_verified=False)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def check_model_reload(ctx: PreflightContext, *, base_loader=None,
                       adapter_attacher=None) -> dict:
    """Save, release the original model, reload it, and compare a forward pass.

    The original model is released (and CUDA cache cleared when available)
    before the reload so the two copies never need to fit in VRAM at the same
    time. The reloaded model replaces ``ctx.model`` for the remaining checks.

    For LoRA the reload uses the base-only loader (no fresh adapter) and then
    attaches the saved adapter exactly once. ``base_loader`` and
    ``adapter_attacher`` are injection seams for CPU/GPU tests.
    """
    import gc
    import tempfile
    from . import adapters
    torch = _torch(ctx)
    base_loader = base_loader or adapters.load_base_model_and_tokenizer
    adapter_attacher = adapter_attacher or adapters.attach_lora_adapter
    device = ctx.scratch.get("device", "cuda")
    if ctx.model is None or ctx.tokenizer is None:
        return _fail("model or tokenizer not loaded")
    plan = ctx.scratch.get("batch_plan")
    loss_before = ctx.scratch.get("eval_loss")
    shape_before = ctx.scratch.get("logits_shape")
    if plan is None or loss_before is None:
        return _fail(
            "the one-step check did not record an eval-mode batch to compare "
            "against; cannot verify a reload")

    directory = Path(tempfile.mkdtemp(prefix="stage1-preflight-reload-"))
    try:
        method = ctx.config.training.method
        ctx.model.save_pretrained(str(directory), safe_serialization=True)
        ctx.tokenizer.save_pretrained(str(directory))
        del ctx.model
        ctx.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            if method == "lora":
                # An adapter is not a standalone model: reload the pinned base
                # without any adapter, then attach the saved one exactly once.
                model, tokenizer, report = base_loader(
                    ctx.config, local_files_only=ctx.local_files_only,
                    cache_dir=ctx.cache_dir, for_training=True)
                model = adapter_attacher(model, directory)
                report["source_kind"] = "lora_adapter"
                report["adapter_attached"] = "once"
                report["adapter"] = str(directory)
            else:
                model, tokenizer, report = adapters.load_model_and_tokenizer(
                    ctx.config, local_files_only=True, cache_dir=ctx.cache_dir,
                    for_training=True, source_override=directory)
        except Exception as exc:
            return _fail(
                f"reload through the supported loading path failed: "
                f"{type(exc).__name__}: {exc}")
        ctx.model = model
        ctx.load_report = report
        batch = {
            "input_ids": torch.tensor(plan.input_ids, dtype=torch.long, device=device),
            "labels": torch.tensor(plan.labels, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(plan.attention_mask, dtype=torch.long,
                                           device=device),
            "position_ids": torch.tensor(plan.position_ids, dtype=torch.long,
                                         device=device),
        }
        model.eval()
        try:
            with torch.no_grad():
                outputs = _forward_no_cache(model, batch)
        except Exception as exc:
            return _fail(
                f"forward pass on the reloaded model failed: "
                f"{type(exc).__name__}: {exc}")
        loss_after = float(outputs.loss)
        shape_after = _materialized_logits_shape(outputs)
        if loss_after != loss_after or loss_after == float("inf"):
            return _fail(f"reloaded loss is not finite: {loss_after}")
        if ((shape_before is None) != (shape_after is None) or
                (shape_before is not None and shape_after != tuple(shape_before))):
            return _fail(
                f"reloaded logits shape {shape_after} != original "
                f"{tuple(shape_before) if shape_before is not None else None}")
        difference = abs(loss_after - float(loss_before))
        if difference > 0.1:
            return _fail(
                f"reloaded loss {loss_after:.6f} differs from the original "
                f"eval loss {float(loss_before):.6f} by {difference:.6f}")
        return _pass(
            f"reloaded through {report['loader']}; eval loss "
            f"{loss_before:.6f} -> {loss_after:.6f} (diff {difference:.2e}), "
            f"logits shape {shape_after if shape_after is not None else 'not materialized'}",
            loss_before=float(loss_before), loss_after=loss_after,
            loss_abs_diff=difference,
            logits_shape=(list(shape_after) if shape_after is not None else None),
            logits_materialized=shape_after is not None,
            reload_verified=True, loader=report["loader"])
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def check_trainer_resume(ctx: PreflightContext) -> dict:
    """Write a minimal Trainer-compatible checkpoint and restore it.

    Verifies, on real objects: the checkpoint file layout our resume path
    requires, the model weights, the optimizer state, the scheduler state and
    the trainer step. The model is reused (already reloaded by the previous
    check), so no second model copy is allocated.
    """
    import tempfile
    from . import checkpointing
    from .errors import CheckpointError
    torch = _torch(ctx)
    if ctx.model is None:
        return _fail("model not loaded")
    plan = ctx.scratch.get("batch_plan")
    if plan is None:
        return _fail("no recorded batch to run the resume check against")

    directory = Path(tempfile.mkdtemp(prefix="stage1-preflight-trainer-"))
    try:
        model = ctx.model
        trainable = [parameter for parameter in model.parameters()
                     if parameter.requires_grad]
        if not trainable:
            return _fail("model has no trainable parameters")
        device = ctx.scratch.get("device", "cuda")
        fused = (ctx.config.training.optim == "adamw_torch_fused"
                 and torch.cuda.is_available())
        optimizer = torch.optim.AdamW(
            trainable, lr=ctx.config.training.learning_rate, fused=fused)
        # The production schedule: transformers cosine with warmup (LambdaLR).
        resume_training_steps = 10
        resume_warmup_steps = 2
        try:
            from transformers import get_scheduler
        except ImportError:
            return _fail("transformers is not importable")
        scheduler = get_scheduler(
            "cosine", optimizer, num_warmup_steps=resume_warmup_steps,
            num_training_steps=resume_training_steps)
        batch = {
            "input_ids": torch.tensor(plan.input_ids, dtype=torch.long, device=device),
            "labels": torch.tensor(plan.labels, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(plan.attention_mask, dtype=torch.long,
                                           device=device),
            "position_ids": torch.tensor(plan.position_ids, dtype=torch.long,
                                         device=device),
        }
        model.train()
        outputs = _forward_no_cache(model, batch)
        if outputs.loss is None or not torch.isfinite(outputs.loss):
            return _fail(f"loss is not finite before the resume check: {outputs.loss}")
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step = 1

        # A minimal checkpoint shaped like the ones the Trainer writes.
        model.save_pretrained(str(directory), safe_serialization=True)
        torch.save(optimizer.state_dict(), directory / "optimizer.pt")
        torch.save(scheduler.state_dict(), directory / "scheduler.pt")
        write_json_atomic(directory / "trainer_state.json", {
            "global_step": step, "epoch": 0.1,
            "log_history": [{"loss": float(outputs.loss)}]})
        method = ctx.config.training.method
        checkpointing.write_checkpoint_meta(
            directory, identity_sha256=ctx.identity["sha256"],
            global_step=step, supervised_tokens_seen=0, epochs_completed=0.1,
            gate=None, run_id=ctx.identity["sha256"][:16],
            model_id=ctx.config.model.id,
            model_revision=ctx.config.model.revision,
            adapter=ctx.config.model.adapter,
            training_method=method,
            base_model_id=ctx.config.model.id if method == "lora" else None,
            base_model_revision=ctx.config.model.revision if method == "lora" else None,
            lora=(ctx.config.lora.to_jsonable()
                  if method == "lora" and ctx.config.lora is not None else None))

        reason = checkpointing.incomplete_reason(directory, method=method)
        if reason:
            return _fail(f"minimal checkpoint is incomplete: {reason}")
        checkpointing.verify_checkpoint(directory, ctx.identity["sha256"],
                                        method=method)

        # Restore the model weights from the saved files (adapter-only for
        # LoRA, complete model for full finetuning).
        try:
            checkpointing.load_artifact_weights(model, directory, method=method)
        except CheckpointError as exc:
            return _fail(f"model restore is not exact: {exc}")

        # Restore optimizer, scheduler and trainer state into fresh objects and
        # require the restored state to equal the state we saved. The scheduler
        # is the production get_scheduler("cosine", ...) LambdaLR.
        try:
            resume = checkpointing.verify_resume_state(
                directory, torch=torch, model=model,
                learning_rate=ctx.config.training.learning_rate, fused=fused,
                num_training_steps=resume_training_steps,
                num_warmup_steps=resume_warmup_steps,
                expected_optimizer_state=optimizer.state_dict(),
                device=device)
        except CheckpointError as exc:
            return _fail(f"resume state restore failed: {exc}")
        checks = {
            "optimizer_state_restored": True,
            "scheduler_state_restored": True,
            "trainer_step_restored": resume["global_step"] == step,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            return _fail(f"trainer resume check failed: {failed}", **checks)

        model.eval()
        with torch.no_grad():
            restored_outputs = _forward_no_cache(model, batch)
        restored_loss = float(restored_outputs.loss)
        if restored_loss != restored_loss or restored_loss == float("inf"):
            return _fail(f"loss after restore is not finite: {restored_loss}")
        return _pass(
            f"checkpoint layout, weights, optimizer, scheduler and step "
            f"{step} restored; loss after restore {restored_loss:.6f}",
            restored_step=step, restored_loss=restored_loss,
            trainer_resume_verified=True, resume=resume, **checks)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _resolve(function_name):
    if callable(function_name):
        return function_name
    return globals()[function_name]


def run_preflight(ctx: PreflightContext, *, checks=None,
                  skip=()) -> dict:
    """Run the check list in order, honoring prerequisites and explicit skips."""
    specs = checks if checks is not None else CHECK_SPECS
    known = {name for name, _, _ in specs}
    unknown_skips = sorted(set(skip) - known)
    if unknown_skips:
        raise DataError(
            f"Unknown --skip-check name(s) {unknown_skips}; known checks are "
            f"{sorted(known)}")
    results = []
    status_by_name = {}
    for name, function_name, requires in specs:
        if name in skip:
            entry = _skip("explicitly skipped by the operator")
            entry["name"] = name
            entry["requires"] = list(requires)
            results.append(entry)
            status_by_name[name] = "skip"
            continue
        failed_requires = [item for item in requires
                           if status_by_name.get(item) not in ("pass",)]
        if failed_requires:
            entry = _skip(f"prerequisite check(s) did not pass: {failed_requires}")
            entry["name"] = name
            entry["requires"] = list(requires)
            results.append(entry)
            status_by_name[name] = "skip"
            continue
        function = _resolve(function_name)
        try:
            outcome = function(ctx)
        except Exception as exc:
            outcome = _fail(f"{type(exc).__name__}: {exc}")
        entry = {"name": name, "requires": list(requires), **outcome}
        results.append(entry)
        status_by_name[name] = entry["status"]
    failed = [entry["name"] for entry in results if entry["status"] == "fail"]
    skipped = [entry["name"] for entry in results if entry["status"] == "skip"]
    status = "failed" if failed else "passed"
    return {
        "status": status,
        "complete": not skipped and not failed,
        "failed": failed,
        "skipped": skipped,
        "checks": results,
        "peak_vram_bytes": ctx.scratch.get("peak_vram_bytes"),
        "load_report": ctx.load_report or None,
        "started_training": False,
    }
