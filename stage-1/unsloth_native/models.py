"""Declarative model registry and native Unsloth model construction."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

from stage1 import adapters, formatting
from stage1.errors import DataError


@dataclass(frozen=True)
class NativeModelSpec:
    key: str
    adapter: str
    loader_name: str
    configs: dict[str, str]
    text_only: bool = True

    def config_path(self, method: str) -> Path:
        try:
            relative = self.configs[method]
        except KeyError as exc:
            raise DataError(
                f"Model {self.key!r} does not support method {method!r}") from exc
        return Path(__file__).resolve().parents[1] / relative


MODEL_SPECS = {
    "minicpm5-2b": NativeModelSpec(
        key="minicpm5-2b",
        adapter="minicpm5",
        loader_name="FastLanguageModel",
        configs={
            "lora": "configs/minicpm5-2b-lora.yaml",
            "full": "configs/minicpm5-2b-full.yaml",
        },
    ),
    "qwen3.5-4b": NativeModelSpec(
        key="qwen3.5-4b",
        adapter="qwen3.5",
        loader_name="FastVisionModel",
        configs={
            "lora": "configs/qwen3.5-4b-lora.yaml",
            "full": "configs/qwen3.5-4b-full.yaml",
        },
    ),
}


def get_spec(key: str) -> NativeModelSpec:
    try:
        return MODEL_SPECS[key]
    except KeyError as exc:
        raise DataError(
            f"Unknown native model {key!r}; choose from {sorted(MODEL_SPECS)}") from exc


def _source(config, *, local_files_only: bool, cache_dir):
    if config.model.local_path is not None:
        return str(config.model.local_path), False
    if not local_files_only:
        return config.model.id, True
    try:
        from huggingface_hub import snapshot_download
        snapshot = snapshot_download(
            repo_id=config.model.id,
            revision=config.model.revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=True,
        )
    except Exception as exc:
        raise DataError(
            f"Pinned snapshot {config.model.id}@{config.model.revision} is not "
            f"available locally ({type(exc).__name__}: {exc})") from exc
    return snapshot, False


def _text_tokenizer(tokenizer_or_processor):
    """Return the tokenizer used by the native text-only training path.

    FastVisionModel returns a multimodal processor for Qwen vision checkpoints,
    whereas FastLanguageModel returns a tokenizer directly.  Stage 1 trains
    pretokenized text only, so all identity, collation, and Trainer operations
    must consistently use the processor's underlying tokenizer.
    """
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    if not hasattr(tokenizer_or_processor, "__len__"):
        raise DataError(
            "Native model loader returned neither a tokenizer nor a processor "
            "with a tokenizer")
    return tokenizer_or_processor


def load_native_model(spec: NativeModelSpec, config, prepared_model: dict, *,
                      local_files_only: bool, cache_dir=None):
    """Load the exact base and apply either native LoRA or full SFT mode."""
    try:
        import unsloth
        import torch
    except ImportError as exc:  # pragma: no cover - GPU environment only
        raise DataError("torch and unsloth are required for native training") from exc

    Loader = getattr(unsloth, spec.loader_name, None)
    if Loader is None:
        raise DataError(
            f"Installed Unsloth has no {spec.loader_name}; cannot load {spec.key}")
    source, pass_revision = _source(
        config, local_files_only=local_files_only, cache_dir=cache_dir)
    load_kwargs = {
        "model_name": source,
        "max_seq_length": config.data.max_seq_len,
        "dtype": torch.bfloat16,
        "load_in_4bit": False,
        "load_in_8bit": False,
        "trust_remote_code": config.model.trust_remote_code,
        "local_files_only": local_files_only,
    }
    if cache_dir:
        load_kwargs["cache_dir"] = str(cache_dir)
    if pass_revision:
        load_kwargs["revision"] = config.model.revision
    if config.training.method == "full":
        load_kwargs["full_finetuning"] = True
    try:
        model, tokenizer_or_processor = Loader.from_pretrained(**load_kwargs)
    except Exception as exc:
        raise DataError(
            f"Native {spec.loader_name} load failed for {config.model.id}@"
            f"{config.model.revision}: {type(exc).__name__}: {exc}") from exc

    tokenizer = _text_tokenizer(tokenizer_or_processor)

    if config.tokenizer.pad_token:
        tokenizer.pad_token = config.tokenizer.pad_token
    if tokenizer.pad_token_id is None:
        raise DataError("Tokenizer has no padding token after native model load")
    if config.model.local_path is None:
        tokenizer.name_or_path = config.model.id

    template = formatting.resolve_chat_template(tokenizer, config.tokenizer)
    fingerprint = adapters.tokenizer_fingerprint(tokenizer, template, config)
    differences = adapters.compare_fingerprints(prepared_model, fingerprint)
    if differences:
        raise DataError(
            "Native model tokenizer differs from dataset preparation:\n" +
            "\n".join(f"  - {line}" for line in differences))

    if config.training.method == "lora":
        lora = config.lora
        if lora is None:
            raise DataError("LoRA method selected without a LoRA configuration")
        model = Loader.get_peft_model(
            model,
            r=lora.rank,
            target_modules=list(lora.target_modules),
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            bias=lora.bias,
            use_gradient_checkpointing=(
                "unsloth" if config.training.gradient_checkpointing else False),
            random_state=config.seed,
        )
    for_training = getattr(Loader, "for_training", None)
    if callable(for_training):
        parameters = inspect.signature(for_training).parameters
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values())
        if "use_gradient_checkpointing" in parameters or accepts_kwargs:
            model = for_training(
                model,
                use_gradient_checkpointing=config.training.gradient_checkpointing)
        else:
            model = for_training(model)
    return model, tokenizer, template, fingerprint
