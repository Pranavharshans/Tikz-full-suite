"""Declarative registry for the audited native training backends."""
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
    backend: str = "unsloth"
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
    "lfm2.5-2.6b": NativeModelSpec(
        key="lfm2.5-2.6b",
        adapter="lfm25-2.6b",
        loader_name="FastLanguageModel",
        configs={"lora": "configs/lfm2.5-2.6b-lora.yaml"},
    ),
    "ling3-tiny": NativeModelSpec(
        key="ling3-tiny",
        adapter="ling3",
        loader_name="AutoModelForCausalLM",
        configs={"lora": "configs/ling3-tiny-lora.yaml"},
        backend="transformers-peft",
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


def _verify_tokenizer(tokenizer, config, prepared_model):
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
    return template, fingerprint


def _matched_lora_modules(model, targets) -> dict[str, list[str]]:
    """Resolve PEFT suffix targets before mutating the model.

    This is particularly important for custom MoE architectures: a typo must
    not produce a partially adapted checkpoint, and generic MLP suffixes must
    not accidentally attach an adapter to every routed expert.
    """
    result = {target: [] for target in targets}
    for name, module in model.named_modules():
        if module.__class__.__name__ != "Linear":
            continue
        for target in targets:
            if name == target or name.endswith(f".{target}"):
                result[target].append(name)
    missing = [target for target, names in result.items() if not names]
    if missing:
        available = sorted({name.rsplit(".", 1)[-1]
                            for name, module in model.named_modules()
                            if module.__class__.__name__ == "Linear"})
        raise DataError(
            f"LoRA target module(s) {missing} do not exist in the loaded model; "
            f"available Linear suffixes are {available}")
    return result


def _load_transformers_peft_model(spec, config, prepared_model, *,
                                  local_files_only: bool, cache_dir=None):
    try:
        import torch
        from peft import LoraConfig as PeftLoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - GPU environment only
        raise DataError(
            "torch, transformers and peft are required for Ling training") from exc

    if config.training.method != "lora":
        raise DataError(
            f"{spec.key} supports audited BF16 LoRA only; full SFT has not "
            "passed a memory and checkpoint probe")
    source, pass_revision = _source(
        config, local_files_only=local_files_only, cache_dir=cache_dir)
    common = {
        "trust_remote_code": config.model.trust_remote_code,
        "local_files_only": local_files_only,
    }
    if cache_dir:
        common["cache_dir"] = str(cache_dir)
    if pass_revision:
        common["revision"] = config.model.revision
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, **common)
        model_kwargs = dict(common, torch_dtype=torch.bfloat16)
        if config.model.attn_implementation:
            model_kwargs["attn_implementation"] = config.model.attn_implementation
        model = AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
    except Exception as exc:
        raise DataError(
            f"Native Transformers load failed for {config.model.id}@"
            f"{config.model.revision}: {type(exc).__name__}: {exc}") from exc

    template, fingerprint = _verify_tokenizer(
        tokenizer, config, prepared_model)
    lora = config.lora
    if lora is None:
        raise DataError("LoRA method selected without a LoRA configuration")
    _matched_lora_modules(model, lora.target_modules)
    peft_config = PeftLoraConfig(
        task_type="CAUSAL_LM",
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias=lora.bias,
        target_modules=list(lora.target_modules),
    )
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant":
                    config.training.gradient_checkpointing_use_reentrant,
            })
        enable_inputs = getattr(model, "enable_input_require_grads", None)
        if callable(enable_inputs):
            enable_inputs()
    model.train()
    # Prove that every configured target produced adapter parameters and that
    # PEFT froze every base parameter. This is stronger than merely observing
    # a non-zero trainable count on a custom remote-code architecture.
    adapters.verify_lora_trainables(model.named_parameters(), lora)
    return model, tokenizer, template, fingerprint


def load_native_model(spec: NativeModelSpec, config, prepared_model: dict, *,
                      local_files_only: bool, cache_dir=None):
    """Load the exact base and apply either native LoRA or full SFT mode."""
    if spec.backend == "transformers-peft":
        return _load_transformers_peft_model(
            spec, config, prepared_model,
            local_files_only=local_files_only, cache_dir=cache_dir)
    if spec.backend != "unsloth":
        raise DataError(f"Unknown native backend {spec.backend!r}")
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

    template, fingerprint = _verify_tokenizer(
        tokenizer, config, prepared_model)

    if config.training.method == "lora":
        lora = config.lora
        if lora is None:
            raise DataError("LoRA method selected without a LoRA configuration")
        adapters.validate_target_modules(
            adapters.module_short_names(model), lora.target_modules)
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
        adapters.verify_lora_trainables(model.named_parameters(), lora)
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
