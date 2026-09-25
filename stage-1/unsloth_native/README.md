# Native Unsloth Stage 1

This is the small, native training path: Unsloth loads and patches the exact
model, TRL `SFTTrainer` owns optimization/checkpoints/resume, and the existing
Stage 1 preparation remains the authority for splits, quarantine and
assistant-only labels.

## Design

- `models.py`: declarative model registry and the only model-specific loader
  switch (`FastLanguageModel` versus `FastVisionModel`).
- `dataset.py`: adapts the verified export/prepared artifacts into pretokenized
  `input_ids` and assistant-only `labels` (`-100` on prompt/padding).
- `runner.py`: one native SFT lifecycle shared by every model.
- `train_minicpm.py` / `train_qwen.py`: intentionally thin entry points.

To add a model, add a `NativeModelSpec` and its Stage 1 YAML configs. Add a thin
launcher only when a dedicated command is useful; do not copy the runner.

## MiniCPM LoRA overfit gate

```bash
python stage-1/unsloth_native/train_minicpm.py \
  --method lora \
  --gate overfit-100 \
  --export /absolute/export \
  --prepared /absolute/prepared-minicpm \
  --run-dir /absolute/runs-native/minicpm-overfit-100 \
  --cache-dir /absolute/huggingface-cache \
  --local-files-only
```

## Qwen LoRA overfit gate

```bash
python stage-1/unsloth_native/train_qwen.py \
  --method lora \
  --gate overfit-100 \
  --export /absolute/export \
  --prepared /absolute/prepared-qwen \
  --run-dir /absolute/runs-native/qwen-overfit-100 \
  --cache-dir /absolute/huggingface-cache \
  --local-files-only
```

`--resume auto` is the default and uses Trainer's newest native checkpoint in
that run directory. Use `--resume none` for a deliberate clean start. Gates do
not chain automatically; use a separate run directory for each gate/method.

Use the model-specific locked environment (`minicpm5-2b.lock` or
`qwen3.5-4b.lock`). LoRA is the recommended A40 path. `--method full` requests
full-parameter BF16 SFT and should only be scheduled after a separate memory
probe on appropriate high-memory hardware; it never silently falls back to
LoRA or quantization.
