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

## Validation status

Last updated: 2026-09-25.

| Model | Method | Gate | Hardware | Result | Evidence |
| --- | --- | --- | --- | --- | --- |
| `openbmb/MiniCPM5-2B` | LoRA BF16 | `overfit-100` | 1x NVIDIA A40 48 GB | **PASS** | Slurm exit `0:0`, 24m03s |
| `openbmb/MiniCPM5-2B` | LoRA BF16 | `smoke-1000` | 1x NVIDIA A40 48 GB | **PASS** | Slurm exit `0:0`, 12m39s |
| `openbmb/MiniCPM5-2B` | LoRA BF16 | native resume drill | 1x NVIDIA A40 48 GB | **NEXT** | Resume from a native smoke checkpoint |
| `openbmb/MiniCPM5-2B` | LoRA BF16 | `full` | 1x NVIDIA A40 48 GB | **BLOCKED** | Run only after the resume drill passes |
| `Qwen/Qwen3.5-4B` | LoRA BF16 | `overfit-100` | Not run | **PENDING** | Requires its own prepared tokenizer artifact and fresh run directory |
| Both models | Full BF16 SFT | Any | Not run | **PENDING** | Requires a separate memory probe on suitable high-memory hardware |

### MiniCPM overfit evidence

Recorded evidence from `metrics.json`:

- `passed: true`; every acceptance check passed.
- 100 training examples, with the same 100 examples used for validation by
  design for the memorization gate.
- 21,493 prompt tokens and 60,059 supervised assistant tokens.
- 1,153 overlength rows excluded using the prepared MiniCPM quarantine.
- Evaluation loss: `0.0120343473` after 40 epochs.
- Training runtime: `1,350.908` seconds; total Slurm elapsed time: `24m03s`.
- Final LoRA adapter and tokenizer artifacts were complete.

This validates the exact A40 model load, audited dataset adapter, MiniCPM chat
template and tokenizer, assistant-only masking, forward/backward optimization,
evaluation, and final adapter save path. It does not yet validate generalization
or native checkpoint resume because this run started with no checkpoint.

### MiniCPM smoke evidence

The independent `smoke-1000` gate also completed successfully:

- `passed: true`; every smoke acceptance check passed.
- 1,000 training examples and 500 held-out validation examples.
- 586,270 supervised training tokens and 315,045 supervised validation tokens.
- Final evaluation loss: `0.7318511605` after one epoch.
- Training runtime: `463.0896` seconds; total Slurm elapsed time: `12m39s`.
- Native checkpoints were written at steps 25, 50 and 63.
- The final LoRA adapter and tokenizer artifacts were complete.

The `first_export_id` statistic is the first row encountered while streaming the
whole export, so it is expected to match between dataset builds. It is not the
first selected row and is not evidence of split overlap. Dataset construction
separately checks that every emitted row ID exactly matches the selected IDs for
the requested split.

## Next gate

Perform one bounded native checkpoint-resume drill and prove that optimizer,
scheduler, trainer state and adapter weights continue from a native checkpoint.
Do not start the full 96K-example training run until that drill passes.
