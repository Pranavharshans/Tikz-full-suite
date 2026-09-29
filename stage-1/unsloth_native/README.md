# Native Stage 1 training

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
- `train_minicpm.py` / `train_qwen.py` / `train_lfm.py`: intentionally thin entry points.

To add a model, add a `NativeModelSpec` and its Stage 1 YAML configs. Add a thin
launcher only when a dedicated command is useful; do not copy the runner. The
Ling integration uses the same runner through `../ling_native`, but selects the
isolated Transformers/PEFT backend rather than importing Unsloth.

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

## LFM2.5-2.6B LoRA gates and Slurm resume

The LFM config pins `LiquidAI/LFM2.5-2.6B` at revision
`654f9463ce32b05d0429d76fe1f580b27d4c1ac0`, uses BF16 LoRA through
`FastLanguageModel`, and applies the model's ChatML markers with a direct
assistant/TikZ target. Stage 1 keeps this run text-only; diagram images remain
available to downstream evaluation.

Install the model-specific lock in its own Python 3.12 environment and prepare
a new tokenizer-specific dataset directory:

```bash
python3.12 -m venv /shared/$USER/envs/lfm2.5-2.6b
/shared/$USER/envs/lfm2.5-2.6b/bin/pip install -r stage-1/locks/lfm2.5-2.6b.lock

/shared/$USER/envs/lfm2.5-2.6b/bin/python stage-1/scripts/prepare_dataset.py \
  --config stage-1/configs/lfm2.5-2.6b-lora.yaml \
  --export /absolute/export \
  --prepared /absolute/prepared-lfm25 \
  --cache-dir /absolute/huggingface-cache
```

Submit one gate per Slurm job from the repository root. The wrapper targets
one RTX PRO 6000, requests no separate `--mem`, and invokes the standalone
LFM trainer. Override wall time on `sbatch` as shown:

```bash
LFM_PYTHON=/absolute/envs/lfm2.5-2.6b/bin/python
LFM_EXPORT=/absolute/export
LFM_PREPARED=/absolute/prepared-lfm25
LFM_RUNS=/absolute/runs/lfm25-v1
LFM_CACHE=/absolute/huggingface-cache

# First, give the bounded overfit gate a 30-minute probe allocation.
sbatch --time=00:30:00 stage-1/unsloth_native/lfm_stage1.sbatch \
  "$LFM_PYTHON" "$LFM_EXPORT" "$LFM_PREPARED" \
  "$LFM_RUNS/overfit-100" overfit-100 none "$LFM_CACHE"
```

Check the log, Slurm state and `metrics.json` before increasing the allocation:

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
tail -n 80 slurm-lfm25-JOB_ID.out
```

After the allocation ends, inspect its state and log. If it hit the time limit
or failed after a checkpoint, resubmit with the same config and
`$LFM_RUNS/overfit-100` path. `auto` restores the newest native Trainer
checkpoint, optimizer and scheduler state. If the short probe ended before its
first checkpoint was written, rerun with `none`; the same run directory is
safe while it contains no checkpoint. Do not change the run path or config
while resuming.

```bash
# Resume the interrupted overfit gate in place.
sbatch --time=02:00:00 stage-1/unsloth_native/lfm_stage1.sbatch \
  "$LFM_PYTHON" "$LFM_EXPORT" "$LFM_PREPARED" \
  "$LFM_RUNS/overfit-100" overfit-100 auto "$LFM_CACHE"

# After metrics.json reports passed=true, run smoke in a fresh directory.
sbatch --time=04:00:00 stage-1/unsloth_native/lfm_stage1.sbatch \
  "$LFM_PYTHON" "$LFM_EXPORT" "$LFM_PREPARED" \
  "$LFM_RUNS/smoke-1000" smoke-1000 none "$LFM_CACHE"

# After smoke passes, run one full-data epoch of LoRA SFT in a fresh directory.
sbatch --time=12:00:00 stage-1/unsloth_native/lfm_stage1.sbatch \
  "$LFM_PYTHON" "$LFM_EXPORT" "$LFM_PREPARED" \
  "$LFM_RUNS/full" full none "$LFM_CACHE"
```

If smoke or full is interrupted, resubmit that same gate and same run
directory with `auto`. A finished gate writes `metrics.json`; keep its evidence
and start the next gate separately. In this native path, `full` means one
epoch over the complete eligible split while still using LoRA, not full-
parameter fine-tuning. Start with `overfit-100`; no later gate is automatic.

`--resume auto` is the default and uses Trainer's newest native checkpoint in
that run directory. An explicit checkpoint from another native run is accepted
only when its recorded model, configuration, gate, dataset and tokenizer
identity exactly match the new run. Use `--resume none` for a deliberate clean
start. Gates do not chain automatically; use a separate run directory for each
gate/method.

Use the model-specific locked environment (`minicpm5-2b.lock` or
`qwen3.5-4b.lock`). LoRA is the recommended A40 path. `--method full` requests
full-parameter BF16 SFT and should only be scheduled after a separate memory
probe on appropriate high-memory hardware; it never silently falls back to
LoRA or quantization.

## Validation status

Last updated: 2026-09-28.

| Model | Method | Gate | Hardware | Result | Evidence |
| --- | --- | --- | --- | --- | --- |
| `openbmb/MiniCPM5-2B` | LoRA BF16 | `overfit-100` | 1x NVIDIA A40 48 GB | **PASS** | Slurm exit `0:0`, 24m03s |
| `openbmb/MiniCPM5-2B` | LoRA BF16 | `smoke-1000` | 1x NVIDIA A40 48 GB | **PASS** | Slurm exit `0:0`, 12m39s |
| `openbmb/MiniCPM5-2B` | LoRA BF16 | native resume drill | 1x NVIDIA A40 48 GB | **PASS** | Resumed step 50 and completed step 63 |
| `openbmb/MiniCPM5-2B` | LoRA BF16 | `full` | 1x NVIDIA A40 48 GB | **PASS** | [Dated full-run report](../results/2026-09-27-minicpm5-2b-tikz-lora.md) |
| `Qwen/Qwen3.5-4B` | LoRA BF16 | `full` | 1x NVIDIA RTX PRO 6000 Blackwell | **PASS** | [Dated full-run report](../results/2026-09-28-qwen3.5-4b-tikz-lora.md) |
| `LiquidAI/LFM2.5-2.6B` | LoRA BF16 | Any | 1x NVIDIA RTX PRO 6000 Blackwell | **IMPLEMENTED / GPU UNVERIFIED** | [Integration record](../results/2026-09-29-lfm2.5-2.6b-integration.md) |
| MiniCPM and Qwen | Full BF16 SFT | Any | Not run | **PENDING** | Requires a separate memory probe on suitable high-memory hardware |
| `inclusionAI/Ling-3.0-tiny` | LoRA BF16 | Any | Not run | **CODE READY / GPU UNVERIFIED** | [Integration and required gates](../ling_native/README.md) |

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

### MiniCPM resume evidence

The native resume drill loaded the smoke run's step-50 checkpoint into a fresh,
identity-matched run. It restored the native Trainer state, optimizer, scheduler
and adapter, continued through step 63, reached epoch 1.0, evaluated all 500
validation examples and wrote a complete final artifact. The training
continuation took `82.2381` seconds. This also exercised the cached Arrow dataset
reuse path used by repeated and resumed runs.

## Next gate

Run one epoch of MiniCPM LoRA SFT on the complete eligible prepared training
split. Keep native checkpointing enabled and use `--resume auto` so a Slurm
interruption can continue in the same run directory. Do not initialize this run
from either the overfit or smoke adapter.
