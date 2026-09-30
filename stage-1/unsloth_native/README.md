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

# After smoke passes, run the LFM config's two full-data epochs of LoRA SFT
# in a fresh directory. The cosine schedule spans both epochs.
sbatch --time=12:00:00 stage-1/unsloth_native/lfm_stage1.sbatch \
  "$LFM_PYTHON" "$LFM_EXPORT" "$LFM_PREPARED" \
  "$LFM_RUNS/full" full none "$LFM_CACHE"
```

If smoke or full is interrupted, resubmit that same gate and same run
directory with `auto`. A finished gate writes `metrics.json`; keep its evidence
and start the next gate separately. For LFM, `full` means two epochs over the
complete eligible split while still using LoRA, not full-parameter fine-tuning.
Start with `overfit-100`; no later gate is automatic.

`--resume auto` is the default and uses Trainer's newest native checkpoint in
that run directory. An explicit checkpoint from another native run is accepted
only when its recorded model, configuration, gate, dataset and tokenizer
identity exactly match the new run. Use `--resume none` for a deliberate clean
start. Gates do not chain automatically; use a separate run directory for each
gate/method.

## LFM2.5-8B-A1B sparse-MoE gates

The 8B-A1B integration is isolated from the dense 2.6B LFM path. It uses
adapter slug `lfm25-8b-a1b` and has its own environment, prepared dataset,
standalone launcher, Slurm wrapper, and run directories. Do not reuse the
2.6B prepared directory.

```bash
python3.12 -m venv /shared/$USER/envs/lfm2.5-8b-a1b
/shared/$USER/envs/lfm2.5-8b-a1b/bin/pip install \
  -r stage-1/locks/lfm2.5-8b-a1b.lock

/shared/$USER/envs/lfm2.5-8b-a1b/bin/python \
  stage-1/scripts/prepare_dataset.py \
  --config stage-1/configs/lfm2.5-8b-a1b-lora.yaml \
  --export /absolute/export \
  --prepared /absolute/prepared-lfm25-8b-a1b \
  --cache-dir /absolute/huggingface-cache

sbatch --time=00:30:00 \
  stage-1/unsloth_native/lfm8b_a1b_stage1.sbatch \
  /shared/$USER/envs/lfm2.5-8b-a1b/bin/python \
  /absolute/export \
  /absolute/prepared-lfm25-8b-a1b \
  /absolute/runs/lfm25-8b-a1b/overfit-100 \
  overfit-100 none /absolute/huggingface-cache
```

The first GPU gate must prove that the pinned stack loads
`Lfm2MoeForCausalLM`, attaches trainable adapters to the requested dense and
expert families, fits the RTX PRO 6000, and saves a resumable adapter. Do not
schedule smoke or full training if any target is filtered or any expert
adapter is missing.

Use the model-specific locked environment (`minicpm5-2b.lock` or
`qwen3.5-4b.lock`). LoRA is the recommended A40 path. `--method full` requests
full-parameter BF16 SFT and should only be scheduled after a separate memory
probe on appropriate high-memory hardware; it never silently falls back to
LoRA or quantization.

## Validation status

Last updated: 2026-09-29.

| Model | Method | Gate | Hardware | Result | Evidence |
| `openbmb/MiniCPM5-2B` | LoRA BF16 | `overfit-100` | 1x NVIDIA A40 48 GB | **PASS** | Slurm exit `0:0`, 24m03s |
| `openbmb/MiniCPM5-2B` | LoRA BF16 | `smoke-1000` | 1x NVIDIA A40 48 GB | **PASS** | Slurm exit `0:0`, 12m39s |
| `openbmb/MiniCPM5-2B` | LoRA BF16 | native resume drill | 1x NVIDIA A40 48 GB | **PASS** | Resumed step 50 and completed step 63 |
| `openbmb/MiniCPM5-2B` | LoRA BF16 | `full` | 1x NVIDIA A40 48 GB | **PASS** | [Dated full-run report](../results/2026-09-27-minicpm5-2b-tikz-lora.md) |
| `Qwen/Qwen3.5-4B` | LoRA BF16 | `full` | 1x NVIDIA RTX PRO 6000 Blackwell | **PASS** | [Dated full-run report](../results/2026-09-28-qwen3.5-4b-tikz-lora.md) |
| `LiquidAI/LFM2.5-2.6B` | LoRA BF16 | `overfit-100` | 1x NVIDIA RTX PRO 6000 Blackwell | **PASS** | Eval loss `0.01573`; 280/280 steps; Slurm job `4401546`. See the [integration record](../results/2026-09-29-lfm2.5-2.6b-integration.md). |
| `LiquidAI/LFM2.5-2.6B` | LoRA BF16 | `smoke-1000` | 1x NVIDIA RTX PRO 6000 Blackwell | **PASS** | Eval loss improved to `0.7829`; 63/63 steps and final artifact completed. |
| `LiquidAI/LFM2.5-8B-A1B` | LoRA BF16 | Any | 1x NVIDIA RTX PRO 6000 Blackwell | **IMPLEMENTED / GPU UNVERIFIED** | [Integration record](../results/2026-09-29-lfm2.5-8b-a1b-integration.md) |
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

## Gemma 4 12B standalone LoRA

`train_gemma12b.py` selects `unsloth/gemma-4-12b-it` at
`55cdba0740a9765956f49501f689a66b098feda3`, the instruction-tuned Unified
checkpoint. It loads BF16 weights through `FastVisionModel`; Stage 1 inputs
are text only, and vision/audio adapter attachment is disabled. The pinned
verbatim template uses Gemma turn markers and an empty thought prefix with
thinking disabled, preserving literal TikZ labels and whitespace.

Use a separate Python 3.12 Linux/NVIDIA environment and run
`bash stage-1/unsloth_native/install_gemma12b.sh`. The base lock pins
Transformers 5.17.0 and Unsloth Zoo 2026.9.8. Unsloth 2026.9.12 still declares
`transformers<=5.5.0`, before Gemma Unified support; the installer first installs
Unsloth with its dependencies, then applies the newer base lock and retains
the exact Unsloth wheel with `--no-deps`. `pip check`
will report this known metadata conflict. This override must pass the GPU
load/forward/backward gates before training is approved. This is a
candidate environment: CPU config/formatting checks do not prove CUDA loading,
LoRA attachment, memory fit, or training. Start with `overfit-100`, require
`metrics.json` to report `passed: true`, then run `smoke-1000` and its resume
drill before the `full` dataset gate. Full means the full dataset with LoRA;
full-parameter SFT is not enabled for this model.

```bash
python stage-1/scripts/prepare_dataset.py \
  --config stage-1/configs/gemma4-12b-lora.yaml \
  --export /absolute/export \
  --prepared /absolute/prepared-gemma4-12b \
  --cache-dir /absolute/huggingface-cache

python stage-1/unsloth_native/train_gemma12b.py \
  --gate overfit-100 \
  --export /absolute/export \
  --prepared /absolute/prepared-gemma4-12b \
  --run-dir /absolute/runs/gemma4-12b/overfit-100 \
  --resume auto \
  --cache-dir /absolute/huggingface-cache \
  --local-files-only
```

After downloading the pinned snapshot and preparing Gemma-specific data,
`gemma12b_stage1.sbatch` accepts the same positional arguments as the other
native wrappers: `PYTHON EXPORT PREPARED RUN_DIR GATE RESUME [CACHE_DIR]`.
It targets one RTX PRO 6000. No Gemma GPU or Slurm job has been run as part of
this integration. Upstream sources: [model](https://huggingface.co/unsloth/gemma-4-12b-it),
[Unsloth guide](https://unsloth.ai/docs/models/gemma-4).
