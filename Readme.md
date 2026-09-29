# TikZ training

Pipeline: data cleaning → supervised fine-tuning → on-policy distillation → online RL.

## Stage 1 model support

On **September 27, 2026**, Stage 1 added integration support for
[`inclusionAI/Ling-3.0-tiny`](stage-1/results/2026-09-27-ling3-tiny-integration.md)
with BF16 LoRA through a dedicated Transformers/PEFT backend. The Ling
configuration, launcher, and tokenizer handling are documented in the
[Stage 1 guide](stage-1/README.md) and [Ling backend guide](stage-1/ling_native/README.md).

The Ling integration has passed code and tokenizer validation. Its GPU gates
and full training run are still pending.

To train Ling, first prepare a Ling-specific tokenizer dataset, then complete
the `overfit-100` and `smoke-1000` GPU gates described in the
[September 27 integration record](stage-1/results/2026-09-27-ling3-tiny-integration.md).

On **September 29, 2026**, Stage 1 added a standalone Unsloth/TRL LoRA path for
[`LiquidAI/LFM2.5-2.6B`](stage-1/results/2026-09-29-lfm2.5-2.6b-integration.md).
The overfit, smoke, resume and full-data Slurm gates are documented, but the
model-specific GPU run has not started.
