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
