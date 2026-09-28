# MiniCPM5-2B TikZ LoRA — 2026-09-27

Stage 1 BF16 LoRA supervised fine-tuning completed successfully for
`openbmb/MiniCPM5-2B`. The full native gate passed and the public package
contains the pinned base checkpoint plus the separate, unmerged adapter.

- Hugging Face: [Praha-Labs/MiniCPM5-2B-TikZ-LoRA](https://huggingface.co/Praha-Labs/MiniCPM5-2B-TikZ-LoRA)
- Base revision: `12a3808a956f869c767195e9266b59c4d21d92e2`
- Trainer: native Unsloth loading with TRL `SFTTrainer`
- Hardware: 1× NVIDIA A40 48 GB
- Maximum sequence length: 8,192
- LoRA: rank 64, alpha 64, dropout 0.0
- Trainable parameters: 100,466,688 of 2,617,223,168 (3.8387%)

## Dataset and tokens

- Training examples: 95,347
- Quarantined overlength examples: 1,153
- Prompt tokens: 19,578,751
- Supervised assistant tokens: 56,063,413
- Unpadded dataset tokens: 75,642,164
- Trainer-reported padded input tokens: 107,296,408

Only assistant/TikZ tokens contributed to loss. Prompt tokens were masked and
overlength examples were quarantined rather than truncated.

## Optimization and results

- Epochs: 1
- Effective batch size: 16 (2 per device × 8 accumulation)
- Peak learning rate: `1e-4`
- Scheduler: cosine
- Full gate: **PASS**
- Validation loss: **0.449503**
- First logged loss: 0.589100
- Final logged loss: 0.572400
- Minimum logged loss: 0.363200
- Mean logged loss: 0.537604
- Total FLOPs: `1.512773958542721e+18`
- Accumulated Slurm allocation time: 10h 09m 38s across six allocations

Five allocations ended at their requested time limits and resumed from native
Trainer checkpoints. The sixth allocation completed the epoch, evaluation,
and final adapter save with exit code `0:0`.

## Interpretation

The validation loss is token-level cross-entropy against one reference TikZ
implementation. It does not directly measure LaTeX compilation, instruction
coverage, or rendered-image quality. Those require a separate compile/render
evaluation.
