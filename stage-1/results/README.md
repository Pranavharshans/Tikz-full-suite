# Stage 1 training results

Stage 1 implementation and training records:

- [2026-09-27 — MiniCPM5-2B TikZ LoRA](2026-09-27-minicpm5-2b-tikz-lora.md)
- [2026-09-27 — Ling-3.0-tiny TikZ LoRA integration](2026-09-27-ling3-tiny-integration.md) — code and tokenizer validated; GPU gates pending
- [2026-09-28 — Qwen3.5-4B TikZ LoRA](2026-09-28-qwen3.5-4b-tikz-lora.md)
- [2026-09-29 — LFM2.5-2.6B TikZ LoRA integration](2026-09-29-lfm2.5-2.6b-integration.md) — `overfit-100` and `smoke-1000` passed on RTX PRO 6000; a two-epoch full-data gate is configured next
- [2026-09-29 — LFM2.5-8B-A1B TikZ LoRA integration](2026-09-29-lfm2.5-8b-a1b-integration.md) — isolated sparse-MoE Unsloth path added; exact GPU gates pending
- [2026-09-30 — Gemma 4 12B TikZ LoRA integration](2026-09-30-gemma4-12b-integration.md) — overfit, smoke, and resume gates passed on RTX PRO 6000; full-data training in progress
- [2026-10-01 — Qwen3.5-9B TikZ LoRA](2026-10-01-qwen3.5-9b-tikz-lora.md) — two-epoch full gate passed and public base-plus-adapter package verified

Completed-run reports summarize the published Hugging Face model cards and
full Stage 1 gate evidence. Integration-only records state their unverified
GPU work explicitly and must not be interpreted as training results. Large
model weights, adapters, checksums, resolved configuration, and
machine-readable metrics remain outside Git.
