# Stage 1: supervised fine-tuning

BF16 supervised fine-tuning for text-to-TikZ generation, trained with Unsloth
on one NVIDIA RTX PRO 6000 Blackwell (96 GB). Two models share one training
implementation and differ only by configuration:

| Config | Model | Adapter |
| --- | --- | --- |
| `configs/qwen3.5-4b-full.yaml` | `Qwen/Qwen3.5-4B` @ `851bf6e8…` | `qwen3.5` (multimodal checkpoint, trained text-only) |
| `configs/minicpm5-2b-full.yaml` | `openbmb/MiniCPM5-2B` @ `12a3808a…` | `minicpm5` (text-only Llama, the comparison baseline) |
| `configs/qwen3.5-4b-lora.yaml` | same model, BF16 LoRA (A40 profile) | `qwen3.5` |
| `configs/minicpm5-2b-lora.yaml` | same model, BF16 LoRA (A40 profile) | `minicpm5` |
| `configs/qwen3.5-4b-lora-rtx.yaml` | same model, BF16 LoRA (RTX PRO 6000 profile) | `qwen3.5` |
| `configs/minicpm5-2b-lora-rtx.yaml` | same model, BF16 LoRA (RTX PRO 6000 profile) | `minicpm5` |

Training mode is explicit and first-class:

```yaml
training:
  method: full   # full-parameter BF16 SFT (default), or
  method: lora   # BF16 LoRA through Unsloth's native PEFT integration
```

LoRA is **not** QLoRA: the base checkpoint is always loaded unquantized in
BF16 (`load_in_4bit: false`, `load_in_8bit: false`), and full-parameter mode
is unchanged. The two methods have different run identities, checkpoint
layouts and final artifacts, so they can never share or resume each other's
work.

The first comparable experiment is **text-only for both models**: no images
are supplied to either model. `png_image` from the export is retained for
evaluation (rendered-image similarity) only.

Stage 1 does **not** support sequence packing. Batches are right-padded only;
`training.packing: true` is rejected by the configuration parser with an
explicit error (see section 5).

## Layout

```
stage-1/
  README.md                 this runbook
  .python-version           interpreter pin (3.12) for the GPU environments
  pyproject.toml            packaging (loose bounds; locks are authoritative)
  configs/                  common.yaml + full and lora YAML per model
  locks/                    pinned pip requirements, one file per model
  schemas/                  JSON Schemas for the artifacts listed in section 9
  src/stage1/               the shared implementation
  scripts/                  seven CLI entry points (see below)
  tests/                    synthetic CPU tests; GPU/network tests are marked
```

`cleaning/`, `stage-2/` and `stage-3/` are untouched. Stage 1 reads the
cleaning export and writes only inside `--prepared` and `--run-dir`.

## 1. Environments

Each model gets its own environment installed from its lock file, so the two
can diverge if the models ever need different Transformers releases. The
interpreter pin lives in `.python-version` (3.12); the lock files are plain
pip requirements files and contain no interpreter pin.

```bash
# Create the virtual environments with the pinned interpreter (3.12), e.g.:
python3.12 -m venv /shared/$USER/envs/qwen3.5-4b
python3.12 -m venv /shared/$USER/envs/minicpm5-2b

# Then install the locked requirements into each environment:
/shared/$USER/envs/qwen3.5-4b/bin/pip install -r stage-1/locks/qwen3.5-4b.lock
/shared/$USER/envs/minicpm5-2b/bin/pip install -r stage-1/locks/minicpm5-2b.lock
```

The locks are model-specific coherent sets inside Unsloth's constraints
(`transformers<=5.5.0`, `trl<=0.24.0`, `torch<2.13.0`). Qwen uses
Transformers 5.5.0; MiniCPM uses its officially documented 4.57.3 fallback to
avoid the incompatible Transformers-v5 path. See `locks/README.md`. The
**preflight verifies the actual environment** (and that the running
interpreter matches `.python-version`); it never trusts the lock. Run it
inside the same environment that will train.

## 2. Prepare the dataset (once, shared by both models)

Input is the audited Parquet export from the cleaning pipeline. Stage 1
re-verifies it: metadata cross-checks, per-shard file and logical checksums
(re-derived with the cleaning pipeline's exact algorithm), per-row TikZ/image
content hashes, global ordering, unique ids and provenance.

```bash
python3 stage-1/scripts/prepare_dataset.py \
  --config stage-1/configs/qwen3.5-4b-full.yaml \
  --config stage-1/configs/minicpm5-2b-full.yaml \
  --export /shared/$USER/tikz-production/export \
  --prepared /shared/$USER/tikz-stage1/prepared
```

Use `--dry-run` to validate configurations only (reads nothing, writes
nothing), and `--local-files-only` to forbid tokenizer downloads.

What preparation produces:

- `split-manifest.json` — self-verifying: `split_manifest_sha256` and the
  `data_identity` digest are recomputed and checked on every load. The hash
  covers only the deterministic payload; `created_at` is provenance and is
  excluded, so identical inputs produce identical hashes regardless of when
  preparation ran.
- Deterministic splits. Rows sharing a `tikz_sha256` **or** an `image_sha256`
  form one duplicate group (union-find); groups are shuffled with the
  configured seed and assigned whole groups to validation/test, so a duplicate
  group can never cross splits. Stable row ids are preserved.
- `models/<adapter>/token-report.json` + `.md` — token-length percentiles,
  supervised-token totals, and **eligible/quarantined counts per split** for
  that tokenizer.
- `models/<adapter>/quarantine.jsonl` — rows whose rendered prompt+assistant
  exceeds `data.max_seq_len`, with explicit reasons and exact token counts.
  The token report records the artifact's SHA256. **Nothing is ever truncated
  silently.** Preparation fails when the quarantine fraction exceeds
  `data.max_quarantined_fraction` (default 2%).
- `models/<adapter>/model.json` — tokenizer/chat-template fingerprint used by
  the preflight and trainer to refuse a different tokenizer.
- `dataset-report.json` — the data-gate evidence (`status: pass`).

### Quarantined rows are excluded everywhere

Every consumer loads and validates the quarantine artifact through
`data.load_eligibility`, which checks the artifact hash against the token
report, the entry count, that every id exists in the split manifest with the
recorded split, and the per-split eligible/quarantined counts. A mismatch is a
hard error, never a silent skip.

### Artifact anchoring chain

`dataset-report.json` records the SHA256 of the model fingerprint, the token
report and the quarantine artifact, and the run identity includes the SHA256 of
`dataset-report.json` itself. Verification follows the chain:

```
run.json identity
  └─ dataset-report.json (hash in the identity)
       ├─ models/<adapter>/model.json
       ├─ models/<adapter>/token-report.json
       │    └─ models/<adapter>/quarantine.jsonl (hash recorded in the token report)
       └─ models/<adapter>/quarantine.jsonl (hash recorded again here)
```

`verify_prepared` and `load_eligibility` verify every link, so replacing a
token report, a model fingerprint or a quarantine artifact - even together with
a consistent token report - is refused unless the dataset report is replaced
too, which changes the run identity and forces a fresh run directory.

Quarantined rows are then excluded from:

- training and validation datasets (`train.build_tokenized_dataset`),
- the gate row selection (`limit` samples are drawn from eligible ids only),
- evaluation selection (`evaluate.select_evaluation_ids`),
- memorization sampling (`evaluate.select_memorization_ids`).

Every tokenized example is re-checked immediately before emission
(`train.check_emission_row`): an overlength row, a quarantined row, or an
unexpected id refuses the dataset. The emitted ids are compared against the
expected eligible set (`train.check_emitted_ids`); a missing expected row or a
quarantined row in the dataset is a hard error.

## 3. Preflight (per model)

```bash
/shared/$USER/envs/qwen3.5-4b/bin/python stage-1/scripts/preflight.py \
  --config stage-1/configs/qwen3.5-4b-full.yaml \
  --export /shared/$USER/tikz-production/export \
  --prepared /shared/$USER/tikz-stage1/prepared \
  --run-dir /shared/$USER/tikz-stage1/runs/qwen3.5-4b
```

Checks: lock/version match, Python pin (config and `.python-version`), disk
space (including an estimated checkpoint footprint), GPU identity and VRAM,
BF16, a CUDA kernel smoke test, fused AdamW construction, Unsloth import and
full-finetuning support, dataset identity, tokenizer load and fingerprint
match, model load of the exact revision, a real forward/backward step with
finite loss and non-zero finite gradients, and three checkpoint checks:

- `checkpoint.weight_serialization` — save, read back, compare sampled tensors
  (serialization only; no reload claim),
- `checkpoint.model_reload` — save, **release the original model** (CUDA cache
  cleared), reload through the same supported loading path, and compare a
  deterministic eval-mode forward pass (finite loss, same logits shape),
- `checkpoint.trainer_resume` — write a minimal Trainer-compatible checkpoint
  using the production `transformers.get_scheduler("cosine", ...)` schedule
  and restore model weights, optimizer state, scheduler state and the trainer
  step into fresh objects, comparing them field by field.

The preflight **never starts training** and does not import the trainer (there
is a test for this). Exit codes: `0` all checks passed, `1` a check failed or
was skipped, `2` handled error. `--skip-check NAME` is recorded in the report
and makes the preflight **incomplete**, which blocks every training gate.

## 4. Gates (never advance automatically)

| Gate | Command | Acceptance criteria |
| --- | --- | --- |
| data | `prepare_dataset.py` | export verified, splits deterministic, quarantine validated |
| preflight | `preflight.py` | environment/GPU/model/checkpoint compatibility |
| overfit-100 | `train.py --gate overfit-100` | finite training, configured loss threshold and/or reduction, checkpoint save + verified reload |
| smoke-1000 | `train.py --gate smoke-1000` | finite losses and gradients, no NaN/Inf, expected optimizer steps completed, same-gate checkpoint resume verified, fixed evaluation generation succeeds, TikZ compile results recorded |
| full | `train.py --gate full` | expected steps/epochs completed, final artifact verified, finite validation loss, fixed validation generation completed, TikZ compilation metrics recorded |

Each training gate refuses to start without passing preflight evidence **for
the same run identity**, and `smoke-1000`/`full` additionally require the
previous gate's evidence. A gate that already passed is not silently repeated:
pass `--rerun-gate` to repeat it deliberately.

**Gates are isolated.** Each gate owns `checkpoints/<gate>/` and
`final/<gate>/`, and only resumes its own interrupted execution. A gate always
starts from the exact pinned base model unless it is resuming a checkpoint
from its own namespace; metadata must name the same gate, so a smoke run
cannot resume an overfit checkpoint even if the directory is copied.

**Intervals must be reachable.** Before training, the deterministic optimizer
step bounds are computed from the eligible dataset size, batch size, gradient
accumulation, epochs and `max_steps`. A `save_steps` larger than the lower
bound is refused with an explicit error, because the gate requires a same-gate
checkpoint to verify resume and none would ever be written. An `eval_steps`
beyond the schedule is recorded (`eval_within_schedule: false`) rather than
refused, since the final evaluation always runs. The shipped gates use
reachable intervals (smoke: ~62 steps, save/eval every 25; overfit: ~240-280
steps, save/eval every 50; full: ~6250 steps, save/eval every 200).

```bash
# 1,000-example smoke run (repeat the same command to resume after a stop)
/shared/$USER/envs/minicpm5-2b/bin/python stage-1/scripts/train.py \
  --config stage-1/configs/minicpm5-2b-full.yaml \
  --export /shared/$USER/tikz-production/export \
  --prepared /shared/$USER/tikz-stage1/prepared \
  --run-dir /shared/$USER/tikz-stage1/runs/minicpm5-2b \
  --gate smoke-1000
```

An incomplete or foreign checkpoint is a hard error and never falls back to an
older one silently. Exit codes: `0` gate passed, `1` gate criteria failed,
`2` handled error, `3` prerequisites missing (run the earlier gate first).

## 5. Training settings

Fixed by configuration validation: full-parameter BF16, cosine schedule, 3%
warmup, max grad norm 1.0, one epoch by default, deterministic seed shared by
splitting and training, right-padded batches only. Configurable: batch sizes,
gradient accumulation, `data.max_seq_len`, save/eval/logging frequencies,
checkpoint retention, dataloader workers, and per-gate criteria knobs
(`success_eval_loss`, `min_loss_reduction`, `generation_samples`,
`check_gradients`). Defaults: `learning_rate: 1e-5`,
`optim: adamw_torch_fused`, `gradient_checkpointing: true`,
`per_device_train_batch_size: 2`, `gradient_accumulation_steps: 8`.

Loss applies only to assistant/TikZ tokens. The mask is computed by
tokenizing the full rendered conversation once with `return_offsets_mapping`
and masking every token that starts before the assistant text boundary; a
token that *spans* the boundary is a hard error. `training.validate_batches`
(default true) re-checks masks and padding on every batch.

### Supervised-token accounting

Runs are compared by **supervised tokens** (assistant tokens that carry loss),
not only example counts or optimizer steps. Accounting is exact and
construction-limited:

- `TrainingMonitor.record_micro_batch` is called from
  `Trainer.training_step` with the labels of the micro-batch actually being
  trained on, so evaluation batches (`prediction_step`) and dataloader
  prefetches that are never consumed cannot contribute;
- `commit_step` runs at `on_step_end`, so only completed optimizer steps are
  committed; an interrupted accumulation cycle stays in
  `supervised_tokens_in_flight`;
- `finalize` commits a trailing cycle only when the Trainer reports a
  completed step;
- resume seeds the committed totals from the checkpoint metadata, so prior
  tokens are never counted twice;
- `assert_within_plan` refuses the run if committed tokens exceed the
  deterministic plan (dataset supervised tokens × epochs), which is the signal
  that accounting leaked.

The shared data collator is pure: it never mutates counters or metrics.
Labels may be flat lists, nested batch lists or 2-D tensors; the counter uses
vectorised comparisons for tensors and never iterates a multi-dimensional
tensor (iterating a 2-D tensor would count examples instead of tokens).
Metrics record `planned_supervised_tokens`,
`supervised_tokens_committed`, `supervised_tokens_in_flight`,
`supervised_tokens_exact: true`, and the accounting method string.

Checkpoint verification loads an older checkpoint's weights into the live
model to prove they load exactly. Immediately afterwards, `run_training`
restores the **final artifact's** weights and proves the restore with a forward
pass before generation or any metric is produced, and records
`final_artifact.restored_after_checkpoint_verification`, `restore_loss` and
`restore_tensors`, so the reported generation never describes a stale
checkpoint.

### LoRA mode (BF16, native PEFT)

`training.method: lora` loads the same pinned unquantized BF16 base model and
attaches adapters through Unsloth's native integration
(`FastLanguageModel`/`FastVisionModel.get_peft_model`, which uses PEFT). No
adapter layers are implemented by hand and no optimizer/scheduler behavior is
recreated: training and resume go through the Hugging Face Trainer exactly as
in full mode.

Defaults (`lora:` section): rank 64, alpha 64, dropout 0.0, bias `none`, and
target modules `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
Default LoRA learning rate is `1.0e-4`, cosine schedule, 3% warmup, max grad
norm 1.0.

**Target validation.** Before adapters are attached, every configured target
name is checked against the loaded model's module names. A target that does
not exist fails the load with the available projection-like module names; a
target is never silently ignored. The load report records the matched-module
count per target, so partial architectural coverage (for example only the
full-attention blocks of a hybrid model) is visible before training.

**Trainable assertions.** After attaching, base parameters must be frozen and
only adapter parameters (plus biases when `bias` is not `none`) may be
trainable; a missing target, an unattached adapter or a trainable base
parameter fails the load. The run records total parameters, trainable
parameters and percentage, rank/alpha/dropout/targets, base-model id and
revision, the tokenizer fingerprint hash and the dataset identity.

**Artifacts.** Full artifacts are complete model directories. LoRA artifacts
are native PEFT adapters (`adapter_config.json` + adapter weights) with
base-model provenance in `stage1-checkpoint.json`/`stage1-final.json`
(`training_method: lora`, `base_model_id`, `base_model_revision`). An adapter
directory is never treated as a standalone model: verification refuses an
adapter without base provenance, and evaluation always loads the pinned BF16
base first and then attaches the adapter.

**Isolation.** The run identity includes `training.method` and the complete
LoRA configuration, and artifact metadata records the method. Full and LoRA
runs cannot share a run directory, a checkpoint namespace or a resume: any
cross-method attempt fails with an explicit message.

**Resume.** Resume uses the Trainer's public `resume_from_checkpoint` path
with the same-gate, same-method checkpoint; nothing reconstructs optimizer or
scheduler objects for resume (the preflight's state check is verification
only, and the dependency-gated integration test exercises a fresh Trainer
continuing from a native checkpoint).

**Training-state restore on adapter reload.** A saved adapter attached to a
fresh base carries no training-mode state; Unsloth's `for_training` only flips
the `gradient_checkpointing` flags, while Transformers dispatches to each
decoder layer's `_gradient_checkpointing_func`. `prepare_model_for_training`
therefore restores the full lifecycle before any training forward:
`patch_peft_model` (the same call Unsloth's own adapter reload and
`get_peft_model` make), then `for_training` with the configured checkpointing
mode, then the exact `gradient_checkpointing_enable(use_reentrant=...)` call
`Trainer._inner_training_loop` makes, and `enable_input_require_grads` when the
loader did not register it. It refuses to return a model that reports
`gradient_checkpointing` without a callable `_gradient_checkpointing_func`, and
the preflight one-step/reload/resume checks and the production training start
run the same diagnostic before the first forward, so an incomplete state is a
clear failure with the missing module names instead of a decoder
`AttributeError`.

**Hardware profiles.** `hardware.profile` selects a built-in expectation
bundle:

| Profile | GPU regex | Min VRAM | Min free disk | Used by |
| --- | --- | --- | --- | --- |
| `rtxpro6000-full` | `RTX PRO 6000` | 88 GiB | 250 GiB | full-parameter SFT |
| `rtxpro6000-lora` | `RTX PRO 6000` | 60 GiB | 60 GiB | RTX LoRA configs |
| `a40-lora` | `\bA40\b` | 44 GiB (strict 44–48) | 60 GiB | A40 LoRA configs |

Explicit `expected_gpu_name_regex` / `min_vram_gib` / `min_free_disk_gib`
values override the profile; unknown profiles are refused. The A40 profile
stays strict (it never matches an RTX PRO 6000), and the RTX LoRA profile
never matches an A40.

**Base-only loading and single attach.** Evaluation, preflight reload and
merge load the pinned BF16 base through `load_base_model_and_tokenizer`, which
can never create a fresh adapter, and then attach the saved adapter exactly
once (`attach_lora_adapter` refuses a model that already carries one). The
same rule applies to base-model evaluation: a LoRA config evaluated without a
checkpoint still loads the base without adapters.

**Adapter restore.** Restoring adapter weights (gate checkpoint verification,
final-weight restore after verification) goes through PEFT's supported adapter
APIs: `peft.set_peft_model_state_dict` applies the saved tensors, then
`peft.get_peft_model_state_dict` reads the adapter back using the same
`adapter_name`, and the normalized adapter tensors are compared (keys, shapes,
dtypes, values). PEFT's key-prefix spellings are normalized on both sides, and
setter "missing" keys that are base-model weights are recorded as diagnostics,
never rejected — only genuine adapter incompatibilities (a missing adapter
parameter, an extra tensor, or a shape/dtype/value mismatch) fail. Raw
`load_state_dict` is used only for full artifacts.

**Merge (separate command).** `scripts/merge_adapter.py` is the only way to
produce a merged model; training never merges automatically. It verifies the
adapter artifact, loads the pinned base base-only, attaches the saved adapter
once, checks that base-plus-adapter and merged logits agree within
`--tolerance` (default `1e-3`, LoRA dropout disabled in eval mode), and writes
the merged model plus a `merge-metadata.json` record. The flow lives in
`src/stage1/merge.py` so it is executable in tests with injected loaders.

```bash
python3 stage-1/scripts/merge_adapter.py \
  --config stage-1/configs/minicpm5-2b-lora.yaml \
  --export /shared/$USER/tikz-production/export \
  --prepared /shared/$USER/tikz-stage1/prepared \
  --run-dir /shared/$USER/tikz-stage1/runs/minicpm5-2b-lora \
  --adapter /shared/$USER/tikz-stage1/runs/minicpm5-2b-lora/final/smoke-1000 \
  --out /shared/$USER/tikz-stage1/merged/minicpm5-2b-smoke
```

**Effective batch.** `per_device_train_batch_size * gradient_accumulation_steps`
must equal 16 for both methods (one A40, no multi-GPU); the config parser
enforces it.

### Sequence packing is not supported

Stage 1 rejects `training.packing: true` at configuration parse time with an
explicit error: resetting `position_ids` under a normal all-ones attention
mask does not stop attention across packed examples, and a backend-specific
varlen implementation has not been validated. There is no cu_seqlens path and
no flash-attention claim in this codebase; batches are right-padded.

### Run directory layout

```
RUN/
  run.json                       immutable run identity (written once)
  resolved-config.json           full resolved config, including paths
  environment.json               dependency snapshot + model load report
  preflight.json                 preflight evidence
  gates/<gate>.json              gate evidence (criteria, metrics path, checks)
  metrics/<gate>.json            full training metrics and log history
  logs/<gate>.jsonl              structured training log (one record per log event)
  checkpoints/<gate>/checkpoint-<step>/   resumable checkpoints + stage1-checkpoint.json
  final/<gate>/                  final model + tokenizer + stage1-final.json
  evaluations/<name>/            metrics.json, rows.jsonl, tex/, summary.md
```

## 6. Evaluation

Evaluates the pinned base checkpoint or a Stage 1 checkpoint/final artifact on
a held-out split (never on training data), with optional TeX compilation:

```bash
# base model, 50 eligible test examples
/shared/$USER/envs/minicpm5-2b/bin/python stage-1/scripts/evaluate.py \
  --config stage-1/configs/minicpm5-2b-full.yaml \
  --export /shared/$USER/tikz-production/export \
  --prepared /shared/$USER/tikz-stage1/prepared \
  --run-dir /shared/$USER/tikz-stage1/runs/minicpm5-2b \
  --name base-test-50 --base --max-examples 50

# final artifact from the smoke gate, whole test split
/shared/$USER/envs/minicpm5-2b/bin/python stage-1/scripts/evaluate.py \
  ... --name sft-full-test \
  --checkpoint /shared/$USER/tikz-stage1/runs/minicpm5-2b/final/smoke-1000 \
  --expect-gate smoke-1000 --all
```

Before loading `--checkpoint`, Stage 1 verifies the artifact metadata against
the current run identity, model id, exact revision, adapter and (optionally)
`--expect-gate`. Missing metadata, foreign identity, wrong model/revision,
wrong adapter, wrong gate and incomplete checkpoints are all refused. The
verified artifact identity is recorded in the evaluation metrics under
`artifact`. LoRA artifacts are loaded base-only first (no fresh adapter) and
the saved adapter is attached exactly once. Quarantined rows are excluded from
the evaluated split and from memorization sampling.

Metrics: generation completion/truncation/empty counts, TikZ extraction,
compilation success with timeout and error categories, output length,
inference latency and tokens/second, duplicate outputs, exact and
near-memorization against sampled training targets, and rendered-image
similarity against `png_image` when Pillow and `pdftoppm` are available
(otherwise the metric explicitly reports `not_available`).

TeX compilation is always safe: `-no-shell-escape`, `openin_any=p` /
`openout_any=p`, a denylist of file/IO primitives (`\input`, `\write`,
`\immediate`, …), a fresh isolated directory per row, a strict timeout that
kills the process group, and the full log captured. Text outside an extracted
`tikzpicture` is never part of the compiled document.

## 7. Compare runs and declare the experiment

```bash
python3 stage-1/scripts/compare_models.py \
  --metrics /shared/$USER/tikz-stage1/runs/minicpm5-2b/metrics/smoke-1000.json \
  --metrics /shared/$USER/tikz-stage1/runs/minicpm5-2b/evaluations/base-test-50 \
  --metrics /shared/$USER/tikz-stage1/runs/minicpm5-2b/evaluations/sft-full-test \
  --labels minicpm-smoke base trained \
  --out /shared/$USER/tikz-stage1/comparison
```

Writes `comparison.json` and a readable `comparison.md`. The table shows the
training method per candidate. The experiment is only declared READY when at
least one base-model evaluation and one trained-artifact evaluation exist for
the **same data identity and split**; otherwise the command exits `1` with the
reasons (`--allow-incomplete` exits 0 for intermediate inspection).

Comparison candidates must also be comparable: same training method (base
evaluations are compatible with either single method, but full and LoRA
candidates are never mixed), same dataset identity, same sequence limit and
the same evaluation set (recorded as `evaluation_set_sha256`). Violations exit
`1` with the specific disagreement unless `--allow-non-comparable` is passed,
and the report always states `Comparability: OK / NOT COMPARABLE`. Compile
improvement is not required until a base-model baseline exists, and missing
fields are shown as `-`, never silently treated as zero.

## 8. Slurm

The generator prints a script; it never writes a file and never calls
`sbatch`:

```bash
python3 stage-1/scripts/make_slurm_script.py \
  --config stage-1/configs/qwen3.5-4b-full.yaml \
  --export /shared/$USER/tikz-production/export \
  --prepared /shared/$USER/tikz-stage1/prepared \
  --run-dir /shared/$USER/tikz-stage1/runs/qwen3.5-4b \
  --gate smoke-1000 \
  --python /shared/$USER/envs/qwen3.5-4b/bin/python \
  --wall-time 12:00:00 \
  > tikz-stage1-qwen-smoke.sbatch

less tikz-stage1-qwen-smoke.sbatch
sbatch tikz-stage1-qwen-smoke.sbatch
```

The generated script requests one GPU (`--gres=gpu:rtxpro6k:1`), runs the
preflight and then exactly one gate. Add `--local-files-only` to export
`HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` (required on offline compute
nodes). Resubmitting the same script resumes the same gate.

## 9. Schemas

Eight JSON Schemas under `schemas/` cover the artifacts that are read by other
tools or by the gate chain:

| Schema | Artifact | Consumer |
| --- | --- | --- |
| `run.schema.json` | `run.json` | every command that verifies the run identity |
| `split-manifest.schema.json` | `split-manifest.json` | training, preflight, evaluation |
| `dataset-report.schema.json` | `dataset-report.json` | the data gate evidence |
| `preflight.schema.json` | `preflight.json` | gate prerequisites |
| `gate-evidence.schema.json` | `gates/<gate>.json` | gate chaining |
| `training-metrics.schema.json` | `metrics/<gate>.json` | `compare_models.py`, reports |
| `eval.schema.json` | `evaluations/<name>/metrics.json` | `compare_models.py`, reports |
| `artifact.schema.json` | `stage1-checkpoint.json` / `stage1-final.json` | resume and evaluation artifact verification |

The comparison report has no schema: it is a derived, human-oriented document
whose structure is asserted directly by the tests. A test
(`test_schema.test_no_orphan_schemas`) fails if a schema file is added or
removed without updating the list.

## 10. Identity and resume rules

The run identity covers: the cleaning export's `dataset_logical_sha256`, the
split-manifest hash, the data-identity hash, the **dataset-report hash**
(which anchors the model/token/quarantine artifacts), model id and exact
revision, tokenizer file hashes, vocab and pad token, chat-template hash, the
resolved configuration (paths excluded), the dependency version snapshot plus
lock hash, the Stage 1 source hash, the repository commit, and the seed.
`run.json` is written once; any later mismatch raises an identity diff and
refuses to continue. Checkpoint and final-artifact metadata bind every artifact
to the same identity plus gate, model id, revision and adapter, and record the
committed supervised-token count for resume accounting.

Resume verification follows the production schedule exactly. Production builds
its scheduler with `transformers.get_scheduler("cosine", optimizer,
num_warmup_steps=..., num_training_steps=...)`, which is a `LambdaLR` (not a
`CosineAnnealingLR`). Verification therefore:

1. validates the saved state is a lambda-based schedule and that its saved
   `num_warmup_steps`/`num_training_steps` match this run's numbers
   (total = `trainer.state.max_steps`, warmup = `ceil(total * warmup_ratio)`);
2. compares the configured initial learning rate against the saved scheduler
   `base_lrs` **before** loading the optimizer state, because loading that
   state restores its own param-group learning rates and would mask a
   configuration mismatch;
3. reconstructs the scheduler with those numbers, probes the freshly built
   lambda values at start/warmup/midpoint/end against the saved lambdas, loads
   the saved state, and verifies `last_epoch`, `_step_count` and `base_lrs`.

A `CosineAnnealingLR` state, mismatched schedule parameters or an unverifiable
schedule are refused rather than guessed at. The full round trip against real
Transformers schedulers is a dependency-gated test
(`tests/test_checkpointing.py::ProductionSchedulerIntegrationTests`); where
Transformers is not installed that test is skipped and the resume gate stays
explicitly unverified.

## 11. Verification

```bash
# Stage 1 tests (CPU, no network, no GPU; optional deps skip cleanly)
python3 -m unittest discover -s stage-1/tests -t stage-1 -p 'test_*.py' -v

# Existing cleaning tests (unchanged)
python3 -m unittest discover -s cleaning -p 'test_*.py'

# Configuration validation only
python3 stage-1/scripts/prepare_dataset.py --dry-run \
  --config stage-1/configs/qwen3.5-4b-full.yaml \
  --config stage-1/configs/minicpm5-2b-full.yaml
```

The marked integration tests run from `stage-1/`:

```bash
cd stage-1

# GPU-marked: runs only on an RTX PRO 6000 host
python3 -m unittest tests.test_gpu_integration -v

# Network-marked: downloads the pinned tokenizer files only
STAGE1_ALLOW_NETWORK_TESTS=1 python3 -m unittest tests.test_network_integration -v
```

`tests/test_cleaning_consistency.py` proves the checksum implementation matches
`cleaning/build_dataset.py` byte for byte. `tests/test_compile_tikz.py`
exercises real `pdflatex` when it is installed (including the shell-escape
refusal). `tests/test_checkpointing.py::ProductionSchedulerIntegrationTests`
runs the resume round trip against the real `transformers.get_scheduler`
cosine schedule, and `tests/test_lora_integration.py` runs the real PEFT
adapter save/attach/merge round trips and a fresh-Trainer resume from a native
checkpoint, when torch, Transformers, PEFT and datasets are installed; without
them those tests are skipped and the corresponding LoRA/resume gates are
explicitly unverified. Tests that need `pyarrow`, `PyYAML`, `Pillow`, `torch`,
`transformers`, `peft`, `datasets`, a GPU or network access skip cleanly with
the reason printed by `unittest -v`; the skip count therefore depends on the
environment (see the run report).

## 12. Known limitations and compatibility risks

- **GPU compatibility is unproven until the preflight passes.** In particular
  `Qwen3.5-4B` is a hybrid linear/full-attention multimodal checkpoint
  (`Qwen3_5ForConditionalGeneration`): text-only batches must load through
  Unsloth's vision loader and train with full finetuning. If Unsloth cannot
  full-finetune it, the preflight fails loudly and the model cannot be used
  until the environment or loader changes (a reviewed config change).
- **MiniCPM5-2B uses its official Transformers 4.57.3 fallback.** The primary
  upstream recommendation is 5.6.x, but pinned Unsloth 2026.9.11 caps
  Transformers at 5.5.0. MiniCPM's separate 4.57.3 environment stays inside
  that cap and avoids the Transformers-v5 adapter conversion path. Qwen keeps
  its independent 5.5.0 environment.
- **No sequence packing**: long examples are padded (slower, but correct).
  Raising `data.max_seq_len` is the supported way to reduce quarantine.
- TeX compilation requires a TeX installation with `standalone` and `tikz`
  (`texlive-latex-extra` or equivalent); the evaluation reports
  `unavailable` when `pdflatex` is missing rather than failing the run.
- Rendered-image similarity requires Pillow and `pdftoppm`; without them the
  metric is `not_available`, and this is recorded, not hidden.
