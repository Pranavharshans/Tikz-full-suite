# Production dataset builder (`cleaning/build_dataset.py`)

Production runbook for the TikZ cleaning stage. `cleaning/build_dataset.py`
builds the text-to-TikZ instruction dataset from a frozen slice of
`nllg/DaTikZ-V4` captioned by `nvidia/Qwen3.8-27B-NVFP4`. Every stage is
explicit, identity-checked and resumable.

`cleaning/benchmark.py` and `cleaning/results/` are preserved untouched: the
benchmark tool and its recorded measurements are the evidence for the baseline
settings below. `build_dataset.py` is a separate production workflow that
imports only two audited helpers from `benchmark.py` (`bench.digest` and
`bench.vllm_runtime_options`) and never runs the benchmark matrix. All
production file writes use the builder's own durable writer (unique temp name,
fsync, atomic rename, directory fsync) rather than `bench.dump`.

Contents:

1. [Architecture](#1-architecture)
2. [Work directory layout](#2-work-directory-layout)
3. [Validated production baseline](#3-validated-production-baseline)
4. [Command-line reference](#4-command-line-reference)
5. [Preparation](#5-preparation)
6. [Slurm generation](#6-slurm-generation)
7. [Run, resume and the state machine](#7-run-resume-and-the-state-machine)
8. [Retry policy](#8-retry-policy)
9. [Export outputs](#9-export-outputs)
10. [Validation rules](#10-validation-rules)
11. [Audit and validate](#11-audit-and-validate)
12. [Recovery procedures](#12-recovery-procedures)
13. [Changing the model or the prompt](#13-changing-the-model-or-the-prompt)
14. [Testing](#14-testing)
15. [Production runbook on Alex](#15-production-runbook-on-alex)
16. [Boundaries](#16-boundaries)

---

## 1. Architecture

### 1.1 Stages

All stages are subcommands of one file. Stage guards and exit codes are
summarized after the table.

| Stage | Purpose | Writes | Slurm guard |
| --- | --- | --- | --- |
| `plan` | Print resolved configuration and provisional run identity | nothing | none |
| `prepare` | Freeze exactly the first 100,000 source rows into a manifest | `manifest.jsonl`, `manifest.meta.json`, `images/` | yes (no GPU count) |
| `run` | Two-replica inference controller | `run.json`, `ledger.sqlite3`, `runtime/`, `ledger-backups/` | yes, exactly 2 visible GPUs, all named `RTX PRO 6000` |
| `status` | Human-readable progress report (read-only) | nothing | none |
| `validate` | Re-validate every accepted instruction | `WORK/validation-report.json` | none |
| `audit` | Verify structural invariants; exit 1 on violation | `WORK/audit-report.json` | none |
| `export` | Deterministic atomic Parquet shards for complete rows | `WORK/export/` | yes (no GPU count) |
| `checkpoint` | Copy the ledger to a timestamped backup | `ledger-backups/ledger-*.sqlite3` | none |
| `slurm-script` | Print an `sbatch` script; never submits | nothing | none |
| `worker` | Hidden internal worker entry point (inside the container) | `runtime/results/`, `runtime/heartbeats/` | invoked by the controller |

Guards:

- `prepare`, `run`, `export` call `require_slurm`. Without `SLURM_JOB_ID` they
  refuse with `Refusing to run outside a Slurm allocation`. `run` additionally
  requires exactly two entries in `CUDA_VISIBLE_DEVICES`.
- A hidden `--allow-non-slurm` flag exists for synthetic fixtures in tests
  only. It is suppressed from `--help` and is not a production option.
- Exit codes: `0` success; `1` audit violations, validate failures, Slurm/GPU/
  disk-space guards (as `SystemExit`); `2` handled `ConfigError`,
  `IdentityMismatch`, `PromptError`, `ManifestError`, `LedgerError`,
  `PipelineIdentityError`, `ExportError`, `WorkerError` (printed as
  `error: ...`); `3` `run` finished with rows still needing work
  (pending/retryable/running), also the worker's engine-failure exit code.

### 1.2 Controller / worker split

`run` is a wave-based controller that runs on the host:

1. Verifies the frozen manifest and writes or verifies `run.json`.
2. Hashes the vLLM SIF (`--vllm-sif`) and probes the two GPUs.
3. Seeds the ledger from the manifest, ingests leftover result files, reclaims
   stale claims, and optionally reprocesses rejected rows.
4. Repeatedly claims currently-eligible rows, splits them deterministically
   between two workers (length-balanced by TikZ character count), writes one
   task file per worker, and spawns one container per worker:

   ```
   apptainer exec --nv --bind WORK --bind CLEANING_DIR --bind MODEL_PATH \
     vllm.sif python3 build_dataset.py worker --job runtime/jobs/<job>.json
   ```

5. Ingests atomic worker result files into the ledger, monitors heartbeats,
   restarts or releases dead/stalled workers, and repeats until no eligible
   rows remain or the runtime budget / stop signal ends the run.

Workers never write the ledger. They read a task file, verify image and TikZ
checksums, build the prompt, generate with vLLM offline (greedy,
`temperature=0`), and write one atomic JSON result file per microbatch plus a
heartbeat. The worker container gets `CUDA_VISIBLE_DEVICES` for one GPU,
`APPTAINERENV_NCCL_P2P_DISABLE` from `--nccl-p2p`, and
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`; the model snapshot path is
bind-mounted, so workers need no network.

Before any field of a result file is used, `validate_result_file` checks its
structure: the document must be an object with `run_id`, `worker`,
`worker_index` and a `results` list, and every result must be an object with
the required keys and types (no bool where an integer is expected, no
non-positive `max_tokens`, no duplicate row ids). A file that is valid JSON
but fails this schema - or that conflicts with the ledger - is quarantined as
evidence and its rows are reclaimed; unexpected programming defects are never
classified this way and still raise normally.

Every batch is validated before its results are trusted: the output count must
match, and each output must echo the request prompt. An engine that does not
echo prompts is refused by default (`--allow-missing-prompt-echo` accepts
engine ordering explicitly, for engines that guarantee it). A batch whose
association is doubtful is recorded as a transient failure for all of its rows
rather than partially committed. Prompts whose length plus the reserved output
ceiling exceeds the context are rejected as `input_too_long` before generation.

Engine construction (in the worker) uses:
`max_model_len=context`, `max_num_seqs=per_replica_concurrency`,
`max_num_batched_tokens=batch_token_budget`, `enable_chunked_prefill=True`,
`enable_prefix_caching=False`, `mm_processor_cache_gb=0`,
`gpu_memory_utilization`, plus `bench.vllm_runtime_options()`
(`disable_custom_all_reduce=True`, `disable_log_stats=False`,
`enforce_eager=False`) and, when `mtp > 0`,
`speculative_config={"method": "mtp", "num_speculative_tokens": mtp}`.
The resolved options and library versions are written to
`runtime/logs/worker-N-engine-options.json` and
`runtime/logs/worker-N-versions.json`.

### 1.3 Single-writer SQLite ledger

`ledger.sqlite3` is the durable state. Only the controller writes it
(`Ledger` opens with `journal_mode=DELETE`, `synchronous=FULL`,
`busy_timeout=30000`, `foreign_keys=ON`, and uses `BEGIN IMMEDIATE`
transactions). Tables:

- `meta`: run identity (`schema_version`, `run_id`, `identity_sha256`,
  `manifest_sha256`, `dataset_revision`, `model_revision`, `prompt_sha256`,
  `tool_version`, `created_at`).
- `rows`: one row per frozen manifest row, `row_id` primary key,
  `source_row_index` unique, state in
  `pending|running|retryable|complete|rejected`, attempt counter, claim
  bookkeeping (`claimed_at`, `worker`, `not_before`), last error, rejection
  reason, and for completed rows the accepted `instruction`, `finish_reason`,
  token counts, model revision, prompt hash and config hash.
- `attempts`: full history (`attempt_no`, `attempt_kind`
  `normal|truncation_retry|recovered`, state `running|complete|retryable|
  rejected|lost`, worker, `max_tokens`, timestamps, finish reason, error
  category/detail, token counts, instruction, hashes).
- `consumed_files`: content SHA256 of every result file already committed, so
  re-applying a file after a crash between commit and file removal is a no-op
  rather than a duplicated attempt.

The ledger independently pins the identity: `Ledger.initialize` inserts
missing metadata but refuses conflicting values, so reusing a work directory
with a different identity fails even if `run.json` were removed.

### 1.4 Atomic manifest and export lifecycles

Manifest freeze: rows are staged under `WORK/.staging-<pid>/`, the manifest
JSONL is flushed and `fsync`ed, the staged manifest and every staged image are
read back and hash-verified, the staging directories are `fsync`ed, the images
directory and manifest are moved into place with `os.replace`, and
`manifest.meta.json` is written last with a durable fsync+rename writer.
`manifest.meta.json` is the only marker that makes a manifest authoritative;
an interrupted freeze leaves no top-level manifest, and `prepare` removes
partial artifacts (`manifest.jsonl`, `images/`, `.staging-*`) before
re-freezing. A torn (non-JSON) `manifest.meta.json` is treated as an
interrupted freeze and re-frozen automatically. `verify_manifest` re-checks
schema, contiguous indices, unique stable ids, TikZ checksums, image presence
and non-emptiness (and hashes when `quick=False`), rejection reasons, row
count, and the manifest content hash against the meta file.

Export: each shard is written to `shard-NNNNN.parquet.tmp`, `fsync`ed,
read back for row count and first-id equality, then renamed atomically; the
parent directory is `fsync`ed. `export.meta.json` is written last and marks a
completed export. Shards from a previous export that this run no longer
produces are deleted. `.tmp` files are never treated as shards (audit ignores
them).

---

## 2. Work directory layout

`--work` defaults to `./tikz-production`; use an absolute shared path on Alex.

```
WORK/
  manifest.jsonl                 frozen 100,000-row manifest (one JSON object per line)
  manifest.meta.json             freeze metadata + hash; authoritative marker
  images/000000.png ...          images for valid rows (index-named)
  run.json                       run identity + provenance (written by the first run)
  ledger.sqlite3                 durable state (single writer: controller)
  ledger-backups/
    ledger-YYYYmmdd-HHMMSS.sqlite3
  runtime/
    tasks/                       worker-N-genW.jsonl and worker-N-genW-rR.jsonl (restarts)
    results/worker-N/batch-W-BBBBB.json   atomic worker outputs (unconsumed)
    consumed/worker-N/...        result files already committed to the ledger
    quarantine/                  unprocessable result files kept as evidence
    jobs/worker-N-genW-rR.json   worker job descriptions
    heartbeats/worker-N.json     progress heartbeats
    logs/worker-N.log            worker stdout/stderr (appended across restarts)
    logs/worker-N-engine-options.json, worker-N-versions.json
    logs/container-sha256.json   cached SIF hash (path|size|mtime key)
    stop                         cooperative stop marker
  export/
    shards/shard-00000.parquet ...
    rejected.parquet
    attempts.parquet
    run-metadata.json
    stats.json
    checksums.json
    validation-report.json
    dataset-card.md
    export.meta.json             written last; marks a completed export
  audit-report.json              written by audit
  validation-report.json         written by validate
  hf/                            default model cache (prepare --model-cache-dir default)
  .staging-*                     transient during prepare only
```

`runtime/logs/worker-N.log` is opened in append mode, so a restart adds to the
existing log instead of discarding the crashed attempt's evidence. Controller
output goes to the Slurm job output (`slurm-tikz-prod-%j.out` by default).

---

## 3. Validated production baseline

The baseline is the measured winner from `cleaning/results/`
(`rtxpro6000-decoding-comparison-2026-09-23.csv`): NVFP4, greedy, two total
TP1 replicas at aggregate concurrency 64, 16K batch budget —
**14,756.96 successful samples/hour** on two NVIDIA RTX PRO 6000 Blackwell
96GB GPUs. The module records `MEASURED_SAMPLES_PER_HOUR = 14_756.96`; `plan`
reports `estimated_generation_hours_at_baseline` (about 6.78 h for 100,000
rows) as a reference, not a promise. The four-replica / two-replicas-per-GPU
topology measured slower (11,685.80 samples/hour) and is explicitly rejected:
`InferenceConfig.validate()` raises if `replicas_per_gpu != 1` or
`(tensor_parallel, replicas) != (1, 2)`.

### 3.1 Fixed (no CLI switch; validated by `InferenceConfig.validate`)

| Setting | Value |
| --- | --- |
| engine | `vllm-offline` |
| tensor parallel | `1` |
| replicas | `2` (one per GPU) |
| replicas per GPU | `1` |
| thinking | disabled |
| decoding | greedy, `temperature=0.0` |
| prefix caching | disabled (`enable_prefix_caching=False`) |
| custom all-reduce | disabled (`disable_custom_all_reduce=True`) |
| dataset / split / rows | `nllg/DaTikZ-V4`, `train`, rows `0..99_999` |

### 3.2 Configurable, but part of the run identity

Changing any of these after `run.json` exists raises `IdentityMismatch` and
requires a fresh `--work` directory.

| CLI flag | Default | Constraint |
| --- | --- | --- |
| `--concurrency` | `64` | aggregate in-flight requests; `>= replicas`; per-replica = `64 // 2 = 32` each |
| `--mtp` | `1` | `0..3`; validated baseline is `1` |
| `--batch-token-budget` | `16384` | positive |
| `--context` | `32768` | `>= 4096` |
| `--max-output-tokens` | `256` | positive |
| `--truncation-retry-tokens` | `384` | must exceed `--max-output-tokens` |
| `--gpu-memory-utilization` | `0.90` | strictly between 0 and 1 |
| `--nccl-p2p` | `disabled` | `disabled` or `auto`; FAX Alex uses `disabled` |
| `--prompt-version` | `caption-v1` | must exist in `cleaning/prompts/registry.json` |
| `--min-instruction-chars` | `20` | validation policy |
| `--max-instruction-chars` | `2000` | validation policy |
| `--max-instruction-words` | `120` | validation policy |
| `--dataset-revision`, `--model-revision` | resolved by `prepare` | pinned 40-char commit SHA |
| `--model`, `--dataset` | `nvidia/Qwen3.8-27B-NVFP4`, `nllg/DaTikZ-V4` | must match the frozen manifest |
| `--vllm-sif` | required by `run` | SIF bytes are hashed into the identity |
| runner/helper code | implicit | editing `build_dataset.py` or `benchmark.py` changes the identity |

Provenance-only fields excluded from the identity hash: `git_commit` (detected
automatically) and `model_path` (the local snapshot path). They are recorded in
`run.json` but do not change the run id.

Chunk, retry and monitoring flags (`--start-index`, `--end-index`,
`--max-rows-this-run`, `--max-runtime-minutes`, `--worker-restarts`,
`--worker-timeout`, `--poll-seconds`, `--shutdown-grace-seconds`,
`--retry-wait-seconds`, `--warmup-samples`, `--reprocess-rejected`,
`--max-transient-attempts`, `--allow-missing-prompt-echo`, backoff flags) are
**not** identity-locked and can differ between resubmissions.

---

## 4. Command-line reference

Every example assumes the repository root as the current directory and a
shared work directory. On Alex, run `prepare`, `run` and `export` inside a
Slurm allocation (or simply submit the generated script).

### plan

```bash
python3 cleaning/build_dataset.py plan \
  --work /shared/$USER/tikz-production
```

Prints JSON: tool version, resolved work path, provisional run id and identity,
stage list, source slice, model, and the measured baseline. No Slurm needed; no
files are written. Add `--dataset-revision` / `--model-revision` to preview a
pinned identity, or `--concurrency 32 --mtp 2` to preview another
configuration.

### prepare

```bash
python3 cleaning/build_dataset.py prepare \
  --work /shared/$USER/tikz-production \
  --model nvidia/Qwen3.8-27B-NVFP4 \
  --dataset nllg/DaTikZ-V4 \
  --model-cache-dir /shared/$USER/tikz-production/hf
```

Runs inside the allocation (the generated script runs it in the container).
`--download-model` is the default; use `--no-download-model` to skip the
snapshot download (then `run` fails until a model path is recorded).

### run

```bash
python3 cleaning/build_dataset.py run \
  --work /shared/$USER/tikz-production \
  --vllm-sif /shared/$USER/containers/vllm.sif \
  --concurrency 64 --mtp 1 \
  --batch-token-budget 16384 --context 32768 \
  --max-output-tokens 256 --truncation-retry-tokens 384 \
  --gpu-memory-utilization 0.90 --nccl-p2p disabled \
  --prompt-version caption-v1 \
  --max-runtime-minutes 690
```

Run inside a Slurm allocation with two visible RTX PRO 6000 GPUs. Bounded
chunk example:

```bash
python3 cleaning/build_dataset.py run \
  --work /shared/$USER/tikz-production \
  --vllm-sif /shared/$USER/containers/vllm.sif \
  --max-rows-this-run 10000 --max-runtime-minutes 600 \
  --start-index 0 --end-index 99999
```

### status

```bash
python3 cleaning/build_dataset.py status --work /shared/$USER/tikz-production
python3 cleaning/build_dataset.py status --work /shared/$USER/tikz-production --json
```

Read-only. Works without Slurm and without the ledger (it warns instead).

### validate

```bash
python3 cleaning/build_dataset.py validate --work /shared/$USER/tikz-production
```

Re-validates every complete row and writes `WORK/validation-report.json`;
exits 1 when any accepted instruction fails the policy.

### audit

```bash
python3 cleaning/build_dataset.py audit --work /shared/$USER/tikz-production
python3 cleaning/build_dataset.py audit --work /shared/$USER/tikz-production --quick
python3 cleaning/build_dataset.py audit --work /shared/$USER/tikz-production \
  --export-dir /shared/$USER/tikz-production/export
```

Exits 1 on any structural violation and writes `WORK/audit-report.json`.
`--quick` skips image hashing and export file hashing.

### export

```bash
python3 cleaning/build_dataset.py export \
  --work /shared/$USER/tikz-production \
  --shard-size 1000
```

Runs in the allocation (the generated script runs it in the container, which
provides `pyarrow`). Re-running is safe and idempotent for unchanged input.

### checkpoint

```bash
python3 cleaning/build_dataset.py checkpoint --work /shared/$USER/tikz-production
```

### slurm-script

```bash
python3 cleaning/build_dataset.py slurm-script \
  --work /shared/$USER/tikz-production \
  --vllm-sif /shared/$USER/containers/vllm.sif \
  --wall-time 12:00:00 > tikz-production.sbatch
```

Prints the script to stdout; it never writes a file and never submits. The
historical spelling `--slurm-script` is accepted as an alias for the
`slurm-script` subcommand.

### worker (hidden)

```bash
apptainer exec --nv \
  --bind /shared/$USER/tikz-production \
  --bind /path/to/repo/cleaning \
  --bind /shared/$USER/tikz-production/hf/models--nvidia--Qwen3.8-27B-NVFP4/snapshots/<sha> \
  /shared/$USER/containers/vllm.sif \
  python3 /path/to/repo/cleaning/build_dataset.py worker \
  --job /shared/$USER/tikz-production/runtime/jobs/worker-0-gen1-r0.json
```

The controller builds this command; manual use is for debugging only.

---

## 5. Preparation

### 5.1 Revision pinning

`prepare` resolves the dataset and model revisions from the Hugging Face Hub
when `--dataset-revision` / `--model-revision` are omitted, then records the
resolved 40-character commit SHAs in `manifest.meta.json`. `run` requires
pinned revisions and the frozen manifest; if you pass a revision that does not
match the frozen manifest, `prepare`/`run` refuse with
`does not match the frozen manifest`.

### 5.2 Exact first 100,000 rows

The source is streamed (`datasets` streaming mode, `png_image` decoded as raw
bytes) and `islice`d to exactly `ROW_LIMIT = 100_000` rows starting at
`ROW_START = 0`. There is no sampling, no shuffling and no replacement:

- `source_row_index` runs contiguously `0..99_999`.
- If the stream ends early, the freeze is a hard failure and no manifest is
  committed.
- Invalid rows are recorded **in place** as `status: "rejected"` with a
  `rejection_reason`; the next valid row is not pulled forward. Reasons:
  `empty_tikz`, `tikz_too_long` (> `--max-tikz-chars`, default 200,000),
  `missing_image`, `image_too_large` (> `--max-image-bytes`, default
  20,000,000), `invalid_image` (not PNG or unreadable/corrupt).
- `manifest.meta.json` records `rows_frozen`, `valid_rows`, `rejected_rows`,
  `rejection_counts`, `freeze_limits` and `image_validation`
  (`"pillow"` when Pillow was importable, else `"signature"`).

### 5.3 Stable row id

```
row_id = sha256("\x1f".join(
    "rid-v1", dataset_id, revision, split, str(source_index),
    tikz_sha256, image_sha256))[:32]
```

It depends only on immutable source information, so the same source row always
gets the same id even across re-freezes. The manifest stores both checksums
(TikZ as SHA256 of the text, image as SHA256 of the raw bytes; rejected rows
still record the checksum of whatever bytes were received).

### 5.4 Image validation

`extract_image_bytes` accepts the shapes the Hub can return with
`decode=False`: `bytes`/`bytearray`, base64 text, a filesystem path, or a dict
with `bytes`/`path`. `inspect_image` then checks, in order: non-empty, size
limit, PNG signature (`\x89PNG\r\n\x1a\n`), and — when Pillow is available —
`Image.verify()` plus a non-degenerate size.

### 5.5 Storage estimation and disk guard

Before freezing, `prepare` streams a bounded sample (`--estimate-rows`,
default 200), computes average image and TikZ sizes, and projects

```
projected_bytes = row_limit * (average_image + average_tikz + 256)
required_bytes  = projected_bytes * 1.5 + 5 GiB reserve
```

and refuses with `Insufficient free space` when the work filesystem has less.

### 5.6 Model snapshot download

`prepare` downloads the pinned snapshot with
`snapshot_download(model, revision=..., cache_dir=...)`. The cache defaults to
`WORK/hf` and can be overridden with `--model-cache-dir`. The resolved local
path is stored as `model_path` in `manifest.meta.json`. `run` refuses to start
when the manifest has no model path. Workers run offline against that path.

### 5.7 Reusing an existing frozen manifest

If `WORK/manifest.meta.json` exists, `prepare`:

1. runs `verify_manifest(quick=True)` (schema, ids, checksums, image presence,
   manifest hash);
2. refuses when the requested dataset/model/revisions disagree with the frozen
   manifest;
3. downloads the model when `--download-model` is set and the recorded
   `model_path` is missing or no longer exists on disk (updating the meta
   file);
4. prints `Reusing frozen manifest: ...` and does not re-read the source.

A second `prepare` against a complete manifest is a no-op.

### 5.8 Reusing a manifest in a fresh work directory

The manifest is dataset-only; copy the three frozen artifacts and let
`prepare` verify them in the new directory:

```bash
mkdir -p /shared/$USER/tikz-production-v2
cp /shared/$USER/tikz-production/manifest.jsonl \
   /shared/$USER/tikz-production/manifest.meta.json \
   /shared/$USER/tikz-production-v2/
cp -r /shared/$USER/tikz-production/images \
      /shared/$USER/tikz-production-v2/images

python3 cleaning/build_dataset.py prepare \
  --work /shared/$USER/tikz-production-v2 \
  --model nvidia/Qwen3.8-27B-NVFP4 \
  --dataset nllg/DaTikZ-V4 \
  --model-cache-dir /shared/$USER/tikz-production-v2/hf
```

`prepare` verifies every image file and the manifest hash, then reuses the
freeze. Two caveats:

- The copied `model_path` must still exist. When it is missing, `prepare
  --download-model` re-resolves the pinned snapshot and updates the meta file.
  `model_path` is provenance-only and not part of the identity, so an operator
  may also update it in `manifest.meta.json` (or delete the key) and re-run
  `prepare --download-model`; never change dataset/model ids or revisions by
  hand.
- A copied manifest is only reusable with the same model id and revision it
  was frozen for. Changing the model requires a fresh freeze (Section 13).

---

## 6. Slurm generation

`slurm-script` prints a complete `sbatch` script and never submits anything.
Redirect it to a file, inspect it, then submit it yourself:

```bash
python3 cleaning/build_dataset.py slurm-script \
  --work /shared/$USER/tikz-production \
  --vllm-sif /shared/$USER/containers/vllm.sif \
  --wall-time 12:00:00 \
  > tikz-production.sbatch

sbatch tikz-production.sbatch
```

Requested allocation (defaults): one node, one task, `--partition=rtxpro6k`,
`--gres=gpu:rtxpro6k:2`, 32 CPUs, the requested wall time, `--export=NONE`,
and `--output=slurm-tikz-prod-%j.out`. Job name defaults to `tikz-prod`.

### 6.1 Example generated script

Paths and arguments are resolved and shell-quoted by the generator; the script
below shows the exact shape for the default configuration.

```bash
#!/bin/bash -l
#SBATCH --job-name=tikz-prod
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-tikz-prod-%j.out
# Set HF_TOKEN in this environment if the model repository is gated.
# Never commit tokens; this generator never writes one.
set -euo pipefail
unset SLURM_EXPORT_ENV
command -v apptainer >/dev/null || module load apptainer
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=$http_proxy
export no_proxy=localhost,127.0.0.1,::1
export NO_PROXY=$no_proxy
WORK=/shared/alice/tikz-production
VLLM=/shared/alice/containers/vllm.sif
SCRIPT=/repo/cleaning/build_dataset.py
mkdir -p "$WORK"
echo "== status before =="
python3 /repo/cleaning/build_dataset.py status --work "$WORK" || true
if [ ! -f "$WORK/manifest.meta.json" ]; then
  echo "== prepare (downloads stay inside the allocation) =="
  apptainer exec --nv --bind /shared/alice/tikz-production --bind /repo/cleaning \
    /shared/alice/containers/vllm.sif python3 /repo/cleaning/build_dataset.py \
    prepare --work /shared/alice/tikz-production --model nvidia/Qwen3.8-27B-NVFP4 \
    --dataset nllg/DaTikZ-V4 --model-cache-dir /shared/alice/tikz-production/hf
fi
echo "== run =="
run_rc=0
python3 /repo/cleaning/build_dataset.py run --work /shared/alice/tikz-production \
  --model nvidia/Qwen3.8-27B-NVFP4 --vllm-sif /shared/alice/containers/vllm.sif \
  --concurrency 64 --mtp 1 --batch-token-budget 16384 --context 32768 \
  --max-output-tokens 256 --truncation-retry-tokens 384 \
  --gpu-memory-utilization 0.9 --nccl-p2p disabled --prompt-version caption-v1 \
  --max-transient-attempts 3 --retry-backoff-base-seconds 30.0 \
  --retry-backoff-cap-seconds 600.0 --max-runtime-minutes 690 \
  --start-index 0 --end-index 99999 || run_rc=$?
echo "== status after =="
python3 /repo/cleaning/build_dataset.py status --work "$WORK" || true
check_rc=0
if [ "$run_rc" -le 3 ]; then
  echo "== export =="
  apptainer exec --nv --bind /shared/alice/tikz-production --bind /repo/cleaning \
    /shared/alice/containers/vllm.sif python3 /repo/cleaning/build_dataset.py \
    export --work /shared/alice/tikz-production --shard-size 1000 \
    --model nvidia/Qwen3.8-27B-NVFP4 --prompt-version caption-v1 || check_rc=$?
  echo "== audit =="
  apptainer exec --nv --bind /shared/alice/tikz-production --bind /repo/cleaning \
    /shared/alice/containers/vllm.sif python3 /repo/cleaning/build_dataset.py \
    audit --work /shared/alice/tikz-production || check_rc=$?
fi
if [ "$check_rc" -ne 0 ]; then
  echo "Export/audit failed with rc=$check_rc; inspect the work directory"
  exit "$check_rc"
fi
if [ "$run_rc" -eq 0 ]; then
  echo "All rows are terminal. Resubmit only after changing the configuration."
elif [ "$run_rc" -eq 3 ]; then
  echo "Rows remain (paused or failed). Resubmit this script to resume."
else
  echo "Run failed with rc=$run_rc; inspect the work directory and logs."
fi
exit "$run_rc"
```

Key behaviors:

- `prepare` runs only when `manifest.meta.json` is missing, so resubmissions
  never re-download the dataset or model.
- `run` executes on the host (standard library only); `prepare`, `export` and
  `audit` execute inside the container (they need `datasets`/`huggingface_hub`
  and `pyarrow`).
- `status` runs on the host before and after the run.
- Export and audit run whenever the run exit code is `0` or `3`.
- The script exits with the run code (`0` all terminal, `3` rows remain) or
  with the export/audit code when those fail.

### 6.2 Wall-time formats and the derived runtime budget

`--wall-time` accepts Slurm's formats (whole minutes are the internal unit):

| Input | Meaning |
| --- | --- |
| `90` | 90 minutes |
| `5:00` | 5 minutes |
| `00:30:00` | 30 minutes |
| `12:00:00` | 12 hours |
| `1-00:00:00` | 1 day |

Minutes/seconds must be `<= 59`; hours must be `<= 23` when days are present;
the total must be positive. Invalid examples: `""`, `abc`, `12:99:00`, `0`,
`1-25:00:00`, `1:2:3:4`.

Unless `--max-runtime-minutes` is given explicitly, the generated `run`
command uses `max(10, wall_minutes - 30)` — a fixed 30-minute reserve inside
the allocation. `12:00:00` therefore produces `--max-runtime-minutes 690`.

### 6.3 Chunk controls and resumability

- `--max-rows-this-run N`: stop claiming after N rows claimed in this process.
- `--start-index A --end-index B`: only claim rows in that source-index window
  (bounds `0 <= A <= B <= 99_999`).
- Both are propagated to the generated script and are **not** identity-locked,
  so a later run (or resubmission) claims the remaining rows.

### 6.4 Container bind behavior

Generated-script container commands bind:

- `WORK` (absolute),
- the directory containing `build_dataset.py` (`CLEANING_DIR`),
- `--model-cache-dir` only when it is outside `WORK`,
- for an absolute `--prepare-python` / `--export-python`, the environment root
  (`Path(python).parent.parent`, e.g. `/envs/prep/bin/python` binds
  `/envs/prep`).

Worker containers (spawned by the controller) bind `WORK`, `CLEANING_DIR` and
the resolved model snapshot path. Binding the same absolute paths preserves
the absolute image paths stored in task files.

`--prepare-python` / `--export-python` select the Python executable used
inside the container. Defaults are `python3` (the image must provide the
dependencies; export/audit need `pyarrow`). Use an absolute path for a
separate environment, as in the `cleaning/README.md` example.

### 6.5 HF_TOKEN

The generator never writes a token. The script only contains:

```bash
# Set HF_TOKEN in this environment if the model repository is gated.
# Never commit tokens; this generator never writes one.
```

Because the script starts from a clean environment (`#SBATCH --export=NONE`),
provide `HF_TOKEN` deliberately — for example by exporting it in your local
copy of the script or through your site's secret mechanism — and never commit
it to the repository.

---

## 7. Run, resume and the state machine

### 7.1 Row states

| State | Meaning |
| --- | --- |
| `pending` | frozen and valid, never claimed (or moved back by `--reprocess-rejected`) |
| `running` | claimed by a worker; has an open attempt |
| `retryable` | failed transiently or released; claimable after `not_before` |
| `complete` | accepted instruction stored; terminal, never regenerated |
| `rejected` | terminal with an explicit reason; excluded from shards |

Every claim opens an attempt row (`normal`, or `truncation_retry` when the row
escalates to the 384-token ceiling). Releasing a running row closes its open
attempt with state `lost`. A result that arrives after a crash without an open
attempt is recorded as a `recovered` attempt. Completed rows are immutable: a
late identical result is a no-op, and a different instruction raises
`refusing to overwrite a completed caption`. Late results for rejected rows are
ignored.

### 7.2 Startup sequence on every run/resubmission

1. Takes the advisory controller lock (`runtime/controller.lock`) and refuses
   to start if another controller holds it.
2. Deletes a leftover `runtime/stop` marker.
3. Initializes/verifies the ledger identity.
4. Seeds the ledger from the manifest (idempotent): valid rows become
   `pending`, manifest-invalid rows become `rejected` with
   `last_error_category = input_invalid`.
5. Ingests any leftover worker result files, committing finished batches.
   Files already committed (tracked by content SHA256 in `consumed_files`) are
   moved without reprocessing, so a crash between commit and file removal is
   harmless. Unprocessable files are moved to `runtime/quarantine/` and their
   rows are reclaimed instead of blocking the resume.
6. Reclaims every row still marked `running` as `stale_claim` — this controller
   provably owns the work directory, so those claims belong to dead workers.
   No retry budget is consumed.
7. With `--reprocess-rejected`, moves rejected rows back to `pending`
   (preserving attempt history), except manifest-invalid rows whose
   `last_error_category` is `input_invalid`, which always stay rejected.
8. Claims and runs waves until no eligible rows remain.

### 7.3 What a resubmission does and does not regenerate

Does:

- commits finished-but-unconsumed worker results;
- reclaims orphaned running rows immediately (no age threshold);
- retries `retryable` rows;
- writes a fresh ledger checkpoint at the end;
- re-runs export and audit (rebuilding shards atomically).

Does not:

- regenerate `complete` rows (they are never claimable again);
- re-freeze or re-verify the source dataset (manifest is verified, not
  re-downloaded);
- re-download the model (the snapshot path is reused; workers are offline);
- rewrite `run.json` (it is verified against the current configuration);
- reset `rejected` rows (unless `--reprocess-rejected` is passed).

### 7.4 Stale claims, stalls and worker restarts

- At startup, every `running` row is released as `stale_claim` (no budget)
  because the controller lock proves no other controller is active.
- During a wave, a worker with no heartbeat and no activity for
  `--worker-timeout` (default 1800 s) is terminated; if it does not exit
  within the stall-kill grace (60 s by default) it is killed. A worker that is
  restarted starts with a clean stall deadline.
- A worker that exits while rows are outstanding is restarted up to
  `--worker-restarts` times (default 2). A restart writes a filtered task file
  (`worker-N-genW-rR.jsonl`) with only the outstanding rows and starts a fresh
  engine (warmup runs again; the worker log is appended, not replaced).
- After the restart budget is exhausted, the outstanding rows are released as
  `worker_lost`, which **does** consume the transient retry budget.
- Worker exit code `0` means it finished or honored the stop marker; `3` means
  engine startup/warmup failure, crash, or a batch-level engine failure.

### 7.5 Stop markers, SIGTERM and the runtime budget

The controller handles `SIGTERM`/`SIGINT` (`StopFlag`). On a signal, or when
`--max-runtime-minutes` is reached, it stops claiming, creates
`runtime/stop`, waits up to `--shutdown-grace-seconds` (default 120) for
workers to finish their current batch, ingests final results, releases
outstanding rows as `run_interrupted` (no retry budget), writes a checkpoint,
and exits `3` if rows remain. Workers check the stop marker between
microbatches.

The runtime budget is measured from controller start and is checked before new
claims and inside each wave; it is not a hard kill of an in-flight batch.
Every runtime budget uses one clock: `time.monotonic()`. Ledger retry
timestamps are epoch seconds (`time.time()`), and the controller translates
them into a duration before comparing them with the monotonic deadline, so
retry waiting, the runtime budget and the wave stop check can never disagree
about which clock they are reading.

---

## 8. Retry policy

`RetryPolicy` defaults: `max_transient_attempts = 3`,
`backoff_base_seconds = 30`, `backoff_cap_seconds = 600`,
`normal_max_tokens = 256`, `truncation_max_tokens = 384`.

### 8.1 Categories

| Category | Members | Effect |
| --- | --- | --- |
| transient (counts against budget) | `engine_transient`, `engine_fatal`, `timeout`, `io`, `transport`, `decode`, `worker_lost`, `worker_stall` | `retryable` with exponential backoff |
| non-counting interruptions | `stale_claim`, `run_interrupted` | `retryable`, no budget, no backoff |
| permanent / semantic | `input_invalid`, `integrity`, `input_too_long`, `invalid_instruction` | `rejected` immediately, never retried |
| terminal reasons | `truncated_at_ceiling`, `transient_exhausted` | `rejected` after exhausting the escalation/budget |

A result is accepted only when `ok` is true, `finish_reason == "stop"`, and the
instruction is non-empty after stripping.

The categories the current code actually emits are `engine_transient`
(worker batch/engine failure, prompt mismatch, output-count mismatch), `io`
(image read failure), `decode` (prompt build failure), `integrity` (checksum
drift), `input_too_long`, `invalid_instruction`, `worker_lost` (controller
release), `run_interrupted`, `stale_claim`, and the terminal reasons
`truncated_at_ceiling` / `transient_exhausted`. The other transient names are
part of the policy and accepted by `release_running`, but are not produced by
the current worker/controller paths.

### 8.2 Transient budget and backoff

The just-finished attempt counts immediately. With the default budget of 3
transient attempts, the first two failures schedule retries and the third
rejects the row as `transient_exhausted`. Backoff is
`min(600, 30 * 2^(counted-1))`: 30 s after the first failure, 60 s after the
second. The controller waits in-process for short backoffs (up to
`--retry-wait-seconds`, default 300); longer waits end the run with exit `3`
and the row is retried on resubmission.

### 8.3 Truncation escalation

1. First attempt runs with `--max-output-tokens` (256). If the engine stops
   with `finish_reason == "length"`, the row becomes `retryable`.
2. The next claim runs with `--truncation-retry-tokens` (384,
   `attempt_kind = truncation_retry`). If it still ends with `length`, the row
   is rejected as `truncated_at_ceiling`.
3. Once any attempt has used 384 tokens, subsequent claims stay at 384.

### 8.4 Semantic failures

`integrity` (image/TikZ checksum mismatch detected in the worker or during
ingestion), `input_too_long` (prompt plus its reserved output ceiling exceeds
the context), and `invalid_instruction` (failed validation after a successful
generation) are rejected without retry: greedy decoding would reproduce the
same invalid text. Manifest-invalid rows are seeded as rejected with
`input_invalid`.

### 8.5 Reprocessing rejected rows

`--reprocess-rejected` moves rejected rows back to `pending` at startup,
preserving attempt history. Manifest-invalid rows (`input_invalid`) are never
moved. Use it deliberately, for example after fixing the cause of an
`integrity` rejection.

---

## 9. Export outputs

`export` writes to `WORK/export` (override with `--export-dir`) and requires
`run.json`, a matching ledger identity, a verified manifest, matching live
validation rules, and `pyarrow`. It also refuses any complete row whose finish
reason, prompt hash, config hash or model revision disagrees with the run.

### 9.1 Shards

`export/shards/shard-NNNNN.parquet`, one row per complete ledger row, ordered
globally by `source_row_index` across shards. Default `--shard-size 1000`
(100,000 complete rows produce 100 shards). Schema (`dataset-v1`):

| Field | Type | Notes |
| --- | --- | --- |
| `id` | string, non-null | stable row id |
| `source_row_index` | int32, non-null | 0-based frozen index |
| `file_id` | string | nullable, from the source row |
| `png_image` | binary, non-null | raw PNG bytes |
| `tikz_code` | string, non-null | source TikZ |
| `instruction` | string, non-null | accepted caption |
| `source_dataset` | string, non-null | `nllg/DaTikZ-V4` |
| `source_revision` | string, non-null | pinned dataset commit |
| `caption_model` | string, non-null | model id |
| `caption_model_revision` | string, non-null | pinned model commit |
| `prompt_version` | string, non-null | e.g. `caption-v1` |
| `image_sha256` | string, non-null | matches the frozen manifest |
| `tikz_sha256` | string, non-null | matches the frozen manifest |

Export refuses to write when a complete row maps to a rejected manifest row,
when TikZ or image bytes drift from the manifest, or when a complete row fails
the validation policy (defense in depth; `validate`/`audit` first).

### 9.2 Companion files

| File | Contents |
| --- | --- |
| `rejected.parquet` | one row per rejected ledger row: `id`, `source_row_index`, `rejection_reason`, `error_category`, `error_detail`, `attempt_count`, `updated_at`, `image_sha256`, `tikz_sha256` |
| `attempts.parquet` | every attempt: `row_id`, `attempt_no`, `attempt_kind`, `state`, `worker`, `max_tokens`, `started_at`, `ended_at`, `finish_reason`, `error_category`, `error_detail`, `prompt_tokens`, `completion_tokens`, `instruction`, `model_revision`, `prompt_sha256`, `config_hash` |
| `run-metadata.json` | run record (identity + provenance), manifest meta, export meta, ledger counts, shard list, rejected/attempts write summaries |
| `stats.json` | state counts, attempt count, truncation count, error categories, rejection reasons, token totals, first/last completion times |
| `checksums.json` | per-shard `name`, `rows`, `logical_sha256`, `file_sha256`; rejected/attempts summaries; `dataset_logical_sha256` |
| `validation-report.json` | `status: "pass"`, `checked` (exported rows), `failures: []`, `rules_sha256` |
| `dataset-card.md` | draft card: source, prompt, counts, schema, intended use, limitations; marked DRAFT / not uploaded |
| `export.meta.json` | `schema_version: export-v1`, `created_at`, `tool_version`, `run_id`, `identity_sha256`, `manifest_sha256`, `rows`, `rejected_rows`, `shards`, `shard_size`, `dataset_logical_sha256`, `complete_rows` |

### 9.3 Atomic shard lifecycle and idempotency

Each shard is written to a `.tmp` file, `fsync`ed, read back for row count and
first-id identity, then renamed atomically. `export.meta.json` is written last
and marks a completed export; audit reports `shards exist without
export.meta.json (partial export)` when it is missing. Re-exporting rebuilds
the shards from the ledger and deletes any `shard-*.parquet` this export does
not produce (for example after changing `--shard-size`).

Idempotency is defined on a **logical** checksum over
`id, source_row_index, image_sha256, tikz_sha256, instruction, prompt_version`
per row; `dataset_logical_sha256` is a stable digest of the shard logical
hashes. Unchanged input reproduces the same logical checksums. File SHA256s
are recorded for integrity and checked by a full audit.

---

## 10. Validation rules

`validate_instruction` is applied to every accepted instruction at ingestion
(authoritative, no retry) and again by `validate`, `audit`, and `export`.
Rules are checked in this order:

| # | Rule name | Condition |
| --- | --- | --- |
| 1 | `not_text` | value is not a string |
| 2 | `empty` | empty after `strip()` |
| 3 | `invalid_utf8` | cannot be UTF-8 encoded |
| 4 | `control_characters` | contains a NUL byte |
| 5 | `thinking_leak` | contains `<think` or `</think` (case-insensitive) |
| 6 | `source_tags` | contains `<source` or `</source` (case-insensitive) |
| 7 | `markdown_fence` | contains ``` ``` ``` |
| 8 | `tikz_code` | contains any of `\begin{`, `\end{`, `\draw`, `\node`, `\path`, `\fill`, `\coordinate`, `\foreach`, `\documentclass`, `\usepackage`, `\usetikzlibrary`, `\tikz`, `\pgf` |
| 9 | `task_reference` | contains any of `supplied image`, `supplied source`, `supplied input`, `provided input`, `provided image`, `provided source`, `source code`, `input image`, `reference image`, `this task`, `the task above`, `markdown fence` (case-insensitive) |
| 10 | `too_short` | fewer than `--min-instruction-chars` characters (default 20) |
| 11 | `too_long` | more than `--max-instruction-chars` characters (default 2000) |
| 12 | `too_many_words` | more than `--max-instruction-words` words (default 120) |

The rule tables and the limits are hashed into the run identity
(`validation_rules_sha256` is also part of the identity), so editing them
changes the identity and a resumed run refuses to continue.

---

## 11. Audit and validate

### 11.1 `audit`

`audit` verifies structural invariants and exits `1` when any check fails;
warnings do not fail. It writes `WORK/audit-report.json` with
`status`, `violations`, `warnings`, and the full `checks` list.

Manifest and identity checks:

- `manifest.verified`
- `manifest.exact_row_count` (`rows_frozen == row_limit`)
- `manifest.unique_stable_ids`
- `run.record_present`
- `run.prompt_matches_registry`
- `run.validation_rules_match` (live rule tables versus the pinned hash)

Ledger checks:

- `ledger.present`
- `ledger.identity_matches_run`
- `ledger.manifest_matches_run`
- `ledger.prompt_hash_consistent`
- `ledger.row_count_matches_manifest`
- `ledger.no_duplicate_ids`
- `ledger.row_set_matches_manifest` (no missing/extra ids)
- `ledger.no_index_above_limit`
- `ledger.manifest_rejections_preserved`
- `ledger.complete_rows_have_instructions`
- `ledger.rejected_rows_have_reasons`
- `ledger.instructions_attached_to_one_row` (**warning**, not a violation:
  identical greedy captions are legitimate; the detail reports how many)
- `ledger.index_pairs_match_manifest` (every ledger row's `(id, index)` pair
  matches the frozen manifest)
- `ledger.attempts_reference_known_rows`
- `ledger.accepted_instructions_valid` (re-runs the validation policy and
  checks checksums, `finish_reason == "stop"`, prompt hash, config hash, and
  model revision for every complete row)
- `runtime.no_quarantined_results` (**warning** when any result file was
  quarantined; its rows were reclaimed and regenerated)

Export checks (when `shards/*.parquet` exist):

- `export.meta_present`
- `export.only_complete_rows`
- `export.rejected_rows_excluded`
- `export.row_count_matches_ledger`
- `export.unique_ids`
- `export.rows_map_to_manifest`
- `export.checksums_match` (logical always; file hashes unless `--quick`)
- `export.companion_checksums_match` (rejected/attempts file hashes unless
  `--quick`)
- `export.shard_count_matches_meta`
- `export.rejected_matches_ledger`
- `export.identity_matches_run`

Without an export, audit records a `warn` on `export.checked` and still exits
`0` if everything else passes. With no `pyarrow`, export checks are skipped
with a warning.

### 11.2 `validate`

`validate` re-applies the validation policy to every complete row, checks
checksums/hashes/finish reason/model revision/prompt hash/config hash against
the run identity and manifest, writes `WORK/validation-report.json`
(`status`, `checked`, `failures`, `rules_sha256`, `manifest_sha256`) and exits
`1` when any row fails. It prints the first ten failures.

---

## 12. Recovery procedures

### 12.1 After a time limit or wall-time stop

The run exits `3`, prints `Runtime budget reached; no new claims`, releases
outstanding rows as `run_interrupted` (no retry budget) and writes a ledger
checkpoint. Resubmit the same script; it resumes from the ledger. Check
`status` to confirm no `running` rows remain.

### 12.2 After a controller or worker crash

- Result files written before the crash are ingested first on restart
  (`Recovered results from a previous run: N rows`).
- Rows left `running` by a dead worker are reclaimed immediately at startup
  (`Reclaimed N orphaned running rows`) because the controller lock guarantees
  no other controller is active. This does not consume the retry budget.
- A crashed worker is normally handled in-process: the controller restarts it
  up to `--worker-restarts` times, then releases its rows as `worker_lost`
  (which consumes the transient budget).
- If orphaned worker processes are still holding GPUs after a hard controller
  death, stop them before resubmitting (the Slurm job/cgroup normally takes
  them down with the job). Completed rows are never regenerated, so a lost
  in-flight batch is retried, not duplicated.
- Never run two controllers against one work directory; the second one refuses
  to start while the lock is held.
- Unprocessable result files are moved to `runtime/quarantine/` (the run
  continues and their rows are reclaimed). This covers malformed-but-valid
  JSON (missing keys, wrong types, invalid containers, duplicate row ids),
  foreign run ids, ownership conflicts, and ledger ingestion conflicts such as
  a conflicting instruction for an already-complete row. Inspect them before
  deleting; `audit` warns while any remain.

### 12.3 Corrupt or incomplete shard

Shards are written atomically, so a crash can leave at most a
`shard-NNNNN.parquet.tmp` plus the previous completed shards. Recovery:```bash
# Optional cleanup of an interrupted write; .tmp files are ignored by audit
# and export, so deleting them is safe.
rm -f /shared/$USER/tikz-production/export/shards/*.parquet.tmp

python3 cleaning/build_dataset.py export \
  --work /shared/$USER/tikz-production --shard-size 1000
```

Re-export rebuilds every shard atomically from the ledger and removes shards
the new export no longer produces. A completed export is marked by
`export.meta.json`; if shards exist without it, the export was partial — just
re-run export. If audit reports `export.checksums_match`, re-export to
regenerate the shards and checksums, then audit again.

### 12.4 Inspecting rejected rows

```bash
# Counts per state
sqlite3 /shared/$USER/tikz-production/ledger.sqlite3 \
  "SELECT state, COUNT(*) FROM rows GROUP BY state;"

# Rejection reasons and categories
sqlite3 -header -column /shared/$USER/tikz-production/ledger.sqlite3 \
  "SELECT rejection_reason, last_error_category, COUNT(*) AS n
     FROM rows WHERE state='rejected'
    GROUP BY 1, 2 ORDER BY n DESC;"

# Individual rejected rows
sqlite3 -header -column /shared/$USER/tikz-production/ledger.sqlite3 \
  "SELECT source_row_index, rejection_reason, last_error_category,
          attempt_count, last_error_detail
     FROM rows WHERE state='rejected'
    ORDER BY source_row_index LIMIT 20;"

# Full attempt history for one row
sqlite3 -header -column /shared/$USER/tikz-production/ledger.sqlite3 \
  "SELECT attempt_no, attempt_kind, state, max_tokens, finish_reason,
          error_category, error_detail, prompt_tokens, completion_tokens
     FROM attempts WHERE row_id='<row_id>' ORDER BY attempt_no;"
```

The same data is exported to `rejected.parquet` and `attempts.parquet`:

```bash
python3 - <<'PY'
import pyarrow.parquet as pq
root = "/shared/alice/tikz-production/export"
print(pq.read_table(f"{root}/rejected.parquet").to_pandas().head(20).to_string())
PY
```

### 12.5 Estimating completion time

`status` reports the recent rate (completions in the last 30 minutes,
extrapolated to an hour), the overall rate, and an ETA for the remaining
pending/retryable/running rows:

```bash
python3 cleaning/build_dataset.py status --work /shared/$USER/tikz-production
```

`status --json` exposes the raw numbers (`successful_per_hour_recent`,
`successful_per_hour_overall`, `estimated_remaining_seconds`) for scripts. The
measured baseline is 14,756.96 successful samples/hour for the validated
configuration (startup, retries and export excluded). When no rows remain, the
rendered ETA line says `done (no rows remain)`; confirm completion from the
state counts (`complete` + `rejected` = 100,000, everything else zero).

### 12.6 Backups

`run` writes a checkpoint when it finishes. Run `checkpoint` manually before
risky operations; backups are SQLite copies under `ledger-backups/` named
`ledger-YYYYmmdd-HHMMSS.sqlite3` (with a numeric suffix on collisions) and are
readable with the same queries as the live ledger.

---

## 13. Changing the model or the prompt

Every identity field is locked once `run.json` exists. To change anything
scientific, start a **new run identity in a fresh work directory**; never
reuse a work directory with a different configuration. The ledger and
`run.json` both refuse mismatches (`Ledger belongs to a different run
identity`, `Work directory run identity does not match the requested
configuration` with a field-by-field diff).

### 13.1 Prompt change

1. Add a new file, e.g. `cleaning/prompts/caption-v2.txt`. Do not edit a
   registered prompt: `load_prompt` refuses when the file hash does not match
   `registry.json` (`Do not edit a registered prompt; create a new version`).
2. Register its normalized SHA256 in `cleaning/prompts/registry.json`
   (`" ".join(text.split())` is the normalization). Helper:

   ```bash
   python3 - <<'PY'
   from pathlib import Path
   import hashlib
   text = " ".join(Path("cleaning/prompts/caption-v2.txt").read_text().split())
   print(hashlib.sha256(text.encode("utf-8")).hexdigest())
   PY
   ```

3. Start a fresh work directory and run with `--prompt-version caption-v2`.
   The prompt version and hash are part of the identity; a resumed run with
   the old `run.json` refuses.

### 13.2 Model change

- Use `--model OWNER/NAME` (and optionally `--model-revision`) with a fresh
  `--work` directory. `prepare` pins the revision, records the snapshot path
  and the model in `manifest.meta.json`; `run` and `export` must pass the same
  `--model`.
- A manifest frozen for one model is not reusable for another: `prepare` and
  `run` refuse when the requested model or revision disagrees with the frozen
  manifest. Re-freeze with the new model (the source slice is unchanged, so
  the stable ids and manifest hash are reproducible, but images are fetched
  again).
- Replacing the SIF changes the container hash and therefore the identity; a
  resumed run refuses until you start a fresh work directory.

### 13.3 New run without contaminating an old one

- Always pass a distinct `--work`.
- You may copy `manifest.jsonl`, `manifest.meta.json` and `images/` into the
  new directory when dataset, revision and model are unchanged (Section 5.8);
  the new ledger and `run.json` start clean.
- Do not update `build_dataset.py` or `benchmark.py` while a run is active:
  the runner/helper hashes are part of the identity, so a mid-run code change
  makes the existing work directory unresumable until the original code is
  restored.

---

## 14. Testing

Local tests are synthetic-only: they never download the dataset or model,
never start an inference engine, and need no GPU, no network and no Slurm
allocation. They exercise the real controller, ledger, task files and worker
loop with a deterministic fake engine, plus export/audit using temporary
directories.

```bash
# All cleaning tests (benchmark + production builder + integration)
python3 -m unittest discover -s cleaning -p 'test_*.py' -v

# Production builder only
python3 -m unittest discover -s cleaning -p 'test_build_dataset.py' -v

# Fault-injection integration tests only
python3 -m unittest discover -s cleaning -p 'test_pipeline_integration.py' -v

# Smoke checks that do not touch the network or a work directory
python3 cleaning/build_dataset.py plan
python3 cleaning/benchmark.py --plan
```

Notes:

- `pyarrow` enables the export tests and the export checks inside audit
  (`unittest.skipUnless(HAS_PYARROW, ...)`); without it, those tests are
  skipped rather than failed.
- Pillow is optional: when importable, `prepare` verifies PNG structure with
  `Image.verify()` and records `"image_validation": "pillow"`; otherwise it
  falls back to signature checks and records `"signature"`.
- A test asserts that importing `build_dataset.py` does not import `pyarrow`,
  `PIL`, `datasets`, `vllm` or `torch`.
- Tests use the hidden `--allow-non-slurm` flag, the hidden
  `--stall-kill-grace-seconds` control and injectable dependency seams;
  production runs must not use them.
- Deadline behavior is covered by integration tests with a non-null
  `--max-runtime-minutes`: workers must be allowed to run, and a retry backoff
  must be waited out under a runtime deadline. Malformed result files are
  covered by a table of resume tests (one per malformed case) plus a ledger
  conflict case.

---

## 15. Production runbook on Alex

Substitute your own checkout path, work directory and SIF path. The pipeline
itself never connects via SSH and never submits a job.

```bash
# 1. Get the code (clone once, then pull) and enter the checkout.
git clone https://github.com/Pranavharshans/Tikz-full-suite.git Tikz-training
# or, for an existing checkout: cd Tikz-training && git pull --ff-only
cd Tikz-training

# 2. Inspect the resolved plan (login node is fine; nothing is written).
python3 cleaning/build_dataset.py plan \
  --work /shared/$USER/tikz-production

# 3. Generate the sbatch script (printed to stdout, never submitted).
python3 cleaning/build_dataset.py slurm-script \
  --work /shared/$USER/tikz-production \
  --vllm-sif /shared/$USER/containers/vllm.sif \
  --wall-time 12:00:00 \
  > tikz-production.sbatch

# 4. Inspect the script, then submit it yourself.
less tikz-production.sbatch
sbatch tikz-production.sbatch

# 5. Monitor from any login node (read-only; no Slurm needed).
python3 cleaning/build_dataset.py status --work /shared/$USER/tikz-production
python3 cleaning/build_dataset.py status --work /shared/$USER/tikz-production --json

# 6. If the job ends with rows remaining (exit code 3), resume by resubmitting.
sbatch tikz-production.sbatch

# 7. Export and audit. The generated script already does this after a run;
#    to repeat manually inside an allocation (container provides pyarrow):
apptainer exec --nv --bind /shared/$USER/tikz-production --bind "$PWD/cleaning" \
  /shared/$USER/containers/vllm.sif python3 "$PWD/cleaning/build_dataset.py" \
  export --work /shared/$USER/tikz-production --shard-size 1000

apptainer exec --nv --bind /shared/$USER/tikz-production --bind "$PWD/cleaning" \
  /shared/$USER/containers/vllm.sif python3 "$PWD/cleaning/build_dataset.py" \
  audit --work /shared/$USER/tikz-production

# 8. Inspect rejected rows and attempts.
sqlite3 -header -column /shared/$USER/tikz-production/ledger.sqlite3 \
  "SELECT rejection_reason, last_error_category, COUNT(*) AS n
     FROM rows WHERE state='rejected' GROUP BY 1, 2 ORDER BY n DESC;"

# 9. Checkpoint the ledger before any maintenance.
python3 cleaning/build_dataset.py checkpoint --work /shared/$USER/tikz-production
```

### 15.1 How to verify success

- `run` exits `0` (the generated script then prints `All rows are terminal.`).
- `status` shows `complete` + `rejected` = 100,000 and `pending=running=
  retryable=0`.
- `validate` prints `Validation passed: <N> accepted rows match the policy and
  identity` and exits `0`.
- `audit` prints `Audit: pass (0 violations, ...)` and exits `0`.
- `export.meta.json` shows `rows` equal to the complete-row count, `shards`
  equal to `ceil(rows / shard_size)`, and `rejected_rows` equal to the
  ledger's rejected count.
- `checksums.json` contains a `dataset_logical_sha256`; re-running export
  reproduces it for unchanged input.
- `dataset-card.md` exists and is marked DRAFT / not uploaded.

---

## 16. Boundaries

- `stage-1/`, `stage-2/` and `stage-3/` are untouched by this pipeline. The
  builder reads only `cleaning/` and writes only inside `--work` (plus the
  container hash cache there). It never writes to the stage directories.
- `cleaning/benchmark.py` is preserved untouched as the benchmark tool; the
  production pipeline reuses only the audited helpers `bench.digest` and
  `bench.vllm_runtime_options`.
- `cleaning/results/` is the preserved record of the measured benchmark
  campaigns, including the RTX PRO 6000 results that selected the validated
  baseline and rejected the four-replica/two-per-GPU topology.
- Publishing the exported dataset is out of scope: the generated dataset card
  is a draft and requires separate authorization.
