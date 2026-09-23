# Cleaning

Benchmark image-plus-TikZ caption generation before cleaning the training dataset.

`benchmark.py` is the single runtime entry point. It freezes 100 DaTikZ-V4 samples,
downloads a pinned Qwen3.8-27B-FP8 snapshot, runs staged experiments in a four-GPU
Slurm allocation, and records results under the selected work directory.

## Fixed experiment

- Official FP8 checkpoint only; no precision sweep or TP1 runs.
- Thinking enabled, `reasoning_effort=xhigh`, temperature 1, top-p .95, top-k 20.
- No independent output-length limit: generation can use the remaining context
  window (default 32768 total tokens). This explicit remainder prevents engine
  defaults such as 16 output tokens. Context exhaustion is a failed/truncated run.
- Full source and image retained; oversized inputs fail without silent truncation.
- 100 unique measured records per configuration; five warmup records per replica.
- Source/code-length stratification within the first 1000 streamed candidates.
  This bounded pool is not a globally representative sample of all dataset rows.
- HTTP outputs keep reasoning and final instruction separate. Only final text is
  a candidate training instruction. Caption quality still requires human review.

## Matrix

| Stage | Settings | Runs |
| --- | --- | ---: |
| Engine | vLLM offline, vLLM HTTP, SGLang HTTP; TP2, MTP2; concurrency 1/8/32 | 9 |
| MTP | Winning engine; off/1/2/3; concurrency 1/2/4/8/16/32/64/100 | 32 |
| Topology | Two TP2 replicas versus one TP4; global concurrency 2/4/8/16/32/64/100 | 14 |
| Scheduling | Random/length ordering; chunked prefill off/on; token budgets 8192/16384/32768 | 12 |
| Finalists | Three best scheduling configurations, three repetitions each | 9 |

This is a staged search, not a full Cartesian product. A complete 100/100 result
is required for selection; MTP-off is a control and cannot become a stage winner.
Different engines use their own native speculative-step semantics. Unsupported
combinations are recorded as failures, never silently replaced with other settings.
A40 FP8 kernel compatibility must be established by the actual run; stored FP8
weights do not imply native FP8 hardware execution. TP2 is the requested minimum,
not a claim that this FP8 checkpoint always needs 60 GB.

Offline concurrency means `max_num_seqs`; HTTP concurrency means in-flight
requests with matching engine sequence capacity. Replicas receive deterministic
code-length-balanced shards and synchronize after warmup. This is static sharding;
long reasoning tails may cause imbalance. A 100-row batch measures finite-batch
completion, not steady-state saturation at concurrency 64/100.

## Run on Alex

Use a dedicated TikZ directory, separate from other experiments. These commands
are for the cluster terminal. The runner never connects via SSH or submits itself.

Provide local Apptainer SIF files with compatible vLLM and SGLang installations.
The vLLM image also needs `datasets`, `huggingface_hub`, Pillow and Transformers.
Pin container versions/digests when obtaining them; the runner hashes actual SIF
bytes for resume identity. No container is silently pulled or modified.

```bash
git clone https://github.com/Pranavharshans/Tikz-full-suite.git
cd Tikz-full-suite
python3 cleaning/benchmark.py --plan

# Replace these paths with your actual local container paths.
python3 cleaning/benchmark.py --slurm-script \
  --work "$WORK/tikz-benchmark" \
  --vllm-sif /absolute/path/vllm.sif \
  --sglang-sif /absolute/path/sglang.sif > benchmark.sbatch

# Inspect the generated job, then submit it yourself.
sbatch benchmark.sbatch
```

The generated job requests one node, four A40s, 64 CPUs, and 24 hours. Preparation
and inference happen in that allocation. Resubmit the same job to resume completed
configurations; an interrupted configuration reruns all 100 rows to retain valid
timing. Many thinking-enabled sweeps can exceed one allocation. Completed results
are reusable only when script, manifest, containers, context and configuration match.
Request timeout defaults to 30 minutes; each configuration has a four-hour outer
deadline. Neither timeout is a token limit. Offline requests are protected by the
outer configuration deadline rather than individual request timeouts.

Outputs: frozen `dataset.json`, source images, checkpoint cache, per-replica logs,
GPU telemetry, streamed HTTP request JSONL, raw speculative metrics, per-run
summaries, aggregate `summary.csv`, `results.json`, and `finalist-repeats.json`.
Offline results are written when the generation call completes. Tokens come from
engine usage or token IDs, never streaming-chunk counts. Raw metrics retain the
evidence for MTP acceptance; acceptance is not synthesized if unavailable.
`wall_s` excludes startup/warmup, while `startup_inclusive_s` includes them.
GPU-hour ranking uses active model GPUs; the job reserves four GPUs even during
TP2-only tests, so allocation billing can be higher than that ranking suggests.

## Local validation

### Split screening on four A40s

Pass `--split-screen --warmup-samples 1` when generating the Slurm script.
The frozen dataset remains 100 samples; only screening subsets change:

| Global concurrency | Measured screening samples |
| --- | --- |
| 1, 2, 4, 8 | 16 |
| 16 | 32 |
| 32 | 64 |
| 64, 100 | 100 |

All original staged settings remain; this is not the full Cartesian product.
Concurrency 8/16/32 is attempted before the slower 1/2 settings. Each engine
instance gets one untimed warmup request with the command above. Subsets are
deterministic nested prefixes interleaving four TikZ source-length strata.
They are not stratified by unknown output length or measured image complexity.
Counts are global across replicas, not counts per GPU. High concurrency is an
upper bound: the queue drains, and memory limits may reduce active concurrency.

Unequal-subset throughput is only a screening heuristic; it can mis-rank candidates.
At each stage the top three successful MTP-enabled candidates are compared again
on the same 100 samples before selecting the next stage's winner. The final three
receive three further 100-sample repeats. This means 67 screening runs, up to 12
stage confirmations, and up to nine repeats (88 logical runs); matching cached
runs may be reused. This reduces sample work, not necessarily the number of launches.
More extensive screening may be needed if rankings are close.

Split mode resumes completed runs and recorded failures instead of repeatedly
retrying unsupported settings. Interrupted runs without a result rerun from scratch.
To retry a recorded failure, preserve its log and move its `failure.json` aside.
No individual offline sample checkpoint is claimed. A short Slurm allocation
does not guarantee completion of all stages or even the active configuration.
Run only one orchestrator against a work directory at a time; do not edit/pull the
runner while a job is active. Old 100-sample mode remains available without the flag.

### Focused non-thinking throughput screen

`--throughput-screen --no-enable-thinking --max-output-tokens 256` runs only
three bounded vLLM-offline MTP2 candidates: one TP2 replica at concurrency 32,
then two TP2 replicas at aggregate concurrency 64 and 100. Each candidate uses
the same 100 frozen samples. Non-thinking mode uses Qwen's recommended instruct
sampling and asks for an instruction under 120 words. A request that exhausts
the 256-token ceiling is recorded as failed rather than accepted as a complete
caption.

This mode writes `throughput-results.json` and `throughput-summary.csv`; it does
not overwrite the original staged benchmark summary. It is a focused capacity
probe, not a replacement for caption-quality review. Generate its Slurm script
with:

```bash
python3 cleaning/benchmark.py --slurm-script \
  --throughput-screen --no-enable-thinking --max-output-tokens 256 \
  --warmup-samples 1 --work /absolute/benchmark-data \
  --vllm-sif /absolute/containers/vllm.sif \
  --sglang-sif /absolute/containers/sglang.sif > throughput.sbatch
```

For two 96GB RTX PRO 6000 Blackwell GPUs, use `--rtx-throughput-screen` instead.
That isolated mode uses TP1 and tests one replica at concurrency 32, two replicas
at aggregate concurrency 64 with MTP 0/1/2/3, and two replicas at aggregate
concurrency 100 with MTP2. It writes `rtx-throughput-results.json` and
`rtx-throughput-summary.csv`, requests the `rtxpro6k` partition and exactly two
GPUs, and retains the same non-thinking 256-token quality controls.

After selecting MTP, `--rtx-concurrency-screen --load-samples 512
--throughput-mtp N` runs a sustained two-replica TP1 load test at aggregate
concurrency 64, 96, 128, 192, and 256. It cycles the immutable 100-row workload
with unique request IDs; these are 512 requests, not 512 unique dataset samples.
Prefix caching remains disabled. This isolates capacity saturation without a
new dataset download and writes `rtx-concurrency-results.json` plus
`rtx-concurrency-summary.csv`. Use it for throughput, not quality statistics.

### Tests

```bash
python3 -m unittest discover -s cleaning -p 'test_*.py' -v
python3 cleaning/benchmark.py --plan
```

GPU/container execution must still be validated on Alex. Local tests cover the
matrix, winner selection, streamed reasoning separation and truncation detection.

Measured hardware results are preserved under [`cleaning/results`](results/).

Resuming an existing `dataset.json` uses only host Python's standard library to
validate image hashes; it does not require `datasets` or reinstall dependencies.
For first-time preparation in a separate container virtual environment, pass
`--prepare-python /absolute/path/preparation-env/bin/python` when generating the
Slurm script. The generator binds that environment into the container.

## Alex A40 runtime findings (updated 2026-09-23)

Current status: offline vLLM TP2 now has successful one-sample eager and compiled
tests, plus two successful ten-sample compiled trials at concurrency 4 with an
engine restart (job 4298526). Both NCCL P2P and vLLM custom all-reduce were
disabled. GPU memory was released after the second trial, despite forced-cleanup
warnings. Trial 2 completed its measured batch in 113.91 seconds. Its cumulative
metrics (including warmup) reported 12,984 draft tokens and 9,741 accepted tokens
(75%); this establishes active MTP, not a speedup over an MTP-off baseline.

The runner now sets `disable_custom_all_reduce=True` for offline vLLM and
`--disable-custom-all-reduce` for vLLM HTTP. Compiled execution stays enabled.
Offline stats are explicitly enabled for MTP metrics, and SGLang is launched
with `--enable-metrics`. Offline engine options are saved in `engine-options.json`;
warmup and measured generation now report progress in worker logs. Changes to
the script hash invalidate old resume identities automatically.

The full sweep is experimental: vLLM HTTP, SGLang, TP4, two replicas, and higher
concurrency have not been validated by these smoke tests. SGLang retains its own
custom-collective defaults; its NCCL P2P setting still comes from the runner.
GPU-pair changes and warmed caches prevent attributing all earlier stalls to
one root cause. Caption quality has not been reviewed from the saved outputs.

### Earlier diagnostic history

The runner now defaults to `--nccl-p2p disabled`, explicitly setting
`NCCL_P2P_DISABLE=1` on the host and through `APPTAINERENV_NCCL_P2P_DISABLE`
for container workers. Generated Slurm scripts pass this option. Use
`--nccl-p2p auto` only for a deliberate comparison on other hardware or after
the cluster issue is resolved. The selected mode is recorded in each replica's
`job.json` and included in resume fingerprints; timings from different modes
must not be mixed. Regenerate existing submission scripts to include the option.

Evidence from user-run jobs on a0429:

- Job 4295289: with P2P disabled, both ranks passed ten checked PyTorch
  all-reduces using SHM/direct. The default P2P path did not show a successful
  collective in the supplied logs. This is a workaround validated on that pair,
  not proof of a cluster-wide root cause or of every TP4/replica topology.
- Job 4295313: with this workaround, vLLM passed the earlier communication
  stall and loaded weights on both ranks (about 15.12 GiB each, 34 seconds).
  Backbone and speculative-head compilation subsequently completed.
- A later startup stall remains unresolved. After the 21:12:56 milestone,
  only engine wait messages appeared for over ten minutes. A symbol-resolved
  TP0 stack showed Triton's `loadBinary` calling CUDA `cuModuleLoadData`.
  TP1's useful stack was not captured. This identifies where TP0 was sampled,
  but does not prove a driver bug, cache corruption, or MTP failure.

Those early logs did not demonstrate completed generation. Subsequent tests
above establish a working configuration, but not the exact cause of the earlier
kernel-loading stall. Do not delete existing caches as an unverified fix. Diagnostic container commands
must use `python3` (the image did not provide a `python` executable).
