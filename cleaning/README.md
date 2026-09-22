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

```bash
python3 -m unittest discover -s cleaning -p 'test_*.py' -v
python3 cleaning/benchmark.py --plan
```

GPU/container execution must still be validated on Alex. Local tests cover the
matrix, winner selection, streamed reasoning separation and truncation detection.

Resuming an existing `dataset.json` uses only host Python's standard library to
validate image hashes; it does not require `datasets` or reinstall dependencies.
For first-time preparation in a separate container virtual environment, pass
`--prepare-python /absolute/path/preparation-env/bin/python` when generating the
Slurm script. The generator binds that environment into the container.

## Alex A40 runtime findings (2026-09-22)

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

No completed caption-generation benchmark has been demonstrated by these logs.
Do not resubmit a full sweep assuming the remaining issue is fixed. The next
diagnostic should isolate CUDA/Triton kernel loading in the same container.
Do not delete existing caches as an unverified fix. Diagnostic container commands
must use `python3` (the image did not provide a `python` executable).
