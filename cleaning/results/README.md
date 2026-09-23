# Throughput results

## FAU Alex A40, 2026-09-23

[`a40-throughput-2026-09-23.csv`](a40-throughput-2026-09-23.csv) records the
completed non-thinking throughput screen from Slurm job `4299285` on node
`a0429`. The served checkpoint was `Qwen/Qwen3.8-27B-FP8`; inference used vLLM
offline, MTP2, NCCL P2P disabled, vLLM custom all-reduce disabled, a 32,768-token
context, and a 256-token output ceiling. Prompts requested instructions under
120 words.

The best measured configuration was two TP2 replicas at aggregate concurrency
64: 94 complete responses in 106.59 seconds, or 3,174.75 successful samples per
wall-clock generation hour across four A40 GPUs. The GPU-hour metric excludes
unused allocated GPUs for the one-replica row. Startup-inclusive time includes
model loading, compilation, warmup, synchronization, and shutdown. Responses
that reached the output ceiling were counted as failures; caption quality was
not evaluated in this run.

## FAU Alex RTX PRO 6000 Blackwell, 2026-09-23

[`rtxpro6000-concurrency-2026-09-23.csv`](rtxpro6000-concurrency-2026-09-23.csv)
records the completed sustained concurrency screen from Slurm job `4300007` on
node `a2041`. It used two independent TP1 replicas on two NVIDIA RTX PRO 6000
Blackwell Server Edition 96GB GPUs, vLLM offline, MTP1, non-thinking generation,
a 16,384-token batch budget and a 256-token output ceiling.

Aggregate concurrency 64 was fastest: 476 complete responses in 153.15 seconds,
or **11,189.15 successful samples per wall-clock generation hour** across both
GPUs. Increasing concurrency did not improve throughput: concurrency 96, 128,
192 and 256 produced 10,621.08, 10,439.17, 10,242.43 and 10,556.42 successful
samples/hour respectively. Concurrency 256 completed the most responses
(485/512) but remained 5.7% slower than concurrency 64 by successful throughput.

This was a capacity test: 512 uniquely identified requests cycled the same 100
frozen dataset samples. Prefix caching was disabled. The speeds exclude engine
startup, compilation and shutdown; `startup_inclusive_s` preserves those costs.
Responses reaching the output ceiling were counted as failures, and caption
quality was not evaluated.

## RTX PRO 6000 decoding and quantization comparison

[`rtxpro6000-decoding-comparison-2026-09-23.csv`](rtxpro6000-decoding-comparison-2026-09-23.csv)
records the isolated RTX experiments on two NVIDIA RTX PRO 6000 Blackwell
96GB GPUs. The two-replica runs assign one TP1 replica per GPU; the four-replica
run assigns two TP1 replicas per GPU. Aggregate concurrency is 64 and the
benchmark uses greedy decoding unless noted. The 512 requests cycle 100 frozen
dataset samples, so this is a throughput benchmark rather than a 512-example
quality evaluation. Output-ceiling responses count as failures, and caption
quality has not yet been reviewed.

| Model / decoding | Replicas | Batch budget | Successful | Wall (s) | Successful samples/hour |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B FP8, sampled | 2 | 16K | 476/512 | 153.15 | 11,189.15 |
| Qwen3.8-27B FP8, greedy | 2 | 16K | 478/512 | 149.27 | 11,527.84 |
| Qwen3.8-27B FP8, greedy | 2 | 32K | 479/512 | 148.95 | 11,576.99 |
| Qwen3.8-27B NVFP4, greedy | 2 | 16K | 483/512 | 117.83 | **14,756.96** |
| Qwen3.8-27B NVFP4, greedy, 12GiB KV/engine | 4 (2/GPU) | 16K | 472/512 | 145.41 | 11,685.80 |

The current winner is **NVFP4 with two total replicas at aggregate
concurrency 64**, at 14,756.96 successful samples/hour. Four colocated
replicas were slower and had more failures, so that topology is rejected for
production. The NVFP4 result is 31.9% faster than the FP8 sampled baseline and
28.0% faster than the equivalent FP8 greedy 16K configuration.
