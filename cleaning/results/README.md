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
