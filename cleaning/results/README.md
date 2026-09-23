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
