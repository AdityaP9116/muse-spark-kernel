# Muse Spark Open Kernel Challenge — Independent Kernel Lab

Independent CUDA and Triton implementations inspired by the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/). This is a **post-competition learning project**, not an official submission or a reproduction of leaderboard performance.

The aim is to understand the full chain:

**ML operator → mathematical formulation → GPU memory and execution model → correct kernel → performance measurement → defensible optimization.**

## Projects

| Project | Official problem | Status |
| --- | --- | --- |
| 01 — BF16 Projections | [`muse_bf16_proj`](https://openkernelchallenge.dev/problems/muse_bf16_proj) | Planning / setup |
| 02 — Attention Decode | `muse_attention_decode` | Backlog |
| 03 — Quantized MoE | `muse_moe_w4a16` | Backlog |
| 04 — Decode Layer | `muse_decode_layer` | Backlog |
| 05 — Prefill Layer | `muse_prefill_layer` | Backlog |

## Current focus: BF16 Projections

See [`muse_bf16_proj/README.md`](muse_bf16_proj/README.md).

**Goal:** investigate the projection `Y = X @ W.T` across small-token decode and larger verification batches. We will build a correct reference, trace its indexing and precision, write educational GPU kernels, benchmark representative shapes, and optimize based on measured bottlenecks.

**Principles**
- Correctness before speed. Follow the official test contract after retrieving the challenge starter.
- Compare implementations on the *same machine, workload, and timing harness*.
- Report software versions, precision, numerical tolerances, benchmarks, and failures.
- Local RTX 5070 Laptop GPU measurements are **not** comparable to official NVIDIA H200 leaderboard runs.
- Preserve competition starter files and their upstream licenses. Do not copy third-party files without attribution.
- Keep the submitted kernel's official interface separate from exploratory code.

## Project coordination

- [Notion — Muse Spark project hub](https://app.notion.com/p/419f393aa7688109a5e4dc0429c4b66d)
- [Notion — BF16 Projections engineering workspace](https://app.notion.com/p/419f393aa76881a09887f864e2018f5f)
- [Official BF16 problem](https://openkernelchallenge.dev/problems/muse_bf16_proj)

## Development environment

Planned: Ubuntu in WSL2, NVIDIA CUDA Toolkit, Python managed with `uv`, PyTorch, and Triton. The environment has **not yet been verified for this repository**; installation commands and pinned dependencies will be recorded after the audit.

## Getting started

1. Clone this repository.
2. Review [the BF16 project brief](muse_bf16_proj/README.md).
3. Retrieve and inspect the official starter files, if still available, **before** adding `submission.py`, `reference.py`, `checks.py`, or other reserved filenames.
4. Run reference correctness tests on feasible local shapes.
5. Implement, benchmark, and document incremental optimization steps.

## Status

Repository scaffold only. **No kernel is implemented, tested, or benchmarked yet.**
