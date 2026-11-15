# Project 01 — BF16 Projections

Official task: [muse_bf16_proj](https://openkernelchallenge.dev/problems/muse_bf16_proj)  
Notion project: [BF16 Projections workspace](https://app.notion.com/p/419f393aa76881a09887f864e2018f5f)

## Objective

Implement and optimize the projection operation `Y = X @ W.T`, examining how batch/token dimension `m`, matrix shapes, and precision affect GPU mapping and performance. We will learn the implementation from first principles before looking at optimized contestant solutions.

## Milestone checklist

- [ ] M0: verify local CUDA / uv / PyTorch / Triton environment and retrieve the official starter
- [ ] M1: validate a PyTorch baseline and numerical acceptance tests
- [ ] M2: implement a correct educational CUDA kernel and explain per-thread mapping
- [ ] M3: optimize CUDA and/or Triton using controlled benchmarks
- [ ] M4: publish performance results, experiments, and limitations

## Directory plan

- `experiments/` — independent trial implementations; **not** competition submission files
- `tests/` — local correctness tests against reference behavior
- `benchmarks/` — local benchmark harness and shape sweeps
- `docs/` — derivations, performance hypotheses, experiment evidence
- `submission.py`, `reference.py`, `baseline.py`, `checks.py`, `TASK.md` — **reserved** for official starter files if obtainable

Do not add reserved starter filenames or assume exact test tolerances until examining the downloaded challenge bundle.

## Ground rules

Record numerical correctness, shape, dtype, timing methodology, hardware and software versions. Start with small cases that fit an 8 GB RTX 5070 Laptop GPU; official H200 performance claims require matching hardware and harness.

**Current status:** Scaffold only — no kernels or test results yet.
