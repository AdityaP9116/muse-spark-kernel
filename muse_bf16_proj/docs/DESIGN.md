# BF16 Projection Design Notes

## Mathematical contract

`Y = X @ W.T`

- `X`: activations, shape `[m, k]`
- `W`: projection weights, shape `[n, k]`
- `Y`: output, shape `[m, n]`

The official problem includes several BF16 projections and an FP32 router; final accuracy conditions and scored shapes must be confirmed against official starter files before implementation.

## Study prompts

1. Derive one output element and its memory addresses for row-major `X` and `W`.
2. Compare low-`m` GEMV-like inference with larger-`m` GEMM.
3. Estimate FLOPs, bytes moved, and idealized arithmetic intensity.
4. Map each output tile to blocks, warps and threads.
5. Identify coalescing, reuse, precision, occupancy and register-pressure tradeoffs.

## Hypotheses (untested)

- Very small `m` may be limited by weight bandwidth.
- Larger `m` offers more opportunities to reuse loaded weights.
- A single kernel configuration may not perform well across all `m`.

## Design decisions

*Pending first implementation and profiling.*
