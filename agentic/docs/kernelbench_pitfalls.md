# KernelBench Interface and Common Pitfalls

## Required interface
The reference module is `Model(nn.Module)`. Your replacement MUST be
`class ModelNew(nn.Module)` with:
- the **same `__init__` signature** and the same constructed parameters/buffers,
- the **same `forward` signature** producing a numerically equivalent output,
- module-level `get_inputs()` and `get_init_inputs()` returning the same shapes/dtypes
  as the reference file.

Output ONLY one ```python``` code block containing the full file (imports, the custom
kernel source, `ModelNew`, and the two input functions). No tests, no prose.

## How it is graded
1. **Compilation** — the extension/Triton kernel must build.
2. **Correctness** — outputs must match the reference within tolerance over several
   random inputs (`num_correct_trials`). Random seeds vary, so the kernel must be
   correct for arbitrary inputs, not one fixed case.
3. **Performance** — runtime is measured with CUDA events over `num_perf_trials`;
   speedup is reference_runtime / your_runtime. "fast_1" means correct AND faster than
   eager PyTorch.

## Frequent failure modes
- **Hard-coded shapes/dtypes**: read sizes from the tensors; support the dtype the
  reference uses (often fp32 here). Accumulate reductions in fp32.
- **Non-contiguous inputs**: call `.contiguous()` before pointer indexing.
- **Forgetting parameters**: keep `nn.Parameter`s (weights/bias) and initialize them
  exactly like the reference, or correctness fails even if the kernel is right.
- **Wrong activation variant**: match erf-GELU vs tanh-GELU, etc., to the reference.
- **Returning correct-but-slower kernels**: passing correctness is necessary but the
  score rewards speedup; fuse and cut memory traffic once correct.
- **Build cache collisions**: give each kernel a unique `load_inline(name=...)`.

## Strategy
Get a correct baseline first (even if it just calls the obvious fused op), confirm it
passes, then iterate on performance using the feedback. A correct slow kernel scores
far better than a fast wrong one (which scores zero).
