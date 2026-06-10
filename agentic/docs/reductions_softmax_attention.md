# Reductions, Softmax, and Attention

## Warp-level reductions
Within a warp, reduce with `__shfl_down_sync` (NVIDIA) / `__shfl_down` (HIP) instead
of shared memory — no `__syncthreads()` needed and very low latency. For block-level
reductions, do a warp reduction, write one partial per warp to shared memory, then
reduce those partials with a single warp.

## Numerically stable softmax
softmax(x)_i = exp(x_i - m) / sum_j exp(x_j - m), with m = max_j x_j. Always subtract
the row max before exponentiating to avoid overflow. This is a two-pass reduction
(max, then sum) unless you use the online algorithm.

## Online / streaming softmax
Compute max and sum in a single pass: maintain running max `m` and running sum `l`;
when a new value `x` arrives, `m_new = max(m, x)`, `l = l * exp(m - m_new) + exp(x - m_new)`,
`m = m_new`. This removes a full pass over the row and is the basis of FlashAttention.

## FlashAttention-style fused attention
Never materialize the full N×N attention matrix. Tile over keys/values; for each query
block, stream key/value blocks, update the online-softmax running statistics and the
output accumulator in shared memory / registers. This turns attention from
memory-bound O(N^2) traffic into a fused, IO-aware kernel and is the standard win for
self-attention modules.

## Reduction correctness
Watch the reduction dtype: accumulate in fp32 even for fp16 inputs to keep the result
within tolerance of the PyTorch reference. Handle the tail when the row length is not a
multiple of the block size with masking.
