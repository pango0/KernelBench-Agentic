# Operator Fusion

Fusion removes intermediate tensors and the global-memory round-trips between them,
which is the dominant cost for memory-bound (low arithmetic-intensity) ops.

## Elementwise chains
Sequences like `bias add -> activation -> scale` are memory bound. Fuse them into one
kernel that reads each input element once, does all the arithmetic in registers, and
writes the result once. A single fused elementwise kernel typically beats N separate
PyTorch ops by ~N x in bandwidth.

## Epilogue fusion after matmul / conv
Fuse the bias, activation (ReLU/GELU/SiLU), residual add, and even the next
elementwise op into the matmul/conv *epilogue*, while the result tile is still in
registers/shared memory. Example: `Linear + bias + GELU` as one kernel avoids writing
and re-reading the pre-activation matrix.

## Normalization fusion
LayerNorm / RMSNorm / BatchNorm combine a reduction (mean/variance) with an
elementwise affine transform. Fuse: compute the reduction in shared memory / via warp
shuffles, then apply scale+shift in the same kernel. Fold the affine `gamma/beta` and,
at inference, fold BatchNorm into the preceding conv weights.

## What not to fuse
Do not fuse across an op that needs a global synchronization or changes the
parallelization axis (e.g. a full matmul between two elementwise stages) unless you
implement that op in the same kernel. Over-fusing can blow up register/shared usage
and crush occupancy.

## Correctness first
Fused kernels must match the reference numerically (within tolerance). Validate the
unfused math first, then fuse; keep the activation formula exact (e.g. tanh-approx vs
erf GELU must match the reference variant).
