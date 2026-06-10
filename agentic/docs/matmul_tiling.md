# Matrix Multiply, Tiling, and Tensor Cores

## Blocked / tiled GEMM
Partition C into BMxBN output tiles, one per thread block. Loop over K in steps of BK:
each step cooperatively loads an BMxBK tile of A and an BKxBN tile of B into shared
memory, `__syncthreads()`, and accumulates partial products into per-thread registers.
This raises arithmetic intensity from O(1) to O(BK) flops per byte.

## Register blocking (thread tiling)
Have each thread compute a small TMxTN micro-tile of C (e.g. 8x8) held in registers,
not a single element. This amortizes shared-memory reads and exposes instruction-level
parallelism. Typical config: 128x128 block tile, 8x8 thread tile.

## Double buffering
Prefetch the next K-tile into a second shared-memory buffer while computing on the
current one, hiding global-memory latency behind compute.

## Tensor Cores / Matrix Cores
For fp16/bf16/tf32, use Tensor Cores (NVIDIA `wmma` / `mma`) or AMD Matrix Cores
(`__builtin_amdgcn_mfma_*`). Operate on 16x16-style fragments. Easiest path from
PyTorch: emit Triton with `tl.dot`, or call cuBLAS/hipBLAS for plain GEMM and only
hand-write kernels for the fused or irregular cases.

## Don't beat cuBLAS at its own game
A naive custom GEMM rarely beats vendor BLAS for plain square matmul. Custom kernels
win when you *fuse* the epilogue, exploit sparsity/structure, use smaller dtypes, or
handle shapes BLAS handles poorly. For KernelBench level-1 matmul, fusing or using
`tl.dot`/library calls is usually the right move.

## Occupancy vs. tile size
Bigger tiles = more reuse but more shared memory / registers = fewer concurrent
blocks. Tune BM/BN/BK and thread tile to balance reuse against occupancy for the GPU.
