# GPU Memory Optimization for Custom Kernels

## Coalesced global memory access
Threads in a warp (32 lanes on NVIDIA, 64 on AMD CDNA) should access consecutive
addresses so the hardware coalesces them into the fewest memory transactions. Index
global arrays as `arr[blockIdx.x * blockDim.x + threadIdx.x]` along the contiguous
(last) dimension. Strided or transposed access patterns waste bandwidth; restructure
the loop or stage data through shared memory instead.

## Shared memory tiling
Load a tile of the input into `__shared__` memory once, `__syncthreads()`, then reuse
it many times from shared memory (≈100x lower latency than global). This is the core
trick for matmul, convolution, and stencils. Choose tile sizes (e.g. 16x16 or 32x32)
so the tile fits in shared memory and keeps occupancy reasonable.

## Bank conflicts
Shared memory is split into 32 banks. If multiple threads in a warp hit the same bank
(different address) the accesses serialize. Pad shared arrays by one column
(`__shared__ float tile[T][T+1];`) to break the conflict for column-major access.

## Vectorized loads
Use `float4` / `int4` (or `reinterpret_cast<float4*>`) to load 128 bits per
instruction, cutting the number of memory instructions and improving bandwidth
utilization. Requires the base pointer and index to be 16-byte aligned.

## Read-only and cached paths
For data that is read but never written by the kernel, use `__ldg()` or mark pointers
`const __restrict__` to let the compiler use the read-only data cache and assume no
aliasing, enabling more reordering and vectorization.

## Minimize host<->device traffic
Keep intermediates on the device. Allocate outputs with `torch::empty_like` / the
right `options()` and avoid `.cpu()`/`.item()` syncs inside the hot path.
