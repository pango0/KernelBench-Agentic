# Writing Custom Kernels with torch.utils.cpp_extension

## load_inline basics
`torch.utils.cpp_extension.load_inline` JIT-compiles CUDA/C++ source at import time:

```python
from torch.utils.cpp_extension import load_inline
mod = load_inline(
    name="my_ext",
    cpp_sources=cpp_src,        # declarations + pybind
    cuda_sources=cuda_src,      # __global__ kernels + launchers
    functions=["my_op"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)
```

Call `mod.my_op(x)` from `ModelNew.forward`. Keep `name` unique per kernel variant so
the build cache does not collide.

## Tensor handling rules
- Accept `torch::Tensor`; call `x.contiguous()` if your indexing assumes contiguity.
- Read shapes with `x.size(i)`; get pointers with `x.data_ptr<float>()` (or
  `at::Half`, etc.). Match the dtype the reference uses.
- Allocate outputs with `torch::empty`/`empty_like` and the input's
  `.options()` so device and dtype match.
- Guard dtype/device with `TORCH_CHECK(x.is_cuda(), "...")` and
  `TORCH_CHECK(x.scalar_type()==at::kFloat, "...")`.

## Launch configuration
Compute grid/block from tensor sizes: `int threads=256; int blocks=(n+threads-1)/threads;`
Always handle `idx < n` with a bounds check inside the kernel.

## Multiple dtypes
Use `AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "my_op", [&]{ ... scalar_t ... });`
to instantiate the kernel for the actual dtype instead of hard-coding `float`.

## Common build/runtime errors
- "CUDA_HOME not set" / "nvcc not found": toolkit not on PATH (build environment issue).
- "undefined symbol": the function name in `functions=[...]` / pybind must match.
- "illegal memory access": out-of-bounds indexing or wrong stride — re-check the
  bounds guard and that tensors are contiguous.
- Shape/stride mismatch vs reference: forgot `.contiguous()` or transposed indexing.

## When to prefer Triton
Triton (`@triton.jit`) is often faster to write correctly than raw CUDA for
elementwise, reductions, softmax, and tiled matmul (`tl.dot`), and it autotunes block
sizes. Use it when you do not need warp-level intrinsics that Triton hides.
