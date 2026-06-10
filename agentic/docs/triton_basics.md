# Triton Kernel Basics

Triton lets you write GPU kernels in Python with block-level semantics; the compiler
handles vectorization, coalescing, and (with autotune) block-size selection.

## Anatomy of a kernel
```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
```
Launch with a grid lambda: `add_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=1024)`.

## Key ideas
- `tl.program_id(axis)` identifies the block; build offsets with `tl.arange`.
- Always pass a boolean `mask` to `tl.load`/`tl.store` for the ragged tail.
- `BLOCK: tl.constexpr` is a compile-time constant; powers of two (256–1024) are typical.
- Reductions: `tl.sum`, `tl.max` along an axis; matmul: `tl.dot(a, b)` (uses Tensor/Matrix cores for fp16/bf16/tf32).

## Autotuning
```python
@triton.autotune(configs=[triton.Config({'BLOCK': b}, num_warps=w)
                          for b in (256, 512, 1024) for w in (4, 8)], key=['n'])
```
Lets Triton pick the best block size / warp count per problem size.

## From PyTorch
Allocate the output with `torch.empty_like(x)`, pass `.data_ptr()` implicitly by passing
the tensors, and read sizes with `x.numel()` / `x.shape`. Keep inputs contiguous.
Accumulate reductions in fp32 (`acc = tl.zeros(..., dtype=tl.float32)`) for accuracy.
