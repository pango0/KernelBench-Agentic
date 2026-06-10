# shellcheck shell=bash
# Build toolchain for compiling KernelBench CUDA extensions.
#
# PyTorch's cpp_extension needs two things that the bare login/compute shell
# does NOT provide here, which is why every kernel in the agentic run failed to
# build ("CUDA_HOME not set", "cuda_runtime.h: No such file", "GCC 9 or later"):
#
#   1. a real CUDA toolkit (nvcc + headers + lib64) reachable via CUDA_HOME
#   2. a host compiler >= GCC 9 (the system default is GCC 8.4.1, which PyTorch
#      refuses) on PATH, with its runtime libs on LD_LIBRARY_PATH
#
# We load the dedicated Lmod modules first (canonical on this TWCC cluster),
# then fall back to the known install paths so it still works if `module` is not
# initialised. CUDA 12.3 is chosen because its major version (12) matches the
# installed torch 2.5.1+cu121; the nvhpc modules also ship nvcc but the readily
# available one is CUDA 11.7 (major 11), which torch would reject.
#
# Source this AFTER `conda activate`. Safe under `set -u`/`set -e`.

# --- CUDA toolkit -----------------------------------------------------------
module load cuda/12.3 2>/dev/null || true
if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  _kb_cuda=/work/HPC_SYS/twnia2/pkg-rocky8/nvidia/cuda/cuda-12.3
  if [[ -x "${_kb_cuda}/bin/nvcc" ]]; then
    export CUDA_HOME="${_kb_cuda}" CUDA_PATH="${_kb_cuda}" CUDA_ROOT="${_kb_cuda}"
    export PATH="${_kb_cuda}/bin:${PATH}"
    export LD_LIBRARY_PATH="${_kb_cuda}/lib64:${LD_LIBRARY_PATH:-}"
  fi
fi

# --- GCC >= 9 host compiler -------------------------------------------------
module load gcc10 2>/dev/null || true
_kb_gccmajor="$(gcc -dumpversion 2>/dev/null | cut -d. -f1)" || true
if [[ "${_kb_gccmajor:-0}" -lt 9 ]]; then
  _kb_gcc=/work/HPC_SYS/devtoolset/devtoolset-10/root
  if [[ -x "${_kb_gcc}/usr/bin/gcc" ]]; then
    export PATH="${_kb_gcc}/usr/bin:${PATH}"
    export LD_LIBRARY_PATH="${_kb_gcc}/usr/lib64:${_kb_gcc}/usr/lib:${LD_LIBRARY_PATH:-}"
  fi
fi

# Pin the host compiler to that GCC>=9. Otherwise an inherited CC/CXX (e.g.
# nvc/nvc++ from an nvhpc module exported into the job via --export=ALL) wins:
# torch feeds $CXX to the host .cpp build and $CC to nvcc's -ccbin, and nvc++ is
# tied to a pre-GCC-9 base, so it fails with the same "GCC 9 or later" error.
_kb_gccmajor="$(gcc -dumpversion 2>/dev/null | cut -d. -f1)" || true
if [[ "${_kb_gccmajor:-0}" -ge 9 ]]; then
  _kb_cc="$(command -v gcc 2>/dev/null)" || true
  _kb_cxx="$(command -v g++ 2>/dev/null)" || true
  if [[ -n "${_kb_cc}" && -n "${_kb_cxx}" ]]; then
    export CC="${_kb_cc}" CXX="${_kb_cxx}" CUDAHOSTCXX="${_kb_cxx}"
  fi
fi

echo "[toolchain] CUDA_HOME=${CUDA_HOME:-<unset>}" \
     "nvcc=$(command -v nvcc 2>/dev/null || echo none)" \
     "gcc=$(gcc -dumpversion 2>/dev/null || echo none) ($(command -v gcc 2>/dev/null || echo none))" \
     "CC=${CC:-<unset>} CXX=${CXX:-<unset>}"
