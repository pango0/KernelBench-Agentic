"""Local KernelBench eval wrapper for one kernel during refinement."""

import hashlib
import os
import shutil
import sys
from pathlib import Path

import torch

KB_SRC = Path(__file__).resolve().parent.parent / "KernelBench" / "src"
if str(KB_SRC) not in sys.path:
    sys.path.insert(0, str(KB_SRC))

from kernelbench import eval as kernel_eval  # noqa: E402


def evaluate_kernel(
    ref_arch_src: str,
    kernel_src: str,
    device_id: int = 0,
    build_dir_root: str = None,
    num_correct_trials: int = 5,
    num_perf_trials: int = 30,
    backend: str = "cuda",
    precision: str = "fp32",
    timing_method: str = "cuda_event",
    verbose: bool = False,
):
    device = torch.device("cuda:{}".format(device_id))
    if build_dir_root is None:
        build_dir_root = str(
            Path(__file__).resolve().parent / "build" / "eval_cache"
        )

    kernel_hash = hashlib.md5(kernel_src.encode()).hexdigest()[:12]
    build_dir = os.path.join(build_dir_root, kernel_hash)
    shutil.rmtree(build_dir, ignore_errors=True)

    dtype = kernel_eval.get_torch_dtype_from_string(precision)

    try:
        return kernel_eval.eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=kernel_src,
            measure_performance=True,
            timing_method=timing_method,
            verbose=verbose,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            build_dir=build_dir,
            device=device,
            backend=backend,
            precision=dtype,
        )
    except Exception as e:
        meta = {
            "runtime_error": str(e),
            "hardware": (
                torch.cuda.get_device_name(device=device)
                if torch.cuda.is_available()
                else "unknown"
            ),
            "device": str(device),
        }
        if "CUDA error" in str(e):
            return kernel_eval.KernelExecResult(
                compiled=False, correctness=False, metadata=meta
            )
        return kernel_eval.KernelExecResult(
            compiled=True, correctness=False, metadata=meta
        )
