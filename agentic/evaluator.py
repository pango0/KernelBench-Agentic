#!/usr/bin/env python3
"""Evaluator agent: run a generated kernel through the KernelBench harness.

This is the only non-LLM agent in the loop. It compiles the candidate `ModelNew`,
checks numerical correctness against the reference, and measures runtime on a GPU,
returning a plain dict. It never raises — failures come back as a result dict so
the loop can react to them.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

_KB_SRC = Path(__file__).resolve().parent.parent / "KernelBench" / "src"


def _sanitize(obj):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, BaseException):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return str(obj)


def eval_result_to_dict(result) -> dict:
    if isinstance(result, dict):
        return _sanitize(result)
    if hasattr(result, "model_dump"):
        return _sanitize(result.model_dump())
    if hasattr(result, "dict"):
        return _sanitize(result.dict())
    return _sanitize({
        "compiled": getattr(result, "compiled", False),
        "correctness": getattr(result, "correctness", False),
        "runtime": getattr(result, "runtime", None),
        "ref_runtime": getattr(result, "ref_runtime", None),
        "metadata": getattr(result, "metadata", {}),
        "error": getattr(result, "error", None),
    })


def evaluate_kernel(ref_arch_src: str, code: str, eval_device_id: int, *,
                    num_correct_trials: int = 5, num_perf_trials: int = 10,
                    backend: str = "cuda", precision: str = "fp32",
                    build_root: Path | str | None = None, verbose: bool = False) -> dict:
    """Compile + check + time a kernel. Returns a result dict; never raises."""
    import torch

    if str(_KB_SRC) not in sys.path:
        sys.path.insert(0, str(_KB_SRC))
    from kernelbench import eval as kernel_eval

    device = torch.device(f"cuda:{eval_device_id}")
    kernel_hash = hashlib.md5(code.encode()).hexdigest()[:12]
    base = Path(build_root) if build_root else (Path(__file__).resolve().parent / "build" / "eval_cache")
    build_dir = str(base / kernel_hash)
    shutil.rmtree(build_dir, ignore_errors=True)

    dtype = kernel_eval.get_torch_dtype_from_string(precision)
    try:
        result = kernel_eval.eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=code,
            measure_performance=True,
            timing_method="cuda_event",
            verbose=verbose,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            build_dir=build_dir,
            device=device,
            backend=backend,
            precision=dtype,
        )
        return eval_result_to_dict(result)
    except Exception as e:  # noqa: BLE001
        return {"compiled": False, "correctness": False, "error": str(e),
                "metadata": {"runtime_error": str(e)}}


def _eval_child(q, ref_arch_src, code, eval_device_id, kwargs):
    """Run one evaluation in a child process and ship the result dict back."""
    try:
        q.put(evaluate_kernel(ref_arch_src, code, eval_device_id, **kwargs))
    except Exception as e:  # noqa: BLE001
        q.put({"compiled": False, "correctness": False, "error": str(e),
               "metadata": {"runtime_error": str(e)}})


def evaluate_kernel_safe(ref_arch_src: str, code: str, eval_device_id: int, *,
                         timeout_s: float = 240.0, **kwargs) -> dict:
    """`evaluate_kernel` with a hard wall-clock timeout.

    Generated CUDA can hang at runtime (infinite loops, device deadlocks); a Python
    signal will not interrupt a stuck CUDA call, so we run the evaluation in a child
    process and SIGKILL it if it overruns. A timed-out kernel is reported as a normal
    (compiled?/incorrect) failure instead of wedging the whole worker forever.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_eval_child,
                    args=(q, ref_arch_src, code, eval_device_id, kwargs))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        return {"compiled": False, "correctness": False,
                "error": f"eval timeout after {timeout_s:.0f}s",
                "metadata": {"runtime_error": f"evaluation hung > {timeout_s:.0f}s (killed)",
                             "timeout": True}}
    try:
        return q.get_nowait()
    except Exception:  # noqa: BLE001
        return {"compiled": False, "correctness": False,
                "error": "eval subprocess produced no result",
                "metadata": {"runtime_error": "eval subprocess died without returning"}}


def format_eval_feedback(ev: dict, turn: int) -> str:
    """Human-readable feedback string passed to the Feedback Analyzer / generator."""
    if not ev:
        return "Evaluation failed (internal error)."
    lines = [f"Evaluation result (turn {turn}):"]
    compiled = ev.get("compiled", False)
    correct = ev.get("correctness", False)
    meta = ev.get("metadata") or {}

    if not compiled:
        lines.append("- Compilation: FAILED")
        for key in ("compilation_error", "cuda_error", "runtime_error", "other_error"):
            if meta.get(key):
                lines.append(f"  {key}: {str(meta[key]).strip()[:2000]}")
        if ev.get("error"):
            lines.append(f"  error: {str(ev['error'])[:1000]}")
        return "\n".join(lines)

    lines.append("- Compilation: OK")
    if not correct:
        lines.append("- Correctness: FAILED (output mismatch vs reference)")
        if meta.get("correctness_issue"):
            lines.append(f"  issue: {str(meta['correctness_issue'])[:1500]}")
        if meta.get("runtime_error"):
            lines.append(f"  runtime_error: {str(meta['runtime_error'])[:1500]}")
        return "\n".join(lines)

    lines.append("- Correctness: PASSED")
    rt = ev.get("runtime", -1.0) or -1.0
    ref = ev.get("ref_runtime", -1.0) or -1.0
    if rt > 0 and ref > 0:
        sp = ref / rt
        lines.append(f"- Performance: {rt:.3f} ms vs reference {ref:.3f} ms (speedup {sp:.3f}x)")
        lines.append("  Correct but slower than reference; optimise further."
                     if sp < 1.0 else "  Correct and faster than reference.")
    else:
        lines.append("- Performance: not measured")
    return "\n".join(lines)
