"""Turn eval results into LLM feedback text."""

from typing import Any, Dict, Optional


def format_eval_feedback(eval_dict: Dict[str, Any], turn: int) -> str:
    """Build human-readable feedback from a KernelExecResult dict."""
    lines = ["Evaluation feedback (turn {}):".format(turn)]

    compiled = eval_dict.get("compiled", False)
    correct = eval_dict.get("correctness", False)
    meta = eval_dict.get("metadata") or {}

    if not compiled:
        lines.append("- Compilation: FAILED")
        for key in (
            "compilation_error",
            "cuda_error",
            "runtime_error",
            "other_error",
        ):
            if meta.get(key):
                lines.append("  {}: {}".format(key, _truncate(str(meta[key]), 2000)))
        if meta.get("compilation_error_name"):
            lines.append("  error_type: {}".format(meta["compilation_error_name"]))
        return "\n".join(lines)

    lines.append("- Compilation: OK")

    if not correct:
        lines.append("- Correctness: FAILED (output mismatch vs reference)")
        if meta.get("correctness_issue"):
            lines.append("  issue: {}".format(_truncate(str(meta["correctness_issue"]), 1500)))
        if meta.get("runtime_error"):
            lines.append("  runtime_error: {}".format(_truncate(str(meta["runtime_error"]), 1500)))
        if meta.get("runtime_error_name"):
            lines.append("  error_type: {}".format(meta["runtime_error_name"]))
        return "\n".join(lines)

    lines.append("- Correctness: PASSED")

    runtime = eval_dict.get("runtime", -1.0)
    ref_runtime = eval_dict.get("ref_runtime", -1.0)
    if runtime and runtime > 0 and ref_runtime and ref_runtime > 0:
        speedup = ref_runtime / runtime
        lines.append(
            "- Performance: kernel {:.3f} ms vs reference {:.3f} ms (speedup {:.3f}x)".format(
                runtime, ref_runtime, speedup
            )
        )
        if speedup < 1.0:
            lines.append(
                "  The kernel is correct but slower than PyTorch; optimize further."
            )
        elif speedup >= 1.0:
            lines.append("  The kernel is correct and faster than or equal to reference.")
    elif runtime is not None and runtime <= 0:
        err = meta.get("error_during_performance") or meta.get("runtime_error")
        lines.append("- Performance: timing failed")
        if err:
            lines.append("  error: {}".format(_truncate(str(err), 1500)))
    else:
        lines.append("- Performance: not measured")

    return "\n".join(lines)


def build_refinement_prompt(
    base_prompt: str,
    previous_code: str,
    feedback: str,
    turn: int,
) -> str:
    """Append refinement instructions to the original zero-shot prompt."""
    return """{base_prompt}

---
Iterative refinement (turn {turn}). Your previous submission was tested on a GPU:

{feedback}

Previous submission:
```python
{previous_code}
```

Fix all compilation and correctness errors. If the kernel is already correct but slow, improve performance.
Output only the complete new ModelNew source in one ```python``` code block. No tests or extra text.
""".format(
        base_prompt=base_prompt.rstrip(),
        turn=turn,
        feedback=feedback.strip(),
        previous_code=previous_code.strip(),
    )


def should_stop_early(
    eval_dict: Dict[str, Any],
    target_speedup: Optional[float],
    stop_on_correct: bool = False,
) -> bool:
    """Stop refinement early when criteria are met."""
    if not eval_dict.get("correctness"):
        return False
    if stop_on_correct and target_speedup is None:
        return True
    if target_speedup is None:
        return False
    runtime = eval_dict.get("runtime", -1.0)
    ref_runtime = eval_dict.get("ref_runtime", -1.0)
    if runtime <= 0 or ref_runtime <= 0:
        return False
    return (ref_runtime / runtime) >= target_speedup


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."
