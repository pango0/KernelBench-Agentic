#!/usr/bin/env python3
"""
Iterative KernelBench: generate → evaluate → feedback → regenerate.

Multi-turn refinement loop where each turn uses actual GPU eval results as feedback.
Runs a local HuggingFace model (multi-GPU, 2 GPUs/worker: LLM + eval).

Usage (local Qwen):
  python main.py --input ../data.json --output results.json
"""

import argparse
import json
import multiprocessing as mp
import os
import queue
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# IMPORTANT: do NOT import torch globally; CUDA must initialize inside each spawned process.


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_for_json(obj):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, BaseException):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return str(obj)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(data), f, indent=2, ensure_ascii=False)


def iter_tasks(data):
    for level_key, problems in data["levels"].items():
        for pid, entry in problems.items():
            yield level_key, pid, entry


def extract_first_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(.*?)```", text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        for hdr in ("python", "cpp"):
            if code.startswith(hdr):
                code = code[len(hdr):].strip()
        return code
    if "class ModelNew" in text or ("import torch" in text and "def get_inputs" in text):
        return text.strip()
    return None


def kernel_path(kernels_dir, level_key, pid, sample_id):
    return (
        Path(kernels_dir)
        / f"level_{level_key}_problem_{pid}_sample_{sample_id}_kernel.py"
    )


def write_kernel_file(kernels_dir, level_key, pid, sample_id, code):
    kp = kernel_path(kernels_dir, level_key, pid, sample_id)
    kp.parent.mkdir(parents=True, exist_ok=True)
    with open(kp, "w", encoding="utf-8") as f:
        f.write(code)
    return kp


def export_prompts_for_problem(entry, prompts_dir, level_key, pid):
    if prompts_dir is None:
        return []
    prompts_dir = Path(prompts_dir)
    written = []
    problem_dir = prompts_dir / "level_{}".format(level_key) / "problem_{}".format(pid)
    problem_dir.mkdir(parents=True, exist_ok=True)

    base = entry.get("base_prompt") or entry.get("prompt", "")
    if base:
        p = problem_dir / "turn_0_base.txt"
        if not p.exists() or p.stat().st_size == 0:
            p.write_text(base, encoding="utf-8")
        written.append(str(p))

    for turn in entry.get("turns", []):
        tnum = turn.get("turn", 0)
        prompt = turn.get("prompt")
        if not prompt:
            continue
        path = problem_dir / "turn_{}_prompt.txt".format(tnum)
        path.write_text(prompt, encoding="utf-8")
        written.append(str(path))
        feedback = turn.get("feedback")
        if feedback:
            fb_path = problem_dir / "turn_{}_feedback.txt".format(tnum)
            fb_path.write_text(feedback, encoding="utf-8")
            written.append(str(fb_path))

    entry["prompt_export_dir"] = str(problem_dir)
    return written


def print_prompts_for_problem(level_key, pid, entry):
    print("\n" + "=" * 80, flush=True)
    print("LEVEL {} PROBLEM {} | status={}".format(level_key, pid, entry.get("status", "?")), flush=True)
    print("=" * 80, flush=True)
    for turn in entry.get("turns", []):
        tnum = turn.get("turn", 0)
        print("\n--- Turn {} (status={}) ---\n".format(tnum, turn.get("status", "?")), flush=True)
        print(turn.get("prompt", "(no prompt stored)"), flush=True)
        if turn.get("feedback"):
            print("\n--- Turn {} feedback ---\n".format(tnum), flush=True)
            print(turn["feedback"], flush=True)
    if entry.get("extracted_code"):
        print("\n--- Best kernel (turn {}) ---\n".format(entry.get("final_turn", "?")), flush=True)
        code = entry["extracted_code"]
        print(code[:2000] + ("..." if len(code) > 2000 else ""), flush=True)


# ---------------------------------------------------------------------------
# Local model helpers
# ---------------------------------------------------------------------------

def prepare_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def generate_one(model, tokenizer, prompt, max_new_tokens):
    import torch

    device = next(model.parameters()).device
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors=None,
    )
    batch = {
        "input_ids": torch.tensor([input_ids], device=device),
        "attention_mask": torch.ones(1, len(input_ids), device=device, dtype=torch.long),
    }
    with torch.inference_mode():
        out = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0, len(input_ids):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Eval helpers (shared)
# ---------------------------------------------------------------------------

def _setup_kernelbench():
    KB_SRC = Path(__file__).resolve().parent.parent / "KernelBench" / "src"
    if str(KB_SRC) not in sys.path:
        sys.path.insert(0, str(KB_SRC))


def _turn_rank(t):
    """Rank a turn by (correct, compiled, speedup) so a later turn never demotes an
    earlier compiling/correct one. Previously the fallback returned the *last* coded
    turn, which could discard a turn that compiled for one that did not — dropping the
    method's compile rate below plain zero-shot."""
    ev = t.get("eval") or {}
    correct = bool(ev.get("correctness"))
    compiled = bool(ev.get("compiled"))
    rt = ev.get("runtime", -1.0) or -1.0
    ref = ev.get("ref_runtime", -1.0) or -1.0
    speedup = (ref / rt) if (correct and rt > 0 and ref > 0) else 0.0
    # turn index is the final tiebreaker so equal-rank turns resolve to the latest
    # attempt (the model's most-refined output), preserving the old fallback semantics.
    return (2 if correct else (1 if compiled else 0), speedup, t.get("turn", 0))


def pick_best_turn(turns):
    coded = [t for t in turns if t.get("extracted_code")]
    if coded:
        # max() keeps the earliest turn on ties, so a correct/compiling result is not
        # silently replaced by a later regression of equal rank.
        return max(coded, key=_turn_rank)
    return turns[-1] if turns else None


def build_refinement_prompt(base_prompt, previous_code, feedback, turn_idx):
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
        turn=turn_idx,
        feedback=feedback.strip(),
        previous_code=previous_code.strip(),
    )


def format_eval_feedback(ev_dict, turn):
    if not ev_dict:
        return "Evaluation failed."
    lines = [
        f"turn={turn}",
        f"compiled={ev_dict.get('compiled')}",
        f"correctness={ev_dict.get('correctness')}",
    ]
    if ev_dict.get("runtime") is not None:
        lines.append(f"runtime_ms={ev_dict.get('runtime')}")
    if ev_dict.get("ref_runtime") is not None:
        lines.append(f"ref_runtime_ms={ev_dict.get('ref_runtime')}")
    rt = ev_dict.get("runtime")
    ref = ev_dict.get("ref_runtime")
    if rt and ref and rt > 0:
        lines.append(f"speedup={ref / rt:.4f}")
    if ev_dict.get("error"):
        lines.append(f"error={ev_dict.get('error')}")
    meta = ev_dict.get("metadata") or {}
    for key in ("compilation_error", "runtime_error", "cuda_error"):
        if meta.get(key):
            lines.append(f"{key}={str(meta[key])[:1500]}")
    return "\n".join(lines)


def should_stop_early(ev_dict, target_speedup=None, stop_on_correct=False):
    if not ev_dict:
        return False
    if stop_on_correct and ev_dict.get("correctness"):
        return True
    if target_speedup is not None and ev_dict.get("correctness"):
        rt = ev_dict.get("runtime")
        ref = ev_dict.get("ref_runtime")
        if rt and ref and rt > 0:
            return (ref / rt) >= target_speedup
    return False


def eval_result_to_dict(eval_result):
    if isinstance(eval_result, dict):
        return sanitize_for_json(eval_result)
    if hasattr(eval_result, "model_dump"):
        return sanitize_for_json(eval_result.model_dump())
    if hasattr(eval_result, "dict"):
        return sanitize_for_json(eval_result.dict())
    return sanitize_for_json(
        {
            "compiled": getattr(eval_result, "compiled", False),
            "correctness": getattr(eval_result, "correctness", False),
            "runtime": getattr(eval_result, "runtime", None),
            "ref_runtime": getattr(eval_result, "ref_runtime", None),
            "metadata": getattr(eval_result, "metadata", {}),
            "error": getattr(eval_result, "error", None),
        }
    )


def read_ref_arch(entry):
    path = entry.get("reference_arch_path") or entry.get("ref_arch_path")
    if not path:
        raise KeyError("reference_arch_path or ref_arch_path missing in data entry")
    return Path(path).read_text(encoding="utf-8")


def evaluate_kernel_inline(ref_arch_src, code, eval_device_id, args):
    """Run KernelBench eval inline. Returns a dict. Never raises."""
    import hashlib, shutil, torch

    _setup_kernelbench()
    from kernelbench import eval as kernel_eval

    device = torch.device("cuda:{}".format(eval_device_id))
    kernel_hash = hashlib.md5(code.encode()).hexdigest()[:12]
    build_dir = str(
        Path(__file__).resolve().parent / "build" / "eval_cache" / kernel_hash
    )
    shutil.rmtree(build_dir, ignore_errors=True)

    dtype = kernel_eval.get_torch_dtype_from_string(args.precision)
    try:
        result = kernel_eval.eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=code,
            measure_performance=True,
            timing_method="cuda_event",
            verbose=args.eval_verbose,
            num_correct_trials=args.num_correct_trials,
            num_perf_trials=args.num_perf_trials,
            build_dir=build_dir,
            device=device,
            backend=args.backend,
            precision=dtype,
        )
        return eval_result_to_dict(result)
    except Exception as e:
        meta = {"runtime_error": str(e)}
        return {"compiled": False, "correctness": False, "error": str(e), "metadata": meta}


# ---------------------------------------------------------------------------
# Core refinement loop (generate_fn is callable(prompt) -> str)
# ---------------------------------------------------------------------------

def run_problem(generate_fn, level_key, pid, entry, args):
    base_prompt = entry["prompt"]
    ref_arch_src = read_ref_arch(entry)
    turns = []
    prompt = base_prompt
    previous_code = ""

    for turn in range(args.max_turns):
        turn_record = {
            "turn": turn,
            "status": "pending",
            "prompt": prompt,
        }
        print(f"L{level_key} P{pid} turn {turn + 1}/{args.max_turns}", flush=True)

        try:
            raw = generate_fn(prompt)
            code = extract_first_code(raw)
            turn_record["raw_response"] = raw
            turn_record["extracted_code"] = code

            if not code:
                turn_record["status"] = "no_code"
                turns.append(turn_record)
                if turn < args.max_turns - 1:
                    prompt = build_refinement_prompt(
                        base_prompt,
                        previous_code or "# empty",
                        "No valid code extracted.",
                        turn + 1,
                    )
                continue

            previous_code = code

            ev_dict = evaluate_kernel_inline(ref_arch_src, code, args.eval_device_id, args)
            turn_record["eval"] = ev_dict
            turn_record["feedback"] = format_eval_feedback(ev_dict, turn)

            if not ev_dict.get("compiled"):
                turn_record["status"] = "compile_failed"
            elif not ev_dict.get("correctness"):
                turn_record["status"] = "incorrect"
            else:
                turn_record["status"] = "correct"

            turns.append(turn_record)

            import torch
            torch.cuda.empty_cache()

            if should_stop_early(ev_dict, args.target_speedup, args.stop_on_correct):
                print(f"L{level_key} P{pid} early stop", flush=True)
                break

            if turn < args.max_turns - 1:
                prompt = build_refinement_prompt(
                    base_prompt,
                    code,
                    turn_record["feedback"],
                    turn + 1,
                )

        except Exception as e:
            turn_record["status"] = "error"
            turn_record["error"] = str(e)
            turns.append(turn_record)
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    best = pick_best_turn(turns)
    out = {
        **entry,
        "base_prompt": base_prompt,
        "turns": turns,
        "num_turns": len(turns),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if best:
        out["final_turn"] = best.get("turn")
        out["final_status"] = best.get("status")
        out["extracted_code"] = best.get("extracted_code")
        out["final_eval"] = best.get("eval")
        out["final_prompt"] = best.get("prompt")
    any_correct = any((t.get("eval") or {}).get("correctness") for t in turns)
    if best and best.get("extracted_code"):
        out["status"] = "success" if any_correct else "best_effort"
    else:
        out["status"] = "failed"
    return out


# ---------------------------------------------------------------------------
# Local GPU worker
# ---------------------------------------------------------------------------

def build_worker_plans(total_gpus, gpus_per_worker):
    if total_gpus < gpus_per_worker:
        raise ValueError("need at least {} GPUs, got {}".format(gpus_per_worker, total_gpus))
    num_workers = total_gpus // gpus_per_worker
    remainder = total_gpus % gpus_per_worker
    if remainder:
        print("Warning: {} GPUs unused (total={}, per_worker={})".format(
            remainder, total_gpus, gpus_per_worker), flush=True)
    plans = []
    for worker_id in range(num_workers):
        base = worker_id * gpus_per_worker
        physical_ids = list(range(base, base + gpus_per_worker))
        plans.append(
            {
                "worker_id": worker_id,
                "cuda_visible_devices": ",".join(str(i) for i in physical_ids),
                "physical_gpu_ids": physical_ids,
                "llm_device_id": 0,
                "eval_device_id": 1 if gpus_per_worker >= 2 else 0,
            }
        )
    return plans


def gpu_worker(worker_plan, tasks, args_dict, result_queue):
    visible = worker_plan["cuda_visible_devices"]
    os.environ["CUDA_VISIBLE_DEVICES"] = visible

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = argparse.Namespace(**args_dict)
    args.llm_device_id = worker_plan["llm_device_id"]
    args.eval_device_id = worker_plan["eval_device_id"]

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    wid = worker_plan["worker_id"]
    print("[worker {}] CUDA_VISIBLE_DEVICES={} | LLM cuda:{} eval cuda:{}".format(
        wid, visible, args.llm_device_id, args.eval_device_id), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prepare_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map={"": args.llm_device_id},
        trust_remote_code=True,
    )
    model.eval()
    print("[worker {}] ready ({} tasks)".format(wid, len(tasks)), flush=True)

    for level_key, pid, entry in tasks:
        try:
            generate_fn = lambda p: generate_one(model, tokenizer, p, args.max_new_tokens)
            out_entry = run_problem(generate_fn, level_key, pid, entry, args)
            out_entry["worker_id"] = wid
            out_entry["cuda_visible_devices"] = visible
            out_entry["llm_device"] = "cuda:{}".format(args.llm_device_id)
            out_entry["eval_device"] = "cuda:{}".format(args.eval_device_id)
            result_queue.put((level_key, pid, out_entry))
        except Exception as e:
            result_queue.put(
                (level_key, pid, {**entry, "status": "error", "error": str(e)})
            )

    print("[worker {}] done".format(wid), flush=True)


def shard_tasks(tasks, num_gpus):
    shards = [[] for _ in range(num_gpus)]
    for i, task in enumerate(tasks):
        shards[i % num_gpus].append(task)
    return shards


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_results_skeleton(args, worker_plans=None):
    return {
        "meta": {
            "method": "iterative",
            "model": args.model,
            "max_turns": args.max_turns,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "levels": {},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="results.json")
    parser.add_argument("--kernels-dir", type=Path, default="iterative")
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default="prompts",
        help="Write per-turn prompt .txt files here",
    )
    parser.add_argument("--export-best-kernels", action="store_true", default=True)
    parser.add_argument("--no-export-best-kernels", action="store_false", dest="export_best_kernels")
    parser.add_argument("--print-prompts", action="store_true")
    parser.add_argument("--sample-id", type=int, default=0)

    # Model selection
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")

    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--dtype", default="float16")

    # Local GPU layout
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--gpus-per-worker", type=int, default=2)

    # Eval options
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=30)
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--eval-verbose", action="store_true")

    # Control
    parser.add_argument("--stop-on-correct", action="store_true")
    parser.add_argument("--target-speedup", type=float, default=None)

    args = parser.parse_args()

    if not args.input.exists():
        print(f"missing input: {args.input}", file=sys.stderr)
        sys.exit(1)

    data = load_json(args.input)
    results = build_results_skeleton(args)
    tasks = list(iter_tasks(data))

    def _save(level_key, pid, out_entry):
        code = out_entry.get("extracted_code")
        write_kernel = code and (
            args.export_best_kernels or out_entry.get("status") == "success"
        )
        if write_kernel:
            kp = write_kernel_file(args.kernels_dir, level_key, pid, args.sample_id, code)
            out_entry["kernel_path"] = str(kp)
        export_prompts_for_problem(out_entry, args.prompts_dir, level_key, pid)
        if args.print_prompts:
            print_prompts_for_problem(level_key, pid, out_entry)
        results["levels"].setdefault(level_key, {})[pid] = out_entry
        save_json(args.output, results)
        print(f"saved L{level_key} P{pid} ({out_entry.get('status', '?')})", flush=True)

    # -----------------------------------------------------------------------
    # Local GPU path
    # -----------------------------------------------------------------------
    import torch

    available = torch.cuda.device_count()
    total_gpus = args.num_gpus if args.num_gpus is not None else available
    total_gpus = min(total_gpus, available)
    if total_gpus < 1:
        print("No CUDA GPUs available.", file=sys.stderr)
        sys.exit(1)

    worker_plans = build_worker_plans(total_gpus, args.gpus_per_worker)
    num_workers = len(worker_plans)
    print("Using {} GPUs, {} workers ({} GPUs/worker)".format(
        total_gpus, num_workers, args.gpus_per_worker), flush=True)
    for plan in worker_plans:
        print("  worker {}: devices {} -> LLM cuda:{}, eval cuda:{}".format(
            plan["worker_id"], plan["cuda_visible_devices"],
            plan["llm_device_id"], plan["eval_device_id"]), flush=True)

    results["meta"].update({
        "total_gpus": total_gpus,
        "gpus_per_worker": args.gpus_per_worker,
        "num_workers": num_workers,
        "worker_gpu_map": worker_plans,
    })

    shards = shard_tasks(tasks, num_workers)
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    args_dict = vars(args)

    for plan, shard in zip(worker_plans, shards):
        if not shard:
            continue
        p = ctx.Process(target=gpu_worker, args=(plan, shard, args_dict, result_queue))
        p.start()
        processes.append(p)

    completed_count = 0
    total = len(tasks)
    while completed_count < total:
        try:
            level_key, pid, out_entry = result_queue.get(timeout=120)
        except queue.Empty:
            if not any(p.is_alive() for p in processes):
                break
            continue
        _save(level_key, pid, out_entry)
        completed_count += 1
        print(f"Progress: {completed_count}/{total}", flush=True)

    for p in processes:
        p.join()

    results["meta"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_json(args.output, results)
    print("done")


if __name__ == "__main__":
    main()
