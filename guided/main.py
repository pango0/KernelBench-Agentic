#!/usr/bin/env python3
"""
Guided (profiling-aware) KernelBench: generate once using profiling-enriched prompts.

Same as zero-shot but uses data_guided.json, which injects reference GPU profiling data
into each prompt so the model can focus on the actual bottlenecks.

Runs a local HuggingFace model with multi-GPU batched generation.

Usage (local Qwen):
  python main.py --input ../data_guided.json --output results.json
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def extract_first_code(output_string: str, code_language_types: list[str]) -> str | None:
    if not output_string or not output_string.strip():
        return None

    trimmed = output_string.strip()

    def _strip_lang_header(code: str) -> str:
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type):].strip()
        return code

    for lang in code_language_types:
        match = re.search(rf"```{lang}\s*\n(.*?)```", trimmed, re.DOTALL | re.IGNORECASE)
        if match:
            return _strip_lang_header(match.group(1).strip())

    match = re.search(r"```(.*?)```", trimmed, re.DOTALL)
    if match:
        return _strip_lang_header(match.group(1).strip())

    if "class ModelNew" in trimmed or (
        "import torch" in trimmed and "def get_inputs" in trimmed
    ):
        return trimmed

    return None


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def kernel_filename(level_key: str, problem_id: str, sample_id: int) -> str:
    return f"level_{int(level_key)}_problem_{problem_id}_sample_{sample_id}_kernel.py"


def kernel_path(kernels_dir: Path, level_key: str, problem_id: str, sample_id: int) -> Path:
    return kernels_dir / kernel_filename(level_key, problem_id, sample_id)


def write_kernel_file(
    kernels_dir: Path, level_key: str, problem_id: str, sample_id: int, code: str
) -> Path:
    kernels_dir.mkdir(parents=True, exist_ok=True)
    path = kernel_path(kernels_dir, level_key, problem_id, sample_id)
    path.write_text(code, encoding="utf-8")
    return path


def iter_tasks(data: dict) -> list[tuple[str, str, dict]]:
    tasks = []
    for level_key, problems in data.get("levels", {}).items():
        for pid, entry in problems.items():
            tasks.append((level_key, str(pid), dict(entry)))
    return tasks


# ---------------------------------------------------------------------------
# Local GPU backend
# ---------------------------------------------------------------------------

def prepare_tokenizer(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    import torch

    input_ids_list = []
    for prompt in prompts:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=None,
        )
        input_ids_list.append(ids)

    max_len = max(len(ids) for ids in input_ids_list)
    pad_id = tokenizer.pad_token_id

    padded_ids = []
    attention_mask = []
    for ids in input_ids_list:
        pad_len = max_len - len(ids)
        padded_ids.append([pad_id] * pad_len + ids)
        attention_mask.append([0] * pad_len + [1] * len(ids))

    batch = {
        "input_ids": torch.tensor(padded_ids, device=model.device),
        "attention_mask": torch.tensor(attention_mask, device=model.device, dtype=torch.long),
    }

    with torch.inference_mode():
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )

    decoded = []
    for i in range(len(prompts)):
        new_tokens = outputs[i, max_len:]
        decoded.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return decoded


def chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i: i + size]


def gpu_worker(
    gpu_id: int,
    tasks: list[tuple[str, str, dict]],
    model_name: str,
    max_new_tokens: int,
    dtype_str: str,
    batch_size: int,
    result_queue: mp.Queue,
) -> None:
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype_str]

    print(
        f"[GPU {gpu_id}] Loading {model_name} ({dtype_str}), "
        f"{len(tasks)} tasks, batch_size={batch_size}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prepare_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()

    tasks = sorted(tasks, key=lambda t: len(t[2]["prompt"]))

    for batch_tasks in chunked(tasks, batch_size):
        prompts = [entry["prompt"] for _, _, entry in batch_tasks]
        names = [entry.get("problem_name", "") for _, _, entry in batch_tasks]
        ids_str = ", ".join(f"L{lk}P{pid}" for lk, pid, _ in batch_tasks)
        print(f"[GPU {gpu_id}] batch [{ids_str}] — {names[0]}{'...' if len(names) > 1 else ''}", flush=True)

        try:
            raws = generate_batch(model, tokenizer, prompts, max_new_tokens)
            for (level_key, pid, entry), raw in zip(batch_tasks, raws):
                code = extract_first_code(raw, ["python", "cpp"])
                result_queue.put(
                    (
                        level_key,
                        pid,
                        {
                            **entry,
                            "raw_response": raw,
                            "extracted_code": code,
                            "status": "success" if code else "no_code_extracted",
                            "gpu_id": gpu_id,
                            "generated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                )
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 1:
                raise
            print(f"[GPU {gpu_id}] OOM at batch_size={batch_size}, retrying one-by-one", flush=True)
            for level_key, pid, entry in batch_tasks:
                try:
                    raw = generate_batch(model, tokenizer, [entry["prompt"]], max_new_tokens)[0]
                    code = extract_first_code(raw, ["python", "cpp"])
                    result_queue.put(
                        (
                            level_key,
                            pid,
                            {
                                **entry,
                                "raw_response": raw,
                                "extracted_code": code,
                                "status": "success" if code else "no_code_extracted",
                                "gpu_id": gpu_id,
                                "generated_at": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    )
                except Exception as e:
                    result_queue.put(
                        (
                            level_key,
                            pid,
                            {
                                **entry,
                                "status": "error",
                                "error": str(e),
                                "gpu_id": gpu_id,
                                "generated_at": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    )
        except Exception as e:
            for level_key, pid, entry in batch_tasks:
                result_queue.put(
                    (
                        level_key,
                        pid,
                        {
                            **entry,
                            "status": "error",
                            "error": str(e),
                            "gpu_id": gpu_id,
                            "generated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                )
            print(f"[GPU {gpu_id}] batch ERROR: {e}", flush=True)

    print(f"[GPU {gpu_id}] Done.", flush=True)


def shard_tasks(
    tasks: list[tuple[str, str, dict]], num_gpus: int
) -> list[list[tuple[str, str, dict]]]:
    shards: list[list[tuple[str, str, dict]]] = [[] for _ in range(num_gpus)]
    for i, task in enumerate(tasks):
        shards[i % num_gpus].append(task)
    return shards


def build_results_skeleton(data: dict, args: argparse.Namespace) -> dict:
    results: dict[str, Any] = {
        "meta": {
            **data.get("meta", {}),
            "method": "guided",
            "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            "input_file": str(args.input),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "dtype": args.dtype,
            "num_gpus": args.num_gpus,
            "batch_size": args.batch_size,
        },
        "levels": {},
    }
    for level_key, problems in data.get("levels", {}).items():
        results["levels"][level_key] = {}
        for pid, entry in problems.items():
            results["levels"][level_key][str(pid)] = dict(entry)
    return results


def filter_tasks_for_resume(
    tasks: list[tuple[str, str, dict]],
    results: dict,
    kernels_dir: Path | None,
    sample_id: int,
) -> list[tuple[str, str, dict]]:
    pending = []
    for level_key, pid, entry in tasks:
        out = results.get("levels", {}).get(level_key, {}).get(pid, {})
        if out.get("status") == "success":
            continue
        if kernels_dir is not None:
            kp = kernel_path(kernels_dir, level_key, pid, sample_id)
            if kp.exists() and kp.stat().st_size > 0:
                continue
        pending.append((level_key, pid, entry))
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data_guided.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results.json",
    )
    parser.add_argument(
        "--kernels-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "guided",
        help="Directory for level_*_problem_*_sample_*_kernel.py files",
    )
    parser.add_argument("--sample-id", type=int, default=0)

    # Model selection
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")

    # Local model options
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if args.batch_size < 1:
        print("--batch-size must be >= 1", file=sys.stderr)
        sys.exit(1)

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    data = load_json(args.input)

    if args.resume and args.output.exists():
        results = load_json(args.output)
    else:
        results = build_results_skeleton(data, args)

    results["meta"]["kernels_dir"] = str(args.kernels_dir)
    results["meta"]["sample_id"] = args.sample_id

    all_tasks = iter_tasks(data)
    if args.resume:
        all_tasks = filter_tasks_for_resume(
            all_tasks, results, args.kernels_dir, args.sample_id
        )

    if not all_tasks:
        print("No tasks to run (all done?).")
        save_results(args.output, results)
        return

    def _save(level_key: str, pid: str, out_entry: dict) -> None:
        code = out_entry.get("extracted_code")
        if code and out_entry.get("status") == "success":
            kp = write_kernel_file(args.kernels_dir, level_key, pid, args.sample_id, code)
            out_entry["kernel_path"] = str(kp)
        results["levels"].setdefault(level_key, {})[pid] = out_entry
        save_results(args.output, results)
        kn = f" -> {out_entry.get('kernel_path', '')}" if out_entry.get("kernel_path") else ""
        print(f"Saved L{level_key} P{pid} ({out_entry.get('status')}){kn}")

    # -----------------------------------------------------------------------
    # Local GPU path
    # -----------------------------------------------------------------------
    import torch

    num_gpus = args.num_gpus or torch.cuda.device_count()
    if num_gpus < 1:
        print("No CUDA GPUs available.", file=sys.stderr)
        sys.exit(1)

    results["meta"]["num_gpus"] = num_gpus
    results["meta"]["batch_size"] = args.batch_size

    shards = shard_tasks(all_tasks, num_gpus)
    for gpu_id, shard in enumerate(shards):
        print(f"GPU {gpu_id}: {len(shard)} problems")

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    processes = []

    for gpu_id, shard in enumerate(shards):
        if not shard:
            continue
        p = ctx.Process(
            target=gpu_worker,
            args=(gpu_id, shard, args.model, args.max_new_tokens, args.dtype, args.batch_size, result_queue),
        )
        p.start()
        processes.append(p)

    completed_count = 0
    total = len(all_tasks)
    while completed_count < total:
        level_key, pid, out_entry = result_queue.get()
        _save(level_key, pid, out_entry)
        completed_count += 1
        print(f"Progress: {completed_count}/{total}", flush=True)

    for p in processes:
        p.join()

    results["meta"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    results["meta"]["output_file"] = str(args.output)
    save_results(args.output, results)
    print(f"Done. Wrote {args.output}")
    print(f"Kernels in {args.kernels_dir}/level_*_problem_*_sample_{args.sample_id}_kernel.py")


if __name__ == "__main__":
    main()
