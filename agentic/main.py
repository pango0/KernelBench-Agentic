#!/usr/bin/env python3
"""
Agentic KernelBench: a multi-agent loop on a single local model.

    PyTorch/HIP kernel -> Code Analyzer -> RAG Researcher <-> Documentation
                                  |              |
                                  v              v
                          Kernel Generator <-----+
                                  |
                              Evaluator -> Feedback Analyzer -> Data collector
                                  ^                  |
                                  +----- loop -------+

Every agent box is the same local HuggingFace model in a different role (see
agents.py); the Evaluator runs the KernelBench harness; the Data collector logs
post-training trajectories. Each box can be toggled to produce ablations.

Usage (full system, local Qwen):
  python main.py --input ../data.json --output results.json --collect-data

Ablations (one switch each):
  python main.py ... --no-rag
  python main.py ... --no-code-analyzer
  python main.py ... --no-feedback-analyzer
  python main.py ... --max-turns 1            # single shot, no loop
  python main.py ... --best-of-n 2            # best-of-n sampling per turn

See docs/AGENTIC_METHOD.md for the full method description.
"""

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))        # agentic/ on path (for workers)

import pipeline as P  # noqa: E402
from datacollector import PostTrainingCollector  # noqa: E402

DEFAULT_DOCS_DIR = Path(__file__).resolve().parent / "docs"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
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


def save_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(data), f, indent=2, ensure_ascii=False)


def kernel_path(kernels_dir, level_key, pid, sample_id):
    return Path(kernels_dir) / f"level_{level_key}_problem_{pid}_sample_{sample_id}_kernel.py"


def write_kernel_file(kernels_dir, level_key, pid, sample_id, code):
    kp = kernel_path(kernels_dir, level_key, pid, sample_id)
    kp.parent.mkdir(parents=True, exist_ok=True)
    kp.write_text(code, encoding="utf-8")
    return kp


def iter_tasks(data):
    for level_key, problems in data["levels"].items():
        for pid, entry in problems.items():
            yield level_key, pid, entry


# ---------------------------------------------------------------------------
# GPU worker layout
# ---------------------------------------------------------------------------

def _build_worker_plans(total_gpus: int, gpus_per_worker: int) -> list:
    if total_gpus < gpus_per_worker:
        raise ValueError(f"Need >= {gpus_per_worker} GPUs, got {total_gpus}")
    plans = []
    for wid in range(total_gpus // gpus_per_worker):
        base = wid * gpus_per_worker
        physical = list(range(base, base + gpus_per_worker))
        plans.append({
            "worker_id": wid,
            "cuda_visible_devices": ",".join(str(i) for i in physical),
            "llm_device_id": 0,
            "eval_device_id": 1 if gpus_per_worker >= 2 else 0,
        })
    return plans


def _shard_tasks(tasks: list, n: int) -> list:
    shards = [[] for _ in range(n)]
    for i, t in enumerate(tasks):
        shards[i % n].append(t)
    return shards


def _config_from_args(args) -> "P.AgenticConfig":
    return P.AgenticConfig(
        max_turns=args.max_turns,
        use_code_analyzer=args.use_code_analyzer,
        use_rag=args.use_rag,
        use_feedback_analyzer=args.use_feedback_analyzer,
        best_of_n=args.best_of_n,
        rag_top_k=args.rag_top_k,
        stop_on_correct=args.stop_on_correct,
        target_speedup=args.target_speedup,
        max_new_tokens=args.max_new_tokens,
        collect_data=args.collect_data,
        fallback_to_reference=args.fallback_to_reference,
    )


def local_gpu_worker(worker_plan: dict, tasks: list, args_dict: dict, result_queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = worker_plan["cuda_visible_devices"]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llm import LocalLLM
    from rag import build_retriever
    from evaluator import evaluate_kernel_safe

    args = argparse.Namespace(**args_dict)
    cfg = _config_from_args(args)
    eval_device_id = worker_plan["eval_device_id"]
    wid = worker_plan["worker_id"]

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    print(f"[worker {wid}] CUDA_VISIBLE_DEVICES={worker_plan['cuda_visible_devices']} | "
          f"LLM cuda:{worker_plan['llm_device_id']} eval cuda:{eval_device_id}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype_map[args.dtype],
        device_map={"": worker_plan["llm_device_id"]}, trust_remote_code=True,
    )
    model.eval()
    llm = LocalLLM(model, tokenizer, default_max_new_tokens=args.max_new_tokens)

    retriever = None
    if cfg.use_rag:
        retriever = build_retriever(args.docs_dir, backend=args.rag_backend)
        n = len(getattr(retriever, "chunks", []) or [])
        print(f"[worker {wid}] RAG corpus: {n} chunks from {args.docs_dir} "
              f"(backend={args.rag_backend})", flush=True)

    def evaluate(ref_src, code):
        return evaluate_kernel_safe(
            ref_src, code, eval_device_id,
            timeout_s=args.eval_timeout,
            num_correct_trials=args.num_correct_trials,
            num_perf_trials=args.num_perf_trials,
            backend=args.backend, precision=args.precision,
            build_root=Path(__file__).resolve().parent / "build" / "eval_cache",
            verbose=args.eval_verbose,
        )

    print(f"[worker {wid}] ready ({len(tasks)} tasks)", flush=True)
    for level_key, pid, entry in tasks:
        try:
            out = P.run_problem(llm, evaluate, retriever, level_key, pid, entry, cfg)
            out["worker_id"] = wid
        except Exception as e:  # noqa: BLE001
            out = {**entry, "status": "error", "error": str(e)}
        result_queue.put((level_key, pid, out))
    print(f"[worker {wid}] done", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results.json")
    parser.add_argument("--kernels-dir", type=Path,
                        default=Path(__file__).resolve().parent / "agentic")
    parser.add_argument("--sample-id", type=int, default=0)

    # Model (local only)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--gpus-per-worker", type=int, default=2,
                        help="1 GPU = LLM+eval shared; 2 = LLM on cuda:0, eval on cuda:1")

    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-turns", type=int, default=3)

    # Agent toggles (ablation switches) — all agents on by default
    parser.add_argument("--no-code-analyzer", dest="use_code_analyzer",
                        action="store_false", default=True)
    parser.add_argument("--no-rag", dest="use_rag", action="store_false", default=True)
    parser.add_argument("--no-feedback-analyzer", dest="use_feedback_analyzer",
                        action="store_false", default=True)
    parser.add_argument("--best-of-n", type=int, default=1,
                        help="candidates generated and evaluated per turn (keep the best)")

    # RAG
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--rag-top-k", type=int, default=5)
    parser.add_argument("--rag-backend", default="bm25", choices=["bm25", "embed"])

    # Post-training data collection
    parser.add_argument("--collect-data", action="store_true")
    parser.add_argument("--post-training-dir", type=Path, default=None,
                        help="default: <output dir>/post_training")

    # Eval
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=10)
    parser.add_argument("--eval-timeout", type=float, default=240.0,
                        help="hard per-kernel eval timeout (s); a hung CUDA kernel is "
                             "killed and scored as a failure instead of wedging the run")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--eval-verbose", action="store_true")

    # Control
    parser.add_argument("--stop-on-correct", action="store_true")
    parser.add_argument("--target-speedup", type=float, default=None)
    parser.add_argument("--fallback-to-reference", action="store_true",
                        help="if no turn compiles, ship ModelNew = Model (correct ~1x) "
                             "as a deployment floor; off by default so passthroughs do "
                             "not inflate benchmark correctness")

    args = parser.parse_args()
    args.docs_dir = str(args.docs_dir)

    if not args.input.exists():
        print(f"missing input: {args.input}", file=sys.stderr)
        sys.exit(1)

    data = load_json(args.input)
    tasks = list(iter_tasks(data))
    print(f"Loaded {len(tasks)} tasks from {args.input}")

    collector = None
    if args.collect_data:
        pt_dir = args.post_training_dir or (args.output.parent / "post_training")
        collector = PostTrainingCollector(pt_dir)
        print(f"Post-training data -> {pt_dir}")

    results: dict = {
        "meta": {
            "method": "agentic",
            "model": args.model,
            "max_turns": args.max_turns,
            "agents": {
                "code_analyzer": args.use_code_analyzer,
                "rag_researcher": args.use_rag,
                "feedback_analyzer": args.use_feedback_analyzer,
            },
            "best_of_n": args.best_of_n,
            "rag_backend": args.rag_backend if args.use_rag else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "levels": {},
    }

    def _save(level_key, pid, out_entry):
        traj = out_entry.pop("_post_training", None)
        if collector is not None and traj is not None:
            collector.add(traj)
        code = out_entry.get("extracted_code")
        if code:
            kp = write_kernel_file(args.kernels_dir, level_key, pid, args.sample_id, code)
            out_entry["kernel_path"] = str(kp)
        results["levels"].setdefault(level_key, {})[pid] = out_entry
        save_json(args.output, results)
        status = out_entry.get("status", "?")
        kp = out_entry.get("kernel_path", "")
        print(f"  saved L{level_key} P{pid} ({status})" + (f" -> {kp}" if kp else ""), flush=True)

    import torch

    available = torch.cuda.device_count()
    total_gpus = min(args.num_gpus if args.num_gpus is not None else available, available)
    if total_gpus < 1:
        print("No CUDA GPUs available.", file=sys.stderr)
        sys.exit(1)

    worker_plans = _build_worker_plans(total_gpus, args.gpus_per_worker)
    num_workers = len(worker_plans)
    print(f"Using {total_gpus} GPUs, {num_workers} workers ({args.gpus_per_worker} GPUs/worker)")
    results["meta"].update({"total_gpus": total_gpus, "gpus_per_worker": args.gpus_per_worker,
                            "num_workers": num_workers})

    shards = _shard_tasks(tasks, num_workers)
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    args_dict = vars(args)
    processes = []
    for plan, shard in zip(worker_plans, shards):
        if not shard:
            continue
        p = ctx.Process(target=local_gpu_worker, args=(plan, shard, args_dict, result_queue))
        p.start()
        processes.append(p)

    completed, total = 0, len(tasks)
    while completed < total:
        try:
            level_key, pid, out_entry = result_queue.get(timeout=120)
        except queue.Empty:
            if not any(p.is_alive() for p in processes):
                break
            continue
        _save(level_key, pid, out_entry)
        completed += 1
        print(f"Progress: {completed}/{total}", flush=True)

    for p in processes:
        p.join()

    results["meta"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    if collector is not None:
        results["meta"]["post_training_counts"] = collector.summary()
    save_json(args.output, results)
    print(f"\nDone. Results -> {args.output}")
    print(f"Kernels -> {args.kernels_dir}/")
    if collector is not None:
        print(f"Post-training records: {collector.summary()}")


if __name__ == "__main__":
    main()
