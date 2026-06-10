import torch
import os
import json
from tqdm import tqdm
from multiprocessing import Process

from kernelbench.dataset import (
    construct_kernelbench_dataset,
    fetch_ref_arch_from_dataset,
)
from kernelbench.timing import measure_ref_program_time

REPO_TOP_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

TIMING_DIR = os.path.join(REPO_TOP_PATH, "results", "timing")
hardware_name = "V100_SXM2_32GB"

NUM_PROBLEMS_PER_LEVEL = int(os.environ.get("NUM_PROBLEMS_PER_LEVEL", "10"))


def run_level(level: int, gpu_id: int, output_dir: str):
    """
    Run first N problems from one level on one GPU.
    """
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    precision = "none" if level == 4 else "fp32"
    results = {f"level{level}": {}}

    dataset = construct_kernelbench_dataset(level)

    # Only take first N problems
    problem_ids = dataset.get_problem_ids()[:NUM_PROBLEMS_PER_LEVEL]

    for problem_id in tqdm(
        problem_ids,
        desc=f"Level {level} GPU {gpu_id}",
    ):
        (
            ref_arch_path,
            ref_arch_name,
            ref_arch_src,
        ) = fetch_ref_arch_from_dataset(dataset, problem_id)

        runtime_stats = measure_ref_program_time(
            ref_arch_name=ref_arch_name,
            ref_arch_src=ref_arch_src,
            use_torch_compile=False,
            torch_compile_backend=None,
            torch_compile_options=None,
            device=device,
            verbose=False,
            precision=precision,
        )

        results[f"level{level}"][ref_arch_name] = runtime_stats

    level_save_path = os.path.join(
        output_dir,
        f"baseline_time_torch_level{level}.json",
    )

    with open(level_save_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[GPU {gpu_id}] Saved level {level} -> {level_save_path}")


def merge_results(output_dir: str):
    """
    Merge all level JSON files into one final JSON.
    """
    merged_results = {}

    for level in range(1, 5):
        level_path = os.path.join(
            output_dir,
            f"baseline_time_torch_level{level}.json",
        )

        with open(level_path, "r") as f:
            level_data = json.load(f)

        merged_results.update(level_data)

    final_save_path = os.path.join(
        output_dir,
        "baseline_time_torch.json",
    )

    with open(final_save_path, "w") as f:
        json.dump(merged_results, f, indent=4)

    print(f"Merged results saved to {final_save_path}")


if __name__ == "__main__":
    output_dir = os.path.join(
        TIMING_DIR,
        hardware_name,
    )

    os.makedirs(output_dir, exist_ok=True)

    # One level per GPU
    gpu_assignments = {
        1: 0,
        2: 1,
        3: 2,
        # 4: 3,
    }

    processes = []

    for level, gpu_id in gpu_assignments.items():
        p = Process(
            target=run_level,
            args=(level, gpu_id, output_dir),
        )

        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    merge_results(output_dir)