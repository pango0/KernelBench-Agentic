# scripts/profile_reference.py

import torch
import json
import os
import sys

from tqdm import tqdm

from kernelbench.dataset import (
    construct_kernelbench_dataset,
    fetch_ref_arch_from_dataset,
)

import os as _os
NUM_PROBLEMS = int(_os.environ.get("NUM_PROBLEMS_PER_LEVEL", "10"))


def profile_with_torch_profiler(
    ref_arch_src,
    device,
    precision="fp32",
):
    """Profile reference model using torch.profiler"""

    namespace = {}
    exec(ref_arch_src, namespace)

    Model = namespace["Model"]
    get_inputs = namespace["get_inputs"]
    get_init_inputs = namespace.get(
        "get_init_inputs",
        lambda: [],
    )

    init_inputs = get_init_inputs()

    model = Model(*init_inputs).to(device)

    inputs = [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in get_inputs()
    ]

    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(*inputs)

    torch.cuda.synchronize()

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=False,
        profile_memory=True,
        record_shapes=True,
    ) as prof:

        with torch.no_grad():
            for _ in range(5):
                model(*inputs)

        torch.cuda.synchronize()

    key_averages = prof.key_averages()

    stats = []

    for evt in key_averages:

        # PyTorch compatibility
        cuda_time_total = getattr(
            evt,
            "cuda_time_total",
            getattr(evt, "device_time_total", 0),
        )

        self_cuda_time_total = getattr(
            evt,
            "self_cuda_time_total",
            getattr(evt, "self_device_time_total", 0),
        )

        cuda_memory_usage = getattr(
            evt,
            "cuda_memory_usage",
            0,
        )

        if cuda_time_total > 0:
            stats.append({
                "name": evt.key,
                "cuda_time_ms": (
                    cuda_time_total / 1000 / 5
                ),
                "cpu_time_ms": (
                    evt.cpu_time_total / 1000 / 5
                ),
                "self_cuda_time_ms": (
                    self_cuda_time_total / 1000 / 5
                ),
                "cuda_memory_usage_mb": (
                    cuda_memory_usage / 1024 / 1024
                ),
            })

    stats.sort(
        key=lambda x: x["cuda_time_ms"],
        reverse=True,
    )

    return {
        "top_ops": stats[:10],
        "total_cuda_time_ms": sum(
            s["cuda_time_ms"] for s in stats
        ),
    }


def profile_level(level, gpu_id=0):

    torch.cuda.set_device(gpu_id)

    device = torch.device(f"cuda:{gpu_id}")

    dataset = construct_kernelbench_dataset(level)

    results = {
        f"level{level}": {}
    }

    # Only first 10 problems
    problem_ids = dataset.get_problem_ids()[:NUM_PROBLEMS]

    for problem_id in tqdm(
        problem_ids,
        desc=f"Level {level}",
    ):

        (
            _,
            ref_arch_name,
            ref_arch_src,
        ) = fetch_ref_arch_from_dataset(
            dataset,
            problem_id,
        )

        try:
            profile_data = profile_with_torch_profiler(
                ref_arch_src,
                device,
            )

            results[f"level{level}"][
                ref_arch_name
            ] = profile_data

            print(f"Profiled: {ref_arch_name}")

        except Exception as e:

            print(
                f"Failed: {ref_arch_name}: {e}"
            )

            results[f"level{level}"][
                ref_arch_name
            ] = {
                "error": str(e)
            }

    save_dir = "results/profiles"

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"reference_profiles_level{level}.json",
    )

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved to {save_path}")


if __name__ == "__main__":

    level = int(sys.argv[1])

    gpu_id = int(sys.argv[2])

    profile_level(
        level=level,
        gpu_id=gpu_id,
    )