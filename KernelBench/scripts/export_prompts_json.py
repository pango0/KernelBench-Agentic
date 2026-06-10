#!/usr/bin/env python3
"""Export KernelBench prompts for problems 1-10 of each level to a JSON file."""

import argparse
import json
import os
import sys

REPO_TOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_TOP, "src"))

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.prompt_constructor_toml import get_prompt_for_backend


def export_prompts(
    output_path: str,
    levels: list[int],
    start_id: int = 1,
    end_id: int = 10,
    source: str = "local",
    prompt_option: str = "zero_shot",
    backend: str = "cuda",
    precision: str = "fp32",
) -> dict:
    payload = {
        "meta": {
            "source": source,
            "prompt_option": prompt_option,
            "backend": backend,
            "precision": precision,
            "problem_id_range": [start_id, end_id],
            "levels": levels,
        },
        "levels": {},
    }

    for level in levels:
        dataset = construct_kernelbench_dataset(
            level=level,
            source=source,
            id_range=(start_id, end_id),
        )
        level_key = str(level)
        payload["levels"][level_key] = {}

        for problem_id in dataset.get_problem_ids():
            problem = dataset.get_problem_by_id(problem_id)
            prompt = get_prompt_for_backend(
                problem.code,
                backend=backend,
                option=prompt_option,
                precision=precision,
            )
            payload["levels"][level_key][str(problem_id)] = {
                "problem_id": problem_id,
                "problem_name": problem.name,
                "reference_arch_path": problem.path,
                "prompt": prompt,
            }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        default=os.path.join(REPO_TOP, "prompts_level1-4_problems1-10_zero_shot.json"),
        help="Output JSON path",
    )
    parser.add_argument("--levels", default="1,2,3,4", help="Comma-separated levels")
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--end-id", type=int, default=10)
    parser.add_argument("--source", default="local", choices=["local", "huggingface"])
    parser.add_argument("--prompt-option", default="zero_shot")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--precision", default="fp32")
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.levels.split(",")]
    export_prompts(
        output_path=args.output,
        levels=levels,
        start_id=args.start_id,
        end_id=args.end_id,
        source=args.source,
        prompt_option=args.prompt_option,
        backend=args.backend,
        precision=args.precision,
    )
    print(f"Wrote prompts to {args.output}")


if __name__ == "__main__":
    main()
