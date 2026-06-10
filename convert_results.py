#!/usr/bin/env python3
"""Convert zeroshot results.json to KernelBench eval layout."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "zeroshot" / "results.json",
    )
    parser.add_argument(
        "--run-name",
        default="zero_shot",
        help="KernelBench run_name (output: KernelBench/runs/{run_name}/)",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).parent / "KernelBench" / "runs",
    )
    parser.add_argument("--sample-id", type=int, default=0)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as f:
        data = json.load(f)

    run_dir = args.runs_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for level_str, problems in data.get("levels", {}).items():
        level = int(level_str)
        for prob_id_str, entry in problems.items():
            code = entry.get("extracted_code")
            if not code or entry.get("status") != "success":
                skipped += 1
                continue

            # KernelBench eval expects this exact filename:
            # level_{L}_problem_{id}_sample_{s}_kernel.py
            out_path = run_dir / f"level_{level}_problem_{prob_id_str}_sample_{args.sample_id}_kernel.py"
            out_path.write_text(code, encoding="utf-8")
            print(f"Written: {out_path}")
            written += 1

    print(f"Done. {written} kernels written, {skipped} skipped.")
    print(f"Eval with:")
    print(
        f"  cd KernelBench && python scripts/eval_from_generations.py "
        f"run_name={args.run_name} dataset_src=local level=1 "
        f"subset='(1,10)' num_gpu_devices=1 gpu_arch=Volta"
    )


if __name__ == "__main__":
    main()
