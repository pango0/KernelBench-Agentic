#!/usr/bin/env python3
"""Build a partial V100 baseline JSON from eval log ref_runtime lines."""

import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEVEL1_NAMES = [
    "1_Square_matrix_multiplication_.py",
    "2_Standard_matrix_multiplication_.py",
    "3_Batched_matrix_multiplication.py",
    "4_Matrix_vector_multiplication_.py",
    "5_Matrix_scalar_multiplication.py",
    "6_Matmul_with_large_K_dimension_.py",
    "7_Matmul_with_small_K_dimension_.py",
    "8_Matmul_with_irregular_shapes_.py",
    "9_Tall_skinny_matrix_multiplication_.py",
    "10_3D_tensor_matrix_multiplication.py",
]


def parse_log(log_path):
    text = log_path.read_text(encoding="utf-8")
    results = {}
    for m in re.finditer(
        r"\[Eval Result\] Problem ID: (\d+), Sample ID: \d+\n"
        r"(.+?)(?=\nAdding Eval Result|\n-{10,}|\Z)",
        text,
        re.DOTALL,
    ):
        pid = int(m.group(1))
        block = m.group(2)
        ref_m = re.search(r"ref_runtime=([-\d.]+)\s+ref_runtime_stats=(\{.+?\})(?:\s|$)", block)
        if not ref_m:
            continue
        ref_runtime = float(ref_m.group(1))
        if ref_runtime < 0:
            continue
        stats = ast.literal_eval(ref_m.group(2))
        results[pid] = stats
    return results


def main():
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "logs" / "zeroshot_918914.log"
    hardware = sys.argv[2] if len(sys.argv) > 2 else "V100_SXM2"
    out_name = sys.argv[3] if len(sys.argv) > 3 else "baseline_time_torch.json"

    timings = parse_log(log_path)
    level1 = {}
    for pid, stats in sorted(timings.items()):
        if 1 <= pid <= 10:
            level1[LEVEL1_NAMES[pid - 1]] = stats

    out_dir = REPO / "results" / "timing" / hardware
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    payload = {"level1": level1, "level2": {}, "level3": {}}
    out_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    print(f"Wrote {len(level1)} level-1 baselines to {out_path}")
    print(f"Analyze with: hardware={hardware} baseline={out_name.replace('.json', '')}")


if __name__ == "__main__":
    main()
