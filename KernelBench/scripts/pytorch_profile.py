from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.utils import read_file
from inspect_kernel_pytorch_profiler import get_torch_profiler_info
import torch
level, problem_id = 1, 1
kernel_path = f"runs/zero_shot/level_{level}_problem_{problem_id}_sample_0_kernel.py"
dataset = construct_kernelbench_dataset(level=level, source="local")
ref_arch_src = dataset.get_problem_by_id(problem_id).code
kernel_src = read_file(kernel_path)
table = get_torch_profiler_info(
  ref_arch_src=ref_arch_src,
  kernel_src=kernel_src,
  build_dir="build/profile_zero_shot",  # separate from eval build dir
  device=torch.device("cuda:0"),
  num_trials=20,
  table_row_limit=15,
)
print(table)
