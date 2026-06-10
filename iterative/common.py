"""Shared helpers for iterative refinement."""

import json
import re
from pathlib import Path
from typing import Any, Optional


def extract_first_code(output_string: str, code_language_types=None) -> Optional[str]:
    if code_language_types is None:
        code_language_types = ["python", "cpp"]
    if not output_string or not output_string.strip():
        return None

    trimmed = output_string.strip()

    def _strip_lang_header(code: str) -> str:
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type) :].strip()
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


def save_json(path: Path, payload: dict) -> None:
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


def iter_tasks(data: dict):
    tasks = []
    for level_key, problems in data.get("levels", {}).items():
        for pid, entry in problems.items():
            tasks.append((level_key, str(pid), dict(entry)))
    return tasks


def read_ref_arch(entry: dict) -> str:
    path = entry.get("reference_arch_path")
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8")
    raise FileNotFoundError(
        "reference_arch_path missing or not found in data entry: {}".format(path)
    )


def eval_result_to_dict(result) -> dict:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return dict(result)
