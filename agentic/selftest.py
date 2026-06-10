#!/usr/bin/env python3
"""Offline self-test for the agentic pipeline (no GPU, no model, no KernelBench).

Stubs the LLM and the evaluator so the orchestration graph (Code Analyzer -> RAG ->
Generator -> Evaluator -> Feedback Analyzer -> loop / Data collector) and the RAG
retriever can be validated on CPU. Run:  python agentic/selftest.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline as P
from rag import build_retriever
from datacollector import build_trajectory

DOCS = Path(__file__).resolve().parent / "docs"

GOOD_CODE = "```python\nimport torch\nimport torch.nn as nn\nclass ModelNew(nn.Module):\n    def forward(self, x):\n        return x\n\ndef get_inputs():\n    return [torch.randn(2)]\n\ndef get_init_inputs():\n    return []\n```"


class StubLLM:
    def __init__(self):
        self.calls = {"analyzer": 0, "researcher": 0, "generator": 0, "feedback": 0}

    def chat(self, messages, max_new_tokens=None, temperature=0.2, top_p=0.95, do_sample=True):
        sysmsg = messages[0]["content"]
        if "senior GPU performance engineer" in sysmsg:
            self.calls["analyzer"] += 1
            return ("## OPERATIONS\nmatmul, elementwise\n"
                    "## BOTTLENECKS\nthe matmul dominates\n"
                    "## FUSION\nfuse bias+relu into the matmul epilogue\n"
                    "## STRATEGY\ntile and use shared memory\n"
                    "## RESEARCH_QUERIES\nmatmul tiling shared memory\noperator fusion epilogue\n")
        if "research assistant" in sysmsg:
            self.calls["researcher"] += 1
            return "- Use shared-memory tiling [matmul_tiling.md#0]\n- Fuse the epilogue [operator_fusion.md#1]"
        if "expert CUDA/HIP kernel engineer" in sysmsg:
            self.calls["generator"] += 1
            return GOOD_CODE
        if "debugging and performance triage" in sysmsg:
            self.calls["feedback"] += 1
            return ("## CATEGORY\ncorrectness\n## ROOT_CAUSE\nwrong indexing\n"
                    "## FIXES\n1. add bounds check\n2. use contiguous\n## DIRECTIVE\nfix the indexing\n")
        return "unknown role"


def make_evaluator():
    """First evaluated kernel is correct-but-slow, second is correct-and-fast."""
    state = {"n": 0}

    def evaluate(ref_src, code):
        state["n"] += 1
        if state["n"] == 1:
            return {"compiled": True, "correctness": True, "runtime": 2.0, "ref_runtime": 1.0,
                    "metadata": {}}
        return {"compiled": True, "correctness": True, "runtime": 0.5, "ref_runtime": 1.0,
                "metadata": {}}

    return evaluate


def main() -> int:
    ok = True

    # 1) RAG retriever over the real docs corpus
    retr = build_retriever(DOCS, backend="bm25")
    assert retr is not None, "retriever should index the docs corpus"
    hits = retr.search("shared memory tiling for matmul", k=3)
    assert hits, "expected BM25 hits for a matmul query"
    print(f"[ok] RAG indexed {len(retr.chunks)} chunks; top hit = {hits[0][0].cite}")

    with tempfile.TemporaryDirectory() as td:
        ref = Path(td) / "ref.py"
        ref.write_text("import torch\nimport torch.nn as nn\nclass Model(nn.Module):\n    def forward(self,x): return x\n")
        entry = {"problem_id": 1, "problem_name": "stub", "reference_arch_path": str(ref),
                 "prompt": "Optimise this module."}

        # 2) Full pipeline with stubs (best-of-n, target speedup)
        llm = StubLLM()
        cfg = P.AgenticConfig(max_turns=3, best_of_n=2, collect_data=True, stop_on_correct=False,
                              target_speedup=1.5)
        out = P.run_problem(llm, make_evaluator(), retr, "1", "1", entry, cfg)

        assert out["status"] == "success", out["status"]
        assert out["analysis"] is not None and out["analysis"]["research_queries"], "analysis missing"
        assert out["research_brief"], "research brief missing"
        assert out["num_turns"] >= 1
        assert out.get("extracted_code"), "should have selected a best kernel"
        assert llm.calls["analyzer"] == 1, llm.calls
        assert llm.calls["researcher"] == 1, llm.calls
        assert llm.calls["generator"] >= 2, llm.calls          # best-of-n => >=2 samples
        print(f"[ok] pipeline status={out['status']} turns={out['num_turns']} "
              f"final_speedup={(out.get('final_eval') or {}).get('ref_runtime')}/"
              f"{(out.get('final_eval') or {}).get('runtime')}  llm_calls={llm.calls}")

        # 3) Post-training trajectory built from the turns
        traj = out["_post_training"]
        assert traj["sft"], "expected an SFT record (best correct kernel)"
        print(f"[ok] post-training records: sft={len(traj['sft'])} "
              f"prefs={len(traj['preferences'])} reflections={len(traj['reflections'])}")

        # 4) Ablation: disabling agents skips their LLM calls
        llm2 = StubLLM()
        cfg2 = P.AgenticConfig(max_turns=1, use_code_analyzer=False, use_rag=False,
                               use_feedback_analyzer=False)
        out2 = P.run_problem(llm2, make_evaluator(), retr, "1", "1", entry, cfg2)
        assert llm2.calls["analyzer"] == 0 and llm2.calls["researcher"] == 0, llm2.calls
        assert out2["analysis"] is None, "analysis should be absent when analyzer is off"
        print(f"[ok] ablation (all agents off) llm_calls={llm2.calls}")

    print("\nALL SELF-TESTS PASSED" if ok else "\nSELF-TESTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
