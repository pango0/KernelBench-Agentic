#!/usr/bin/env python3
"""Thin chat wrapper around a local HuggingFace causal LM.

Every agent role (Code Analyzer, RAG Researcher, Kernel Generator, Feedback
Analyzer) talks to the *same* local model through this object, so the multi-agent
system stays single-model and fully offline. `chat()` accepts a standard
[{"role": ..., "content": ...}] message list (system + user/assistant turns) and
returns the decoded assistant text.
"""

from __future__ import annotations

from typing import Optional


class LocalLLM:
    """Stateless chat interface over a loaded HF model + tokenizer."""

    def __init__(self, model, tokenizer, default_max_new_tokens: int = 4096):
        self.model = model
        self.tokenizer = tokenizer
        self.default_max_new_tokens = default_max_new_tokens

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        import torch

        max_new_tokens = max_new_tokens or self.default_max_new_tokens
        device = next(self.model.parameters()).device
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=None,
        )
        batch = {
            "input_ids": torch.tensor([input_ids], device=device),
            "attention_mask": torch.ones(1, len(input_ids), device=device, dtype=torch.long),
        }
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id)
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.inference_mode():
            out = self.model.generate(**batch, **gen_kwargs)
        new_tokens = out[0, len(input_ids):]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
