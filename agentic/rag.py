#!/usr/bin/env python3
"""Documentation retrieval for the RAG Researcher agent.

The retriever indexes a directory of optimization notes (Markdown / text, plus PDF
when `pypdf` is installed) and returns the most relevant chunks for a query. The
default backend is a dependency-free BM25 (Okapi) implemented here, so the system
runs fully offline with no extra packages. An optional dense backend
(sentence-transformers) is used when `backend="embed"` and the package is present.

This module is intentionally self-contained and importable without torch.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

_WORD_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class Chunk:
    source: str          # file name the chunk came from
    chunk_id: int        # index within that file
    text: str            # chunk body

    @property
    def cite(self) -> str:
        return f"{self.source}#{self.chunk_id}"


# ---------------------------------------------------------------------------
# Corpus loading + chunking
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        import pypdf
    except Exception:
        return ""
    try:
        reader = pypdf.PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks at paragraph boundaries when possible."""
    text = text.strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paras:
        if len(para) > max_chars:
            # hard-split an oversized paragraph
            if buf:
                chunks.append(buf)
                buf = ""
            for i in range(0, len(para), max_chars - overlap):
                chunks.append(para[i: i + max_chars])
            continue
        if len(buf) + len(para) + 2 <= max_chars:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail}\n\n{para}" if tail else para
    if buf:
        chunks.append(buf)
    return chunks


def load_chunks(docs_dir: str | Path, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    docs_dir = Path(docs_dir)
    chunks: list[Chunk] = []
    if not docs_dir.exists():
        return chunks
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        suf = path.suffix.lower()
        if suf in (".md", ".markdown", ".txt", ".rst"):
            raw = _read_text(path)
        elif suf == ".pdf":
            raw = _read_pdf(path)
        else:
            continue
        for i, body in enumerate(chunk_text(raw, max_chars, overlap)):
            chunks.append(Chunk(source=path.name, chunk_id=i, text=body))
    return chunks


# ---------------------------------------------------------------------------
# BM25 retriever (default, dependency-free)
# ---------------------------------------------------------------------------

class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs_tokens = [_tokenize(c.text) for c in chunks]
        self.doc_len = [len(t) for t in self.docs_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        # document frequency
        self.df: dict[str, int] = {}
        for toks in self.docs_tokens:
            for term in set(toks):
                self.df[term] = self.df.get(term, 0) + 1
        n = max(len(chunks), 1)
        self.idf = {
            term: math.log(1 + (n - dfi + 0.5) / (dfi + 0.5))
            for term, dfi in self.df.items()
        }
        # per-doc term frequencies
        self.tf: list[dict[str, int]] = []
        for toks in self.docs_tokens:
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q_terms = _tokenize(query)
        scores = [0.0] * len(self.chunks)
        for i, counts in enumerate(self.tf):
            dl = self.doc_len[i] or 1
            s = 0.0
            for term in q_terms:
                f = counts.get(term)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += idf * (f * (self.k1 + 1)) / denom
            scores[i] = s
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = [(self.chunks[i], scores[i]) for i in ranked[:k] if scores[i] > 0]
        return out


# ---------------------------------------------------------------------------
# Optional dense retriever (sentence-transformers)
# ---------------------------------------------------------------------------

class EmbeddingRetriever:
    def __init__(self, chunks: list[Chunk], model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # may raise
        import numpy as np

        self.np = np
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        texts = [c.text for c in chunks] or [""]
        self.emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self.emb @ q
        idx = self.np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in idx]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_retriever(docs_dir: str | Path, backend: str = "bm25",
                    max_chars: int = 1200, overlap: int = 150):
    """Build a retriever over `docs_dir`. Falls back to BM25 if `embed` is requested
    but sentence-transformers is unavailable. Returns None if the corpus is empty."""
    chunks = load_chunks(docs_dir, max_chars, overlap)
    if not chunks:
        return None
    if backend == "embed":
        try:
            return EmbeddingRetriever(chunks)
        except Exception as e:  # noqa: BLE001
            print(f"[rag] embedding backend unavailable ({e}); using BM25", flush=True)
    return BM25Retriever(chunks)
