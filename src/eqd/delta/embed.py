"""delta/embed.py — embedding-space filing change, the semantic doc_similarity.

Motivated by Didisheim, Kelly, Pourmohammadi & Tian, "The Inefficient Pricing
of News" (NBER w35093): represent text as an embedding vector and treat the
part not explained by firm characteristics as the signal. Here the text is a
filing section (Item 1A), the vector is the mean of chunk embeddings, and the
raw signal is the cosine distance between consecutive years' vectors:

  emb_delta = 1 - cos(vec(prior Item 1A), vec(current Item 1A))

0 means the section says the same thing (even if reworded); larger means the
meaning actually moved. The mechanical diff counts changed sentences; this
measures how far the content drifted.

Model: bge-large-en-v1.5 via fastembed (ONNX, free, local; runs on the GPU
when onnxruntime's CUDA provider is available, CPU otherwise — override with
EQD_EMBED_MODEL). Sections are chunked to fit the 512-token window and
mean-pooled — the same aggregation the paper applies to article embeddings.
`model` is injectable everywhere: any object with
.embed(list[str]) -> iterable of vectors works, so tests never touch the
real model.
"""

from __future__ import annotations

import os
import re

import numpy as np

MODEL_NAME = os.getenv("EQD_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
CHUNK_CHARS = 1800  # ~450 tokens, safely inside the model's 512-token window

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def get_model(cache_dir: str | None = None):
    """The real embedding model (downloaded once, then cached). GPU if available."""
    from fastembed import TextEmbedding

    providers = None
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()  # locate pip-installed CUDA/cuDNN DLLs on Windows
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except ImportError:
        pass
    return TextEmbedding(MODEL_NAME, cache_dir=cache_dir, providers=providers)


def chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Split a section into chunks of <= max_chars, breaking on sentence ends."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks, cur = [], ""
    for sent in _SENT_SPLIT.split(text):
        if cur and len(cur) + 1 + len(sent) > max_chars:
            chunks.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}" if cur else sent
        while len(cur) > max_chars:  # pathological run-on with no sentence end
            chunks.append(cur[:max_chars])
            cur = cur[max_chars:]
    if cur:
        chunks.append(cur)
    return chunks


def section_vector(text: str, model) -> np.ndarray | None:
    """Mean-pooled, L2-normalized embedding of a whole section."""
    chunks = chunk_text(text)
    if not chunks:
        return None
    vecs = np.asarray(list(model.embed(chunks)), dtype=np.float64)
    v = vecs.mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def embed_delta(prior_vec: np.ndarray, cur_vec: np.ndarray) -> float:
    """Cosine distance between two normalized section vectors, in [0, 2]."""
    return float(1.0 - np.dot(prior_vec, cur_vec))
