"""Shared multilingual sentence embeddings (for placing any corpus on a common map).

Uses a small multilingual sentence-transformer so personal (N=1) and DreamSeer dreams live
in ONE space (EN+RU comparable). Same env hardening as dreams/score.py.
"""
from __future__ import annotations
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path[:] = [p for p in sys.path if "/.local/lib/" not in p]

DEFAULT_ST = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_M = {}


def embed_texts(texts, model=DEFAULT_ST, batch=64, device=None, max_seq_length=256):
    from sentence_transformers import SentenceTransformer
    import torch
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if model not in _M:
        m = SentenceTransformer(model, device=device)
        m.max_seq_length = max_seq_length
        _M[model] = m
    return _M[model].encode(list(texts), batch_size=batch, show_progress_bar=False,
                            normalize_embeddings=True)
