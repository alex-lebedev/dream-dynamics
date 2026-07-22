"""Generic build-once/reuse multilingual embeddings for ANY dream corpus (external replication).

Caches vectors to 10-data/interim/emb_<name>.npz keyed by a content SHA1 (no text stored), so the
cache is de-identified and stable across row reorderings. Same encoder/params as the DreamSeer
cache (MiniLM, 256-char truncation) so re-identification/geometry are comparable across corpora.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .. import config as C
from ..maps.embed import embed_texts


def _key(texts):
    return np.array([hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest() for t in texts])


def _mps_clear():
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def embed_corpus(name: str, texts, max_chars: int = 256, batch: int = 64, chunk: int = 1500,
                 rebuild: bool = False, device: str | None = None):
    """Return an (N, d) float32 array aligned to `texts` (built once, cached by content hash).

    Clears the MPS cache between (small) chunks to avoid the memory-creep stall that hits long runs.
    """
    cache = C.INTERIM / f"emb_{name}.npz"
    ids = _key(texts)
    if cache.exists() and not rebuild:
        z = np.load(cache, allow_pickle=True)
        lut = {k: i for i, k in enumerate(z["ids"])}
        if all(k in lut for k in ids):
            return z["emb"][[lut[k] for k in ids]].astype(np.float32)
    trunc = [t[:max_chars] for t in texts]
    parts = []
    for i in range(0, len(trunc), chunk):
        parts.append(np.asarray(embed_texts(trunc[i:i + chunk], batch=batch, device=device),
                                dtype=np.float32))
        _mps_clear()
        print(f"[emb_{name}] {min(i + chunk, len(trunc))}/{len(trunc)}", flush=True)
    emb = np.vstack(parts)
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    uniq, first = np.unique(ids, return_index=True)
    np.savez(cache, emb=emb[first], ids=ids[first])
    print(f"[emb_{name}] cached {emb.shape} -> {cache}", flush=True)
    return emb
