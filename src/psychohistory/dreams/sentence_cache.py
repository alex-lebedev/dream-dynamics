"""Build-once / reuse SENTENCE-level embeddings for DreamSeer dreams.

Every prior structure/physics probe collapsed a dream to a single vector. This opens the
*intra-dream* resolution: each dream (or its LLM `interpretation`, used as a waking-prose
control) is split into sentences and embedded with the SAME multilingual MiniLM as the
dream-level cache, so within-dream semantic trajectories are comparable to everything else.

Cache (``10-data/interim/emb_sent_<field>.npz``) is keyed by ``documentID`` + sentence index
only (no text, no userID) -> de-identified like the dream-level cache (docs/ETHICS.md S1).

    from psychohistory.dreams.sentence_cache import load_or_build_sentences
    docs, counts, emb, nwords, meta = load_or_build_sentences("text")
    # trajectory of dream k: emb[off[k]:off[k]+counts[k]]  where off = cumsum offsets
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .. import config as C
from ..maps.embed import embed_texts
from .embed_cache import load_meta

# split on sentence enders (incl. Cyrillic-friendly) and hard newlines; keep it simple + fast.
_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+", re.UNICODE)


def split_sentences(text: str, min_words: int = 2, max_chars: int = 200) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    out = []
    for p in _SPLIT.split(text.strip()):
        p = p.strip()
        if len(p.split()) >= min_words:
            out.append(p[:max_chars])
    return out


def _raw_field(doc_ids, field: str) -> dict:
    raw = pd.read_csv(C.DREAMSEER_RAW, dtype=str, keep_default_na=False,
                      usecols=["documentID", field])
    m = dict(zip(raw.documentID, raw[field]))
    return m


def load_or_build_sentences(field: str = "text", start: str = "2024-03-01",
                            min_words: int = 2, min_sent: int = 1, limit: int | None = None,
                            batch: int = 128, chunk: int = 4000, rebuild: bool = False,
                            seed: int = 0):
    """Return (docs, counts, emb, nwords, meta).

    - docs   : (D,) documentID in build order
    - counts : (D,) #sentences per doc (>= min_sent)
    - emb    : (S, 384) L2-normalized sentence embeddings, doc-contiguous in `docs` order
    - nwords : (S,) word count per sentence
    - meta   : doc-level DataFrame aligned to `docs` (lang, userID, date, nightmare_index, ...)
    `limit` (optional) caps #docs (random subsample) — used for the interpretation control.
    """
    cache = C.INTERIM / f"emb_sent_{field}.npz"
    meta_all = load_meta(start=start, min_chars=20).copy()
    if limit is not None and limit < len(meta_all):
        meta_all = meta_all.sample(limit, random_state=seed).sort_index().reset_index(drop=True)
    fmap = _raw_field(meta_all.documentID.values, field)

    keep_docs, counts, sents, nwords = [], [], [], []
    keep_mask = np.zeros(len(meta_all), dtype=bool)
    for i, d in enumerate(meta_all.documentID.values):
        ss = split_sentences(fmap.get(d, ""), min_words=min_words)
        if len(ss) < min_sent:
            continue
        keep_mask[i] = True
        keep_docs.append(d)
        counts.append(len(ss))
        sents.extend(ss)
        nwords.extend(len(s.split()) for s in ss)
    meta = meta_all[keep_mask].reset_index(drop=True)
    docs = np.array(keep_docs)
    counts = np.array(counts, dtype=np.int32)
    nwords = np.array(nwords, dtype=np.int32)

    if cache.exists() and not rebuild:
        z = np.load(cache, allow_pickle=True)
        if len(z["docs"]) == len(docs) and bool((z["docs"] == docs).all()):
            return docs, counts, z["emb"].astype(np.float32), nwords, meta
        print(f"[sent_{field}] cache stale -> rebuild", flush=True)

    parts = []
    for i in range(0, len(sents), chunk):
        parts.append(np.asarray(embed_texts(sents[i:i + chunk], batch=batch), dtype=np.float32))
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        print(f"[sent_{field}] {min(i + chunk, len(sents))}/{len(sents)}", flush=True)
    emb = np.vstack(parts) if parts else np.zeros((0, 384), np.float32)
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    np.savez(cache, emb=emb, docs=docs, counts=counts, nwords=nwords)
    print(f"[sent_{field}] wrote {cache} sentences={emb.shape}", flush=True)
    return docs, counts, emb, nwords, meta


def offsets(counts: np.ndarray) -> np.ndarray:
    """Start index of each doc's sentence block in the contiguous emb array."""
    return np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)


if __name__ == "__main__":
    docs, counts, emb, nwords, meta = load_or_build_sentences("text")
    print(f"[sent] docs={len(docs)} sentences={len(emb)} "
          f"median_sent/doc={np.median(counts):.0f} langs={meta.lang.value_counts().to_dict()}")
