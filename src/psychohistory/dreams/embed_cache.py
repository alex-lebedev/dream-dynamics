"""Build-once / reuse DreamSeer dream-level multilingual embeddings.

The last bold-probe run stalled re-embedding ~30k dreams every time; this caches the
vectors to ``10-data/interim/ds_embeddings.npz`` keyed by ``documentID`` ONLY (no text,
no userID) so the cache artifact is itself de-identified (docs/ETHICS.md S1).

Text is loaded transiently from raw at build time, truncated to ~256 chars, embedded on
MPS with the shared multilingual MiniLM (batch 64), and never persisted.

    from psychohistory.dreams.embed_cache import load_or_build
    meta, emb = load_or_build()      # meta rows align 1:1 with emb rows
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..maps.embed import embed_texts

CACHE = C.INTERIM / "ds_embeddings.npz"

# per-dream fields the probes need (kept small; all S1-safe once aggregated)
_META_COLS = [
    "documentID", "userID", "date", "lang", "textlen", "words",
    "nightmare_index", "fear", "negativity", "danger", "weapon",
]


def load_meta(start: str = "2024-03-01", min_chars: int = 20) -> pd.DataFrame:
    lv = pd.read_parquet(C.DREAMS_OUT / "dreamseer_dream_level.parquet", columns=_META_COLS)
    lv["date"] = pd.to_datetime(lv["date"])
    lv = lv[(lv.date >= start) & (lv.textlen >= min_chars)].reset_index(drop=True)
    return lv


def _load_text_map(doc_ids) -> list[str]:
    raw = pd.read_csv(C.DREAMSEER_RAW, dtype=str, keep_default_na=False,
                      usecols=["documentID", "text"])
    m = dict(zip(raw.documentID, raw.text))
    return [m.get(d, "") for d in doc_ids]


def load_or_build(start: str = "2024-03-01", min_chars: int = 20, max_chars: int = 256,
                  batch: int = 64, rebuild: bool = False, chunk: int = 2000):
    """Return (meta_df, emb) with rows aligned. Builds + caches on first call."""
    meta = load_meta(start=start, min_chars=min_chars)
    texts = _load_text_map(meta.documentID.values)
    # drop any dream whose raw text failed to join / is too short after join
    ok = np.array([len(t) >= min_chars for t in texts])
    meta = meta[ok].reset_index(drop=True)
    texts = [t for t, k in zip(texts, ok) if k]
    ids = meta.documentID.values

    if CACHE.exists() and not rebuild:
        z = np.load(CACHE, allow_pickle=True)
        if len(z["ids"]) == len(ids) and bool((z["ids"] == ids).all()):
            return meta, z["emb"].astype(np.float32)
        print("[embed_cache] cache stale (id mismatch) -> rebuilding", flush=True)

    trunc = [t[:max_chars] for t in texts]
    parts = []
    for i in range(0, len(trunc), chunk):
        e = embed_texts(trunc[i:i + chunk], batch=batch)  # L2-normalized
        parts.append(np.asarray(e, dtype=np.float32))
        print(f"[embed_cache] {min(i + chunk, len(trunc))}/{len(trunc)}", flush=True)
    emb = np.vstack(parts)
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, emb=emb, ids=ids)
    print(f"[embed_cache] wrote {CACHE} shape={emb.shape}", flush=True)
    return meta, emb


if __name__ == "__main__":
    m, e = load_or_build()
    assert CACHE.exists() and len(m) == len(e)
    print(f"[embed_cache] OK: {len(m)} dreams, dim={e.shape[1]}, "
          f"langs={m.lang.value_counts().to_dict()}")
