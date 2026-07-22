"""A valence axis in the shared embedding space.

Learns a single direction w (+intercept) mapping a MiniLM embedding -> model sentiment,
fitted on the dream-level cache against the XLM-R dream sentiment (dreamseer_sentiment.parquet).
Lets us assign a fast, consistent *per-sentence* valence (proj onto w) for within-dream arc /
irreversibility work without re-scoring 160k sentences. Validate proj-valence vs the real
scorer on a subsample before headlining (see analyses).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from .embed_cache import load_or_build


def fit_valence_axis(alpha: float = 10.0, lang: str | None = None):
    """Return (w, b, meta_r) where sentiment_hat = emb @ w + b (ridge on dream-level)."""
    from sklearn.linear_model import Ridge

    meta, emb = load_or_build()
    sent = pd.read_parquet(C.DREAMS_OUT / "dreamseer_sentiment.parquet",
                           columns=["documentID", "sentiment"])
    smap = dict(zip(sent.documentID, sent.sentiment))
    y = np.array([smap.get(d, np.nan) for d in meta.documentID.values], dtype=float)
    ok = np.isfinite(y)
    if lang is not None:
        ok &= (meta.lang.values == lang)
    X = emb[ok].astype(float)
    yy = y[ok]
    r = Ridge(alpha=alpha).fit(X, yy)
    w = r.coef_.astype(np.float32)
    b = float(r.intercept_)
    # in-sample fit quality (diagnostic)
    r2 = float(r.score(X, yy))
    return w, b, {"n_fit": int(ok.sum()), "r2_insample": r2}


def project(emb: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return emb.astype(np.float32) @ w + b
