"""Consistent, model-based sentiment extraction from dream text.

Scores EVERY corpus (DreamSeer, personal N=1, external) with the SAME open multilingual model,
so valence is measured identically across languages and datasets — rather than relying on
DreamSeer's pre-baked, corpus-specific numeric columns (docs/METHODS.md §5, measurement
consistency).

Model: `cardiffnlp/twitter-xlm-roberta-base-sentiment` — multilingual (verified coherent on
EN and RU dream text), neg/neu/pos. Needs `sentencepiece`.

Env robustness (this box): transformers pulls TF + accelerate/boto3; we force torch-only and
drop the broken ~/.local urllib3 so Anaconda's urllib3 1.26 is used. Run with `python3`.
"""
from __future__ import annotations
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# defensive: remove the broken user-site (urllib3 2.x lacks DEFAULT_CIPHERS -> breaks boto3)
sys.path[:] = [p for p in sys.path if "/.local/lib/" not in p]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
_PIPE = {}


def _auto_device(device):
    if device is not None:
        return device
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return -1  # CPU


def get_pipeline(model=DEFAULT_MODEL, device=None):
    dev = _auto_device(device)
    key = (model, str(dev))
    if key not in _PIPE:
        from transformers import pipeline
        # max_length=256 keeps MPS memory in bounds (dreams are short; median ~58 words)
        _PIPE[key] = pipeline("text-classification", model=model, top_k=None,
                              truncation=True, max_length=256, device=dev)
    return _PIPE[key]


def score_texts(texts, model=DEFAULT_MODEL, batch=32, device=None, progress=True,
                chunk=2000) -> pd.DataFrame:
    """Return p_negative/p_neutral/p_positive and sentiment = P(pos)-P(neg) for each text.

    Passes `batch_size` to the pipeline (critical: without it the pipeline runs one-by-one,
    ~10x slower). On Apple MPS this is ~19 texts/sec vs ~2/sec unbatched.
    """
    pipe = get_pipeline(model, device)
    texts = [t if (isinstance(t, str) and t.strip()) else " " for t in texts]
    cols = {"p_negative": [], "p_neutral": [], "p_positive": []}
    n = len(texts)
    for i in range(0, n, chunk):
        part = texts[i:i + chunk]
        for ps in pipe(part, batch_size=batch):
            d = {p["label"].strip().lower(): float(p["score"]) for p in ps}
            cols["p_negative"].append(d.get("negative", np.nan))
            cols["p_neutral"].append(d.get("neutral", np.nan))
            cols["p_positive"].append(d.get("positive", np.nan))
        try:  # prevent MPS memory creep over long runs (avoids OOM)
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        if progress:
            print(f"    scored {min(i + chunk, n)}/{n}", flush=True)
    df = pd.DataFrame(cols)
    df["sentiment"] = df["p_positive"] - df["p_negative"]
    return df
