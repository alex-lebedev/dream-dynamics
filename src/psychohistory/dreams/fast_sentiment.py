"""Batched per-sentence XLM-R sentiment, fast enough to score every corpus rather than a subset.

`score.score_texts` wraps the HuggingFace `text-classification` pipeline, which pads every batch
to a fixed `max_length`. Sentences vary from three words to two hundred characters, so most of
that compute is spent on padding: on this machine the pipeline sustains ~40 sentences/s, which
makes true per-sentence scoring of ~500,000 sentences an overnight job on its own.

This module keeps the identical model and the identical output definition
(`sentiment = P(positive) - P(negative)`, `cardiffnlp/twitter-xlm-roberta-base-sentiment`) and
changes only the batching: sentences are sorted by token length, batched, dynamically padded to
the longest member of each batch, and restored to input order. Same instrument, same numbers,
several times the throughput.

    from psychohistory.dreams.fast_sentiment import score_sentences
    vals = score_sentences(list_of_sentences)      # np.float32, aligned to input order
"""
from __future__ import annotations

import os
import time

import numpy as np

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
_M = {}


def _load(model: str, device: str | None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    key = (model, device)
    if key not in _M:
        tok = AutoTokenizer.from_pretrained(model)
        mdl = AutoModelForSequenceClassification.from_pretrained(model).to(device).eval()
        lab = {v.strip().lower(): k for k, v in mdl.config.id2label.items()}
        _M[key] = (tok, mdl, device, lab["positive"], lab["negative"])
    return _M[key]


BATCH = int(os.environ.get("XLMR_BATCH", 128))


def score_sentences(sents, model: str = MODEL, batch: int = BATCH, device: str | None = None,
                    max_length: int = 128, progress_every: int = 50000) -> np.ndarray:
    """`P(positive) - P(negative)` per sentence, in input order."""
    import torch
    tok, mdl, dev, i_pos, i_neg = _load(model, device)
    texts = [s if (isinstance(s, str) and s.strip()) else " " for s in sents]
    n = len(texts)
    out = np.full(n, np.nan, dtype=np.float32)
    # length-sorted batching: dynamic padding then costs the batch's own longest member
    order = np.argsort([len(t) for t in texts], kind="stable")
    mps = dev == "mps"

    def run(sel):
        enc = tok([texts[j] for j in sel], padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt").to(dev)
        p = torch.softmax(mdl(**enc).logits.float(), -1).cpu().numpy()
        out[sel] = (p[:, i_pos] - p[:, i_neg]).astype(np.float32)

    t0, done, i = time.time(), 0, 0
    with torch.inference_mode():
        while i < n:
            sel = order[i:i + batch]
            try:
                run(sel)
            except RuntimeError as e:
                # MPS shares one memory pool with any other job on the machine, so a batch that
                # fit a minute ago can fail now. Halve and retry rather than lose the corpus.
                if "out of memory" not in str(e).lower() or batch <= 8:
                    raise
                if mps:
                    torch.mps.empty_cache()
                batch = max(8, batch // 2)
                print(f"    [xlmr] out of memory -> batch {batch}", flush=True)
                continue
            i += len(sel)
            done += len(sel)
            if mps and done % 20000 < batch:
                torch.mps.empty_cache()
            if progress_every and done % progress_every < batch:
                r = done / max(time.time() - t0, 1e-9)
                print(f"    [xlmr] {done:,}/{n:,} ({r:.0f}/s, eta {(n-done)/max(r,1e-9)/60:.0f}m)",
                      flush=True)
    return out
