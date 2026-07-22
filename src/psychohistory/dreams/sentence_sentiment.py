"""Per-sentence XLM-R sentiment for DreamSeer, aligned to the sentence-embedding cache.

The valence *proxy* (projection onto a dream-level axis) transfers poorly to short sentences
(r~0.27 vs the real scorer), so the emotional-arc analysis needs true per-sentence sentiment.
We re-split each dream with the SAME splitter as `sentence_cache` (so order/counts align 1:1 with
the embedding cache) and score with the shared multilingual model. Only dreams with >= `min_sent`
sentences are scored (the arc set); others are left NaN. Cache is de-identified (docs + counts +
sentiment only; no text).

    from psychohistory.dreams.sentence_sentiment import load_or_build_sentence_sentiment
    docs, counts, sent = load_or_build_sentence_sentiment("text", min_sent=5)
"""
from __future__ import annotations

import numpy as np

from .. import config as C
from .sentence_cache import _raw_field, load_or_build_sentences, offsets, split_sentences
from .score import score_texts


def load_or_build_sentence_sentiment(field: str = "text", min_sent: int = 5, max_sent: int = 25,
                                     max_docs_per_lang: int = 3000, device=-1, seed: int = 0,
                                     rebuild: bool = False, batch: int = 128):
    """Per-sentence sentiment for a language-stratified, length-capped subset of multi-sentence
    dreams (arcs need trajectories, not every dream). CPU by default (MPS creeps on long runs).
    Docs outside the selection are left NaN; alignment to the embedding cache is preserved."""
    cache = C.INTERIM / f"sent_sentiment_{field}_ms{min_sent}.npz"
    docs, counts, _emb, _nw, meta = load_or_build_sentences(field)
    if cache.exists() and not rebuild:
        z = np.load(cache, allow_pickle=True)
        if len(z["docs"]) == len(docs) and bool((z["docs"] == docs).all()):
            return docs, counts, z["sent"].astype(np.float32)
        print(f"[sent_sent_{field}] cache stale -> rebuild", flush=True)

    rng = np.random.default_rng(seed)
    off = offsets(counts)
    sent = np.full(int(counts.sum()), np.nan, dtype=np.float32)
    # language-stratified, length-capped doc selection
    elig = np.where((counts >= min_sent) & (counts <= max_sent))[0]
    langs = meta["lang"].to_numpy()
    sel = []
    for lg in np.unique(langs[elig]):
        pool = elig[langs[elig] == lg]
        if len(pool) > max_docs_per_lang:
            pool = rng.choice(pool, max_docs_per_lang, replace=False)
        sel.append(pool)
    sel = np.sort(np.concatenate(sel))

    fmap = _raw_field(docs, field)
    texts, positions = [], []
    for k in sel:
        ss = split_sentences(fmap.get(docs[k], ""))
        for j, s in enumerate(ss[:counts[k]]):
            texts.append(s)
            positions.append(off[k] + j)
    print(f"[sent_sent_{field}] scoring {len(texts)} sentences from {len(sel)} dreams "
          f"(langs {dict(zip(*np.unique(langs[sel], return_counts=True)))}) on device={device} ...",
          flush=True)
    scores = score_texts(texts, batch=batch, device=device)["sentiment"].to_numpy()
    sent[np.array(positions, dtype=np.int64)] = scores.astype(np.float32)
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    np.savez(cache, docs=docs, counts=counts, sent=sent)
    print(f"[sent_sent_{field}] wrote {cache} ({np.isfinite(sent).sum()} scored)", flush=True)
    return docs, counts, sent


if __name__ == "__main__":
    d, c, s = load_or_build_sentence_sentiment("text", min_sent=5)
    print(f"[sent_sent] scored sentences: {np.isfinite(s).sum()} / {len(s)}")
