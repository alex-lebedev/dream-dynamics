"""Is dream-space low-dimensional, or is that just what a sentence encoder does to prose?

The published geometry compares dream embeddings against two SYNTHETIC nulls: an isotropic
Gaussian (~131 dimensions) and a variance-preserving feature shuffle (~112). Both show the
observed ~25 dimensions are not an artifact of distance concentration or of the encoder's
marginal variance profile. Neither can show the result is about *dreams*, because transformer
sentence embeddings are themselves anisotropic and strongly structured for any coherent text.
The same gap applies to the percolation "continent".

The missing control is empirical rather than synthetic: ordinary human language through the
identical pipeline. This runs TwoNN and percolation on

  dream corpora   : Dreamseer EN (the published reference), DreamBank, Reddit r/Dreams, SDDb
  human non-dream : r/confession personal narratives, Project Gutenberg fiction, Wikipedia
  machine prose   : Dreamseer LLM interpretations of the same dreams
  synthetic nulls : feature shuffle + isotropic Gaussian (the published nulls, recomputed)

every corpus embedded with the SAME encoder (paraphrase-multilingual-MiniLM-L12-v2), the SAME
256-character truncation and L2 normalization, at the same sample sizes. Because truncation
interacts with length, each non-dream corpus is run twice: under the plain 256-character rule,
and LENGTH-MATCHED, where every document is truncated to a character count drawn from the
Dreamseer English length distribution.

If ordinary narrative language also lands near 25 dimensions, the honest claim is that this
geometry belongs to natural language as encoded, not to dreaming — and the paper must say so.

    cd psychohistory && PYTHONPATH=src python3 \
        analyses/2026-08-20-04-geometry-language-baseline.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from psychohistory import config as C
from psychohistory.maps.geometry import (feature_shuffle, gaussian_ref, participation_ratio,
                                         percolation, two_nn_dim)

OUT = C.RESULTS / "showcase"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = C.INTERIM / "geometry_baseline_emb.npz"
MAX_CHARS = 256          # identical to psychohistory.dreams.embed_cache
N_DOCS = 12000           # >= the 8,000 TwoNN sample and the 5,000 percolation sample
SEEDS = (0, 1, 2, 3, 4)
BASE = C.EXTERNAL / "language_baselines"


# ---------------- text sources -----------------------------------------------------------------
def dreamseer_en_texts():
    lv = pd.read_parquet(C.DREAMS_OUT / "dreamseer_dream_level.parquet",
                         columns=["documentID", "lang", "date"])
    lv = lv[(lv.lang == "en") & (pd.to_datetime(lv.date) >= "2024-03-01")]
    raw = pd.read_csv(C.DREAMSEER_RAW, dtype=str, keep_default_na=False,
                      usecols=["documentID", "text"])
    d = lv.merge(raw, on="documentID")
    return [t for t in d.text.tolist() if isinstance(t, str) and len(t) >= 20]


def dreamseer_interp_texts():
    raw = pd.read_csv(C.DREAMSEER_RAW, dtype=str, keep_default_na=False,
                      usecols=["interpretation"])
    return [t for t in raw.interpretation.tolist() if isinstance(t, str) and len(t) >= 60]


def dreambank_texts():
    db = pd.read_csv(C.DREAMBANK_RAW, usecols=["content"], dtype=str, keep_default_na=False)
    return [t for t in db.content.tolist() if isinstance(t, str) and len(t) >= 60]


def reddit_dreams_texts():
    r = pd.read_csv(C.RAW / "mallett" / "r-dreams.csv", usecols=["selftext"], dtype=str,
                    keep_default_na=False)
    return [t for t in r.selftext.tolist() if isinstance(t, str) and len(t) >= 60]


def sddb_texts():
    f = sorted(glob.glob(str(C.RAW / "sddb" / "*.csv")))[0]
    s = pd.read_csv(f, usecols=["Dream Text"], dtype=str, keep_default_na=False)
    return [t for t in s["Dream Text"].tolist() if isinstance(t, str) and len(t) >= 60]


def baseline_texts(name):
    f = BASE / f"{name}.csv"
    if not f.exists():
        return []
    d = pd.read_csv(f, dtype=str, keep_default_na=False)
    return [t for t in d.text.tolist() if isinstance(t, str) and len(t) >= 60]


SOURCES = [
    ("Dreamseer EN (reference)", "dream", dreamseer_en_texts),
    ("DreamBank", "dream", dreambank_texts),
    ("Reddit r/Dreams", "dream", reddit_dreams_texts),
    ("SDDb", "dream", sddb_texts),
    ("Dreamseer LLM interpretations", "machine prose", dreamseer_interp_texts),
    ("Reddit r/confession (personal narrative)", "human non-dream",
     lambda: baseline_texts("personal-narrative")),
    ("Project Gutenberg fiction", "human non-dream", lambda: baseline_texts("fiction")),
    ("Wikipedia openings", "human non-dream", lambda: baseline_texts("encyclopedic")),
]


# ---------------- embedding ---------------------------------------------------------------------
def truncate(texts, lengths=None, seed=0):
    """Plain 256-char truncation, or length-matched truncation to a drawn character count."""
    if lengths is None:
        return [t[:MAX_CHARS] for t in texts]
    rng = np.random.default_rng(seed)
    draw = rng.choice(lengths, size=len(texts), replace=True)
    return [t[:int(min(MAX_CHARS, L))] for t, L in zip(texts, draw)]


def build():
    """Embed every corpus (both truncation rules for non-dream ones); cache to interim.

    Returns the cache *handle*, not a dict of arrays: on a 16 GB machine, materializing all
    seventeen 12,000x384 blocks at once alongside the 5,000x5,000 similarity matrix that
    percolation needs pushes the process into swap, where a five-minute measurement takes half an
    hour. Callers index one key at a time and drop it.
    """
    from psychohistory.maps.embed import embed_texts
    if CACHE.exists():
        return np.load(CACHE, allow_pickle=True)

    rng = np.random.default_rng(0)
    store = {}
    ds_texts = dreamseer_en_texts()
    # the reference length distribution (pre-truncation character counts, capped)
    ds_lengths = np.minimum(np.array([len(t) for t in ds_texts]), MAX_CHARS)
    store["_ds_lengths"] = ds_lengths

    for name, kind, loader in SOURCES:
        texts = loader()
        if len(texts) < 2000:
            print(f"[{name}] only {len(texts)} docs -> skip", flush=True); continue
        if len(texts) > N_DOCS:
            texts = [texts[i] for i in rng.choice(len(texts), N_DOCS, replace=False)]
        variants = [("trunc256", truncate(texts))]
        if not name.startswith("Dreamseer EN"):    # the reference defines the length distribution
            variants.append(("lenmatched", truncate(texts, ds_lengths, seed=1)))
        for vname, tt in variants:
            key = f"{name}||{kind}||{vname}"
            print(f"[embed] {key}  n={len(tt):,} "
                  f"mean_chars={np.mean([len(x) for x in tt]):.0f}", flush=True)
            E = np.asarray(embed_texts(tt, batch=64), dtype=np.float32)
            store[key] = E
    np.savez(CACHE, **store)
    print(f"[cache] wrote {CACHE}", flush=True)
    del store
    return np.load(CACHE, allow_pickle=True)


# ---------------- measurement -------------------------------------------------------------------
def measure(label, kind, variant, E):
    dnn = [two_nn_dim(E, seed=s) for s in SEEDS]
    perc = percolation(E, n=min(5000, len(E)))
    g50 = float(perc.loc[np.isclose(perc.theta, 0.50), "giant_frac"].iloc[0])
    g60 = float(perc.loc[np.isclose(perc.theta, 0.59), "giant_frac"].iloc[0])
    s50 = float(perc.loc[np.isclose(perc.theta, 0.50), "n_singletons"].iloc[0])
    sub = E[np.random.default_rng(0).choice(len(E), min(8000, len(E)), replace=False)]
    row = {"corpus": label, "kind": kind, "variant": variant, "n": len(E),
           "twonn": float(np.median(dnn)), "twonn_lo": float(min(dnn)),
           "twonn_hi": float(max(dnn)), "giant_050": g50, "giant_059": g60,
           "singletons_050": s50, "participation_ratio": participation_ratio(sub),
           "mean_cos": float(np.mean(sub @ sub[:1500].T))}
    print(f"  {label[:44]:44} {variant:11} TwoNN={row['twonn']:6.1f} "
          f"[{row['twonn_lo']:.1f},{row['twonn_hi']:.1f}]  giant@.50={g50:5.1%} "
          f"giant@.59={g60:5.1%}  PR={row['participation_ratio']:6.1f}", flush=True)
    return row


CKPT = C.INTERIM / "geometry_baseline_rows.csv"


def _measured(rows, label, variant):
    return any(r["corpus"] == label and r["variant"] == variant for r in rows)


def _checkpoint(rows):
    """Persist finished rows after each measurement: one row costs several minutes, and losing
    the batch to an interruption means paying for all of them again."""
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(CKPT, index=False)


def main():
    store = build()
    keys = [k for k in store.files if not k.startswith("_")]
    rows = pd.read_csv(CKPT).to_dict("records") if CKPT.exists() else []
    if rows:
        print(f"[resume] {len(rows)} rows already measured in {CKPT.name}", flush=True)
    print("\n=== measuring ===", flush=True)
    for key in keys:
        label, kind, variant = key.split("||")
        if _measured(rows, label, variant):
            continue
        rows.append(measure(label, kind, variant, np.asarray(store[key])))
        _checkpoint(rows)

    # the published synthetic nulls, recomputed on the reference for continuity
    ref_key = next(k for k in keys if k.startswith("Dreamseer EN (reference)"))
    for label, maker in [
        ("feature-shuffle null (Dreamseer EN)",
         lambda R: feature_shuffle(R[:min(12000, len(R))])),
        ("isotropic-Gaussian null", lambda R: gaussian_ref(min(12000, len(R)), R.shape[1])),
    ]:
        if _measured(rows, label, "trunc256"):
            continue
        rows.append(measure(label, "synthetic null", "trunc256",
                            maker(np.asarray(store[ref_key]))))
        _checkpoint(rows)

    df = pd.DataFrame(rows).sort_values(["kind", "corpus", "variant"])
    df.to_csv(OUT / "geometry_language_baseline.csv", index=False)
    _write_md(df)
    print("\n[geometry-baseline] wrote", OUT / "geometry_language_baseline.md")


def _write_md(df):
    L = ["# Is the geometry about dreams, or about the encoder? A matched-language baseline", "",
         "*The published intrinsic-dimension and percolation results are benchmarked only against "
         "synthetic nulls (isotropic Gaussian, variance-preserving feature shuffle). Those rule out "
         "distance concentration and marginal-variance artifacts, but not the possibility that a "
         "sentence encoder imposes this geometry on any coherent prose. Here every corpus goes "
         "through the identical pipeline: same encoder "
         "(`paraphrase-multilingual-MiniLM-L12-v2`), same 256-character truncation, same L2 "
         "normalization, TwoNN median over 5 seeds, percolation averaged over 3 seeds of 5,000 "
         "documents.*", "",
         "`lenmatched` truncates each document to a character count drawn from the Dreamseer "
         "English length distribution, so the comparison is not driven by baseline documents "
         "filling the 256-character window more often than dreams do.", "",
         "| corpus | register | truncation | n | TwoNN ID [min,max] | giant @ θ=0.50 | "
         "giant @ θ=0.59 | participation ratio |",
         "|---|---|---|--:|--:|--:|--:|--:|"]
    for _, r in df.iterrows():
        L.append(f"| {r.corpus} | {r.kind} | {r.variant} | {int(r.n):,} | "
                 f"**{r.twonn:.1f}** [{r.twonn_lo:.1f},{r.twonn_hi:.1f}] | {r.giant_050:.1%} | "
                 f"{r.giant_059:.1%} | {r.participation_ratio:.1f} |")
    dreams = df[(df.kind == "dream") & (df.variant == "trunc256")]
    human = df[(df.kind == "human non-dream") & (df.variant == "lenmatched")]
    L += ["", "## Read-out"]
    if len(dreams) and len(human):
        L += [f"- Dream corpora (256-char rule): TwoNN {dreams.twonn.min():.1f}–"
              f"{dreams.twonn.max():.1f}.",
              f"- Human non-dream corpora (length-matched): TwoNN {human.twonn.min():.1f}–"
              f"{human.twonn.max():.1f}.",
              "- The comparison that matters is dream vs human non-dream, not dream vs synthetic "
              "null. Whichever way it falls, it should be reported as the specificity test for the "
              "intrinsic-dimension and percolation claims."]
    L += ["", "*Aggregate-only: dimensions, connectivity and summary statistics; no text and no "
          "document-level vectors are written.*"]
    (OUT / "geometry_language_baseline.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
