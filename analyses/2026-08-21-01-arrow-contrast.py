"""Formal dream-vs-non-dream contrasts for the emotional arrow, plus the preregistered skew.

`2026-08-20-05-arrow-true-xlmr.py` established that every corpus we score drifts negative --
dream reports steeply, waking narrative weakly -- and reported each corpus with its own cluster
bootstrap interval. Claiming a difference from two non-overlapping intervals is not a test, and
the multiplier ("3.8x as steeply") is a ratio of two point estimates with no interval at all.
This script runs the contrast the claim actually needs:

    1. Per-author drift, OLS slope and increment skew for all nine corpora, reconstructed from
       the committed per-sentence valence caches (no re-scoring, no GPU).
    2. A paired cluster bootstrap of the DIFFERENCE (dream minus comparator), resampling authors
       independently in both corpora, giving a difference interval, a ratio interval and a
       two-sided bootstrap p, BH-adjusted across the full 5 x 3 contrast family.
    3. The same contrast on SENTENCE-COUNT-MATCHED subsets. End-minus-start is mechanically
       length-dependent and the corpora are not length-matched, so each contrast is repeated
       after stratifying both corpora to a common sentence-count histogram.
    4. The per-sentence OLS slope as a length-insensitive alternative estimand.
    5. Increment skew (gamma), the statistic named in the preregistration for H3, which the
       primary-instrument run never reported.

Note on sampling: the primary-instrument run takes the FIRST `MAXDOC` qualifying documents of
each corpus in file order, not a random draw. This script inherits that sample exactly, because
it reads the caches that run produced; the document-order caveat therefore applies to these
contrasts too and is reported in the output rather than silently corrected.

Aggregate-only output: per-corpus and per-contrast statistics, no document text, no author keys.

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 \
        python3 analyses/2026-08-21-01-arrow-contrast.py
"""
from __future__ import annotations

import gc
import os
import sys

import numpy as np
import pandas as pd

from psychohistory import config as C
from psychohistory.dreams.arrow import TEXT_CORPORA, MIN_SENT_ARROW
from psychohistory.dreams.sentence_cache import split_sentences

OUT = C.RESULTS / "showcase"
OUT.mkdir(parents=True, exist_ok=True)

MAXDOC = int(os.environ.get("MAXDOC", 6000))
MAX_SENT = int(os.environ.get("MAX_SENT", 25))
B_BOOT = int(os.environ.get("B_BOOT", 4000))
N_MATCH_SEEDS = int(os.environ.get("N_MATCH_SEEDS", 50))

DREAMS = ["DreamSeer EN", "DreamSeer RU", "DreamBank", "Reddit r/Dreams", "SDDb"]
COMPARATORS = ["r/confession (waking narrative)", "Gutenberg fiction", "Wikipedia openings"]


def load(name):
    """Reconstruct (trajectories, authors, sentence counts) from the committed valence cache.

    Replays the same deterministic document selection as the run that built the cache and
    refuses to proceed unless the per-document sentence counts match exactly, which is what
    licenses slicing the cached valence vector by document. The raw corpus is released as soon
    as the counts are known: several loaders hold their whole source file in memory, which is
    enough to lose the process if another analysis is running alongside it.
    """
    stem, kind, loader = TEXT_CORPORA[name]
    cache = C.INTERIM / f"true_xlmr_arrow_{stem}.npz"
    texts, authors = loader()
    sents, auth = [], []
    for t, a in zip(texts, authors):
        ss = split_sentences(t, min_words=2)[:MAX_SENT]
        if len(ss) < MIN_SENT_ARROW:
            continue
        sents.append(len(ss)); auth.append(a)
        if len(sents) >= MAXDOC:
            break
    del texts, authors
    gc.collect()
    counts = np.array(sents, np.int32)
    z = np.load(cache)
    if len(z["counts"]) != len(counts) or not bool((z["counts"] == counts).all()):
        raise RuntimeError(f"[{name}] replay does not match cache "
                           f"(cache n={len(z['counts'])}, replay n={len(counts)})")
    off = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    vals = z["vals"]
    ser = [np.asarray(vals[o:o + c], float) for o, c in zip(off, counts)]
    # Author identity is only ever needed as a grouping key, so keep integer codes.
    codes = pd.factorize(pd.Series(auth, dtype="object"))[0].astype(np.int32)
    return ser, codes, counts, kind


def prepare(name):
    """Per-document statistics for one corpus, checkpointed so a kill does not lose the work."""
    stem = TEXT_CORPORA[name][0]
    ck = C.INTERIM / f"arrow_contrast_{stem}.npz"
    if ck.exists():
        z = np.load(ck)
        print(f"  [ckpt] {name}", flush=True)
        return {k: z[k] for k in ("drift", "slope", "skew", "auth", "counts")} | \
               {"kind": str(z["kind"])}
    ser, codes, counts, kind = load(name)
    drift, slope, skew = per_doc(ser)
    del ser
    gc.collect()
    np.savez(ck, drift=drift, slope=slope, skew=skew, auth=codes, counts=counts, kind=kind)
    return {"drift": drift, "slope": slope, "skew": skew, "auth": codes,
            "counts": counts, "kind": kind}


def per_doc(ser):
    """Three per-document statistics: end-minus-start, OLS slope, increment skew."""
    drift = np.array([v[-1] - v[0] for v in ser])
    slope = np.array([np.polyfit(np.linspace(0, 1, len(v)), v, 1)[0] for v in ser])
    skew = np.empty(len(ser))
    for i, v in enumerate(ser):
        d = np.diff(v)
        s = d.std()
        skew[i] = np.nan if s < 1e-12 else float(((d - d.mean()) ** 3).mean() / s ** 3)
    return drift, slope, skew


def by_author(x, auth):
    """Author-level means, dropping documents with an undefined statistic."""
    ok = np.isfinite(x)
    df = pd.DataFrame({"a": auth[ok], "x": x[ok]})
    return df.groupby("a", sort=True)["x"].mean().to_numpy()


def boot_mean(a, B=B_BOOT, seed=0):
    """Cluster bootstrap of an author-weighted mean; returns (mean, lo, hi, draws)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(B, n))
    draws = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), draws


def sign_flip_p(a, B=B_BOOT, seed=0):
    """Author-level sign-flip test that the author-weighted mean is zero."""
    rng = np.random.default_rng(seed)
    obs = abs(a.mean())
    s = rng.choice([-1.0, 1.0], size=(B, len(a)))
    null = np.abs((s * a).mean(axis=1))
    return float((1 + (null >= obs).sum()) / (B + 1))


def contrast(dream_a, comp_a, B=B_BOOT, seed=0):
    """Bootstrap the difference and ratio of two author-weighted means.

    Authors are resampled independently within each corpus, which is the right null for two
    disjoint author populations. The p-value is the two-sided bootstrap probability that the
    difference is on the wrong side of zero, floored at 1/(B+1).
    """
    rng = np.random.default_rng(seed)
    d = dream_a[rng.integers(0, len(dream_a), size=(B, len(dream_a)))].mean(axis=1)
    c = comp_a[rng.integers(0, len(comp_a), size=(B, len(comp_a)))].mean(axis=1)
    diff = d - c
    obs = float(dream_a.mean() - comp_a.mean())
    p = 2 * min((diff >= 0).mean(), (diff <= 0).mean())
    p = max(float(p), 1.0 / (B + 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(c) > 1e-9, d / c, np.nan)
    rr = ratio[np.isfinite(ratio)]
    return {
        "diff": obs,
        "diff_lo": float(np.percentile(diff, 2.5)),
        "diff_hi": float(np.percentile(diff, 97.5)),
        "p": p,
        "ratio": float(dream_a.mean() / comp_a.mean()) if abs(comp_a.mean()) > 1e-9 else np.nan,
        "ratio_lo": float(np.percentile(rr, 2.5)) if len(rr) else np.nan,
        "ratio_hi": float(np.percentile(rr, 97.5)) if len(rr) else np.nan,
    }


def bh(p):
    """Benjamini-Hochberg adjusted q-values."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    m = len(p)
    running = 1.0
    for rank, i in enumerate(o[::-1]):
        running = min(running, p[i] * m / (m - rank))
        q[i] = running
    return q


def match_counts(k_a, k_b, seed=0):
    """Indices into a and b whose sentence-count histograms are identical by construction.

    For every sentence count present in both corpora, keep min(n_a, n_b) documents from each,
    sampled without replacement. This removes the length difference that end-minus-start is
    mechanically sensitive to, at the cost of sample size.
    """
    rng = np.random.default_rng(seed)
    ia, ib = [], []
    for k in np.intersect1d(np.unique(k_a), np.unique(k_b)):
        pa = np.flatnonzero(k_a == k)
        pb = np.flatnonzero(k_b == k)
        m = min(len(pa), len(pb))
        ia.append(rng.choice(pa, m, replace=False))
        ib.append(rng.choice(pb, m, replace=False))
    if not ia:
        return np.array([], int), np.array([], int)
    return np.concatenate(ia), np.concatenate(ib)


def matched_over_seeds(D, K, stat, seeds=N_MATCH_SEEDS):
    """Repeat the count-matched contrast over many matching draws.

    A single stratified draw is one realisation of an arbitrary choice, and reporting its
    p-value as though it were "the" matched result hides that variability. We report the median
    across draws together with its range.
    """
    ps, ds, n = [], [], 0
    for s in range(seeds):
        ia, ib = match_counts(D["counts"], K["counts"], seed=s)
        if not len(ia):
            continue
        r = contrast(by_author(D[stat][ia], D["auth"][ia]),
                     by_author(K[stat][ib], K["auth"][ib]), B=1000, seed=s)
        ps.append(r["p"]); ds.append(r["diff"]); n = len(ia)
    if not ps:
        return {}
    return {"mat_diff": float(np.median(ds)), "mat_p": float(np.median(ps)),
            "mat_p_lo": float(np.min(ps)), "mat_p_hi": float(np.max(ps)), "matched_n": n}


def main():
    store, rows = {}, []
    for name in DREAMS + COMPARATORS + ["DreamSeer interp."]:
        try:
            rec = prepare(name)
        except Exception as e:
            print(f"[{name}] SKIP: {type(e).__name__}: {e}", flush=True)
            continue
        store[name] = rec
        drift, slope, skew, auth = rec["drift"], rec["slope"], rec["skew"], rec["auth"]
        counts, kind = rec["counts"], rec["kind"]
        da, sa, ka = by_author(drift, auth), by_author(slope, auth), by_author(skew, auth)
        md, lo, hi, _ = boot_mean(da)
        ms, slo, shi, _ = boot_mean(sa)
        mk, klo, khi, _ = boot_mean(ka)
        rows.append({
            "corpus": name, "kind": kind, "n_docs": len(drift), "n_authors": len(da),
            "sent_median": float(np.median(counts)),
            "sent_q1": float(np.percentile(counts, 25)),
            "sent_q3": float(np.percentile(counts, 75)),
            "drift_auth": md, "drift_lo": lo, "drift_hi": hi, "drift_p": sign_flip_p(da),
            "slope_auth": ms, "slope_lo": slo, "slope_hi": shi, "slope_p": sign_flip_p(sa),
            "skew_auth": mk, "skew_lo": klo, "skew_hi": khi, "skew_p": sign_flip_p(ka),
        })
        print(f"  {name:34s} n={len(drift):5,} auth={len(da):5,} sent~{np.median(counts):4.1f} "
              f"drift={md:+.4f} slope={ms:+.4f} skew={mk:+.3f}", flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(OUT / "arrow_contrast_corpora.csv", index=False)

    con = []
    for d in DREAMS:
        if d not in store:
            continue
        for c in COMPARATORS:
            if c not in store:
                continue
            D, K = store[d], store[c]
            for stat in ("drift", "slope"):
                raw = contrast(by_author(D[stat], D["auth"]), by_author(K[stat], K["auth"]))
                con.append({
                    "dream": d, "comparator": c, "stat": stat,
                    **{f"raw_{k}": v for k, v in raw.items()},
                    **matched_over_seeds(D, K, stat),
                })
    con = pd.DataFrame(con)
    for stat in con.stat.unique():
        m = con.stat == stat
        con.loc[m, "raw_q"] = bh(con.loc[m, "raw_p"].to_numpy())
        con.loc[m, "mat_q"] = bh(con.loc[m, "mat_p"].to_numpy())
    # The paper's claim is a conjunction over two estimands, so also report the more
    # conservative correction across the pooled 30-test family.
    con["pooled_q"] = bh(con["raw_p"].to_numpy())
    con.to_csv(OUT / "arrow_contrast_tests.csv", index=False)

    with open(OUT / "arrow_contrast.md", "w") as f:
        f.write("# Emotional arrow: formal dream-vs-non-dream contrasts\n\n")
        f.write(f"True per-sentence XLM-R. Author-weighted means, {B_BOOT}-draw cluster "
                "bootstrap over authors. Documents are the first "
                f"{MAXDOC:,} qualifying per corpus in file order (inherited from the "
                "primary-instrument run), capped at "
                f"{MAX_SENT} sentences, minimum {MIN_SENT_ARROW}.\n\n")
        f.write("## Per corpus\n\n")
        f.write(per.to_markdown(index=False, floatfmt=".4f") + "\n\n")
        f.write("## Contrasts (dream minus comparator), BH-adjusted within each statistic\n\n")
        f.write("`raw` = as sampled; `mat` = sentence-count-matched subsets.\n\n")
        keep = ["dream", "comparator", "stat", "raw_diff", "raw_diff_lo", "raw_diff_hi",
                "raw_ratio", "raw_ratio_lo", "raw_ratio_hi", "raw_q", "pooled_q",
                "matched_n", "mat_diff", "mat_q", "mat_p_lo", "mat_p_hi"]
        f.write(con[keep].to_markdown(index=False, floatfmt=".4f") + "\n")
    print(f"[contrast] wrote {OUT/'arrow_contrast.md'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
