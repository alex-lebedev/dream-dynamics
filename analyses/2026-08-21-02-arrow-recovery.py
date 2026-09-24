"""Does the narrative RESOLVE at its close? A direct test, replacing a curvature inference.

`2026-08-21-01-arrow-contrast.py` showed that waking narrative's end-minus-start drift (-0.034)
is far smaller than its whole-trajectory OLS slope (-0.082) would predict, while dream reports'
two estimands agree. It is tempting to read that gap as "waking narrative recovers in its closing
sentence" -- but the gap is a CURVATURE statistic. Any departure from a straight line produces it:
a dark opening, a mid-document trough, a gradual lift over the final third. Drift and slope cannot
localise anything to the ending, and the one direct estimate already in hand (r/confession's
drop-last-sentence arm) pointed the other way.

So measure the thing:

    1. RECOVERY STEP. On documents of >=6 sentences, the closing step v[-1] - v[-2], author-
       weighted, cluster-bootstrapped. This is exactly "how much does the last sentence lift the
       trajectory", with no curvature assumption.
    2. NORMALISED RECOVERY. The closing step as a fraction of the descent that preceded it,
       step / -(v[-2] - v[0]), formed from the two author-weighted means inside every bootstrap
       draw so the ratio carries an interval.
    3. TERMINAL FLATTENING. Second-half minus first-half OLS slope. Catches resolution spread
       over a closing passage rather than concentrated in one sentence, which a single-sentence
       step would miss.
    4. EDGE PROFILES. Mean valence at each of the first and last five sentence positions,
       author-weighted -- the descriptive shape behind the three statistics above, and the only
       way to see WHERE a drift-versus-slope disagreement actually sits.
    5. OPENING-MATCHED CONTRAST. The edge profiles show the corpora do not start from the same
       place: dream reports open near neutral, r/confession opens at its emotional floor. A
       corpus that starts low has less room to fall, so "dream reports descend more steeply"
       could be nothing more than "dream reports start higher". Each contrast is therefore
       repeated on subsets stratified to a common opening-valence histogram. This is the
       deflationary alternative to the paper's flagship claim, and it deserves a direct test
       rather than a caveat.

       READ THAT ARM WITH CARE. Conditioning on a noisy first-sentence score when the corpora sit
       at different mean valences induces regression toward each corpus's own mean, which biases
       the descent of the lower-mean corpus downward at any matched starting point. It is Lord's
       paradox, and it means opening-matching does not identify the contrast in either direction.
       We report it because a reader will ask, and we report why it cannot settle the question.
    6. SCALE-FREE DESCENT. The honest way to remove a dynamic-range difference without
       conditioning on a baseline: express each corpus's descent in units of its own
       within-document sentence-valence dispersion. A corpus sitting near the floor of the scale
       has less room to fall in raw units but the standardised statistic is comparable.

Each is contrasted dream-minus-comparator with authors resampled independently in both corpora,
BH-adjusted across the 5 x 3 family, and repeated on sentence-count-matched subsets.

Two honesty controls the contrast script did not carry:

    - COMPARATOR CLUSTERING. `author` is a post id for r/confession and an article id for
      Wikipedia (the HuggingFace sources carry no username field), so those corpora are clustered
      at the document level while the dream corpora are clustered at the person level. That makes
      the comparator interval too narrow by an unknown design effect, and the difference
      anti-conservative in the direction the paper claims. We report the design effect that would
      be needed to overturn each contrast, so the reader can price the risk.
    - SENTENCE CAP. Documents truncated at 25 sentences have an artificial "ending". We report the
      capped fraction per corpus and repeat the headline contrasts with capped documents dropped.

Aggregate-only output: no document text, no author keys.

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 \
        python3 analyses/2026-08-21-02-arrow-recovery.py
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
MIN_SENT_RECOV = int(os.environ.get("MIN_SENT_RECOV", 6))
B_BOOT = int(os.environ.get("B_BOOT", 4000))
N_MATCH_SEEDS = int(os.environ.get("N_MATCH_SEEDS", 50))
TAIL = 5

DREAMS = ["DreamSeer EN", "DreamSeer RU", "DreamBank", "Reddit r/Dreams", "SDDb"]
COMPARATORS = ["r/confession (waking narrative)", "Gutenberg fiction", "Wikipedia openings"]
# Corpora whose `author` key is a document/article id rather than a person.
DOC_CLUSTERED = {"r/confession (waking narrative)", "Wikipedia openings"}

STATS = ["step", "flatten", "drift", "slope", "pre", "v0", "sd", "vbar"]


def load(name):
    """Reconstruct per-document valence trajectories from the committed per-sentence cache.

    Replays the primary-instrument run's deterministic document selection and refuses to proceed
    unless the per-document sentence counts match the cache element-wise, which is what licenses
    slicing the cached valence vector by document.
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
    codes = pd.factorize(pd.Series(auth, dtype="object"))[0].astype(np.int32)
    return ser, codes, counts, kind


def per_doc(ser):
    """Closing step, opening step, pre-close descent, drift, half-slope gap, and edge profiles."""
    n = len(ser)
    step = np.empty(n); pre = np.empty(n); drift = np.empty(n)
    flatten = np.empty(n); slope = np.empty(n); v0 = np.empty(n)
    sd = np.empty(n); vbar = np.empty(n)
    tail = np.full((n, TAIL), np.nan)
    head = np.full((n, TAIL), np.nan)
    for i, v in enumerate(ser):
        step[i] = v[-1] - v[-2]
        pre[i] = v[-2] - v[0]
        drift[i] = v[-1] - v[0]
        v0[i] = v[0]
        sd[i] = v.std(ddof=1)
        vbar[i] = v.mean()
        slope[i] = np.polyfit(np.linspace(0, 1, len(v)), v, 1)[0]
        h = len(v) // 2
        a, b = v[:h], v[h:]
        sa = np.polyfit(np.linspace(0, 1, len(a)), a, 1)[0]
        sb = np.polyfit(np.linspace(0, 1, len(b)), b, 1)[0]
        flatten[i] = sb - sa
        tail[i] = v[-TAIL:]
        head[i] = v[:TAIL]
    return {"step": step, "pre": pre, "drift": drift, "slope": slope, "flatten": flatten,
            "v0": v0, "sd": sd, "vbar": vbar, "tail": tail, "head": head}


def prepare(name):
    """Per-document statistics for one corpus, checkpointed so a kill does not lose the work."""
    stem = TEXT_CORPORA[name][0]
    ck = C.INTERIM / f"arrow_recovery4_{stem}.npz"
    if ck.exists():
        z = np.load(ck)
        print(f"  [ckpt] {name}", flush=True)
        return {k: z[k] for k in STATS + ["tail", "head", "auth", "counts"]} | \
               {"kind": str(z["kind"])}
    ser, codes, counts, kind = load(name)
    keep = np.flatnonzero(counts >= MIN_SENT_RECOV)
    ser = [ser[i] for i in keep]
    codes, counts = codes[keep], counts[keep]
    rec = per_doc(ser)
    del ser
    gc.collect()
    rec |= {"auth": codes, "counts": counts, "kind": kind}
    np.savez(ck, **rec)
    return rec


def by_author(x, auth):
    """Author-level means, dropping documents with an undefined statistic."""
    ok = np.isfinite(x)
    df = pd.DataFrame({"a": auth[ok], "x": x[ok]})
    return df.groupby("a", sort=True)["x"].mean().to_numpy()


def boot_draws(a, B=B_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    return a[rng.integers(0, len(a), size=(B, len(a)))].mean(axis=1)


def ci(draws):
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def sign_flip_p(a, B=B_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    obs = abs(a.mean())
    null = np.abs((rng.choice([-1.0, 1.0], size=(B, len(a))) * a).mean(axis=1))
    return float((1 + (null >= obs).sum()) / (B + 1))


def recovery_ratio(step_a, pre_a, B=B_BOOT, seed=0):
    """step / -(pre), with the ratio re-formed inside each bootstrap draw so it carries a CI.

    Undefined when the pre-close descent is not negative; reported as NaN rather than papered
    over, because a corpus that does not descend cannot be said to recover.
    """
    ds = boot_draws(step_a, B, seed)
    dp = boot_draws(pre_a, B, seed + 1)
    obs = float(step_a.mean() / -pre_a.mean()) if pre_a.mean() < -1e-9 else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(dp < -1e-9, ds / -dp, np.nan)
    r = r[np.isfinite(r)]
    lo, hi = ci(r) if len(r) > 100 else (np.nan, np.nan)
    return obs, lo, hi


def contrast(dream_a, comp_a, B=B_BOOT, seed=0):
    """Bootstrap the dream-minus-comparator difference, authors resampled independently."""
    d = boot_draws(dream_a, B, seed)
    c = boot_draws(comp_a, B, seed + 7919)
    diff = d - c
    obs = float(dream_a.mean() - comp_a.mean())
    p = max(float(2 * min((diff >= 0).mean(), (diff <= 0).mean())), 1.0 / (B + 1))
    lo, hi = ci(diff)
    # Design effect that would be needed to overturn the contrast: inflate the comparator's
    # bootstrap spread by sqrt(k) until the difference interval covers zero.
    cm = c.mean()
    need = np.nan
    for k in np.arange(1.0, 20.01, 0.25):
        dd = d - (cm + (c - cm) * np.sqrt(k))
        l, h = ci(dd)
        if l <= 0 <= h:
            need = float(k)
            break
    return {"diff": obs, "diff_lo": lo, "diff_hi": hi, "p": p, "deff_to_overturn": need}


def bh(p):
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
    """Indices whose sentence-count histograms are identical by construction."""
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


def match_values(x_a, x_b, seed=0, bins=20):
    """Indices whose distributions of a continuous covariate are matched by binning.

    Bin edges come from the pooled distribution so both corpora are cut identically; within each
    bin we keep min(n_a, n_b) documents from each side. Bins outside the common support drop out,
    which is the intended behaviour: there is no honest comparison where the corpora do not
    overlap.
    """
    rng = np.random.default_rng(seed)
    edges = np.quantile(np.concatenate([x_a, x_b]), np.linspace(0, 1, bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    ba = np.digitize(x_a, edges[1:-1])
    bb = np.digitize(x_b, edges[1:-1])
    ia, ib = [], []
    for k in range(bins):
        pa = np.flatnonzero(ba == k)
        pb = np.flatnonzero(bb == k)
        m = min(len(pa), len(pb))
        if m == 0:
            continue
        ia.append(rng.choice(pa, m, replace=False))
        ib.append(rng.choice(pb, m, replace=False))
    if not ia:
        return np.array([], int), np.array([], int)
    return np.concatenate(ia), np.concatenate(ib)


def matched_over_seeds(D, K, stat, seeds=N_MATCH_SEEDS):
    """Repeat the count-matched contrast over many matching draws.

    A single stratified draw is one realisation of an arbitrary choice; reporting its p-value as
    though it were the matched result hides that variability. We report the median and range.
    """
    ps, ds = [], []
    for s in range(seeds):
        ia, ib = match_counts(D["counts"], K["counts"], seed=s)
        if not len(ia):
            continue
        r = contrast(by_author(D[stat][ia], D["auth"][ia]),
                     by_author(K[stat][ib], K["auth"][ib]), B=1000, seed=s)
        ps.append(r["p"]); ds.append(r["diff"])
    if not ps:
        return {}
    return {"mat_diff": float(np.median(ds)), "mat_p": float(np.median(ps)),
            "mat_p_lo": float(np.min(ps)), "mat_p_hi": float(np.max(ps)),
            "mat_n": int(len(ia))}


def main():
    store, rows, tails = {}, [], []
    for name in DREAMS + COMPARATORS + ["DreamSeer interp."]:
        try:
            rec = prepare(name)
        except Exception as e:
            print(f"[{name}] SKIP: {type(e).__name__}: {e}", flush=True)
            continue
        # Standardised descent: each document's drift and slope in units of the corpus's typical
        # within-document sentence-valence dispersion, so a corpus sitting low on the scale is
        # not penalised for having less room to fall.
        sd_pool = float(np.sqrt(np.nanmean(rec["sd"] ** 2)))
        rec["drift_z"] = rec["drift"] / sd_pool
        rec["slope_z"] = rec["slope"] / sd_pool
        store[name] = rec
        auth, counts = rec["auth"], rec["counts"]
        sa = {s: by_author(rec[s], auth) for s in STATS + ["drift_z", "slope_z"]}
        rr, rlo, rhi = recovery_ratio(sa["step"], sa["pre"])
        row = {"corpus": name, "kind": rec["kind"],
               "n_docs": int(len(counts)), "n_authors": int(len(sa["step"])),
               "author_unit": "document" if name in DOC_CLUSTERED else "person",
               "frac_at_cap": float((counts >= MAX_SENT).mean()),
               "sent_median": float(np.median(counts)),
               "sd_within": sd_pool,
               "recovery": rr, "recovery_lo": rlo, "recovery_hi": rhi}
        for s in STATS + ["drift_z", "slope_z"]:
            m = float(sa[s].mean())
            lo, hi = ci(boot_draws(sa[s]))
            row |= {f"{s}_auth": m, f"{s}_lo": lo, f"{s}_hi": hi, f"{s}_p": sign_flip_p(sa[s])}
        rows.append(row)
        # author-weighted edge profiles: position 1..5 from the start, and 5..1 from the end
        for edge, key in (("start", "head"), ("end", "tail")):
            for j in range(TAIL):
                av = by_author(rec[key][:, j], auth)
                lo, hi = ci(boot_draws(av))
                tails.append({"corpus": name, "edge": edge,
                              "pos": j + 1 if edge == "start" else TAIL - j,
                              "mean": float(av.mean()), "lo": lo, "hi": hi})
        print(f"  {name:34s} n={len(counts):5,} step={row['step_auth']:+.4f} "
              f"pre={row['pre_auth']:+.4f} recov={rr if rr==rr else float('nan'):+.3f} "
              f"flatten={row['flatten_auth']:+.4f}", flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(OUT / "arrow_recovery_corpora.csv", index=False)
    pd.DataFrame(tails).to_csv(OUT / "arrow_recovery_edges.csv", index=False)

    con = []
    for d in DREAMS:
        if d not in store:
            continue
        for c in COMPARATORS:
            if c not in store:
                continue
            D, K = store[d], store[c]
            for stat in ("step", "flatten", "drift", "slope", "drift_z", "slope_z"):
                raw = contrast(by_author(D[stat], D["auth"]), by_author(K[stat], K["auth"]))
                mat = matched_over_seeds(D, K, stat)
                # capped documents dropped: the "ending" of a truncated document is an artifact
                du = np.flatnonzero(D["counts"] < MAX_SENT)
                ku = np.flatnonzero(K["counts"] < MAX_SENT)
                unc = contrast(by_author(D[stat][du], D["auth"][du]),
                               by_author(K[stat][ku], K["auth"][ku]))
                # opening-valence matched: does the difference survive a common starting point?
                oa, ob = match_values(D["v0"], K["v0"])
                opn = contrast(by_author(D[stat][oa], D["auth"][oa]),
                               by_author(K[stat][ob], K["auth"][ob])) if len(oa) else {}
                con.append({"dream": d, "comparator": c, "stat": stat,
                            **{f"raw_{k}": v for k, v in raw.items()}, **mat,
                            "uncapped_diff": unc["diff"], "uncapped_p": unc["p"],
                            "openmatch_n": int(len(oa)),
                            "openmatch_diff": opn.get("diff", np.nan),
                            "openmatch_lo": opn.get("diff_lo", np.nan),
                            "openmatch_hi": opn.get("diff_hi", np.nan),
                            "openmatch_p": opn.get("p", np.nan)})
    con = pd.DataFrame(con)
    for stat in con.stat.unique():
        m = con.stat == stat
        con.loc[m, "raw_q"] = bh(con.loc[m, "raw_p"].to_numpy())
        con.loc[m, "mat_q"] = bh(con.loc[m, "mat_p"].to_numpy())
        con.loc[m, "openmatch_q"] = bh(con.loc[m, "openmatch_p"].to_numpy())
    con["pooled_q"] = bh(con["raw_p"].to_numpy())      # across all statistics jointly
    con.to_csv(OUT / "arrow_recovery_tests.csv", index=False)

    with open(OUT / "arrow_recovery.md", "w") as f:
        f.write("# Does the narrative resolve? Direct measurement of the closing sentence\n\n")
        f.write(f"True per-sentence XLM-R, documents of >={MIN_SENT_RECOV} sentences (capped at "
                f"{MAX_SENT}), author-weighted means, {B_BOOT}-draw cluster bootstrap.\n\n"
                "`step` = v[-1] - v[-2] (closing step). `pre` = v[-2] - v[0] (descent before the "
                "close). `recovery` = step / -pre. `flatten` = second-half minus first-half OLS "
                "slope.\n\n"
                "`author_unit` = document means the corpus has no person key and its interval is "
                "therefore too narrow by an unknown design effect; `deff_to_overturn` is the "
                "variance inflation that would be needed to make the contrast cross zero.\n\n")
        f.write("## Per corpus\n\n")
        f.write(per.to_markdown(index=False, floatfmt=".4f") + "\n\n")
        f.write("## Contrasts (dream minus comparator)\n\n")
        keep = ["dream", "comparator", "stat", "raw_diff", "raw_diff_lo", "raw_diff_hi",
                "raw_q", "pooled_q", "raw_deff_to_overturn",
                "mat_diff", "mat_q", "mat_p_lo", "mat_p_hi", "uncapped_diff", "uncapped_p",
                "openmatch_n", "openmatch_diff", "openmatch_lo", "openmatch_hi", "openmatch_q"]
        f.write(con[keep].to_markdown(index=False, floatfmt=".4f") + "\n\n")
        f.write("## Author-weighted edge profiles (first five and last five positions)\n\n")
        f.write(pd.DataFrame(tails).to_markdown(index=False, floatfmt=".4f") + "\n")
    print(f"[recovery] wrote {OUT/'arrow_recovery.md'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
