"""ARROW OF TIME under AUTHOR-CLUSTERED inference (+ a raised permutation budget).

The published external replication (`analyses/2026-07-22-01-arrow-arcs-external.py`) grouped its
cross-validation by DREAM and treated each report as an independent unit. Author concentration
makes that untenable in two of the five corpus-language samples: DreamBank's 15,000 analyzed
reports come from 50 dreamers, and a single SDDb participant contributes 31% of that sample. A
reviewer would rightly read the published n as an effective sample size, which it is not.

This re-runs the arrow with the author as the unit of clustering, changing nothing else:

  1. AUC   — GroupKFold by AUTHOR (no author in both train and test) and a group-wise 80/20
             split for the label-flip permutation, at B=10,000 instead of the published B=120
             (whose p floor of .008 the reviewer flagged).
  2. drift — cluster bootstrap over authors; the mean of per-author means (equal weight per
             author); and an author-level sign-flip permutation, which is the correct
             exchangeability unit when one dreamer can supply 1,853 reports.
  3. drift — one randomly chosen dream per author, averaged over 200 draws (n = #authors).
  4. Dreamseer EN/RU additionally with TRUE per-sentence XLM-R sentiment rather than the
     embedding-axis proxy (r~.27 per sentence), closing the measurement gap on the one corpus
     where true per-sentence scores exist.

Aggregate-only outputs. Author keys are re-attached in memory and never written (docs/ETHICS.md).

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 USE_TF=0 \
        /opt/anaconda3/bin/python3 analyses/2026-08-20-01-arrow-user-clustered.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from psychohistory import config as C
from psychohistory.dreams.arrow import (CORPORA, arrow_auc, drift_stats, load_corpus,
                                        one_per_author, series)
from psychohistory.dreams.valence import fit_valence_axis, project

OUT = C.RESULTS / "showcase"
OUT.mkdir(parents=True, exist_ok=True)
B_PERM = int(os.environ.get("B_PERM", 10000))
B_BOOT = int(os.environ.get("B_BOOT", 4000))


def true_xlmr_dreamseer():
    """Per-sentence XLM-R valence for the language-stratified Dreamseer subset, with userID.

    Returns {lang: (series, authors)} using the same >=4-sentence arrow filter. The cache scores
    a stratified subset (<=3,000 docs/language, 5..25 sentences); unscored docs are NaN.
    """
    from psychohistory.dreams.sentence_cache import load_or_build_sentences, offsets
    from psychohistory.dreams.sentence_sentiment import load_or_build_sentence_sentiment
    docs, counts, sent = load_or_build_sentence_sentiment("text", min_sent=5)
    _d, _c, _e, _nw, meta = load_or_build_sentences("text")
    off = offsets(counts)
    out = {}
    for lang in ("en", "ru"):
        sel = np.where((meta.lang.values == lang) & (counts >= 4))[0]
        ser, auth = [], []
        for k in sel:
            v = sent[off[k]:off[k] + counts[k]]
            if np.isfinite(v).all() and len(v) >= 4:
                ser.append(v.astype(float)); auth.append(meta.userID.values[k])
        out[lang] = (ser, np.array(auth, dtype=object))
    return out


def analyze(name, ser, authors, dream_groups=None, tag="embed_axis"):
    """One row: published dream-grouped AUC (reproduction) + author-clustered everything."""
    if dream_groups is None:
        dream_groups = np.arange(len(ser))
    auc_d, _p, _ = arrow_auc(ser, dream_groups, B=0)             # reproduction check only
    auc_a, p_a, n_auth = arrow_auc(ser, authors, B=B_PERM)
    ds = drift_stats(ser, authors, B=B_BOOT)
    opa = one_per_author(ser, authors)
    row = {"corpus": name, "proxy": tag, "n_dreams": ds["n_dreams"], "n_authors": n_auth,
           "auc_dreamgrouped": auc_d, "auc_authorgrouped": auc_a, "auc_p_authorgrouped": p_a,
           **{k: v for k, v in ds.items() if k not in ("n_dreams", "n_authors")},
           **{k: v for k, v in opa.items() if k != "n_authors"}}
    print(f"[{name}/{tag}] n={ds['n_dreams']:,} authors={n_auth:,} | "
          f"AUC dream={auc_d:.3f} author={auc_a:.3f} (p={p_a:.4f}) | "
          f"drift={ds['drift']:+.3f}[{ds['drift_lo']:+.3f},{ds['drift_hi']:+.3f}] "
          f"author-weighted={ds['drift_auth']:+.3f}"
          f"[{ds['drift_auth_lo']:+.3f},{ds['drift_auth_hi']:+.3f}] "
          f"p_cluster={ds['p_cluster']:.4f} | 1/author={opa['drift_1pa']:+.3f} | "
          f"authors darkening={ds['frac_authors_neg']:.1%}", flush=True)
    return row


def main():
    w, b, vinfo = fit_valence_axis()
    print(f"[valence] axis fit n={vinfo['n_fit']} R2={vinfo['r2_insample']:.3f}", flush=True)
    rows = []
    for name in CORPORA:
        try:
            cp = load_corpus(name)
        except Exception as e:
            print(f"[{name}] SKIP {e}", flush=True); continue
        proj = project(cp.emb, w, b)
        ser, idx = series(proj, cp.counts)
        rows.append(analyze(name, ser, cp.authors[idx]))

    # true per-sentence XLM-R on Dreamseer (the instrument the proxy stands in for)
    try:
        tx = true_xlmr_dreamseer()
        for lang, (ser, auth) in tx.items():
            if len(ser) >= 200:
                rows.append(analyze(f"DreamSeer {lang.upper()}", ser, auth, tag="true_xlmr"))
    except Exception as e:
        print(f"[true-xlmr] SKIP {e}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "arrow_user_clustered.csv", index=False)
    _write_md(df)
    print("\n[arrow-user-clustered] wrote", OUT / "arrow_user_clustered.md")


def _write_md(df):
    L = ["# The emotional arrow under author-clustered inference", "",
         "*Re-analysis of `analyses/2026-07-22-01-arrow-arcs-external.py` with the AUTHOR as the "
         "unit of clustering. Nothing else changes: same corpora, same filters, same sentence "
         "splitter, same five arc features, same valence axis. Author keys are re-attached in "
         "memory by replaying each loader; alignment to the cached per-sentence embeddings is "
         "asserted element-wise against the cached sentence counts.*", "",
         f"*Permutation budget B={B_PERM:,} (published run: B=120, p floor .008). "
         f"Cluster bootstrap B={B_BOOT:,}.*", "",
         "## Why this matters",
         "Analyzed reports per distinct author: DreamBank 15,000/50, SDDb 15,000/1,931 (one "
         "participant = 31%), Reddit 15,000/12,737, Dreamseer EN 9,459/1,795, RU 2,743/646. "
         "Dream-level inference implicitly claims the first two corpora carry 15,000 independent "
         "observations each; they do not.", "",
         "## Arrow of time — dream-grouped vs author-grouped",
         "| corpus | proxy | n dreams | n authors | AUC (dream-grouped) | AUC (author-grouped) | "
         "perm p | end−start drift [cluster 95% CI] | author-weighted drift [95% CI] | "
         "author-level p | 1 dream/author | authors darkening |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for _, r in df.iterrows():
        L.append(
            f"| {r.corpus} | {r.proxy} | {int(r.n_dreams):,} | {int(r.n_authors):,} | "
            f"{r.auc_dreamgrouped:.3f} | **{r.auc_authorgrouped:.3f}** | {r.auc_p_authorgrouped:.4f} | "
            f"{r.drift:+.3f} [{r.drift_lo:+.3f},{r.drift_hi:+.3f}] | "
            f"{r.drift_auth:+.3f} [{r.drift_auth_lo:+.3f},{r.drift_auth_hi:+.3f}] | "
            f"{r.p_cluster:.4f} | {r.drift_1pa:+.3f} [{r.drift_1pa_lo:+.3f},{r.drift_1pa_hi:+.3f}] | "
            f"{r.frac_authors_neg:.0%} |")
    dream = df[~df.corpus.str.contains("interp")]
    L += ["", "## Read-out",
          f"- Author-grouped AUC stays above 0.5 in {int((dream.auc_authorgrouped>0.5).sum())}/"
          f"{len(dream)} dream rows; the drift stays negative in "
          f"{int((dream.drift_auth<0).sum())}/{len(dream)} under equal author weighting.",
          "- The author-weighted drift and the one-dream-per-author estimate are the honest "
          "effect sizes; the cluster CI is the honest precision.",
          "", "*Aggregate-only. No author key, user ID or dream text is written by this script.*"]
    (OUT / "arrow_user_clustered.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
