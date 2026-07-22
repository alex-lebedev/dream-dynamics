"""Publication-grade figures for the physics-&-structure-of-dreams paper.

Re-renders the CORE pillars with the shared `plotstyle` (colorblind-safe, 400-dpi PNG + vector PDF,
panel letters, non-misleading axes) — WITHOUT touching the analysis scripts. Data come from the
committed result CSVs where available; the intrinsic-dimension panel is recomputed on the cached
embeddings so the plotted quantity MATCHES the headline (fixes the old fig-44 two-NN/D2 mismatch).

Outputs -> 60-results/showcase/pub/ :
    fig1_scaling_manifold.{png,pdf}   Zipf + dream-length + intrinsic-dim-vs-nulls   (F0054)
    fig2_geometry.{png,pdf}           percolation continent + composition control    (F0035)
    fig3_arcs.{png,pdf}               SVD arc basis + prototype shapes                (F0036)
    (arrow-of-time & cross-corpus arcs = 45_arrow_arcs_external.{png,pdf}, from -22-01)

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 USE_TF=0 python3 \
        analyses/2026-07-22-02-figures-publication.py
"""
from __future__ import annotations

import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psychohistory import config as C
from psychohistory.utils.plotstyle import apply_style, PALETTE, panel_label, save_fig

SHOW = C.RESULTS / "showcase"
PUB = SHOW / "pub"
PUB.mkdir(parents=True, exist_ok=True)


# ---- TwoNN intrinsic dimension (Facco et al. 2017) — identical to analyses/2026-07-21-04 -------
def two_nn_dim(X, sample=8000, seed=0):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    if len(X) > sample:
        X = X[rng.choice(len(X), sample, replace=False)]
    tree = cKDTree(X)
    dd, _ = tree.query(X, k=3)
    r1, r2 = dd[:, 1], dd[:, 2]
    ok = (r1 > 0)
    mu = np.sort(r2[ok] / r1[ok])
    Femp = np.arange(1, len(mu) + 1) / len(mu)
    m = Femp < 0.9
    d = np.polyfit(np.log(mu[m]), -np.log(1 - Femp[m]), 1)[0]
    return float(d)


def _en_text():
    lv = pd.read_parquet(C.DREAMS_OUT / "dreamseer_dream_level.parquet", columns=["documentID", "lang"])
    lv = lv[lv.lang == "en"]
    raw = pd.read_csv(C.DREAMSEER_RAW, dtype=str, keep_default_na=False, usecols=["documentID", "text"])
    d = lv.merge(raw, on="documentID")
    return [t for t in d.text.tolist() if isinstance(t, str) and t.strip()]


# ---- Figure 1: scaling laws + low-dimensional manifold (F0054) --------------------------------
def fig1_scaling_manifold():
    texts = _en_text()
    tok = re.compile(r"[a-z']+")
    freqs = Counter()
    lengths = []
    for t in texts:
        ws = tok.findall(t.lower())
        lengths.append(len(ws))
        freqs.update(ws)
    ranks = np.arange(1, len(freqs) + 1)
    fvals = np.array(sorted(freqs.values(), reverse=True))
    lo, hi = 10, min(1000, len(fvals))
    zexp = -np.polyfit(np.log(ranks[lo:hi]), np.log(fvals[lo:hi]), 1)[0]
    lengths = np.array([l for l in lengths if l > 0])

    from psychohistory.dreams.embed_cache import load_or_build
    meta, emb = load_or_build()
    E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    Xen = E[meta.lang.values == "en"]
    dnn = np.median([two_nn_dim(Xen, seed=s) for s in range(5)])
    rng = np.random.default_rng(1)
    G = rng.standard_normal((min(len(Xen), 8000), E.shape[1])); G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-9
    dgauss = two_nn_dim(G)
    Sh = np.column_stack([rng.permutation(Xen[:, j]) for j in range(Xen.shape[1])])
    dshuf = two_nn_dim(Sh)

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    # (a) Zipf
    ax[0].loglog(ranks, fvals, ".", ms=2.2, color=PALETTE["blue"], alpha=0.7)
    xf = np.array([ranks[lo], ranks[hi - 1]])
    ax[0].loglog(xf, fvals[lo] * (xf / ranks[lo]) ** (-zexp), color=PALETTE["red"], lw=2,
                 label=f"slope {zexp:.2f}")
    ax[0].set_xlabel("word rank"); ax[0].set_ylabel("frequency")
    ax[0].set_title("Zipf's law"); ax[0].legend(loc="upper right")
    panel_label(ax[0], "a")
    # (b) dream length distribution (log scale) + log-normal fit line
    ax[1].hist(lengths, bins=np.logspace(0, np.log10(lengths.max()), 40),
               color=PALETTE["sky"], edgecolor="white", linewidth=0.3)
    ax[1].set_xscale("log")
    ax[1].axvline(np.median(lengths), color=PALETTE["red"], lw=1.6, ls="--",
                  label=f"median {int(np.median(lengths))} w")
    ax[1].set_xlabel("dream length (words)"); ax[1].set_ylabel("dreams")
    ax[1].set_title("Heavy-tailed length"); ax[1].legend()
    panel_label(ax[1], "b")
    # (c) intrinsic dimension vs nulls (CORRECTED headline panel)
    labels = ["real\ndreams", "feature\nshuffle", "isotropic\nGaussian", "ambient"]
    vals = [dnn, dshuf, dgauss, E.shape[1]]
    cols = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["grey"]]
    bars = ax[2].bar(labels, vals, color=cols)
    for b, v in zip(bars, vals):
        ax[2].text(b.get_x() + b.get_width() / 2, v + 4, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax[2].set_ylabel("intrinsic dimension (two-NN)")
    ax[2].set_title("Low-dimensional manifold"); ax[2].set_ylim(0, E.shape[1] * 1.12)
    panel_label(ax[2], "c")
    fig.suptitle("Dreams obey natural scaling laws and occupy a low-dimensional manifold",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, PUB / "fig1_scaling_manifold")
    print(f"[fig1] Zipf {zexp:.2f} | median len {np.median(lengths):.0f} | two-NN {dnn:.0f} "
          f"(shuf {dshuf:.0f}, gauss {dgauss:.0f}, ambient {E.shape[1]})", flush=True)


# ---- Figure 2: percolation continent + composition-controlled geometry (F0035) ----------------
def fig2_geometry():
    perc = pd.read_csv(SHOW / "collective_percolation.csv")
    pan = pd.read_csv(SHOW / "collective_expanding_universe_panel.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(perc.theta.to_numpy(), (perc.giant_frac * 100).to_numpy(), "o-",
               color=PALETTE["blue"], label="real dreams")
    ax[0].axhline(0, color=PALETTE["orange"], lw=2, ls="--", label="feature-shuffle null (≈0%)")
    ax[0].set_xlabel("cosine threshold θ"); ax[0].set_ylabel("giant component (% of dreams)")
    ax[0].set_title("One dense 'continent'"); ax[0].legend(); ax[0].set_ylim(-4, 104)
    panel_label(ax[0], "a")
    # stable-panel geometry over time (composition control): participation ratio ~ flat
    m = pd.to_datetime(pan.month + "-01").to_numpy()
    pr = ((pan.participation_ratio - pan.participation_ratio.mean()) / pan.participation_ratio.std()).to_numpy()
    mp = ((pan.mean_pairwise_dist - pan.mean_pairwise_dist.mean()) / pan.mean_pairwise_dist.std()).to_numpy()
    ax[1].plot(m, pr, "o-", color=PALETTE["blue"], ms=3, label="participation ratio (z)")
    ax[1].plot(m, mp, "s-", color=PALETTE["green"], ms=3, label="mean pairwise dist (z)")
    ax[1].axhline(0, color=PALETTE["black"], lw=0.6)
    ax[1].set_xlabel("month"); ax[1].set_ylabel("stable-panel geometry (z)")
    ax[1].set_title("Canalization is composition (stable panel ~ flat)"); ax[1].legend(fontsize=8)
    fig.autofmt_xdate(rotation=45)
    panel_label(ax[1], "b")
    fig.suptitle("Dream-space is a single dense continent; apparent contraction is user turnover",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, PUB / "fig2_geometry")
    print("[fig2] percolation + stable-panel geometry rendered", flush=True)


# ---- Figure 3: emotional-arc basis (F0036) — NAMED shapes (Reagan-style) ----------------------
def _name_shape(p):
    """Label a normalized arc shape (Reagan et al. 2016 vocabulary)."""
    p = np.asarray(p, float)
    a, m, z = p[:3].mean(), p[len(p) // 2 - 1:len(p) // 2 + 1].mean(), p[-3:].mean()
    if z - a > 0.12:
        return "rise (rags→riches)"
    if a - z > 0.12:
        return "fall (tragedy)"
    if m < a - 0.08 and m < z - 0.08:
        return "man-in-a-hole (fall→rise)"
    if m > a + 0.08 and m > z + 0.08:
        return "Icarus (rise→fall)"
    return "flat / steady"


def fig3_arcs():
    modes = pd.read_csv(SHOW / "arcs_svd_modes_en.csv")
    protos = pd.read_csv(SHOW / "arcs_prototypes_en.csv")
    x = np.linspace(0, 1, len(modes))
    # try to size the prototypes (dreams per cluster) from the KMeans md, else equal weight
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    var_lbl = {"mode1": "mode 1 (25%)", "mode2": "mode 2 (19%)", "mode3": "mode 3 (14%)"}
    for i, c in enumerate(["mode1", "mode2", "mode3"]):
        ax[0].plot(x, modes[c].to_numpy(), lw=2,
                   color=[PALETTE["blue"], PALETTE["orange"], PALETTE["green"]][i], label=var_lbl[c])
    ax[0].axhline(0, color=PALETTE["grey"], lw=0.6)
    ax[0].set_xlabel("normalized dream time"); ax[0].set_ylabel("valence (shape)")
    ax[0].set_title("Low-dimensional arc basis (SVD modes)"); ax[0].legend(loc="lower left")
    panel_label(ax[0], "a")
    # prototypes: label each by its Reagan-style shape, order by shape for a clean legend
    pal = [PALETTE[k] for k in ("blue", "orange", "green", "red", "purple", "sky")]
    cols = list(protos.columns)
    order = sorted(range(len(cols)), key=lambda j: _name_shape(protos[cols[j]]))
    seen = {}
    for k, j in enumerate(order):
        c = cols[j]; nm = _name_shape(protos[c])
        seen[nm] = seen.get(nm, 0) + 1
        lbl = nm if seen[nm] == 1 else f"{nm} ({seen[nm]})"
        ax[1].plot(x, protos[c].to_numpy(), color=pal[k % 6], lw=2, label=lbl)
    ax[1].axhline(0, color=PALETTE["grey"], lw=0.6)
    ax[1].set_xlabel("normalized dream time"); ax[1].set_ylabel("valence (shape)")
    ax[1].set_title("Six prototype dream arcs (named)")
    ax[1].legend(loc="upper right", fontsize=8, ncol=1)
    panel_label(ax[1], "b")
    fig.suptitle("Dream emotional arcs reduce to a few basic shapes (Reagan-style), dominated by the fall",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, PUB / "fig3_arcs")
    print("[fig3] arc basis rendered (named shapes)", flush=True)


def main():
    apply_style()
    fig2_geometry()
    fig3_arcs()
    fig1_scaling_manifold()   # heaviest (tokenize + two-NN) last
    print("[pub-figures] done ->", PUB)


if __name__ == "__main__":
    main()
