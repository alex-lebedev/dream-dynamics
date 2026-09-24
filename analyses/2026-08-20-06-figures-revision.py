"""Figures for the two revision claims that the original figure set could not support.

The published Figure 1c compared the intrinsic dimension of dream-space against an isotropic
Gaussian and a feature shuffle. Both are synthetic, and both destroy exactly the cross-coordinate
dependence that makes an embedding of natural language an embedding of natural language, so the
panel establishes `real semantic embeddings != independent coordinates` rather than anything about
dreams. The published Figure 3 measured the emotional arrow with a document-fit valence proxy on
one unit of clustering (the report) with no waking-human comparator.

This script renders the two panels that close those gaps:

    fig1_scaling_manifold.{png,pdf}   REBUILT as 2x2: Zipf, length, ID vs synthetic nulls,
                                      and ID vs MATCHED natural language (the new panel d)
    fig2_geometry.{png,pdf}           REBUILT as 1x3: percolation vs null, percolation across
                                      dream and non-dream corpora, stable-panel composition
    fig4_arrow_robustness.{png,pdf}   NEW: author-clustered drift, the awakening-artifact arms,
                                      and dream vs waking-narrative drift under true XLM-R

Reads only committed aggregate CSVs plus the embedding cache; writes no per-document quantity.

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 \
        python3 analyses/2026-08-20-06-figures-revision.py
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

# Short display names: the figures are crowded and the full corpus labels do not fit.
SHORT = {
    "Dreamseer EN (reference)": "Dreamseer EN",
    "Reddit r/confession (personal narrative)": "r/confession",
    "Project Gutenberg fiction": "fiction",
    "Wikipedia openings": "Wikipedia",
    "Dreamseer LLM interpretations": "LLM interp.",
    "Reddit r/Dreams": "r/Dreams",
    "DreamSeer EN": "Dreamseer EN",
    "DreamSeer RU": "Dreamseer RU",
    "DreamSeer interp.": "LLM interp.",
    "r/confession (waking narrative)": "r/confession",
    "Gutenberg fiction": "fiction",
    "Wikipedia openings ": "Wikipedia",
}
KIND_COLOR = {"dream": PALETTE["blue"], "human non-dream": PALETTE["green"],
              "machine prose": PALETTE["purple"], "synthetic": PALETTE["orange"],
              "waking narrative": PALETTE["green"], "written narrative": PALETTE["sky"],
              "non-narrative": PALETTE["grey"], "waking": PALETTE["purple"]}


def short(name):
    return SHORT.get(str(name).strip(), str(name))


# ---- Figure 1: scaling, length, and intrinsic dimension against both null families -------------
def two_nn_dim(X, sample=8000, seed=0):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    if len(X) > sample:
        X = X[rng.choice(len(X), sample, replace=False)]
    tree = cKDTree(X)
    dd, _ = tree.query(X, k=3)
    r1, r2 = dd[:, 1], dd[:, 2]
    ok = r1 > 0
    mu = np.sort(r2[ok] / r1[ok])
    F = np.arange(1, len(mu) + 1) / len(mu)
    m = F < 0.9
    return float(np.polyfit(np.log(mu[m]), -np.log(1 - F[m]), 1)[0])


def _en_text():
    lv = pd.read_parquet(C.DREAMS_OUT / "dreamseer_dream_level.parquet",
                         columns=["documentID", "lang"])
    lv = lv[lv.lang == "en"]
    raw = pd.read_csv(C.DREAMSEER_RAW, dtype=str, keep_default_na=False,
                      usecols=["documentID", "text"])
    d = lv.merge(raw, on="documentID")
    return [t for t in d.text.tolist() if isinstance(t, str) and t.strip()]


def fig1():
    texts = _en_text()
    tok = re.compile(r"[a-z']+")
    freqs, lengths = Counter(), []
    for t in texts:
        ws = tok.findall(t.lower())
        lengths.append(len(ws))
        freqs.update(ws)
    ranks = np.arange(1, len(freqs) + 1)
    fvals = np.array(sorted(freqs.values(), reverse=True))
    lo, hi = 10, min(1000, len(fvals))
    zexp = -np.polyfit(np.log(ranks[lo:hi]), np.log(fvals[lo:hi]), 1)[0]
    lengths = np.array([x for x in lengths if x > 0])

    from psychohistory.dreams.embed_cache import load_or_build
    meta, emb = load_or_build()
    E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    Xen = E[meta.lang.values == "en"]
    dnn = np.median([two_nn_dim(Xen, seed=s) for s in range(5)])
    rng = np.random.default_rng(1)
    G = rng.standard_normal((min(len(Xen), 8000), E.shape[1]))
    G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-9
    dgauss = two_nn_dim(G)
    dshuf = two_nn_dim(np.column_stack([rng.permutation(Xen[:, j]) for j in range(Xen.shape[1])]))

    gb = pd.read_csv(SHOW / "geometry_language_baseline.csv")

    fig, axs = plt.subplots(2, 2, figsize=(12.4, 8.6))
    ax = axs.ravel()

    ax[0].loglog(ranks, fvals, ".", ms=2.2, color=PALETTE["blue"], alpha=0.7)
    xf = np.array([ranks[lo], ranks[hi - 1]])
    ax[0].loglog(xf, fvals[lo] * (xf / ranks[lo]) ** (-zexp), color=PALETTE["red"], lw=2,
                 label=f"slope {zexp:.2f}")
    ax[0].set_xlabel("word rank"); ax[0].set_ylabel("frequency")
    ax[0].set_title("Zipf's law"); ax[0].legend(loc="upper right")
    panel_label(ax[0], "a")

    ax[1].hist(lengths, bins=np.logspace(0, np.log10(lengths.max()), 40),
               color=PALETTE["sky"], edgecolor="white", linewidth=0.3)
    ax[1].set_xscale("log")
    ax[1].axvline(np.median(lengths), color=PALETTE["red"], lw=1.6, ls="--",
                  label=f"median {int(np.median(lengths))} w")
    ax[1].set_xlabel("report length (words)"); ax[1].set_ylabel("reports")
    ax[1].set_title("Heavy-tailed length"); ax[1].legend()
    panel_label(ax[1], "b")

    labels = ["real\ndreams", "feature\nshuffle", "isotropic\nGaussian", "ambient"]
    vals = [dnn, dshuf, dgauss, E.shape[1]]
    cols = [PALETTE["blue"], PALETTE["orange"], PALETTE["orange"], PALETTE["grey"]]
    bars = ax[2].bar(labels, vals, color=cols)
    for b, v in zip(bars, vals):
        ax[2].text(b.get_x() + b.get_width() / 2, v + 6, f"{v:.0f}", ha="center", fontsize=9,
                   fontweight="bold")
    ax[2].set_ylabel("intrinsic dimension (two-NN)")
    ax[2].set_title("Against synthetic nulls"); ax[2].set_ylim(0, E.shape[1] * 1.14)
    panel_label(ax[2], "c")

    # (d) the comparison the synthetic nulls cannot make: matched human non-dream language.
    # Sorted purely by estimate, so the reader sees a single continuum rather than a grouping
    # imposed by us; the answer to panel c's implied question is that there is no dream band.
    m = gb[(gb.variant == "lenmatched") | (gb.corpus.str.startswith("Dreamseer EN"))].copy()
    m = m[~m.kind.str.contains("synthetic")].sort_values("twonn")
    m["lbl"] = [short(c) for c in m.corpus]
    y = np.arange(len(m))
    ax[3].barh(y, m.twonn.to_numpy(),
               xerr=[m.twonn.to_numpy() - m.twonn_lo.to_numpy(),
                     m.twonn_hi.to_numpy() - m.twonn.to_numpy()],
               color=[KIND_COLOR.get(k, PALETTE["grey"]) for k in m.kind],
               error_kw={"lw": 1.0, "capsize": 2})
    ax[3].set_yticks(y); ax[3].set_yticklabels(m.lbl.tolist(), fontsize=8.5)
    ax[3].set_xlabel("intrinsic dimension (two-NN), length-matched")
    ax[3].set_title("Against matched natural language")
    ax[3].set_xlim(0, float(m.twonn.max()) * 1.42)
    ax[3].annotate(f"synthetic nulls: {dshuf:.0f} and {dgauss:.0f}\n(off scale, panel c)",
                   xy=(0.975, 0.06), xycoords="axes fraction", ha="right", fontsize=7.8,
                   color=PALETTE["orange"], fontweight="bold")
    for k in ["dream", "human non-dream", "machine prose"]:
        if (m.kind == k).any():
            ax[3].scatter([], [], color=KIND_COLOR[k], marker="s", s=42, label=k)
    ax[3].legend(loc="upper right", fontsize=8, framealpha=0.92)
    panel_label(ax[3], "d")

    fig.suptitle("Dream text obeys ordinary language statistics, and so does its semantic "
                 "geometry: every natural-language corpus is thin, dream or not",
                 fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, PUB / "fig1_scaling_manifold")
    print(f"[fig1] Zipf {zexp:.2f} | two-NN {dnn:.0f} (shuf {dshuf:.0f}, gauss {dgauss:.0f}) "
          f"| {len(m)} baseline rows", flush=True)


# ---- Figure 2: percolation against the null, across corpora, and the composition control -------
def fig2():
    perc = pd.read_csv(SHOW / "collective_percolation.csv")
    pan = pd.read_csv(SHOW / "collective_expanding_universe_panel.csv")
    gb = pd.read_csv(SHOW / "geometry_language_baseline.csv")

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))

    ax[0].plot(perc.theta.to_numpy(), (perc.giant_frac * 100).to_numpy(), "o-",
               color=PALETTE["blue"], label="real dreams")
    ax[0].axhline(0, color=PALETTE["orange"], lw=2, ls="--", label="feature-shuffle null (≈0%)")
    ax[0].set_xlabel("cosine threshold θ"); ax[0].set_ylabel("giant component (% of reports)")
    ax[0].set_title("One dense continent"); ax[0].legend(); ax[0].set_ylim(-4, 104)
    panel_label(ax[0], "a")

    # (b) is the connectivity a property of dreams or of coherent language? Two thresholds, all
    # corpora, length-matched where a matched variant exists. Coloured by kind rather than by
    # threshold, because the question the panel answers is which corpora group together.
    m = gb[(gb.variant == "lenmatched") | (gb.corpus.str.startswith("Dreamseer EN"))].copy()
    m = m[~m.kind.str.contains("synthetic")].sort_values("giant_050", ascending=False)
    m["lbl"] = [short(c) for c in m.corpus]
    x = np.arange(len(m))
    cols = [KIND_COLOR.get(k, PALETTE["grey"]) for k in m.kind]
    ax[1].bar(x - 0.19, (m.giant_050 * 100).to_numpy(), width=0.38, color=cols)
    ax[1].bar(x + 0.19, (m.giant_059 * 100).to_numpy(), width=0.38, color=cols, alpha=0.45)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(m.lbl.tolist(), rotation=35, ha="right", fontsize=8.5)
    ax[1].set_ylabel("giant component (%)"); ax[1].set_ylim(0, 116)
    ax[1].set_title("Connectivity across corpora")
    for k in ["dream", "human non-dream", "machine prose"]:
        if (m.kind == k).any():
            ax[1].scatter([], [], color=KIND_COLOR[k], marker="s", s=40, label=k)
    ax[1].bar([], [], color=PALETTE["grey"], label="left θ=0.50, right θ=0.59")
    ax[1].legend(fontsize=7.2, ncol=2, loc="upper right", framealpha=0.92)
    panel_label(ax[1], "b")

    mm = pd.to_datetime(pan.month + "-01").to_numpy()
    pr = ((pan.participation_ratio - pan.participation_ratio.mean())
          / pan.participation_ratio.std()).to_numpy()
    mp = ((pan.mean_pairwise_dist - pan.mean_pairwise_dist.mean())
          / pan.mean_pairwise_dist.std()).to_numpy()
    ax[2].plot(mm, pr, "o-", color=PALETTE["blue"], ms=3, label="participation ratio (z)")
    ax[2].plot(mm, mp, "s-", color=PALETTE["green"], ms=3, label="mean pairwise dist (z)")
    ax[2].axhline(0, color=PALETTE["black"], lw=0.6)
    ax[2].set_xlabel("month"); ax[2].set_ylabel("stable-panel geometry (z)")
    ax[2].set_title("Contraction is composition"); ax[2].legend(fontsize=8)
    fig.autofmt_xdate(rotation=40)
    panel_label(ax[2], "c")

    fig.suptitle("Dream-space is one connected continent — but so is every narrative corpus, "
                 "and the apparent contraction over calendar time is user turnover",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, PUB / "fig2_geometry")
    print(f"[fig2] percolation + {len(m)} corpora + stable panel", flush=True)


# ---- Figure 4: what the arrow survives ---------------------------------------------------------
# Panel-d label offsets, in points. SDDb and fiction land almost on top of each other and
# Dreamseer EN sits at the corner, so a few labels need placing by hand.
NUDGE = {"SDDb": (6, 7), "fiction": (6, -11), "Wikipedia": (-10, 9), "r/confession": (7, 5),
         "Dreamseer EN": (9, 4), "DreamBank": (9, -9), "Dreamseer RU": (8, 5),
         "r/Dreams": (8, -9)}

ARM_ORDER = ["full", "no-wake-sentences", "drop-last-sentence", "drop-last-2-sentences",
             "interior (drop first+last)", "reports NOT ending at awakening",
             "reports ending at awakening", "no-nightmare (drop top-decile threat)"]
ARM_SHORT = {"full": "full report", "no-wake-sentences": "drop awakening sentences",
             "drop-last-sentence": "drop last sentence",
             "drop-last-2-sentences": "drop last two",
             "interior (drop first+last)": "interior only",
             "reports NOT ending at awakening": "not ending awake",
             "reports ending at awakening": "ending awake",
             "no-nightmare (drop top-decile threat)": "drop nightmares"}


def fig4():
    uc = pd.read_csv(SHOW / "arrow_user_clustered.csv")
    aw = pd.read_csv(SHOW / "arrow_awakening_audit.csv")
    tx = pd.read_csv(SHOW / "arrow_true_xlmr.csv")

    rc = pd.read_csv(SHOW / "arrow_recovery_corpora.csv")
    re_edge = pd.read_csv(SHOW / "arrow_recovery_edges.csv")

    fig, ax = plt.subplots(3, 2, figsize=(13.6, 14.2))
    ax = ax.ravel()

    # (a) author-clustered drift: does the arrow survive equal weight per contributor?
    u = uc[uc.proxy == "embed_axis"].copy()
    u["lbl"] = [f"{short(c)}  ({int(n):,})" for c, n in zip(u.corpus, u.n_authors)]
    u = u.sort_values("drift_auth")
    y = np.arange(len(u))
    cols = [PALETTE["purple"] if "interp" in str(c) else PALETTE["blue"] for c in u.corpus]
    ax[0].barh(y, u.drift_auth.to_numpy(),
               xerr=[u.drift_auth.to_numpy() - u.drift_auth_lo.to_numpy(),
                     u.drift_auth_hi.to_numpy() - u.drift_auth.to_numpy()],
               color=cols, error_kw={"lw": 1.1, "capsize": 2.5})
    ax[0].axvline(0, color=PALETTE["black"], lw=0.9)
    ax[0].set_yticks(y); ax[0].set_yticklabels(u.lbl.tolist(), fontsize=8.5)
    ax[0].set_xlabel("author-weighted end − start valence")
    ax[0].set_title("Clustered on the contributor")
    ax[0].text(0.99, 0.02, "(n contributors)", transform=ax[0].transAxes, ha="right",
               fontsize=7.5, color=PALETTE["grey"])
    panel_label(ax[0], "a")

    # (b) the awakening arms: delete the ending and the descent should vanish if it is an artifact
    a = aw[(aw.proxy == "embed_axis") & (aw.corpus != "DreamSeer interp.")].copy()
    arms = [x for x in ARM_ORDER if x in set(a.arm)]
    corpora = sorted(a.corpus.unique(), key=lambda c: short(c))
    w = 0.8 / len(corpora)
    for j, c in enumerate(corpora):
        s = a[a.corpus == c].set_index("arm").reindex(arms)
        ax[1].bar(np.arange(len(arms)) + (j - (len(corpora) - 1) / 2) * w,
                  s.drift_auth.to_numpy(), width=w, label=short(c))
    ax[1].axhline(0, color=PALETTE["black"], lw=0.9)
    ax[1].set_xticks(np.arange(len(arms)))
    ax[1].set_xticklabels([ARM_SHORT.get(x, x) for x in arms], rotation=35, ha="right",
                          fontsize=8)
    ax[1].set_ylabel("author-weighted end − start valence")
    ax[1].set_title("Deleting the ending")
    ax[1].legend(fontsize=7.5, ncol=2, loc="upper left")
    panel_label(ax[1], "b")

    # (c) the comparator that puts the descent on a scale. The message is the GAP between the
    # dream corpora and the waking-narrative baseline, so the baseline is drawn as a reference.
    t = tx[tx.arm == "full"].copy()
    # The LLM corpus was written to CSV under an earlier kind label; it is machine prose.
    t.loc[t.corpus.str.contains("interp"), "kind"] = "machine prose"
    t["lbl"] = [f"{short(c)}  ({int(n):,})" for c, n in zip(t.corpus, t.n_authors)]
    t = t.sort_values("drift_auth")
    y = np.arange(len(t))
    base = float(t.loc[t.corpus.str.startswith("r/confession"), "drift_auth"].iloc[0])
    ax[2].axvspan(base, 0, color=PALETTE["grey"], alpha=0.16, zorder=0)
    ax[2].axvline(base, color=PALETTE["green"], lw=1.4, ls="--", zorder=1,
                  label=f"waking-narrative baseline ({base:+.3f})")
    ax[2].barh(y, t.drift_auth.to_numpy(),
               xerr=[np.abs(t.drift_auth.to_numpy() - t.drift_auth_lo.to_numpy()),
                     np.abs(t.drift_auth_hi.to_numpy() - t.drift_auth.to_numpy())],
               color=[KIND_COLOR.get(k, PALETTE["grey"]) for k in t.kind], zorder=2,
               error_kw={"lw": 1.1, "capsize": 2.5, "zorder": 3})
    ax[2].axvline(0, color=PALETTE["black"], lw=0.9, zorder=1)
    ax[2].set_yticks(y); ax[2].set_yticklabels(t.lbl.tolist(), fontsize=8.5)
    ax[2].set_xlabel("author-weighted end − start valence (true XLM-R)")
    ax[2].set_title("Dreams against matched non-dream prose")
    seen = []
    for k in t.kind:
        if k not in seen:
            ax[2].scatter([], [], color=KIND_COLOR.get(k, PALETTE["grey"]), marker="s", s=42,
                          label=k)
            seen.append(k)
    ax[2].legend(fontsize=7.5, loc="lower right", ncol=1, framealpha=0.92)
    panel_label(ax[2], "c")

    # (d) the same contrast on a scale-free footing. Raw valence units reward a corpus for
    # starting high; dividing by each corpus's own within-document sentence dispersion asks how
    # deep the descent is relative to that corpus's ordinary sentence-to-sentence movement.
    z = rc[~rc.corpus.str.contains("interp")].copy()
    z["lbl"] = [short(c) for c in z.corpus]
    z = z.sort_values("drift_z_auth")
    y = np.arange(len(z))
    zbase = float(z.loc[z.corpus.str.startswith("r/confession"), "drift_z_auth"].iloc[0])
    ax[3].axvspan(zbase, 0, color=PALETTE["grey"], alpha=0.16, zorder=0)
    ax[3].axvline(zbase, color=PALETTE["green"], lw=1.4, ls="--", zorder=1,
                  label=f"waking-narrative baseline ({zbase:+.2f} SD)")
    ax[3].barh(y, z.drift_z_auth.to_numpy(),
               xerr=[np.abs(z.drift_z_auth.to_numpy() - z.drift_z_lo.to_numpy()),
                     np.abs(z.drift_z_hi.to_numpy() - z.drift_z_auth.to_numpy())],
               color=[KIND_COLOR.get(k, PALETTE["grey"]) for k in z.kind], zorder=2,
               error_kw={"lw": 1.1, "capsize": 2.5, "zorder": 3})
    ax[3].axvline(0, color=PALETTE["black"], lw=0.9, zorder=1)
    ax[3].set_yticks(y); ax[3].set_yticklabels(z.lbl.tolist(), fontsize=8.5)
    ax[3].set_xlabel("end − start, in within-document SD of sentence valence")
    ax[3].set_title("The descent on a scale-free footing")
    ax[3].legend(fontsize=7.5, loc="upper left", framealpha=0.92)
    panel_label(ax[3], "d")

    # (e) the hypothesis we rejected. If dream reports differed from waking narrative by failing
    # to resolve, their closing step would sit below the waking baseline. It does not.
    s = rc.copy()
    s["lbl"] = [short(c) for c in s.corpus]
    s = s.sort_values("step_auth")
    y = np.arange(len(s))
    sbase = float(s.loc[s.corpus.str.startswith("r/confession"), "step_auth"].iloc[0])
    ax[4].axvline(sbase, color=PALETTE["green"], lw=1.4, ls="--", zorder=1,
                  label=f"waking-narrative baseline ({sbase:+.3f})")
    ax[4].barh(y, s.step_auth.to_numpy(),
               xerr=[np.abs(s.step_auth.to_numpy() - s.step_lo.to_numpy()),
                     np.abs(s.step_hi.to_numpy() - s.step_auth.to_numpy())],
               color=[KIND_COLOR.get(k, PALETTE["grey"]) for k in s.kind], zorder=2,
               error_kw={"lw": 1.1, "capsize": 2.5, "zorder": 3})
    ax[4].axvline(0, color=PALETTE["black"], lw=0.9, zorder=1)
    ax[4].set_yticks(y); ax[4].set_yticklabels(s.lbl.tolist(), fontsize=8.5)
    ax[4].set_xlabel("closing step: last sentence − penultimate sentence")
    ax[4].set_title("Does the narrative resolve? Only the machine's")
    ax[4].legend(fontsize=7.5, loc="lower right", framealpha=0.92)
    panel_label(ax[4], "e")

    # (f) where the corpora actually differ in shape: the opening, not the ending.
    show = {"DreamSeer EN": PALETTE["blue"], "Reddit r/Dreams": PALETTE["sky"],
            "r/confession (waking narrative)": PALETTE["green"],
            "Gutenberg fiction": PALETTE["orange"], "DreamSeer interp.": PALETTE["purple"]}
    for c, col in show.items():
        e = re_edge[re_edge.corpus == c]
        h = e[e.edge == "start"].sort_values("pos")
        t2 = e[e.edge == "end"].sort_values("pos", ascending=False)
        ls = "--" if "interp" in c else "-"
        ax[5].plot(np.arange(1, 6), h["mean"].to_numpy(), marker="o", ms=4, color=col, ls=ls,
                   label=short(c))
        ax[5].plot(np.arange(7, 12), t2["mean"].to_numpy(), marker="o", ms=4, color=col, ls=ls)
    ax[5].axvline(6, color=PALETTE["grey"], lw=6, alpha=0.25)
    ax[5].axhline(0, color=PALETTE["black"], lw=0.9)
    ax[5].set_xticks(list(range(1, 6)) + [6] + list(range(7, 12)))
    ax[5].set_xticklabels(["1", "2", "3", "4", "5", "⋯", "−5", "−4", "−3", "−2", "−1"],
                          fontsize=8)
    ax[5].set_xlabel("sentence position (from the start, then from the end)")
    ax[5].set_ylabel("author-weighted valence")
    ax[5].set_title("The corpora diverge at the opening, not the close")
    ax[5].legend(fontsize=7.5, loc="upper left", ncol=2, framealpha=0.92)
    panel_label(ax[5], "f")

    fig.suptitle("The arrow survives contributor clustering, deletion of the ending and a "
                 "scale-free re-expression;\nthe structural difference we conjectured — a "
                 "missing resolution — is not there",
                 fontsize=12.5, fontweight="bold", y=1.003)
    fig.tight_layout()
    save_fig(fig, PUB / "fig4_arrow_robustness")
    print(f"[fig4] {len(u)} clustered rows | {len(arms)} arms | {len(t)} true-XLM-R corpora "
          f"| {len(z)} standardised | {len(s)} closing-step corpora", flush=True)


def main():
    apply_style()
    for name, fn in [("fig4", fig4), ("fig2", fig2), ("fig1", fig1)]:
        try:
            fn()
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
    print("[revision-figures] done ->", PUB)


if __name__ == "__main__":
    main()
