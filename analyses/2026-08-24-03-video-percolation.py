"""Video 2 — the semantic continent dissolving, and the null that never had one.

Animates Section 3.2. Dream reports embedded in a 384-dimensional semantic space form a single
dense connected "continent": raise the cosine-similarity threshold and a giant component survives
far past the point where a variance-preserving null has shattered into dust. The manuscript's
reading of this is a *calibration* rather than a discovery — matched non-dream narrative prose
produces the same continent, so the structure belongs to narrative language rather than to dreaming
— and the video is built to show the comparison that establishes it is real structure at all.

Method mirrors `analyses/2026-07-19-19-collective-dynamics.py::percolation`: 5,000 English reports
per seed, unit-normalised embeddings, cosine threshold swept on the same grid, giant-component
fraction averaged over three seeds. The feature-shuffle null permutes each embedding coordinate
independently across reports and renormalises, which preserves every coordinate's marginal
distribution and destroys only the cross-coordinate structure that makes two reports similar.

The recomputed curve is checked against the published `collective_percolation.csv` and the script
fails if they disagree, so the video cannot silently drift from the figure it illustrates.

Privacy: nodes are unlabelled points standing for de-identified reports. No text, no identifier and
no per-report attribute is displayed, and the quantitative panel is a population curve.

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 USE_TF=0 python3 \
        analyses/2026-08-24-03-video-percolation.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from psychohistory import config as C
from psychohistory.dreams.embed_cache import load_or_build
from psychohistory.utils.anim import PALETTE, apply_video_style, render

OUT = C.RESULTS / "videos"
PUBLISHED = C.RESULTS / "showcase" / "collective_percolation.csv"

N_GRAPH = 5000          # reports per seed, as in the analysis
SEEDS = (0, 1, 2)
N_LAYOUT = 420          # nodes drawn in the node-link panels
MAX_EDGES = 2600        # edges drawn; the true count is reported in text
FPS = 24
HOLD = 40


def unit(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def feature_shuffle(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute each coordinate independently: keeps every marginal, destroys the joint."""
    return unit(np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])]))


def giant_curve(X: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    S = X @ X.T
    np.fill_diagonal(S, 0.0)
    out = np.empty(len(thetas))
    for i, th in enumerate(thetas):
        A = csr_matrix(S > th)
        n, lab = connected_components(A, directed=False)
        out[i] = np.bincount(lab).max() / len(X)
    return out


def curves(emb: np.ndarray, idx_all: np.ndarray, thetas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    real, shuf = [], []
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        idx = rng.choice(idx_all, min(N_GRAPH, len(idx_all)), replace=False)
        X = unit(emb[idx].astype(np.float32))
        real.append(giant_curve(X, thetas))
        shuf.append(giant_curve(feature_shuffle(X, rng), thetas))
        print(f"   seed {sd} done", flush=True)
    return np.mean(real, 0), np.mean(shuf, 0)


def pca2(X: np.ndarray) -> np.ndarray:
    Z = X - X.mean(0)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    return U[:, :2] * S[:2]


def edge_list(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All upper-triangle pairs with their cosine, in a fixed shuffled order.

    The order is fixed once so that the subset of edges *drawn* is stable from frame to frame:
    an edge that is present at two thresholds is drawn at both, rather than flickering because a
    fresh random subsample was taken.
    """
    S = X @ X.T
    iu = np.triu_indices(len(X), k=1)
    w = S[iu]
    order = np.random.default_rng(7).permutation(len(w))
    return np.stack([iu[0][order], iu[1][order]], axis=1), w[order]


def main() -> None:
    meta, emb = load_or_build()
    idx_all = np.where(meta.lang.values == "en")[0]
    print(f"[video2] {len(idx_all):,} English reports with embeddings", flush=True)

    thetas = np.round(np.arange(0.02, 0.90, 0.03), 2)
    print("[video2] recomputing percolation curves ...", flush=True)
    g_real, g_shuf = curves(emb, idx_all, thetas)

    # Fail rather than drift: the video must illustrate the published figure, not a variant of it.
    pub = pd.read_csv(PUBLISHED).set_index("theta").giant_frac.reindex(thetas).to_numpy()
    dev = np.nanmax(np.abs(pub - g_real))
    print(f"[video2] max deviation from published curve: {dev:.4f}")
    if dev > 0.05:
        raise SystemExit(f"recomputed curve departs from {PUBLISHED.name} by {dev:.3f} — "
                         "the method has drifted; fix before rendering.")

    rng = np.random.default_rng(0)
    sub = rng.choice(idx_all, N_LAYOUT, replace=False)
    Xr = unit(emb[sub].astype(np.float32))
    Xs = feature_shuffle(Xr, np.random.default_rng(0))
    Pr, Ps = pca2(Xr), pca2(Xs)
    Er, Wr = edge_list(Xr)
    Es, Ws = edge_list(Xs)

    sweep = thetas[(thetas >= 0.29) & (thetas <= 0.86)]
    frames = np.concatenate([sweep, np.repeat(sweep[-1], HOLD)])

    apply_video_style()
    fig = plt.figure(figsize=(15.36, 7.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], left=0.035, right=0.975,
                          bottom=0.13, top=0.80, wspace=0.20)
    ax_r, ax_s, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

    for ax, P, title, sub_t in (
        (ax_r, Pr, "Dream reports", f"{N_LAYOUT}-report sample, real embedding"),
        (ax_s, Ps, "Feature-shuffled null", f"{N_LAYOUT}-report sample, joint destroyed"),
    ):
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        pad = 0.08 * (P.max(0) - P.min(0))
        ax.set_xlim(P[:, 0].min() - pad[0], P[:, 0].max() + pad[0])
        ax.set_ylim(P[:, 1].min() - pad[1], P[:, 1].max() + pad[1])
        ax.set_title(f"{title}\n{sub_t}", fontsize=14)

    lc_r = LineCollection([], colors=PALETTE["blue"], linewidths=0.35, alpha=0.10, zorder=1)
    lc_s = LineCollection([], colors=PALETTE["orange"], linewidths=0.35, alpha=0.10, zorder=1)
    ax_r.add_collection(lc_r); ax_s.add_collection(lc_s)
    sc_r = ax_r.scatter(Pr[:, 0], Pr[:, 1], s=16, zorder=3, linewidths=0)
    sc_s = ax_s.scatter(Ps[:, 0], Ps[:, 1], s=16, zorder=3, linewidths=0)
    txt_r = ax_r.text(0.02, 0.02, "", transform=ax_r.transAxes, fontsize=11.5, color="#333333")
    txt_s = ax_s.text(0.02, 0.02, "", transform=ax_s.transAxes, fontsize=11.5, color="#333333")

    ax_c.set_xlim(sweep[0], sweep[-1])
    ax_c.set_ylim(-3, 104)
    ax_c.set_xlabel("cosine similarity threshold  θ")
    ax_c.set_ylabel("largest connected component (% of reports)")
    ax_c.set_title(f"The continent survives; the null does not\nfull measurement: "
                   f"{N_GRAPH:,} reports, {len(SEEDS)} seeds", fontsize=14)
    ax_c.plot(thetas, g_real * 100, color=PALETTE["blue"], lw=1.0, alpha=0.28)
    ax_c.plot(thetas, g_shuf * 100, color=PALETTE["orange"], lw=1.0, alpha=0.28)
    (cr,) = ax_c.plot([], [], color=PALETTE["blue"], lw=3.0, label="dream reports")
    (cs,) = ax_c.plot([], [], color=PALETTE["orange"], lw=3.0, label="feature-shuffled null")
    mk_r = ax_c.scatter([], [], s=70, color=PALETTE["blue"], zorder=5)
    mk_s = ax_c.scatter([], [], s=70, color=PALETTE["orange"], zorder=5)
    ax_c.legend(loc="lower left")
    vline = ax_c.axvline(sweep[0], color=PALETTE["grey"], lw=1.0, ls=":")

    sup = fig.suptitle("", fontsize=17, fontweight="bold", y=0.965)
    note = fig.text(0.5, 0.90, "", ha="center", fontsize=12.5, color="#333333")
    fig.text(0.5, 0.052, f"The two panels draw a {N_LAYOUT}-report sample for legibility; a sparser "
                         f"sample has fewer neighbours per report, so it fragments at a lower θ than "
                         f"the {N_GRAPH:,}-report measurement at right.",
             ha="center", fontsize=11, color="#555555")
    fig.text(0.5, 0.016, "Matched non-dream narrative prose produces the same continent (Section 3.2) — "
                         "the structure belongs to narrative language, not to dreaming.",
             ha="center", fontsize=11, color="#555555", style="italic")

    def paint(ax_txt, sc, lc, P, E, W, th, base):
        keep = W > th
        n_true = int(keep.sum())
        sel = E[keep][:MAX_EDGES]
        lc.set_segments(list(np.stack([P[sel[:, 0]], P[sel[:, 1]]], axis=1)) if len(sel) else [])
        A = csr_matrix((np.ones(n_true, bool), (E[keep][:, 0], E[keep][:, 1])),
                       shape=(len(P), len(P))) if n_true else csr_matrix((len(P), len(P)), dtype=bool)
        _, lab = connected_components(A, directed=False)
        big = np.bincount(lab).argmax()
        inbig = lab == big
        sc.set_color(np.where(inbig, base, PALETTE["grey"]))
        sc.set_alpha(None)
        ax_txt.set_text(f"largest component: {100 * inbig.mean():.0f}% of the {len(P)} shown\n"
                        f"{n_true:,} edges present"
                        + (f" ({len(sel):,} drawn)" if n_true > MAX_EDGES else ""))
        return inbig.mean()

    def update(i):
        th = float(frames[min(i, len(frames) - 1)])
        paint(txt_r, sc_r, lc_r, Pr, Er, Wr, th, PALETTE["blue"])
        paint(txt_s, sc_s, lc_s, Ps, Es, Ws, th, PALETTE["orange"])

        m = thetas <= th + 1e-9
        cr.set_data(thetas[m], g_real[m] * 100)
        cs.set_data(thetas[m], g_shuf[m] * 100)
        j = int(np.argmin(np.abs(thetas - th)))
        mk_r.set_offsets([[thetas[j], g_real[j] * 100]])
        mk_s.set_offsets([[thetas[j], g_shuf[j] * 100]])
        vline.set_xdata([th, th])

        sup.set_text(f"A single dense semantic continent — at θ = {th:.2f}")
        note.set_text(f"giant component:   dream reports {g_real[j] * 100:.0f}%    "
                      f"|    feature-shuffled null {g_shuf[j] * 100:.0f}%    "
                      f"(DreamSeer English, 5,000 reports)")
        return ()

    render(fig, update, n_frames=len(frames), out_stem=OUT / "semantic-continent", fps=FPS)

    at = {t: (g_real[np.argmin(np.abs(thetas - t))], g_shuf[np.argmin(np.abs(thetas - t))])
          for t in (0.50, 0.59)}
    (OUT / "semantic-continent.md").write_text(
        "# Video 2 — the semantic continent\n\n"
        f"DreamSeer English, {N_GRAPH:,} reports per seed, {len(SEEDS)} seeds, 384-dimensional "
        "unit-normalised embeddings, cosine-similarity graph.\n\n"
        + "".join(f"- At θ = {t:.2f}: giant component **{r * 100:.0f}%** of reports, against "
                  f"**{s * 100:.0f}%** for the feature-shuffle null.\n" for t, (r, s) in at.items())
        + f"\nMaximum deviation from the published curve "
          f"(`60-results/showcase/collective_percolation.csv`): {dev:.4f}.\n\n"
        "The null permutes each embedding coordinate independently across reports and renormalises, "
        "preserving every coordinate's marginal distribution and destroying only the "
        "cross-coordinate structure that makes two reports similar. Matched non-dream narrative "
        "corpora reach 89-96% at these thresholds, which is why the manuscript reads the continent "
        "as a property of narrative language rather than of dreaming.\n")
    print("[video2] wrote", OUT / "semantic-continent.md")


if __name__ == "__main__":
    main()
