"""Video 1 — the emotional arrow of time, and the order control that licenses it.

Animates the manuscript's central population claim (Sections 3.3-3.4): individual dream reports
follow heterogeneous emotional trajectories, yet averaging over reports reveals a stable descending
mean arc — and that descent is *order-sensitive*, so it is a property of narrative sequence rather
than of the sentiment instrument or the marginal distribution of sentence scores.

The control is the point of the video, not a footnote. Left panel: reports in their real sentence
order. Right panel: the same reports with sentence order permuted within each report before
resampling, which preserves every sentence score and destroys only the sequence. If the descent were
an artifact of the scorer or of how sentences are distributed, both means would descend. Only the
left one does.

Panel (c) tracks the end-minus-start drift of both running means as reports accumulate, which is
what "the macroscopic regularity stabilises" looks like: the real estimate converges away from zero
and the shuffled one converges onto it.

Privacy: every frame is aggregate or ensemble. The faint lines are a fixed anonymous subsample of
de-identified 20-point trajectories, never labelled, isolated or linked to a contributor, and no
report text is used. The running mean is over all accumulated reports.

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 USE_TF=0 python3 \
        analyses/2026-08-24-02-video-arrow.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from psychohistory import config as C
from psychohistory.dreams.sentence_cache import load_or_build_sentences, offsets
from psychohistory.dreams.sentence_sentiment import load_or_build_sentence_sentiment
from psychohistory.utils.anim import PALETTE, apply_video_style, render
from psychohistory.utils.disclosure import MIN_REPORTS, MIN_USERS, min_prefix, n_users_in

OUT = C.RESULTS / "videos"
L_ARC = 20
MIN_SENT = 5
N_FRAMES = 260
FPS = 24
N_FAINT = 260          # anonymous trajectories drawn as an ensemble
SEED = 0


def resample(v, L=L_ARC):
    return np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(v)), v)


def build_pair(sent, counts, mask, rng, min_sent=MIN_SENT):
    """Real and within-report order-shuffled arcs from the same per-sentence scores.

    The permutation is applied to the per-sentence values *before* resampling, so the shuffled arm
    keeps each report's exact multiset of sentence scores and its length, and differs only in order.
    """
    off = offsets(counts)
    real, shuf, keep = [], [], []
    for k in np.where(mask & (counts >= min_sent))[0]:
        v = sent[off[k]:off[k] + counts[k]]
        if not np.all(np.isfinite(v)):
            continue
        real.append(resample(v))
        shuf.append(resample(rng.permutation(v)))
        keep.append(k)
    return np.asarray(real), np.asarray(shuf), np.asarray(keep)


def drift(A):
    """End-minus-start valence, the manuscript's scalar summary of the arrow."""
    return A[:, -3:].mean(1) - A[:, :3].mean(1)


def main() -> None:
    rng = np.random.default_rng(SEED)
    print("[video1] loading sentence cache + per-sentence sentiment ...", flush=True)
    _dt, ct, _et, _u, mt = load_or_build_sentences("text")
    _d, _c, sent = load_or_build_sentence_sentiment("text", min_sent=MIN_SENT)

    real, shuf, idx = build_pair(sent, ct, mt.lang.values == "en", rng)
    n = len(real)

    order = rng.permutation(n)
    real, shuf, idx = real[order], shuf[order], idx[order]
    users = mt.userID.values[idx]
    d_real, d_shuf = drift(real), drift(shuf)
    print(f"[video1] {n:,} English reports with >={MIN_SENT} sentences "
          f"from {n_users_in(users):,} contributors", flush=True)

    # ETHICS §4: the running mean is displayed on screen, so the sweep may not begin below the
    # disclosure floor — 20 reports drawn from at least 5 distinct contributors.
    k0 = min_prefix(users)
    print(f"[video1] sweep starts at n={k0} ({MIN_USERS}+ contributors, {MIN_REPORTS}+ reports)")

    # Geometric frame schedule: the early frames are where the mean is visibly unstable, so spend
    # frames there rather than on the last few thousand reports that move nothing.
    ks = np.unique(np.round(np.geomspace(k0, n, N_FRAMES)).astype(int))
    ks = np.concatenate([ks, np.repeat(ks[-1], 60)])          # hold on the final frame
    x = np.linspace(0, 1, L_ARC)

    apply_video_style()
    fig = plt.figure(figsize=(15.36, 7.2))                     # 1536x720 at dpi=100
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95], left=0.055, right=0.985,
                          bottom=0.115, top=0.80, wspace=0.30)
    ax_r, ax_s, ax_d = (fig.add_subplot(gs[0, i]) for i in range(3))

    lo = float(min(real.min(), shuf.min())), float(max(real.max(), shuf.max()))
    for ax, title, sub in (
        (ax_r, "Real sentence order", "the reports as written"),
        (ax_s, "Sentence order shuffled", "same scores, sequence destroyed"),
    ):
        ax.set_xlim(0, 1)
        ax.set_ylim(lo[0] * 1.02, lo[1] * 1.02)
        ax.set_xlabel("normalized dream time")
        ax.set_title(f"{title}\n{sub}", fontsize=14)
        ax.axhline(0, color=PALETTE["grey"], lw=0.9, ls=":", zorder=1)
    ax_r.set_ylabel("sentence valence")

    faint_r = LineCollection([], colors=PALETTE["blue"], linewidths=0.5, alpha=0.09, zorder=2)
    faint_s = LineCollection([], colors=PALETTE["orange"], linewidths=0.5, alpha=0.09, zorder=2)
    ax_r.add_collection(faint_r)
    ax_s.add_collection(faint_s)
    (mean_r,) = ax_r.plot([], [], color=PALETTE["blue"], lw=3.4, zorder=5, label="mean arc")
    (mean_s,) = ax_s.plot([], [], color=PALETTE["orange"], lw=3.4, zorder=5, label="mean arc")
    band_r = band_s = None
    ax_r.legend(loc="lower left")
    ax_s.legend(loc="lower left")

    ax_d.set_xscale("log")
    ax_d.set_xlim(k0, n * 1.05)
    ax_d.set_xlabel("reports averaged (log scale)")
    ax_d.set_ylabel("end − start valence")
    ax_d.set_title("The regularity stabilises\nrunning drift of each mean", fontsize=14)
    ax_d.axhline(0, color=PALETTE["grey"], lw=1.1, ls=":")
    (tr_r,) = ax_d.plot([], [], color=PALETTE["blue"], lw=2.6, label="real order")
    (tr_s,) = ax_d.plot([], [], color=PALETTE["orange"], lw=2.6, label="shuffled")
    ax_d.legend(loc="upper right")
    span = max(abs(d_real[:200].mean()), 0.05) * 2.2
    ax_d.set_ylim(-span, span * 0.6)

    sup = fig.suptitle("", fontsize=17, fontweight="bold", y=0.965)
    note = fig.text(0.5, 0.905, "", ha="center", fontsize=12.5, color="#333333")
    fig.text(0.5, 0.052, "Shuffling preserves each report's sentence scores and its overall level; "
                         "it removes only their order. The comparison is the slope, not the height.",
             ha="center", fontsize=11.5, color="#555555", style="italic")
    fig.text(0.5, 0.014, "True per-sentence XLM-R on the language-stratified arc sample; the "
                         "manuscript's headline arrow estimate uses a larger per-corpus prefix "
                         "(Section 3.3) and is contributor-clustered.",
             ha="center", fontsize=10.5, color="#777777")
    seg_x = np.tile(x, (N_FAINT, 1))
    tr_ks, tr_dr, tr_ds = [], [], []

    def update(i):
        nonlocal band_r, band_s
        k = int(ks[min(i, len(ks) - 1)])
        nf = min(k, N_FAINT)
        faint_r.set_segments(list(np.stack([seg_x[:nf], real[:nf]], axis=-1)))
        faint_s.set_segments(list(np.stack([seg_x[:nf], shuf[:nf]], axis=-1)))

        mr, ms = real[:k].mean(0), shuf[:k].mean(0)
        mean_r.set_data(x, mr)
        mean_s.set_data(x, ms)
        # 95% band on the mean arc; it is wide at n=12 and invisible by n=6000, which is the point.
        se_r = real[:k].std(0, ddof=1) / np.sqrt(k) if k > 1 else np.zeros(L_ARC)
        se_s = shuf[:k].std(0, ddof=1) / np.sqrt(k) if k > 1 else np.zeros(L_ARC)
        for b in (band_r, band_s):
            if b is not None:
                b.remove()
        band_r = ax_r.fill_between(x, mr - 1.96 * se_r, mr + 1.96 * se_r,
                                   color=PALETTE["blue"], alpha=0.20, lw=0, zorder=4)
        band_s = ax_s.fill_between(x, ms - 1.96 * se_s, ms + 1.96 * se_s,
                                   color=PALETTE["orange"], alpha=0.20, lw=0, zorder=4)

        if not tr_ks or k != tr_ks[-1]:
            tr_ks.append(k)
            tr_dr.append(float(d_real[:k].mean()))
            tr_ds.append(float(d_shuf[:k].mean()))
        tr_r.set_data(tr_ks, tr_dr)
        tr_s.set_data(tr_ks, tr_ds)

        sup.set_text(f"Dream reports end darker than they begin — averaged over {k:,} reports")
        note.set_text(
            f"mean arc, start → end:    real order {mr[:3].mean():+.3f} → {mr[-3:].mean():+.3f}    "
            f"|    shuffled {ms[:3].mean():+.3f} → {ms[-3:].mean():+.3f}    "
            f"(DreamSeer English, {MIN_SENT}–25 sentences; {N_FAINT} anonymous trajectories shown)"
        )
        return ()

    render(fig, update, n_frames=len(ks), out_stem=OUT / "arrow-of-time", fps=FPS)

    # Final numbers, so the video's endpoint is checkable against the manuscript.
    md = OUT / "arrow-of-time.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(
        "# Video 1 — the emotional arrow of time\n\n"
        f"DreamSeer English, {n:,} reports, true per-sentence XLM-R "
        "(`cardiffnlp/twitter-xlm-roberta-base-sentiment`) resampled to 20 points of normalized "
        f"dream time.\n\nSample: the language-stratified arc cache "
        f"(`sentence_sentiment.load_or_build_sentence_sentiment`), which draws up to 3,000 documents "
        f"per language and admits reports of {MIN_SENT}–25 sentences. This is the same instrument as "
        "the manuscript's primary arrow analysis but a different sample: the headline estimates in "
        "Section 3.3 use the first 6,000 qualifying documents per corpus in file order with a "
        "four-sentence minimum. The video therefore animates the regularity, not the reported "
        "effect size — for that, cite the manuscript.\n\n"
        f"- Mean end-minus-start drift, real sentence order: **{d_real.mean():+.4f}** "
        f"(SE {d_real.std(ddof=1)/np.sqrt(n):.4f})\n"
        f"- Mean end-minus-start drift, order shuffled within report: **{d_shuf.mean():+.4f}** "
        f"(SE {d_shuf.std(ddof=1)/np.sqrt(n):.4f})\n\n"
        "The shuffled arm preserves each report's sentence scores and length and permutes only "
        "their order, so the gap between these two numbers is attributable to narrative sequence "
        "rather than to the sentiment instrument or the marginal distribution of scores.\n\n"
        f"The running mean shown on screen starts at n = {k0}, not at a single report: under "
        f"`docs/ETHICS.md` §4 no displayed aggregate may rest on fewer than {MIN_REPORTS} reports "
        f"or {MIN_USERS} "
        "distinct contributors, and a mean over a handful of reports approaches publishing those "
        f"trajectories individually. The sample spans {n_users_in(users):,} contributors.\n\n"
        "Report-level inference in the manuscript is author-clustered (Section 3.3); the standard "
        "errors above treat reports as independent and are shown only to indicate the precision of "
        "the animated running mean.\n"
    )
    print("[video1] wrote", md)


if __name__ == "__main__":
    main()
