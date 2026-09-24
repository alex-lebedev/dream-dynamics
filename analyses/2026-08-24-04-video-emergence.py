"""Video 3 — a population regularity condensing out of individual noise.

Video 1 shows *that* the mean arc descends and that the descent is order-sensitive. This one shows
*how the shape arrives*: a single dream report's emotional trajectory is jagged and idiosyncratic,
and the smooth descending arc is not visible in any one of them. It appears only as reports are
averaged, at close to the rate averaging predicts.

Three panels. (a) and (b) redraw the running mean arc as reports accumulate, in real sentence order
and with order shuffled within each report, leaving a fading trail of earlier estimates so the funnel
narrowing is visible. (c) measures the convergence *out of sample*: contributors are split into two
disjoint groups, the running mean is computed on group A, and its distance is measured against the
final mean of group B. Comparing a running mean to its own endpoint would force the curve to zero by
construction and prove nothing; comparing it to a held-out group cannot, so what panel (c) shows is
genuine convergence onto a stable population shape rather than an arithmetic identity.

The split is on *contributors*, not reports. A report-level split would let one person's reports fall
on both sides, so the running mean could close on the reference partly by recognising the same people
rather than the same population regularity — and the contributor is the unit the manuscript estimates
the arrow on (Section 3.3), for the reason that one SDDb participant supplies 31% of that corpus.

The 1/sqrt(n) reference line is what *independent* averaging predicts, and the fitted decay is
shallower than that (see the generated note). This is expected and is not a failure of the claim:
reports are clustered within contributors, so successive reports carry less new information than
independent draws would, which slows convergence. The claim the panel supports is that the arc is a
population-level regularity converging on a stable shape, not that reports are independent.

Privacy: aggregate and ensemble only. The single highlighted trajectory in the opening frames is a
de-identified 20-point vector drawn from the same anonymous subsample used elsewhere, never labelled
or linked to a contributor, and no report text is used.

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 USE_TF=0 python3 \
        analyses/2026-08-24-04-video-emergence.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from psychohistory import config as C
from psychohistory.dreams.sentence_cache import load_or_build_sentences, offsets
from psychohistory.dreams.sentence_sentiment import load_or_build_sentence_sentiment
from psychohistory.utils.anim import PALETTE, apply_video_style, render
from psychohistory.utils.disclosure import MIN_REPORTS, MIN_USERS, min_prefix, n_users_in

OUT = C.RESULTS / "videos"
L_ARC = 20
MIN_SENT = 5
N_FRAMES = 250
FPS = 24
N_TRAIL = 16           # fading earlier estimates retained per panel
SEED = 0


def resample(v, L=L_ARC):
    return np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(v)), v)


def build_pair(sent, counts, mask, rng, min_sent=MIN_SENT):
    """Real and within-report order-shuffled arcs, plus the row index of each kept report.

    The index is returned so contributor identifiers can be attached downstream: the animation
    accumulates reports, and the disclosure floor has to be checked on contributors, not just on
    report counts.
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


def main() -> None:
    rng = np.random.default_rng(SEED)
    print("[video3] loading sentence cache + per-sentence sentiment ...", flush=True)
    _dt, ct, _et, _u, mt = load_or_build_sentences("text")
    _d, _c, sent = load_or_build_sentence_sentiment("text", min_sent=MIN_SENT)

    real, shuf, idx = build_pair(sent, ct, mt.lang.values == "en", rng)
    n = len(real)
    order = rng.permutation(n)
    real, shuf, idx = real[order], shuf[order], idx[order]
    users = mt.userID.values[idx]
    print(f"[video3] {n:,} English reports with >={MIN_SENT} sentences "
          f"from {n_users_in(users):,} contributors", flush=True)

    # Author-disjoint halves. Splitting on reports would let one contributor's reports sit on both
    # sides, so a running mean could converge on the held-out arc partly by recognising the same
    # people rather than the same population regularity — and the contributor is the unit the
    # manuscript's arrow analysis is estimated on (Section 3.3). No contributor appears in both.
    uniq = np.unique(users)
    rng.shuffle(uniq)
    in_a = np.isin(users, uniq[:len(uniq) // 2])
    A_r, B_r = real[in_a], real[~in_a]
    A_s, B_s = shuf[in_a], shuf[~in_a]
    users_a = users[in_a]
    h = len(A_r)
    ref_r, ref_s = B_r.mean(0), B_s.mean(0)
    print(f"[video3] author-disjoint halves: {h:,} reports from {n_users_in(users_a):,} contributors "
          f"vs {len(B_r):,} from {n_users_in(users[~in_a]):,}", flush=True)

    # ETHICS §4: a running mean may not be displayed until the prefix behind it clears both floors.
    # A mean arc over two reports is close to publishing the two trajectories themselves.
    k0 = min_prefix(users_a)
    print(f"[video3] sweep starts at n={k0} ({MIN_USERS}+ contributors, {MIN_REPORTS}+ reports)")

    ks = np.unique(np.round(np.geomspace(k0, h, N_FRAMES)).astype(int))
    ks = np.concatenate([np.repeat(ks[0], 18), ks, np.repeat(ks[-1], 46)])
    x = np.linspace(0, 1, L_ARC)

    rms_r = np.array([np.sqrt(np.mean((A_r[:k].mean(0) - ref_r) ** 2)) for k in ks])
    rms_s = np.array([np.sqrt(np.mean((A_s[:k].mean(0) - ref_s) ** 2)) for k in ks])

    apply_video_style()
    fig = plt.figure(figsize=(15.36, 7.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95], left=0.055, right=0.985,
                          bottom=0.205, top=0.79, wspace=0.30)
    ax_r, ax_s, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

    ylo, yhi = float(min(real.min(), shuf.min())), float(max(real.max(), shuf.max()))
    for ax, title, sub in (
        (ax_r, "Real sentence order", "the reports as written"),
        (ax_s, "Sentence order shuffled", "same scores, sequence destroyed"),
    ):
        ax.set_xlim(0, 1)
        ax.set_ylim(ylo * 1.02, yhi * 1.02)
        ax.set_xlabel("normalized dream time")
        ax.set_title(f"{title}\n{sub}", fontsize=14)
        ax.axhline(0, color=PALETTE["grey"], lw=0.9, ls=":", zorder=1)
    ax_r.set_ylabel("sentence valence")

    ax_r.plot(x, ref_r, color=PALETTE["blue"], lw=1.6, ls="--", alpha=0.55,
              zorder=3, label="held-out contributors")
    ax_s.plot(x, ref_s, color=PALETTE["orange"], lw=1.6, ls="--", alpha=0.55,
              zorder=3, label="held-out contributors")
    # Fixed-alpha lines rather than a per-segment-alpha LineCollection: the trail is short, and
    # assigning an alpha array to a collection is not portable across matplotlib versions.
    alphas = np.linspace(0.05, 0.32, N_TRAIL)
    trail_r = [ax_r.plot([], [], color=PALETTE["blue"], lw=1.2, alpha=a, zorder=2)[0]
               for a in alphas]
    trail_s = [ax_s.plot([], [], color=PALETTE["orange"], lw=1.2, alpha=a, zorder=2)[0]
               for a in alphas]
    (cur_r,) = ax_r.plot([], [], color=PALETTE["blue"], lw=3.6, zorder=6, label="running mean")
    (cur_s,) = ax_s.plot([], [], color=PALETTE["orange"], lw=3.6, zorder=6, label="running mean")
    ax_r.legend(loc="lower left", fontsize=11)
    ax_s.legend(loc="lower left", fontsize=11)

    ax_c.set_xscale("log"); ax_c.set_yscale("log")
    ax_c.set_xlim(2, h * 1.05)
    ax_c.set_xlabel("reports averaged (log scale)")
    ax_c.set_ylabel("distance to the held-out contributors' arc (RMS)")
    ax_c.set_title("Sharpening, near the 1/√n rate\nout-of-sample convergence", fontsize=14)
    guide = rms_r[0] * np.sqrt(ks[0]) / np.sqrt(ks)
    ax_c.plot(ks, guide, color=PALETTE["grey"], lw=1.4, ls="--", label="1/√n reference")
    (cv_r,) = ax_c.plot([], [], color=PALETTE["blue"], lw=2.8, label="real order")
    (cv_s,) = ax_c.plot([], [], color=PALETTE["orange"], lw=2.8, label="shuffled")
    ax_c.legend(loc="lower left", fontsize=11)

    sup = fig.suptitle("", fontsize=17, fontweight="bold", y=0.965)
    note = fig.text(0.5, 0.895, "", ha="center", fontsize=12.5, color="#333333")
    fig.text(0.5, 0.076, "The right panel compares the running mean against the final arc of a "
                         "disjoint set of contributors, so the convergence is out-of-sample and "
                         "cannot be driven to zero by construction.",
             ha="center", fontsize=11, color="#555555")
    fig.text(0.5, 0.042, "The shape is aggregate — no single report carries it. The direction is not: "
                         "56–88% of individual contributors darken on their own averages "
                         "(Section 3.4).",
             ha="center", fontsize=11, color="#555555", style="italic")
    fig.text(0.5, 0.011, "True per-sentence XLM-R on the language-stratified arc sample; the "
                         "manuscript's headline arrow estimate uses a larger per-corpus prefix "
                         "(Section 3.3) and is contributor-clustered.",
             ha="center", fontsize=10.5, color="#777777")

    hist_r: list[np.ndarray] = []
    hist_s: list[np.ndarray] = []
    last_k = [-1]

    def update(i):
        j = min(i, len(ks) - 1)
        k = int(ks[j])
        mr, ms = A_r[:k].mean(0), A_s[:k].mean(0)
        if k != last_k[0]:                      # hold frames must not flood the trail
            hist_r.append(mr); hist_s.append(ms)
            last_k[0] = k
        cur_r.set_data(x, mr); cur_s.set_data(x, ms)
        for lines, hist in ((trail_r, hist_r), (trail_s, hist_s)):
            keep = hist[-N_TRAIL:]
            pad = N_TRAIL - len(keep)           # oldest lines stay empty until the trail fills
            for ln in lines[:pad]:
                ln.set_data([], [])
            for ln, m in zip(lines[pad:], keep):
                ln.set_data(x, m)

        cv_r.set_data(ks[:j + 1], rms_r[:j + 1])
        cv_s.set_data(ks[:j + 1], rms_s[:j + 1])

        sup.set_text(f"A regularity no single dream contains — averaging {k:,} "
                     f"report{'s' if k != 1 else ''}")
        note.set_text(f"running mean, start → end:    real {mr[:3].mean():+.3f} → "
                      f"{mr[-3:].mean():+.3f}    |    shuffled {ms[:3].mean():+.3f} → "
                      f"{ms[-3:].mean():+.3f}    (DreamSeer English, {MIN_SENT}–25 sentences)")
        return ()

    render(fig, update, n_frames=len(ks), out_stem=OUT / "regularity-emerging", fps=FPS)

    slope = np.polyfit(np.log(ks[ks <= h // 2]), np.log(rms_r[ks <= h // 2]), 1)[0]
    (OUT / "regularity-emerging.md").write_text(
        "# Video 3 — a population regularity condensing out of individual noise\n\n"
        f"DreamSeer English, {n:,} reports of {MIN_SENT}–25 sentences from the language-stratified "
        f"arc cache, split by **contributor** into disjoint groups of {h:,} reports "
        f"({n_users_in(users_a):,} contributors) and {n - h:,} reports "
        f"({n_users_in(users[~in_a]):,} contributors). The running mean of group A is compared "
        "against the final mean arc of group B, so the convergence measured is out-of-sample and no "
        "contributor is on both sides.\n\n"
        "This is an illustration of a point the manuscript already makes in Section 3.4 — that the "
        "arc's *shape* is aggregate while its *direction* is not — and not a separate result. It is "
        "not in the manuscript, the findings ledger or the reproducibility table.\n\n"
        f"The sweep starts at n = {k0} rather than at a single report: under `docs/ETHICS.md` §4 "
        "no displayed "
        f"aggregate may rest on fewer than {MIN_REPORTS} reports or {MIN_USERS} distinct "
        "contributors, and a mean arc over a handful of reports is close to publishing those "
        f"trajectories individually. The full sample spans {n_users_in(users):,} contributors.\n\n"
        f"- Distance to the held-out contributors' arc falls from **{rms_r[0]:.3f}** at "
        f"n = {int(ks[0])} to **{rms_r[-1]:.4f}** at n = {h:,} (real sentence order).\n"
        f"- Fitted log-log slope over the first half of the range: **{slope:+.2f}**, against the "
        f"−0.50 that *independent* averaging predicts. The shortfall is expected: these "
        f"{n:,} reports come from {n_users_in(users):,} contributors, so successive reports are not "
        "independent draws and within-contributor similarity slows convergence.\n"
        f"- Shuffled order converges too, onto a flat arc: **{rms_s[-1]:.4f}**. Convergence is not "
        "the claim; the *shape* converged onto is.\n\n"
        "Comparing a running mean to its own endpoint would force the curve to zero by construction. "
        "A held-out set of *different contributors* cannot be driven to zero this way, so the decay "
        "is evidence that the mean arc is a stable population quantity rather than an artifact of "
        "accumulating the same reports or of recognising the same people.\n\n"
        "Instrument and sample: true per-sentence XLM-R "
        "(`cardiffnlp/twitter-xlm-roberta-base-sentiment`), the same scorer as the manuscript's "
        "primary arrow analysis, applied to the language-stratified arc cache rather than to the "
        "6,000-document per-corpus prefix behind the Section 3.3 estimates. The video animates the "
        "regularity; the reported effect sizes belong to the manuscript.\n\n"
        "Report-level inference in the manuscript is author-clustered (Section 3.3). This video "
        "treats reports as exchangeable, which is appropriate for illustrating convergence of the "
        "population mean but is not the manuscript's inferential model.\n")
    print("[video3] wrote", OUT / "regularity-emerging.md")


if __name__ == "__main__":
    main()
