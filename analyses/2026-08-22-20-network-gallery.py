"""Descriptive network gallery: how the dream feature network looks, breathes, and flows.

This is an EXPLORATORY, DESCRIPTIVE deliverable, not a confirmatory family. Nothing here is
pre-registered and nothing here is a finding. Each panel carries the comparison that keeps it
honest (label permutation, placebo anchors, or a chronology rotation), so the pictures can be
read without being believed too hard.

Five figures:
  1. nightmare vs calm affect/content constellation on a shared layout, plus the max-statistic
     permutation threshold for edge changes;
  2. the arrow of time drawn as a flow network of narrative mood, dreams vs matched prose;
  3. event-locked network breathing around ten curated global shocks, against placebo anchors;
  4. the same integration statistics rolled through calendar time against a chronology rotation;
  5. a raw-embedding orientation map with k-nearest-neighbour threads and a smoothed native
     nightmare-score overlay; explicitly not a connectivity or regionalization test.

Outputs -> 60-results/network-gallery/{figures,tables}. Aggregates only: no text, no user IDs,
no cell below the disclosure floor.
"""
from __future__ import annotations

import gc
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

from psychohistory import config as C
from psychohistory.dreams.arrow import MIN_SENT_ARROW, TEXT_CORPORA
from psychohistory.dreams.sentence_cache import split_sentences
from psychohistory.maps.state_networks import (
    load_state_panel,
    network_stats,
    placebo_anchors,
    spearman_matrix,
    top_edges,
    window_masks,
)

OUT = C.RESULTS / "network-gallery"
FIGS = OUT / "figures"
TABLES = OUT / "tables"
SEED = 20260822
MIN_CELL = 20  # ETHICS.md S4 disclosure floor: >= 20 reports ...
MIN_USERS = 5  # ... and >= 5 distinct contributors, in every drawn or written cell

BG = "#0B0E14"
PANEL = "#11151F"
FG = "#ECE8E1"
MUTED = "#7C879B"
WARM = "#FF7B54"
COOL = "#4FA8F7"
GOLD = "#F2C14E"
MINT = "#6FD9A6"
VIOLET = "#B48CF2"


def styled(figsize, nrows=1, ncols=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=BG, **kw)
    for ax in np.atleast_1d(np.asarray(axes)).ravel():
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color("#232A38")
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(FG)
    return fig, axes


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    # Matplotlib's `wrap=True` is backend/version dependent and can make `bbox_inches="tight"`
    # expand a plate to several times its intended width. Wrap long footer text deterministically.
    for text in fig.texts:
        value = text.get_text()
        if len(value) > 180:
            text.set_text(textwrap.fill(value, width=150, break_long_words=False))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=220, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"[gallery] wrote {name}", flush=True)


# --------------------------------------------------------------------------------------
# shared layout


def layout_and_communities(corr, names, seed=SEED):
    """Spring layout and modularity communities from the pooled network."""
    g = nx.Graph()
    g.add_nodes_from(range(len(names)))
    iu = np.triu_indices_from(corr, k=1)
    for i, j in zip(*iu):
        w = abs(corr[i, j])
        if w > 0.02:
            g.add_edge(int(i), int(j), weight=float(w))
    pos = nx.spring_layout(g, weight="weight", seed=seed, iterations=600, k=0.55)
    communities = nx.community.louvain_communities(g, weight="weight", seed=seed)
    membership = np.zeros(len(names), int)
    for c, nodes in enumerate(communities):
        for n in nodes:
            membership[n] = c
    return pos, membership


def draw_network(ax, corr, pos, edges, names, membership, node_size, *, scale=1.0, title=""):
    palette = [GOLD, COOL, MINT, VIOLET, WARM, "#E86A92"]
    segs, colors, widths = [], [], []
    for i, j in edges:
        r = corr[i, j]
        segs.append([pos[i], pos[j]])
        colors.append(WARM if r > 0 else COOL)
        widths.append(min(6.0, 9.0 * abs(r) * scale))
    alphas = np.clip(np.array(widths) / 6.0, 0.12, 0.85)
    ax.add_collection(
        LineCollection(
            segs,
            colors=[matplotlib.colors.to_rgba(c, a) for c, a in zip(colors, alphas)],
            linewidths=widths,
            zorder=1,
        )
    )
    xy = np.array([pos[i] for i in range(len(names))])
    ax.scatter(
        xy[:, 0], xy[:, 1], s=node_size * 2.6, c=[palette[m % len(palette)] for m in membership],
        alpha=0.18, zorder=2, linewidths=0,
    )
    ax.scatter(
        xy[:, 0], xy[:, 1], s=node_size, c=[palette[m % len(palette)] for m in membership],
        zorder=3, linewidths=0.6, edgecolors=BG,
    )
    for i, name in enumerate(names):
        ax.annotate(
            name.replace("place_", "").replace("situation_", "").replace("object_", ""),
            pos[i], fontsize=6.4, color=FG, ha="center", va="center",
            xytext=(0, -9), textcoords="offset points", alpha=0.75,
        )
    bare(ax)
    ax.set_title(title, fontsize=10, pad=10, color=FG)
    ax.autoscale_view()
    ax.margins(0.12)


# --------------------------------------------------------------------------------------
# figure 1: nightmare vs calm constellation


def circular_positions(membership, gap=0.22):
    """Nodes on a circle, grouped by community, with a gap between groups."""
    order = np.argsort(membership, kind="stable")
    groups = np.unique(membership)
    total = len(membership) + gap * len(groups) * len(membership) / len(membership)
    angles = np.zeros(len(membership))
    step = 2 * np.pi / (len(membership) + gap * len(groups))
    a = np.pi / 2
    prev = membership[order[0]]
    for node in order:
        if membership[node] != prev:
            a -= gap * step
            prev = membership[node]
        angles[node] = a
        a -= step
    pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(len(membership))}
    return pos, angles


def draw_chords(ax, pairs, values, pos, angles, membership, names, *, vmax=None):
    palette = [GOLD, COOL, MINT, VIOLET, WARM, "#E86A92"]
    vmax = vmax or (np.abs(values).max() + 1e-9)
    for (i, j), v in sorted(zip(pairs, values), key=lambda kv: abs(kv[1])):
        p0, p1 = np.array(pos[i]), np.array(pos[j])
        pull = 0.28 + 0.55 * (np.linalg.norm(p0 - p1) / 2.0)
        c0, c1 = p0 * (1 - pull), p1 * (1 - pull)
        frac = abs(v) / vmax
        ax.add_patch(PathPatch(
            MplPath([p0, c0, c1, p1],
                    [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]),
            facecolor="none", lw=0.5 + 4.5 * frac,
            edgecolor=matplotlib.colors.to_rgba(WARM if v > 0 else COOL, 0.25 + 0.6 * frac),
            zorder=1, capstyle="round"))
    for i, name in enumerate(names):
        colour = palette[membership[i] % len(palette)]
        ax.scatter(*pos[i], s=46, color=colour, zorder=3, linewidths=0.7, edgecolors=BG)
        deg = np.degrees(angles[i])
        rot = deg if -90 <= deg <= 90 else deg + 180
        ax.annotate(name.replace("place_", "").replace("situation_", "").replace("object_", ""),
                    np.array(pos[i]) * 1.09, fontsize=7.2, color=FG,
                    ha="left" if -90 <= deg <= 90 else "right", va="center",
                    rotation=rot, rotation_mode="anchor", alpha=0.9)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    bare(ax)


def threat_marked(document_ids):
    """Text-derived threat marker: the frozen anxiety or death lexicon fires in the report.

    The app's nightmare index is the mean of fear, danger and sadness minus joy — a
    deterministic function of four axes plotted in this figure — so splitting on it would
    condition the network on its own nodes. This splitter is an independent instrument.
    """
    from psychohistory.dreams.lexicons import COMPARATOR, LATENT, score

    raw = pd.read_csv(C.DREAMSEER_RAW, dtype=str, keep_default_na=False,
                      usecols=["documentID", "text"])
    text = pd.Series(dict(zip(raw.documentID, raw.text))).reindex(document_ids).fillna("")
    hits = score(text, {**COMPARATOR, "death": LATENT["death"]})
    return hits.to_numpy().max(axis=1).astype(bool)


def figure_nightmare_constellation(panel, rng):
    names = list(panel.feature_names)
    x = panel.state
    marked = threat_marked(panel.frame["documentID"].to_numpy())
    finite = np.isfinite(x).all(axis=1)
    x, marked = x[finite], marked[finite]
    dates = panel.frame.loc[finite, "date"].dt.to_period("M").to_numpy()
    person = panel.frame.loc[finite, "person"].to_numpy()

    n = int(min(marked.sum(), (~marked).sum()))
    idx_hi = rng.choice(np.flatnonzero(marked), n, replace=False)
    idx_lo = rng.choice(np.flatnonzero(~marked), n, replace=False)

    corr_hi, corr_lo = spearman_matrix(x[idx_hi]), spearman_matrix(x[idx_lo])
    pooled = spearman_matrix(x)
    pos, membership = layout_and_communities(pooled, names)
    edges = top_edges(pooled, density=0.13)

    # max-statistic permutation: relabel nightmare status within calendar month.
    delta = corr_hi - corr_lo
    iu = np.triu_indices_from(delta, k=1)
    both = np.concatenate([idx_hi, idx_lo])
    strata = dates[both]
    maxstat = np.empty(300)
    for b in range(len(maxstat)):
        lab = np.zeros(len(both), bool)
        for s in np.unique(strata):
            m = strata == s
            k = m.sum() // 2
            lab[np.flatnonzero(m)[rng.permutation(m.sum())[:k]]] = True
        d = spearman_matrix(x[both[lab]]) - spearman_matrix(x[both[~lab]])
        maxstat[b] = np.abs(d[iu]).max()
    thresh = float(np.quantile(maxstat, 0.95))
    survivors = [(int(i), int(j)) for i, j in zip(*iu) if abs(delta[i, j]) > thresh]

    fig = plt.figure(figsize=(18.0, 7.4), facecolor=BG)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.12, 0.78], wspace=0.10)

    ax = fig.add_subplot(gs[0, 0], facecolor=PANEL)
    node_size = 40 + 900 * np.abs(x).mean(axis=0)
    draw_network(ax, pooled, pos, edges, names, membership, node_size,
                 title=f"The atlas — all English dreams (n={len(x):,})")

    ax = fig.add_subplot(gs[0, 1], facecolor=PANEL)
    cpos, cangles = circular_positions(membership)
    top = sorted(survivors, key=lambda e: abs(delta[e]), reverse=True)[:45]
    draw_chords(ax, top, [delta[e] for e in top], cpos, cangles, membership, names)
    ax.set_title(f"Δρ after a text-lexicon split — {len(top)} strongest of "
                 f"{len(survivors)} changes\nbeyond the max-|Δρ| threshold {thresh:.3f} "
                 f"({len(iu[0])} edges tested)", fontsize=10, color=FG, pad=10)

    ax = fig.add_subplot(gs[0, 2], facecolor=PANEL)
    for spine in ax.spines.values():
        spine.set_color("#232A38")
    ax.tick_params(colors=MUTED, labelsize=7.5)
    # share of the network's total coupling, so overall shrinkage cannot masquerade as rewiring
    raw_hi = np.abs(corr_hi).sum(axis=1) - 1.0
    raw_lo = np.abs(corr_lo).sum(axis=1) - 1.0
    strength_hi = 100 * raw_hi / raw_hi.sum()
    strength_lo = 100 * raw_lo / raw_lo.sum()
    order = np.argsort(strength_hi - strength_lo)
    y = np.arange(len(names))
    for k, i in enumerate(order):
        rising = strength_hi[i] >= strength_lo[i]
        ax.plot([strength_lo[i], strength_hi[i]], [k, k],
                color=WARM if rising else COOL, lw=2.0, alpha=0.75, zorder=1,
                solid_capstyle="round")
    ax.scatter(strength_lo[order], y, s=26, color=MUTED, zorder=3, linewidths=0, label="unmarked")
    ax.scatter(strength_hi[order], y, s=34, color=GOLD, zorder=4, linewidths=0,
               label="threat-marked")
    ax.set_yticks(y)
    ax.set_yticklabels([names[i].replace("place_", "").replace("situation_", "")
                        .replace("object_", "") for i in order], fontsize=7.2, color=FG)
    ax.set_ylim(-0.8, len(names) - 0.2)
    ax.set_xlabel("share of the network's total |ρ|   (%)", fontsize=8.5)
    ax.set_title("Which axes take a larger share of the coupling", fontsize=10, color=FG, pad=10)
    leg = ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    for t in leg.get_texts():
        t.set_color(MUTED)

    fig.suptitle(
        "Same 29 axes, two states of the night — English dreams, within-person deviations",
        color=FG, fontsize=14, y=1.0)
    fig.text(0.5, -0.02,
             "Edges are Spearman correlations of leave-one-out within-person deviations, so a "
             "contributor's stable style is removed before anything is correlated. Left: pooled "
             "network at 13% density, node colour = Louvain community, node size = mean absolute "
             "deviation. Middle: change in edge strength between reports whose text fires the "
             f"frozen anxiety-or-death lexicon and reports that do not ({n:,} each, warm = "
             "strengthened on threatening nights, cool = weakened), keeping only edges beyond the "
             "95th percentile of the maximum |Δρ| under 300 month-stratified label permutations. "
             "Right: each axis's share of its network's total coupling, so an overall change in "
             "correlation magnitude cannot masquerade as rewiring. The splitter is text-derived "
             "on purpose: the app's nightmare index is fear, danger and sadness minus joy, so "
             "conditioning on it would condition the network on four of its own nodes. Selecting "
             "threatening reports still compresses variance in the threat axes, which lowers "
             "their correlations mechanically — read the right panel as redistribution of "
             "coupling, not as fear disconnecting. The lexicon is an independent code path but "
             "not an independent construct: it fires on words the annotator also reads. Unit is "
             "the report, not the contributor-night. Descriptive and not pre-registered.",
             color=MUTED, fontsize=8, ha="center", va="top", wrap=True)
    save(fig, "fig1-nightmare-constellation")

    pd.DataFrame({
        "feature_i": [names[i] for i, _ in survivors],
        "feature_j": [names[j] for _, j in survivors],
        "rho_threat_marked": [corr_hi[i, j] for i, j in survivors],
        "rho_unmarked": [corr_lo[i, j] for i, j in survivors],
        "delta": [delta[i, j] for i, j in survivors],
    }).sort_values("delta", key=np.abs, ascending=False).to_csv(
        TABLES / "nightmare-vs-calm-edges.csv", index=False)
    pd.DataFrame([{
        "n_panel_reports": len(x),
        "n_panel_users": int(np.unique(person).size),
        "n_per_group": n,
        "n_threat_group_users": int(np.unique(person[idx_hi]).size),
        "n_unmarked_group_users": int(np.unique(person[idx_lo]).size),
        "edges_tested": len(iu[0]),
        "edges_surviving": len(survivors),
        "maxstat_threshold": thresh,
    }]).to_csv(TABLES / "nightmare-network-summary.csv", index=False)
    return {"n_per_group": int(n), "edges_surviving": len(survivors), "maxstat_threshold": thresh}


# --------------------------------------------------------------------------------------
# figure 2: the arrow of time as a flow network

CORPORA = [
    ("true_xlmr_arrow_DreamSeer_EN.npz", "DreamSeer EN", "DreamSeer dreams (EN)", GOLD),
    ("true_xlmr_arrow_Reddit_rDreams.npz", "Reddit r/Dreams", "Reddit r/Dreams", MINT),
    ("true_xlmr_arrow_confession.npz", "r/confession (waking narrative)",
     "r/confession (prose control)", COOL),
    ("true_xlmr_arrow_wikipedia.npz", "Wikipedia openings",
     "Wikipedia (prose control)", VIOLET),
]


def stage_means(counts, vals, n_stage=6, min_sent=6):
    """Mean sentiment per narrative stage and indices of retained documents."""
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rows, keep = [], []
    for i, (off, c) in enumerate(zip(offsets, counts)):
        if c < min_sent:
            continue
        v = vals[off:off + c]
        if not np.isfinite(v).all():
            continue
        pos = (np.arange(c) + 0.5) / c
        stage = np.clip((pos * n_stage).astype(int), 0, n_stage - 1)
        row = [v[stage == s].mean() for s in range(n_stage)]
        rows.append(row)
        keep.append(i)
    return np.asarray(rows, float), np.asarray(keep, np.int32)


def phase_author_codes(source, expected_counts):
    """Replay the primary-instrument sample and return opaque contributor codes.

    The score cache intentionally carries no author key. Replaying the deterministic loader is
    therefore required to enforce the five-contributor publication floor without writing IDs.
    """
    _stem, _kind, loader = TEXT_CORPORA[source]
    texts, authors = loader()
    counts, kept_authors = [], []
    for text, author in zip(texts, authors):
        sentences = split_sentences(text, min_words=2)[:25]
        if len(sentences) < MIN_SENT_ARROW:
            continue
        counts.append(len(sentences))
        kept_authors.append(author)
        if len(counts) >= len(expected_counts):
            break
    del texts, authors
    gc.collect()
    counts = np.asarray(counts, np.int32)
    if len(counts) != len(expected_counts) or not np.array_equal(counts, expected_counts):
        raise RuntimeError(f"{source}: contributor replay does not match sentiment cache")
    return pd.factorize(pd.Series(kept_authors, dtype="object"))[0].astype(np.int32)


def _alluvial(ax, band, authors, n_stage, n_band, band_colors, gap=0.018, node_w=0.055):
    """Stacked-band alluvial: node heights are occupancy, ribbon heights are transition mass."""
    n_doc = len(band)
    occupancy = np.stack([np.bincount(band[:, s], minlength=n_band) for s in range(n_stage)])
    span = 1.0 - gap * (n_band - 1)
    bottoms = np.zeros((n_stage, n_band))
    heights = occupancy / n_doc * span
    for s in range(n_stage):
        y = 0.0
        for b in range(n_band):
            bottoms[s, b] = y
            y += heights[s, b] + gap

    for s in range(n_stage - 1):
        pairs = np.zeros((n_band, n_band))
        for a, b in zip(band[:, s], band[:, s + 1]):
            pairs[a, b] += 1
        src = bottoms[s].copy()
        # allocate outgoing slots bottom-up by target band so ribbons cross minimally
        for a in range(n_band):
            for b in range(n_band):
                w = pairs[a, b]
                pair_mask = (band[:, s] == a) & (band[:, s + 1] == b)
                if w < MIN_CELL or np.unique(authors[pair_mask]).size < MIN_USERS:
                    src[a] += w / n_doc * span
                    continue
                h = w / n_doc * span
                tgt = bottoms[s + 1, b] + pairs[:a, b].sum() / n_doc * span
                x0, x1 = s + node_w, s + 1 - node_w
                mid = (x0 + x1) / 2
                lower = [(x0, src[a]), (mid, src[a]), (mid, tgt), (x1, tgt)]
                upper = [(x1, tgt + h), (mid, tgt + h), (mid, src[a] + h), (x0, src[a] + h)]
                verts = lower + upper + [(x0, src[a])]
                codes = ([MplPath.MOVETO] + [MplPath.CURVE4] * 3
                         + [MplPath.LINETO] + [MplPath.CURVE4] * 3 + [MplPath.CLOSEPOLY])
                colour = WARM if b > a else COOL if b < a else "#5A6478"
                ax.add_patch(PathPatch(MplPath(verts, codes), lw=0,
                                       facecolor=matplotlib.colors.to_rgba(colour, 0.42),
                                       zorder=1))
                src[a] += h

    for s in range(n_stage):
        for b in range(n_band):
            ax.add_patch(plt.Rectangle((s - node_w, bottoms[s, b]), 2 * node_w, heights[s, b],
                                       facecolor=band_colors[b], lw=0, zorder=3))
    return bottoms, heights


def figure_arrow_flow(n_stage=6, n_level=9, n_boot=400, n_thread=140):
    """Phase portrait: where mood drifts next, given where in the story you are and how dark."""
    rng = np.random.default_rng(SEED)
    loaded = []
    for fname, source, label, color in CORPORA:
        z = np.load(C.INTERIM / fname, allow_pickle=True)
        authors = phase_author_codes(source, z["counts"])
        means, keep = stage_means(z["counts"], z["vals"], n_stage)
        loaded.append((label, color, means, authors[keep]))
    pooled = np.concatenate([m.ravel() for _, _, m, _ in loaded])
    edges_y = np.quantile(pooled, np.linspace(0.02, 0.98, n_level + 1))
    centres = (edges_y[:-1] + edges_y[1:]) / 2

    fig, axes = styled((18.0, 6.2), 1, len(loaded), sharey=True,
                       gridspec_kw={"wspace": 0.08})
    summary = []
    cell_summary = []
    for ax, (label, color, m, authors) in zip(axes, loaded):
        level = np.clip(np.digitize(m, edges_y[1:-1]), 0, n_level - 1)
        density = np.full((n_level, n_stage), np.nan)
        for s in range(n_stage):
            for lv in range(n_level):
                sel = level[:, s] == lv
                if sel.sum() >= MIN_CELL and np.unique(authors[sel]).size >= MIN_USERS:
                    density[lv, s] = sel.sum() / len(m)
                    cell_summary.append({
                        "corpus": label,
                        "stage": s,
                        "level": lv,
                        "n_reports": int(sel.sum()),
                        "n_contributors": int(np.unique(authors[sel]).size),
                        "has_next_stage_arrow": s < n_stage - 1,
                    })
        ax.imshow(density, origin="lower", aspect="auto", cmap="magma", alpha=0.62,
                  extent=(-0.5, n_stage - 0.5, edges_y[0], edges_y[-1]),
                  interpolation="bicubic", zorder=0)

        thread = rng.choice(len(m), min(n_thread, len(m)), replace=False)
        ax.plot(np.arange(n_stage), m[thread].T, color="white", lw=0.5, alpha=0.10, zorder=1)

        for s in range(n_stage - 1):
            for lv in range(n_level):
                sel = level[:, s] == lv
                if sel.sum() < MIN_CELL or np.unique(authors[sel]).size < MIN_USERS:
                    continue
                dy = float(np.mean(m[sel, s + 1] - m[sel, s]))
                ax.annotate("", xy=(s + 0.86, centres[lv] + 0.86 * dy),
                            xytext=(s + 0.06, centres[lv]),
                            arrowprops=dict(arrowstyle="-|>", lw=0.6 + 2.6 * min(1, abs(dy) / 0.35),
                                            color=matplotlib.colors.to_rgba(
                                                WARM if dy > 0 else COOL,
                                                0.35 + 0.5 * min(1, abs(dy) / 0.35)),
                                            shrinkA=0, shrinkB=0), zorder=3)

        mean = m.mean(axis=0)
        boot = np.array([m[rng.integers(len(m), size=len(m))].mean(axis=0) for _ in range(n_boot)])
        lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
        ax.fill_between(range(n_stage), lo, hi, color=FG, alpha=0.28, lw=0, zorder=4)
        ax.plot(range(n_stage), mean, color=FG, lw=3.0, zorder=5)
        ax.plot(range(n_stage), mean, color=color, lw=1.6, marker="o", ms=4.5, zorder=6)

        drift = mean[-1] - mean[0]
        drift_ci = np.quantile(boot[:, -1] - boot[:, 0], [0.025, 0.975])
        ax.set_title(f"{label}\nn = {len(m):,}   ·   close − open = {drift:+.3f}\n"
                     f"[{drift_ci[0]:+.3f}, {drift_ci[1]:+.3f}]", fontsize=9.5, pad=8)
        ax.set_xticks(range(n_stage))
        ax.set_xticklabels(["open", "", "", "", "", "close"], fontsize=8.5)
        ax.set_xlim(-0.5, n_stage - 0.5)
        ax.set_ylim(edges_y[0], edges_y[-1])
        ax.set_xlabel("narrative position", fontsize=9)
        summary.append({"corpus": label, "n_docs": len(m), "mean_open": mean[0],
                        "mean_close": mean[-1], "close_minus_open": drift,
                        "ci_lo": drift_ci[0], "ci_hi": drift_ci[1]})
    axes[0].set_ylabel("sentence sentiment within the report", fontsize=9)

    handles = [plt.Line2D([], [], color=WARM, lw=2.4), plt.Line2D([], [], color=COOL, lw=2.4),
               plt.Line2D([], [], color=FG, lw=3.0)]
    leg = axes[-1].legend(handles, ["mood rises next", "mood falls next", "mean trajectory"],
                          fontsize=7.5, frameon=False, loc="lower left")
    for t in leg.get_texts():
        t.set_color(MUTED)

    fig.suptitle("The phase portrait of a narrative: where mood drifts next, given where you are "
                 "in the story", color=FG, fontsize=14, y=1.02)
    fig.text(0.5, -0.09,
             "Every document is cut into six narrative stages and its stage mood binned on a "
             "common scale pooled across all four corpora. Arrows are the mean next-stage change "
             "for documents currently at that stage and mood level (warm = rises, cool = falls); "
             f"cells holding fewer than {MIN_CELL} documents or {MIN_USERS} distinct contributors "
             "are left empty. Background heat is "
             "the share of documents at each level within a stage, faint white threads are "
             f"{n_thread} individual trajectories, and the bright line is the mean with a "
             f"{n_boot}-draw report-level bootstrap band. All four corpora drift down — the "
             "descent is not dreaming's property — and DreamSeer falls furthest; r/Dreams and "
             "r/confession are not separated by these intervals. The "
             "arrows flip sign with height because a stage scored from a handful of sentences "
             "regresses toward the mean; what distinguishes the corpora is where that balance "
             "point sits and how fast the mean line falls, not the fan itself. Report-level "
             "resampling here, not the contributor-clustered inference behind the published "
             "arrow estimates.",
             color=MUTED, fontsize=8, ha="center", va="top", wrap=True)
    save(fig, "fig2-arrow-phase-portrait")
    pd.DataFrame(summary).to_csv(TABLES / "arrow-flow-drift.csv", index=False)
    pd.DataFrame(cell_summary).to_csv(TABLES / "phase-portrait-cells.csv", index=False)
    return summary


# --------------------------------------------------------------------------------------
# figure 3: event-locked breathing

WINDOWS = {"−21..−8 d": (-21, -8), "−7..−1 d": (-7, -1), "0..+6 d": (0, 6), "+7..+20 d": (7, 20)}


def figure_event_breathing(panel, rng, offsets=range(28, 259, 7)):
    names = list(panel.feature_names)
    x = panel.state
    ok = np.isfinite(x).all(axis=1)
    x = x[ok]
    dates = panel.frame.loc[ok, "date"].reset_index(drop=True)

    shocks = pd.read_csv(C.EVENTS_DIR / "curated_global_shocks.csv", parse_dates=["date"])
    shocks = shocks[shocks.cohorts.str.contains("en")]
    covered = shocks[(shocks.date >= dates.min() + pd.Timedelta(days=25))
                     & (shocks.date <= dates.max() - pd.Timedelta(days=25))]
    anchors = list(covered.date)

    person = panel.frame.loc[ok, "person"].to_numpy()

    masks = window_masks(dates, anchors, WINDOWS)
    observed, sizes, users, corrs = {}, {}, {}, {}
    for name, m in masks.items():
        sizes[name] = int(m.sum())
        users[name] = int(np.unique(person[m]).size)
        if sizes[name] < MIN_CELL or users[name] < MIN_USERS:
            raise ValueError(f"window {name} below the disclosure floor")
        corrs[name] = spearman_matrix(x[m])
        observed[name] = network_stats(corrs[name])

    # Two placebo-in-time families, because the answer depends on which one you grant.
    # LOCAL: shift the cohort by a modest common offset, so the comparison stays in the same
    # era of the platform. ROTATION: wrap the cohort anywhere in the window, a broader null
    # that also compares across eras in which the platform itself was different.
    lo_day = dates.min() + pd.Timedelta(days=25)
    hi_day = dates.max() - pd.Timedelta(days=25)
    span = (hi_day - lo_day).days
    step = int(np.diff(list(offsets)[:2])[0])

    local_sets = []
    for sign in (-1, 1):
        for off in offsets:
            fake = [a + sign * pd.Timedelta(days=int(off)) for a in anchors]
            if min(fake) >= lo_day and max(fake) <= hi_day:
                local_sets.append(fake)
    rotation_sets = [[lo_day + pd.Timedelta(days=int(((a - lo_day).days + off) % span))
                      for a in anchors] for off in range(min(offsets), span - min(offsets), step)]

    def build_null(anchor_sets):
        acc = {k: {s: [] for s in observed[k]} for k in masks}
        for fake in anchor_sets:
            for name, m in window_masks(dates, fake, WINDOWS).items():
                if m.sum() < MIN_CELL or np.unique(person[m]).size < MIN_USERS:
                    continue
                for k, v in network_stats(spearman_matrix(x[m])).items():
                    acc[name][k].append(v)
        return acc

    nulls = {"local": build_null(local_sets), "rotation": build_null(rotation_sets)}
    null = nulls["local"]
    n_used = len(local_sets)
    n_rot = len(rotation_sets)

    pooled = spearman_matrix(x)
    pos, membership = layout_and_communities(pooled, names)
    edges = top_edges(pooled, density=0.13)

    # leave-one-event-out: does one shock carry the whole picture?
    loo = []
    for drop in anchors:
        rest = [a for a in anchors if a != drop]
        m = window_masks(dates, rest, {"late": WINDOWS["+7..+20 d"]})["late"]
        st = network_stats(spearman_matrix(x[m]))
        loo.append({"dropped": f"{drop:%Y-%m-%d}", "n": int(m.sum()),
                    "n_users": int(np.unique(person[m]).size), **st})
    loo = pd.DataFrame(loo)

    fig = plt.figure(figsize=(17.6, 9.8), facecolor=BG)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1.0], hspace=0.36, wspace=0.22)
    deltas = {k: corrs[k] - pooled for k in corrs}
    vmax = max(np.abs(d[np.triu_indices_from(d, 1)]).max() for d in deltas.values())
    for c, (name, corr) in enumerate(corrs.items()):
        ax = fig.add_subplot(gs[0, c], facecolor=PANEL)
        node_size = 40 + 900 * np.abs(x[masks[name]]).mean(axis=0)
        d = deltas[name]
        segs, colors, widths = [], [], []
        for i, j in edges:
            v = d[i, j]
            segs.append([pos[i], pos[j]])
            colors.append(matplotlib.colors.to_rgba(WARM if v > 0 else COOL,
                                                    0.15 + 0.8 * min(1, abs(v) / vmax)))
            widths.append(0.4 + 5.0 * min(1, abs(v) / vmax))
        ax.add_collection(LineCollection(segs, colors=colors, linewidths=widths, zorder=1))
        palette = [GOLD, COOL, MINT, VIOLET, WARM, "#E86A92"]
        xy = np.array([pos[i] for i in range(len(names))])
        ax.scatter(xy[:, 0], xy[:, 1], s=node_size * 0.5,
                   c=[palette[m % len(palette)] for m in membership], zorder=3,
                   linewidths=0.5, edgecolors=BG)
        for i, nm_ in enumerate(names):
            ax.annotate(nm_.replace("place_", "").replace("situation_", "").replace("object_", ""),
                        pos[i], fontsize=5.8, color=FG, ha="center", va="center",
                        xytext=(0, -8), textcoords="offset points", alpha=0.6)
        bare(ax)
        ax.autoscale_view()
        ax.margins(0.12)
        ax.set_title(f"{name}   n={sizes[name]:,}\nmean|ρ| = "
                     f"{observed[name]['mean_abs_rho']:.3f}", fontsize=9.5, color=FG, pad=8)

    labels = list(WINDOWS)
    stats_meta = [("mean_abs_rho", "integration   mean |ρ|"),
                  ("lambda1_share", "dominance   λ₁ share"),
                  ("participation_ratio", "effective dimensionality")]
    rows = []
    for c, (key, title) in enumerate(stats_meta):
        ax = fig.add_subplot(gs[1, c], facecolor=PANEL)
        for spine in ax.spines.values():
            spine.set_color("#232A38")
        ax.tick_params(colors=MUTED, labelsize=8)
        obs = np.array([observed[n][key] for n in labels])
        wide = np.array([np.quantile(nulls["rotation"][n][key], [0.025, 0.975]) for n in labels])
        tight = np.array([np.quantile(nulls["local"][n][key], [0.025, 0.5, 0.975])
                          for n in labels])
        ax.fill_between(range(len(labels)), wide[:, 0], wide[:, 1], color=MUTED, alpha=0.14,
                        lw=0, label="rotation placebo 95%")
        ax.fill_between(range(len(labels)), tight[:, 0], tight[:, 2], color=MUTED, alpha=0.30,
                        lw=0, label="local-shift placebo 95%")
        ax.plot(range(len(labels)), tight[:, 1], color=MUTED, lw=1.0, ls="--")
        ax.plot(range(len(labels)), obs, color=GOLD, lw=2.2, marker="o", ms=5, label="shocks")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, color=MUTED)
        ax.set_title(title, fontsize=9.5, color=FG)
        if c == 0:
            leg = ax.legend(fontsize=7, frameon=False, loc="best")
            for t in leg.get_texts():
                t.set_color(MUTED)
        for i, n in enumerate(labels):
            row = {"window": n, "statistic": key, "n_reports": sizes[n], "n_users": users[n],
                   "observed": obs[i]}
            for fam in ("local", "rotation"):
                draws = np.asarray(nulls[fam][n][key], float)
                exceed = int((np.abs(draws - draws.mean()) >= abs(obs[i] - draws.mean())).sum())
                row[f"{fam}_draws"] = len(draws)
                row[f"{fam}_mean"] = draws.mean()
                row[f"{fam}_exceedances"] = exceed
                # +1 rule: a Monte-Carlo p can never be zero, and the local family has 12 draws
                row[f"{fam}_p"] = (1 + exceed) / (1 + len(draws))
            rows.append(row)

    ax = fig.add_subplot(gs[1, 3], facecolor=PANEL)
    bare(ax)
    pr = loo["participation_ratio"]
    cut = float(np.quantile(nulls["local"]["+7..+20 d"]["participation_ratio"], 0.975))
    cut_rot = float(np.quantile(nulls["rotation"]["+7..+20 d"]["participation_ratio"], 0.975))
    loo["local_pr_q975"] = cut
    loo["rotation_pr_q975"] = cut_rot
    loo["clears_local_pr_q975"] = loo["participation_ratio"] > cut
    loo["clears_rotation_pr_q975"] = loo["participation_ratio"] > cut_rot
    lines = [f"{len(covered)} curated global shocks"]
    lines += [f"  {d:%Y-%m-%d}  {l[:42]}" for d, l in zip(covered.date, covered.label)]
    lines += ["", "Leave-one-event-out, +7..+20 d",
              f"  effective dimensionality {pr.min():.2f} – {pr.max():.2f}",
              f"  {int((pr > cut).sum())} of {len(loo)} subsets clear the local band ({cut:.2f})",
              f"  {int((pr > cut_rot).sum())} of {len(loo)} clear the rotation band "
              f"({cut_rot:.2f})",
              "", f"Placebo cohorts: {n_used} local, {n_rot} rotation",
              "  local is an era-matched envelope,",
              "  not a p-value at this resolution"]
    ax.text(0.0, 1.0, "\n".join(lines), color=MUTED, fontsize=7.4, va="top", ha="left",
            transform=ax.transAxes, linespacing=1.55)

    fig.suptitle("Does the dream network breathe around global shocks?", color=FG, fontsize=13.5,
                 y=0.965)
    fig.text(0.5, 0.045,
             "Top: the same fixed-density skeleton drawn from event-locked pools of English "
             "within-person deviations, with each edge shaded by its departure from the pooled "
             "network (warm = stronger inside that window, cool = weaker) so small changes are "
             "visible at all. Bottom: three threshold-free integration statistics against "
             "two families of placebo-in-time cohorts, both preserving inter-event spacing and "
             f"moving only calendar alignment: {n_used} local shifts that keep the cohort in the "
             f"same era of the platform, and {n_rot} full-window rotations. Read the local family "
             "as an era-matched envelope rather than a p-value — twelve draws cannot resolve one "
             "— and the rotation family as the calibrated comparison. The late window falls "
             "outside all twelve local shifts and inside the rotation band, and that "
             "disagreement is the result: the platform's own network drifts across eras "
             "(Figure 4), so a rotation compares windows drawn from different instruments. The "
             "unit is the report, not the contributor-night, so several reports from one person "
             "on one night all count; the four windows overlap in event support and are not "
             "independent; the three statistics are functions of the same matrix; nothing is "
             "corrected for multiplicity and none of this is a test of a hypothesis.",
             color=MUTED, fontsize=8, ha="center", va="top", wrap=True)
    save(fig, "fig3-event-breathing")
    pd.DataFrame(rows).to_csv(TABLES / "event-breathing-stats.csv", index=False)
    loo.to_csv(TABLES / "event-breathing-leave-one-out.csv", index=False)
    return rows


# --------------------------------------------------------------------------------------
# figure 4: breathing through calendar time


def rolling_network_series(x, day, person, grid, win, min_n):
    out = []
    for centre in grid:
        m = (day >= centre - win // 2) & (day < centre + win // 2)
        n_users = int(np.unique(person[m]).size)
        if m.sum() < min_n or n_users < MIN_USERS:
            out.append({"mean_abs_rho": np.nan, "lambda1_share": np.nan,
                        "participation_ratio": np.nan, "n": int(m.sum()),
                        "n_users": n_users})
            continue
        st = network_stats(spearman_matrix(x[m]))
        st["n"] = int(m.sum())
        st["n_users"] = n_users
        out.append(st)
    return pd.DataFrame(out, index=pd.DatetimeIndex(grid))


def figure_time_breathing(panel, rng, win=91, step=7, min_n=150, n_null=120):
    x = panel.state
    ok = np.isfinite(x).all(axis=1)
    x = x[ok]
    frame = panel.frame.loc[ok].reset_index(drop=True)
    order = np.argsort(frame["date"].to_numpy())
    x, frame = x[order], frame.iloc[order].reset_index(drop=True)
    day = frame["date"].to_numpy("datetime64[D]")
    person = frame["person"].to_numpy()

    grid = pd.date_range(frame["date"].min() + pd.Timedelta(days=win // 2),
                         frame["date"].max() - pd.Timedelta(days=win // 2), freq=f"{step}D")
    grid_d = grid.to_numpy("datetime64[D]")
    obs = rolling_network_series(x, day, person, grid_d, np.timedelta64(win, "D"), min_n)

    # chronology rotation: keep each window's size and the local content blocks, break alignment.
    null = {k: np.full((n_null, len(grid)), np.nan) for k in
            ("mean_abs_rho", "lambda1_share", "participation_ratio")}
    for b in range(n_null):
        shift = rng.integers(len(x))
        rolled = np.roll(x, shift, axis=0)
        s = rolling_network_series(rolled, day, person, grid_d, np.timedelta64(win, "D"), min_n)
        for k in null:
            null[k][b] = s[k].to_numpy()

    daily = frame.assign(neg=panel.frame.loc[ok, "nightmare_index"].to_numpy()[order]).groupby(
        frame["date"].dt.normalize())["nightmare_index"].agg(["mean", "size"])
    daily = daily[daily["size"] >= MIN_CELL]
    resid = daily["mean"] - daily["mean"].rolling(180, min_periods=60, center=True).median()
    var = resid.rolling(56, min_periods=28).var()
    ar1 = resid.rolling(56, min_periods=28).apply(
        lambda a: np.corrcoef(a[:-1], a[1:])[0, 1] if np.isfinite(a).all() else np.nan, raw=True)

    shocks = pd.read_csv(C.EVENTS_DIR / "curated_global_shocks.csv", parse_dates=["date"])
    shocks = shocks[(shocks.date >= grid.min()) & (shocks.date <= grid.max())]

    volume = pd.read_parquet(C.DREAMS_OUT / "dreamseer_dream_level.parquet",
                             columns=["userID", "date", "lang"])
    volume = volume[volume.lang.eq("en")]
    daily_vol = volume.groupby(volume.date.dt.normalize()).agg(
        reports=("userID", "size"), contributors=("userID", "nunique"))
    daily_vol = daily_vol.reindex(
        pd.date_range(daily_vol.index.min(), daily_vol.index.max())).fillna(0).rolling(28).mean()

    fig, axes = styled((15.5, 11.6), 4, 1, sharex=True,
                       gridspec_kw={"height_ratios": [1, 1, 1, 0.8], "hspace": 0.30})
    specs = [("mean_abs_rho", "integration   mean |ρ|", GOLD),
             ("participation_ratio", "effective dimensionality", MINT),
             ("ews", "volatility diagnostic: variance and lag-1 autocorrelation of the daily "
                     "nightmare-index residual", COOL),
             ("volume", "what the instrument was doing: English reports and contributors per day",
              VIOLET)]
    for ax, (key, title, color) in zip(axes, specs):
        if key == "ews":
            ax.plot(var.index.to_numpy(), var.to_numpy(), color=COOL, lw=1.8,
                    label="rolling variance")
            twin = ax.twinx()
            twin.plot(ar1.index.to_numpy(), ar1.to_numpy(), color=WARM, lw=1.4, alpha=0.9)
            twin.set_ylabel("lag-1 autocorrelation", color=WARM, fontsize=9)
            twin.tick_params(colors=WARM, labelsize=8)
            for spine in twin.spines.values():
                spine.set_color("#232A38")
            ax.set_ylabel("variance", fontsize=9)
        elif key == "volume":
            ax.fill_between(daily_vol.index.to_numpy(), daily_vol.reports.to_numpy(),
                            color=VIOLET, alpha=0.35, lw=0,
                            label="reports / day (28-day mean)")
            ax.plot(daily_vol.index.to_numpy(), daily_vol.contributors.to_numpy(),
                    color=GOLD, lw=1.5,
                    label="distinct contributors / day")
            ax.set_ylabel("per day", fontsize=9)
        else:
            lo, hi = np.nanquantile(null[key], [0.025, 0.975], axis=0)
            grid_values = grid.to_numpy()
            ax.fill_between(grid_values, lo, hi, color=MUTED, alpha=0.22, lw=0,
                            label="chronology-rotation 95%")
            ax.plot(grid_values, obs[key].to_numpy(), color=color, lw=2.0, label="observed")
            ax.set_ylabel(title.split("   ")[-1], fontsize=9)
        ax.set_title(title, fontsize=10, loc="left", color=FG)
        for _, s in shocks.iterrows():
            ax.axvline(s.date, color=VIOLET, lw=0.8, alpha=0.5, zorder=0)
        leg = ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        for t in leg.get_texts():
            t.set_color(MUTED)
    axes[-1].set_xlabel("date")
    axes[-1].set_xlim(grid.min() - pd.Timedelta(days=60), grid.max() + pd.Timedelta(days=60))

    fig.suptitle("Does it breathe on its own? Network integration through calendar time",
                 color=FG, fontsize=13.5, y=0.945)
    fig.text(0.5, 0.045,
             f"{win}-day rolling windows stepped every {step} days over English within-person "
             f"deviations (windows below {min_n} reports are blank). The grey band is "
             f"{n_null} circular rotations of the report block against the date axis, which keeps "
             "every window's size and local content structure but destroys calendar alignment. "
             "Violet lines mark the ten curated shocks. Read this figure as a warning before "
             "reading it as a result: the network tightens and effective dimensionality falls "
             "from 7.38 in the first window to 6.11 in the last (1.27 dimensions), while the bottom "
             "panel shows the platform growing "
             "underneath it, so contributor composition and any change in the annotation "
             "instrument are live explanations for the trend. The rotation keeps the dates and "
             "window sizes where they are and shifts the report block against them, so "
             "composition-by-calendar survives it and remains a rival explanation. The third "
             "panel is a volatility diagnostic on the daily residual, not an early-warning test: "
             "no run-up window, no surrogate family, no declared transition.",
             color=MUTED, fontsize=8, ha="center", va="top", wrap=True)
    save(fig, "fig4-time-breathing")

    out = obs.copy()
    out.index.name = "window_centre"
    for k in null:
        out[f"{k}_null_lo"] = np.nanquantile(null[k], 0.025, axis=0)
        out[f"{k}_null_hi"] = np.nanquantile(null[k], 0.975, axis=0)
    out.to_csv(TABLES / "time-breathing-series.csv")
    return out


# --------------------------------------------------------------------------------------
# figure 5: the dream galaxy


def figure_galaxy(rng, n_sample=2600, k=8, n_cluster=12):
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.neighbors import NearestNeighbors

    from psychohistory.dreams.embed_cache import load_or_build

    meta, emb = load_or_build()
    keep = np.flatnonzero((meta.lang == "en").to_numpy())
    pick = rng.choice(keep, min(n_sample, len(keep)), replace=False)
    sub, e = meta.iloc[pick].reset_index(drop=True), emb[pick]

    xy = TSNE(n_components=2, perplexity=32, init="pca", random_state=SEED,
              metric="cosine").fit_transform(e)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(e)
    _, nbr = nn.kneighbors(e)
    pairs = {(min(i, j), max(i, j)) for i, row in enumerate(nbr) for j in row[1:]}
    # cluster in the plane: a label has to describe the region it is printed on
    km = KMeans(n_cluster, n_init=10, random_state=SEED).fit(xy)

    feats = C.EMOTIONS + C.TAGS
    lv = pd.read_parquet(C.DREAMS_OUT / "dreamseer_dream_level.parquet",
                         columns=["documentID", *feats]).set_index("documentID")
    f = lv.reindex(sub.documentID.values).to_numpy(float)
    z = (f - np.nanmean(f, axis=0)) / (np.nanstd(f, axis=0) + 1e-9)

    from scipy.ndimage import gaussian_filter
    from scipy.stats import binned_statistic_2d

    nm = sub["nightmare_index"].to_numpy(float)
    fig, ax = styled((13.6, 11.6))

    # smoothed nightmare field over the semantic map, masked where the map is empty
    nbin = 190
    rng_x = (xy[:, 0].min() - 2, xy[:, 0].max() + 2)
    rng_y = (xy[:, 1].min() - 2, xy[:, 1].max() + 2)
    total = binned_statistic_2d(xy[:, 0], xy[:, 1], nm, "sum", bins=nbin,
                                range=[rng_x, rng_y]).statistic
    count = binned_statistic_2d(xy[:, 0], xy[:, 1], nm, "count", bins=nbin,
                                range=[rng_x, rng_y]).statistic
    smooth_total = gaussian_filter(np.nan_to_num(total), 5.0)
    smooth_count = gaussian_filter(count, 5.0)
    field = np.where(smooth_count > 1e-3, smooth_total / np.maximum(smooth_count, 1e-9), np.nan)
    field = np.ma.masked_invalid(np.where(smooth_count < np.nanmax(smooth_count) * 0.02,
                                          np.nan, field)).T
    extent = (*rng_x, *rng_y)
    im = ax.imshow(field, origin="lower", extent=extent, cmap="inferno", alpha=0.95, zorder=1,
                   interpolation="bicubic",
                   vmin=np.nanpercentile(field.compressed(), 3),
                   vmax=np.nanpercentile(field.compressed(), 97))
    levels = np.nanpercentile(field.compressed(), [75, 88, 96])
    ax.contour(field, levels=levels, extent=extent, colors=[FG], linewidths=0.7, alpha=0.45,
               zorder=3)

    ax.add_collection(LineCollection([[xy[i], xy[j]] for i, j in pairs],
                                     colors=[(1, 1, 1, 0.05)], linewidths=0.3, zorder=2))
    ax.scatter(xy[:, 0], xy[:, 1], s=4.5, color="white", alpha=0.42, zorder=4, linewidths=0)

    for c in range(n_cluster):
        m = km.labels_ == c
        if m.sum() < MIN_CELL or pd.unique(sub.loc[m, "userID"]).size < MIN_USERS:
            continue
        best = np.nanmean(z[m], axis=0)
        top2 = " + ".join(feats[i].replace("place_", "").replace("object_", "")
                          for i in np.argsort(best)[::-1][:2])
        centre = np.median(xy[m], axis=0)
        ax.annotate(f"{top2}  ·  {m.sum()}", centre, color=FG, fontsize=8.5, ha="center",
                    va="center", alpha=0.95, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec="#3A4557", alpha=0.80))
    bare(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.01)
    cb.set_label("locally averaged nightmare index", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_edgecolor("#232A38")
    ax.set_title(f"A raw embedding orientation map — {len(sub):,} English dream reports",
                 fontsize=13, pad=12)
    fig.text(0.5, 0.045,
             "Dreams are placed by t-SNE on multilingual sentence-embedding geometry; faint "
             f"threads are the {k}-nearest-neighbour graph in the full 384-dimensional space, and "
             "white points are individual reports. The heat is the locally averaged nightmare "
             "index — a smoothed field, not a per-dream value — with contours at its 75th, 88th "
             "and 96th percentiles. Labels give the two strongest standardised content axes of a "
             "k-means region of the map and its size. This raw-embedding t-SNE is an orientation "
             "map: it does not use within-person deviations, preserve global distances, test "
             "connectivity, or establish natural regions. No text and no contributor identifiers "
             "are shown.",
             color=MUTED, fontsize=8, ha="center", va="top", wrap=True)
    save(fig, "fig5-dream-galaxy")

    users = sub["userID"].to_numpy()
    top = [np.argsort(np.nanmean(z[km.labels_ == c], axis=0))[::-1][:2]
           for c in range(n_cluster)]
    out = pd.DataFrame({
        "cluster": range(n_cluster),
        "n_reports": [int((km.labels_ == c).sum()) for c in range(n_cluster)],
        "n_users": [int(pd.unique(users[km.labels_ == c]).size) for c in range(n_cluster)],
        "top_axis_1": [feats[int(idx[0])] for idx in top],
        "top_axis_2": [feats[int(idx[1])] for idx in top],
        "mean_nightmare_index": [float(np.nanmean(nm[km.labels_ == c]))
                                 for c in range(n_cluster)],
    })
    return out[(out.n_reports >= MIN_CELL) & (out.n_users >= MIN_USERS)]


# --------------------------------------------------------------------------------------


def animate_breathing(panel, *, win=91, step=7, min_n=150, fps=9):
    """The network as a moving object: one frame per rolling window."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    names = list(panel.feature_names)
    x = panel.state
    ok = np.isfinite(x).all(axis=1)
    x = x[ok]
    frame = panel.frame.loc[ok].reset_index(drop=True)
    order = np.argsort(frame["date"].to_numpy())
    x, frame = x[order], frame.iloc[order].reset_index(drop=True)
    day = frame["date"].to_numpy("datetime64[D]")

    pooled = spearman_matrix(x)
    pos, membership = layout_and_communities(pooled, names)
    edges = top_edges(pooled, density=0.13)
    xy = np.array([pos[i] for i in range(len(names))])

    grid = pd.date_range(frame["date"].min() + pd.Timedelta(days=win // 2),
                         frame["date"].max() - pd.Timedelta(days=win // 2), freq=f"{step}D")
    frames = []
    for centre in grid.to_numpy("datetime64[D]"):
        m = (day >= centre - np.timedelta64(win // 2, "D")) & (
            day < centre + np.timedelta64(win // 2, "D"))
        n_users = int(frame.loc[m, "person"].nunique())
        if m.sum() < min_n or n_users < MIN_USERS:
            continue
        corr = spearman_matrix(x[m])
        frames.append({"date": pd.Timestamp(centre), "corr": corr, "n": int(m.sum()),
                       "n_users": n_users,
                       "size": 40 + 900 * np.abs(x[m]).mean(axis=0),
                       **network_stats(corr)})
    if not frames:
        return None
    vmax = max(np.abs(f["corr"] - pooled)[np.triu_indices(len(names), 1)].max() for f in frames)
    series = pd.DataFrame(frames)

    shocks = pd.read_csv(C.EVENTS_DIR / "curated_global_shocks.csv", parse_dates=["date"])
    shocks = shocks[(shocks.date >= series.date.min()) & (shocks.date <= series.date.max())]

    fig = plt.figure(figsize=(11.0, 8.6), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.16)
    ax = fig.add_subplot(gs[0], facecolor=PANEL)
    axl = fig.add_subplot(gs[1], facecolor=PANEL)
    for spine in axl.spines.values():
        spine.set_color("#232A38")
    axl.tick_params(colors=MUTED, labelsize=8)
    axl.plot(series.date.to_numpy(), series.participation_ratio.to_numpy(), color=MINT, lw=1.6)
    for _, s in shocks.iterrows():
        axl.axvline(s.date, color=VIOLET, lw=0.8, alpha=0.55)
    axl.set_ylabel("effective\ndimensionality", color=MUTED, fontsize=8.5)
    cursor = axl.axvline(series.date.iloc[0], color=GOLD, lw=1.8)
    palette = [GOLD, COOL, MINT, VIOLET, WARM, "#E86A92"]
    node_colors = [palette[m % len(palette)] for m in membership]

    def draw(idx):
        ax.clear()
        ax.set_facecolor(PANEL)
        f = frames[idx]
        d = f["corr"] - pooled
        segs, colors, widths = [], [], []
        for i, j in edges:
            v = d[i, j]
            segs.append([pos[i], pos[j]])
            colors.append(matplotlib.colors.to_rgba(WARM if v > 0 else COOL,
                                                    0.12 + 0.8 * min(1, abs(v) / vmax)))
            widths.append(0.4 + 5.5 * min(1, abs(v) / vmax))
        ax.add_collection(LineCollection(segs, colors=colors, linewidths=widths, zorder=1))
        ax.scatter(xy[:, 0], xy[:, 1], s=f["size"], c=node_colors, zorder=3, linewidths=0.5,
                   edgecolors=BG)
        for i, name in enumerate(names):
            ax.annotate(name.replace("place_", "").replace("situation_", "")
                        .replace("object_", ""), pos[i], fontsize=6.2, color=FG, ha="center",
                        va="center", xytext=(0, -9), textcoords="offset points", alpha=0.7)
        bare(ax)
        ax.margins(0.14)
        ax.autoscale_view()
        near = shocks[(shocks.date >= f["date"] - pd.Timedelta(days=win // 2))
                      & (shocks.date <= f["date"] + pd.Timedelta(days=win // 2))]
        tag = "   ·   ".join(near.label.str[:38]) if len(near) else ""
        ax.set_title(f"{f['date']:%B %Y}    n = {f['n']:,}    mean|ρ| = "
                     f"{f['mean_abs_rho']:.3f}    effective dim = "
                     f"{f['participation_ratio']:.2f}", fontsize=11, color=FG, pad=10)
        ax.text(0.5, -0.02, tag, transform=ax.transAxes, color=VIOLET, fontsize=8.5,
                ha="center", va="top")
        cursor.set_xdata([f["date"], f["date"]])
        return ()

    fig.suptitle("A network breathing: 91-day windows of the English dream feature network",
                 color=FG, fontsize=13, y=0.965)
    fig.text(0.5, 0.015,
             "Edges are shaded by their departure from the whole-period network (warm = stronger "
             "than usual, cool = weaker); node size is the mean absolute within-person deviation. "
             "Violet lines mark the curated shocks. Much of the slow motion is the platform "
             "growing and its contributor mix changing, not dreams changing — see Figure 4.",
             color=MUTED, fontsize=8, ha="center", va="top", wrap=True)

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps, blit=False)
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "network-breathing.gif"
    anim.save(out, writer=PillowWriter(fps=fps), dpi=110, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print(f"[gallery] wrote {out.name} ({len(frames)} frames)", flush=True)
    return series.drop(columns=["corr", "size"])


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    panel = load_state_panel()
    print(f"[gallery] panel: {len(panel.frame):,} English dreams, "
          f"{len(panel.feature_names)} axes", flush=True)

    nightmare = figure_nightmare_constellation(panel, rng)
    print(f"[gallery] fig1 {nightmare}", flush=True)
    drift = figure_arrow_flow()
    print(f"[gallery] fig2 {drift}", flush=True)
    figure_event_breathing(panel, rng)
    figure_time_breathing(panel, rng)
    galaxy = figure_galaxy(rng)
    galaxy.to_csv(TABLES / "galaxy-clusters.csv", index=False)
    series = animate_breathing(panel)
    if series is not None:
        series.to_csv(TABLES / "network-breathing-frames.csv", index=False)
    print("[gallery] done", flush=True)


if __name__ == "__main__":
    main()
