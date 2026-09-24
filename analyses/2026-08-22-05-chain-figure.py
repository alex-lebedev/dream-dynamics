"""Conceptual schematic: from societal environment to population observables.

The manuscript's central ambiguity is that "a dream", "a dream report" and "a population statistic
of dream reports" are three different objects, and that only some of the arrows between the world
and the last of them are tested here. This figure states the chain once, marks which links this
design measures and which it assumes, and lists the observables a population of reports admits.

Emits `60-results/showcase/pub/fig0_chain.{png,pdf}` (manuscript Figure 1). No data are read; the
figure is a diagram, and it is committed because a schematic that is not reproducible from a script
drifts away from the results it summarises.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from psychohistory import config as C

OUT = C.RESULTS / "showcase" / "pub"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1b1b1b"
TESTED = "#1f5fa8"      # a link this study estimates
ASSUMED = "#9a9a9a"     # a link this study does not observe
ACCENT = "#b3452b"

# (label, sublabel) for the vertical chain, top to bottom.
CHAIN = [
    ("Societal environment", "news · institutions · work calendar · crises"),
    ("Individual exposure, physiology, waking affect", "attention · stress · sleep timing"),
    ("Sleep, memory processing, dream generation", "the dream — not observed here"),
    ("Recall and narrative construction", "what survives to morning, in language"),
    ("Dream report", "the unit this study measures"),
    ("Population observables", "aggregates over reports, contributors and days"),
]

# Link status between consecutive chain boxes: (label, colour, style)
LINKS = [
    ("assumed", ASSUMED, "dashed"),
    ("assumed", ASSUMED, "dashed"),
    ("assumed", ASSUMED, "dashed"),
    ("measured\n(§3.3)", TESTED, "solid"),
    ("measured\n(§3.5)", TESTED, "solid"),
]

OBSERVABLES = [
    ("mean", "level of affect", "raw series moves;\nthe movement is\ncontributor turnover,\nplus a slow drift\nwe cannot attribute"),
    ("variance", "volatility of the aggregate", "precursor: null when\ntransitions defined\nindependently"),
    ("trajectory shape", "within-report arrow", "replicated; residual-\ndominated, no\ncommon temporal\ncomponent"),
    ("distribution", "tails, arc mixture, entropy", "no calendar\nstructure beyond\nsampling"),
    ("synchrony", "same-night convergence", "null, bounded"),
]

# Direct-forcing arrows this study estimates from the environment onto the observables.
SIDE = [
    ("social time\n(weekday, holiday, DST)", "reaches the sleeper through sleep timing", "tested — nothing\nsurvives correction\nafter within-person\nand composition control"),
    ("informational events\n(news, markets, attention)", "reaches the sleeper through attention", "tested — null,\nbounded by power"),
]


def box(ax, x, y, w, h, title, sub, fc="#ffffff", ec=INK, lw=1.1, fs=10.5, subfs=8.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title, ha="center", va="center",
            fontsize=fs, color=INK, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center", fontsize=subfs,
                color="#4a4a4a", style="italic", zorder=3)


def arrow(ax, x, y0, y1, colour, style, label):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.5, color=colour, linestyle=style,
                                 shrinkA=0, shrinkB=0, zorder=2))
    ax.text(0.470, (y0 + y1) / 2, label, ha="left", va="center", fontsize=7.8, color=colour,
            linespacing=1.2)


def main() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 9.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bx, bw, bh = 0.055, 0.40, 0.076
    gap = 0.036
    top = 0.965
    ys = [top - i * (bh + gap) for i in range(len(CHAIN))]

    for i, (title, sub) in enumerate(CHAIN):
        fc = "#eef3fa" if i in (4, 5) else "#ffffff"
        ec = TESTED if i in (4, 5) else INK
        box(ax, bx, ys[i] - bh, bw, bh, title, sub, fc=fc, ec=ec, lw=1.6 if i in (4, 5) else 1.1)

    for i, (label, colour, style) in enumerate(LINKS):
        arrow(ax, bx + bw / 2, ys[i] - bh, ys[i + 1] + 0.001, colour, style, label)

    # ------------------------------------------------------------------ observables panel ----
    ox, ow = 0.545, 0.415
    oy_top = 0.830
    ax.text(ox, oy_top + 0.038, "Observables of a population of reports", fontsize=11,
            color=INK, ha="left")
    ax.text(ox, oy_top + 0.014, "what this study finds for each", fontsize=8.6, color="#4a4a4a",
            style="italic", ha="left")
    rh = 0.088
    for j, (name, what, verdict) in enumerate(OBSERVABLES):
        y = oy_top - (j + 1) * rh
        ax.add_patch(FancyBboxPatch((ox, y), ow, rh - 0.014,
                                    boxstyle="round,pad=0.008,rounding_size=0.015",
                                    linewidth=0.9, edgecolor="#b9c6d6", facecolor="#fbfcfe",
                                    zorder=2))
        ax.text(ox + 0.014, y + (rh - 0.014) * 0.66, name, fontsize=9.6, color=TESTED,
                ha="left", va="center", zorder=3)
        ax.text(ox + 0.014, y + (rh - 0.014) * 0.26, what, fontsize=8.0, color="#4a4a4a",
                ha="left", va="center", style="italic", zorder=3)
        ax.text(ox + ow - 0.014, y + (rh - 0.014) * 0.5, verdict, fontsize=7.9, color=ACCENT,
                ha="right", va="center", zorder=3, linespacing=1.25)

    # bracket from the population-observables box to the panel
    ax.add_patch(FancyArrowPatch((bx + bw + 0.004, ys[-1] - bh / 2), (ox - 0.008, oy_top - 2.5 * rh),
                                 arrowstyle="-|>", mutation_scale=13, linewidth=1.5,
                                 color=TESTED, connectionstyle="arc3,rad=0.26",
                                 shrinkA=2, shrinkB=2, zorder=1))

    # ------------------------------------------------------------------- forcing channels ----
    sy = ys[-1] - bh - 0.062
    ax.text(bx, sy + 0.014, "Two channels from the environment, tested separately",
            fontsize=10.4, color=INK, ha="left")
    for k, (name, route, verdict) in enumerate(SIDE):
        y = sy - (k + 1) * 0.088
        ax.add_patch(FancyBboxPatch((bx, y), 0.905, 0.078,
                                    boxstyle="round,pad=0.008,rounding_size=0.015",
                                    linewidth=0.9, edgecolor="#d5c3bd", facecolor="#fdf7f5",
                                    zorder=2))
        ax.text(bx + 0.016, y + 0.039, name, fontsize=9.4, color=INK, ha="left", va="center",
                zorder=3, linespacing=1.3)
        ax.text(bx + 0.245, y + 0.039, route, fontsize=8.6, color="#4a4a4a", style="italic",
                ha="left", va="center", zorder=3)
        ax.text(bx + 0.895, y + 0.039, verdict, fontsize=8.2, color=ACCENT, ha="right",
                va="center", zorder=3, linespacing=1.25)

    ax.text(bx, 0.005,
            "Solid arrows are links this design estimates; dashed arrows are links it assumes and "
            "does not observe.\nA dream, a dream report and a population statistic of dream reports "
            "are three different objects.",
            fontsize=8.4, color="#4a4a4a", ha="left", va="bottom", linespacing=1.45)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig0_chain.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    print(f"[write] {OUT/'fig0_chain.png'}")


if __name__ == "__main__":
    main()
