"""Publication-grade matplotlib style for `psychohistory` figures.

One import to make every figure consistent, colorblind-safe, and print-ready:

    from psychohistory.utils.plotstyle import apply_style, PALETTE, panel_label, save_fig
    apply_style()
    ...
    save_fig(fig, OUT / "44_physics_of_dreams")   # writes .png (400 dpi) + .pdf (vector)

Design choices for a high-tier venue:
- Okabe-Ito colorblind-safe palette.
- Clean spines (top/right off), light grid, readable fonts sized for a 1-2 column figure.
- Vector PDF + 400-dpi PNG on every save; tight bounding box.
- `panel_label` places bold (a)/(b)/(c) labels in figure/axes coords.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito (colorblind-safe). Keys are semantic-ish so callers can pick stable colors.
PALETTE = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
    "grey":   "#8C8C8C",
}
CYCLE = [PALETTE[k] for k in ("blue", "orange", "green", "red", "purple", "sky", "yellow", "black")]


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        # DejaVu Sans is matplotlib's bundled single-file TTF -> embeds cleanly in vector PDF.
        # (macOS Helvetica/Arial are .ttc collections that break fontTools PDF subsetting.)
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "pdf.fonttype": 42,   # editable text in vector output
        "ps.fonttype": 42,
    })


def panel_label(ax, letter: str, x: float = -0.12, y: float = 1.05, **kw):
    """Bold (a)/(b)/(c) panel label in axes coordinates."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="bottom", ha="right", **kw)


def save_fig(fig, path_noext, formats=("png", "pdf")):
    """Save a figure to multiple formats (vector + raster). `path_noext` has no extension."""
    p = Path(path_noext)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in formats:
        f = p.with_suffix("." + ext)
        fig.savefig(f)
        out.append(str(f))
    plt.close(fig)
    return out
