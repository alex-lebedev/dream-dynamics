"""CLI: build DreamSeer per-dream features + de-identified daily/weekly aggregates.

Usage:
    python -m psychohistory.dreams.build_features \
        --input 10-data/raw/dreamseer/dreamseer_data.csv \
        --outdir 10-data/processed/dreams

Outputs
    <outdir>/dreamseer_dream_level.parquet   (GITIGNORED: has userID/text)
    <outdir>/dreamseer_daily.csv             (tracked: de-identified aggregate)
    <outdir>/dreamseer_weekly.csv            (tracked)
    60-results/tables/dreamseer_quality_report.md
    60-results/tables/anomalous_affect_days.csv
    60-results/figures/dreamseer_volume_and_affect.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .. import config as C
from ..utils.lang import backend
from ..utils.io import write_datacard
from ..stats.inference import rolling_robust_z
from . import features as F


def _quality_report(raw, clean_df, daily) -> str:
    alld = daily[daily.lang == "all"].sort_values("date")
    lang_counts = clean_df["lang"].value_counts()
    dense = alld[alld.n_dreams >= 15]
    lines = [
        "# DreamSeer — data-quality & profile report",
        "",
        f"- generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}",
        f"- language backend: `{backend()}`",
        f"- raw rows: {len(raw):,} | cleaned dreams: {len(clean_df):,} "
        f"(dropped {clean_df.attrs.get('dropped', 0):,}: dupes/bad/pre-{C.APP_START[:4]} dates)",
        f"- date span: {clean_df['date'].min():%Y-%m-%d} \u2192 {clean_df['date'].max():%Y-%m-%d}",
        f"- unique users: {clean_df['userID'].nunique():,}",
        "",
        "## Language mix (share of cleaned dreams)",
    ]
    for lang, n in lang_counts.items():
        lines.append(f"- {lang}: {n:,} ({100*n/len(clean_df):.1f}%)")
    lines += [
        "",
        "## Daily coverage",
        f"- active days: {alld['date'].nunique():,}",
        f"- days with \u226515 dreams: {len(dense):,} "
        f"({dense['date'].min():%Y-%m-%d} \u2192 {dense['date'].max():%Y-%m-%d})",
        f"- median dreams/day (all): {alld['n_dreams'].median():.0f}; last-90d mean: "
        f"{alld.tail(90)['n_dreams'].mean():.0f}",
        "",
        "## Mean affect (cleaned, per-dream)",
    ]
    for c in ["fear", "anger", "sadness", "joy", "trust", "negativity", "nightmare_index"]:
        lines.append(f"- {c}: {clean_df[c].mean():.3f}")
    lines += [
        "",
        "> Aggregates in `dreamseer_daily.csv` / `dreamseer_weekly.csv` are de-identified ",
        "> (no userID, no text) and safe to track in git. The per-dream parquet is gitignored.",
    ]
    return "\n".join(lines)


def _anomalous_days(daily, min_dreams=20, min_users=5, z_thresh=3.0):
    """Days whose affect is a robust-z outlier, for the quality report.

    ``min_dreams`` and ``min_users`` are DISCLOSURE THRESHOLDS, not statistical ones: this table is
    tracked and released, so docs/ETHICS.md §4 forbids emitting a cell computed from fewer than 20
    dreams or fewer than 5 distinct users. ``min_dreams`` was 15 until 2026-08-21, which put 33
    released cells below the rule the manuscript asserts the release meets. Do not lower either.

    ``n_users`` is carried into the output for a reason worth stating: without it the second half of
    the rule is unverifiable downstream, since a 20-report cell can be drawn from two people filing
    ten dreams each. Emitting the denominator is what lets `scripts/check_release_cells.py` check
    the whole rule rather than half of it — and a count over the population is not itself disclosive.
    """
    cols = ["date", "lang", "feature", "value", "z", "n_dreams", "n_users"]
    rows = []
    for lang in daily.lang.unique():
        sub = daily[daily.lang == lang].sort_values("date").reset_index(drop=True)
        for feat in ["negativity_mean", "nightmare_index_mean", "fear_mean", "anger_mean", "sadness_mean"]:
            z = rolling_robust_z(sub[feat])
            keep = (z.abs() >= z_thresh) & (sub.n_dreams >= min_dreams)
            if "n_users" in sub.columns:
                keep &= sub.n_users >= min_users
            hit = sub[keep].copy()
            hit["feature"] = feat
            hit["z"] = z[hit.index].round(2)
            hit = hit.rename(columns={feat: "value"})
            rows.append(hit.reindex(columns=cols))
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["lang", "date"]).reset_index(drop=True)


def _figure(daily, path):
    alld = daily[daily.lang == "all"].sort_values("date")
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].fill_between(alld.date.to_numpy(), alld.n_dreams.to_numpy(), color="#4C72B0", alpha=0.6)
    ax[0].set_ylabel("dreams / day")
    ax[0].set_title("DreamSeer volume and negative affect (7-day rolling)")
    for lang, color in [("all", "black"), ("en", "#C44E52"), ("ru", "#55A868")]:
        s = daily[daily.lang == lang].sort_values("date")
        if len(s) > 7:
            roll = s.negativity_mean.rolling(7, min_periods=3).mean()
            ax[1].plot(s.date.to_numpy(), roll.to_numpy(), label=lang, color=color, lw=1.3)
    ax[1].axhline(0, color="grey", lw=0.6, ls="--")
    ax[1].set_ylabel("negativity (neg\u2212pos)")
    ax[1].legend(title="language", loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(C.DREAMSEER_RAW))
    ap.add_argument("--outdir", default=str(C.DREAMS_OUT))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    C.TABLES.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.input}")
    raw = F.load_dreamseer(args.input)
    clean_df = F.clean(raw)
    clean_df = F.add_features(clean_df)
    print(f"[clean] {len(clean_df):,} dreams | langs: {dict(clean_df['lang'].value_counts())}")

    daily = F.aggregate_daily(clean_df)
    weekly = F.aggregate_weekly(daily)

    # per-dream (SENSITIVE -> parquet, gitignored)
    dl_cols = ["documentID", "userID", "date", "lang", "textlen", "words"] + F.FEATURES + ["X", "Y", "Z"]
    clean_df[dl_cols].to_parquet(outdir / "dreamseer_dream_level.parquet", index=False)

    # aggregates (de-identified -> csv, tracked)
    daily.to_csv(outdir / "dreamseer_daily.csv", index=False)
    weekly.to_csv(outdir / "dreamseer_weekly.csv", index=False)

    (C.TABLES / "dreamseer_quality_report.md").write_text(_quality_report(raw, clean_df, daily))
    anom = _anomalous_days(daily)
    anom.to_csv(C.TABLES / "anomalous_affect_days.csv", index=False)
    _figure(daily, C.FIGURES / "dreamseer_volume_and_affect.png")

    write_datacard(
        C.MANIFESTS, "dreamseer_processed",
        source="DreamSeer app export (private)", license="private",
        sensitivity="S1 (aggregates) / S3 (dream-level parquet)",
        local_path=str(outdir / "dreamseer_daily.csv"),
        notes="Daily/weekly de-identified aggregates + gitignored per-dream parquet.",
        extra={"n_dreams": int(len(clean_df)), "n_users": int(clean_df["userID"].nunique()),
               "lang_backend": backend()},
    )

    print(f"[write] daily rows={len(daily):,} weekly rows={len(weekly):,} anomalous-day rows={len(anom):,}")
    print(f"[done] tables -> {C.TABLES} | figure -> {C.FIGURES}")


if __name__ == "__main__":
    main()
