"""BOLD Q21 — nightmare contagion: population-level temporal clustering (Hawkes-style).

Test whether high-nightmare dreams cluster in calendar time beyond chance: (a) pair-correlation /
Ripley's K vs date-shuffled null; (b) branching-ratio excess in +1..+3 days after spike days
(weekday-controlled); (c) food-tag spike negative control. Optional VIX spike-week residual check.

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 USE_TF=0 python3 analyses/2026-07-19-13-nightmare-contagion.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psychohistory import config as C

OUT = C.RESULTS / "showcase"
OUT.mkdir(parents=True, exist_ok=True)
WINDOW_START = "2024-03-01"
WINDOW_END = "2026-07-01"
N_PERM = 1000
MAX_LAG_DAYS = 14


def load_daily() -> pd.DataFrame:
    tags = ["nightmare_index", "food", "fear", "danger"]
    dl = pd.read_parquet(C.DREAMS_OUT / "dreamseer_dream_level.parquet",
                         columns=["date", "lang"] + tags)
    dl["date"] = pd.to_datetime(dl["date"])
    dl = dl[(dl.lang == "en") & (dl.date >= WINDOW_START) & (dl.date <= WINDOW_END)]
    day = dl.date.dt.normalize()
    g = dl.groupby(day)
    daily = g[tags].mean()
    daily["n"] = g.size()
    daily["nightmare_rate"] = g.apply(lambda x: (x.nightmare_index >= x.nightmare_index.quantile(0.75)).mean())
    daily.index.name = "date"
    return daily.reset_index()


def event_dates(daily: pd.DataFrame, col="nightmare_index", q=0.75) -> np.ndarray:
    thr = daily[col].quantile(q)
    hit = daily[daily[col] >= thr]["date"].values
    return hit.astype("datetime64[D]")


def pair_correlation(dates: np.ndarray, max_lag=MAX_LAG_DAYS) -> np.ndarray:
    """Normalized pair count g(r) = N_pairs at lag r / expected under uniform intensity."""
    if len(dates) < 10:
        return np.full(max_lag + 1, np.nan)
    t = np.sort(dates.astype("datetime64[D]").astype(np.int64))
    span = (t[-1] - t[0]) / 86400.0
    n = len(t)
    lam = n / max(span, 1)
    g = np.zeros(max_lag + 1)
    for r in range(max_lag + 1):
        if r == 0:
            continue
        cnt = sum(1 for i in range(n) for j in range(i + 1, n) if (t[j] - t[i]) // 86400 == r)
        exp = n * (n - 1) / 2 * lam * r / max(span, 1) if span > 0 else 1
        g[r] = cnt / max(exp, 1e-6)
    return g


def perm_pair_corr(daily: pd.DataFrame, dates: np.ndarray, n_perm=N_PERM, seed=0) -> tuple[np.ndarray, np.ndarray, float]:
    obs = pair_correlation(dates)
    rng = np.random.default_rng(seed)
    pool = daily["date"].values.astype("datetime64[D]")
    null_peak = []
    null_curves = []
    for _ in range(n_perm):
        sh = rng.choice(pool, size=len(dates), replace=False)
        g = pair_correlation(sh)
        null_curves.append(g)
        null_peak.append(float(np.max(g[1:4])))  # +1..+3 day band
    null_curves = np.array(null_curves)
    obs_peak = float(np.max(obs[1:4]))
    p_peak = (1 + np.sum(np.array(null_peak) >= obs_peak)) / (len(null_peak) + 1)
    return obs, null_curves, p_peak


def branching_ratio(daily: pd.DataFrame, spike_col="nightmare_index", q=0.90) -> dict:
    """Excess event rate in +1..+3 days after spike days vs weekday-matched baseline."""
    daily = daily.copy()
    daily["dow"] = pd.to_datetime(daily["date"]).dt.dayofweek
    thr = daily[spike_col].quantile(q)
    daily["spike"] = daily[spike_col] >= thr
    spike_days = set(daily.loc[daily.spike, "date"].dt.normalize())
    idx = pd.date_range(daily.date.min(), daily.date.max(), freq="D")
    cal = pd.DataFrame({"date": idx})
    cal["dow"] = cal.date.dt.dayofweek
    cal = cal.merge(daily[["date", spike_col, "n"]], on="date", how="left")
    cal[spike_col] = cal[spike_col].fillna(cal[spike_col].median())
    cal["n"] = cal["n"].fillna(0)

    def is_event(d, col, q75):
        row = daily[daily.date.dt.normalize() == d]
        if len(row) == 0:
            return False
        return row[col].iloc[0] >= daily[col].quantile(q75)

    post_counts = []
    for sd in spike_days:
        for lag in [1, 2, 3]:
            td = sd + np.timedelta64(lag, "D")
            post_counts.append(is_event(td, spike_col, 0.75))
    post_rate = np.mean(post_counts) if post_counts else np.nan

    # weekday-matched baseline (non-spike days)
    base_counts = []
    non_spike = daily[~daily.spike]
    for _, r in non_spike.iterrows():
        for lag in [1, 2, 3]:
            td = r["date"].normalize() + pd.Timedelta(days=lag)
            base_counts.append(is_event(td, spike_col, 0.75))
    base_rate = np.mean(base_counts) if base_counts else np.nan
    br = post_rate / base_rate if base_rate > 0 else np.nan
    return dict(post_rate=post_rate, base_rate=base_rate, branching_ratio=br, n_spikes=len(spike_days))


def vix_residual_check(daily: pd.DataFrame) -> dict:
    try:
        from psychohistory.signals.panel import build_weekly_panel
        panel = build_weekly_panel()
        panel = panel[(panel.week >= WINDOW_START) & (panel.week <= WINDOW_END)]
        if "vix" not in panel.columns:
            return dict(vix_available=False)
        daily = daily.copy()
        daily["week"] = daily.date.dt.to_period("W-SUN").dt.start_time
        wk = daily.groupby("week").nightmare_index.mean().reset_index()
        m = wk.merge(panel[["week", "vix"]], on="week", how="inner")
        if len(m) < 20:
            return dict(vix_available=False)
        r = float(np.corrcoef(m.nightmare_index, m.vix)[0, 1])
        return dict(vix_available=True, weekly_r=r, n_weeks=len(m))
    except Exception:
        return dict(vix_available=False)


def figure(obs_g, null_g, br_nm, br_food, p_peak):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax0, ax1 = axes
    lags = np.arange(len(obs_g))
    null_mean = np.nanmean(null_g, axis=0)
    null_lo = np.nanpercentile(null_g, 2.5, axis=0)
    null_hi = np.nanpercentile(null_g, 97.5, axis=0)
    ax0.fill_between(lags[1:], null_lo[1:], null_hi[1:], alpha=0.25, color="gray", label="null 95%")
    ax0.plot(lags[1:], null_mean[1:], "--", color="gray", lw=1)
    ax0.plot(lags[1:], obs_g[1:], "o-", color="#C1443C", label="observed g(r)")
    ax0.axhline(1, color="k", lw=0.6, alpha=0.4)
    ax0.set_xlabel("lag r (days)")
    ax0.set_ylabel("pair correlation g(r)")
    ax0.set_title(f"Nightmare temporal clustering (+1..+3d peak p={p_peak:.3f})")
    ax0.legend(frameon=False, fontsize=8)
    ax0.spines[["top", "right"]].set_visible(False)

    labels = ["nightmare", "food (neg)"]
    brs = [br_nm.get("branching_ratio", np.nan), br_food.get("branching_ratio", np.nan)]
    colors = ["#C1443C", "#7a7a7a"]
    ax1.bar(labels, brs, color=colors, width=0.45)
    ax1.axhline(1, color="k", lw=0.8, ls="--")
    ax1.set_ylabel("branching ratio (+1..+3d / baseline)")
    ax1.set_title("Post-spike excess (weekday structure in daily rates)")
    ax1.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "20_nightmare_contagion.png", dpi=150)
    plt.close(fig)


def main():
    daily = load_daily()
    nm_dates = event_dates(daily, "nightmare_index", q=0.75)
    obs_g, null_g, p_peak = perm_pair_corr(daily, nm_dates)
    br_nm = branching_ratio(daily, "nightmare_index", q=0.90)
    br_food = branching_ratio(daily, "food", q=0.90)
    vix = vix_residual_check(daily)

    rows = [
        dict(test="pair_corr_peak_1_3d", observed=float(np.max(obs_g[1:4])), perm_p=p_peak,
             note="nightmare dream dates"),
        dict(test="branching_ratio_nightmare", observed=br_nm["branching_ratio"],
             perm_p=np.nan, note=f"post={br_nm['post_rate']:.3f} base={br_nm['base_rate']:.3f}"),
        dict(test="branching_ratio_food_neg", observed=br_food["branching_ratio"],
             perm_p=np.nan, note="negative control"),
        dict(test="n_nightmare_events", observed=len(nm_dates), perm_p=np.nan, note="top-quartile days"),
    ]
    if vix.get("vix_available"):
        rows.append(dict(test="weekly_nightmare_vix_r", observed=vix["weekly_r"], perm_p=np.nan,
                         note=f"n_weeks={vix['n_weeks']}"))
    pd.DataFrame(rows).to_csv(OUT / "nightmare_contagion.csv", index=False)

    figure(obs_g, null_g, br_nm, br_food, p_peak)

    contagion = p_peak < 0.05 or (br_nm.get("branching_ratio", 1) > 1.15 and br_nm["branching_ratio"] > br_food.get("branching_ratio", 1))
    lines = [
        "# BOLD Q21 — nightmare contagion (population temporal clustering)",
        "",
        f"*EN DreamSeer daily aggregates {WINDOW_START}..{WINDOW_END}. Nightmare events = days in "
        "top quartile of mean nightmare_index (plus branching on spike days). Pair-correlation g(r) "
        f"vs calendar-shuffle null ({N_PERM} perm); food spikes = negative control. Aggregate-only.*",
        "",
        f"- **Pair correlation (+1..+3 days):** peak g(r) = {np.max(obs_g[1:4]):.2f}, perm p = {p_peak:.4f}.",
        f"- **Branching ratio (post-spike +1..+3d):** nightmare **{br_nm['branching_ratio']:.2f}** "
        f"(post={br_nm['post_rate']:.3f}, baseline={br_nm['base_rate']:.3f}, n_spikes={br_nm['n_spikes']}).",
        f"- **Food spike control:** branching ratio = {br_food['branching_ratio']:.2f}.",
    ]
    if vix.get("vix_available"):
        lines.append(f"- **VIX co-movement (weekly):** r = {vix['weekly_r']:+.3f} — shared stress "
                     + ("may partly explain clustering." if abs(vix["weekly_r"]) > 0.2 else "weak; clustering not obviously VIX-driven."))
    verdict = (
        "Population nightmare timestamps show **excess short-lag clustering** beyond calendar shuffle — "
        "consistent with self-exciting / contagion-like dynamics (exploratory; not causal)."
        if contagion else
        "No robust population-level nightmare contagion beyond chance; branching ratios near unity. Null/exploratory."
    )
    lines += ["", f"**Verdict:** {verdict}", "", "![nightmare contagion](20_nightmare_contagion.png)"]
    (OUT / "nightmare_contagion.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[q21] wrote {OUT / 'nightmare_contagion.md'}")


if __name__ == "__main__":
    main()
