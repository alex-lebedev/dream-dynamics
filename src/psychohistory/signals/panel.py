"""Assemble a unified WEEKLY societal-signal panel from 10-data/external/signals/.

Defensive: includes whatever files are present (missing signals are skipped). Weeks use the
same W-SUN start-time convention as the dream weekly aggregates so they merge on `week`.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from .. import config as C


def _to_weekly(df, date_col, val_col, how="mean") -> pd.DataFrame:
    d = df[[date_col, val_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[val_col] = pd.to_numeric(d[val_col], errors="coerce")
    d = d.dropna()
    d["week"] = d[date_col].dt.to_period("W-SUN").apply(lambda p: p.start_time)
    return d.groupby("week")[val_col].agg(how).reset_index().rename(columns={val_col: "v"})


def _merge(panel, w, name):
    w = w.rename(columns={"v": name})
    return w if panel is None else panel.merge(w, on="week", how="outer")


def build_weekly_panel(sig_dir=None) -> pd.DataFrame:
    sig_dir = Path(sig_dir or (C.EXTERNAL / "signals"))
    panel = None

    # FRED level series -> weekly mean
    fred = {"vix": "fred_VIXCLS.csv", "dff": "fred_DFF.csv", "dollar": "fred_DTWEXBGS.csv",
            "yield_2s10s": "fred_T10Y2Y.csv", "hy_spread": "fred_BAMLH0A0HYM2.csv"}
    for name, fn in fred.items():
        p = sig_dir / fn
        if p.exists():
            panel = _merge(panel, _to_weekly(pd.read_csv(p), "date", "value"), name)

    # S&P 500 -> weekly return
    p = sig_dir / "fred_SP500.csv"
    if p.exists():
        s = pd.read_csv(p)
        s["date"] = pd.to_datetime(s["date"], errors="coerce")
        s["value"] = pd.to_numeric(s["value"], errors="coerce")
        s = s.dropna().sort_values("date")
        s["week"] = s["date"].dt.to_period("W-SUN").apply(lambda x: x.start_time)
        wk = s.groupby("week")["value"].last().pct_change().reset_index().rename(columns={"value": "v"})
        panel = _merge(panel, wk, "sp500_ret")

    # Geotone global news tone (mean tone across tracked topics)
    p = sig_dir / "geotone_topic_daily.csv"
    if p.exists():
        g = pd.read_csv(p)
        if {"date", "tone"}.issubset(g.columns):
            daily = g.groupby("date")["tone"].mean().reset_index()
            panel = _merge(panel, _to_weekly(daily, "date", "tone"), "news_tone")

    # Geotone per-country tone (US, RU) for cross-cultural coupling
    p = sig_dir / "geotone_country_daily.csv"
    if p.exists():
        c = pd.read_csv(p)
        iso_col = next((col for col in ["iso2", "iso", "country_iso2"] if col in c.columns), None)
        if iso_col and "tone" in c.columns:
            for iso, label in [("US", "tone_us"), ("RU", "tone_ru")]:
                sub = c[c[iso_col] == iso]
                if len(sub):
                    panel = _merge(panel, _to_weekly(sub, "date", "tone"), label)

    # EPU daily (day/month/year -> date)
    p = sig_dir / "epu_daily.csv"
    if p.exists():
        e = pd.read_csv(p)
        if {"day", "month", "year"}.issubset(e.columns):
            e["date"] = pd.to_datetime(dict(year=e.year, month=e.month, day=e.day), errors="coerce")
            vcol = next((col for col in e.columns if "index" in col.lower()), None)
            if vcol:
                panel = _merge(panel, _to_weekly(e, "date", vcol), "epu")

    if panel is None:
        return pd.DataFrame(columns=["week"])
    return panel.sort_values("week").reset_index(drop=True)


def signal_columns(panel) -> list:
    return [c for c in panel.columns if c != "week"]
