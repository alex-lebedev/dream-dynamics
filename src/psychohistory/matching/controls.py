"""Dream-appropriate control-day matching.

The event cohorts ship celestial-matched controls; those are wrong for a dream outcome.
Here we match each event day to k control days that resemble it on the *dream-sampling*
confounds — weekday, calendar month (season), local volume regime — while excluding days
too close to any event. See docs/METHODS.md §2-3.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def eligible_days(daily_all: pd.DataFrame, min_dreams: int = 15) -> pd.DataFrame:
    d = daily_all[daily_all.n_dreams >= min_dreams].copy()
    d["weekday"] = d.date.dt.weekday
    d["month"] = d.date.dt.month
    d["vol_tercile"] = pd.qcut(d.n_dreams.rank(method="first"), 3, labels=[0, 1, 2])
    return d.set_index("date")


def match_controls(daily_all, event_dates, k: int = 5, buffer_days: int = 10,
                   min_dreams: int = 15, seed: int = 0) -> dict:
    """For each event date, choose up to k control dates sharing weekday & month &
    volume tercile, at least `buffer_days` from any event. Returns {event_date: [controls]}.
    """
    rng = np.random.default_rng(seed)
    pool = eligible_days(daily_all, min_dreams)
    ev = pd.to_datetime(pd.Series(sorted(set(event_dates))))
    # exclusion mask: any day within buffer of an event is not a valid control
    excluded = set()
    for e in ev:
        for off in range(-buffer_days, buffer_days + 1):
            excluded.add(e + pd.Timedelta(days=off))
    out = {}
    for e in ev:
        if e not in pool.index:
            continue
        row = pool.loc[e]
        cand = pool[(pool.weekday == row.weekday) & (pool.month == row.month)
                    & (pool.vol_tercile == row.vol_tercile)]
        cand = cand[~cand.index.isin(excluded)]
        idx = list(cand.index)
        if len(idx) > k:
            idx = list(rng.choice(idx, size=k, replace=False))
        out[e] = sorted(idx)
    return out
