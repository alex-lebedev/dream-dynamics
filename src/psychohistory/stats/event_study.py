"""Event-study for dream outcomes (case-crossover + permutation), fully vectorised.

Design (docs/METHODS.md §3A): for each event, compare the mean outcome in a POST window to a
matched PRE baseline; the pooled pre->post difference is tested by a permutation null that draws
pseudo-events from the same eligible day pool (preserving count), so slow adoption trends and
weekday effects cannot manufacture significance.

Speed: the daily outcome is rasterised to a dense integer day-index with cumulative sum/count
arrays, so any window mean is O(1) and the entire permutation batch is a single numpy op.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _prep(daily_lang, outcome, min_dreams):
    d = daily_lang[daily_lang.n_dreams >= min_dreams].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    base = d["date"].min()
    span = int((d["date"].max() - base).days) + 1
    vals = np.full(span, np.nan)
    offs = (d["date"] - base).dt.days.to_numpy()
    vals[offs] = d[outcome].to_numpy()
    present = ~np.isnan(vals)
    v0 = np.where(present, vals, 0.0)
    C = np.concatenate([[0.0], np.cumsum(v0)])                 # prefix sums, len span+1
    N = np.concatenate([[0.0], np.cumsum(present.astype(float))])
    pool = offs.copy()                                          # eligible day offsets
    return C, N, span, base, pool


def _win_mean(C, N, span, offs, lo, hi):
    offs = np.asarray(offs)
    a = np.clip(offs + lo, 0, span)
    b = np.clip(offs + hi + 1, 0, span)
    num = C[b] - C[a]
    den = N[b] - N[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(den > 0, num / den, np.nan)
    return m


def event_study(daily_lang, event_dates, outcome="negativity_mean",
                post=(1, 7), pre=(-21, -4), min_dreams=15, n_perm=20000, seed=0):
    """Return dict: observed pooled pre->post effect, permutation p, per-event effects."""
    C, N, span, base, pool = _prep(daily_lang, outcome, min_dreams)
    pool_set = set(pool.tolist())
    ev_off = np.array(sorted({int((pd.Timestamp(d) - base).days)
                              for d in pd.to_datetime(list(event_dates))} & pool_set), dtype=int)
    if ev_off.size == 0:
        return {"outcome": outcome, "n_events": 0}

    obs_diffs = _win_mean(C, N, span, ev_off, *post) - _win_mean(C, N, span, ev_off, *pre)
    n_events = int(np.isfinite(obs_diffs).sum())
    if n_events == 0:
        return {"outcome": outcome, "n_events": 0}
    obs = float(np.nanmean(obs_diffs))

    rng = np.random.default_rng(seed)
    upool = np.array(sorted(pool_set), dtype=int)
    k = ev_off.size
    # batched pseudo-events (with replacement across the pool; negligible for k << |pool|)
    samp = upool[rng.integers(0, len(upool), size=(n_perm, k))]
    postp = _win_mean(C, N, span, samp.ravel(), *post).reshape(n_perm, k)
    prep = _win_mean(C, N, span, samp.ravel(), *pre).reshape(n_perm, k)
    null = np.nanmean(postp - prep, axis=1)
    null = null[np.isfinite(null)]
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)
    return {
        "outcome": outcome,
        "n_events": n_events,
        "effect_post_minus_pre": obs,
        "null_mean": float(np.mean(null)),
        "null_sd": float(np.std(null)),
        "perm_p": float(p),
        "per_event": obs_diffs,
    }
