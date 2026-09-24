"""Continuous coupling: lead-lag cross-correlation between a weekly dream outcome and a
weekly societal signal, with circular-permutation p-values and BH-FDR across lags
(docs/METHODS.md §3B).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .inference import benjamini_hochberg, circular_shift_pvalue_corr


def _align(dream_weekly, signal_weekly, outcome, signal_col):
    a = dream_weekly[["week", outcome]].rename(columns={outcome: "y"})
    b = signal_weekly[["week", signal_col]].rename(columns={signal_col: "x"})
    m = a.merge(b, on="week", how="inner").dropna().sort_values("week")
    return m


def lagged_crosscorr(dream_weekly, signal_weekly, outcome, signal_col,
                     max_lag=8, n_perm=2000, seed=0) -> pd.DataFrame:
    """corr(y_t, x_{t-lag}). lag>0 => signal leads dreams; lag<0 => dreams lead signal."""
    m = _align(dream_weekly, signal_weekly, outcome, signal_col)
    y = m["y"].to_numpy()
    x = m["x"].to_numpy()
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:            # signal leads dreams: corr(y_t, x_{t-lag})
            yy, xx = y[lag:], x[: len(x) - lag]
        elif lag < 0:          # dreams lead signal
            yy, xx = y[: len(y) + lag], x[-lag:]
        else:
            yy, xx = y, x
        if len(yy) < 12:
            continue
        r, p = circular_shift_pvalue_corr(xx, yy)  # exact all-shifts null, O(n log n)
        rows.append({"lag_weeks": lag, "r": r, "perm_p": p, "n": len(yy)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_fdr"] = benjamini_hochberg(out["perm_p"].to_numpy())
    return out
