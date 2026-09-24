"""Change points and regimes, defined without reference to the precursor that is tested against them.

The manuscript's early-warning result has a circularity that its own text names but does not repair:
the episodes are top-decile forward excursions *of the same series* whose rolling variance supplies
the precursor, so the finding is a statement about one signal's internal dynamics under a selection
rule tuned to that signal. A claim about *state transitions* requires the states to be defined first,
by a criterion that does not mention variance, and only then asked what precedes them.

This module supplies that definition. `binary_segmentation` finds mean shifts by recursive CUSUM
with a minimum segment length and a Bayesian-information-criterion stopping rule; `regime_stats`
summarizes the resulting partition; `precursor_at` evaluates an arbitrary indicator over the window
preceding each detected change point. None of the three looks at variance.

Change-point algorithms find change points in noise, so every statistic here is meaningful only
against surrogates passed through the identical pipeline — the same segmentation, the same penalty,
the same run-up rule — which is what the calling analysis does with the surrogate families in
`psychohistory.stats.ews`.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["binary_segmentation", "regime_stats", "precursor_at"]


def _best_split(y: np.ndarray, lo: int, hi: int, min_seg: int):
    """Location and cost reduction of the best single mean shift inside y[lo:hi]."""
    n = hi - lo
    if n < 2 * min_seg:
        return None, 0.0
    seg = y[lo:hi]
    c = np.cumsum(seg)
    tot = c[-1]
    k = np.arange(min_seg, n - min_seg + 1)
    left, right = c[k - 1], tot - c[k - 1]
    # reduction in residual sum of squares from splitting at k
    gain = left ** 2 / k + right ** 2 / (n - k) - tot ** 2 / n
    j = int(np.argmax(gain))
    return lo + int(k[j]), float(gain[j])


def binary_segmentation(y, min_seg: int = 4, max_k: int = 12, penalty: float | None = None):
    """Recursive binary segmentation for mean shifts, stopped by a BIC-style penalty.

    `penalty` defaults to `sigma^2 * log(n)`, the usual Schwarz cost of one extra parameter, with
    sigma estimated robustly from first differences so that a series containing genuine shifts does
    not inflate its own threshold. Returns the sorted interior change-point indices.
    """
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 4 * min_seg:
        return np.array([], dtype=int)
    if penalty is None:
        sigma2 = float(np.median(np.abs(np.diff(y))) / 0.6745) ** 2 / 2.0
        penalty = sigma2 * np.log(n)
    cps = []
    segments = [(0, n)]
    while segments and len(cps) < max_k:
        best = (None, 0.0, None)
        for (lo, hi) in segments:
            loc, gain = _best_split(y, lo, hi, min_seg)
            if loc is not None and gain > best[1]:
                best = (loc, gain, (lo, hi))
        loc, gain, seg = best
        if loc is None or gain <= penalty:
            break
        cps.append(loc)
        lo, hi = seg
        segments.remove(seg)
        segments += [(lo, loc), (loc, hi)]
    return np.array(sorted(cps), dtype=int)


def regime_stats(y, cps) -> dict:
    """Partition summary: how many regimes, how separated, how long they last."""
    y = np.asarray(y, float)
    n = len(y)
    bounds = np.concatenate([[0], np.asarray(cps, int), [n]])
    means = np.array([y[bounds[i]:bounds[i + 1]].mean() for i in range(len(bounds) - 1)])
    sizes = np.diff(bounds).astype(float)
    grand = y.mean()
    ssb = float(np.sum(sizes * (means - grand) ** 2))
    sst = float(np.sum((y - grand) ** 2))
    shifts = np.abs(np.diff(means)) if len(means) > 1 else np.array([0.0])
    return {"k_changepoints": int(len(cps)), "n": int(n),
            "between_share": ssb / sst if sst > 0 else np.nan,
            "mean_abs_shift": float(shifts.mean()),
            "max_abs_shift": float(shifts.max()),
            "median_dwell": float(np.median(sizes)), "min_dwell": float(sizes.min())}


def precursor_at(cps, indicator, runup: int, min_pts: int = 6) -> float:
    """Mean Kendall tau of an indicator over the `runup` observations before each change point."""
    taus = []
    arr = np.asarray(indicator, float)
    for p in np.asarray(cps, int):
        seg = arr[max(0, p - runup):p]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= min_pts:
            taus.append(stats.kendalltau(np.arange(len(seg)), seg).correlation)
    return float(np.nanmean(taus)) if taus else np.nan
