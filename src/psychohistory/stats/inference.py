"""Inference primitives suited to autocorrelated, multiple-comparison-heavy time series.

See docs/METHODS.md §4. These are used everywhere so effect sizes come with honest,
resampling-based uncertainty and FDR control rather than naive p-values.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def benjamini_hochberg(pvals) -> np.ndarray:
    """Return BH-adjusted q-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def rolling_robust_z(series, window: int = 28, min_periods: int = 14) -> pd.Series:
    """Robust z of each point vs its *trailing* window (median/MAD), excluding itself.

    Used to flag anomalous-affect days without leaking the day into its own baseline.
    """
    s = pd.Series(np.asarray(series, dtype=float))
    base = s.shift(1)
    med = base.rolling(window, min_periods=min_periods).median()
    mad = base.rolling(window, min_periods=min_periods).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    scale = 1.4826 * mad.replace(0, np.nan)
    return (s - med) / scale


def circular_shift_pvalue_corr(x, y):
    """EXACT two-sided p-value for Pearson r under ALL circular shifts of y, in O(n log n).

    The null of r over every circular shift is the normalised circular cross-correlation of
    the centred series (computed by FFT), so no sampling is needed — faster and more rigorous
    than a Monte-Carlo permutation for the linear-correlation case (docs/METHODS.md §4).
    Returns (observed_r, p_two_sided).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    if denom == 0 or n < 4:
        return 0.0, 1.0
    obs = float((xc * yc).sum() / denom)
    cc = np.fft.irfft(np.fft.rfft(xc) * np.conj(np.fft.rfft(yc)), n=n)  # numerator per shift
    r_all = cc / denom
    p = float(np.sum(np.abs(r_all) >= abs(obs) - 1e-12) / n)  # includes observed shift
    return obs, p


def circular_permutation_pvalue(x, y, stat_fn, n: int = 2000, seed: int = 0):
    """Generic Monte-Carlo circular-shift p-value (kept for non-linear stats). For plain
    Pearson r prefer `circular_shift_pvalue_corr` (exact + fast)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    obs = stat_fn(x, y)
    m = len(y)
    count = 0
    for _ in range(n):
        shift = int(rng.integers(1, m))
        if abs(stat_fn(x, np.roll(y, shift))) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n + 1)


def block_bootstrap_ci(x, stat_fn, block: int = 7, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Moving-block bootstrap CI for a statistic of one autocorrelated series."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    N = len(x)
    nblocks = int(np.ceil(N / block))
    stats = np.empty(n)
    for b in range(n):
        idx = []
        for _ in range(nblocks):
            start = int(rng.integers(0, N))
            idx.extend((start + k) % N for k in range(block))
        stats[b] = stat_fn(x[np.asarray(idx[:N])])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(stats)), float(lo), float(hi)
