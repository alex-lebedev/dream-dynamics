"""Early-warning-signal estimators for the aggregate dream stream.

These are the estimators behind the nightmare-variance precursor of the manuscript's
Section 3.5. They were originally written inline in
``analyses/2026-07-19-19-collective-dynamics.py`` and are promoted here verbatim in
behaviour so that the surrogate-hardening analysis can reuse the *identical* pipeline
rather than a re-implementation of it. ``analyses/2026-08-21-04-precursor-hardening.py``
asserts that this module reproduces the published observed statistic, episode count and
day count, so the equivalence is checked rather than assumed.

Two families of surrogate live here:

*Spectrum-preserving* (``ar1_surrogate``, ``phase_surrogate``) reproduce linear serial
dependence. A Gaussian AR(1) has constant conditional variance, and Fourier phase
randomization preserves the power spectrum while Gaussianizing the amplitude
distribution — that construction is designed to *destroy* nonlinear structure, of which
volatility clustering is the standard example. These nulls therefore bound linear serial
dependence and not conditional heteroskedasticity.

*Volatility-preserving* (``garch_surrogate_factory``, ``stationary_bootstrap_factory``,
``iaaft_surrogate``) reproduce dependence in the magnitudes as well. Because rolling
variance is exactly the statistic that conditional heteroskedasticity inflates, these are
the nulls a variance precursor has to clear. ``volatility_diagnostic`` reports what any
given family actually preserved, so the claim is evidenced rather than asserted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "daily_residual", "covariate_residual", "ews_indicators", "runup_tau", "dedupe_anchors",
    "pipeline_taus", "episode_anchors",
    "ar1_surrogate", "phase_surrogate", "iaaft_surrogate",
    "garch_surrogate_factory", "stationary_bootstrap_factory",
    "volatility_diagnostic",
]


# --------------------------------------------------------------- residual ----
def daily_residual(meta, col, lang="en", min_n=20):
    """Deseasonalized, adoption-detrended, volume/standard-error-standardized daily series.

    Returns the gap-interpolated residual and the mask of genuinely observed days.
    """
    sub = meta[meta.lang == lang]
    g = sub.groupby(sub.date.dt.normalize())[col]
    d = pd.DataFrame({"m": g.mean(), "sd": g.std(), "n": g.size()})
    d = d[d.n >= min_n]
    idx = d.index
    t = (idx - idx.min()).days.values.astype(float)
    tz = (t - t.mean()) / (t.std() + 1e-9)
    logn = np.log(d.n.values)
    dow = pd.get_dummies(idx.dayofweek).values[:, 1:]
    doy = idx.dayofyear.values.astype(float)
    harm = np.column_stack([f(2 * np.pi * k * doy / 365.25)
                            for k in (1, 2) for f in (np.sin, np.cos)])
    X = np.column_stack([np.ones_like(tz), tz, logn, dow, harm])
    w = np.sqrt(d.n.values)
    beta = np.linalg.lstsq(X * w[:, None], d.m.values * w, rcond=None)[0]
    se = d.sd.values / np.sqrt(d.n.values)
    resid = (d.m.values - X @ beta) / (se + 1e-9)
    full = pd.date_range(idx.min(), idx.max(), freq="D")
    s = pd.Series(resid, index=idx).reindex(full)
    return s.interpolate(limit_direction="both").values, s.notna().values


def covariate_residual(values, n, index):
    """Apply ``daily_residual``'s covariate adjustment to an arbitrary daily statistic.

    Exists so that a *derived* daily quantity — a between-contributor variance, a report count —
    can be put on the same footing as the aggregate mean before its run-up is compared with the
    aggregate's. The design matrix, the sqrt(n) weighting and the log-volume term are identical to
    ``daily_residual``; what necessarily differs is the final division, since a variance component
    has no per-day standard error of the same form. We standardize by the residual's own standard
    deviation instead, which fixes the scale without importing a distributional assumption.

    Comparing a run-up computed on a rolling mean of a raw, undetrended series against one computed
    on a rolling variance of a detrended, volume-standardized series — as an earlier version of the
    decomposition did — differs in the input, the preprocessing and the summarizing operator at
    once, so the two numbers share no scale and neither bounds the other.
    """
    v = np.asarray(values, float)
    n = np.asarray(n, float)
    t = (index - index.min()).days.values.astype(float)
    tz = (t - t.mean()) / (t.std() + 1e-9)
    dow = pd.get_dummies(index.dayofweek).values[:, 1:]
    doy = index.dayofyear.values.astype(float)
    harm = np.column_stack([f(2 * np.pi * k * doy / 365.25)
                            for k in (1, 2) for f in (np.sin, np.cos)])
    X = np.column_stack([np.ones_like(tz), tz, np.log(n), dow, harm])
    w = np.sqrt(n)
    ok = np.isfinite(v)
    beta = np.linalg.lstsq((X[ok] * w[ok, None]), v[ok] * w[ok], rcond=None)[0]
    resid = v - X @ beta
    resid = resid / (np.nanstd(resid) + 1e-9)
    full = pd.date_range(index.min(), index.max(), freq="D")
    s = pd.Series(resid, index=index).reindex(full)
    return s.interpolate(limit_direction="both").values


# ------------------------------------------------------------- indicators ----
def ews_indicators(x, win=28):
    """Rolling variance and rolling lag-1 autocorrelation."""
    var = pd.Series(x).rolling(win, min_periods=win // 2).var().values

    def _ar1(a):
        a = a[np.isfinite(a)]
        return np.corrcoef(a[:-1], a[1:])[0, 1] if len(a) > 5 else np.nan

    ac = pd.Series(x).rolling(win, min_periods=win // 2).apply(_ar1, raw=True).values
    return var, ac


def runup_tau(pos, arr, runup):
    """Kendall's tau between position and indicator value over the run-up window."""
    seg = arr[max(0, pos - runup):pos]
    seg = seg[np.isfinite(seg)]
    if len(seg) < 10:
        return np.nan
    return stats.kendalltau(np.arange(len(seg)), seg).correlation


def dedupe_anchors(pos, gap):
    """Retain an anchor only if it lies at least `gap` days after the last retained one."""
    keep, last = [], -10 ** 9
    for p in np.sort(pos):
        if p - last >= gap:
            keep.append(p)
            last = p
    return np.array(keep)


def episode_anchors(x, runup=42):
    """Top-decile forward-shift excursions of the series itself, deduplicated."""
    fwd = (pd.Series(x).shift(-14).rolling(14).mean() - pd.Series(x).rolling(7).mean()).values
    fin = np.isfinite(fwd)
    return dedupe_anchors(np.where(fin & (fwd >= np.nanquantile(fwd, 0.90)))[0], runup)


def pipeline_taus(x, win=28, runup=42):
    """The full detrend-free EWS -> endogenous-anchor -> mean run-up tau pipeline."""
    var, ac = ews_indicators(x, win)
    anc = episode_anchors(x, runup)
    return ({k: np.nanmean([runup_tau(a, arr, runup) for a in anc])
             for k, arr in [("variance", var), ("ar1", ac)]}, len(anc))


# --------------------------------------------------- spectrum-preserving nulls ----
def ar1_surrogate(x, rng):
    x = np.asarray(x, float) - np.mean(x)
    phi = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    e = rng.standard_normal(len(x)) * np.std(x) * np.sqrt(max(1 - phi ** 2, 1e-6))
    y = np.empty(len(x))
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = phi * y[t - 1] + e[t]
    return y


def phase_surrogate(x, rng):
    x = np.asarray(x, float)
    X = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, len(X))
    ph[0] = 0.0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(x)) + x.mean()


# ------------------------------------------------- volatility-preserving nulls ----
def garch_surrogate_factory(x, grid=161):
    """AR(1) mean equation with GARCH(1,1) innovations, fitted by variance-targeted QMLE.

    The unconditional variance is targeted at the sample innovation variance, so
    ``omega = v * (1 - alpha - beta)`` and the Gaussian quasi-likelihood is maximized over
    a deterministic two-dimensional (alpha, beta) grid. This is a standard estimator and
    it keeps the fit reproducible without an optimizer dependency.

    Returns a generator with the same signature as the spectrum-preserving surrogates,
    together with the fitted parameters. Preserves linear serial dependence *and*
    conditional heteroskedasticity.
    """
    x = np.asarray(x, float)
    mu = float(x.mean())
    xc = x - mu
    phi = float(np.corrcoef(xc[:-1], xc[1:])[0, 1])
    e = xc[1:] - phi * xc[:-1]
    v = float(np.var(e))

    def _nll(alpha, beta):
        omega = v * (1.0 - alpha - beta)
        s2 = np.empty(len(e))
        s2[0] = v
        for t in range(1, len(e)):
            s2[t] = omega + alpha * e[t - 1] ** 2 + beta * s2[t - 1]
        s2 = np.maximum(s2, 1e-12)
        return 0.5 * float(np.sum(np.log(s2) + e ** 2 / s2))

    alphas = np.linspace(0.0, 0.60, grid)
    betas = np.linspace(0.0, 0.98, grid)
    best, best_f = (0.0, 0.0), np.inf
    for a in alphas:
        for b in betas:
            if a + b >= 0.999:
                continue
            f = _nll(float(a), float(b))
            if f < best_f:
                best, best_f = (float(a), float(b)), f
    alpha, beta = best
    omega = v * (1.0 - alpha - beta)
    params = {"phi": phi, "omega": omega, "alpha": alpha, "beta": beta,
              "alpha_plus_beta": alpha + beta, "innovation_var": v, "neg_loglik": best_f}

    def gen(_x, rng):
        n = len(x)
        s2 = np.empty(n)
        ee = np.empty(n)
        s2[0] = v
        ee[0] = rng.standard_normal() * np.sqrt(max(v, 1e-12))
        for t in range(1, n):
            s2[t] = omega + alpha * ee[t - 1] ** 2 + beta * s2[t - 1]
            ee[t] = rng.standard_normal() * np.sqrt(max(s2[t], 1e-12))
        y = np.empty(n)
        y[0] = xc[0]
        for t in range(1, n):
            y[t] = phi * y[t - 1] + ee[t]
        return y + mu

    return gen, params


def stationary_bootstrap_factory(mean_block):
    """Politis & Romano stationary bootstrap: geometric block lengths, wrapped."""
    p = 1.0 / float(mean_block)

    def gen(x, rng):
        x = np.asarray(x, float)
        n = len(x)
        out = np.empty(n)
        i = int(rng.integers(n))
        for t in range(n):
            out[t] = x[i]
            i = int(rng.integers(n)) if rng.random() < p else (i + 1) % n
        return out

    return gen


def iaaft_surrogate(x, rng, n_iter=60):
    """Iterated amplitude-adjusted Fourier transform: preserves spectrum and amplitudes."""
    x = np.asarray(x, float)
    n = len(x)
    target_amp = np.abs(np.fft.rfft(x))
    sorted_x = np.sort(x)
    y = rng.permutation(x)
    for _ in range(n_iter):
        Y = np.fft.rfft(y)
        y = np.fft.irfft(target_amp * np.exp(1j * np.angle(Y)), n=n)
        y = sorted_x[np.argsort(np.argsort(y))]
    return y


def volatility_diagnostic(x):
    """What a surrogate family preserved: dependence in the raw series and the magnitudes."""
    x = np.asarray(x, float)
    a = np.abs(x - x.mean())
    s = (x - x.mean()) ** 2
    return {"raw_ar1": float(np.corrcoef(x[:-1], x[1:])[0, 1]),
            "abs_ar1": float(np.corrcoef(a[:-1], a[1:])[0, 1]),
            "sq_ar1": float(np.corrcoef(s[:-1], s[1:])[0, 1])}
