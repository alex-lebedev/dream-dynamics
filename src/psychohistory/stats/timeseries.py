"""Rigorous time-series tools for defensible causality/coupling claims.

- stationarity: ADF (null: unit root) + KPSS (null: stationary) — reported together.
- Toda & Yamamoto (1995) Granger causality: lag-augmented VAR(p+dmax), Wald-test ONLY the
  first p lags of the cause, with HAC covariance. Valid regardless of integration/cointegration
  order and robust to autocorrelation — the recommended test when unit-root status is uncertain.
- HAC (Newey-West) OLS helper and a prewhitened cross-correlation function.

Requires statsmodels.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def stationarity(series) -> dict:
    from statsmodels.tsa.stattools import adfuller, kpss
    s = pd.Series(series).dropna().astype(float)
    out = {"n": int(len(s))}
    try:
        out["adf_p"] = float(adfuller(s, autolag="AIC")[1])            # <0.05 => stationary
    except Exception:
        out["adf_p"] = np.nan
    try:
        out["kpss_p"] = float(kpss(s, regression="c", nlags="auto")[1])  # <0.05 => NON-stationary
    except Exception:
        out["kpss_p"] = np.nan
    # verdict: order of integration guess
    adf_stat = out["adf_p"] < 0.05
    kpss_nonstat = out["kpss_p"] < 0.05
    out["verdict"] = ("I(0) stationary" if adf_stat and not kpss_nonstat
                      else "I(1)/nonstationary" if (not adf_stat and kpss_nonstat)
                      else "ambiguous")
    return out


def select_lag(df2, maxlags=8):
    """VAR AIC lag order on a 2-col frame (levels)."""
    from statsmodels.tsa.api import VAR
    d = df2.dropna()
    try:
        return max(1, int(VAR(d.values).select_order(maxlags=maxlags).aic))
    except Exception:
        return 2


def toda_yamamoto(df, cause, effect, p=None, dmax=1, maxlags=8):
    """Toda-Yamamoto Granger causality  cause -> effect.

    Fit lag-augmented model with (p+dmax) lags of both series (single-equation OLS with HAC),
    then Wald-test that the FIRST p lags of `cause` are jointly zero. Returns chi2, df, p-value.
    """
    import statsmodels.api as sm
    d = df[[effect, cause]].dropna().astype(float)
    if p is None:
        p = select_lag(d, maxlags=maxlags)
    k = p + dmax
    X = pd.DataFrame(index=d.index)
    for i in range(1, k + 1):
        X[f"{effect}_l{i}"] = d[effect].shift(i)
        X[f"{cause}_l{i}"] = d[cause].shift(i)
    X = sm.add_constant(X)
    dat = pd.concat([d[effect].rename("y"), X], axis=1).dropna()
    res = sm.OLS(dat["y"], dat[X.columns]).fit(cov_type="HAC", cov_kwds={"maxlags": max(p, 1)})
    restr = [f"{cause}_l{i}" for i in range(1, p + 1)]        # test only first p lags (TY)
    wald = res.wald_test(np.array([[1.0 if c == r else 0.0 for c in X.columns] for r in restr]),
                         scalar=False)
    return {"cause": cause, "effect": effect, "p": int(p), "dmax": dmax,
            "chi2": float(np.ravel(wald.statistic)[0]), "df": len(restr),
            "pvalue": float(wald.pvalue), "n": int(len(dat))}


def hac_ols(y, X, maxlags=None):
    """OLS with Newey-West (HAC) covariance. X should already include a constant if wanted."""
    import statsmodels.api as sm
    df = pd.concat([pd.Series(y, name="y"), pd.DataFrame(X)], axis=1).dropna()
    L = maxlags if maxlags is not None else int(round(4 * (len(df) / 100.0) ** (2 / 9)))
    return sm.OLS(df["y"], df.drop(columns="y")).fit(cov_type="HAC", cov_kwds={"maxlags": max(L, 1)})


def prewhiten_ccf(x, y, arlags=4, maxlag=8):
    """Cross-correlation after prewhitening x by its AR model and applying the same filter to y
    (removes spurious CCF from shared autocorrelation). Returns dict lag->corr (lag>0: x leads y)."""
    from statsmodels.tsa.ar_model import AutoReg
    x = pd.Series(x).dropna().astype(float).reset_index(drop=True)
    y = pd.Series(y).dropna().astype(float).reset_index(drop=True)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    ar = AutoReg(x, lags=arlags, old_names=False).fit()
    phi = ar.params.values[1:]
    fx = x.copy().values
    fy = y.copy().values
    wx = fx[arlags:] - sum(phi[i - 1] * fx[arlags - i:-i] for i in range(1, arlags + 1))
    wy = fy[arlags:] - sum(phi[i - 1] * fy[arlags - i:-i] for i in range(1, arlags + 1))
    out = {}
    for lag in range(-maxlag, maxlag + 1):
        if lag >= 0:
            a, b = wx[:len(wx) - lag] if lag else wx, wy[lag:]
        else:
            a, b = wx[-lag:], wy[:len(wy) + lag]
        m = min(len(a), len(b))
        if m > 10:
            out[lag] = float(np.corrcoef(a[:m], b[:m])[0, 1])
    return out
