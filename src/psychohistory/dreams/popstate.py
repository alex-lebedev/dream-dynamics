"""Trait/state decomposition of a dreaming population.

A dream corpus mixes two things a population-level reading has to keep apart. Some of the
variation between reports is *who wrote them* — the stable dreamer-specific level established by
the author-fingerprint result — and some is *when they were written*. An aggregate mean confounds
the two, because the contributing population turns over: a shift in the daily mean can be a shift
in the population's state or a shift in who is reporting that week, and the mean cannot tell them
apart.

This module implements the decomposition the distinction requires,

    y_it = alpha_i + beta_t + eps_it,

and two estimators of the time component, because they answer different questions.

`loo_deviation` removes each contributor's own level using a LEAVE-ONE-OUT mean, so a report is
compared against the same person's other reports and never against itself. The daily mean of those
deviations is a *local* state: it measures where the population sits relative to the recent
baseline of whoever is present, and is therefore insensitive to adoption drift and to composition
turnover by construction. Its cost is that it high-pass filters — a slow common movement shared by
every contributor is partly absorbed into each contributor's own baseline.

`twoway_fe` estimates alpha and beta jointly by alternating projections, which recovers the global
time path (up to an additive constant) provided the contributor-by-day design is connected. Its
cost is the mirror image: with short contributor spans, slow components of beta lean on the
connectivity of the design rather than on any single person's observations.

Both are reported everywhere, and their agreement is a reported quantity rather than an assumption.

    from psychohistory.dreams.popstate import loo_deviation, twoway_fe, day_effect_test
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- estimators ----
def collapse_person_day(df: pd.DataFrame, value: str, person: str = "userID",
                        day: str = "day") -> pd.DataFrame:
    """One row per (person, day): a contributor filing twice in a night is one observation.

    Necessary before any independence-based null: two reports by the same person on the same day
    are not two draws from the population, and a leave-one-out baseline computed across them
    leaves each one partly in the other's comparator.
    """
    g = df.groupby([person, day], sort=False)[value]
    out = g.mean().rename(value).reset_index()
    out["n_rep"] = g.size().values
    return out


def loo_deviation(df: pd.DataFrame, value: str, person: str = "userID") -> np.ndarray:
    """y_it minus the mean of the SAME contributor's other observations.

    The plug-in person mean includes the observation being centred, which correlates the residuals
    of a contributor with one another by construction and biases any variance statistic computed on
    them. Leaving the observation out costs a factor n_i/(n_i-1) in variance and removes the bias.
    Contributors with a single observation return NaN and drop out of every statistic here.
    """
    y = df[value].to_numpy(float)
    codes, _ = pd.factorize(df[person], sort=False)
    tot = np.bincount(codes, weights=y)
    cnt = np.bincount(codes).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        base = (tot[codes] - y) / (cnt[codes] - 1.0)
    base[cnt[codes] < 2] = np.nan
    return y - base


def twoway_fe(df: pd.DataFrame, value: str, person: str = "userID", day: str = "day",
              iters: int = 500, tol: float = 1e-10):
    """Alternating-projection fit of y_it = alpha_i + beta_t + eps_it on an unbalanced panel.

    Returns (alpha, beta, resid, info). `beta` is a Series indexed by the time key, centred to mean
    zero over observations so the level is carried by alpha; the split between the two is
    identified only up to that constant, which is why every statistic taken from beta below is a
    variance or a contrast rather than a level.
    """
    y = df[value].to_numpy(float)
    pc, pu = pd.factorize(df[person], sort=False)
    tc, tu = pd.factorize(df[day], sort=False)
    npers, ntime = len(pu), len(tu)
    pn = np.bincount(pc, minlength=npers).astype(float)
    tn = np.bincount(tc, minlength=ntime).astype(float)
    alpha = np.zeros(npers)
    beta = np.zeros(ntime)
    for it in range(iters):
        prev = beta.copy()
        alpha = np.bincount(pc, weights=y - beta[tc], minlength=npers) / np.maximum(pn, 1)
        beta = np.bincount(tc, weights=y - alpha[pc], minlength=ntime) / np.maximum(tn, 1)
        m = float(np.average(beta[tc]))          # observation-weighted centring
        beta -= m
        alpha += m
        if np.max(np.abs(beta - prev)) < tol:
            break
    fit = alpha[pc] + beta[tc]
    resid = y - fit
    # `var_resid` is the plug-in SSR/N and understates sigma^2 because I+T-1 effects were fitted;
    # `sigma2_dof` is the degrees-of-freedom-adjusted estimate and is the one the noise corrections
    # must use, since an understated sigma^2 under-corrects both variance components and so inflates
    # both the trait and the state share.
    n_obs = len(y)
    dof = max(n_obs - npers - ntime + 1, 1)
    info = {"iters": int(it + 1), "n_person": npers, "n_time": ntime,
            "converged": bool(np.max(np.abs(beta - prev)) < tol),
            "var_y": float(np.var(y)), "var_alpha": float(np.var(alpha[pc])),
            "var_beta": float(np.var(beta[tc])), "var_resid": float(np.var(resid)),
            "sigma2_dof": float(np.sum(resid ** 2) / dof), "dof": int(dof),
            "cov_alpha_beta": float(np.cov(alpha[pc], beta[tc])[0, 1])}
    return (pd.Series(alpha, index=pu), pd.Series(beta, index=tu),
            pd.Series(resid, index=df.index), info)


# ---------------------------------------------------------------- day effect ----
def _anova(dev: np.ndarray, tcode: np.ndarray, ntime: int):
    """Between-day and within-day mean squares of a deviation vector, plus the day means."""
    n = np.bincount(tcode, minlength=ntime).astype(float)
    s = np.bincount(tcode, weights=dev, minlength=ntime)
    m = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
    grand = dev.mean()
    ssb = float(np.sum(n * (m - grand) ** 2))
    ssw = float(np.sum((dev - m[tcode]) ** 2))
    k = int((n > 0).sum())
    msb = ssb / max(k - 1, 1)
    msw = ssw / max(len(dev) - k, 1)
    # harmonic-style n0 for the unbalanced one-way random-effects variance component
    n0 = (len(dev) - float(np.sum(n[n > 0] ** 2)) / len(dev)) / max(k - 1, 1)
    return {"ms_between": msb, "ms_within": msw, "F": msb / msw if msw > 0 else np.nan,
            "k_days": k, "n_obs": int(len(dev)), "n0": float(n0),
            "sd_day": float(np.sqrt(max((msb - msw) / max(n0, 1e-9), 0.0))),
            "icc_day": float(max((msb - msw), 0.0) / max(msb + (max(n0, 1e-9) - 1) * msw, 1e-12)),
            "day_mean": m, "day_n": n}


def day_effect_test(df: pd.DataFrame, dev_col: str, person: str = "userID", day: str = "day",
                    B: int = 1000, seed: int = 0, adjust: pd.DataFrame | None = None):
    """Is there a common day component in within-person deviations, beyond independent noise?

    The statistic is the one-way between-day mean square of the deviations; the null permutes each
    contributor's deviation values across that contributor's OWN observation dates. That preserves
    exactly what a composition explanation would exploit — how many reports each day carries, which
    contributors are present on it, and every contributor's own distribution of deviations — and
    destroys only the alignment between different contributors on the same date. A day effect that
    survives it is a shared temporal component, not a turnover artifact.

    `adjust`, if given, is a design matrix on the day key: the day means are residualised on it
    inside every surrogate as well as in the observed statistic, so the reported effect is the part
    of the common component that calendar structure and reporting volume do not explain.
    """
    d = df.dropna(subset=[dev_col]).copy()
    dev = d[dev_col].to_numpy(float)
    pc, _ = pd.factorize(d[person], sort=False)
    tc, tu = pd.factorize(d[day], sort=False)
    ntime = len(tu)

    Xadj = None
    if adjust is not None:
        A = adjust.reindex(tu)
        Xadj = np.column_stack([np.ones(len(A))] + [A[c].to_numpy(float) for c in A.columns])
        Xadj = np.nan_to_num(Xadj)

    def stat(v):
        r = _anova(v, tc, ntime)
        if Xadj is None:
            return r["F"], r
        keep = r["day_n"] > 0
        w = r["day_n"][keep]
        m = r["day_mean"][keep]
        Xw = Xadj[keep] * np.sqrt(w)[:, None]
        b, *_ = np.linalg.lstsq(Xw, m * np.sqrt(w), rcond=None)
        e = m - Xadj[keep] @ b
        ssb = float(np.sum(w * (e - np.average(e, weights=w)) ** 2))
        msb = ssb / max(keep.sum() - Xadj.shape[1], 1)
        r = dict(r, ms_between=msb, F=msb / r["ms_within"] if r["ms_within"] > 0 else np.nan,
                 sd_day=float(np.sqrt(max((msb - r["ms_within"]) / max(r["n0"], 1e-9), 0.0))))
        return r["F"], r

    obs, rec = stat(dev)
    rng = np.random.default_rng(seed)
    order = np.argsort(pc, kind="stable")
    blocks = np.split(order, np.flatnonzero(np.diff(pc[order])) + 1)
    blocks = [b for b in blocks if len(b) > 1]
    null = np.empty(B)
    v = dev.copy()
    for j in range(B):
        for b in blocks:
            v[b] = dev[rng.permutation(b)]
        null[j], _ = stat(v)
    p = float((1 + np.sum(null >= obs)) / (B + 1))
    q95 = float(np.percentile(null, 95))
    # A null is worth a bound: the largest common day component that would still have failed to
    # clear the permutation threshold, in the units of the outcome.
    sd_bound = float(np.sqrt(max((q95 - 1.0) * rec["ms_within"] / max(rec["n0"], 1e-9), 0.0)))
    return {"F_obs": float(obs), "p_perm": p, "B": B,
            "null_mean_F": float(np.mean(null)), "null_q95_F": q95,
            "ms_between": rec["ms_between"], "ms_within": rec["ms_within"],
            "sd_day": rec["sd_day"], "sd_day_bound_q95": sd_bound, "icc_day": rec["icc_day"],
            "k_days": rec["k_days"], "n_obs": rec["n_obs"]}


# --------------------------------------------------- structured calendar test ----
def _within_person_blocks(pc: np.ndarray):
    order = np.argsort(pc, kind="stable")
    blocks = np.split(order, np.flatnonzero(np.diff(pc[order])) + 1)
    return [b for b in blocks if len(b) > 1]


def design_test(df: pd.DataFrame, dev_col: str, design: pd.DataFrame, groups: dict,
                contrasts: dict | None = None, person: str = "userID", day: str = "day",
                B: int = 1000, seed: int = 0) -> dict:
    """Low-dimensional calendar structure in within-person deviations, permutation-tested.

    `day_effect_test` asks whether days differ from one another at all, which spends one degree of
    freedom per day and is therefore weak against a smooth or periodic component: a weekly rhythm
    worth a thousandth of the daily variance is invisible to it. This asks the targeted question
    instead — does a *named* calendar structure (weekday, annual harmonic, trend) explain the
    deviations — at a handful of degrees of freedom.

    The null is the same within-contributor shuffle used everywhere in this module: each
    contributor's deviations are permuted across that contributor's own reporting dates. Reporting
    volume, which contributors are present on which date, and every contributor's own distribution
    of deviations are preserved exactly; only the alignment between the calendar and the deviations
    is destroyed. `groups` maps a name to the design columns tested jointly; `contrasts` maps a name
    to a coefficient-weight dict (e.g. Friday minus Monday).
    """
    d = df.dropna(subset=[dev_col])
    base = d[dev_col].to_numpy(float)
    pc, _ = pd.factorize(d[person], sort=False)
    A = np.nan_to_num(design.reindex(pd.DatetimeIndex(d[day])).to_numpy(float))
    X = np.column_stack([np.ones(len(d)), A])
    names = ["const"] + list(design.columns)
    n, p = X.shape

    # The design is fixed across permutations; only the response moves. Pseudo-inverses are
    # therefore formed once and every replicate costs two matrix-vector products per model.
    P_full = np.linalg.pinv(X)
    reduced = {}
    for gname, gcols in groups.items():
        keep = np.array([i for i, nm in enumerate(names) if nm not in set(gcols)])
        k = p - len(keep)
        Xr = X[:, keep]
        reduced[gname] = (Xr, np.linalg.pinv(Xr), max(k, 1))
    sd_y = float(np.std(base))
    cidx = {cname: np.array([w.get(nm, 0.0) for nm in names])
            for cname, w in (contrasts or {}).items()}

    def stats(y):
        b = P_full @ y
        rss_full = float(np.sum((y - X @ b) ** 2))
        out = {}
        for gname, (Xr, Pr, k) in reduced.items():
            rss_red = float(np.sum((y - Xr @ (Pr @ y)) ** 2))
            out[gname] = ((rss_red - rss_full) / k) / (rss_full / max(n - p, 1))
        return out, {cname: float(c @ b) for cname, c in cidx.items()}

    obs_f, obs_c = stats(base)
    rng = np.random.default_rng(seed)
    blocks = _within_person_blocks(pc)
    nf = {g: np.empty(B) for g in groups}
    nc = {c: np.empty(B) for c in cidx}
    v = base.copy()
    for j in range(B):
        for b_ in blocks:
            v[b_] = base[rng.permutation(b_)]
        f_j, c_j = stats(v)
        for g in groups:
            nf[g][j] = f_j[g]
        for c in nc:
            nc[c][j] = c_j[c]

    rows = []
    for g in groups:
        rows.append({"term": g, "kind": "group", "stat": obs_f[g],
                     "null_mean": float(np.mean(nf[g])), "null_q95": float(np.percentile(nf[g], 95)),
                     "p_perm": float((1 + np.sum(nf[g] >= obs_f[g])) / (B + 1)),
                     "effect_sd_units": np.nan})
    for c in nc:
        z = np.abs(nc[c]) >= abs(obs_c[c])
        rows.append({"term": c, "kind": "contrast", "stat": obs_c[c],
                     "null_mean": float(np.mean(nc[c])), "null_q95": float(np.percentile(np.abs(nc[c]), 95)),
                     "p_perm": float((1 + np.sum(z)) / (B + 1)),
                     "effect_sd_units": obs_c[c] / sd_y if sd_y > 0 else np.nan})
    return {"rows": rows, "n_obs": int(n), "B": B, "sd_dev": sd_y}


# ---------------------------------------------------------------- reliability ----
def spearman_brown(r: float) -> float:
    return 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan


def split_half_series(df: pd.DataFrame, value: str, key: str, person: str = "userID",
                      by: str = "report", kind: str = "state", n_splits: int = 200,
                      min_cell: int = 5, seed: int = 0, detrend: bool = False) -> float:
    """Mean split-half correlation of a period-level series, under a stated split unit.

    `by="report"` splits observations at random, which is the scheme the repository's published
    reliability audit uses; `by="user"` splits CONTRIBUTORS, which additionally charges the series
    for the composition noise a population-state reading has to survive. The two are not
    interchangeable and both are reported.

    `kind="raw"` correlates the period mean of y; `kind="state"` correlates the period mean of the
    leave-one-out within-person deviation, recomputed inside each half so the halves stay
    independent.
    """
    rng = np.random.default_rng(seed)
    out = []
    users = df[person].to_numpy()
    uniq = pd.unique(users)
    for _ in range(n_splits):
        if by == "user":
            pick = pd.Series(rng.integers(0, 2, len(uniq)), index=uniq)
            h = pick.reindex(users).to_numpy()
        else:
            h = rng.integers(0, 2, len(df))
        series = {}
        for side in (0, 1):
            sub = df[h == side]
            if kind == "state":
                sub = sub.assign(_d=loo_deviation(sub, value, person)).dropna(subset=["_d"])
                col = "_d"
            else:
                col = value
            g = sub.groupby(key)[col]
            s = g.mean()[g.size() >= min_cell]
            series[side] = s
        a, b = series[0].align(series[1], join="inner")
        if len(a) <= 10:
            continue
        if detrend:
            t = np.arange(len(a), dtype=float)
            X = np.column_stack([np.ones_like(t), (t - t.mean()) / (t.std() + 1e-9)])
            a = pd.Series(a.to_numpy() - X @ np.linalg.lstsq(X, a.to_numpy(), rcond=None)[0],
                          index=a.index)
            b = pd.Series(b.to_numpy() - X @ np.linalg.lstsq(X, b.to_numpy(), rcond=None)[0],
                          index=b.index)
        out.append(float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1]))
    return float(np.mean(out)) if out else np.nan


def calendar_design(days: pd.DatetimeIndex, logn: pd.Series | None = None) -> pd.DataFrame:
    """Weekday indicators, two annual harmonics, a linear trend and (optionally) log volume."""
    idx = pd.DatetimeIndex(days)
    doy = idx.dayofyear.to_numpy(float)
    t = (idx - idx.min()).days.to_numpy(float)
    X = pd.DataFrame(index=idx)
    X["trend"] = (t - t.mean()) / (t.std() + 1e-9)
    for k in (1, 2):
        X[f"s{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        X[f"c{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    dow = idx.dayofweek.to_numpy()
    for k in range(1, 7):
        X[f"dow{k}"] = (dow == k).astype(float)
    if logn is not None:
        X["logn"] = logn.reindex(idx).to_numpy(float)
        X["logn"] = X["logn"] - np.nanmean(X["logn"])
    return X
