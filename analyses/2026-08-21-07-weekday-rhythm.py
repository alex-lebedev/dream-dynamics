"""Re-estimate the Friday-minus-Monday rhythm with inference matched to the design.

Why re-estimate. The published figure (beta=+0.021, p=.0086, permutation p=.0055) comes from
`2026-07-18-11-validation-confounds.py`, whose OLS clusters standard errors on *user* and whose
permutation shuffles day-of-week labels across individual reports. Both are wrong for this contrast,
in the same direction:

  * Weekday is a deterministic function of the calendar date, so every report filed on a given
    Friday shares that Friday's shocks. Clustering on user leaves that dependence in the residuals,
    and the effective sample size is on the order of the ~120 Fridays and ~120 Mondays in the window
    rather than the ~30,000 reports. Standard errors are therefore too small.
  * The permutation reshuffles weekday labels across reports globally, which destroys date-level
    clustering entirely rather than preserving it under the null, and it permutes a *raw* difference
    of means while being presented as corroborating the covariate-adjusted beta.

This matters beyond the decimal places, because Section 3.5 reconciles the significant dream-level
regression against a null aggregate periodogram by attributing the gap entirely to the periodogram's
low power. Anticonservative dream-level inference predicts the same pattern. Both explanations have
to be on the table, and only a correctly clustered estimate can tell them apart.

What is estimated here:

  1. Dream-level OLS with the published specification, reported under three variance estimators —
     user-clustered (as published), date-clustered, and two-way user+date — so the cost of the
     original choice is visible rather than asserted.
  2. A date-block permutation: whole dates keep their reports and their weekday label is permuted
     across dates, so date-level clustering is preserved under the null. The permuted statistic is
     the covariate-adjusted contrast, not a raw mean difference.
  3. A day-level regression on daily means, weighted by sqrt(n), which is the aggregate-scale test
     the periodogram is closest to and which cannot be inflated by within-day dependence.
  4. Language-stratified estimates (EN, RU) and a pooled estimate, since METHODS forbids leading
     with a pooled number, plus the composition check that the pooled model implicitly assumes: is
     the Russian share of reports flat across weekdays? The RU-EN sentiment gap is several times the
     weekday effect, so weekday variation in language mix would load straight onto the contrast.

    PYTHONPATH=src PYTHONNOUSERSITE=1 python3 analyses/2026-08-21-07-weekday-rhythm.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from psychohistory import config as CFG

TABLES = CFG.RESULTS / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

N_PERM = 2000
SEED = 0
COVARS = ["textlen_z", "s1", "c1", "s2", "c2"]


def load() -> pd.DataFrame:
    """Identical construction to the published script, so differences are inferential only."""
    lv = pd.read_parquet(CFG.DREAMS_OUT / "dreamseer_dream_level.parquet")
    sc = pd.read_parquet(CFG.DREAMS_OUT / "dreamseer_sentiment.parquet")[["documentID", "sentiment"]]
    d = lv.merge(sc, on="documentID", how="inner")
    d["date"] = pd.to_datetime(d["date"])
    d = d[d.date >= "2024-03-01"].copy()
    d["dow"] = d.date.dt.weekday
    doy = d.date.dt.dayofyear
    d["s1"], d["c1"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)
    d["s2"], d["c2"] = np.sin(4 * np.pi * doy / 365), np.cos(4 * np.pi * doy / 365)
    d["textlen_z"] = (d.textlen - d.textlen.mean()) / d.textlen.std()
    d["year"] = d.date.dt.year
    d["ucode"] = d.userID.astype("category").cat.codes
    d["dcode"] = d.date.dt.normalize().astype("int64")
    return d


def _fri_mon_design(d: pd.DataFrame):
    """Restrict to Friday and Monday and build the design for a single contrast.

    Estimating the two-day contrast directly rather than reading one level off a six-dummy weekday
    model keeps the reported quantity and the permuted quantity the same object.
    """
    sub = d[d.dow.isin([0, 4])].copy()
    sub["fri"] = (sub.dow == 4).astype(float)
    X = np.column_stack([np.ones(len(sub)), sub["fri"].values,
                         *(sub[c].values for c in COVARS)])
    return sub, X, sub["sentiment"].values


def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    return beta, resid, XtX_inv


def _cluster_se(X, resid, XtX_inv, groups) -> float:
    """Cluster-robust SE for the contrast coefficient (column 1)."""
    meat = np.zeros((X.shape[1], X.shape[1]))
    order = np.argsort(groups)
    Xs, rs, gs = X[order], resid[order], np.asarray(groups)[order]
    bounds = np.flatnonzero(np.diff(gs)) + 1
    for blk in np.split(np.arange(len(gs)), bounds):
        u = Xs[blk].T @ rs[blk]
        meat += np.outer(u, u)
    V = XtX_inv @ meat @ XtX_inv
    g = len(bounds) + 1
    n, k = X.shape
    adj = (g / (g - 1)) * ((n - 1) / (n - k))
    return float(np.sqrt(max(V[1, 1] * adj, 0)))


def _twoway_se(X, resid, XtX_inv, g1, g2) -> float:
    """Cameron-Gelbach-Miller two-way: V(g1) + V(g2) - V(intersection)."""
    def V(groups):
        meat = np.zeros((X.shape[1], X.shape[1]))
        order = np.argsort(groups)
        Xs, rs, gs = X[order], resid[order], np.asarray(groups)[order]
        bounds = np.flatnonzero(np.diff(gs)) + 1
        for blk in np.split(np.arange(len(gs)), bounds):
            u = Xs[blk].T @ rs[blk]
            meat += np.outer(u, u)
        return (XtX_inv @ meat @ XtX_inv)[1, 1]
    pairs = np.char.add(np.char.add(np.asarray(g1).astype(str), "|"),
                        np.asarray(g2).astype(str))
    _, inter = np.unique(pairs, return_inverse=True)
    v = V(g1) + V(g2) - V(inter)
    return float(np.sqrt(max(v, 0)))


def dream_level(d: pd.DataFrame, label: str) -> dict:
    from scipy import stats as st
    sub, X, y = _fri_mon_design(d)
    beta, resid, XtX_inv = _ols(X, y)
    b = float(beta[1])
    se_u = _cluster_se(X, resid, XtX_inv, sub["ucode"].values)
    se_d = _cluster_se(X, resid, XtX_inv, sub["dcode"].values)
    se_2 = _twoway_se(X, resid, XtX_inv, sub["ucode"].values, sub["dcode"].values)
    out = {"stratum": label, "n_reports": len(sub),
           "n_dates": int(sub["dcode"].nunique()), "n_users": int(sub["ucode"].nunique()),
           "beta_fri_minus_mon": round(b, 5)}
    for tag, se in (("user", se_u), ("date", se_d), ("twoway", se_2)):
        z = b / se if se > 0 else np.nan
        out[f"se_{tag}"] = round(se, 5)
        out[f"p_{tag}"] = round(float(2 * st.norm.sf(abs(z))), 5) if np.isfinite(z) else np.nan
    return out


def date_block_permutation(d: pd.DataFrame, n_perm=N_PERM, seed=SEED) -> dict:
    """Permute the weekday label across whole dates, keeping each date's reports together.

    This is the null the design implies: under it, which dates are 'Fridays' is arbitrary, but the
    clustering of reports within a date and the covariate structure are untouched. The statistic
    permuted is the adjusted contrast, matching what is reported.
    """
    sub, X, y = _fri_mon_design(d)
    obs = float(_ols(X, y)[0][1])

    dates = sub["dcode"].values
    uniq = np.unique(dates)
    is_fri = pd.Series(sub["fri"].values, index=dates).groupby(level=0).first().reindex(uniq).values
    pos = {dd: i for i, dd in enumerate(uniq)}
    idx = np.fromiter((pos[dd] for dd in dates), int, len(dates))

    rng = np.random.default_rng(seed)
    Xp = X.copy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        lab = rng.permutation(is_fri)
        Xp[:, 1] = lab[idx]
        null[i] = _ols(Xp, y)[0][1]
    p = float((1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1))
    return {"observed": round(obs, 5), "n_perm": n_perm,
            "null_mean": round(float(null.mean()), 5),
            "null_sd": round(float(null.std()), 5), "perm_p_date_block": round(p, 5)}


def day_level(d: pd.DataFrame) -> dict:
    """Weighted regression on daily means: the aggregate-scale test, immune to within-day dependence."""
    from scipy import stats as st
    day = d.date.dt.normalize()
    g = d.groupby(day)
    D = pd.DataFrame({"sentiment": g["sentiment"].mean(), "n": g.size(),
                      "textlen_z": g["textlen_z"].mean()})
    D["dow"] = D.index.dayofweek
    doy = D.index.dayofyear
    D["s1"], D["c1"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)
    D["s2"], D["c2"] = np.sin(4 * np.pi * doy / 365), np.cos(4 * np.pi * doy / 365)
    D = D[D.n >= 20]
    sub = D[D.dow.isin([0, 4])]
    X = np.column_stack([np.ones(len(sub)), (sub.dow == 4).astype(float).values,
                         *(sub[c].values for c in COVARS)])
    w = np.sqrt(sub["n"].values)
    Xw, yw = X * w[:, None], sub["sentiment"].values * w
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    dof = max(len(sub) - X.shape[1], 1)
    V = np.linalg.pinv(Xw.T @ Xw) * float(resid @ resid) / dof
    se = float(np.sqrt(max(V[1, 1], 0)))
    b = float(beta[1])
    return {"n_days": int(len(sub)), "beta_fri_minus_mon": round(b, 5), "se": round(se, 5),
            "p": round(float(2 * st.t.sf(abs(b / se), dof)), 5) if se > 0 else np.nan}


def composition(d: pd.DataFrame) -> pd.DataFrame:
    """Russian share of reports by weekday — the assumption the pooled model makes silently."""
    day = d.date.dt.weekday
    tab = pd.crosstab(day, d.lang, normalize="index")
    tab.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return tab


def main() -> None:
    from scipy import stats as st
    d = load()
    print(f"[data] {len(d):,} reports, {d.date.dt.normalize().nunique()} dates, "
          f"langs={d.lang.value_counts().to_dict()}")

    rows = [dream_level(d, "pooled (as published)")]
    for lang in ("en", "ru"):
        sub = d[d.lang == lang]
        if len(sub) > 500:
            rows.append(dream_level(sub, lang))
    R = pd.DataFrame(rows)
    print("\n=== dream-level Friday-minus-Monday contrast, three variance estimators ===")
    for _, r in R.iterrows():
        print(f"    {r.stratum:22s} beta={r.beta_fri_minus_mon:+.4f}  "
              f"n={int(r.n_reports):6,d} reports / {int(r.n_dates)} dates | "
              f"p(user)={r.p_user:.4f}  p(date)={r.p_date:.4f}  p(2way)={r.p_twoway:.4f}")
    print(f"    -> date-clustered SE is {R.loc[0, 'se_date'] / R.loc[0, 'se_user']:.1f}x the "
          f"user-clustered SE on the pooled estimate")

    print("\n=== date-block permutation (pooled, then EN) ===")
    perms = {}
    for label, sub in (("pooled", d), ("en", d[d.lang == "en"])):
        pr = date_block_permutation(sub)
        perms[label] = pr
        print(f"    {label:7s} observed {pr['observed']:+.4f} | null {pr['null_mean']:+.4f} "
              f"(sd {pr['null_sd']:.4f}) | date-block permutation p={pr['perm_p_date_block']:.4f}")

    print("\n=== day-level (aggregate scale, >=20 reports/day) ===")
    dl = {}
    for label, sub in (("pooled", d), ("en", d[d.lang == "en"])):
        r = day_level(sub)
        dl[label] = r
        print(f"    {label:7s} beta={r['beta_fri_minus_mon']:+.4f} (se {r['se']:.4f}) "
              f"p={r['p']:.4f} on {r['n_days']} days")

    print("\n=== language composition by weekday (pooled model's silent assumption) ===")
    comp = composition(d)
    print(comp.round(4).to_string())
    ru = comp["ru"] if "ru" in comp else pd.Series(dtype=float)
    if len(ru):
        chi = st.chi2_contingency(pd.crosstab(d.date.dt.weekday, d.lang).values)
        print(f"    RU share range {ru.min():.3f}..{ru.max():.3f} (spread {ru.max()-ru.min():.3f}); "
              f"chi2 test of independence p={chi.pvalue:.3g}")

    R["perm_p_date_block_pooled"] = perms["pooled"]["perm_p_date_block"]
    R["perm_p_date_block_en"] = perms["en"]["perm_p_date_block"]
    R["day_level_beta_pooled"] = dl["pooled"]["beta_fri_minus_mon"]
    R["day_level_p_pooled"] = dl["pooled"]["p"]
    R["day_level_beta_en"] = dl["en"]["beta_fri_minus_mon"]
    R["day_level_p_en"] = dl["en"]["p"]
    R.to_csv(TABLES / "weekday_rhythm_reinference.csv", index=False)
    comp.round(5).to_csv(TABLES / "weekday_language_composition.csv")
    print(f"\n[write] weekday_rhythm_reinference.csv, weekday_language_composition.csv")


if __name__ == "__main__":
    main()
