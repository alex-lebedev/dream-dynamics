"""Per-report arrow statistics as a panel: the within-report trajectory as a population variable.

The manuscript's strongest result is a within-report fact — dream narratives descend emotionally —
and its weakest layer is a between-day fact about aggregate mood. They are reported as if they were
about different objects. They need not be: if the *shape* of the trajectory varies over calendar
time more than sampling noise allows, then the arrow is itself a population observable, and the
strongest measurement in the corpus becomes available to the societal question.

This module turns the per-sentence valence caches into one row per report — drift, slope, curvature,
volatility, arc class — so that the same trait/state machinery in `popstate` can be applied to them.

    from psychohistory.dreams.arrow_dynamics import report_panel, ARC_CLASSES
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C

# Four arc classes from the sign pattern of the two third-to-third steps. Coarse on purpose: a
# mixture over four bins is estimable month by month, a continuous shape distribution is not.
ARC_CLASSES = ("fall", "rise", "peak", "trough")


def _arc_class(t1: float, t2: float, t3: float) -> str:
    up1, up2 = t2 >= t1, t3 >= t2
    if not up1 and not up2:
        return "fall"
    if up1 and up2:
        return "rise"
    return "peak" if up1 else "trough"


def report_stats(v: np.ndarray) -> dict:
    """Arrow statistics of one report's sentence-valence trajectory.

    `drift` is the published endpoint statistic. `drift_trim` drops the final sentence, which is
    where an awakening clause lands, and is the artifact-robust counterpart. `slope` uses every
    sentence and is therefore the lower-variance estimate of the same direction; `half` is the
    second-half-minus-first-half contrast. `sd_within` is the trajectory's own volatility, which is
    the within-report analogue of the population-volatility measure the manuscript reports.
    """
    v = np.asarray(v, float)
    k = len(v)
    x = np.linspace(0.0, 1.0, k)
    xc = x - x.mean()
    slope = float(np.dot(xc, v - v.mean()) / np.dot(xc, xc))
    h = k // 2
    j = max(k // 3, 1)
    t1, t3 = float(v[:j].mean()), float(v[-j:].mean())
    t2 = float(v[j:k - j].mean()) if k - 2 * j > 0 else float(v.mean())
    return {"n_sent": k, "drift": float(v[-1] - v[0]),
            "drift_trim": float(v[-2] - v[0]) if k >= 3 else np.nan,
            "slope": slope, "half": float(v[h:].mean() - v[:h].mean()),
            "third": t3 - t1, "level": float(v.mean()), "sd_within": float(v.std()),
            "min_pos": float(np.argmin(v) / (k - 1)), "max_pos": float(np.argmax(v) / (k - 1)),
            "arc": _arc_class(t1, t2, t3)}


def report_panel(docs: np.ndarray, counts: np.ndarray, vals: np.ndarray,
                 start: str = "2024-03-01") -> pd.DataFrame:
    """One row per report: arrow statistics joined to date, contributor and language.

    The contributor key and the date are attached in memory from the dream-level table; the cache
    on disk holds neither (docs/ETHICS.md S1), and nothing written by callers of this function is
    below the release floor.
    """
    off = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    rows = []
    for k in range(len(counts)):
        v = vals[off[k]:off[k] + counts[k]]
        if not np.isfinite(v).all():
            continue
        r = report_stats(v)
        r["documentID"] = docs[k]
        rows.append(r)
    P = pd.DataFrame(rows)
    lv = pd.read_parquet(C.DREAMS_OUT / "dreamseer_dream_level.parquet",
                         columns=["documentID", "userID", "date", "lang", "textlen",
                                  "negativity", "nightmare_index"])
    lv["date"] = pd.to_datetime(lv["date"])
    P = P.merge(lv, on="documentID", how="inner")
    P = P[P.date >= start].copy()
    P["day"] = P.date.dt.normalize()
    for cls in ARC_CLASSES:
        P[f"arc_{cls}"] = (P["arc"] == cls).astype(float)
    return P.reset_index(drop=True)


def mixture_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


# ------------------------------------------------- distributional dynamics test ----
def _period_stats(code: np.ndarray, k: int, y: np.ndarray, tail: np.ndarray,
                  arc: np.ndarray, n_arc: int):
    """Per-period mean, dispersion, tail rate and arc-mixture entropy, by bincount."""
    n = np.bincount(code, minlength=k).astype(float)
    s = np.bincount(code, weights=y, minlength=k)
    s2 = np.bincount(code, weights=y * y, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / n
        var = np.maximum(s2 / n - mean ** 2, 0.0)
        tailr = np.bincount(code, weights=tail, minlength=k) / n
        props = np.stack([np.bincount(code[arc == a], minlength=k).astype(float) / n
                          for a in range(n_arc)])
    ent = -np.nansum(np.where(props > 0, props * np.log(np.where(props > 0, props, 1.0)), 0.0),
                     axis=0)
    return n, {"mean": mean, "sd": np.sqrt(var), "tail_rate": tailr, "arc_entropy": ent,
               "p_fall": props[0]}


def distribution_dynamics_test(df: pd.DataFrame, value: str, period: str, person: str = "userID",
                               tail_q: float = 0.10, min_cell: int = 30, B: int = 1000,
                               seed: int = 0) -> pd.DataFrame:
    """Do the higher moments of the arrow distribution move over calendar time?

    The manuscript's aggregate layer is a series of means. A population state can express itself in
    dispersion, in the weight of the lower tail, or in the mixture of trajectory shapes while the
    mean stays put, and those are different claims about what a dreaming population is doing. Each
    statistic is computed per period; the test statistic is its volume-weighted variance across
    periods, and the null permutes each contributor's reports across that contributor's own periods.
    Volume per period, composition per period and each contributor's own distribution of values all
    survive that permutation; only the alignment between the calendar and the reports does not.
    """
    d = df.dropna(subset=[value]).copy()
    y = d[value].to_numpy(float)
    thr = float(np.quantile(y, tail_q))
    tail = (y <= thr).astype(float)
    arc = pd.Categorical(d["arc"], categories=list(ARC_CLASSES)).codes.astype(int)
    code, uniq = pd.factorize(d[period], sort=True)
    pc, _ = pd.factorize(d[person], sort=False)
    k = len(uniq)

    def wvar(vals, n, keep):
        v, w = vals[keep], n[keep]
        m = np.average(v, weights=w)
        return float(np.average((v - m) ** 2, weights=w))

    n_obs, obs = _period_stats(code, k, y, tail, arc, len(ARC_CLASSES))
    keep = n_obs >= min_cell
    observed = {nm: wvar(val, n_obs, keep) for nm, val in obs.items()}

    rng = np.random.default_rng(seed)
    order = np.argsort(pc, kind="stable")
    blocks = [b for b in np.split(order, np.flatnonzero(np.diff(pc[order])) + 1) if len(b) > 1]
    null = {nm: np.empty(B) for nm in observed}
    c = code.copy()
    for j in range(B):
        for b in blocks:
            c[b] = code[rng.permutation(b)]
        n_j, st_j = _period_stats(c, k, y, tail, arc, len(ARC_CLASSES))
        kj = n_j >= min_cell
        for nm, val in st_j.items():
            null[nm][j] = wvar(val, n_j, kj)

    rows = []
    for nm, o in observed.items():
        nl = null[nm]
        rows.append({"statistic": nm, "between_period_var": o,
                     "null_mean": float(np.mean(nl)), "null_q95": float(np.percentile(nl, 95)),
                     "excess_sd": float(np.sqrt(max(o - np.mean(nl), 0.0))),
                     "bound_sd_q95": float(np.sqrt(max(np.percentile(nl, 95) - np.mean(nl), 0.0))),
                     "p_perm": float((1 + np.sum(nl >= o)) / (B + 1)),
                     "n_periods": int(keep.sum()), "n_obs": int(len(d)), "B": B})
    return pd.DataFrame(rows)
