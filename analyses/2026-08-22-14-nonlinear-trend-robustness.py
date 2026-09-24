"""Are the non-linear survivors dependence, or a shared ramp?

The non-linear battery (``2026-08-22-12``) returned 19 FDR survivors, and their
distribution is suspicious in three specific ways:

1. **They are confined to the two streams the battery did not detrend.** The 13
   barometer axes were adoption-detrended (residual correlation with time
   r≈0.003) and returned **0/117**. The person-adjusted and arrow streams were
   coupled as raw weekly levels, and they carry every survivor.
2. **They are confined to the three most strongly trending indicators** in the
   panel: ``manifold_risk`` (r with time −0.67), ``wiki_crisis`` (−0.65) and
   ``newsentiment`` (−0.63). Nothing survives on the six flatter indicators.
3. **The single strongest cell is ``padj_food``** (p=2e-4). Food is not a mood
   channel; it is closer to a negative control. A collective-mood result that
   peaks on food is a statement about the corpus, not about dreaming.

Phase and IAAFT surrogates preserve the power *spectrum* but randomise phase, so
they reproduce low-frequency wander without reproducing a *monotone* ramp. Two
series that both ramp will therefore beat their own surrogates even when neither
tells you anything about the other. Distance correlation is the most exposed
estimator here, because it is sensitive to any monotone structure plus scale.

So this re-runs those two streams under three designs on the identical harness:

``as_run``
    Reproduces the battery exactly. A harness check: it must return the same
    survivors, otherwise nothing below is interpretable.

``detrended``
    METHODS section 3, applied symmetrically: the dream series is residualised on
    a quadratic in time **and** log report volume (the adoption/composition
    control the barometer stream already had), and the indicator is residualised
    on the same quadratic in time, so no shared ramp is left on either side.

``differenced``
    Week-over-week change on both sides. Assumes nothing about trend shape, costs
    power at high frequency, and answers a subtly different question — does the
    *movement* co-move — which is the question a "dreams track society" claim
    actually needs.

A survivor that is real should be visible in all three. A survivor that is a
ramp appears only in ``as_run``.

Run: PYTHONPATH=src python3 analyses/2026-08-22-14-nonlinear-trend-robustness.py
"""
from __future__ import annotations

import importlib.util as iu
from pathlib import Path

import numpy as np
import pandas as pd

from psychohistory import config as C
from psychohistory.stats.inference import benjamini_hochberg

B_SCREEN = 200
B_REFINE = 5000
REFINE_AT = 0.10
FREQ = "W"
MIN_OVERLAP = 60
COHORT = "en"                    # RU had no usable series in either stream
DESIGNS = ("as_run", "detrended", "differenced")
SEED = 20260822


def _load(path: str, name: str):
    spec = iu.spec_from_file_location(name, Path(C.ROOT) / path)
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ volume ----
def popstate_volume(cohort: str) -> pd.Series:
    """Weekly report volume behind the person-adjusted state (the adoption term)."""
    d = pd.read_csv(C.TABLES / "popstate_daily_series.csv", parse_dates=["date"])
    d = d[(d.lang == cohort)].dropna(subset=["n_state"])
    v = d.groupby("date")["n_state"].max().resample("W-SUN").sum()
    v.index = v.index.to_period("W-SUN").start_time
    return v[v > 0]


def arrow_volume(cohort: str) -> pd.Series:
    a = pd.read_csv(C.TABLES / "arrowpop_period_series.csv", parse_dates=["period_start"])
    a = a[(a.lang == cohort) & (a.freq == FREQ)].set_index("period_start").sort_index()
    return a["n_reports"].astype(float)


# ------------------------------------------------------------------ designs ----
def _resid(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Residual of y on design matrix X (constant added here)."""
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def apply_design(y: np.ndarray, x: np.ndarray, logn: np.ndarray, design: str):
    """Transform a (dream, indicator) pair. Returns the pair, possibly shortened."""
    if design == "as_run":
        return y, x
    if design == "differenced":
        return np.diff(y), np.diff(x)
    t = np.arange(len(y), dtype=float)
    t = (t - t.mean()) / t.std()
    poly = np.column_stack([t, t ** 2])
    # the dream side also loses adoption/composition; the indicator only loses time
    return _resid(y, np.column_stack([poly, logn])), _resid(x, poly)


# ------------------------------------------------------------------- main ----
def main() -> None:
    rng = np.random.default_rng(SEED)
    nb = _load("analyses/2026-08-22-12-nonlinear-coupling.py", "nb")
    v5 = _load("analyses/2026-07-20-13-barometer-v5-search-culture.py", "v5")

    panel = v5.load_nonfin_v5(FREQ).dropna(how="any")
    streams = {
        "popstate_person_adjusted": (nb.popstate_series(COHORT), popstate_volume(COHORT)),
        "arrow": (nb.arrow_series(COHORT), arrow_volume(COHORT)),
    }

    rows = []
    for stream, (frame, vol) in streams.items():
        for series_name in frame.columns:
            s = frame[series_name].dropna()
            idx = s.index.intersection(panel.index).intersection(vol.index)
            if len(idx) < MIN_OVERLAP:
                continue
            y_raw = s.loc[idx].to_numpy(float)
            logn = np.log(vol.loc[idx].to_numpy(float))
            for design in DESIGNS:
                # the design transform of the dream series does not depend on the
                # indicator, so one surrogate bank per (series, design) serves all nine
                y, _ = apply_design(y_raw, y_raw, logn, design)
                banks = {nl: nb.surrogate_bank(y, nl, B_SCREEN, rng) for nl in nb.NULLS}
                cache: dict[str, np.ndarray] = {}
                for ind in panel.columns:
                    _, x = apply_design(y_raw, panel.loc[idx, ind].to_numpy(float),
                                        logn, design)
                    for ename, efn in nb.ESTIMATORS.items():
                        obs = efn(y, x)
                        for null, bank in banks.items():
                            p, nm, q95 = nb._mc_p(efn, obs, x, bank)
                            if p <= REFINE_AT:
                                if null not in cache:
                                    cache[null] = nb.surrogate_bank(y, null, B_REFINE, rng)
                                p, nm, q95 = nb._mc_p(efn, obs, x, cache[null])
                            rows.append({
                                "cohort": COHORT, "stream": stream, "series": series_name,
                                "design": design, "indicator": ind, "estimator": ename,
                                "null": null, "n": len(y), "stat": obs, "null_mean": nm,
                                "null_q95": q95, "p": p,
                                "B": B_REFINE if p <= REFINE_AT else B_SCREEN})
            print(f"[robust] {stream}/{series_name}: {len(DESIGNS)} designs done", flush=True)

    R = pd.DataFrame(rows)
    R["q"] = R.groupby(["design", "stream", "estimator", "null"])["p"].transform(
        lambda p: benjamini_hochberg(p.to_numpy()))
    R.to_csv(C.TABLES / "nonlinear_trend_robustness.csv", index=False)

    print("\n=== survivors (q<.10) per design x stream x estimator ===")
    fam = R.groupby(["design", "stream", "estimator", "null"]).agg(
        tests=("p", "size"), nominal=("p", lambda p: int((p < .05).sum())),
        expected=("p", lambda p: round(.05 * len(p), 1)),
        q10=("q", lambda q: int((q < .10).sum()))).reset_index()
    print(fam.to_string(index=False))

    print("\n=== do the 19 as-run survivors persist? ===")
    key = ["stream", "series", "indicator", "estimator", "null"]
    base = R[(R.design == "as_run") & (R.q < .10)]
    for _, r in base.sort_values("p").iterrows():
        line = f"  {r.series:20s} x {r.indicator:14s} {r.estimator:8s} {r.null:5s}"
        for design in DESIGNS[1:]:
            m = R[(R.design == design) & np.logical_and.reduce(
                [R[k] == r[k] for k in key])]
            if m.empty:
                line += f"  {design}=NA"
            else:
                mm = m.iloc[0]
                line += f"  {design}: p={mm.p:.3f} q={mm.q:.3f}{'*' if mm.q < .10 else ''}"
        print(line)

    print(f"\n[robust] -> {C.TABLES/'nonlinear_trend_robustness.csv'}")


if __name__ == "__main__":
    main()
