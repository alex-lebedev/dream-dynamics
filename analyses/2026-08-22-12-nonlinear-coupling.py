"""Non-linear coupling: does mutual information find what Pearson missed?

The repo's central negative result — no continuous coupling between the dream
aggregate and any societal indicator (F0037 to F0048, 117 tests) — is a *linear*
null. It was estimated with Pearson correlation and phase-surrogate inference.
The only non-linear work (F0053) covered 5 affect metrics against 4 indicators,
EN only, and returned 4/320 hits, below chance. So a fair question survives:
is there dependence that a correlation coefficient cannot see?

This runs the full grid with two non-linear estimators and closes three gaps.

**Streams.** (1) The 13 barometer axes against all 9 indicators — the F0048
grid, re-estimated. (2) The **person-adjusted** population-state series, which
has never been coupled to an external indicator at all; the manuscript's own
limitations section names this as owed, because the paper otherwise tests
social time on the composition-controlled series and the informational channel
on the raw aggregate. (3) The **arrow** (endpoint drift and trajectory slope),
also never coupled externally.

**Estimators.** Pearson and Spearman (for reference), distance correlation, and
KSG mutual information — the same ``mutual_info_regression(n_neighbors=3)``
estimator F0053 used, so the two are comparable.

**Nulls.** Mutual information and distance correlation are biased upward, are
non-negative by construction, and are sensitive to the marginal distribution,
so an asymptotic p-value is meaningless here. Primary inference is an **IAAFT**
surrogate, which preserves both the spectrum and the amplitude distribution of
the dream series; a **phase-randomisation** arm is reported alongside it for
comparability with F0048 and F0053, which used it.

**Injection calibration.** A null from a biased estimator at n≈120 is worthless
without knowing what the estimator could have caught. Dependence of known
strength and known *shape* — including a purely quadratic dependence that
Pearson is blind to by construction — is injected through the identical
pipeline, giving the strength at which each estimator reaches 80% power.

Run: PYTHONPATH=src python3 analyses/2026-08-22-12-nonlinear-coupling.py
"""
from __future__ import annotations

import importlib.util as iu
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_selection import mutual_info_regression

from psychohistory import config as C
from psychohistory.stats.ews import iaaft_surrogate, phase_surrogate
from psychohistory.stats.inference import benjamini_hochberg

#: Two-stage Monte-Carlo inference. A single pass at a few hundred surrogates
#: cannot support Benjamini-Hochberg over a 117-test family: the smallest
#: attainable p-value is 1/(B+1), so at B=500 even a perfect cell reaches only
#: q=0.23 and no true effect could ever survive. Every cell is therefore
#: screened at ``B_SCREEN``, and any cell that could plausibly matter
#: (screen p <= ``REFINE_AT``) is re-estimated at ``B_REFINE``, which resolves
#: p down to 2e-4 where BH has room to work.
B_SCREEN = 200
B_REFINE = 5000
REFINE_AT = 0.10

FREQ = "W"                      # monthly leaves n=30: too few for a k-NN MI estimator
MIN_OVERLAP = 60
COHORTS = ("en", "ru")
NULLS = ("iaaft", "phase")
SEED = 20260822


def _load(path: str, name: str):
    spec = iu.spec_from_file_location(name, Path(C.ROOT) / path)
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------ estimators ----
def est_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return abs(float((a * b).sum() / d)) if d > 0 else 0.0


def est_spearman(a: np.ndarray, b: np.ndarray) -> float:
    return est_pearson(rankdata(a), rankdata(b))


def est_dcor(a: np.ndarray, b: np.ndarray) -> float:
    """Szekely distance correlation (univariate, double-centred)."""
    A = np.abs(a[:, None] - a[None, :])
    Bm = np.abs(b[:, None] - b[None, :])
    A = A - A.mean(0)[None, :] - A.mean(1)[:, None] + A.mean()
    Bm = Bm - Bm.mean(0)[None, :] - Bm.mean(1)[:, None] + Bm.mean()
    dcov = (A * Bm).mean()
    vx, vy = (A * A).mean(), (Bm * Bm).mean()
    return float(np.sqrt(max(dcov, 0.0) / np.sqrt(vx * vy))) if vx * vy > 0 else 0.0


def est_mi(a: np.ndarray, b: np.ndarray) -> float:
    return float(mutual_info_regression(a.reshape(-1, 1), b, random_state=0, n_neighbors=3)[0])


ESTIMATORS = {"pearson": est_pearson, "spearman": est_spearman,
              "dcor": est_dcor, "mi": est_mi}


# -------------------------------------------------------------- surrogates ----
def surrogate_bank(x: np.ndarray, null: str, B: int, rng) -> np.ndarray:
    """B surrogates of the dream series, generated once and reused across indicators."""
    gen = iaaft_surrogate if null == "iaaft" else phase_surrogate
    return np.stack([gen(x, rng) for _ in range(B)])


def _mc_p(efn, obs: float, x: np.ndarray, bank: np.ndarray) -> tuple[float, float, float]:
    nulls = np.array([efn(s, x) for s in bank])
    p = (1 + int(np.sum(nulls >= obs))) / (len(nulls) + 1)
    return float(p), float(nulls.mean()), float(np.quantile(nulls, 0.95))


def test_series(y: np.ndarray, panel: pd.DataFrame, idx, banks: dict[str, np.ndarray],
                big_bank) -> list[dict]:
    """All indicators x estimators x nulls for one dream series, with refinement.

    ``big_bank(null)`` lazily returns a ``B_REFINE`` bank, built at most once per
    (series, null) and only if some cell survives the screen.
    """
    rows = []
    for ind in panel.columns:
        x = panel.loc[idx, ind].to_numpy(float)
        for ename, efn in ESTIMATORS.items():
            obs = efn(y, x)
            for null, bank in banks.items():
                p, nm, q95 = _mc_p(efn, obs, x, bank)
                rows.append({"indicator": ind, "estimator": ename, "null": null,
                             "stat": obs, "null_mean": nm, "null_q95": q95,
                             "p": p, "B": len(bank), "refined": False})
    for r in rows:
        if r["p"] > REFINE_AT:
            continue
        x = panel.loc[idx, r["indicator"]].to_numpy(float)
        efn = ESTIMATORS[r["estimator"]]
        p, nm, q95 = _mc_p(efn, r["stat"], x, big_bank(r["null"]))
        r.update(p=p, null_mean=nm, null_q95=q95, B=B_REFINE, refined=True)
    return rows


# ---------------------------------------------------------- dream streams ----
def barometer_axes(cohort: str, w24) -> pd.DataFrame:
    """The 13 F0048 axes, adoption-detrended exactly as that battery did."""
    ax = w24.dream_axes(FREQ, cohort)
    out = {}
    for a in ax.columns:
        if a in ("n", "logn") or a not in w24.TYPE:
            continue
        s = ax[a].dropna()
        if len(s) < MIN_OVERLAP:
            continue
        out[a] = pd.Series(w24.detrend(s.to_numpy(float), ax.loc[s.index, "logn"].to_numpy(float)),
                           index=s.index)
    return pd.DataFrame(out)


def popstate_series(cohort: str) -> pd.DataFrame:
    """Person-adjusted daily state, aggregated to weeks. Never externally coupled before."""
    d = pd.read_csv(C.TABLES / "popstate_daily_series.csv", parse_dates=["date"])
    d = d[d.lang == cohort]
    out = {}
    for oc, g in d.groupby("outcome"):
        g = g.dropna(subset=["state"])
        if g.empty:
            continue
        wk = (g.set_index("date")["state"]
              .resample("W-SUN").mean().dropna())
        wk.index = wk.index.to_period("W-SUN").start_time
        if len(wk) >= MIN_OVERLAP:
            out[f"padj_{oc}"] = wk
    return pd.DataFrame(out)


def arrow_series(cohort: str) -> pd.DataFrame:
    """Weekly arrow drift and slope. Never externally coupled before."""
    a = pd.read_csv(C.TABLES / "arrowpop_period_series.csv", parse_dates=["period_start"])
    a = a[(a.lang == cohort) & (a.freq == "W")].set_index("period_start").sort_index()
    out = {}
    for col, name in (("drift_mean", "arrow_drift"), ("slope_mean", "arrow_slope")):
        if col in a.columns:
            s = a[col].dropna()
            if len(s) >= MIN_OVERLAP:
                out[name] = s
    return pd.DataFrame(out)


# ------------------------------------------------------------- injection ----
def injection_power(y: np.ndarray, x: np.ndarray, shape: str, rhos: np.ndarray,
                    n_rep: int = 100, B: int = 150, seed: int = 0) -> dict[float, dict[str, float]]:
    """Power of each estimator against dependence of known strength and shape.

    ``y`` supplies realistic autocorrelated noise; the injected signal is a
    function of the real indicator ``x``. ``quadratic`` is the case of interest:
    it is a genuine dependence that Pearson is blind to by construction, so if
    the linear null were hiding non-linear structure, this is the shape that
    would reveal the difference in sensitivity.
    """
    rng = np.random.default_rng(seed)
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    g = {"linear": z(x), "quadratic": z(z(x) ** 2), "abs": z(np.abs(z(x))),
         "cosine": z(np.cos(2.5 * z(x)))}[shape]

    out = {}
    for rho in rhos:
        hits = {e: 0 for e in ESTIMATORS}
        for _ in range(n_rep):
            noise = phase_surrogate(y, rng)
            y_inj = np.sqrt(max(1 - rho ** 2, 0.0)) * z(noise) + rho * g
            bank = surrogate_bank(y_inj, "iaaft", B, rng)
            for ename, efn in ESTIMATORS.items():
                if _mc_p(efn, efn(y_inj, x), x, bank)[0] < 0.05:
                    hits[ename] += 1
        out[float(rho)] = {e: hits[e] / n_rep for e in ESTIMATORS}
    return out


# ------------------------------------------------------------------ main ----
def main() -> None:
    rng = np.random.default_rng(SEED)
    w24 = _load("analyses/2026-07-19-24-collective-wave-poc.py", "w24")
    v5 = _load("analyses/2026-07-20-13-barometer-v5-search-culture.py", "v5")

    panel = v5.load_nonfin_v5(FREQ).dropna(how="any")
    print(f"[nonlinear] indicators={list(panel.columns)} n={len(panel)}")

    rows = []
    for cohort in COHORTS:
        streams = {"barometer": barometer_axes(cohort, w24),
                   "popstate_person_adjusted": popstate_series(cohort),
                   "arrow": arrow_series(cohort)}
        for stream, frame in streams.items():
            if frame.empty:
                print(f"[nonlinear] {cohort}/{stream}: no usable series")
                continue
            for series_name in frame.columns:
                s = frame[series_name].dropna()
                idx = s.index.intersection(panel.index)
                if len(idx) < MIN_OVERLAP:
                    continue
                y = s.loc[idx].to_numpy(float)
                # one surrogate bank per dream series, reused across all 9 indicators
                banks = {nl: surrogate_bank(y, nl, B_SCREEN, rng) for nl in NULLS}
                cache: dict[str, np.ndarray] = {}

                def big_bank(null: str, _y=y) -> np.ndarray:
                    if null not in cache:
                        cache[null] = surrogate_bank(_y, null, B_REFINE, rng)
                    return cache[null]

                for r in test_series(y, panel, idx, banks, big_bank):
                    rows.append({"cohort": cohort, "stream": stream,
                                 "series": series_name, "n": len(idx), **r})
            print(f"[nonlinear] {cohort}/{stream}: {frame.shape[1]} series done")

    R = pd.DataFrame(rows)
    # families: one per stream x estimator x null x cohort (the F0048 grid is one such family)
    R["q"] = R.groupby(["stream", "estimator", "null", "cohort"])["p"].transform(
        lambda p: benjamini_hochberg(p.to_numpy()))
    C.TABLES.mkdir(parents=True, exist_ok=True)
    R.to_csv(C.TABLES / "nonlinear_coupling.csv", index=False)

    print("\n=== tests, nominal hits and survivors per family ===")
    fam = R.groupby(["stream", "estimator", "null", "cohort"]).agg(
        tests=("p", "size"), nominal=("p", lambda p: int((p < .05).sum())),
        expected=("p", lambda p: round(.05 * len(p), 1)),
        q10=("q", lambda q: int((q < .10).sum())), min_p=("p", "min")).reset_index()
    print(fam.to_string(index=False))

    # --- injection calibration on a representative EN cell ------------------
    base = barometer_axes("en", w24)
    key = "threat" if "threat" in base.columns else base.columns[0]
    s = base[key].dropna()
    idx = s.index.intersection(panel.index)
    y0 = s.loc[idx].to_numpy(float)
    x0 = panel.loc[idx, "gdelt_tone"].to_numpy(float)
    rhos = np.array([0.15, 0.25, 0.35, 0.45, 0.60])
    inj_rows = []
    for shape in ("linear", "quadratic", "cosine"):
        curve = injection_power(y0, x0, shape, rhos, seed=SEED)
        for rho, powers in curve.items():
            inj_rows.append({"shape": shape, "rho": rho, "n": len(idx), **powers})
    I = pd.DataFrame(inj_rows)
    I.to_csv(C.TABLES / "nonlinear_injection_power.csv", index=False)

    print(f"\n=== injection power (EN {key} vs gdelt_tone, n={len(idx)}, IAAFT null) ===")
    print(I.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nsmallest rho reaching 80% power:")
    for shape, g in I.groupby("shape"):
        mins = {e: (g.loc[g[e] >= .8, "rho"].min() if (g[e] >= .8).any() else np.nan)
                for e in ESTIMATORS}
        print(f"  {shape:10s} " + "  ".join(f"{e}={mins[e]!s:>5}" for e in ESTIMATORS))
    print(f"\n[nonlinear] -> {C.TABLES/'nonlinear_coupling.csv'}")


if __name__ == "__main__":
    main()
