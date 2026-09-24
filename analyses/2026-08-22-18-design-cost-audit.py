"""What do the corrections COST? Injection-recovery across as-run / detrended / deseasonalised.

Alex's objection, and it is a fair one: the trend and seasonal corrections were declared clean
without anyone measuring what they remove. A null under a design that has been stripped of most of
its power is not a null, it is a non-detection, and the two must not be reported the same way. The
corrections were mandated ex ante (METHODS section 2) so they stay primary either way — but their
*cost* is an empirical quantity and it has never been measured in this repo.

So: inject dependence of known strength into the **raw** dream series (trend and season intact,
exactly as the real data arrives), push it through each design, and ask two questions per design.

1. **Attenuation** — what fraction of the injected association survives the correction? This needs
   no surrogates and no inference; it is a direct measurement of signal loss.
2. **Power** — how often does the corrected series beat its own IAAFT null at alpha=.05?

The distinction that decides how every null in F0069 should be worded:

- If the deseasonalised design retains most of a true effect, its nulls are **nulls**, and the two MI
  candidates it killed were genuinely seasonal artifacts.
- If it retains little, its nulls are **underpowered non-detections**, F0069's wording is too strong,
  and the MI deaths are ambiguous between "artifact removed" and "signal removed" — which is the
  competing explanation the round-2 critic was asked to adjudicate.

Note this cuts both ways on purpose. It is the same instrument that could vindicate the corrections
or indict them, and it is run on the candidate's own cell so the answer is about the series that
actually matters, not a convenient one.

Efficiency note: the critical value is estimated once per (design, delta, shape) from a surrogate
bank of B, then all n_rep noise realisations are tested against it, rather than rebuilding a bank
per realisation. The null's shape is stable across realisations because the spectrum is preserved by
construction, so this is a standard and mild approximation; it is disclosed rather than hidden.

Run: PYTHONPATH=src python3 analyses/2026-08-22-18-design-cost-audit.py
"""
from __future__ import annotations

import importlib.util as iu
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from psychohistory import config as C

B = 1000
N_REP = 60
ALPHA = 0.05
DELTAS = (0.0, 0.15, 0.25, 0.35, 0.50, 0.70)
SEED = 20260822
FREQ = "W"
COHORT = "en"

# The candidate's own cell, plus a barometer cell from the null grid as a reference.
CELLS = [
    ("popstate_person_adjusted", "padj_neg_sentiment", "gtrends_war"),
    ("popstate_person_adjusted", "padj_negativity", "gdelt_vol"),
]


def _load(path: str, name: str):
    spec = iu.spec_from_file_location(name, Path(C.ROOT) / path)
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def g_linear(z):
    return z


def g_quad(z):
    q = z ** 2
    return (q - q.mean()) / q.std()


SHAPES = {"linear": g_linear, "quadratic": g_quad}


def resid(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def designs(idx, logn, words):
    """The three designs as bases to residualise on (None = leave the series alone)."""
    t = np.arange(len(idx), dtype=float)
    t = (t - t.mean()) / t.std()
    poly = np.column_stack([t, t ** 2])
    doy = pd.DatetimeIndex(idx).dayofyear.to_numpy(float)
    harm = np.column_stack([np.sin(2 * np.pi * h * doy / 365.25) for h in (1, 2)]
                           + [np.cos(2 * np.pi * h * doy / 365.25) for h in (1, 2)])
    return {
        "as_run": (None, None),
        "detrended": (np.column_stack([poly, logn]), poly),
        "detrended_seasonal": (np.column_stack([poly, harm, logn, words]),
                              np.column_stack([poly, harm])),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    nb = _load("analyses/2026-08-22-12-nonlinear-coupling.py", "nb")
    tr = _load("analyses/2026-08-22-14-nonlinear-trend-robustness.py", "tr")
    v5 = _load("analyses/2026-07-20-13-barometer-v5-search-culture.py", "v5")
    au = _load("analyses/2026-08-22-17-nonlinear-candidate-audit.py", "au")

    panel = v5.load_nonfin_v5(FREQ).dropna(how="any")
    words_s = au.weekly_words()
    frame = nb.popstate_series(COHORT)
    vol = tr.popstate_volume(COHORT)

    rows = []
    for stream, series_name, ind in CELLS:
        s = frame[series_name].dropna()
        idx = s.index.intersection(panel.index).intersection(vol.index).intersection(words_s.index)
        y_raw = s.loc[idx].to_numpy(float)
        x = panel.loc[idx, ind].to_numpy(float)
        logn = np.log(vol.loc[idx].to_numpy(float))
        words = words_s.loc[idx].to_numpy(float)
        n = len(idx)
        D = designs(idx, logn, words)

        # scale for the injection: SD of the fully corrected residual, so "delta" means
        # the same thing in signal-to-noise terms regardless of which design is applied later
        yb, _ = D["detrended_seasonal"]
        sd = resid(y_raw, yb).std()
        z = (x - x.mean()) / x.std()

        for shape, gfn in SHAPES.items():
            gz = gfn(z)
            for delta in DELTAS:
                # one reference realisation per (shape, delta) to set critical values
                for dname, (yb, xb) in D.items():
                    efn_names = ("spearman", "dcor", "mi")
                    crit = {}
                    y_ref = y_raw + delta * sd * gz
                    yy = y_ref if yb is None else resid(y_ref, yb)
                    xx = x if xb is None else resid(x, xb)
                    bank = nb.surrogate_bank(yy, "iaaft", B, rng)
                    for en in efn_names:
                        null = np.array([nb.ESTIMATORS[en](sur, xx) for sur in bank])
                        crit[en] = float(np.quantile(null, 1 - ALPHA))

                    hits = {en: 0 for en in efn_names}
                    att = {en: [] for en in efn_names}
                    rhos = []
                    for _ in range(N_REP):
                        # circular rotation of the dream series = spectrum-preserving noise
                        # realisation that keeps the trend/season structure intact
                        k = int(rng.integers(1, n))
                        y_rep = np.roll(y_raw, k) + delta * sd * gz
                        yy = y_rep if yb is None else resid(y_rep, yb)
                        xx = x if xb is None else resid(x, xb)
                        rhos.append(float(spearmanr(yy, xx).statistic))
                        for en in efn_names:
                            v = nb.ESTIMATORS[en](yy, xx)
                            att[en].append(v)
                            hits[en] += int(v > crit[en])
                    for en in efn_names:
                        rows.append({
                            "series": series_name, "indicator": ind, "shape": shape,
                            "design": dname, "estimator": en, "delta": delta, "n": n,
                            "crit": crit[en], "mean_stat": float(np.mean(att[en])),
                            "mean_spearman": float(np.mean(rhos)),
                            "power": hits[en] / N_REP})
                print(f"[cost] {series_name} x {ind} | {shape} | delta={delta:.2f} done", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(C.TABLES / "nonlinear_design_cost.csv", index=False)

    # ---- attenuation: how much of the injected association each design retains ----
    print("\n=== ATTENUATION: mean recovered Spearman rho by design (linear injection) ===")
    lin = R[(R["shape"] == "linear") & (R.estimator == "spearman")]
    piv = lin.pivot_table(index=["series", "delta"], columns="design", values="mean_spearman")
    print(piv.to_string(float_format=lambda v: f"{v:+.3f}"))
    for ser in lin.series.unique():
        sub = lin[(lin.series == ser) & (lin.delta > 0)]
        p2 = sub.pivot_table(index="delta", columns="design", values="mean_spearman")
        keep = (p2["detrended_seasonal"] / p2["as_run"]).mean()
        keep_d = (p2["detrended"] / p2["as_run"]).mean()
        print(f"  {ser}: detrended retains {100*keep_d:.0f}% of the as-run rho, "
              f"deseasonalised retains {100*keep:.0f}%")

    print("\n=== POWER at alpha=.05 by design (Spearman, linear injection) ===")
    print(lin.pivot_table(index=["series", "delta"], columns="design", values="power")
          .to_string(float_format=lambda v: f"{v:.2f}"))

    print("\n=== POWER, quadratic injection (dCor — the shape Pearson cannot see) ===")
    qd = R[(R["shape"] == "quadratic") & (R.estimator == "dcor")]
    print(qd.pivot_table(index=["series", "delta"], columns="design", values="power")
          .to_string(float_format=lambda v: f"{v:.2f}"))

    print("\n=== MI power by design (linear injection) ===")
    mi = R[(R["shape"] == "linear") & (R.estimator == "mi")]
    print(mi.pivot_table(index=["series", "delta"], columns="design", values="power")
          .to_string(float_format=lambda v: f"{v:.2f}"))

    print(f"\n[cost] -> {C.TABLES/'nonlinear_design_cost.csv'}")


if __name__ == "__main__":
    main()
