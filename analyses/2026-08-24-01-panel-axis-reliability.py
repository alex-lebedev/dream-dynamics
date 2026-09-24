"""PANEL-MATCHED SPLIT-HALF RELIABILITY for the 13 barometer axes, with sampling intervals.

Why this exists. `dream-dynamics.md` §2.5/§3.6/§4 read the external-coupling null against a
reliability figure, and the figure it had been reading was the wrong object three times over:
`2026-07-19-07-reliability-audit.py` is all-language, detrends on time alone, and covers only
three of the panel's thirteen axes (threat, sentiment, negativity). The panel
(`2026-08-22-12-nonlinear-coupling.py` -> `barometer_axes`) correlates EN-only weekly means,
gated at n>=30 reports/week, residualised on [1, standardised time, centred log n].

This audit matches all three: same cohort, same period key and gate, same detrend, all 13 axes.
Reported per axis: the split-half correlation of the DETRENDED series (the quantity the panel
correlates), the same for LEVELS, and the Spearman-Brown step-up of each to full length. The
step-up is needed because a split-half r is the reliability of a period mean taken over HALF the
period's reports, whereas the panel correlates the mean over all of them; here it is more than a
psychometric analogy, since a period mean's sampling-error variance goes as sigma^2/n, so halving
the reports doubles it, which is exactly what 2r/(1+r) assumes.

Four things the first version of this script did not do, each of which the manuscript's arithmetic
turned out to need:

1.  SAMPLING INTERVALS. Averaging over many half-splits removes the variance due to which reports
    landed on which side, but every split sees the same ~121 weeks. The worst axes sit at split-half
    r near 0.10, where the interval at that number of weeks is wide enough that the disattenuation
    factor is effectively unbounded above -- so the point estimate alone cannot carry a
    "noise-free association of about X" claim. Weeks are resampled in circular blocks and *every*
    split assignment is re-evaluated inside each draw, so assignment spread is not counted twice
    (the idiom is `2026-08-22-07-arrow-affect-coupling.py::split_corr`).
2.  UNIT OF SPLIT. Splitting reports leaves a prolific contributor on both sides of the split, so
    their idiosyncratic level is counted as true score rather than as error, which inflates
    reliability. Both units are reported: `report`, which is the panel's own aggregation unit and
    the more flattering, and `contributor`, which treats contributor sampling as error and is the
    right estimand if the target is a population state a fresh sample of dreamers should reproduce.
3.  DETREND SENSITIVITY. Each half is residualised on its own log n above (log n_half = log n +
    log f, so the adoption component the panel removes survives and the extra term is noise), but
    the panel's regressor is the *full* period's log n; the `fulln` columns run it that way.
4.  REALISED PERIOD COUNTS, per axis, rather than only the size of the gate set -- the dispersion
    axis carries its own embedded-count gate and need not rest on exactly the same periods.

Conventions, so the released numbers are reconstructable: correlations are averaged untransformed
across splits (not Fisher-z), and Spearman-Brown is applied to that mean rather than averaged over
per-split step-ups. Both choices are small and partly offsetting; the z-mean differs in the third
decimal.

Aggregate-only output; no report text, no identifiers.

    PYTHONPATH=src PYTHONNOUSERSITE=1 python3 analyses/2026-08-24-01-panel-axis-reliability.py
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from psychohistory import config as C

OUT = C.RESULTS / "showcase"
COHORT = "en"
N_SPLITS = 200
BOOT_SPLITS = 50
N_BOOT = 500
BLOCK = {"W": 8, "M": 3}
MIN_PERIODS = 12
MIN_HALF_EMB = 5
GATE = {"W": 30, "M": 120}
SEED = 0

EMO = ["fear", "anger", "sadness", "disgust", "joy", "trust", "anticipation", "surprise"]
THREAT = ["danger", "weapon", "fear", "nightmare_index"]
NEUTRAL = ["family", "food", "flying", "animal", "social"]
AGENCY = re.compile(r"\b(I|I'm|I've|I'd|me|my|mine|myself)\b", re.I)
NUM_AXES = ["valence_neg", "neg_sent", "intensity", "arousal_hi", "threat", "grief",
            *NEUTRAL, "agency"]
AXES = [*NUM_AXES, "dispersion"]


def _load(fn: str, name: str):
    """Import a dated analysis script as a module (the barometer scripts' own idiom)."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / fn)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sb(r: float) -> float:
    """Spearman-Brown step-up from a half-reports split-half r to full-length reliability."""
    return 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan


def period_key(dates: pd.Series, freq: str) -> pd.Series:
    return (dates.dt.to_period("W-SUN") if freq == "W" else dates.dt.to_period("M")).dt.start_time


def build_dream_level(w24) -> tuple[pd.DataFrame, np.ndarray]:
    """EN dream-level frame carrying every per-report axis, plus the unit-normed embeddings."""
    lv = pd.read_parquet(
        C.DREAMS_OUT / "dreamseer_dream_level.parquet",
        columns=["documentID", "userID", "date", "lang"] + EMO
        + ["danger", "weapon", "nightmare_index", "negativity"] + NEUTRAL + ["textlen"])
    lv["date"] = pd.to_datetime(lv["date"])
    lv = lv[lv.date >= "2024-03-01"]
    se = pd.read_parquet(C.DREAMS_OUT / "dreamseer_sentiment.parquet",
                         columns=["documentID", "sentiment"])
    lv = lv.merge(se, on="documentID", how="left")
    lv = lv[lv.lang == COHORT].reset_index(drop=True)

    lv["valence_neg"] = lv["negativity"]
    lv["neg_sent"] = -lv["sentiment"]
    lv["intensity"] = lv[EMO].mean(axis=1)
    lv["arousal_hi"] = lv[["fear", "anger", "surprise", "anticipation"]].mean(axis=1)
    lv["threat"] = lv[THREAT].mean(axis=1)
    lv["grief"] = lv["sadness"]

    raw = pd.read_csv(C.DREAMSEER_RAW, dtype=str, keep_default_na=False,
                      usecols=["documentID", "text"])
    tmap = dict(zip(raw.documentID, raw.text))
    lv["agency"] = lv.documentID.map(
        lambda d: len(AGENCY.findall(tmap.get(d, ""))) / max(len(tmap.get(d, "").split()), 1))

    meta, emb = w24.load_or_build()
    E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    idx = pd.Series(np.arange(len(meta)), index=meta.documentID.values)
    lv["_ei"] = lv.documentID.map(idx).astype("Int64")
    return lv, E


# --------------------------------------------------------------- estimators ----
def _resid(y: np.ndarray, cols: list[np.ndarray], m: np.ndarray) -> np.ndarray:
    """Masked OLS residual of every row of `y` on `cols`, batched over rows (splits)."""
    X = np.where(m[..., None], np.stack([np.nan_to_num(np.broadcast_to(c, y.shape))
                                         for c in cols], -1), 0.0)
    ym = np.where(m, np.nan_to_num(y), 0.0)
    XtX = np.einsum("spk,spl->skl", X, X) + 1e-9 * np.eye(X.shape[-1])
    beta = np.linalg.solve(XtX, np.einsum("spk,sp->sk", X, ym))
    return (ym - np.einsum("spk,sk->sp", X, beta)) * m


def _corr(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Masked Pearson r per row, NaN where a row has too few usable periods."""
    n = m.sum(1).astype(float)
    A, B = np.where(m, np.nan_to_num(a), 0.0), np.where(m, np.nan_to_num(b), 0.0)
    d = np.maximum(n, 1.0)
    S = lambda x, y: (x * y).sum(1) - x.sum(1) * y.sum(1) / d
    with np.errstate(invalid="ignore", divide="ignore"):
        r = S(A, B) / np.sqrt(S(A, A) * S(B, B))
    return np.where((n >= MIN_PERIODS) & np.isfinite(r), r, np.nan)


def _split_r(P: dict, idx: np.ndarray, rows: slice, mode: str) -> float:
    """Mean split-half r over split assignments, on the periods `idx`, in one of three modes.

    `idx` is a set of calendar positions -- the whole gate set for the point estimate, a circular
    block resample of it for a bootstrap draw. Detrending uses those calendar positions rather than
    the position a week happens to occupy in the resample, so a draw estimates the same functional.
    """
    a, b = P["a"][rows][:, idx], P["b"][rows][:, idx]
    m = P["m"][rows][:, idx]
    if mode == "level":
        return float(np.nanmean(_corr(a, b, m)))
    t = (idx - idx.mean()) / (idx.std() + 1e-9)
    la, lb = np.log(P["na"][rows][:, idx]), np.log(P["nb"][rows][:, idx])
    if mode == "fulln":
        la = lb = np.log(P["nt"][idx])[None, :]
    ra = _resid(a, [np.ones_like(t), t, la - np.nanmean(np.where(m, la, np.nan))], m)
    rb = _resid(b, [np.ones_like(t), t, lb - np.nanmean(np.where(m, lb, np.nan))], m)
    return float(np.nanmean(_corr(ra, rb, m)))


def half_panels(lv: pd.DataFrame, E: np.ndarray, freq: str, unit: str,
                rng: np.random.Generator) -> dict[str, dict]:
    """Per-axis (n_splits x n_periods) matrices of the two halves' period means and counts."""
    gate = GATE[freq]
    key = period_key(lv["date"], freq)
    counts = key.value_counts()
    periods = np.sort(np.array([p for p in counts.index if counts[p] >= gate]))
    code = pd.Series(np.arange(len(periods)), index=periods).reindex(key).to_numpy()
    keep = np.isfinite(code)
    sub, code = lv[keep].reset_index(drop=True), code[keep].astype(int)
    P = len(periods)

    ei = sub["_ei"].to_numpy()
    emb_ok = pd.notna(ei)
    ei_ok = ei[emb_ok].astype(int)
    # the panel gates dispersion on the period's own embedded count, so we do too
    emb_tot = np.bincount(code[emb_ok], minlength=P).astype(float)
    n_tot = np.bincount(code, minlength=P).astype(float)
    users = sub.userID.to_numpy()
    uniq = np.unique(users)
    Y = {a: sub[a].to_numpy(float) for a in NUM_AXES}

    acc = {a: {k: [] for k in ("a", "b", "na", "nb", "m")} for a in AXES}
    for _ in range(N_SPLITS):
        if unit == "contributor":
            h = pd.Series(rng.integers(0, 2, len(uniq)), index=uniq).reindex(users).to_numpy()
        else:
            h = rng.integers(0, 2, len(sub))
        g = code * 2 + h
        for a in NUM_AXES:
            y = Y[a]
            ok = np.isfinite(y)
            c = np.bincount(g[ok], minlength=2 * P).reshape(P, 2).astype(float)
            s = np.bincount(g[ok], weights=y[ok], minlength=2 * P).reshape(P, 2)
            with np.errstate(invalid="ignore", divide="ignore"):
                v = np.where(c > 0, s / np.maximum(c, 1), np.nan)
            acc[a]["a"].append(v[:, 0]), acc[a]["b"].append(v[:, 1])
            acc[a]["na"].append(np.maximum(c[:, 0], 1)), acc[a]["nb"].append(np.maximum(c[:, 1], 1))
            acc[a]["m"].append((c > 0).all(1) & np.isfinite(v).all(1))

        ge = g[emb_ok]
        ce = np.bincount(ge, minlength=2 * P).reshape(P, 2).astype(float)
        M = sp.csr_matrix((np.ones(len(ge)), (ge, np.arange(len(ge)))), shape=(2 * P, len(ge)))
        norm = np.linalg.norm(M @ E[ei_ok], axis=1).reshape(P, 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(ce > 0, 1 - norm / np.maximum(ce, 1), np.nan)
        acc["dispersion"]["a"].append(v[:, 0]), acc["dispersion"]["b"].append(v[:, 1])
        acc["dispersion"]["na"].append(np.maximum(ce[:, 0], 1))
        acc["dispersion"]["nb"].append(np.maximum(ce[:, 1], 1))
        acc["dispersion"]["m"].append((ce >= MIN_HALF_EMB).all(1) & (emb_tot >= gate)
                                      & np.isfinite(v).all(1))

    out = {}
    for a in AXES:
        d = {k: np.vstack(acc[a][k]) for k in acc[a]}
        d["nt"] = emb_tot if a == "dispersion" else n_tot
        d["n_gate"] = P
        out[a] = d
    return out


def main() -> None:
    w24 = _load("2026-07-19-24-collective-wave-poc.py", "wave24")
    lv, E = build_dream_level(w24)
    rows = []

    for unit in ("report", "contributor"):
        for freq in ("W", "M"):
            rng = np.random.default_rng(SEED)
            panels = half_panels(lv, E, freq, unit, rng)
            L = BLOCK[freq]
            for axis in AXES:
                Pn = panels[axis]
                full = np.arange(Pn["n_gate"])
                if Pn["m"].sum(1).mean() < MIN_PERIODS:
                    rows.append({"unit": unit, "axis": axis, "freq": freq,
                                 "n_gate_periods": Pn["n_gate"]})
                    continue
                rec = {"unit": unit, "axis": axis, "freq": freq, "n_gate_periods": Pn["n_gate"],
                       "n_used_mean": round(float(Pn["m"].sum(1).mean()), 1)}
                for mode, tag in (("level", "level"), ("det", "det"), ("fulln", "det_fulln")):
                    rec[f"{tag}_r"] = round(_split_r(Pn, full, slice(None), mode), 3)
                    rec[f"{tag}_SB"] = round(sb(rec[f"{tag}_r"]), 3)

                W, k = len(full), int(np.ceil(len(full) / L))
                draws = np.empty(N_BOOT)
                bo = np.random.default_rng(SEED + 1)
                for j in range(N_BOOT):
                    st = bo.integers(0, W, k)
                    idx = ((st[:, None] + np.arange(L)[None, :]) % W).ravel()[:W]
                    draws[j] = _split_r(Pn, idx, slice(0, BOOT_SPLITS), "det")
                draws = draws[np.isfinite(draws)]
                rec["det_r_lo"] = round(float(np.percentile(draws, 2.5)), 3)
                rec["det_r_hi"] = round(float(np.percentile(draws, 97.5)), 3)
                rec["det_SB_lo"] = round(sb(rec["det_r_lo"]), 3)
                rec["det_SB_hi"] = round(sb(rec["det_r_hi"]), 3)
                rec["latent_floor"] = (round(0.30 / np.sqrt(rec["det_SB"]), 2)
                                       if rec["det_SB"] > 0 else np.nan)
                rec["latent_floor_hi"] = (round(0.30 / np.sqrt(rec["det_SB_lo"]), 2)
                                          if rec["det_SB_lo"] > 0 else np.inf)
                rows.append(rec)

    R = pd.DataFrame(rows).sort_values(["unit", "freq", "det_SB"], ascending=[True, True, True])
    R.to_csv(OUT / "panel_axis_reliability.csv", index=False)

    def table(unit: str, freq: str) -> list[str]:
        T = R[(R.unit == unit) & (R.freq == freq) & R.det_r.notna()]
        out = ["| axis | periods used | level r | level SB | detrended r [95% CI] | detrended SB "
               "[95% CI] | full-n detrend r | latent floor at &#124;r&#124;=0.30 |",
               "|---|--:|--:|--:|--:|--:|--:|--:|"]
        for _, r in T.iterrows():
            # every value in this column is a lower bound: the indicator side is treated as
            # error-free, so the prefix belongs on all of them
            fl = f"&ge;{r.latent_floor:.2f}"
            hi = "unbounded" if not np.isfinite(r.latent_floor_hi) else f"{r.latent_floor_hi:.2f}"
            out.append(f"| {r.axis} | {r.n_used_mean:.0f} | {r.level_r:.3f} | {r.level_SB:.3f} | "
                       f"{r.det_r:.3f} [{r.det_r_lo:.3f}, {r.det_r_hi:.3f}] | "
                       f"{r.det_SB:.3f} [{r.det_SB_lo:.3f}, {r.det_SB_hi:.3f}] | "
                       f"{r.det_fulln_r:.3f} | {fl} (to {hi}) |")
        return out

    Wr = R[(R.unit == "report") & (R.freq == "W") & R.det_r.notna()]
    Wc = R[(R.unit == "contributor") & (R.freq == "W") & R.det_r.notna()]
    Mr = R[(R.unit == "report") & (R.freq == "M") & R.det_r.notna()]
    shift = float((R.det_r - R.det_fulln_r).abs().max())
    lines = [
        "# Panel-matched split-half reliability — the 13 barometer axes (EN)", "",
        f"*Mean over {N_SPLITS} random half-splits. Cohort, period key, n>={GATE['W']}/week gate "
        "and detrend ([1, standardised time, centred log n]) all match "
        "`2026-08-22-12-nonlinear-coupling.py::barometer_axes`. `SB` = Spearman–Brown step-up of "
        "the split-half r to full length, which is the reliability of the series the panel actually "
        f"correlates. Intervals are {N_BOOT} circular block bootstrap draws over periods (block "
        f"{BLOCK['W']} weeks / {BLOCK['M']} months), with every one of {BOOT_SPLITS} split "
        "assignments re-evaluated inside each draw so assignment spread is not counted twice. "
        "`latent floor` = 0.30/sqrt(SB): the noise-free association corresponding to the panel's "
        "monotone observable floor, treating the indicator side as error-free (a lower bound).*",
        "", "## Weekly, split on reports (the panel's own aggregation unit)", "", *table("report", "W"),
        "", "## Weekly, split on contributors (contributor sampling counted as error)", "",
        *table("contributor", "W"),
        "", "## Monthly, split on reports", "", *table("report", "M"),
        "", "## Summary", "",
        f"- **Weekly, report-split, detrended:** {Wr.det_r.min():.3f}–{Wr.det_r.max():.3f} as split "
        f"halves, {Wr.det_SB.min():.3f}–{Wr.det_SB.max():.3f} stepped up to full length.",
        f"- **Weekly, contributor-split, detrended:** {Wc.det_r.min():.3f}–{Wc.det_r.max():.3f} as "
        f"split halves, {Wc.det_SB.min():.3f}–{Wc.det_SB.max():.3f} stepped up.",
        f"- **Monthly, report-split, detrended:** {Mr.det_SB.min():.3f}–{Mr.det_SB.max():.3f} "
        f"stepped up (levels {Mr.level_SB.min():.3f}–{Mr.level_SB.max():.3f}).",
        f"- **Detrend sensitivity:** residualising both halves on the full period's log n rather "
        f"than each half's own moves no axis by more than {shift:.3f} in split-half r.",
        f"- **Latent floor at the panel's |r|≈0.30 monotone floor (weekly, report-split):** "
        f"{Wr.latent_floor.min():.2f} on the best-measured axis to {Wr.latent_floor.max():.2f} on "
        f"the worst; on the {int((~np.isfinite(Wr.latent_floor_hi)).sum())} axes whose interval "
        "reaches zero reliability the correction is not bounded above at this number of periods.",
    ]
    (OUT / "panel_axis_reliability.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
