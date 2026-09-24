"""Hardening the nightmare-variance precursor (§3.5) against four reviewer objections.

The published null for the variance run-up is an AR(1)-plus-phase-randomized surrogate.
Two independent external reviews observed that such surrogates are linear-Gaussian and
spectrum-preserving: they reproduce *linear* serial dependence but not conditional
heteroskedasticity, so the manuscript's claim that they "preserve volatility clustering"
was not earned by the construction. Rolling variance is precisely the statistic that
conditional heteroskedasticity inflates, so the objection is substantive rather than
verbal, and it sits under the paper's most exposed finding.

This script settles it empirically and answers three adjacent objections:

  (A) VOLATILITY-PRESERVING NULLS. Re-run the identical detrend -> EWS -> endogenous-anchor
      -> run-up-tau pipeline against nulls that do preserve time-varying volatility: an
      AR(1)-GARCH(1,1) surrogate (parametric conditional variance), a stationary block
      bootstrap at two mean block lengths (preserves local dependence including local
      volatility clumps) and IAAFT (preserves the spectrum *and* the amplitude
      distribution). Each family carries a diagnostic stating what it actually preserved:
      the lag-1 autocorrelation of the raw, absolute and squared series.

  (B) LEAVE-ONE-EPISODE-OUT. Whether the mean Kendall tau is carried by one episode.

  (C) WITHIN- VERSUS BETWEEN-PERSON. Whether the aggregate variance run-up is carried by
      individuals becoming more volatile or by the population dispersing.

  (D) COMPOSITION. Whether report volume or contributor count shows the same run-up,
      which would make the precursor a sampling artifact rather than an affective one.

The estimators come from `psychohistory.stats.ews`, promoted verbatim in behaviour from
the committed script that produced the published numbers. This script asserts that the
module reproduces the published observed statistic, episode count and day count, so the
equivalence of the pipeline is checked rather than assumed.

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 USE_TF=0 python3 \
        analyses/2026-08-21-04-precursor-hardening.py
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats

from psychohistory import config as C
from psychohistory.stats import ews

OUT = C.RESULTS / "tables"
OUT.mkdir(parents=True, exist_ok=True)

WIN, RUNUP, N_SURR, SEED = 28, 42, 1000, 0


def _cell_rng(family, series):
    """Deterministic per-cell stream, so a p-value does not depend on loop order.

    Uses blake2b rather than the builtin ``hash``. ``hash()`` of a tuple containing strings is
    salted per interpreter process (PEP 456), so the earlier version of this function produced a
    different stream on every run and the p-values it emitted were not reproducible by anyone,
    including us. A surrogate p-value that moves between runs is not a p-value.
    """
    key = f"{SEED}|{family}|{series}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))

# Published values from the manuscript's Section 3.5, reproduced as a pipeline check.
PUBLISHED = {"tau_variance": 0.236, "tau_ar1": 0.054, "n_episodes": 12, "n_days": 601}


def check_pipeline(x, mask):
    obs, n_anc = ews.pipeline_taus(x, WIN, RUNUP)
    got = {"tau_variance": round(float(obs["variance"]), 3),
           "tau_ar1": round(float(obs["ar1"]), 3),
           "n_episodes": int(n_anc), "n_days": int(mask.sum())}
    ok = all(abs(got[k] - PUBLISHED[k]) < (0.002 if "tau" in k else 0.5) for k in PUBLISHED)
    print(f"[pipeline check] published={PUBLISHED} reproduced={got} -> "
          f"{'IDENTICAL' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("pipeline does not reproduce the published statistic; aborting")
    return obs


def _bh(pvals):
    """Benjamini-Hochberg q-values, order preserved."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, p[i] * m / (rank + 1))
        q[i] = running
    return q


def full_family_under_each_null(meta):
    """Re-run the whole pre-stated four-member family under every surrogate family.

    The manuscript corrects across {variance, AR(1)} x {nightmare, negativity}. Changing
    the null changes every member's p-value, so the correction has to be recomputed
    inside each null rather than carried over from the published one. Each cell draws from
    its own deterministically seeded stream, so a p-value does not depend on loop order.

    Alongside every cell, the diagnostic states what that null actually preserved: the
    lag-1 autocorrelation of the raw series (linear serial dependence) and of the absolute
    and squared series (volatility clustering).
    """
    series = {col: ews.daily_residual(meta, col)[0]
              for col in ("nightmare_index", "negativity")}
    garch = {col: ews.garch_surrogate_factory(x) for col, x in series.items()}

    def gens_for(name, col):
        return {
            "ar1_plus_phase (published)": ([ews.ar1_surrogate, ews.phase_surrogate], "spectrum"),
            "ar1_garch11": ([garch[col][0]], "volatility (parametric)"),
            "stationary_bootstrap_L28": ([ews.stationary_bootstrap_factory(28)], "volatility (block)"),
            "stationary_bootstrap_L14": ([ews.stationary_bootstrap_factory(14)], "volatility (block)"),
            "iaaft": ([ews.iaaft_surrogate], "spectrum + amplitudes"),
        }[name]

    names = ["ar1_plus_phase (published)", "ar1_garch11",
             "stationary_bootstrap_L28", "stationary_bootstrap_L14", "iaaft"]
    rows = []
    for name in names:
        cells, pvals = [], []
        for col, x in series.items():
            obs, _n = ews.pipeline_taus(x, WIN, RUNUP)
            obs_diag = ews.volatility_diagnostic(x)
            gens, kind = gens_for(name, col)
            rng = _cell_rng(name, col)
            tv, ta, diags = [], [], []
            per_gen = max(N_SURR // len(gens), 1)
            for gen in gens:
                for _ in range(per_gen):
                    y = gen(x, rng)
                    t, _k = ews.pipeline_taus(y, WIN, RUNUP)
                    tv.append(t["variance"])
                    ta.append(t["ar1"])
                    diags.append(ews.volatility_diagnostic(y))
            for ind, samp in (("variance", tv), ("ar1", ta)):
                s = np.asarray(samp, float)
                s = s[np.isfinite(s)]
                p = float((1 + np.sum(s >= obs[ind])) / (len(s) + 1))
                pvals.append(p)
                cells.append({
                    "surrogate_family": name, "preserves": kind, "series": col,
                    "indicator": ind, "obs_tau": round(float(obs[ind]), 4),
                    "n_surrogates": int(len(s)),
                    "surr_mean_tau": round(float(s.mean()), 4),
                    "surr_q95_tau": round(float(np.quantile(s, 0.95)), 4),
                    "obs_raw_ar1": round(obs_diag["raw_ar1"], 4),
                    "surr_raw_ar1_mean": round(float(np.mean([d["raw_ar1"] for d in diags])), 4),
                    "obs_abs_ar1": round(obs_diag["abs_ar1"], 4),
                    "surr_abs_ar1_mean": round(float(np.mean([d["abs_ar1"] for d in diags])), 4),
                    "obs_sq_ar1": round(obs_diag["sq_ar1"], 4),
                    "surr_sq_ar1_mean": round(float(np.mean([d["sq_ar1"] for d in diags])), 4),
                })
        qvals = _bh(pvals)
        for cell, p, q in zip(cells, pvals, qvals):
            cell["p"] = round(p, 4)
            cell["q_bh_within_family"] = round(float(q), 4)
            rows.append(cell)
    garch_params = {col: garch[col][1] for col in series}
    return pd.DataFrame(rows), garch_params


# --------------------------------------------------- (B) leave-one-episode-out ----
def episode_jackknife(x):
    var, _ac = ews.ews_indicators(x, WIN)
    anc = ews.episode_anchors(x, RUNUP)
    per_ep = np.array([ews.runup_tau(a, var, RUNUP) for a in anc], float)
    rows = [{"dropped_episode_index": i, "dropped_day_index": int(a),
             "dropped_episode_tau": round(float(per_ep[i]), 4),
             "mean_tau_without_it": round(float(np.nanmean(np.delete(per_ep, i))), 4)}
            for i, a in enumerate(anc)]
    return pd.DataFrame(rows), per_ep, anc


# ------------------------------------------- (C)/(D) decomposition & composition ----
def within_between(meta, anchors):
    """Split each day's nightmare dispersion into between- and within-contributor parts.

    Between = variance of per-contributor daily means (the population dispersing).
    Within  = pooled variance within contributors who filed more than once (individuals
    becoming noisier). If the run-up is carried by the between term, the moving quantity
    is collective; if by volume or contributor count, it is compositional.

    Each component is put through the aggregate's own covariate adjustment and then scored on two
    statistics against its own surrogate null (see the comments below for why both are needed).
    Read the result as a bound, not a refutation: the between-contributor variance is estimated
    from roughly 25 contributors on a median day, and that estimation noise attenuates a Kendall
    trend toward zero, so a genuine desynchronization signal would be biased downward here.
    """
    sub = meta[meta.lang == "en"].copy()
    sub["day"] = sub["date"].dt.normalize()
    rows = []
    for day, g in sub.groupby("day"):
        if len(g) < 20:
            continue
        by_user = g.groupby("userID")["nightmare_index"]
        means, counts = by_user.mean(), by_user.size()
        var_u = by_user.var(ddof=1)
        num = float(np.nansum((var_u.fillna(0.0).values) * (counts.values - 1)))
        den = max(int(len(g) - len(means)), 1)
        rows.append({"day": day, "n_reports": int(len(g)), "n_users": int(len(means)),
                     "between_var": float(means.var(ddof=1)) if len(means) > 1 else np.nan,
                     "within_var": num / den if len(g) > len(means) else np.nan,
                     "total_var": float(g["nightmare_index"].var(ddof=1))})
    D = pd.DataFrame(rows).set_index("day").sort_index()
    D = D.reindex(pd.date_range(D.index.min(), D.index.max(), freq="D")).interpolate(
        limit_direction="both")
    idx = pd.DatetimeIndex(D.index)

    out = []
    for col in ("between_var", "within_var", "total_var", "n_reports", "n_users"):
        # Matched preprocessing: same covariates and weighting as the aggregate residual.
        r = ews.covariate_residual(D[col].values, D["n_reports"].values, idx)

        # Two statistics, because they answer different questions and only one is like-for-like
        # with the aggregate. "instability" is the aggregate's own indicator (rolling variance of
        # the residual) and is the only one comparable to the +0.237 headline. "level" is a rolling
        # mean, which is what the desynchronization reading actually predicts: dispersion should
        # *rise*, not merely become erratic. Reporting only one of the two would answer a question
        # nobody asked while appearing to settle the one they did.
        stats_by_kind = {"instability": ews.ews_indicators(r, WIN)[0],
                         "level": pd.Series(r).rolling(WIN, min_periods=WIN // 2).mean().values}

        for kind, arr in stats_by_kind.items():
            taus = [ews.runup_tau(a, arr, RUNUP) for a in anchors if a < len(arr)]
            taus = [t for t in taus if np.isfinite(t)]
            obs_t = float(np.mean(taus)) if taus else np.nan

            # Own surrogate null, so the number is calibrated rather than eyeballed against the
            # aggregate's. Without this, "+0.06 is far below +0.24" asserts a yardstick that does
            # not exist: under the published null the aggregate's own 95th percentile is +0.11.
            rng = _cell_rng("within_between", f"{col}|{kind}")
            surr = []
            for gen in (ews.ar1_surrogate, ews.phase_surrogate):
                for _ in range(N_SURR // 2):
                    y = gen(r, rng)
                    a2 = (ews.ews_indicators(y, WIN)[0] if kind == "instability"
                          else pd.Series(y).rolling(WIN, min_periods=WIN // 2).mean().values)
                    tt = [ews.runup_tau(a, a2, RUNUP) for a in anchors if a < len(a2)]
                    tt = [t for t in tt if np.isfinite(t)]
                    if tt:
                        surr.append(float(np.mean(tt)))
            s = np.asarray(surr, float)
            s = s[np.isfinite(s)]
            p = float((1 + np.sum(s >= obs_t)) / (len(s) + 1)) if len(s) else np.nan
            out.append({"component": col, "statistic": kind, "n_episodes_scored": len(taus),
                        "mean_runup_tau": round(obs_t, 4),
                        "surr_mean_tau": round(float(s.mean()), 4) if len(s) else np.nan,
                        "surr_q95_tau": round(float(np.quantile(s, 0.95)), 4) if len(s) else np.nan,
                        "p": round(p, 4) if np.isfinite(p) else np.nan})
    return pd.DataFrame(out), D


def main() -> None:
    meta = pd.read_parquet(C.PROCESSED / "dreams" / "dreamseer_dream_level.parquet")
    meta["date"] = pd.to_datetime(meta["date"])

    x, mask = ews.daily_residual(meta, "nightmare_index")
    obs = check_pipeline(x, mask)

    fam, garch_params = full_family_under_each_null(meta)
    fam.to_csv(OUT / "precursor_family_by_null.csv", index=False)
    print("\n=== (A) what each null preserves, and whether the run-up clears it ===")
    print(fam[fam.indicator == "variance"][
        ["surrogate_family", "preserves", "series", "obs_tau", "surr_mean_tau",
         "surr_q95_tau", "p", "q_bh_within_family",
         "obs_abs_ar1", "surr_abs_ar1_mean", "obs_sq_ar1", "surr_sq_ar1_mean"]
    ].to_string(index=False))
    print("\n--- the pre-stated four-member family, corrected inside each null ---")
    print(fam[["surrogate_family", "series", "indicator", "obs_tau", "p",
               "q_bh_within_family"]].to_string(index=False))
    print(f"\nGARCH(1,1) fits: {json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in garch_params.items()})}")

    jack, per_ep, anc = episode_jackknife(x)
    jack.to_csv(OUT / "precursor_episode_jackknife.csv", index=False)
    print("\n=== (B) leave-one-episode-out ===")
    print(f"per-episode tau: {np.round(per_ep, 3).tolist()}")
    print(f"positive in {int(np.nansum(per_ep > 0))} of {len(per_ep)} episodes; "
          f"jackknifed mean tau ranges {jack.mean_tau_without_it.min():+.3f} to "
          f"{jack.mean_tau_without_it.max():+.3f}")

    wb, _D = within_between(meta, anc)
    wb.to_csv(OUT / "precursor_within_between.csv", index=False)
    print("\n=== (C)/(D) variance decomposition and composition ===")
    print(wb.to_string(index=False))

    (OUT / "precursor_hardening_summary.json").write_text(json.dumps({
        "observed_tau_variance": float(obs["variance"]),
        "observed_tau_ar1": float(obs["ar1"]),
        "n_days": int(mask.sum()),
        "n_episodes": int(len(anc)),
        "garch_params": garch_params,
        "family_by_null": fam.to_dict(orient="records"),
        "episode_taus": [None if not np.isfinite(v) else float(v) for v in per_ep],
        "n_positive_episodes": int(np.nansum(per_ep > 0)),
        "jackknife_min_mean_tau": float(jack.mean_tau_without_it.min()),
        "jackknife_max_mean_tau": float(jack.mean_tau_without_it.max()),
        "decomposition": wb.to_dict(orient="records"),
    }, indent=2))
    print(f"\n[written] {OUT / 'precursor_family_by_null.csv'}")
    print(f"[written] {OUT / 'precursor_episode_jackknife.csv'}")
    print(f"[written] {OUT / 'precursor_within_between.csv'}")


if __name__ == "__main__":
    main()
