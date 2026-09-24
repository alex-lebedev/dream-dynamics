"""Round 4 / Q26 + Q29 + Q32 — collective dynamics & geometry of the dream cloud.

(Q26) CRITICAL SLOWING DOWN — does the collective dream show the tipping-point early-warning
    signature (rising variance + rising lag-1 autocorrelation) BEFORE big darkenings of mood?
    Daily EN nightmare/negativity residual (detrended for adoption trend + log-volume); 28-day
    rolling variance & AR(1); Kendall-tau of each EWS indicator over the run-up to top-decile
    "darkening" anchors vs random-anchor permutation null. (Scheffer et al.; the right tool to
    complement the Hurst-null F0016 and prophetic-null F0024.) Reliability-limited -> honest.

(Q29) PERCOLATION — is the collective unconscious ONE continent or an ARCHIPELAGO? Sweep the
    cosine threshold on the EN dream-similarity graph; track the giant-component fraction and
    #components; locate the percolation transition; compare to a feature-shuffled null.

(Q32) THE EXPANDING DREAM UNIVERSE — is dream content-space inflating (diversifying) or
    canalizing (contracting) over 2024->2026? Sample-size-FIXED monthly geometry (mean pairwise
    distance, participation ratio, log-generalized-variance) to remove the volume confound that
    drove the earlier dispersion result (F0016). Trend = true content spread, not sampling.

Aggregate-only. Exploratory/Frontier.

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 USE_TF=0 python3 \
        analyses/2026-07-19-19-collective-dynamics.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from psychohistory import config as C
from psychohistory.dreams.embed_cache import load_or_build

OUT = C.RESULTS / "showcase"
OUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------- (Q26) variance precursor ----
# The random-anchor null could not separate a genuine run-up from endogenous anchor selection, so the
# test here is the observed run-up Kendall-tau vs AR(1)- and phase-randomized SURROGATES run through
# the IDENTICAL detrend->EWS->endogenous-anchor->run-up-tau pipeline. Residual is deseasonalized +
# volume/SE-standardized so low-volume days don't inflate the rolling variance; anchors are
# episode-deduped.
#
# CORRECTION (2026-08-21): this block's own output text used to claim these surrogates "reproduce
# ordinary volatility-clustering". They do not. AR(1) is linear-Gaussian with constant conditional
# variance and phase randomization preserves only the power spectrum; volatility clustering is a
# conditional-variance property. Measured, these surrogates reproduce ~1/4 of the observed
# abs/squared lag-1 autocorrelation. Since rolling variance is exactly what conditional
# heteroskedasticity inflates, that was the load-bearing objection, and it is now tested properly in
# analyses/2026-08-21-04-precursor-hardening.py against GARCH(1,1), stationary-bootstrap and IAAFT
# nulls (estimators re-exported from this file via src/psychohistory/stats/ews.py, so the statistic
# below reproduces exactly). Result: the run-up FAILS BH under the stationary bootstrap at L=28.
# Anything downstream should treat the block bootstrap, not this pair, as the primary null.
def _daily_resid(meta, col, lang="en", min_n=20):
    sub = meta[meta.lang == lang]
    g = sub.groupby(sub.date.dt.normalize())[col]
    d = pd.DataFrame({"m": g.mean(), "sd": g.std(), "n": g.size()})
    d = d[d.n >= min_n]
    idx = d.index
    t = (idx - idx.min()).days.values.astype(float)
    tz = (t - t.mean()) / (t.std() + 1e-9)
    logn = np.log(d.n.values)
    dow = pd.get_dummies(idx.dayofweek).values[:, 1:]          # weekday FE
    doy = idx.dayofyear.values.astype(float)
    harm = np.column_stack([f(2 * np.pi * k * doy / 365.25)    # annual harmonics
                            for k in (1, 2) for f in (np.sin, np.cos)])
    X = np.column_stack([np.ones_like(tz), tz, logn, dow, harm])
    w = np.sqrt(d.n.values)                                     # WLS by volume
    beta = np.linalg.lstsq(X * w[:, None], d.m.values * w, rcond=None)[0]
    se = d.sd.values / np.sqrt(d.n.values)
    resid = (d.m.values - X @ beta) / (se + 1e-9)               # SE-standardized
    full = pd.date_range(idx.min(), idx.max(), freq="D")
    s = pd.Series(resid, index=idx).reindex(full)
    return s.interpolate(limit_direction="both").values, s.notna().values


def _ews(x, win=28):
    var = pd.Series(x).rolling(win, min_periods=win // 2).var().values
    def ar1(a):
        a = a[np.isfinite(a)]
        return np.corrcoef(a[:-1], a[1:])[0, 1] if len(a) > 5 else np.nan
    ac = pd.Series(x).rolling(win, min_periods=win // 2).apply(ar1, raw=True).values
    return var, ac


def _tau_runup(pos, arr, runup):
    seg = arr[max(0, pos - runup):pos]
    seg = seg[np.isfinite(seg)]
    if len(seg) < 10:
        return np.nan
    return stats.kendalltau(np.arange(len(seg)), seg).correlation


def _dedupe(pos, gap):
    keep, last = [], -10 ** 9
    for p in np.sort(pos):
        if p - last >= gap:
            keep.append(p); last = p
    return np.array(keep)


def _pipeline_taus(x, win, runup):
    var, ac = _ews(x, win)
    fwd = (pd.Series(x).shift(-14).rolling(14).mean() - pd.Series(x).rolling(7).mean()).values
    fin = np.isfinite(fwd)
    anc = _dedupe(np.where(fin & (fwd >= np.nanquantile(fwd, 0.90)))[0], runup)
    return ({k: np.nanmean([_tau_runup(a, arr, runup) for a in anc])
             for k, arr in [("variance", var), ("ar1", ac)]}, len(anc))


def _ar1_surrogate(x, rng):
    x = x - x.mean()
    phi = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    e = rng.standard_normal(len(x)) * np.std(x) * np.sqrt(max(1 - phi ** 2, 1e-6))
    y = np.empty(len(x)); y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = phi * y[t - 1] + e[t]
    return y


def _phase_surrogate(x, rng):
    X = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, len(X)); ph[0] = 0.0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(x)) + x.mean()


def csd_test(meta, col="nightmare_index", win=28, runup=42, seed=0, n_surr=400):
    rng = np.random.default_rng(seed)
    x, mask = _daily_resid(meta, col)
    obs, n_anc = _pipeline_taus(x, win, runup)
    surr = {"variance": [], "ar1": []}
    for gen in (_ar1_surrogate, _phase_surrogate):
        for _ in range(n_surr):
            t, _ = _pipeline_taus(gen(x, rng), win, runup)
            surr["variance"].append(t["variance"]); surr["ar1"].append(t["ar1"])
    out = {}
    for ind in ("variance", "ar1"):
        s = np.array(surr[ind]); s = s[np.isfinite(s)]
        out[ind] = {"obs_tau": float(obs[ind]), "surr_mean": float(s.mean()),
                    "surr_p": float((1 + np.sum(s >= obs[ind])) / (len(s) + 1))}
    return out, {"n_anchors": int(n_anc), "n_days": int(mask.sum())}, x, win


# --------------------------------------------------------- (Q29) percolation ----
def _percolation_once(X, thetas):
    S = X @ X.T
    np.fill_diagonal(S, 0.0)
    N = len(X)
    rows = []
    for th in thetas:
        ncomp, labels = connected_components(csr_matrix(S > th), directed=False)
        sizes = np.bincount(labels)
        rows.append({"theta": float(th), "giant_frac": float(sizes.max() / N),
                     "n_components": int(ncomp), "mean_degree": float((S > th).sum() / N),
                     "n_singletons": int((sizes == 1).sum())})
    return pd.DataFrame(rows)


def percolation(emb, meta, lang="en", n=5000, seeds=(0, 1, 2), shuffled=False):
    # de-censored grid extended well below 0.30 (critic: prior 0.30 floor was an artifact) + multi-seed
    thetas = np.round(np.arange(0.02, 0.90, 0.03), 2)
    idx_all = np.where(meta.lang.values == lang)[0]
    parts = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        idx = rng.choice(idx_all, min(n, len(idx_all)), replace=False)
        X = emb[idx].astype(np.float32)
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        if shuffled:
            X = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        parts.append(_percolation_once(X, thetas).set_index("theta"))
    mean = sum(parts) / len(parts)
    return mean.reset_index(), min(n, len(idx_all))


# ------------------------------------------------ (Q32) expanding universe ----
def _circular_shift_slope_p(y, n_perm=5000, seed=0):
    """Autocorrelation-valid trend p: observed OLS slope vs circularly-shifted-series slopes."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    t = np.arange(len(y), dtype=float)
    b_obs = np.polyfit(t, y, 1)[0]
    null = np.array([np.polyfit(t, np.roll(y, int(k)), 1)[0]
                     for k in rng.integers(1, len(y), n_perm)])
    return float(b_obs), float((1 + np.sum(np.abs(null) >= abs(b_obs))) / (n_perm + 1))


def expanding_universe(emb, meta, lang="en", k_per_month=120, reps=25, seed=0, m_pca=15,
                       restrict_users=None):
    rng = np.random.default_rng(seed)
    sub = meta[meta.lang == lang].copy()
    if restrict_users is not None:
        sub = sub[sub.userID.isin(restrict_users)]
    sub = sub.assign(month=sub.date.dt.to_period("M"))
    E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    rows = []
    for mo, g in sub.groupby("month"):
        pos = g.index.values
        if len(pos) < k_per_month:
            continue
        mpd, pr, lgv = [], [], []
        for _ in range(reps):
            s = rng.choice(pos, k_per_month, replace=False)
            V = E[s]
            cen = V.mean(0)
            mpd.append(1 - (V @ cen / (np.linalg.norm(cen) + 1e-9)).mean())
            Vc = V - V.mean(0)
            lam = np.linalg.svd(Vc, compute_uv=False) ** 2
            lam = lam / lam.sum()
            pr.append((lam.sum() ** 2) / (lam ** 2).sum())
            lgv.append(np.sum(np.log(lam[:m_pca] + 1e-12)))
        rows.append({"month": str(mo), "n": len(pos), "mean_pairwise_dist": np.mean(mpd),
                     "participation_ratio": np.mean(pr), "log_gen_var": np.mean(lgv)})
    D = pd.DataFrame(rows)
    trends = {}
    for c in ["mean_pairwise_dist", "participation_ratio", "log_gen_var"]:
        b, p = _circular_shift_slope_p(D[c].values)
        trends[c] = {"slope_per_month": b, "p_circshift": p,
                     "r": float(stats.pearsonr(np.arange(len(D)), D[c].values)[0])}
    return D, trends


def _stable_panel_users(meta, lang="en", k_per_month=120):
    """Users active in BOTH the first and last third of the dense window (composition control)."""
    sub = meta[meta.lang == lang]
    t0, t1 = sub.date.min(), sub.date.max()
    cut1 = t0 + (t1 - t0) / 3
    cut2 = t1 - (t1 - t0) / 3
    early = set(sub[sub.date <= cut1].userID)
    late = set(sub[sub.date >= cut2].userID)
    return early & late


def main():
    meta, emb = load_or_build()
    L = ["# Round 4 · Q26 + Q29 + Q32 — collective dynamics & geometry", "",
         "*DreamSeer dense window, EN. Aggregate-only. Reliability-limited dynamical claims are "
         "flagged exploratory.*", ""]

    # Q26
    L += ["## (Q26) Variance precursor — does variance rise faster than a linear-dependence surrogate?"]
    try:
        csd_fig = None
        for col in ["nightmare_index", "negativity"]:
            res, info, x, win = csd_test(meta, col=col)
            L.append(f"- **{col}** (n_days={info['n_days']}, {info['n_anchors']} episode-deduped "
                     f"darkening anchors): variance run-up τ={res['variance']['obs_tau']:+.3f} vs "
                     f"**AR(1)+phase surrogates** (mean {res['variance']['surr_mean']:+.3f}, "
                     f"**p={res['variance']['surr_p']:.3f}**); AR(1) run-up τ={res['ar1']['obs_tau']:+.3f} "
                     f"(surrogate p={res['ar1']['surr_p']:.3f}).")
            csd_fig = (x, win)
        L += ["  Test = observed run-up τ vs **AR(1)/phase-randomized surrogates** through the identical "
              "pipeline, so the null absorbs the anchor-selection rule (the prior random-anchor null "
              "could not). **These surrogates preserve LINEAR serial dependence only — NOT volatility "
              "clustering**, contrary to what this line used to say; they reproduce ~1/4 of the observed "
              "abs/squared lag-1 autocorrelation. Rolling variance is exactly what conditional "
              "heteroskedasticity inflates, so the volatility-preserving nulls in "
              "`analyses/2026-08-21-04-precursor-hardening.py` are the ones that matter: the run-up "
              "**FAILS BH under a stationary block bootstrap at L=28** (q=.160). The autocorrelation "
              "indicator fails everywhere, so this is a *variance precursor*, not critical slowing "
              "down. Also reliability-limited (daily nightmare split-half ~.17) and only 12 episodes.", ""]
    except Exception as e:  # keep the batch alive
        L += [f"  (CSD failed: {e})", ""]
        csd_fig = None

    # Q29 — de-censored grid; compare real vs feature-shuffled giant-fraction curves at fixed thresholds
    perc, N = percolation(emb, meta)
    perc_sh, _ = percolation(emb, meta, shuffled=True)
    perc.to_csv(OUT / "collective_percolation.csv", index=False)
    def at(df, th):
        return df.iloc[(df.theta - th).abs().argmin()]
    r5, s5 = at(perc, 0.5), at(perc_sh, 0.5)
    r6, s6 = at(perc, 0.6), at(perc_sh, 0.6)
    L += ["## (Q29) Percolation — one continent or an archipelago?",
          f"- N={N} EN dreams; grid de-censored to θ∈[0.02,0.9]. Dream-space forms a **giant component "
          f"at near-zero θ** (transition below 0.02) → densely connected, *not* a modular archipelago.",
          f"- At θ=0.50: giant {r5.giant_frac*100:.0f}% ({int(r5.n_singletons)} island-singletons) vs "
          f"feature-shuffled {s5.giant_frac*100:.0f}% ({int(s5.n_singletons)}); at θ=0.60: "
          f"{r6.giant_frac*100:.0f}% ({int(r6.n_singletons)}) vs {s6.giant_frac*100:.0f}% "
          f"({int(s6.n_singletons)}).",
          ("- Real dreams retain a **larger giant component + fewer islands** than the feature-shuffled "
           "null at the same θ → genuine (not random) local structure holds the continent together."
           if r6.giant_frac > s6.giant_frac else
           "- Real vs shuffled giant-fractions are similar → connectivity is largely diffuse/dense-blob."),
          ""]

    # Q32 — full pool vs stable-panel (composition control) + circular-shift trend p
    D, trends = expanding_universe(emb, meta)
    D.to_csv(OUT / "collective_expanding_universe.csv", index=False)
    panel = _stable_panel_users(meta)
    Dp, trends_p = expanding_universe(emb, meta, restrict_users=panel)
    Dp.to_csv(OUT / "collective_expanding_universe_panel.csv", index=False)
    L += ["## (Q32) Expanding/contracting dream universe (sample-size-fixed monthly geometry)",
          f"- **Full pool** (n months={len(D)}), circular-shift trend p:"]
    for c, tr in trends.items():
        direction = "inflating" if tr["slope_per_month"] > 0 else "contracting"
        L.append(f"  - {c}: slope={tr['slope_per_month']:+.4f}/mo (r={tr['r']:+.2f}, "
                 f"p_circshift={tr['p_circshift']:.3f}) → {direction}.")
    L += [f"- **Stable panel** ({len(panel)} users active across the window, {len(Dp)} months) — "
          "isolates *within-cohort* change from composition drift:"]
    for c, tr in trends_p.items():
        direction = "inflating" if tr["slope_per_month"] > 0 else "contracting"
        L.append(f"  - {c}: slope={tr['slope_per_month']:+.4f}/mo (r={tr['r']:+.2f}, "
                 f"p_circshift={tr['p_circshift']:.3f}) → {direction}.")
    L += ["  If contraction **persists in the stable panel**, it is within-cohort canalization; if it "
          "**weakens/vanishes**, the full-pool contraction is user-composition drift (cf F0031). "
          "Trend p via circular-shift permutation (autocorrelation-valid).", ""]

    fig_dynamics(csd_fig, perc, perc_sh, D)
    L += ["**Figures:** 28_collective_csd.png · 29_dream_percolation.png · 30_expanding_universe.png",
          "", "*Verdict tier: Frontier / exploratory pending methodology-critic.*"]
    (OUT / "collective_dynamics.md").write_text("\n".join(L))
    print("\n".join(L))
    print("\n[q26/q29/q32] wrote", OUT / "collective_dynamics.md")


def fig_dynamics(csd_fig, perc, perc_sh, D):
    if csd_fig is not None:
        x, win = csd_fig
        var, ac = _ews(x, win)
        fig, ax = plt.subplots(figsize=(9, 4))
        t = np.arange(len(x))
        ax.plot(t, (var - np.nanmean(var)) / np.nanstd(var), label="rolling variance (z)", color="#4C72B0")
        ax.plot(t, (ac - np.nanmean(ac)) / np.nanstd(ac), label="rolling AR(1) (z)", color="#C1443C", alpha=0.8)
        ax.set_xlabel("day index"); ax.set_title("Critical-slowing-down indicators (EN nightmare residual)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / "28_collective_csd.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(perc.theta.to_numpy(), perc.giant_frac.to_numpy(), "-o", ms=3, label="dreams", color="#C1443C")
    ax.plot(perc_sh.theta.to_numpy(), perc_sh.giant_frac.to_numpy(), "-o", ms=3,
            label="feature-shuffled", color="#7a7a7a")
    ax.set_xlabel("cosine threshold θ"); ax.set_ylabel("giant-component fraction")
    ax.set_title("Percolation of dream-space")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "29_dream_percolation.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    mm = pd.to_datetime(D.month).to_numpy()
    pr_z = ((D.participation_ratio - D.participation_ratio.mean()) / D.participation_ratio.std()).to_numpy()
    mpd_z = ((D.mean_pairwise_dist - D.mean_pairwise_dist.mean()) / D.mean_pairwise_dist.std()).to_numpy()
    ax.plot(mm, pr_z, "-o", ms=3, label="participation ratio (z)", color="#4C72B0")
    ax.plot(mm, mpd_z, "-o", ms=3, label="mean pairwise dist (z)", color="#C1443C")
    ax.set_title("Expanding dream universe? (sample-size-fixed)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "30_expanding_universe.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
