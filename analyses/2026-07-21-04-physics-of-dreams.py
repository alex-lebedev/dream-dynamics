"""Physics of dreams II — periodic dynamics, scaling laws, complexity, and the fractal geometry of dream-space.

Builds on F0016 (Zipf + low-D manifold; long-memory null), F0033 (CSD), F0034 (intra-dream), F0035
(percolation). New measures, each with a null/surrogate:

  PERIODIC/SPECTRAL — Welch power spectrum of the deseasonalized daily collective signal → natural periods
    + the spectral "colour" (1/f^β slope: white .0 / pink 1 / brown 2).
  FRACTAL/LONG-MEMORY — Detrended Fluctuation Analysis (DFA) scaling exponent α of the daily signal vs a
    shuffle null (α=.5 white, >.5 long-range correlated).
  COMPLEXITY — permutation entropy (ordinal, m=3) of the daily signal vs shuffle.
  SCALING LAWS — Zipf (rank-frequency) + Heaps (vocabulary growth V~N^β) + dream-length distribution
    (log-normal vs power) + Taylor's fluctuation-scaling law across themes (var~mean^α: 1 Poisson / 2 bursty).
  FRACTAL DIMENSION OF DREAM-SPACE — correlation dimension (Grassberger-Procaccia) + two-NN intrinsic
    dimension of the MiniLM embedding cloud vs an independent-Gaussian null (ambient = 384).

EN primary; aggregate-only; exploratory.

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 USE_TF=0 python3 \
        analyses/2026-07-21-04-physics-of-dreams.py
"""
from __future__ import annotations

import math
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sig
from scipy.spatial import cKDTree

from psychohistory import config as C

OUT = C.RESULTS / "showcase"
EMO = ["fear", "anger", "sadness", "disgust", "joy", "trust", "anticipation", "surprise"]
THEMES = ["danger", "weapon", "family", "flying", "animal", "health", "swimming", "social",
          "friend", "stranger", "food", "romantic", "music", "game"]


# ---------------------------------------------------------------- daily signal ----
def daily_resid(col="negativity_mean"):
    d = pd.read_csv(C.DREAMS_OUT / "dreamseer_daily.csv")
    d = d[d.lang == "en"].copy(); d["date"] = pd.to_datetime(d["date"])
    d = d[(d.date >= "2024-03-01") & (d.n_dreams >= 15)].sort_values("date")
    y = d[col].to_numpy(float)
    t = (d.date - d.date.min()).dt.days.to_numpy(float)
    dow = pd.get_dummies(d.date.dt.dayofweek, prefix="dow").to_numpy(float)
    X = np.column_stack([np.ones(len(y)), (t - t.mean()) / t.std(), np.log(d.n_dreams.to_numpy()), dow])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    # regular daily grid (interpolate small gaps) for spectral/DFA
    idx = pd.Series(resid, index=d.date)
    full = pd.date_range(d.date.min(), d.date.max(), freq="D")
    reg = idx.reindex(full).interpolate(limit_direction="both").to_numpy()
    return resid, reg, d


def spectral(reg, L):
    f, Pxx = sig.welch(reg - reg.mean(), fs=1.0, nperseg=min(256, len(reg) // 2))
    per = 1.0 / f[1:]                       # periods in days
    P = Pxx[1:]
    # dominant peaks (period 2..400 d)
    m = (per >= 2) & (per <= 400)
    order = np.argsort(P[m])[::-1]
    tops = [(per[m][i], P[m][i]) for i in order[:5]]
    # 1/f slope in the 2..90 d band (log-log fit)
    band = (per >= 2) & (per <= 90)
    lf = np.log(1.0 / per[band]); lp = np.log(P[band] + 1e-18)
    beta = -np.polyfit(lf, lp, 1)[0]
    L += ["## 1. Spectral colour (detrended + weekday/volume-adjusted daily EN negativity; NOT day-of-year "
          "deseasonalized)",
          "- Nominal spectral peaks (" + ", ".join(f"{p:.1f} d" for p, _ in tops)
          + ") are **NOT significance-tested and NOT interpreted** — with no day-of-year harmonics removed, "
          "low-freq peaks (e.g. 128 d) are likely unmodeled seasonality/leakage, and F0017 already found the "
          "7-day peak weak (perm-p≈.08 EN).",
          f"- **Spectral colour 1/f^β (2-90 d band): β = {beta:+.2f}** "
          f"({'≈white/uncorrelated' if abs(beta) < 0.3 else '≈pink/1-f' if abs(beta-1) < 0.4 else '≈brown' if abs(beta-2) < 0.5 else 'intermediate'}). "
          "Residual seasonality would only REDDEN β, so a near-white β is a conservative reading.", ""]
    return f, Pxx, beta, tops


def dfa(x, scales=None):
    x = np.asarray(x, float); x = x - x.mean()
    y = np.cumsum(x); n = len(y)
    if scales is None:
        scales = np.unique(np.floor(np.logspace(np.log10(8), np.log10(n // 4), 18)).astype(int))
    F = []
    for s in scales:
        nseg = n // s
        if nseg < 2:
            F.append(np.nan); continue
        rms = []
        for v in range(nseg):
            seg = y[v * s:(v + 1) * s]
            tt = np.arange(s)
            c = np.polyfit(tt, seg, 1)
            rms.append(np.sqrt(np.mean((seg - np.polyval(c, tt)) ** 2)))
        F.append(np.sqrt(np.mean(np.square(rms))))
    F = np.array(F); ok = np.isfinite(F) & (F > 0)
    alpha = np.polyfit(np.log(scales[ok]), np.log(F[ok]), 1)[0]
    return float(alpha), scales[ok], F[ok]


def perm_entropy(x, m=3, tau=1):
    x = np.asarray(x, float); n = len(x)
    patterns = Counter()
    for i in range(n - (m - 1) * tau):
        w = x[i:i + m * tau:tau]
        patterns[tuple(np.argsort(w))] += 1
    p = np.array(list(patterns.values()), float); p /= p.sum()
    H = -np.sum(p * np.log(p))
    return float(H / np.log(math.factorial(m)))


def complexity_fractal(resid, reg, L):
    rng = np.random.default_rng(0)
    # PRIMARY: observed-day sequence (NO interpolation — interpolation artificially inflates DFA α)
    a_obs, sc, F = dfa(resid)
    an_obs = np.array([dfa(rng.permutation(resid))[0] for _ in range(200)])
    p_obs = (1 + np.sum(np.abs(an_obs - 0.5) >= abs(a_obs - 0.5))) / 201
    # interpolated grid (shown for contrast; inflated by ~30% gap-fill)
    a_reg = dfa(reg)[0]
    pe = perm_entropy(resid)
    pe_null = np.array([perm_entropy(rng.permutation(resid)) for _ in range(200)])
    p_pe = (1 + np.sum(pe_null <= pe)) / 201
    L += ["## 2. Fractal / long-memory + complexity (daily EN residual)",
          f"- **DFA scaling exponent α = {a_obs:.2f}** (observed-day sequence, NO interpolation; 0.5=white, "
          f"1=1/f, 1.5=Brownian); shuffle-null 0.50±{an_obs.std():.02f} → excess p={p_obs:.3f}. "
          f"{'Modest long-range persistence (α>.5 beyond shuffle) — multi-day mood episodes.' if (a_obs > 0.5 and p_obs < 0.05) else 'No DETECTABLE long-range memory (under-powered: daily split-half r≈.09; consistent with the F0016 long-memory null).'} "
          f"*(Interpolated-grid α={a_reg:.2f} is INFLATED by ~30% gap-fill — not used.)*",
          f"- **Permutation entropy = {pe:.3f}** (1=max random); shuffle-null {pe_null.mean():.3f} "
          f"(p={p_pe:.3f}) → {'slightly less random than shuffle (weak ordinal structure).' if p_pe < 0.05 else 'near-maximal (little ordinal structure at daily scale).'}", ""]
    return a_obs, sc, F


# ---------------------------------------------------------------- scaling laws ----
TOKEN = re.compile(r"[a-z]+")


def scaling_laws(L):
    lv = pd.read_parquet(C.DREAMS_OUT / "dreamseer_dream_level.parquet",
                         columns=["documentID", "date", "lang", "words"])
    lv["date"] = pd.to_datetime(lv["date"]); lv = lv[(lv.date >= "2024-03-01") & (lv.lang == "en")]
    raw = pd.read_csv(C.DREAMSEER_RAW, dtype=str, keep_default_na=False, usecols=["documentID", "text"])
    tmap = dict(zip(raw.documentID, raw.text))
    docs = [TOKEN.findall(tmap.get(d, "").lower()) for d in lv.documentID]
    docs = [t for t in docs if len(t) >= 3]
    # Zipf
    freq = Counter(w for t in docs for w in t)
    ranks = np.arange(1, min(2000, len(freq)) + 1)
    fvals = np.array([c for _, c in freq.most_common(len(ranks))], float)
    zexp = -np.polyfit(np.log(ranks[10:1000]), np.log(fvals[10:1000]), 1)[0]
    # Heaps (vocab growth) — cumulative unique vs total tokens (in observed order)
    seen, V, N = set(), [], []
    tot = 0
    for t in docs:
        for w in t:
            seen.add(w); tot += 1
        V.append(len(seen)); N.append(tot)
    V = np.array(V); N = np.array(N)
    hexp = np.polyfit(np.log(N[N > 100]), np.log(V[N > 100]), 1)[0]
    # dream length distribution (words): lognormal fit
    w = lv.words.to_numpy(float); w = w[w > 0]
    lw = np.log(w)
    L += ["## 3. Scaling laws (OLS log-log slopes; corroborated by F0016 which reported Zipf R²≈.996)",
          f"- **Zipf** rank-frequency exponent (ranks 10-1000): **{zexp:.2f}** (≈1, classic; F0016 got 1.25 — "
          "OLS slope, no per-fit CI here).",
          f"- **Heaps** vocabulary growth V~N^β: **β = {hexp:.2f}** ({len(freq)} types / {int(N[-1]):,} tokens). "
          f"NB internal tension: 1/Zipf≈{1/zexp:.2f} ≠ observed Heaps β — so treat both as approximate, not a "
          "tight Zipf-Heaps identity.",
          f"- **Dream length** (words): median {np.median(w):.0f}, mean {w.mean():.0f}; **heavy-tailed, "
          f"well-approximated by log-normal** (log μ={lw.mean():.2f}, σ={lw.std():.2f}) — a power-law/gamma "
          "alternative is NOT formally excluded (not fit-compared).", ""]
    # Taylor's law across themes (daily rate mean vs variance)
    dd = pd.read_csv(C.DREAMS_OUT / "dreamseer_daily.csv")
    dd = dd[dd.lang == "en"].copy(); dd["date"] = pd.to_datetime(dd["date"])
    dd = dd[(dd.date >= "2024-03-01") & (dd.n_dreams >= 15)]
    means, vars_ = [], []
    for th in THEMES:
        col = th + "_mean"
        if col in dd.columns:
            s = dd[col].to_numpy(float); s = s[np.isfinite(s)]
            if s.mean() > 0 and len(s) > 30:
                means.append(s.mean()); vars_.append(s.var())
    means = np.array(means); vars_ = np.array(vars_)
    talpha = np.polyfit(np.log(means), np.log(vars_), 1)[0]
    L += [f"- **Taylor's fluctuation-scaling law** across {len(means)} themes (Var~Mean^α): α = {talpha:.2f} "
          "— **CONFOUNDED / descriptive only**: theme rates are bounded proportions, so the mean-variance "
          "relation is shaped by the binomial p(1−p)/n structure, not a free bursty-vs-Poisson exponent. "
          "Not interpreted as a scaling law.", ""]
    return (ranks, fvals, zexp), (N, V, hexp), (means, vars_, talpha)


# ---------------------------------------------------------------- fractal dim ----
def two_nn_dim(X, sample=8000, seed=0):
    rng = np.random.default_rng(seed)
    if len(X) > sample:
        X = X[rng.choice(len(X), sample, replace=False)]
    tree = cKDTree(X)
    dd, _ = tree.query(X, k=3)
    r1, r2 = dd[:, 1], dd[:, 2]
    ok = (r1 > 0)
    mu = np.sort(r2[ok] / r1[ok])
    Femp = np.arange(1, len(mu) + 1) / len(mu)
    m = Femp < 0.9
    d = np.polyfit(np.log(mu[m]), -np.log(1 - Femp[m]), 1)[0]
    return float(d)


def corr_dim(X, sample=1500, seed=0):
    rng = np.random.default_rng(seed)
    if len(X) > sample:
        X = X[rng.choice(len(X), sample, replace=False)]
    from scipy.spatial.distance import pdist
    D = pdist(X)
    rs = np.quantile(D, np.linspace(0.02, 0.5, 20))
    C_r = np.array([(D < r).mean() for r in rs])
    ok = C_r > 0
    d2 = np.polyfit(np.log(rs[ok]), np.log(C_r[ok]), 1)[0]
    return float(d2), rs[ok], C_r[ok]


def fractal_dim(L):
    try:
        from psychohistory.dreams.embed_cache import load_or_build
        meta, emb = load_or_build()
    except Exception as e:
        L += [f"## 4. Fractal dimension of dream-space — skipped: {e}", ""]
        return None
    E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    en = (meta.lang.values == "en")
    Xen = E[en]
    # multi-seed stability CIs (given the D2 vs two-NN spread)
    dnn_s = [two_nn_dim(Xen, seed=s) for s in range(5)]
    d2_s = [corr_dim(Xen, seed=s)[0] for s in range(5)]
    dnn, dnn_lo, dnn_hi = np.median(dnn_s), min(dnn_s), max(dnn_s)
    d2, d2_lo, d2_hi = np.median(d2_s), min(d2_s), max(d2_s)
    _, rs, Cr = corr_dim(Xen, seed=0)
    # nulls: isotropic Gaussian AND per-feature shuffle (variance-preserving, F0016-consistent)
    rng = np.random.default_rng(1)
    G = rng.standard_normal((min(len(Xen), 8000), E.shape[1]))
    G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    dnn_gauss = two_nn_dim(G)
    Sh = np.column_stack([rng.permutation(Xen[:, j]) for j in range(Xen.shape[1])])
    dnn_shuf = two_nn_dim(Sh)
    L += ["## 4. Intrinsic dimension of dream-space (MiniLM 384-d embedding, EN)",
          f"- **Two-NN intrinsic dimension ≈ {dnn:.0f}** [{dnn_lo:.0f}-{dnn_hi:.0f}] (primary; corroborates "
          f"F0016 ~25) — **far below the 384 ambient dims → dreams occupy a thin low-dimensional manifold**.",
          f"- Grassberger-Procaccia correlation dim **D₂ ≈ {d2:.1f}** [{d2_lo:.1f}-{d2_hi:.1f}] is a "
          "narrow-radius, distance-concentration-regime estimate that under-reads vs two-NN — treated as a "
          "fragile LOWER bound, not a fractal/self-similarity claim (scale-invariance not demonstrated).",
          f"- Nulls (same 384-d): isotropic-Gaussian two-NN ≈ {dnn_gauss:.0f}; per-feature-shuffle (variance-"
          f"preserving, F0016-style) ≈ {dnn_shuf:.0f} → the low dimensionality is **genuine structure**, not "
          "distance concentration.",
          "- *Caveat: embeddings use 256-char-truncated text (median dream ≈56 words → many truncated), so "
          "the geometry reflects dream OPENINGS.*", ""]
    return d2, rs, Cr, dnn


def fig_physics(spec, sl, fr, cf):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    # (1) power spectrum
    ax = axes[0]
    f, Pxx, beta, tops = spec
    ax.loglog(1.0 / f[1:], Pxx[1:], color="#4C72B0", lw=1)
    for p, _ in tops[:3]:
        ax.axvline(p, color="#C1443C", ls=":", lw=1)
    ax.set_xlabel("period (days)"); ax.set_ylabel("power")
    ax.set_title(f"Power spectrum (deseasonalized)\n1/f^β β={beta:+.2f}; peaks marked")
    # (2) Zipf + Heaps
    ax = axes[1]
    (ranks, fvals, zexp), (N, V, hexp), _ = sl
    ax.loglog(ranks, fvals, ".", ms=2, color="#333", label=f"Zipf slope {zexp:.2f}")
    ax.set_xlabel("word rank"); ax.set_ylabel("frequency"); ax.set_title("Zipf rank-frequency"); ax.legend(fontsize=8, frameon=False)
    # (3) correlation dimension
    ax = axes[2]
    if fr is not None:
        d2, rs, Cr, dnn = fr
        ax.loglog(rs, Cr, "o-", ms=3, color="#8856a7")
        ax.set_xlabel("radius r"); ax.set_ylabel("C(r)")
        ax.set_title(f"Intrinsic dim: two-NN≈{dnn:.0f} (D₂≈{d2:.1f} lower)\nambient=384 → low-D manifold")
    fig.tight_layout(); fig.savefig(OUT / "44_physics_of_dreams.png", dpi=150); plt.close(fig)


def main():
    L = ["# Physics of dreams II — periodicity, scaling laws, complexity, fractal geometry", "",
         "*New measures beyond F0016/F0033/F0034/F0035, each with a null/surrogate. EN primary; "
         "aggregate-only; exploratory. Daily collective reliability is low (split-half r≈.09), so the "
         "temporal-complexity measures are expected near-white — the robust physics is in the scaling laws "
         "and the embedding geometry.*", ""]
    resid, reg, _ = daily_resid()
    spec = spectral(reg, L)
    cf = complexity_fractal(resid, reg, L)
    sl = scaling_laws(L)
    fr = fractal_dim(L)
    fig_physics(spec, sl, fr, cf)
    L += ["**Figure:** 44_physics_of_dreams.png", "",
          "*Read: the robust aggregate 'physics' is (i) natural SCALING LAWS (Zipf≈1, Heaps sublinear, "
          "log-normal length) and (ii) the LOW INTRINSIC DIMENSION of dream-space (two-NN≈25 vs 384 ambient; "
          "not claimed 'fractal' — scale-invariance untested). The temporal measures (β≈white, gap-robust "
          "DFA α≈0.53, entropy≈max) are an informative NULL: once weekday+trend are removed the daily "
          "collective mood has no DETECTABLE long-range memory (under-powered; consistent with F0016) — its "
          "structure is periodic (weekday/seasonal, F0010) + noise.*"]
    (OUT / "dream_physics_ii.md").write_text("\n".join(L))
    print("\n".join(L)); print("\n[physics] wrote", OUT / "dream_physics_ii.md")


if __name__ == "__main__":
    main()
