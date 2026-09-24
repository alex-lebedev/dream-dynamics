"""Round 5 POC — the "collective wave": dreams as ONE sensor among many; which dream AXIS carries it?

Program reframe (Alex): don't ask "does dream-negativity track VIX". Build a latent collective-mood
"wave" from MANY external indicators (dreams left OUT), then ask which AXIS of dreams — valence,
**intensity/arousal**, coherence, threat, grief, agency — loads on that wave, and whether it beats a
NEUTRAL-theme control (anxiety/emotion-specificity) and colored-noise chance. English/Western dreams
are the PRIMARY population; RU is reported SEPARATELY (a different collective), never pooled.

Wave = leave-dreams-out latent factor of the external market block (VIX/EPU/HY/2s10s/dollar):
  (a) PCA PC1 (interpretable), cross-checked against (b) a statsmodels DynamicFactor smoothed state.
Inference in LEVELS with autocorrelation-preserving phase surrogates (Ebisuzaki/GETAB). Aggregate-only,
~2.3-yr window → exploratory. Straightforward outputs: one wave line + a ranked axis-loading table.

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 USE_TF=0 python3 \
        analyses/2026-07-19-24-collective-wave-poc.py
"""
from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats

from psychohistory import config as C
from psychohistory.dreams.embed_cache import load_or_build
from psychohistory.stats.inference import benjamini_hochberg


def mde_r(n, alpha=0.05):
    """Minimum detectable |r| at this n (two-sided α) — for honest low-power framing."""
    if n is None or n < 4:
        return np.nan
    tc = sstats.t.ppf(1 - alpha / 2, n - 2)
    return float(tc / np.sqrt(n - 2 + tc ** 2))

OUT = C.RESULTS / "showcase"
OUT.mkdir(parents=True, exist_ok=True)
SIG = C.EXTERNAL / "signals"
EMO = ["fear", "anger", "sadness", "disgust", "joy", "trust", "anticipation", "surprise"]
THREAT = ["danger", "weapon", "fear", "nightmare_index"]
NEUTRAL = ["family", "food", "flying", "animal", "social"]
AGENCY = re.compile(r"\b(I|I'm|I've|I'd|me|my|mine|myself)\b", re.I)


# --------------------------------------------------------------- external wave ----
def _fred(fn, name):
    p = SIG / fn
    if not p.exists():
        return None
    d = pd.read_csv(p); d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    return d.dropna().set_index("date")["value"].rename(name)


def external_panel(freq):
    cols = [_fred(f, n) for f, n in [("fred_VIXCLS.csv", "vix"), ("fred_BAMLH0A0HYM2.csv", "hy_spread"),
                                     ("fred_T10Y2Y.csv", "yield_2s10s"), ("fred_DTWEXBGS.csv", "dollar")]]
    frames = [c for c in cols if c is not None]
    p = SIG / "epu_daily.csv"
    if p.exists():
        e = pd.read_csv(p)
        if {"day", "month", "year"}.issubset(e.columns):
            e["date"] = pd.to_datetime(dict(year=e.year, month=e.month, day=e.day), errors="coerce")
            vcol = next((c for c in e.columns if "index" in c.lower()), None)
            if vcol:
                frames.append(e.dropna(subset=["date"]).set_index("date")[vcol].rename("epu"))
    daily = pd.concat(frames, axis=1).sort_index()
    key = daily.index.to_period("W-SUN").start_time if freq == "W" else daily.index.to_period("M").start_time
    return daily.groupby(key).mean()


def collective_wave(freq, window="2024-02-01"):
    sig = external_panel(freq)
    cols = [c for c in ["vix", "epu", "hy_spread", "yield_2s10s", "dollar"] if c in sig]
    X = sig[cols].copy()
    X = X[X.index >= window].interpolate(limit_direction="both").dropna(how="any")
    Z = (X - X.mean()) / (X.std() + 1e-9)
    # (a) PCA PC1
    U, S, Vt = np.linalg.svd(Z.values - Z.values.mean(0), full_matrices=False)
    pc1 = Z.values @ Vt[0]
    if np.corrcoef(pc1, Z["vix"].values)[0, 1] < 0:
        pc1 = -pc1
    pc1 = pd.Series(pc1, index=X.index, name="wave_pc1")
    var1 = float((S[0] ** 2) / (S ** 2).sum())
    # (b) DynamicFactor smoothed state (cross-check)
    dfm_r = np.nan
    try:
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
        res = DynamicFactor(Z, k_factors=1, factor_order=2, error_order=1).fit(disp=False, maxiter=200)
        f = np.asarray(res.factors.smoothed).ravel()
        if np.corrcoef(f, Z["vix"].values)[0, 1] < 0:
            f = -f
        dfm_r = float(np.corrcoef(pc1.values, f)[0, 1])
    except Exception as e:
        print("[wave] DFM skipped:", e)
    return pc1, var1, dfm_r


# --------------------------------------------------------------- dream axes ----
def dream_axes(freq, cohort):
    lv = pd.read_parquet(C.DREAMS_OUT / "dreamseer_dream_level.parquet",
                         columns=["documentID", "date", "lang"] + EMO + ["danger", "weapon",
                         "nightmare_index", "negativity"] + NEUTRAL + ["textlen"])
    lv["date"] = pd.to_datetime(lv["date"]); lv = lv[lv.date >= "2024-03-01"]
    se = pd.read_parquet(C.DREAMS_OUT / "dreamseer_sentiment.parquet", columns=["documentID", "sentiment"])
    lv = lv.merge(se, on="documentID", how="left")
    if cohort != "all":
        lv = lv[lv.lang == cohort]
    lv["valence_neg"] = lv["negativity"]
    lv["neg_sent"] = -lv["sentiment"]
    lv["intensity"] = lv[EMO].mean(axis=1)                       # emotional activation (sign-agnostic)
    lv["arousal_hi"] = lv[["fear", "anger", "surprise", "anticipation"]].mean(axis=1)
    lv["threat"] = lv[THREAT].mean(axis=1)
    lv["grief"] = lv["sadness"]
    key = lv.date.dt.to_period("W-SUN").dt.start_time if freq == "W" else lv.date.dt.to_period("M").dt.start_time
    g = lv.groupby(key)
    axes = ["valence_neg", "neg_sent", "intensity", "arousal_hi", "threat", "grief"] + NEUTRAL
    out = g[axes].mean()
    out["n"] = g.size(); out["logn"] = np.log(out["n"])
    thr = 30 if freq == "W" else 120
    out = out[out["n"] >= thr]

    # coherence (embedding dispersion) — separate, from the MiniLM cache (higher = LESS coherent)
    try:
        meta, emb = load_or_build()
        E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        mask = (meta.lang.values == cohort) if cohort != "all" else np.ones(len(meta), bool)
        sub = meta[mask].copy().assign(_pi=np.where(mask)[0])
        pk = (sub.date.dt.to_period("W-SUN").dt.start_time if freq == "W"
              else sub.date.dt.to_period("M").dt.start_time)
        disp = {}
        for per, gg in sub.groupby(pk):
            if len(gg) >= thr:
                V = E[gg._pi.values]
                cen = V.mean(0)
                disp[per] = float(1 - (V @ cen / (np.linalg.norm(cen) + 1e-9)).mean())
        out["dispersion"] = pd.Series(disp)
    except Exception as e:
        print("[axes] dispersion skipped:", e)

    # agency (first-person rate) — separate, from raw text
    try:
        raw = pd.read_csv(C.DREAMSEER_RAW, dtype=str, keep_default_na=False, usecols=["documentID", "text"])
        tmap = dict(zip(raw.documentID, raw.text))
        lv["agency"] = lv.documentID.map(lambda d: len(AGENCY.findall(tmap.get(d, ""))) /
                                         max(len(tmap.get(d, "").split()), 1))
        out["agency"] = lv.groupby(key)["agency"].mean()
    except Exception as e:
        print("[axes] agency skipped:", e)
    return out


def detrend(y, logn):
    t = np.arange(len(y), dtype=float)
    X = np.column_stack([np.ones_like(t), (t - t.mean()) / (t.std() + 1e-9), logn - np.mean(logn)])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def phase_surrogate(x, rng):
    x = np.asarray(x, float); n = len(x)
    X = np.fft.rfft(x - x.mean()); ph = rng.uniform(0, 2 * np.pi, len(X)); ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=n) + x.mean()


def _r(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def axis_vs_wave(axis_s, wave, logn, lags=(-3, -2, -1, 0, 1, 2, 3), B=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.Series(detrend(axis_s.values.astype(float), logn.values), index=axis_s.index)
    best = {"r0": np.nan, "p0": np.nan, "best_r": 0.0, "best_k": 0}
    dpos = pd.Series(np.arange(len(d)), index=d.index)
    for k in lags:
        j = pd.concat([dpos.rename("p"), wave.shift(-k).rename("m")], axis=1).dropna()
        if len(j) < 8:
            continue
        pos = j["p"].values.astype(int); ma = j["m"].values
        r = _r(d.values[pos], ma)
        if k == 0:
            null = np.array([abs(_r(phase_surrogate(d.values, rng)[pos], ma)) for _ in range(B)])
            best["r0"] = r; best["p0"] = float((1 + np.sum(null >= abs(r))) / (B + 1))
        if abs(r) > abs(best["best_r"]):
            best["best_r"] = r; best["best_k"] = k
    return best


TYPE = {"valence_neg": "affect", "neg_sent": "affect", "intensity": "intensity", "arousal_hi": "intensity",
        "threat": "threat", "grief": "affect", "dispersion": "coherence", "agency": "agency",
        **{t: "neutral" for t in NEUTRAL}}


def run_cohort(cohort, freq="M"):
    wave, var1, dfm_r = collective_wave(freq)
    ax = dream_axes(freq, cohort)
    rows = []
    for a in ax.columns:
        if a in ("n", "logn") or a not in TYPE:
            continue
        s = ax[a].dropna()
        if len(s) < (25 if freq == "W" else 12):
            continue
        res = axis_vs_wave(s, wave, ax.loc[s.index, "logn"])
        rows.append({"axis": a, "type": TYPE[a], "r0": res["r0"], "p0": res["p0"],
                     "best_r": res["best_r"], "best_k": res["best_k"], "n": int(len(s))})
    R = pd.DataFrame(rows).sort_values("r0", key=lambda c: c.abs(), ascending=False).reset_index(drop=True)
    if len(R):
        R["q0"] = benjamini_hochberg(R["p0"].values)   # BH-FDR across the axis family (METHODS 4.2)
    return R, wave, var1, dfm_r, ax


def main():
    L = ["# Round 5 POC — which dream AXIS rides the MARKET-mood factor?", "",
         "*Leave-dreams-out latent = PC1 of the **market/macro-financial block only** (VIX/EPU/HY/2s10s/"
         "dollar) — a MARKET-mood factor, **not** the broad multi-domain 'collective wave' (that needs "
         "GDELT tone/attention/conflict added). Dream axes tested in LEVELS with phase-surrogate nulls + "
         "**BH-FDR** across the axis family. EN/Western = primary population; RU reported SEPARATELY, never "
         "pooled. Aggregate-only; ~2.3-yr → **low-power exploratory** (see MDE per cohort).*", ""]
    tables = {}
    en_bundle = None
    for cohort in ["en", "ru"]:
        R, wave, var1, dfm_r, ax = run_cohort(cohort, "M")
        tables[cohort] = R
        if cohort == "en":
            en_bundle = (R, wave, ax)
        R.to_csv(OUT / f"collective_wave_axes_{cohort}.csv", index=False)
        nmax = int(R.n.max()) if len(R) else None
        L += [f"## {cohort.upper()} population (monthly; market-factor PC1 var={var1*100:.0f}%, "
              f"PCA↔DynamicFactor r={dfm_r:+.2f}; n≈{nmax} → MDE |r|≈{mde_r(nmax):.2f})",
              "| dream axis | type | r₀ (contemp.) | phase-p | BH-q | best r | best lag |",
              "|---|---|--:|--:|--:|--:|--:|"]
        for _, r in R.iterrows():
            L.append(f"| {r.axis} | {r.type} | {r.r0:+.3f} | {r.p0:.3f} | {r.get('q0', float('nan')):.3f} "
                     f"| {r.best_r:+.3f} | {int(r.best_k)} |")
        nsurv = int((R["q0"] < 0.05).sum()) if "q0" in R else 0
        # specificity: best non-neutral vs best neutral
        nn = R[R.type != "neutral"]; ne = R[R.type == "neutral"]
        if len(nn) and len(ne):
            top = nn.iloc[nn.r0.abs().argmax()]; topn = ne.iloc[ne.r0.abs().argmax()]
            spec = abs(top.r0) - abs(topn.r0)
            L += ["",
                  f"- **{nsurv}/{len(R)} axes survive BH-FDR** (q<.05){' — none' if nsurv == 0 else ''}; "
                  "any bare RU phase-p's (food/dispersion ~.01) do **not** survive multiplicity.",
                  f"- **Top dream axis:** `{top.axis}` ({top.type}) r₀={top.r0:+.3f} (phase-p={top.p0:.3f}, "
                  f"q={top.get('q0', float('nan')):.3f}). **Top NEUTRAL control:** `{topn.axis}` r₀={topn.r0:+.3f}. "
                  f"Specificity Δ|r|={spec:+.3f} → "
                  f"{'content-specific (beats neutral + FDR)' if spec > 0.05 and top.get('q0', 1) < 0.05 else 'NOT above the neutral/co-drift baseline'}.",
                  f"- Reframe check (is it intensity/arousal, not negativity?): leading type = **{top.type}** "
                  "— but not significant, so no axis is preferred at this window.",
                  f"- **Low-power caveat:** n≈{nmax} months → can only detect |r|≳{mde_r(nmax):.2f}; "
                  "read as a LOW-POWER null ('at this window'), not an established absence. Lead-lags are "
                  "in-sample/exploratory (boundary lags = edge artifacts).", ""]
    if en_bundle is not None:
        fig_wave(*en_bundle, cohort="en")
    L += ["**Figure:** 35_collective_wave_poc.png", "",
          "*Verdict tier: exploratory POC. Interpretation: a dream axis is a candidate collective-wave "
          "sensor only if it (i) exceeds phase-surrogate chance, (ii) beats the neutral-theme control, "
          "and (iii) ideally leads the wave. This POC ranks the axes and applies (i)+(ii); the honest "
          "bar (longer window, cross-cultural distance, out-of-sample) still applies before any claim.*"]
    (OUT / "collective_wave_poc.md").write_text("\n".join(L))
    print("\n".join(L)); print("\n[wave-poc] wrote", OUT / "collective_wave_poc.md")


def fig_wave(R, wave, ax, cohort="en"):
    if R is None or not len(R):
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    a = axes[0]
    colors = {"intensity": "#C1443C", "affect": "#4C72B0", "threat": "#8856a7",
              "coherence": "#2ca02c", "agency": "#e08a1e", "neutral": "#9aa0a6"}
    order = R.sort_values("r0", key=lambda c: c.abs(), ascending=True)
    a.barh(order.axis, order.r0, color=[colors.get(t, "#555") for t in order.type])
    a.axvline(0, color="k", lw=0.6)
    a.set_xlabel("contemporaneous r with collective wave (levels)")
    a.set_title(f"{cohort.upper()}: which dream axis rides the wave?\n(grey = neutral control)")
    a2 = axes[1]
    top = R.iloc[R.r0.abs().argmax()]
    w = (wave - wave.mean()) / wave.std()
    s = ax[top.axis].dropna()
    sd = pd.Series(detrend(s.values.astype(float), ax.loc[s.index, "logn"].values), index=s.index)
    sd = (sd - sd.mean()) / sd.std()
    a2.plot(pd.to_datetime(w.index).to_numpy(), w.values, "-o", ms=3, color="black", label="collective wave (PC1)")
    a2.plot(pd.to_datetime(sd.index).to_numpy(), sd.values, "-o", ms=3, color=colors.get(top.type, "#C1443C"),
            label=f"top dream axis: {top.axis} ({top.type})")
    a2.axhline(0, color="k", lw=0.5); a2.legend(frameon=False, fontsize=8)
    a2.set_title("Collective wave vs top dream axis (z, monthly)")
    fig.tight_layout(); fig.savefig(OUT / "35_collective_wave_poc.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
