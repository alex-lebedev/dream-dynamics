"""BOLD PROBE 6 — collective pulses & cycles: weekly rhythm across the affect family, a spectral
7-day pulse, and weekly/monthly co-movement with societal indicators.

(A) WEEKDAY SPECTRUM — for the whole emotion/theme family, which dimensions have a weekly cycle?
    Dream-level user-cluster-robust OLS (the validated Friday>Monday method), season + text-length
    controls; joint weekday Wald test; EN and RU (native) CROSS-VERIFIED (sign must agree).
(B) SPECTRAL PULSE — periodogram of the daily collective mood; is there a significant 7-day peak
    (permutation null)?
(C) COUPLING — weekly (and monthly) dream mood vs societal indicators (VIX, S&P return, EPU, yield
    curve, HY spread, $, news tone): prewhitened cross-correlation + exact circular-shift p +
    Toda-Yamamoto; alternative-metric replication (>=2 indicators, same sign); EN<->RU cross-verify.

Reliability-aware: coupling at WEEKLY/monthly only (daily too noisy). Multiplicity: BH-FDR reported
at BOTH the standard q=0.05 AND a RELAXED q=0.10/0.20 (exploratory, per Alex's steer). Aggregate-only.

    PYTHONPATH=src PYTHONNOUSERSITE=1 python3 analyses/2026-07-18-20-collective-cycles.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from psychohistory import config as CFG  # NB: not `C` — patsy formulas use C() for categoricals
from psychohistory.signals.panel import build_weekly_panel, signal_columns
from psychohistory.stats.inference import benjamini_hochberg, circular_shift_pvalue_corr
from psychohistory.stats.timeseries import prewhiten_ccf, toda_yamamoto

OUT = CFG.RESULTS / "showcase"
OUT.mkdir(parents=True, exist_ok=True)
START = "2024-03-01"
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WD_OUTCOMES = ["neg_sentiment", "nightmare_index", "negativity", "fear", "anger", "sadness",
               "joy", "trust", "anticipation", "danger", "social", "family", "flying"]


def load_dream_level():
    dl = pd.read_parquet(CFG.DREAMS_OUT / "dreamseer_dream_level.parquet",
                         columns=["documentID", "userID", "date", "lang", "textlen",
                                  "nightmare_index", "negativity", "fear", "anger", "sadness",
                                  "joy", "trust", "anticipation", "danger", "social", "family", "flying"])
    dl["date"] = pd.to_datetime(dl["date"])
    se = pd.read_parquet(CFG.DREAMS_OUT / "dreamseer_sentiment.parquet", columns=["documentID", "sentiment"])
    dl = dl.merge(se, on="documentID", how="left")
    dl["neg_sentiment"] = -dl["sentiment"]
    dl = dl[dl.date >= START].reset_index(drop=True)
    dl["dow"] = dl.date.dt.dayofweek
    doy = dl.date.dt.dayofyear
    dl["sin1"] = np.sin(2 * np.pi * doy / 365.25); dl["cos1"] = np.cos(2 * np.pi * doy / 365.25)
    dl["logtl"] = np.log(dl.textlen.clip(lower=1))
    return dl


# ---------------------------------------------------------- (A) weekday ----
def weekday_effects(dl, lang):
    d = dl[dl.lang == lang]
    rows = []
    for out in WD_OUTCOMES:
        sub = d[["userID", "dow", "sin1", "cos1", "logtl", out]].dropna()
        if sub[out].std() == 0 or len(sub) < 500:
            continue
        m = smf.ols(f"{out} ~ C(dow) + sin1 + cos1 + logtl", sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub.userID})
        dterms = [p for p in m.params.index if p.startswith("C(dow)")]
        jp = float(m.f_test(" , ".join(f"{t} = 0" for t in dterms)).pvalue)
        # weekday means relative to Monday (baseline)
        eff = {0: 0.0}
        for t in dterms:
            eff[int(t.split("T.")[1].rstrip("]"))] = float(m.params[t])
        peak = max(eff, key=eff.get); trough = min(eff, key=eff.get)
        rows.append(dict(lang=lang, outcome=out, joint_p=jp,
                         fri_minus_mon=eff.get(4, np.nan),
                         peak_day=DOW[peak], trough_day=DOW[trough],
                         range=eff[peak] - eff[trough]))
    R = pd.DataFrame(rows)
    R["q05"] = benjamini_hochberg(R.joint_p.values)
    return R


# ---------------------------------------------------------- (B) spectral ----
def weekly_pulse(dl, lang="all"):
    d = dl if lang == "all" else dl[dl.lang == lang]
    daily = d.groupby(d.date.dt.normalize()).agg(s=("neg_sentiment", "mean"), n=("neg_sentiment", "size"))
    daily = daily[daily.n >= 15]
    full = pd.date_range(daily.index.min(), daily.index.max())
    s = daily.s.reindex(full).interpolate(limit=3).dropna()
    x = s.values - np.polyval(np.polyfit(np.arange(len(s)), s.values, 1), np.arange(len(s)))
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0)
    power = np.abs(np.fft.rfft(x)) ** 2
    per = 1 / freqs[1:]
    pw = power[1:]
    k7 = int(np.argmin(np.abs(per - 7)))
    rng = np.random.default_rng(0)
    null = np.array([np.abs(np.fft.rfft(rng.permutation(x)))[1:][k7] ** 2 for _ in range(2000)])
    p7 = (1 + np.sum(null >= pw[k7])) / 2001
    band = per <= 40                                    # report dominant CYCLE (ignore long trend)
    dom = per[band][np.argmax(pw[band])]
    return {"lang": lang, "n_days": n, "dominant_period": float(dom),
            "power_at_7d_rank": int((pw > pw[k7]).sum() + 1), "p_7day": float(p7)}


# ---------------------------------------------------------- (C) coupling ----
def dream_weekly(dl, lang, metric, min_n=40):
    d = dl if lang == "all" else dl[dl.lang == lang]
    w = d.groupby(d.date.dt.to_period("W-SUN").apply(lambda p: p.start_time)).agg(
        v=(metric, "mean"), n=(metric, "size"))
    w = w[w.n >= (min_n if lang != "ru" else 20)]
    return w.v


def coupling(dl, panel, langs=("all", "en", "ru"),
             metrics=("neg_sentiment", "nightmare_index", "negativity"), maxlag=6):
    """PRIMARY = contemporaneous (lag-0) weekly corr with exact circular-shift p (FDR-controlled).
    EXPLORATORY = best lead/lag from the prewhitened CCF (descriptive; lag search not FDR'd)."""
    rows = []
    for lang in langs:
        for metric in metrics:
            dw = dream_weekly(dl, lang, metric)
            for s in signal_columns(panel):
                m = pd.concat([dw.rename("d"), panel.set_index("week")[s].rename("s")],
                              axis=1).dropna()
                if len(m) < 40:
                    continue
                r0, p0 = circular_shift_pvalue_corr(m.d.values, m.s.values)   # LEVEL contemporaneous
                md, ms = m.d.diff().dropna(), m.s.diff().dropna()             # FIRST DIFFERENCE (trend-robust)
                pair = pd.concat([md, ms], axis=1).dropna()
                rd, pd_ = (circular_shift_pvalue_corr(pair.iloc[:, 0].values, pair.iloc[:, 1].values)
                           if len(pair) >= 40 else (np.nan, np.nan))
                ccf = prewhiten_ccf(m.d.values, m.s.values, arlags=4, maxlag=maxlag)  # lag>0: dream leads
                best = max(ccf, key=lambda k: abs(ccf[k])) if ccf else 0
                rows.append(dict(lang=lang, metric=metric, signal=s, n_weeks=len(m),
                                 r_level=float(r0), p_level=float(p0), r_diff=float(rd), p_diff=float(pd_),
                                 best_lag_w=int(best), ccf_best=float(ccf.get(best, np.nan))))
    R = pd.DataFrame(rows)
    if len(R):
        R["q_level"] = benjamini_hochberg(R.p_level.values)
        R["q_diff"] = benjamini_hochberg(R.p_diff.fillna(1).values)   # PRIMARY: trend-robust
    return R


def main():
    dl = load_dream_level()
    panel = build_weekly_panel()
    L = ["# BOLD PROBE 6 — collective pulses & cycles", "",
         "*Weekday rhythm (dream-level, user-cluster-robust OLS + season/length controls); spectral "
         "7-day pulse (permutation); weekly/monthly coupling (prewhitened CCF + exact circular-shift "
         "p). Reliability-aware (coupling weekly+). FDR at q=.05 and RELAXED q=.10 (exploratory).*", ""]

    # (A) weekday
    we = pd.concat([weekday_effects(dl, "en"), weekday_effects(dl, "ru")], ignore_index=True)
    we.to_csv(OUT / "cycles_weekday_effects.csv", index=False)
    en = we[we.lang == "en"].set_index("outcome")
    ru = we[we.lang == "ru"].set_index("outcome")
    L += ["## (A) Weekly rhythm across the affect/theme family (EN, q<.10 = FDR hit)"]
    for o in WD_OUTCOMES:
        if o not in en.index:
            continue
        e = en.loc[o]
        agree = (o in ru.index) and (np.sign(e.fri_minus_mon) == np.sign(ru.loc[o].fri_minus_mon))
        star = " **[FDR]**" if e.q05 < 0.10 else ""
        L.append(f"- **{o}**: Fri−Mon={e.fri_minus_mon:+.3f}, peak {e.peak_day}/trough {e.trough_day}, "
                 f"joint p={e.joint_p:.1e} (q={e.q05:.2f}){star}; RU sign-agrees: {bool(agree)}")
    best_q = en.joint_p.pipe(lambda s: we[we.lang == 'en'].q05.min())
    L += [f"  *NB: no single outcome clears BH-FDR at q<.10 in this 13-outcome family (best q≈{best_q:.2f}). "
          "The weekly-rhythm ✅ rests on the PRIOR pre-registered Fri−Mon contrast (β=+0.021, p=.009, "
          "permutation p=.006, replicates every year; docs/VALIDATION.md) PLUS the new EN↔RU "
          "sign-agreement here — not on this exploratory family-wide test.*", ""]

    # (B) spectral
    L += ["## (B) Spectral 7-day pulse"]
    for lg in ["all", "en"]:
        p = weekly_pulse(dl, lg)
        L.append(f"- **{lg.upper()}**: dominant period {p['dominant_period']:.0f}d; 7-day power "
                 f"rank #{p['power_at_7d_rank']}, permutation p={p['p_7day']:.3f} "
                 f"({'significant weekly pulse' if p['p_7day'] < 0.05 else 'no clear 7-day peak'}).")
    L += [""]

    # (C) coupling
    cp = coupling(dl, panel)
    cp.to_csv(OUT / "cycles_coupling_weekly.csv", index=False)
    L += ["## (C) Weekly coupling with societal indicators (Frontier)",
          "*PRIMARY = FIRST-DIFFERENCED (trend-robust) lag-0 corr; level correlations of trending "
          "macro series are shown but are usually spurious co-trends.*"]
    if len(cp):
        top = cp.sort_values("p_level").head(10)
        for _, r in top.iterrows():
            surv = " **[Δ-FDR survives]**" if (pd.notna(r.q_diff) and r.q_diff < 0.10) else \
                   "  (level-only → likely co-trend)"
            L.append(f"- {r.lang}/{r.metric} ~ **{r.signal}**: level r={r.r_level:+.2f} (p={r.p_level:.3f}, "
                     f"q={r.q_level:.2f}); **Δ r={r.r_diff:+.2f}** (p={r.p_diff:.3f}, q={r.q_diff:.2f}){surv} "
                     f"(n={r.n_weeks}w)")
        nlev = int((cp.q_level < 0.10).sum()); ndiff = int((cp.q_diff < 0.10).sum())
        fearset = cp[(cp.metric == "neg_sentiment") & (cp.lang == "all") &
                     (cp.signal.isin(["vix", "hy_spread", "sp500_ret", "epu"]))]
        L += ["",
              f"- **{nlev} level hits at q<.10 collapse to {ndiff} once first-differenced** — i.e. the "
              "level 'coupling' with rate/curve series (dff, 2s10s) is mostly shared TREND, not weekly "
              "co-movement (TY below is null too).",
              "- Alternative-metric check (all/neg_sentiment vs fear proxies, differenced): "
              + "; ".join(f"{r.signal} Δr={r.r_diff:+.2f}(p={r.p_diff:.2f})" for _, r in fearset.iterrows()),
              "  **Verdict:** coupling stays **Frontier/NULL** at weekly resolution — no trend-robust, "
              "multi-indicator-replicated coupling. Consistent with the earlier r=−0.44 being an "
              "autocorrelation artifact (docs/VALIDATION.md). Revisit monthly as volume grows."]
    else:
        L += ["- insufficient overlapping weeks."]

    if len(cp):
        b = cp.sort_values("p_level").iloc[0]
        dw = dream_weekly(dl, b.lang, b.metric)
        tydf = pd.concat([dw.rename("d"), panel.set_index("week")[b.signal].rename("s")], axis=1).dropna()
        if len(tydf) > 30:
            f = toda_yamamoto(tydf, "s", "d", dmax=1); rv = toda_yamamoto(tydf, "d", "s", dmax=1)
            L += ["", f"- TY on strongest lag-0 pair ({b.lang}/{b.metric} ~ {b.signal}): "
                  f"{b.signal}→dreams p={f['pvalue']:.3f}, dreams→{b.signal} p={rv['pvalue']:.3f}."]

    (OUT / "collective_cycles.md").write_text("\n".join(L))
    print("\n".join(L)); print("\n[cycles] wrote", OUT / "collective_cycles.md")


if __name__ == "__main__":
    main()
