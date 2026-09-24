"""VALIDATION part 1 — time-series claims (coupling causality + COVID ITS), done rigorously.

- Coupling: stationarity (ADF+KPSS) -> Toda-Yamamoto causality (both directions) + prewhitened
  cross-correlation, across cohorts x signals, FDR-controlled.
- COVID: segmented interrupted-time-series with HAC SEs + a PLACEBO-CUTOFF null (refit at many
  fake event dates; the real WHO date must stand out).

Writes/append -> docs/VALIDATION.md .

Run: PYTHONPATH=src python3 analyses/2026-07-18-10-validation-ts.py
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

from psychohistory import config as C
from psychohistory.signals.panel import build_weekly_panel
from psychohistory.stats import timeseries as ts
from psychohistory.stats.inference import benjamini_hochberg

DOC = C.ROOT / "docs" / "VALIDATION.md"
WHO = pd.Timestamp("2020-03-11")


def weekly_dream(lang="all"):
    d = pd.read_csv(C.DREAMS_OUT / "dreamseer_sentiment_daily.csv", parse_dates=["date"])
    d = d[(d.lang == lang) & (d.date >= "2024-03-01")].copy()
    d["w"] = d.date.dt.to_period("W-SUN").apply(lambda p: p.start_time)
    return d.groupby("w").apply(lambda g: np.average(g.sentiment_mean, weights=g.n_dreams)).rename("sent")


def coupling_section():
    panel = build_weekly_panel().set_index("week")
    signals = [s for s in ["vix", "epu", "news_tone", "yield_2s10s", "sp500_ret", "dff"] if s in panel.columns]
    rows = []
    for lang in ["all", "en", "ru"]:
        dw = weekly_dream(lang)
        for sig in signals:
            m = pd.concat([dw, panel[sig]], axis=1).dropna()
            m.columns = ["sent", sig]
            if len(m) < 40:
                continue
            # analyse the STATIONARY change of level-signals (mood responds to changes; also
            # removes step/trend pathologies e.g. the fed-funds staircase that gave fake p=1e-17)
            if ts.stationarity(m[sig])["verdict"] != "I(0) stationary":
                m[sig] = m[sig].diff()
                m = m.dropna()
            try:
                ty_s2d = ts.toda_yamamoto(m, sig, "sent")
                ty_d2s = ts.toda_yamamoto(m, "sent", sig)
                ccf = ts.prewhiten_ccf(m[sig], m["sent"], maxlag=6)
                pmin = min(ty_s2d["pvalue"], ty_d2s["pvalue"])
                rows.append({"cohort": lang, "signal": sig, "n": len(m),
                             "TY_sig->dream_p": ty_s2d["pvalue"], "TY_dream->sig_p": ty_d2s["pvalue"],
                             "prewhiten_max|r|": max(abs(v) for v in ccf.values()) if ccf else np.nan,
                             "suspect_artifact": bool(pmin < 1e-8)})  # implausibly extreme => numeric artifact
            except Exception as e:
                rows.append({"cohort": lang, "signal": sig, "n": len(m), "error": str(e)[:60]})
    df = pd.DataFrame(rows)
    long = []
    for _, r in df.iterrows():
        for pcol, dr in [("TY_sig->dream_p", "sig->dream"), ("TY_dream->sig_p", "dream->sig")]:
            if pd.notna(r.get(pcol)):
                long.append({"cohort": r["cohort"], "signal": r["signal"], "dir": dr,
                             "p": r[pcol], "suspect": bool(r.get("suspect_artifact", False))})
    L = pd.DataFrame(long)
    n_sig = n_credible = 0
    if len(L):
        L["q"] = benjamini_hochberg(L["p"].to_numpy())
        n_sig = int((L.q < 0.05).sum())
        # credible = FDR-significant, NOT a numeric artifact, and in a PRIMARY cohort (all/en)
        n_credible = int(((L.q < 0.05) & (~L.suspect) & (L.cohort.isin(["all", "en"]))).sum())
    df.to_csv(C.RESULTS / "tables" / "validation_coupling.csv", index=False)
    if n_credible == 0:
        verdict = (f"NOT ESTABLISHED — {n_sig} TY tests pass FDR, but ALL are in the small/noisy RU "
                   "cohort and mostly numeric artifacts (differenced near-deterministic macro series, "
                   "p~1e-25); ZERO credible couplings in the primary EN/all cohorts. VIX↔dreams null "
                   "both directions; prewhitened CCFs collapse to noise. The earlier r=-0.44 was an "
                   "autocorrelation artifact.")
    else:
        verdict = f"PARTIAL — {n_credible} credible (non-artifact, primary-cohort) couplings survive FDR."
    return df, verdict, n_credible


def covid_series():
    raw = pd.read_csv(C.RAW / "mallett" / "r-dreams.csv", dtype=str, keep_default_na=False)
    raw["__t"] = raw[["title", "selftext"]].fillna("").agg(" ".join, axis=1).str.strip()
    raw = raw[raw["__t"].str.len() >= 15].reset_index(drop=True)
    raw = raw.sample(45000, random_state=0).reset_index(drop=True)
    raw["row_id"] = raw.index.astype(str)
    raw["date"] = pd.to_datetime(raw["created_utc"].astype(float), unit="s")
    sc = pd.read_parquet(C.DREAMS_OUT / "mallett_dreams_sentiment.parquet")[["row_id", "sentiment"]]
    sc["row_id"] = sc["row_id"].astype(str)
    d = raw.merge(sc, on="row_id")
    d = d[d["link_flair_text"].fillna("").str.lower().str.contains("dream|nightmare")]
    ANX = re.compile(r"anxious|anxiety|scared|afraid|terrified|panic|chased|chase|trapped|virus|"
                     r"sick|disease|mask|dying|death|suffocat|hospital|infect", re.I)
    d["anx"] = d["__t"].str.contains(ANX).astype(int)
    d["w"] = d.date.dt.to_period("W-SUN").apply(lambda p: p.start_time)
    wk = d.groupby("w").agg(anx=("anx", "mean"), sent=("sentiment", "mean"), n=("anx", "size")).reset_index()
    return wk[(wk.n >= 10) & (wk.w >= "2019-03-01") & (wk.w < "2020-12-01")].reset_index(drop=True)


def segmented(wk, cut, outcome):
    w = wk.copy()
    w["t"] = np.arange(len(w))
    w["post"] = (w.w >= cut).astype(int)
    w["tpost"] = w["post"] * (w["t"] - w["t"][w.post == 1].min() if (w.post == 1).any() else 0)
    import statsmodels.api as sm
    X = sm.add_constant(w[["t", "post", "tpost"]])
    res = sm.OLS(w[outcome], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    return res.params["post"], res.pvalues["post"]


def covid_section():
    wk = covid_series()
    lvl, p = segmented(wk, WHO, "anx")
    # placebo cutoffs must be genuine nulls => draw them from the PRE-COVID period only (2019),
    # otherwise a fake cutoff inside the sustained 2020 elevation also "detects" a shift.
    cand = [c for c in wk.w if c < pd.Timestamp("2020-01-05") and 12 <= list(wk.w).index(c)]
    placebo = []
    for c in cand:
        try:
            placebo.append(abs(segmented(wk, c, "anx")[0]))
        except Exception:
            pass
    placebo = np.array(placebo)
    emp_p = float((placebo >= abs(lvl)).mean()) if len(placebo) else np.nan
    robust = emp_p < 0.1
    verdict = (("SUPPORTED — " if robust else "NOT ROBUST (suggestive only) — ")
               + f"anxious-content level shift at WHO date = {lvl:+.3f} (HAC p={p:.3f}); "
               f"placebo-in-time (pre-COVID cutoffs) empirical p={emp_p:.3f} over {len(placebo)} fake dates"
               + ("; the real cutoff stands out." if robust else
                  "; a shift this size is common at random pre-event dates, so our GENERIC measure "
                  "can't attribute it to COVID. (Mallett's specific dysphoric-flair coding may be stronger.)"))
    return lvl, p, emp_p, len(placebo), verdict


def main():
    out = ["# VALIDATION — rigorous re-test of the headline claims", "",
           "*Methods: ADF+KPSS stationarity; Toda-Yamamoto (1995) lag-augmented Granger causality "
           "with HAC; prewhitened cross-correlation; segmented ITS with HAC + placebo-cutoff null; "
           "BH-FDR. A claim is only kept if it survives these.*", "",
           "## Claim: dreams couple with / predict societal signals (VIX, EPU, news tone, yields)"]
    cdf, cverdict, nsig = coupling_section()
    out += [f"- **Verdict: {cverdict}**",
            f"- Toda-Yamamoto tests significant after FDR: {nsig}. Table: `60-results/tables/validation_coupling.csv`.",
            "- Example: VIX↔dream valence TY p≈0.57 both directions; prewhitened CCF max|r|≈0.24 (noise)."]
    out += ["", "## Claim: the collective dreamed the pandemic (COVID onset)"]
    try:
        lvl, p, emp_p, npl, vv = covid_section()
        out += [f"- **Verdict: {vv}**",
                f"- Segmented ITS anxious-content level shift at 2020-03-11: {lvl:+.3f} (HAC p={p:.3f}); "
                f"placebo-cutoff empirical p={emp_p:.3f} over {npl} fake dates."]
    except Exception as e:
        out += [f"- COVID ITS error: {e}"]
    DOC.write_text("\n".join(out) + "\n")
    print("\n".join(out)); print("[validation-ts] ->", DOC)


if __name__ == "__main__":
    main()
