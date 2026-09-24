"""Run the full analysis battery and rank candidate findings by impact.

Deterministic, unattended-safe (each cell wrapped in try/except). Produces:
  - event-study results (family x cohort x outcome), FDR-controlled
  - coupling results (signal x cohort x outcome x lag), FDR-controlled across ALL lag tests
  - a cross-cultural (EN vs RU) divergence table
  - a ranked findings table + a human-readable report, most-impactful first

Everything is EXPLORATORY unless preregistered + externally replicated (docs/METHODS.md).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .event_study import event_study
from .coupling import lagged_crosscorr
from .inference import benjamini_hochberg

# Outcomes: primary dysphoria/affect + a few tags; last two are NEGATIVE CONTROLS.
PRIMARY_OUTCOMES = ["fear_mean", "anger_mean", "sadness_mean", "negativity_mean",
                    "nightmare_index_mean", "danger_mean", "trust_mean", "joy_mean"]
NEG_CONTROL_OUTCOMES = ["food_mean", "music_mean"]
ALL_OUTCOMES = PRIMARY_OUTCOMES + NEG_CONTROL_OUTCOMES


# ---------------- event battery ----------------
def run_event_battery(daily, events_by_family, langs, outcomes=ALL_OUTCOMES,
                      n_perm=20000, post=(1, 7), pre=(-21, -4)):
    rows = []
    for fam, ev in events_by_family.items():
        dates = list(ev["date"]) if len(ev) else []
        if len(dates) < 5:
            continue
        for lang in langs:
            dl = daily[daily.lang == lang]
            for oc in outcomes:
                if oc not in dl.columns:
                    continue
                try:
                    r = event_study(dl, dates, oc, post=post, pre=pre, n_perm=n_perm)
                except Exception as e:  # keep the battery alive
                    r = {"n_events": 0, "error": str(e)}
                if r.get("n_events", 0) >= 5:
                    sd = r.get("null_sd") or np.nan
                    z = r["effect_post_minus_pre"] / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
                    rows.append({"category": "event_study", "family": fam, "cohort": lang,
                                 "outcome": oc, "is_neg_control": oc in NEG_CONTROL_OUTCOMES,
                                 "n": r["n_events"], "effect": r["effect_post_minus_pre"],
                                 "z": z, "p": r["perm_p"]})
    df = pd.DataFrame(rows)
    if len(df):
        # FDR over the non-negative-control tests only
        mask = ~df.is_neg_control
        q = np.full(len(df), np.nan)
        if mask.any():
            q[mask.values] = benjamini_hochberg(df.loc[mask, "p"].to_numpy())
        df["q"] = q
    return df


# ---------------- coupling battery ----------------
def run_coupling_battery(dream_weekly, panel, langs, signals, outcomes=ALL_OUTCOMES,
                         max_lag=8, n_perm=20000, min_n=30):
    full = []
    for lang in langs:
        dw = dream_weekly[dream_weekly.lang == lang]
        for sig in signals:
            if sig not in panel.columns:
                continue
            sig_w = panel[["week", sig]].dropna()
            for oc in outcomes:
                if oc not in dw.columns:
                    continue
                try:
                    cc = lagged_crosscorr(dw, sig_w, oc, sig, max_lag=max_lag, n_perm=n_perm)
                except Exception:
                    continue
                cc = cc[cc.n >= min_n]
                if cc.empty:
                    continue
                cc = cc.assign(signal=sig, cohort=lang, outcome=oc,
                               is_neg_control=oc in NEG_CONTROL_OUTCOMES)
                full.append(cc)
    if not full:
        return pd.DataFrame(), pd.DataFrame()
    full = pd.concat(full, ignore_index=True)
    # FDR across ALL lag tests (non-neg-control) -> honest about the lag search
    mask = ~full.is_neg_control
    q = np.full(len(full), np.nan)
    if mask.any():
        q[mask.values] = benjamini_hochberg(full.loc[mask, "perm_p"].to_numpy())
    full["q"] = q
    # best (min-q) row per (cohort, signal, outcome) for the ranked report
    best = (full[~full.is_neg_control].sort_values("q")
            .groupby(["cohort", "signal", "outcome"], as_index=False).first())
    return full, best


# ---------------- cross-cultural (EN vs RU) ----------------
def cross_cultural_divergence(coupling_best):
    if coupling_best.empty:
        return pd.DataFrame()
    piv = coupling_best.pivot_table(index=["signal", "outcome"], columns="cohort", values="r")
    if not {"en", "ru"}.issubset(piv.columns):
        return pd.DataFrame()
    piv = piv.dropna(subset=["en", "ru"]).reset_index()
    piv["abs_divergence"] = (piv["en"] - piv["ru"]).abs()
    return piv.sort_values("abs_divergence", ascending=False)


# ---------------- ranking ----------------
def _impact(effect_mag, q, weight):
    q = min(max(q, 1e-6), 1.0) if np.isfinite(q) else 1.0
    return float(effect_mag) * (-np.log10(q)) * weight


def rank_findings(event_df, coupling_best):
    items = []
    if len(event_df):
        for _, r in event_df[~event_df.is_neg_control].iterrows():
            if not np.isfinite(r.get("z", np.nan)) or not np.isfinite(r.get("q", np.nan)):
                continue
            direction = "rises" if r.effect > 0 else "falls"
            head = (f"After **{r.family}** events, **{r.cohort.upper()}** dream "
                    f"`{r.outcome.replace('_mean','')}` {direction} vs baseline "
                    f"(z={r.z:+.1f}, {int(r.n)} events)")
            items.append({"category": "event_study", "cohort": r.cohort, "headline": head,
                          "effect": round(float(r.effect), 4), "stat": f"z={r.z:+.2f}",
                          "p": round(float(r.p), 4), "q": round(float(r.q), 4), "n": int(r.n),
                          "reference": f"event_results.csv [{r.family}/{r.cohort}/{r.outcome}]",
                          "impact": _impact(abs(r.z), r.q, 1.2)})
    if len(coupling_best):
        for _, r in coupling_best.iterrows():
            if not np.isfinite(r.get("q", np.nan)):
                continue
            lead = "signal leads dreams" if r.lag_weeks > 0 else ("dreams lead signal" if r.lag_weeks < 0 else "contemporaneous")
            head = (f"**{r.cohort.upper()}** dream `{r.outcome.replace('_mean','')}` couples with "
                    f"**{r.signal}** at lag {int(r.lag_weeks):+d}w (r={r.r:+.2f}; {lead})")
            items.append({"category": "coupling", "cohort": r.cohort, "headline": head,
                          "effect": round(float(r.r), 3), "stat": f"r={r.r:+.2f}@{int(r.lag_weeks):+d}w",
                          "p": round(float(r.perm_p), 4), "q": round(float(r.q), 4), "n": int(r.n),
                          "reference": f"coupling_results.csv [{r.cohort}/{r.signal}/{r.outcome}/lag{int(r.lag_weeks)}]",
                          "impact": _impact(abs(r.r), r.q, 1.0)})
    ranked = pd.DataFrame(items)
    if ranked.empty:
        return ranked
    ranked = ranked.sort_values("impact", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    m = ranked["impact"].max() or 1.0
    ranked["impact_0_100"] = (100 * ranked["impact"] / m).round(1)
    return ranked


def _tier(q):
    if not np.isfinite(q):
        return "n/a"
    if q < 0.05:
        return "headline (q<0.05)"
    if q < 0.20:
        return "suggestive (q<0.20)"
    return "not significant"


def write_report(ranked, event_df, coupling_full, xcult, path, meta):
    L = ["# Overnight findings — ranked by impact (most mind-blowing first)", "",
         f"*Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
         f"n_perm={meta.get('n_perm')}, cohorts={meta.get('langs')}, "
         f"event families={meta.get('families')}, signals={meta.get('signals')}.*", "",
         "> **EXPLORATORY.** Not preregistered; single-corpus (DreamSeer) unless noted. "
         "Nothing here is a confirmed finding until it passes the methodology-critic AND "
         "replicates externally (Reddit) or reproduces a published positive control "
         "(see docs/METHODS.md §5). FDR is across the non-negative-control test family.", ""]

    if ranked.empty:
        L.append("_No rankable results (check data coverage)._")
    else:
        L.append("## Top candidates")
        for _, r in ranked.head(25).iterrows():
            L.append(f"{int(r['rank'])}. **[{r['impact_0_100']:.0f}/100]** {r['headline']}  ")
            L.append(f"   · {r['stat']}, perm p={r['p']}, **q={r['q']}** ({_tier(r['q'])}), "
                     f"N={r['n']} · `{r['reference']}`")
        # negative-control sanity
        L += ["", "## Negative-control sanity (should be mostly null)"]
        nc = event_df[event_df.is_neg_control] if len(event_df) else pd.DataFrame()
        if len(nc):
            sig_nc = nc[nc.p < 0.05]
            L.append(f"- event-study neg-controls firing at p<0.05: {len(sig_nc)}/{len(nc)} "
                     f"({'OK' if len(sig_nc) <= 0.1*len(nc)+1 else 'WATCH — possible leakage'})")

    if len(xcult):
        L += ["", "## Biggest EN vs RU divergences (cross-cultural / socionomic-distance leads)"]
        for _, r in xcult.head(10).iterrows():
            L.append(f"- **{r['signal']} ~ {r['outcome'].replace('_mean','')}**: "
                     f"r(EN)={r['en']:+.2f} vs r(RU)={r['ru']:+.2f} (|Δ|={r['abs_divergence']:.2f})")

    L += ["", "## How to read this", "- **impact** = |effect| × −log10(q) × design-weight, scaled 0-100.",
          "- Promote a candidate only after: methodology-critic PASS, negative controls clean, "
          "and external replication. Then move it into `50-wiki/findings.md`.",
          "- Full tables: `event_results.csv`, `coupling_results.csv`, `cross_cultural.csv`."]
    path.write_text("\n".join(L))
