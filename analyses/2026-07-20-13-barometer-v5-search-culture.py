"""Round 6 / Barometer v5 — add KEYED public-API domains: SEARCH (Google Trends) + CULTURE (horror-share).

Completes the no-approval public-API battery on top of v4:
  - SEARCH: SerpApi Google Trends (per-term, full resolution) — an economic-anxiety worry index
    (recession/layoffs/inflation/anxiety) + war-search attention. US/EN.
  - CULTURE: TMDB monthly horror-genre SHARE of film releases (collective-anxiety-in-entertainment).

Same critic-approved machinery as v3/v4: adoption-detrended dream axis vs raw indicator + symmetric-detrend
robustness; phase-surrogate nulls + BH-FDR (within each indicator's 13-axis family) + neutral control;
EN primary / RU separate; monthly + weekly (culture is monthly-native → step-filled for weekly, low info).
Sign-stable block-bootstrap coherence checks on the new indicators. Aggregate-only; ~2.3-yr → exploratory
(mde_r = single-test alpha-critical |r|, a lower bound).

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 USE_TF=0 python3 \
        analyses/2026-07-20-13-barometer-v5-search-culture.py
"""
from __future__ import annotations

import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psychohistory import config as C


def _load(mod_file, name):
    spec = importlib.util.spec_from_file_location(name, C.ROOT / "analyses" / mod_file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


w24 = _load("2026-07-19-24-collective-wave-poc.py", "wave24")
w2 = _load("2026-07-20-01-collective-barometer-v2.py", "barov2")
v3 = _load("2026-07-20-11-barometer-v3-nonfinancial.py", "barov3")
v4 = _load("2026-07-20-12-barometer-v4-public-apis.py", "barov4")

OUT = C.RESULTS / "showcase"
SIG = C.EXTERNAL / "signals"

NONFIN_DOMAIN = {**v4.NONFIN_DOMAIN, "gtrends_worry": "search", "gtrends_war": "search",
                 "tmdb_horror": "culture"}
WORRY = ["recession", "layoffs", "inflation", "anxiety"]


def load_nonfin_v5(freq, window="2024-02-01"):
    base = v4.load_nonfin_v4(freq, window)

    # SEARCH: Google Trends per-term -> z each term -> worry composite (econ-anxiety) + war
    t = pd.read_csv(SIG / "serpapi_trends.csv", parse_dates=["date"])
    piv = t.pivot_table(index="date", columns="query", values="value", aggfunc="mean").sort_index()
    tz = (piv - piv.mean()) / (piv.std() + 1e-9)
    gt = pd.DataFrame(index=piv.index)
    have = [c for c in WORRY if c in tz.columns]
    if have:
        gt["gtrends_worry"] = tz[have].mean(axis=1)
    if "war" in tz.columns:
        gt["gtrends_war"] = tz["war"]

    # CULTURE: TMDB monthly horror-share (drop low-count months); step-fill to daily
    h = pd.read_csv(SIG / "tmdb_horror.csv", parse_dates=["date"])
    h = h[h["total"] >= 20].set_index("date")["horror_share"].rename("tmdb_horror").sort_index()

    extra = gt.join(h, how="outer").sort_index()
    extra = extra.reindex(pd.date_range(extra.index.min(), extra.index.max(), freq="D")).ffill(limit=40)
    key = (extra.index.to_period("W-SUN").start_time if freq == "W"
           else extra.index.to_period("M").start_time)
    extra = extra.groupby(key).mean()
    panel = base.join(extra, how="outer").sort_index()
    return panel[panel.index >= window]


def direct_battery(cohort, freq="M", detrend_indicator=False):
    nf = load_nonfin_v5(freq)
    if detrend_indicator:
        facs = {c: pd.Series(w24.detrend(nf[c].dropna().values.astype(float),
                                         np.zeros(nf[c].dropna().shape[0])), index=nf[c].dropna().index)
                for c in nf.columns}
    else:
        facs = {c: nf[c].dropna() for c in nf.columns}
    return w2.battery(facs, cohort, freq)


def main():
    DOMAIN = {**w2.DOMAIN, **NONFIN_DOMAIN}
    order = ["wiki_crisis", "wiki_hope", "newsentiment", "gdelt_tone", "gdelt_vol", "manifold_risk",
             "gtrends_worry", "gtrends_war", "tmdb_horror"]
    L = ["# Round 6 / Barometer v5 — SEARCH (Google Trends) + CULTURE (horror-share) added", "",
         "*Completes the no-approval public-API battery: adds SerpApi **Google Trends** (per-term, full "
         "resolution — an econ-anxiety worry index [recession/layoffs/inflation/anxiety] + war-search) and "
         "**TMDB** monthly horror-genre SHARE of releases. Same critic-approved design as v3/v4 "
         "(adoption-detrended dream vs raw indicator; symmetric-detrend robustness; phase-surrogate + "
         "BH-FDR per 13-axis family + neutral control; EN primary, RU separate; monthly + weekly). "
         "`mde_r` = single-test α-critical |r| (lower bound). Aggregate-only, ~2.3 yr → exploratory.*", ""]

    # ---------- (1) DIRECT ----------
    L += ["## (1) DIRECT — does a dream axis track SEARCH or CULTURE, beyond neutral control?", ""]
    direct = {}
    for cohort in ["en", "ru"]:
        R, nmax = direct_battery(cohort, "M")
        direct[cohort] = R
        R.to_csv(OUT / f"barometer_v5_direct_{cohort}.csv", index=False)
        L += [f"### {cohort.upper()} (monthly; n≈{nmax} → single-test α-crit |r|≈{w24.mde_r(nmax):.2f})"]
        for ind in order:
            Rf = R[R.factor == ind]
            if not len(Rf):
                continue
            nn = Rf[Rf.type != "neutral"]; ne = Rf[Rf.type == "neutral"]
            top = nn.iloc[nn.r0.abs().argmax()]
            topn = ne.iloc[ne.r0.abs().argmax()] if len(ne) else None
            nsurv = int((Rf["q0"] < 0.05).sum())
            tag = " ⟵NEW" if ind in ("gtrends_worry", "gtrends_war", "tmdb_horror") else ""
            line = (f"- **{ind}** ({DOMAIN.get(ind,'?')}){tag}: top `{top.axis}`/{top.type} r₀={top.r0:+.3f} "
                    f"(p={top.p0:.3f}, q={top.q0:.3f}); {nsurv}/{len(Rf)} FDR")
            if topn is not None:
                line += f"; Δ|r| vs neutral={abs(top.r0)-abs(topn.r0):+.3f}"
            L.append(line + ".")
        L.append("")

    # symmetric-detrend robustness + weekly power-check (search only; culture is monthly-native)
    L += ["*Symmetric-detrend robustness (linear-detrend indicator too):*"]
    for cohort in ["en", "ru"]:
        Rs, _ = direct_battery(cohort, "M", detrend_indicator=True)
        if len(Rs):
            nsurv = int((Rs["q0"] < 0.05).sum()); top = Rs.iloc[Rs.r0.abs().argmax()]
            L.append(f"- {cohort.upper()}: {nsurv}/{len(Rs)} FDR; strongest |r₀|={abs(top.r0):.3f} "
                     f"(`{top.axis}`/{top.type} vs {top.factor}, q={top.q0:.3f}).")
    L += ["", "## Weekly power-check (~5× points; culture is step-filled monthly → low info)"]
    for cohort in ["en", "ru"]:
        try:
            RW, nW = direct_battery(cohort, "W")
        except Exception as e:
            L.append(f"- {cohort.upper()} weekly failed: {e}"); continue
        if len(RW):
            nsurv = int((RW["q0"] < 0.05).sum()); top = RW.iloc[RW.r0.abs().argmax()]
            L.append(f"- **{cohort.upper()} weekly** (n≈{nW}; α-crit|r|≈{w24.mde_r(nW):.2f}): "
                     f"**{nsurv}/{len(RW)} survive FDR**; strongest |r₀|={abs(top.r0):.3f} "
                     f"(`{top.axis}`/{top.type} vs {top.factor}, q={top.q0:.3f}).")
    L.append("")

    # ---------- (2) BROAD + coherence ----------
    L += ["## (2) BROAD — fold search+culture into the factor + coherence sanity-check"]
    base = w2.load_broad_panel("M")
    nf = load_nonfin_v5("M")
    panelM = base.join(nf, how="inner").interpolate(limit_direction="both").dropna(how="any")
    facs, loadings, var = w2.factors(panelM, k=2)
    panel_v2 = w2.load_broad_panel("M"); f2, _l, _v = w2.factors(panel_v2, k=2)
    j = pd.concat([facs["F1"].rename("v5"), f2["F1"].rename("v2")], axis=1).dropna()
    f1_shift = float(np.corrcoef(j["v5"], j["v2"])[0, 1]) if len(j) > 3 else np.nan
    L += [f"- Panel now {len(panelM.columns)} indicators ({len(panelM)} months). F1(v5)↔F1(v2) r={f1_shift:+.2f}.",
          "", "### External-wave coherence — sign-stable direct r (moving-block bootstrap 95% CI)"]
    for ind, anchor, lbl in [("gtrends_worry", "vix", "Trends worry-search ↔ VIX"),
                             ("gtrends_worry", "unrate", "Trends worry-search ↔ unemployment"),
                             ("tmdb_horror", "vix", "TMDB horror-share ↔ VIX")]:
        if ind in nf.columns and anchor in panelM.columns:
            jj = pd.concat([nf[ind], panelM[anchor]], axis=1).dropna()
            if len(jj) > 6:
                r0, lo, hi = v3._block_boot_r(jj.iloc[:, 0].values, jj.iloc[:, 1].values)
                sig = "excludes 0" if (lo > 0 or hi < 0) else "includes 0 (ns)"
                L.append(f"- {lbl}: r={r0:+.2f} [{lo:+.2f}, {hi:+.2f}] — CI {sig}.")

    for cohort in ["en", "ru"]:
        R, nmax = w2.battery(facs, cohort, "M")
        R.to_csv(OUT / f"barometer_v5_broad_{cohort}.csv", index=False)
        L += ["", f"### {cohort.upper()} broad-factor battery (monthly; n≈{nmax})"]
        for fn in ["F1", "F2"]:
            Rf = R[R.factor == fn]
            if not len(Rf):
                continue
            nsurv = int((Rf["q0"] < 0.05).sum()); top = Rf.iloc[Rf.r0.abs().argmax()]
            L.append(f"- **{fn}**: top `{top.axis}`/{top.type} |r₀|={abs(top.r0):.3f} (q={top.q0:.3f}); "
                     f"{nsurv}/{len(Rf)} survive FDR.")

    fig_v5(direct.get("en"), nf, DOMAIN)
    total = sum(int((direct[c]["q0"] < 0.05).sum()) for c in direct if len(direct[c]))
    L += ["", "**Figure:** 39_barometer_v5.png", "",
          f"*Verdict tier: exploratory. BH-FDR is per-indicator (13-axis family); pooling all tests is "
          f"weakly more conservative → the result holds a fortiori. Adding SEARCH (Google Trends worry + "
          f"war) and CULTURE (horror-share): **{total} dream axis×indicator survive FDR** (monthly, both "
          f"cohorts). {'A broad bounded null holds' if total == 0 else 'A survivor appeared — inspect'} for "
          "**CONTINUOUS aggregate coupling** across markets + attention + news-tone×2 + news-volume + "
          "prediction + search×2 + culture (EN/US). **Three scope limits — do NOT over-generalize:** "
          "(1) this is the continuous-coupling design (+ reverse-anomaly F0043); the repo's pre-registered "
          "PRIMARY **forward event-study / case-crossover around curated shocks (METHODS §3A) has NOT been "
          "run** → event-locked transient responses remain untested; the hypothesis is narrowed, not closed. "
          "(2) all indicators are **EN/US** → the breadth holds for the EN/US population only; RU (~20%) / "
          "other (~15%) coupling stays OPEN pending language-matched signals (RU tested vs EN referents, "
          "n=14 monthly). (3) the |r|≳0.25–0.30 bound is **EN-weekly**; the TMDB culture arm is monthly-only, "
          "low-power, production-lagging, and itself decoupled from the barometer (horror↔VIX ns) → weakest "
          "evidence, down-weighted. Consolidates the F0037→F0047 continuous-coupling thread.*"]
    (OUT / "barometer_v5.md").write_text("\n".join(L))
    print("\n".join(L)); print("\n[barometer-v5] wrote", OUT / "barometer_v5.md")


def fig_v5(direct_en, nf, DOMAIN):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    if direct_en is not None and len(direct_en):
        Rf = direct_en[direct_en.factor == "gtrends_worry"].sort_values("r0", key=lambda c: c.abs())
        if len(Rf):
            cols = ["#9aa0a6" if t == "neutral" else "#C1443C" for t in Rf.type]
            ax.barh(Rf.axis, Rf.r0, color=cols); ax.axvline(0, color="k", lw=.6)
            ax.set_title("EN: dream axes vs Google-Trends worry\n(grey=neutral)")
            ax.set_xlabel("contemporaneous r")
    ax = axes[1]
    for c, col in [("gtrends_worry", "#C1443C"), ("tmdb_horror", "#8856a7"), ("gtrends_war", "#e08a1e")]:
        if c in nf.columns:
            s = nf[c].dropna(); s = (s - s.mean()) / (s.std() + 1e-9)
            ax.plot(pd.to_datetime(s.index).to_numpy(), s.to_numpy(), "-o", ms=3, label=c)
    ax.axhline(0, color="k", lw=.5); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Search + culture signals (z, monthly)")
    ax = axes[2]
    # dream anxiety vs worry-search (both z, monthly) — the headline comparison
    try:
        axd = w24.dream_axes("M", "en")
        s = axd["neg_sent"].dropna()
        sd = pd.Series(w24.detrend(s.values.astype(float), axd.loc[s.index, "logn"].values), index=s.index)
        sd = (sd - sd.mean()) / sd.std()
        ax.plot(pd.to_datetime(sd.index).to_numpy(), sd.to_numpy(), "-o", ms=3, color="#4C72B0",
                label="EN dream neg-sentiment (detrended)")
    except Exception:
        pass
    if "gtrends_worry" in nf.columns:
        w = nf["gtrends_worry"].dropna(); w = (w - w.mean()) / (w.std() + 1e-9)
        ax.plot(pd.to_datetime(w.index).to_numpy(), w.to_numpy(), "-o", ms=3, color="#C1443C",
                label="Google worry-search")
    ax.axhline(0, color="k", lw=.5); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Dream negativity vs worry-search (z)")
    fig.tight_layout(); fig.savefig(OUT / "39_barometer_v5.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
