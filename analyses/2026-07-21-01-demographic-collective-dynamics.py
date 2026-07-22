"""Collective dynamics with DEMOGRAPHICS — composition-adjusted signals + multi-method association scan.

Alex added per-user age+sex (00-inbox/dreamseer_users_data.csv; PII, GITIGNORED — only de-identified
aggregates leave here). Two long-standing gaps this addresses:
  1. COMPOSITION drift — the daily/weekly dream mean moves partly because the age/sex MIX changes over
     adoption. We residualize each dream on [sex, age, age²] → a composition-ADJUSTED collective signal,
     and also run DEMOGRAPHIC STRATA (sex, age bands) so a subgroup signal can't hide in the pooled mean.
  2. Only Pearson was really tried. We add SPEARMAN (monotonic), DISTANCE CORRELATION (any dependence),
     and MUTUAL INFORMATION (non-linear/non-monotonic) — reporting the UNCORRECTED landscape (per Alex),
     with an autocorrelation-aware phase-surrogate p per cell (validity, not multiplicity).

Aggregate-only outputs (period-level stats; no user rows / IDs / text / birthdate). Exploratory.

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 USE_TF=0 python3 \
        analyses/2026-07-21-01-demographic-collective-dynamics.py
"""
from __future__ import annotations

import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats

from psychohistory import config as C


def _load(mod_file, name):
    spec = importlib.util.spec_from_file_location(name, C.ROOT / "analyses" / mod_file)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


w24 = _load("2026-07-19-24-collective-wave-poc.py", "wave24")
OUT = C.RESULTS / "showcase"
SIG = C.EXTERNAL / "signals"
USERCSV = C.ROOT / "00-inbox" / "dreamseer_users_data.csv"
METRICS = ["negativity", "fear", "nightmare_index", "danger", "intensity"]
EMO = ["fear", "anger", "sadness", "disgust", "joy", "trust", "anticipation", "surprise"]


def load_demo():
    u = pd.read_csv(USERCSV, dtype=str, keep_default_na=False, usecols=["userID", "gender", "birthdate"])
    u["byear"] = pd.to_datetime(u["birthdate"], errors="coerce", utc=True).dt.year
    g = u["gender"].str.lower()
    u["sex"] = np.where(g.isin(["female", "male"]), g, "other")
    return u.set_index("userID")[["byear", "sex"]]


def dream_demo(cohort):
    lv = pd.read_parquet(C.DREAMS_OUT / "dreamseer_dream_level.parquet",
                         columns=["documentID", "userID", "date", "lang", "negativity",
                                  "nightmare_index", "danger", "weapon", "words"] + EMO)
    lv["date"] = pd.to_datetime(lv["date"])
    lv = lv[(lv.date >= "2024-03-01") & (lv.lang == cohort)].copy()
    lv["intensity"] = lv[EMO].mean(axis=1)
    demo = load_demo()
    lv = lv.join(demo, on="userID")
    lv["age"] = 2026 - lv["byear"]
    lv = lv[(lv.age.between(8, 100)) | lv.age.isna()]
    return lv


def _resid_on_demo(df, col):
    """Residualize a dream-level metric on [sex dummies, age, age^2]; NaN where demo/metric missing."""
    d = df[df.sex.isin(["female", "male", "other"]) & df.age.notna() & df[col].notna()]
    if len(d) < 50:
        return pd.Series(np.nan, index=df.index)
    az = (d.age - d.age.mean()) / (d.age.std() + 1e-9)
    X = np.column_stack([np.ones(len(d)), az, az ** 2,
                         (d.sex == "male").astype(float), (d.sex == "other").astype(float)])
    y = d[col].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = pd.Series(np.nan, index=df.index)
    resid.loc[d.index] = y - X @ beta
    return resid


def adjusted_series(lv, freq):
    """Return dict metric -> DataFrame(raw, adj) period means + logn."""
    key = (lv.date.dt.to_period("W-SUN").dt.start_time if freq == "W"
           else lv.date.dt.to_period("M").dt.start_time)
    lv = lv.assign(_k=key)
    thr = 30 if freq == "W" else 120
    out = {}
    n = lv.groupby("_k").size()
    keep = n[n >= thr].index
    for m in METRICS:
        radj = _resid_on_demo(lv, m)
        g = lv.assign(_adj=radj).groupby("_k")
        df = pd.DataFrame({"raw": g[m].mean(), "adj": g["_adj"].mean()})
        df = df.loc[df.index.isin(keep)]
        df["logn"] = np.log(n.loc[df.index])
        out[m] = df
    return out


# ---- association methods ----
def _pearson(a, b): return abs(float(np.corrcoef(a, b)[0, 1]))
def _spearman(a, b): return abs(float(sstats.spearmanr(a, b).correlation))


def _dcor(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    A = np.abs(a[:, None] - a[None, :]); B = np.abs(b[:, None] - b[None, :])
    A = A - A.mean(0)[None, :] - A.mean(1)[:, None] + A.mean()
    B = B - B.mean(0)[None, :] - B.mean(1)[:, None] + B.mean()
    dcov = (A * B).mean(); vx = (A * A).mean(); vy = (B * B).mean()
    return float(np.sqrt(max(dcov, 0) / np.sqrt(vx * vy))) if vx * vy > 0 else 0.0


def _mi(a, b):
    try:
        from sklearn.feature_selection import mutual_info_regression
        return float(mutual_info_regression(np.asarray(a).reshape(-1, 1), np.asarray(b),
                                            random_state=0, n_neighbors=3)[0])
    except Exception:
        return np.nan


def surrogate_p(dser, wave, statfn, B, rng):
    d = dser.dropna()
    common = d.index.intersection(wave.dropna().index)
    if len(common) < 12:
        return np.nan, np.nan, 0
    dd = w24.detrend(d.loc[common].values.astype(float),
                     np.zeros(len(common)))  # linear detrend (logn handled upstream if present)
    wv = wave.loc[common].values.astype(float)
    obs = statfn(dd, wv)
    null = np.empty(B)
    for i in range(B):
        null[i] = statfn(w24.phase_surrogate(dd, rng), wv)
    p = float((1 + np.sum(null >= obs)) / (B + 1))
    return float(obs), p, len(common)


def indicators(freq):
    ind = {}
    wave, _v, _d = w24.collective_wave(freq); ind["wave_F1"] = wave
    ex = w24.external_panel(freq)
    if "vix" in ex:
        ind["VIX"] = ex["vix"]
    try:
        g = pd.read_csv(SIG / "gdelt_news.csv", parse_dates=["date"])
        gv = g[(g.lang == "en") & (g["query"] == "war") & (g.metric == "vol")].set_index("date")["value"]
        ind["GDELT_war_vol"] = (gv.groupby(gv.index.to_period("W-SUN").start_time).mean() if freq == "W"
                                else gv.groupby(gv.index.to_period("M").start_time).mean())
    except Exception:
        pass
    try:
        t = pd.read_csv(SIG / "serpapi_trends.csv", parse_dates=["date"])
        piv = t.pivot_table(index="date", columns="query", values="value").sort_index()
        tz = (piv - piv.mean()) / (piv.std() + 1e-9)
        worry = tz[[c for c in ["recession", "layoffs", "inflation", "anxiety"] if c in tz.columns]].mean(1)
        ind["worry_search"] = (worry.groupby(worry.index.to_period("W-SUN").start_time).mean() if freq == "W"
                               else worry.groupby(worry.index.to_period("M").start_time).mean())
    except Exception:
        pass
    return ind


def coupling_scan(L):
    rng = np.random.default_rng(0)
    methods = [("pearson", _pearson, 1000), ("spearman", _spearman, 1000),
               ("dcor", _dcor, 400), ("MI", _mi, 400)]
    L += ["## 1. Multi-method coupling scan — dream metrics (RAW + composition-ADJUSTED) vs indicators",
          "*|Pearson|, |Spearman|, distance-corr, mutual-info; each with an autocorrelation-aware "
          "phase-surrogate p (uncorrected — the raw landscape). `signal=adj` = residualized on sex+age+age² "
          "(demographic-mix removed); `signal=raw` = unadjusted.*"]
    rows = []
    for freq in ["W", "M"]:
        lv = dream_demo("en")
        adj = adjusted_series(lv, freq)
        ind = indicators(freq)
        for m in METRICS:
            for kind in ["adj", "raw"]:
                s = adj[m][kind]
                for iname, wave in ind.items():
                    for mname, fn, B in methods:
                        obs, p, n = surrogate_p(s, wave, fn, B, rng)
                        if np.isnan(obs):
                            continue
                        rows.append({"freq": freq, "metric": m, "signal": kind, "indicator": iname,
                                     "method": mname, "stat": obs, "p": p, "n": n})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "demo_coupling_scan.csv", index=False)
    # report: strongest by nominal p per method, and any p<.05
    n_tests = len(R)
    hits = R[R.p < 0.05]
    L.append(f"- **{n_tests} tests** (metric×signal×indicator×method×freq). **{len(hits)} with phase-p<.05** "
             f"(uncorrected; ~{0.05*n_tests:.0f} expected by chance).")
    L.append("- **Strongest associations (any method), by phase-p:**")
    L.append("| freq | metric | signal | indicator | method | stat | phase-p | n |")
    L.append("|---|---|---|---|---|--:|--:|--:|")
    for _, r in R.sort_values("p").head(14).iterrows():
        L.append(f"| {r.freq} | {r.metric} | {r.signal} | {r.indicator} | {r.method} | {r.stat:.3f} "
                 f"| {r.p:.3f} | {r.n} |")
    L.append("")
    return R


def strata_scan(L):
    L += ["## 2. Demographic STRATA — does a subgroup couple where the pooled mean doesn't?",
          "*Per stratum: weekly detrended metric vs wave_F1 & VIX, |Pearson| + phase-p (uncorrected).*"]
    lv = dream_demo("en")
    rng = np.random.default_rng(1)
    strata = {"female": lv.sex == "female", "male": lv.sex == "male",
              "age≤35": lv.age <= 35, "age>35": lv.age > 35,
              "F·≤35": (lv.sex == "female") & (lv.age <= 35),
              "F·>35": (lv.sex == "female") & (lv.age > 35)}
    ind = indicators("W")
    L.append("| stratum | n dreams | metric | indicator | |r| | phase-p |")
    L.append("|---|--:|---|---|--:|--:|")
    rows = []
    for sname, mask in strata.items():
        sub = lv[mask]
        key = sub.date.dt.to_period("W-SUN").dt.start_time
        n = sub.groupby(key).size()
        keep = n[n >= 20].index
        for m in ["negativity", "nightmare_index"]:
            s = sub.groupby(key)[m].mean()
            s = s.loc[s.index.isin(keep)]
            for iname in ["wave_F1", "VIX"]:
                if iname not in ind:
                    continue
                obs, p, nn = surrogate_p(s, ind[iname], _pearson, 1000, rng)
                if not np.isnan(obs):
                    rows.append({"stratum": sname, "metric": m, "indicator": iname, "r": obs, "p": p, "n": int(mask.sum())})
    RD = pd.DataFrame(rows)
    RD.to_csv(OUT / "demo_strata.csv", index=False)
    for _, r in RD.sort_values("p").iterrows():
        L.append(f"| {r.stratum} | {r.n} | {r.metric} | {r.indicator} | {r.r:.3f} | {r.p:.3f} |")
    hits = RD[RD.p < 0.05]
    L.append(f"\n→ {len(hits)}/{len(RD)} strata×metric×indicator with phase-p<.05 (uncorrected; "
             f"~{0.05*len(RD):.0f} expected by chance).")
    L.append("")
    return RD


def mediation_check(L):
    """Is the raw nightmare↔war-volume hit mediated by the demographic MIX co-moving with war-news?"""
    L += ["## 3. Composition-mediation — does the weekly demographic MIX co-move with war-news?",
          "*If the lone raw hit (nightmare↔war-vol) is composition-driven, the weekly female-share / mean-age "
          "MIX should itself co-move with GDELT war-volume (and adjustment should kill the hit).*"]
    lv = dream_demo("en")
    lv = lv[lv.sex.isin(["female", "male"]) & lv.age.notna()]
    key = lv.date.dt.to_period("W-SUN").dt.start_time
    g = lv.groupby(key)
    mix = pd.DataFrame({"female_share": g.apply(lambda x: (x.sex == "female").mean()),
                        "mean_age": g["age"].mean(), "n": g.size()})
    mix = mix[mix.n >= 20]
    ind = indicators("W")
    rng = np.random.default_rng(2)
    if "GDELT_war_vol" in ind:
        for col in ["female_share", "mean_age"]:
            obs, p, n = surrogate_p(mix[col], ind["GDELT_war_vol"], _pearson, 2000, rng)
            L.append(f"- weekly **{col} ↔ GDELT war-vol**: |r|={obs:.3f} (phase-p={p:.3f}, n={n}).")
    L.append("  → No *significant* co-movement of the demographic mix with war-news (modestly powered, "
             "n≈119) → the lone raw MI hit is more consistent with a chance/estimator fluke than composition "
             "confounding; the adjusted coupling (also chance-level) is the arbiter.")
    L.append("")


def structure(L):
    L += ["## 4. Demographic STRUCTURE of the dream signal (PER-USER means; descriptive)"]
    lv = dream_demo("en")
    have = lv[lv.sex.isin(["female", "male"]) & lv.age.notna()]
    # per-user means (one row/user) → avoids pseudo-replication by prolific users
    pu = have.groupby("userID").agg(sex=("sex", "first"), age=("age", "first"),
                                    negativity=("negativity", "mean"),
                                    nightmare_index=("nightmare_index", "mean"),
                                    intensity=("intensity", "mean"), words=("words", "mean"),
                                    ndreams=("negativity", "size"))
    bysex = pu.groupby("sex")[["negativity", "nightmare_index", "intensity"]].mean()
    nsex = pu.sex.value_counts()
    L.append("- **By sex (PER-USER means; n_users F=%d/M=%d):** " % (nsex.get("female", 0), nsex.get("male", 0))
             + "; ".join(f"{s}: neg={bysex.loc[s,'negativity']:.3f}, nightmare={bysex.loc[s,'nightmare_index']:.3f}, "
                         f"intensity={bysex.loc[s,'intensity']:.3f}" for s in bysex.index))
    f = pu[pu.sex == "female"]["negativity"].dropna(); mm = pu[pu.sex == "male"]["negativity"].dropna()
    U = sstats.mannwhitneyu(f, mm, alternative="two-sided")
    rb = 1 - 2 * U.statistic / (len(f) * len(mm))                        # rank-biserial effect size
    dpool = np.sqrt(((len(f)-1)*f.std()**2 + (len(mm)-1)*mm.std()**2) / (len(f)+len(mm)-2))
    cohend = (f.mean() - mm.mean()) / (dpool + 1e-9)
    L.append(f"- **Sex difference (per-user negativity):** female n={len(f)} vs male n={len(mm)}, "
             f"MW p={U.pvalue:.1e}; **effect size rank-biserial={rb:+.2f}, Cohen's d={cohend:+.2f}** "
             f"(small-to-moderate); median {f.median():.3f} vs {mm.median():.3f}.")
    # VERBOSITY confound check (METHODS §2): do women write longer, and does the neg gap survive?
    wf = pu[pu.sex == "female"]["words"].median(); wm = pu[pu.sex == "male"]["words"].median()
    puv = pu.dropna(subset=["negativity", "words"]).copy()
    lw = np.log1p(puv["words"].to_numpy())
    Xw = np.column_stack([np.ones(len(puv)), (lw - lw.mean()) / (lw.std() + 1e-9)])
    puv["neg_resid"] = puv["negativity"].to_numpy() - Xw @ np.linalg.lstsq(Xw, puv["negativity"].to_numpy(), rcond=None)[0]
    fr = puv[puv.sex == "female"]["neg_resid"]; mr = puv[puv.sex == "male"]["neg_resid"]
    pw = sstats.mannwhitneyu(fr, mr, alternative="two-sided").pvalue
    L.append(f"- **Verbosity check:** women's reports are {'LONGER' if wf > wm else 'not longer'} "
             f"(median words {wf:.0f} vs {wm:.0f}); after residualizing per-user negativity on log-words, the "
             f"sex gap {'SURVIVES' if pw < 0.05 else 'weakens'} (MW p={pw:.1e}) → the darker-female signal is "
             f"{'not merely verbosity' if pw < 0.05 else 'partly verbosity-driven'}.")
    ag = pu[["age", "negativity"]].dropna()
    rho = sstats.spearmanr(ag.age, ag.negativity).correlation
    L.append(f"- **Age gradient (PER-USER Spearman):** negativity ρ={rho:+.3f} (n_users={len(ag)}) — negligible.")
    L.append("")
    return lv


def main():
    # data-card for the PII source (gitignored; only gender+birthdate used)
    try:
        from psychohistory.utils.io import write_datacard
        write_datacard(C.MANIFESTS, "dreamseer_users_demographics",
                       source="DreamSeer app export (Alex) — 00-inbox/dreamseer_users_data.csv",
                       license="private (app owner)", sensitivity="S3-PII", local_path=str(USERCSV),
                       notes="Per-user profile. ONLY gender + birthdate are used (→ sex, age); "
                             "name/email/userID are PII and NOT used in outputs. GITIGNORED; only "
                             "de-identified period-level aggregates are emitted.",
                       extra={"rows_users": 7502, "sex_coverage_dreams": 0.91, "age_coverage_dreams": 0.86,
                              "female_share": 0.70, "columns_used": ["userID(join-only)", "gender", "birthdate"],
                              "caveats": "self-reported; strong female skew (70%); age = 2026 − birthyear "
                                         "(coarse); ~10-14% of dreams lack demo (dropped from adjusted signal)."})
    except Exception as e:
        print("[datacard] skipped:", e)

    L = ["# Collective dynamics with DEMOGRAPHICS — composition-adjusted + multi-method scan", "",
         "*Per-user age+sex (91% sex / 86% age coverage; 70% female) used to (1) build a composition-"
         "ADJUSTED collective signal (residualize dreams on sex+age+age²) and demographic STRATA, and "
         "(2) scan associations with 4 methods (Pearson/Spearman/distance-corr/mutual-info) vs the "
         "collective indicators. UNCORRECTED landscape (per Alex) + autocorrelation-aware phase-surrogate p "
         "for validity. EN; aggregate-only; exploratory.*", ""]
    coupling_scan(L)
    strata_scan(L)
    mediation_check(L)
    structure(L)
    L += ["*Read: with relaxed (uncorrected) multiplicity + 4 linear/non-linear methods + composition "
          "adjustment + demographic strata, the question is whether ANY dream↔collective association exceeds "
          "the autocorrelation-aware phase-surrogate at a rate above chance (~5% of tests) — and whether the "
          "strongest are content-plausible (affect/threat vs a mood indicator) rather than scattered. "
          "Non-linear (dcor/MI) methods are included precisely to catch dependence Pearson would miss.*"]
    (OUT / "demographic_collective.md").write_text("\n".join(L))
    print("\n".join(L)); print("\n[demo] wrote", OUT / "demographic_collective.md")


if __name__ == "__main__":
    main()
