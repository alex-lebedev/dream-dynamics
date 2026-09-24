"""Death / contagion / contamination themes in DreamSeer: seasonality + curated shocks.

Two jobs, both using the same declared lexicons as the Mallett COVID test so
the two corpora are measured on one instrument
(``psychohistory.dreams.lexicons``).

1. **Independent seasonality check.** The Mallett analysis found that contagion,
   mask and illness content carry annual structure while death, contamination
   and the neutral controls do not — which is what makes an uncontrolled
   March-2020 cutoff dangerous. That estimate rests on 63 pre-COVID weeks, only
   1.2 annual cycles, where a harmonic is barely separable from smooth trend.
   DreamSeer's dense window gives ~2.3 cycles on a different population,
   platform and elicitation, and supports a within-person arm that Reddit does
   not. If the seasonal claim is real it should appear here too.

2. **Content-matched event study.** The repo's forward event-study (F0050)
   tested affect and threat outcomes around ten curated shocks and was null.
   Death *content* was never tested, and one of the ten events is a
   death-category shock (Pope Francis, 2025-04-21) which yields a directional
   content-matched prediction. Contagion and contamination have no matching
   event in the window, so in-window they serve as extra negative controls.

Only aggregates are written; the lexicon is scored on the gitignored raw text
locally and nothing below the release floor leaves this script.

Run: PYTHONPATH=src python3 analyses/2026-08-22-11-dreamseer-content-events.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from psychohistory import config as C
from psychohistory.dreams import lexicons as L
from psychohistory.stats.event_study import event_study
from psychohistory.stats.inference import benjamini_hochberg

DENSE_START = "2024-03-25"
MIN_DREAMS = 15
HARMONICS = ["sin1", "cos1", "sin2", "cos2"]
CACHE = C.INTERIM / f"dreamseer_lexicon_scored_{L.version()}.parquet"

THREAT = {"war", "violence", "econ"}
LATENT = list(L.LATENT)
NEUTRAL = list(L.NEUTRAL)
TESTED = LATENT + list(L.COMPARATOR) + NEUTRAL


# --------------------------------------------------------------- corpus ----
def load_scored(rebuild: bool = False) -> pd.DataFrame:
    """Dream-level lexicon hits for the English cohort (aggregate-safe columns only)."""
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)

    raw = pd.read_csv(C.DREAMSEER_RAW, dtype=str, keep_default_na=False)
    text = raw["text"].fillna("")
    hits = L.score(text)
    meta = pd.DataFrame({
        "documentID": raw["documentID"].astype(str),
        "n_words": text.str.split().str.len().fillna(0).astype(int),
    })
    scored = pd.concat([meta, hits], axis=1)

    dl = pd.read_parquet(C.DREAMS_OUT / "dreamseer_dream_level.parquet")
    dl["documentID"] = dl["documentID"].astype(str)
    d = dl[["documentID", "userID", "date", "lang", "words"]].merge(
        scored.drop(columns=["n_words"]), on="documentID", how="inner")
    d["date"] = pd.to_datetime(d["date"])

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(CACHE, index=False)
    return d


def daily(d: pd.DataFrame, lang: str = "en", person_adjust: bool = False) -> pd.DataFrame:
    """Daily presence-rates, optionally after removing each contributor's own mean.

    Person-adjustment is the composition control the manuscript's population-state
    layer uses: a day's level is then the average *deviation* of that day's
    contributors from their own habit, so turnover in who is writing cannot
    manufacture a temporal pattern.
    """
    x = d[(d.lang == lang) & (d.date >= DENSE_START)].copy()
    axes = list(L.ALL_AXES)
    if person_adjust:
        for a in axes:
            x[a] = x[a] - x.groupby("userID")[a].transform("mean")
    agg = x.groupby("date").agg(
        n_dreams=("documentID", "size"),
        n_users=("userID", "nunique"),
        words_mean=("words", "mean"),
        **{f"{a}_mean": (a, "mean") for a in axes},
    ).reset_index()
    doy = agg.date.dt.dayofyear.to_numpy(dtype=float)
    for h in (1, 2):
        agg[f"sin{h}"] = np.sin(2 * np.pi * h * doy / 365.25)
        agg[f"cos{h}"] = np.cos(2 * np.pi * h * doy / 365.25)
    return agg


# ---------------------------------------------------------- seasonality ----
def seasonality(agg: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Joint test that the annual harmonics are zero, weighted by daily volume."""
    w = agg[agg.n_dreams >= MIN_DREAMS].copy()
    t = np.arange(len(w), dtype=float)
    rows = []
    for a in TESTED:
        y = w[f"{a}_mean"].to_numpy(dtype=float)
        X = sm.add_constant(pd.DataFrame(
            {"t": t, "logn": np.log(w.n_dreams.to_numpy()),
             "log_words": np.log(w.words_mean.to_numpy()),
             **{h: w[h].to_numpy() for h in HARMONICS}}))
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 14})
        cols = list(X.columns)
        R = np.zeros((len(HARMONICS), len(cols)))
        for i, h in enumerate(HARMONICS):
            R[i, cols.index(h)] = 1.0
        amp = float(np.hypot(res.params["sin1"], res.params["cos1"]))
        base = float(w[f"{a}_mean"].mean()) if arm == "raw" else float(
            agg[f"{a}_mean"].abs().mean())
        rows.append({"axis": a, "family": L.family_of(a), "arm": arm, "n_days": len(w),
                     "annual_amp": amp, "amp_rel": amp / base if base > 0 else np.nan,
                     "harmonics_p": float(res.f_test(R).pvalue)})
    out = pd.DataFrame(rows)
    out["q"] = out.groupby("family")["harmonics_p"].transform(
        lambda p: benjamini_hochberg(p.to_numpy()))
    return out


# --------------------------------------------------------- event study ----
def detrend_local(agg: pd.DataFrame, axes: list[str], win: int = 61) -> pd.DataFrame:
    """Subtract a centred rolling median: the case-crossover tests event-locked deviations."""
    a = agg.copy()
    for ax in axes:
        col = f"{ax}_mean"
        a[f"{col}_dt"] = a[col] - a[col].rolling(win, center=True, min_periods=20).median()
    return a


def space_events(dates, min_days: int = 28):
    kept = []
    for d in sorted(pd.to_datetime(list(dates))):
        if not kept or (d - kept[-1]).days >= min_days:
            kept.append(d)
    return kept


def event_battery(agg: pd.DataFrame, dates, label: str, arm: str) -> pd.DataFrame:
    rows = []
    for ax in TESTED:
        r = event_study(agg, dates, outcome=f"{ax}_mean_dt", post=(1, 7), pre=(-21, -4),
                        min_dreams=MIN_DREAMS, n_perm=20000)
        if r.get("n_events", 0) == 0:
            continue
        rows.append({"axis": ax, "family": L.family_of(ax), "cohort_set": label, "arm": arm,
                     "n_events": r["n_events"], "effect": r["effect_post_minus_pre"],
                     "perm_p": r["perm_p"], "null_sd": r["null_sd"]})
    R = pd.DataFrame(rows)
    if len(R):
        R["q"] = R.groupby("family")["perm_p"].transform(
            lambda p: benjamini_hochberg(p.to_numpy()))
    return R


def injected_mde(agg: pd.DataFrame, dates, axis: str,
                 deltas=(0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12),
                 n_rep: int = 30, n_perm: int = 2000, seed: int = 0) -> float:
    """Smallest injected post-window step detected at **80% power**.

    Three things the first version got optimistically wrong, all corrected here:

    1. It injected into the already-detrended column, so the step escaped the
       attenuation that ``detrend_local``'s centred 61-day rolling median
       inflicts on a real event-locked step. The step is now added to the raw
       daily mean and the series is re-detrended, exactly as observed data is.
    2. It tested a single noise realisation, so the first delta reaching p<.05
       was roughly the 50%-power point, not 80%. Power is now estimated over
       ``n_rep`` realisations generated by circularly rotating the series (which
       preserves its autocorrelation and marginal distribution) before injection.
    3. The delta grid was coarse (1,2,3,5,8,12pp), giving +/-1pp resolution at
       the reported value. It is finer below 4pp, where the answers live.
    """
    rng = np.random.default_rng(seed)
    raw_col = f"{axis}_mean"
    idx = pd.to_datetime(agg["date"]).to_numpy()
    step = np.zeros(len(agg))
    for e in pd.to_datetime(list(dates)):
        m = ((idx >= np.datetime64(e + pd.Timedelta(days=1)))
             & (idx <= np.datetime64(e + pd.Timedelta(days=7))))
        step[m] += 1.0
    base = agg[raw_col].to_numpy(float)

    for delta in deltas:
        hits = 0
        for _ in range(n_rep):
            a = agg.copy()
            shifted = np.roll(base, int(rng.integers(len(base))))
            a[raw_col] = shifted + delta * step
            a = detrend_local(a, [axis])
            r = event_study(a, dates, outcome=f"{raw_col}_dt", post=(1, 7),
                            pre=(-21, -4), min_dreams=MIN_DREAMS, n_perm=n_perm)
            hits += r.get("perm_p", 1.0) < 0.05
        if hits / n_rep >= 0.80:
            return float(delta)
    return np.nan


# ------------------------------------------------------------------ main ----
def main() -> None:
    d = load_scored()
    print(f"[ds-content] scored dreams={len(d)}; "
          f"EN dense={(d.lang.eq('en') & d.date.ge(DENSE_START)).sum()}")

    # --- seasonality, raw and person-adjusted -----------------------------
    S = pd.concat([seasonality(daily(d, "en", False), "raw"),
                   seasonality(daily(d, "en", True), "person_adjusted")],
                  ignore_index=True)

    # --- content-matched event study --------------------------------------
    evs = pd.read_csv(C.EVENTS_DIR / "curated_global_shocks.csv", parse_dates=["date"])
    evs = evs[evs.cohorts.str.contains("en")]
    agg_raw = detrend_local(daily(d, "en", False), TESTED)
    agg_adj = detrend_local(daily(d, "en", True), TESTED)

    sets = {
        "threat(war/violence/econ)": space_events(evs[evs.category.isin(THREAT)].date),
        "death-category(Pope Francis)": space_events(evs[evs.category.eq("death")].date),
        "all-curated": space_events(evs.date),
    }
    E = pd.concat([event_battery(a, dts, name, arm)
                   for arm, a in (("raw", agg_raw), ("person_adjusted", agg_adj))
                   for name, dts in sets.items()], ignore_index=True)

    mde = pd.DataFrame([
        {"axis": ax, "cohort_set": name, "power": 0.80, "alpha": 0.05,
         "pre_rate": float(agg_raw[f"{ax}_mean"].mean()),
         "mde80": injected_mde(agg_raw, dts, ax)}
        for name, dts in sets.items() for ax in ["death", "contagion", "contamination"]
    ])
    mde["mde80_rel"] = mde.mde80 / mde.pre_rate

    C.TABLES.mkdir(parents=True, exist_ok=True)
    S.to_csv(C.TABLES / "dscontent_seasonality.csv", index=False)
    E.to_csv(C.TABLES / "dscontent_event_study.csv", index=False)
    mde.to_csv(C.TABLES / "dscontent_event_mde.csv", index=False)
    # The event study gates at MIN_DREAMS=15 for comparability with F0050, whose
    # gate was chosen for event count. The *released* series is suppressed at the
    # stricter ETHICS section 4 floor, so released and analysed differ; the
    # analysis gate is stated rather than quietly applied to the release.
    keep = ["date", "n_dreams", "n_users", "words_mean"] + [f"{a}_mean" for a in list(L.ALL_AXES)]
    rel = daily(d, "en", False)[keep].copy()
    below = (rel.n_dreams < 20) | (rel.n_users < 5)
    rel.loc[below, [c for c in rel.columns
                    if c not in ("date", "n_dreams", "n_users")]] = np.nan
    print(f"[ds-content] released daily cells suppressed below floor: "
          f"{int(below.sum())}/{len(rel)}")
    rel.to_csv(C.TABLES / "dscontent_daily_en.csv", index=False)

    print("\n=== annual seasonality (EN, DreamSeer) ===")
    print(S.pivot_table(index=["family", "axis"], columns="arm",
                        values=["amp_rel", "harmonics_p"]).round(4).to_string())
    print("\n=== content-matched event study (person-adjusted arm) ===")
    e = E[E.arm == "person_adjusted"]
    print(e.pivot_table(index=["family", "axis"], columns="cohort_set",
                        values=["effect", "perm_p"]).round(4).to_string())
    print("\n=== detectable post-window step at p<.05 ===")
    print(mde.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n[ds-content] -> {C.TABLES/'dscontent_event_study.csv'}")


if __name__ == "__main__":
    main()
