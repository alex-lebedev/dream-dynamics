"""BOLD PROBE 7 — co-dreaming WITHIN THE WEEK (does aggregation reveal collective synchrony?).

The within-DAY test (Probe 1) was NULL in EN. Aggregating to the WEEK lifts reliability (weekly
split-half r=.21 vs daily .09), so a weekly collective pulse could surface even if the nightly one
is buried in noise. Same rigorous design, one scale up:

  unit = (user, ISO-week) mean embedding (cross-user pairs only, by construction).
  observed = within-week cross-user cohesion.
  null = a random same-size group drawn from the LOCAL +/-8-week neighbourhood (controls slow
         topical drift at the ~4-month scale); 1000x/week.
  inference = block-bootstrap CI + sign test + Stouffer; week-shuffle NEGATIVE CONTROL.
  cohorts = EN (primary), RU, pooled (native multilingual MiniLM space).

Exploratory add: do "high-synchrony weeks" line up with societal stress (EPU / |ΔVIX|)?
Aggregate-only outputs.

    PYTHONPATH=src PYTHONNOUSERSITE=1 python3 analyses/2026-07-18-21-coweek-synchrony.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from psychohistory import config as C
from psychohistory.dreams.embed_cache import load_or_build
from psychohistory.signals.panel import build_weekly_panel
from psychohistory.stats.inference import block_bootstrap_ci, circular_shift_pvalue_corr

OUT = C.RESULTS / "showcase"
OUT.mkdir(parents=True, exist_ok=True)
MIN_UNITS = 30          # >=30 user-weeks (distinct users) to test a week
MIN_DREAMS = 60
WIN_W = 8               # local null neighbourhood (+/- weeks)
B = 1000


def _cohesion(sum_vec, n):
    return (float((sum_vec ** 2).sum()) - n) / (n * (n - 1)) if n > 1 else np.nan


def user_week_units(sub, E):
    wk = sub.date.dt.to_period("W-SUN").dt.start_time
    codes, uniq = pd.factorize(list(zip(sub.userID.values, wk.values)))
    M = codes.max() + 1
    S = np.zeros((M, E.shape[1])); np.add.at(S, codes, E)
    U = S / np.bincount(codes, minlength=M)[:, None]
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9)
    uw_week = pd.to_datetime([t[1] for t in uniq])
    dreams_per_week = wk.value_counts()
    return U, uw_week.values, dreams_per_week


def synchrony_week(meta, emb, lang, seed=0, shuffle=False):
    sub = (meta if lang == "all" else meta[meta.lang == lang]).reset_index(drop=True)
    E = emb if lang == "all" else emb[meta.lang.values == lang]
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(seed)
    U, uw_week, dpw = user_week_units(sub, E)
    if shuffle:
        uw_week = rng.permutation(uw_week)
    wk_series = pd.Series(uw_week)
    weeks = np.array(sorted(wk_series.unique()))
    rows = []
    for w in weeks:
        w_ts = pd.Timestamp(w)
        idx = np.where(uw_week == w)[0]
        n = len(idx)
        ndreams = int(dpw.get(w_ts, n))
        if n < MIN_UNITS or ndreams < MIN_DREAMS:
            continue
        obs = _cohesion(U[idx].sum(0), n)
        deltas = np.abs((wk_series - w_ts).dt.days.values)
        pool = np.where((deltas > 0) & (deltas <= WIN_W * 7))[0]
        if len(pool) < n:
            continue
        sel = rng.random((B, len(pool))).argsort(1)[:, :n]
        sums = U[pool][sel].sum(1)
        null = ((sums ** 2).sum(1) - n) / (n * (n - 1))
        mu, sd = float(null.mean()), float(null.std()) + 1e-12
        rows.append((w_ts, ndreams, n, obs, mu, obs - mu, (obs - mu) / sd,
                     (1 + int((null >= obs).sum())) / (B + 1)))
    R = pd.DataFrame(rows, columns=["week", "n_dreams", "n_users", "obs", "null_mean",
                                    "excess", "z", "p"])
    if R.empty:
        return {"lang": lang, "n_weeks": 0, "_frame": R}
    exc = R.excess.values
    _, lo, hi = block_bootstrap_ci(exc, np.mean, block=4, n=5000, seed=seed)
    zst = float(np.sum(R.z.values) / np.sqrt(len(R)))
    return {"lang": lang, "n_weeks": len(R), "mean_excess": float(exc.mean()),
            "ci": (float(lo), float(hi)), "frac_pos": float((exc > 0).mean()),
            "sign_p": float(stats.binomtest((exc > 0).sum(), len(exc), 0.5, alternative="greater").pvalue),
            "z_stouffer": zst, "stouffer_p": float(stats.norm.sf(zst)),
            "n_sig": int((R.p < 0.05).sum()), "expected_sig": round(0.05 * len(R), 1),
            "top": [(str(pd.Timestamp(t).date()), int(nd), round(z, 2))
                    for t, nd, z in zip(*[R.sort_values('z', ascending=False).head(6)[c]
                                          for c in ['week', 'n_dreams', 'z']])],
            "_frame": R}


def tier(s):
    if s.get("n_weeks", 0) == 0:
        return "n/a"
    if s["ci"][0] > 0 and s["sign_p"] < 0.05:
        return "robust positive"
    if s["ci"][0] > 0 or s["stouffer_p"] < 0.05:
        return "suggestive"
    return "null"


def main():
    meta, emb = load_or_build()
    en = synchrony_week(meta, emb, "en", seed=1)
    ru = synchrony_week(meta, emb, "ru", seed=2)
    allc = synchrony_week(meta, emb, "all", seed=3)
    ctrl = synchrony_week(meta, emb, "en", seed=7, shuffle=True)

    # exploratory: do high-synchrony EN weeks track societal stress?
    stress_line = "- (stress link: insufficient overlap)"
    try:
        panel = build_weekly_panel().set_index("week")
        Rw = en["_frame"].set_index("week")
        j = Rw.join(panel[["epu", "vix"]]).dropna(subset=["excess"])
        j["dvix"] = j["vix"].diff().abs()
        parts = []
        for sig in ["epu", "dvix"]:
            k = j[["excess", sig]].dropna()
            if len(k) >= 30:
                r, p = circular_shift_pvalue_corr(k["excess"].values, k[sig].values)
                parts.append(f"{sig} r={r:+.2f}(p={p:.2f})")
        if parts:
            stress_line = "- Exploratory stress link (EN weekly synchrony-excess vs): " + "; ".join(parts)
    except Exception as e:
        stress_line = f"- (stress link skipped: {type(e).__name__})"

    def line(s):
        if s.get("n_weeks", 0) == 0:
            return f"- **{s['lang'].upper()}**: no eligible weeks."
        return (f"- **{s['lang'].upper()}** ({s['n_weeks']} weeks): mean excess "
                f"**{s['mean_excess']:+.4f}** (95% CI {s['ci'][0]:+.4f}..{s['ci'][1]:+.4f}); "
                f"{s['frac_pos']:.0%} weeks positive (sign p={s['sign_p']:.1e}); "
                f"Stouffer p={s['stouffer_p']:.1e}; {s['n_sig']}/{s['n_weeks']} weeks p<.05 "
                f"(exp {s['expected_sig']}) → **{tier(s)}**")

    L = ["# BOLD PROBE 7 — do we dream together WITHIN THE WEEK?", "",
         "*User-week embedding units; within-week cross-user cohesion vs a matched local +/-8-week "
         "permutation null (1000x/week); block-bootstrap CI; week-shuffle negative control. Weekly "
         "aggregation lifts reliability vs the (null) nightly test (Probe 1).*", "",
         "## Within-week collective synchrony",
         line(en), line(ru), line(allc),
         f"- **Negative control** (EN weeks shuffled): excess {ctrl.get('mean_excess', float('nan')):+.4f} "
         f"(95% CI {ctrl.get('ci', (0, 0))[0]:+.4f}..{ctrl.get('ci', (0, 0))[1]:+.4f}) — must be ~0.",
         f"- Peak-synchrony weeks (EN, week·dreams·z): {en.get('top', [])}",
         stress_line,
         "",
         "## Verdict",
         f"- EN within-week synchrony: **{tier(en)}**; RU: **{tier(ru)}**; pooled: **{tier(allc)}** "
         "(pooled mixes languages → cohesion partly reflects EN/RU clustering; read EN/RU separately).",
         "- Compare to Probe 1 (within-DAY): "
         + ("weekly aggregation surfaces synchrony that nightly noise hid." if tier(en) != "null"
            else "still null in EN at the weekly scale too — the collective-unconscious 'shared field' "
            "does not appear in English even after reliability-boosting aggregation."),
        f"- RU within-week is **{tier(ru)}**: the RU within-DAY positive (Probe 1) "
        + ("persists weekly too → kept Frontier (likely news-residue)." if tier(ru) != "null" else
           "does NOT persist at the weekly scale → it was **day-specific** (acute shared-news "
           "residue), not a sustained weekly collective field. 'We dream alone' holds at both scales.")]
    (OUT / "coweek_synchrony.md").write_text("\n".join(L))
    for s in (en, ru, allc):
        if s.get("n_weeks", 0):
            s["_frame"].assign(week=lambda x: x.week.dt.date).to_csv(
                OUT / f"coweek_synchrony_{s['lang']}.csv", index=False)
    print("\n".join(L)); print("\n[coweek] wrote", OUT / "coweek_synchrony.md")


if __name__ == "__main__":
    main()
