"""COVID content-correspondence: did death / contagion / contamination themes rise?

The question is content correspondence rather than state correspondence: not
"did dreaming become more negative" but "did dreams acquire the *specific
content* of the event". DreamSeer cannot answer it — its dense usable window
begins 2024-03-25, four years after the onset — so this runs on the Mallett
r/Dreams corpus (2019-01-01 to 2020-12-31), the repo's designated COVID
positive-control corpus.

The repo already has a COVID test (``analyses/2026-07-18-10-validation-ts.py``).
It used one conflated "anxious content" regex that mixed anxiety words with
health words, had no composition controls, and failed placebo-in-time
(p=.625). This analysis decomposes that measure and hardens the design:

1. **Declared axis families** (``psychohistory.dreams.lexicons``), fixed before
   estimation. The ``explicit`` family (covid/pandemic/quarantine/lockdown) is
   vocabulary that barely existed before 2020; it is a manipulation check that
   the design can see a real step, and a screen, never evidence about dreams.
   The ``latent`` family (contagion, illness, contamination, death, mask) is
   the actual test. ``neutral`` axes are negative controls.
2. **Composition controls.** Post volume rises 25% into the lockdown quarter,
   narrative-flair usage drifts 39%->61% across the window, and median post
   length falls. Every lexicon axis — including all seven neutral controls —
   drifts *down* about a point, so the raw contrast is uninterpretable. Arms:
   raw, covariate-adjusted, explicit-screened, and differenced against the
   neutral composite.
3. **Placebo-in-time**, the arm that killed the earlier claim, on every axis.
   Two designs: the full series (more power, unequal precision across cutoffs)
   and a symmetric +/-13-week window (every fit identical in size, so the
   placebo distribution is directly comparable).
4. **Power by injection**, so a null is a bound rather than an absence.

Aggregates only: no post text, no author identifiers, nothing below the
release floor.

Run: PYTHONPATH=src python3 analyses/2026-08-22-10-covid-content-correspondence.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from psychohistory import config as C
from psychohistory.dreams import lexicons as L
from psychohistory.stats.inference import benjamini_hochberg

WHO = pd.Timestamp("2020-03-11")          # WHO pandemic declaration
NARRATIVE_FLAIRS = {"short dream", "medium dream", "long dream",
                    "recurring dream", "nightmare"}
MIN_POSTS_PER_WEEK = 10
HAC_LAGS = 4
HALF_WIDTH = 13                            # symmetric-window design, weeks each side
COMPOSITION = ["logn", "log_words", "flair_rate", "new_author_share"]
#: Annual harmonics (METHODS section 2 mandates a seasonality control). Only
#: estimable in the full-series design: a 26-week symmetric window spans half a
#: cycle, where sin/cos are near-collinear with the linear trend.
HARMONICS = ["sin1", "cos1", "sin2", "cos2"]
CACHE = C.INTERIM / f"mallett_lexicon_scored_{L.version()}.parquet"

LATENT_AXES = list(L.LATENT)
TESTABLE = list(L.EXPLICIT) + LATENT_AXES + list(L.COMPARATOR)
NEUTRAL_AXES = list(L.NEUTRAL)


# --------------------------------------------------------------- corpus ----
def load_scored(rebuild: bool = False) -> pd.DataFrame:
    """Post-level table: date, narrative flag, length, author novelty, axis hits."""
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)

    raw = pd.read_csv(C.RAW / "mallett" / "r-dreams.csv", dtype=str,
                      keep_default_na=False)
    raw["date"] = pd.to_datetime(raw["created_utc"].astype(float), unit="s")
    text = (raw["title"].fillna("") + " " + raw["selftext"].fillna("")).str.strip()
    flair = raw["link_flair_text"].fillna("").str.lower()

    d = pd.DataFrame({
        "date": raw["date"],
        "author": raw["author"].fillna(""),
        "narrative": flair.isin(NARRATIVE_FLAIRS),
        "n_words": text.str.split().str.len().fillna(0).astype(int),
        "long_enough": text.str.len() >= 15,
    })
    d = pd.concat([d, L.score(text)], axis=1)

    # author novelty on the whole corpus, before any subsetting
    first_seen = d.loc[d.author.ne("") & d.author.ne("[deleted]")].groupby("author").date.min()
    d["new_author"] = d.author.map(first_seen).eq(d.date).fillna(False)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(CACHE, index=False)
    return d


def weekly(d: pd.DataFrame, screen_explicit: bool = False) -> pd.DataFrame:
    """Weekly axis presence-rates plus the composition covariates.

    ``flair_rate`` is computed over *all* posts in the week (not just the
    narrative subset) because it measures the selection into that subset.
    """
    d = d[d.long_enough].copy()
    d["week"] = d.date.dt.to_period("W-SUN").apply(lambda p: p.start_time)
    flair_rate = d.groupby("week").narrative.mean().rename("flair_rate")

    n = d[d.narrative].copy()
    if screen_explicit:
        n = n[n[list(L.EXPLICIT)].sum(axis=1) == 0]

    axes = list(L.ALL_AXES)
    agg = n.groupby("week").agg(
        n_posts=("narrative", "size"),
        n_authors=("author", "nunique"),
        mean_words=("n_words", "mean"),
        new_author_share=("new_author", "mean"),
        **{a: (a, "mean") for a in axes},
    )
    agg = agg.join(flair_rate)
    agg["logn"] = np.log(agg.n_posts)
    agg["log_words"] = np.log(agg.mean_words)
    agg["neutral_mean"] = agg[NEUTRAL_AXES].mean(axis=1)
    doy = agg.index.dayofyear.to_numpy(dtype=float)
    for h in (1, 2):
        agg[f"sin{h}"] = np.sin(2 * np.pi * h * doy / 365.25)
        agg[f"cos{h}"] = np.cos(2 * np.pi * h * doy / 365.25)
    return agg[agg.n_posts >= MIN_POSTS_PER_WEEK].reset_index()


# ------------------------------------------------------ segmented models ----
def _design(w: pd.DataFrame, cut: pd.Timestamp, covariates: list[str]) -> pd.DataFrame:
    t = np.arange(len(w), dtype=float)
    post = (w.week >= cut).to_numpy().astype(float)
    tpost = np.where(post > 0, t - t[post > 0].min(), 0.0) if post.any() else np.zeros(len(w))
    X = pd.DataFrame({"t": t, "post": post, "tpost": tpost}, index=w.index)
    for c in covariates:
        X[c] = w[c].to_numpy()
    return X


def its(w: pd.DataFrame, cut: pd.Timestamp, outcome: str,
        covariates: list[str] | None = None,
        weights: np.ndarray | None = None) -> tuple[float, float, float, float]:
    """Segmented interrupted time series.

    Returns (level shift, HAC p, HAC CI low, HAC CI high). METHODS section 7
    requires an interval on every reported effect, and for a claim whose content
    is a null the interval is the primary quantity: it is what separates "we did
    not detect" from "we can exclude".

    ``weights`` enables the inverse-variance (WLS) sensitivity arm: weekly
    presence rates are estimated from between 10 and several hundred posts, and
    that precision rises across the window, so it is correlated with treatment
    status. ``logn`` as a covariate absorbs a mean shift, not the variance.
    """
    covariates = covariates or []
    X = sm.add_constant(_design(w, cut, covariates))
    y = w[outcome].to_numpy(dtype=float)
    model = (sm.WLS(y, X, weights=weights) if weights is not None else sm.OLS(y, X))
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    lo, hi = res.conf_int().loc["post"]
    return (float(res.params["post"]), float(res.pvalues["post"]),
            float(lo), float(hi))


def its_fast(y: np.ndarray, X: np.ndarray, post_col: int) -> float:
    """Level shift only, via least squares — for the injection loops."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[post_col])


def placebo_cutoffs(w: pd.DataFrame, design: str) -> list[pd.Timestamp]:
    """Candidate fake cutoffs whose *fits* are genuine nulls.

    A pre-COVID cutoff is not sufficient: what has to be COVID-free is the data
    the placebo model sees. In the full-series design the placebo is therefore
    fitted on the pre-WHO subframe only (see ``run_arm``), so a candidate needs
    at least 12 weeks before it and 8 weeks of pre-WHO data after it. In the
    symmetric design the whole +/-HALF_WIDTH window must close before the WHO
    date, which it already did.
    """
    weeks = list(w.week)
    if design == "symmetric":
        return [c for i, c in enumerate(weeks)
                if i >= HALF_WIDTH and c + pd.Timedelta(weeks=HALF_WIDTH) <= WHO]
    pre = [c for c in weeks if c < WHO]
    return [c for i, c in enumerate(pre) if i >= 12 and i <= len(pre) - 8]


def window(w: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    lo, hi = cut - pd.Timedelta(weeks=HALF_WIDTH), cut + pd.Timedelta(weeks=HALF_WIDTH)
    return w[(w.week >= lo) & (w.week < hi)].reset_index(drop=True)


def run_arm(w: pd.DataFrame, arm: str, design: str, axes: list[str],
            covariates: list[str], harmonics: bool = True,
            weighted: bool = False) -> pd.DataFrame:
    """Real effect + placebo-in-time empirical p for each axis.

    The real fit uses the whole frame (full design) or a symmetric window. The
    placebo fit is restricted so that the model never sees post-WHO data: in the
    full design that means the pre-WHO subframe, which costs the placebo fits
    precision relative to the real fit and is reported as ``n_weeks_placebo``.
    The previous version fitted placebos on the entire series, so every fake
    cutoff's post-period contained COVID; that inflates the placebo shift
    distribution and biases the empirical p-value *upward*, which is
    anti-conservative for a null claim.
    """
    rows = []
    if harmonics and design == "full":
        covariates = covariates + HARMONICS
    wt = (lambda f: f.n_posts.to_numpy(float)) if weighted else (lambda f: None)
    if design == "symmetric":
        real_frame, placebo_frame = window(w, WHO), (lambda c: window(w, c))
    else:
        pre = w[w.week < WHO].reset_index(drop=True)
        real_frame, placebo_frame = w, (lambda c: pre)
    cands = placebo_cutoffs(w, design)
    for axis in axes:
        if len(real_frame) < 20:
            continue
        shift, hac_p, lo, hi = its(real_frame, WHO, axis, covariates, wt(real_frame))
        placebo, n_pl_weeks = [], np.nan
        for c in cands:
            f = placebo_frame(c)
            if len(f) < 20:
                continue
            n_pl_weeks = len(f)
            try:
                placebo.append(abs(its(f, c, axis, covariates, wt(f))[0]))
            except Exception:
                pass
        placebo = np.asarray(placebo, dtype=float)
        # (1 + #{>=}) / (1 + B): a Monte-Carlo p-value cannot be exactly zero,
        # and this makes the resolution floor self-documenting in the table
        emp_p = (float((1 + (placebo >= abs(shift)).sum()) / (1 + placebo.size))
                 if placebo.size else np.nan)
        rows.append({
            "axis": axis, "family": L.family_of(axis) if axis in L.ALL_AXES else "composite",
            "arm": arm, "design": design, "harmonics": harmonics, "weighted": weighted,
            "n_weeks": len(real_frame), "n_weeks_placebo": n_pl_weeks,
            "shift": shift, "hac_p": hac_p, "ci_lo": lo, "ci_hi": hi,
            "placebo_p": emp_p, "n_placebo": int(placebo.size),
            "placebo_q90": float(np.quantile(placebo, 0.90)) if placebo.size else np.nan,
            "placebo_q95": float(np.quantile(placebo, 0.95)) if placebo.size else np.nan,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------ power by injection ----
def injection_mde(w: pd.DataFrame, axis: str, covariates: list[str],
                  crit: float, deltas: np.ndarray, B: int = 300,
                  block: int = 4, seed: int = 0) -> dict:
    """Smallest injected step this design detects at 80% power.

    The series has its estimated step removed, making it null; a moving-block
    bootstrap then resamples it, a known step is injected at the WHO date, and
    the same placebo-calibrated decision rule is applied. ``crit`` is a quantile
    of that axis's own placebo shift distribution, so the calibration inherits
    the test actually reported rather than an asymptotic threshold — and it must
    be the **q95** (two-sided alpha=.05), matching the threshold the nulls are
    declared at. Calibrating at q90 while declaring nulls at p<.05 states the
    bound under a rule twice as permissive as the test that produced it.
    """
    rng = np.random.default_rng(seed)
    X = sm.add_constant(_design(w, WHO, covariates)).to_numpy(dtype=float)
    post_col = list(sm.add_constant(_design(w, WHO, covariates)).columns).index("post")
    y = w[axis].to_numpy(dtype=float)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_null = y - beta[post_col] * X[:, post_col]          # step removed
    resid = y_null - X @ np.linalg.lstsq(X, y_null, rcond=None)[0]
    fitted = y_null - resid

    n = len(y)
    nblocks = int(np.ceil(n / block))
    power = {}
    for delta in deltas:
        hits = 0
        for _ in range(B):
            idx = np.concatenate([(int(rng.integers(n)) + np.arange(block)) % n
                                  for _ in range(nblocks)])[:n]
            y_b = fitted + resid[idx] + delta * X[:, post_col]
            hits += abs(its_fast(y_b, X, post_col)) > crit
        power[float(delta)] = hits / B
    ok = [d for d, p in power.items() if p >= 0.80]
    return {"axis": axis, "mde80": min(ok) if ok else np.nan,
            "crit": crit, "power_curve": power}


# ------------------------------------------------------------------ main ----
def main() -> None:
    d = load_scored()
    w_all = weekly(d, screen_explicit=False)
    w_scr = weekly(d, screen_explicit=True)
    print(f"[covid] weeks={len(w_all)} "
          f"({w_all.week.min():%Y-%m-%d}..{w_all.week.max():%Y-%m-%d}), "
          f"posts={int(w_all.n_posts.sum())}; screened posts={int(w_scr.n_posts.sum())}")

    # differenced-against-neutral outcome: axis minus the neutral composite
    w_did = w_all.copy()
    for a in TESTABLE:
        w_did[a] = w_all[a] - w_all["neutral_mean"]

    # Two CO-PRIMARY arms. The explicit screen conditions on a post-treatment
    # variable and does so differentially in time (it removes 0.4% of pre-period
    # posts but 3.7% of post-period ones, and the removed posts are exactly those
    # enriched in the latent vocabulary under test), so it is biased toward the
    # null and cannot stand alone. The unscreened composition-adjusted arm is
    # biased the other way — the pandemic is free to enter through explicit
    # mentions — so the pair brackets the answer.
    arms = [
        ("raw", w_all, [], list(L.ALL_AXES), False, False),
        ("adjusted", w_all, COMPOSITION, list(L.ALL_AXES), True, False),
        ("adjusted_noharm", w_all, COMPOSITION, list(L.ALL_AXES), False, False),
        ("screened", w_scr, COMPOSITION, LATENT_AXES + list(L.COMPARATOR) + NEUTRAL_AXES,
         True, False),
        ("screened_noharm", w_scr, COMPOSITION,
         LATENT_AXES + list(L.COMPARATOR) + NEUTRAL_AXES, False, False),
        ("adjusted_wls", w_all, COMPOSITION, list(L.ALL_AXES), True, True),
        ("did_neutral", w_did, COMPOSITION, TESTABLE, True, False),
    ]
    out = []
    for design in ("full", "symmetric"):
        for arm, frame, covs, axes, harm, wls in arms:
            out.append(run_arm(frame, arm, design, axes, covs, harm, wls))
    R = pd.concat(out, ignore_index=True)

    # multiplicity is corrected inside each declared family x arm x design cell
    R["q"] = R.groupby(["family", "arm", "design"])["placebo_p"].transform(
        lambda p: benjamini_hochberg(p.fillna(1.0).to_numpy()))

    # Power by injection, on BOTH co-primary arms, calibrated at the q95 that
    # matches the alpha=.05 rule the nulls are declared at.
    deltas = np.round(np.arange(0.002, 0.081, 0.002), 4)
    mde_rows = []
    for arm, frame in (("screened", w_scr), ("adjusted", w_all)):
        prim = R[(R.arm == arm) & (R.design == "full")].set_index("axis")
        for axis in LATENT_AXES + list(L.COMPARATOR):
            if axis not in prim.index or not np.isfinite(prim.loc[axis, "placebo_q95"]):
                continue
            crit = float(prim.loc[axis, "placebo_q95"])
            m = injection_mde(frame, axis, COMPOSITION + HARMONICS, crit, deltas)
            base = float(frame.loc[frame.week < WHO, axis].mean())
            mde = m["mde80"]
            mde_rows.append({
                "arm": arm, "axis": axis, "pre_rate": base, "crit": crit,
                "crit_alpha": 0.05, "crit_quantile": "placebo_q95",
                "mde80_pp": 100 * mde if np.isfinite(mde) else np.nan,
                "mde80_rel": mde / base if np.isfinite(mde) and base > 0 else np.nan,
                "observed_shift_pp": 100 * float(prim.loc[axis, "shift"]),
                "ci_lo_pp": 100 * float(prim.loc[axis, "ci_lo"]),
                "ci_hi_pp": 100 * float(prim.loc[axis, "ci_hi"])})
    M = pd.DataFrame(mde_rows)

    # How much of the screened-vs-unscreened gap is mechanical selection rather
    # than confound removal? (share removed) x (excess rate among removed).
    dd = d[d.long_enough & d.narrative].copy()
    dd["post"] = dd.date >= WHO
    expl = dd[list(L.EXPLICIT)].sum(axis=1) > 0
    att_rows = []
    for axis in LATENT_AXES + list(L.COMPARATOR):
        for post in (False, True):
            g = dd[dd.post == post]
            rem, kept = g[expl.loc[g.index]], g[~expl.loc[g.index]]
            share = len(rem) / max(len(g), 1)
            r_rem = float(rem[axis].mean()) if len(rem) else np.nan
            r_kept = float(kept[axis].mean()) if len(kept) else np.nan
            att_rows.append({"axis": axis, "period": "post" if post else "pre",
                             "share_removed": share, "rate_removed": r_rem,
                             "rate_kept": r_kept,
                             "attenuation_pp": 100 * share * (r_rem - r_kept)})
    A = pd.DataFrame(att_rows)

    C.TABLES.mkdir(parents=True, exist_ok=True)
    R.to_csv(C.TABLES / "covid_content_its.csv", index=False)
    M.to_csv(C.TABLES / "covid_content_mde.csv", index=False)
    A.to_csv(C.TABLES / "covid_content_screen_attenuation.csv", index=False)
    # Released under the ETHICS section 4 floor (>=20 reports, >=5 contributors per
    # cell). The count columns are named so that scripts/check_release_cells.py
    # actually inspects them: a floor nothing checks is not a floor.
    keep = ["week", "n_posts", "n_authors", "mean_words", "new_author_share",
            "flair_rate", "neutral_mean"] + list(L.ALL_AXES)
    rel = w_all[keep].rename(columns={"n_posts": "n_reports", "n_authors": "n_authors"})
    below = (rel.n_reports < 20) | (rel.n_authors < 5)
    rel.loc[below, [c for c in rel.columns
                    if c not in ("week", "n_reports", "n_authors")]] = np.nan
    print(f"[covid] released weekly cells suppressed below floor: {int(below.sum())}/{len(rel)}")
    rel.to_csv(C.TABLES / "covid_content_weekly.csv", index=False)

    fmt = lambda v: f"{v:.4f}"
    for arm in ("screened", "adjusted"):
        print(f"\n=== co-primary: latent family, {arm}, full series ===")
        p = R[(R.arm == arm) & (R.design == "full") & (R.family == "latent")]
        print(p[["axis", "shift", "ci_lo", "ci_hi", "hac_p", "placebo_p", "q"]]
              .to_string(index=False, float_format=fmt))
    print("\n=== manipulation check: explicit vocabulary ===")
    e = R[(R.family == "explicit") & (R.design == "full")]
    print(e[["arm", "shift", "hac_p", "placebo_p"]].to_string(index=False, float_format=fmt))

    print("\n=== negative-control diagnostic: EVERY arm x design ===")
    print("(a neutral axis moving is a composition or measurement problem, not a finding)")
    nc = R[R.family == "neutral"]
    diag = nc.groupby(["design", "arm"]).apply(
        lambda g: pd.Series({
            "hits_p<.05": f"{int((g.placebo_p < .05).sum())}/{len(g)}",
            "max|shift|pp": 100 * g["shift"].abs().max(),
            "worst_axis": g.loc[g["shift"].abs().idxmax(), "axis"]}))
    print(diag.to_string())

    print("\n=== harmonics sensitivity: contagion & death, full design ===")
    h = R[(R.design == "full") & R.axis.isin(["contagion", "death"])
          & R.arm.isin(["raw", "adjusted", "adjusted_noharm", "screened",
                        "screened_noharm", "adjusted_wls"])]
    print(h.pivot_table(index="arm", columns="axis", values="shift").mul(100).round(2)
          .to_string())

    print("\n=== explicit-screen mechanical attenuation (pp of latent rate) ===")
    print(A.pivot_table(index="axis", columns="period",
                        values="attenuation_pp").round(3).to_string())

    print("\n=== detectable step at 80% power, crit=placebo q95 (alpha=.05) ===")
    print(M.to_string(index=False, float_format=fmt))
    print(f"\n[covid] -> {C.TABLES/'covid_content_its.csv'}")


if __name__ == "__main__":
    main()
