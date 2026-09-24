"""Emit the Russian peak-synchrony night list, and test the two diagnostics that dissent from the
'we dream alone' null.

Why this exists. Section 3.5 of the manuscript set aside the one same-night synchrony result that
points *toward* collective co-movement — the Russian cohort's positive excess — by asserting that its
"peak nights include 24 February 2025", i.e. that the effect is shared war news reaching a small,
geographically concentrated cohort on particular days. That attribution had no computed support.
`2026-07-18-15-bold-synchrony-biometric.py` calls `synchrony(..., "ru")`, which builds the peak-night
list internally, but only ever writes `synchrony_en_by_night.csv` — so no Russian night list existed
anywhere in the repository, and 2025-02-24 is in fact in the *English* peak list (n=27, z=2.88). The
date reached the manuscript from a hardcoded string in that script's verdict prose.

Setting aside a dissenting result on a datum belonging to the other cohort is exactly the move this
paper criticizes elsewhere, so this script computes the thing that should have been computed. It also
tests two diagnostics that were reported as bare numbers inside paragraphs asserting a null:

  1. The English cohort has 44 of 601 nights nominally significant at p<.05 against 30.1 expected.
     That 1.46x excess was never tested. Nights are not independent (adjacent nights share
     contributors and topic drift), so a plain binomial is anticonservative; a block bootstrap over
     7-day blocks is reported alongside it, matching the block length the published CI already uses.

  2. The strongest available *power* statement was missing. The English mean excess has a 95% CI
     whose upper bound can be compared with the Russian point estimate: if the bound excludes it, the
     English design would have detected an effect of the only magnitude observed anywhere in these
     data, which converts an underpowered null into a bounded one.

Seeds match the published run exactly (EN seed=1, RU seed=2) so the numbers here are the numbers in
Section 3.5 rather than a near miss.

    PYTHONPATH=src PYTHONNOUSERSITE=1 python3 analyses/2026-08-21-05-synchrony-ru-peaks.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from psychohistory import config as C
from psychohistory.dreams.embed_cache import load_or_build
from psychohistory.stats.inference import block_bootstrap_ci

OUT = C.RESULTS / "showcase"
TABLES = C.RESULTS / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

SRC = Path("analyses/2026-07-18-15-bold-synchrony-biometric.py")


def _load_published_module():
    """Import the published synchrony estimator rather than reimplementing it.

    Reimplementing would put a second copy of the estimator in the repo and invite exactly the
    silent divergence that the Figure 6 / Table S3 duplication already risks.
    """
    spec = importlib.util.spec_from_file_location("bold_synchrony", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def peak_nights(frame: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    """Same min-N gate and ordering the published EN list uses (>=20 dreams, >=5 users, top by z)."""
    ok = frame[(frame.n >= 20) & (frame.n_users >= 5)]
    return ok.sort_values("z", ascending=False).head(k)


def nominal_excess(frame: pd.DataFrame, seed: int = 0) -> dict:
    """Is the count of nominally-significant nights itself more than chance?

    Reported two ways. The binomial treats nights as independent, which they are not, so it is the
    anticonservative bound. The block bootstrap resamples 7-day blocks of the per-night indicator —
    the same block length the published mean-excess CI uses — and is the number to read.
    """
    hit = (frame.p.values < 0.05).astype(float)
    n, k = len(hit), int(hit.sum())
    binom = stats.binomtest(k, n, 0.05, alternative="greater").pvalue
    mean, lo, hi = block_bootstrap_ci(hit, np.mean, block=7, n=5000, seed=seed)

    # Which tail? The per-night p is one-sided (obs >= null), so p>.95 marks nights that are
    # *less* cohesive than their local matched pool. If BOTH tails are enriched, the per-night
    # excess is over-dispersed relative to the null rather than shifted, and a mean excess of
    # zero alongside an enriched upper tail is then the expected signature — which is a
    # different claim from same-night synchrony, and is also what a miscalibrated per-night
    # permutation null would produce. Reporting both keeps those two readings distinguishable.
    low = int(np.sum(frame.p.values > 0.95))
    sd_z = float(np.std(frame.z.values, ddof=1))
    return {"n_nights": n, "n_nominal": k, "expected": round(0.05 * n, 1),
            "ratio": round(k / (0.05 * n), 2), "binom_p": float(binom),
            "rate": float(np.mean(hit)), "rate_ci_lo": float(lo), "rate_ci_hi": float(hi),
            "excludes_chance": bool(lo > 0.05),
            "n_low_tail": low, "low_ratio": round(low / (0.05 * n), 2),
            "sd_z": sd_z, "frac_pos": float(np.mean(frame.excess.values > 0))}


def main() -> None:
    mod = _load_published_module()
    meta, emb = load_or_build()
    print(f"[data] {len(meta)} dreams; langs={meta.lang.value_counts().to_dict()}", flush=True)

    print("[synchrony] EN (seed=1, published) ...", flush=True)
    syn_en = mod.synchrony(meta, emb, "en", seed=1)
    print("[synchrony] RU (seed=2, published) ...", flush=True)
    syn_ru = mod.synchrony(meta, emb, "ru", seed=2)

    en, ru = syn_en["_frame"], syn_ru["_frame"]

    # ---- 1. the Russian night list that never existed -------------------------------------
    ru_out = ru.copy()
    ru_out["date"] = pd.to_datetime(ru_out["date"]).dt.date
    ru_out.round(6).to_csv(TABLES / "synchrony_ru_by_night.csv", index=False)
    print(f"\n[write] synchrony_ru_by_night.csv: {len(ru_out)} nights")

    ru_peaks, en_peaks = peak_nights(ru), peak_nights(en)
    print("\n=== RU peak-synchrony nights (date, n dreams, n users, z, p) ===")
    for _, r in ru_peaks.iterrows():
        print(f"    {pd.Timestamp(r.date).date()}  n={int(r.n):3d}  users={int(r.n_users):3d}  "
              f"z={r.z:+.2f}  p={r.p:.3f}")
    print("\n=== EN peak-synchrony nights (for comparison) ===")
    for _, r in en_peaks.iterrows():
        print(f"    {pd.Timestamp(r.date).date()}  n={int(r.n):3d}  users={int(r.n_users):3d}  "
              f"z={r.z:+.2f}  p={r.p:.3f}")

    # The specific claim the manuscript made, checked in both cohorts.
    target = pd.Timestamp("2025-02-24")
    for name, pk, fr in (("RU", ru_peaks, ru), ("EN", en_peaks, en)):
        in_peak = target in set(pd.to_datetime(pk.date))
        in_any = target in set(pd.to_datetime(fr.date))
        print(f"[check] 2025-02-24 in {name} peak list: {in_peak} | scored at all in {name}: {in_any}")

    # ---- 2. the untested nominal-significance excess --------------------------------------
    print("\n=== nominally-significant nights vs chance ===")
    rows = []
    for name, fr in (("en", en), ("ru", ru)):
        d = nominal_excess(fr, seed=11)
        rows.append({"lang": name, **d})
        print(f"    {name.upper()}: {d['n_nominal']}/{d['n_nights']} nominal vs {d['expected']} "
              f"expected ({d['ratio']}x); binomial p={d['binom_p']:.2g}; "
              f"block-bootstrap rate {d['rate']:.3f} [{d['rate_ci_lo']:.3f}, {d['rate_ci_hi']:.3f}]; "
              f"CI excludes 0.05: {d['excludes_chance']}")
        print(f"        lower tail (p>.95): {d['n_low_tail']} vs {d['expected']} expected "
              f"({d['low_ratio']}x) | SD of per-night z = {d['sd_z']:.2f} (null expects ~1.0) | "
              f"frac nights positive = {d['frac_pos']:.2f}")

    # ---- 3. the power statement that was missing ------------------------------------------
    en_hi, ru_pt = syn_en["ci"][1], syn_ru["mean_excess"]
    print("\n=== power: does the EN design bound the only effect size observed here? ===")
    print(f"    EN mean excess {syn_en['mean_excess']:+.4f}, 95% CI "
          f"[{syn_en['ci'][0]:+.4f}, {syn_en['ci'][1]:+.4f}]")
    print(f"    RU mean excess {ru_pt:+.4f}, 95% CI "
          f"[{syn_ru['ci'][0]:+.4f}, {syn_ru['ci'][1]:+.4f}]  (n={syn_ru['n_days']} nights)")
    print(f"    EN upper bound {en_hi:+.4f} {'EXCLUDES' if en_hi < ru_pt else 'includes'} "
          f"the RU point estimate {ru_pt:+.4f} "
          f"-> EN null is {'bounded' if en_hi < ru_pt else 'merely underpowered'} "
          f"at the RU magnitude (ratio {ru_pt / en_hi:.1f}x)")

    summary = pd.DataFrame(rows)
    summary["en_ci_hi"] = syn_en["ci"][1]
    summary["ru_point"] = ru_pt
    summary["en_bounds_ru_magnitude"] = en_hi < ru_pt
    summary.round(6).to_csv(TABLES / "synchrony_dissent_diagnostics.csv", index=False)
    print(f"\n[write] synchrony_dissent_diagnostics.csv")

    ru_peaks_out = ru_peaks[["date", "n", "n_users", "z", "p"]].copy()
    ru_peaks_out["date"] = pd.to_datetime(ru_peaks_out["date"]).dt.date
    ru_peaks_out.round(4).to_csv(TABLES / "synchrony_ru_peak_nights.csv", index=False)
    print("[write] synchrony_ru_peak_nights.csv")


if __name__ == "__main__":
    main()
