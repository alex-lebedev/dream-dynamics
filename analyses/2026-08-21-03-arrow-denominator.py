"""Sensitivity of the dispersion-standardized descent to a self-normalization objection.

The standardized descent divides each corpus's drift by the pooled within-document standard
deviation of sentence valence. A document that descends contributes that descent to its own
SD, so the denominator is not independent of the numerator: a steeply descending corpus
inflates its own denominator and is penalised for the very effect being measured. The
objection cuts against us, but it should be measured rather than argued.

This re-standardizes by the *residual* within-document SD, after removing each document's
own linear trend, and re-runs the nine endpoint contrasts that carry the magnitude claim.
For an OLS fit on equally spaced positions the residual variance is available in closed form
from the quantities already checkpointed by the recovery run, so no re-scoring is needed:

    sd_res^2 = [ (k-1) * sd^2  -  slope^2 * (k-1) * var(x) ] / (k-2),   var(x) = (k+1)/(12(k-1))

Run: PYTHONPATH=src python3 analyses/2026-08-21-03-arrow-denominator.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from psychohistory import config as C

B_BOOT = 4000
OUT = C.RESULTS / "showcase"

STEMS = {
    "DreamSeer EN": "DreamSeer_EN",
    "DreamSeer RU": "DreamSeer_RU",
    "DreamBank": "DreamBank",
    "Reddit r/Dreams": "Reddit_rDreams",
    "SDDb": "SDDb",
    "r/confession (waking narrative)": "confession",
    "Gutenberg fiction": "fiction",
    "Wikipedia openings": "wikipedia",
    "DreamSeer interp.": "DreamSeer_interp.",
}
DREAMS = ["DreamSeer EN", "DreamSeer RU", "DreamBank"]
COMPS = ["r/confession (waking narrative)", "Gutenberg fiction", "Wikipedia openings"]


def residual_sd(sd, slope, k):
    """Within-document SD after removing the document's own OLS trend, in closed form."""
    k = k.astype(float)
    var_x = (k + 1.0) / (12.0 * (k - 1.0))
    ss_tot = (k - 1.0) * sd ** 2
    ss_fit = slope ** 2 * (k - 1.0) * var_x
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.sqrt(np.clip(ss_tot - ss_fit, 0.0, None) / (k - 2.0))
    return out


def by_author(x, auth):
    ok = np.isfinite(x)
    return pd.DataFrame({"a": auth[ok], "x": x[ok]}).groupby("a", sort=True)["x"].mean().to_numpy()


def boot(a, seed):
    rng = np.random.default_rng(seed)
    return a[rng.integers(0, len(a), size=(B_BOOT, len(a)))].mean(axis=1)


def contrast(d, c, seed=0):
    diff = boot(d, seed) - boot(c, seed + 7919)
    p = max(float(2 * min((diff >= 0).mean(), (diff <= 0).mean())), 1.0 / (B_BOOT + 1))
    return float(d.mean() - c.mean()), float(np.percentile(diff, 2.5)), \
        float(np.percentile(diff, 97.5)), p


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


rows, per_corpus = {}, []
for name, stem in STEMS.items():
    z = np.load(C.INTERIM / f"arrow_recovery4_{stem}.npz")
    sd, slope, drift, k, auth = z["sd"], z["slope"], z["drift"], z["counts"], z["auth"]
    sd_res = residual_sd(sd, slope, k)
    pool_raw = float(np.sqrt(np.nanmean(sd ** 2)))
    pool_res = float(np.sqrt(np.nanmean(sd_res ** 2)))
    rows[name] = {
        "drift_z": by_author(drift / pool_raw, auth),
        "drift_zr": by_author(drift / pool_res, auth),
        "slope_zr": by_author(slope / pool_res, auth),
    }
    per_corpus.append({
        "corpus": name, "n_docs": len(sd), "sd_within": pool_raw, "sd_residual": pool_res,
        "shrinkage": 1 - pool_res / pool_raw,
        "drift_z": float(rows[name]["drift_z"].mean()),
        "drift_zr": float(rows[name]["drift_zr"].mean()),
        "slope_zr": float(rows[name]["slope_zr"].mean()),
    })

pc = pd.DataFrame(per_corpus)
print("\n== dispersion denominators, raw and trend-removed ==")
print(pc.round(4).to_string(index=False))

out = []
for stat in ["drift_zr", "slope_zr"]:
    for d in DREAMS:
        for c in COMPS:
            diff, lo, hi, p = contrast(rows[d][stat], rows[c][stat])
            out.append({"stat": stat, "dream": d, "comparator": c,
                        "diff": diff, "lo": lo, "hi": hi, "p": p})
res = pd.DataFrame(out)
for stat in res.stat.unique():
    m = res.stat == stat
    res.loc[m, "q"] = bh(res.loc[m, "p"].to_numpy())

print("\n== the nine endpoint contrasts, denominator = trend-removed SD ==")
print(res[res.stat == "drift_zr"].round(4).to_string(index=False))
print(f"\n  significant: {int((res[res.stat=='drift_zr'].q < .05).sum())} of 9, "
      f"max q = {res[res.stat=='drift_zr'].q.max():.4f}")
print("\n== the nine slope contrasts, same denominator ==")
print(res[res.stat == "slope_zr"].round(4).to_string(index=False))
print(f"\n  significant: {int((res[res.stat=='slope_zr'].q < .05).sum())} of 9, "
      f"max q = {res[res.stat=='slope_zr'].q.max():.4f}")

ratios = [pc.set_index("corpus").drift_zr[d] / pc.set_index("corpus").drift_zr[c]
          for d in DREAMS for c in COMPS]
print(f"\n  multiplier on the trend-removed denominator: "
      f"{min(ratios):.1f} to {max(ratios):.1f}")

pc.to_csv(OUT / "arrow_denominator_corpora.csv", index=False)
res.to_csv(OUT / "arrow_denominator_tests.csv", index=False)
print(f"\nwrote {OUT}/arrow_denominator_{{corpora,tests}}.csv\n")
