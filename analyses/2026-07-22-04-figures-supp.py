"""Updated/《supplementary》 publication figures — Figure 5 (CSD) + Supplementary Figure S2 (event study).

Renders two figures in the shared `plotstyle` (colorblind-safe, 400-dpi PNG + vector PDF, panel
letters), REUSING the exact committed analysis logic — the statistics are imported from the two
committed analysis scripts, not re-implemented, so every plotted quantity reproduces its committed value:

  - **Figure 5** (F0033, script 2026-07-19-19): critical slowing down. Rolling 28-day variance and
    AR(1) of the deseasonalized, volume/SE-standardized EN nightmare residual, with the 12 *endogenous*
    top-decile mood-darkening episodes and their 42-day run-up windows marked, plus the observed
    variance run-up Kendall-tau against AR(1)/phase-randomized surrogates (the decisive null).
  - **Supplementary Figure S2** (F0050, script 2026-07-20-15): the forward event-study around 10
    curated global shocks — a NULL specificity control. Event-aligned detrended affect (with a neutral
    control) + per-event Delta negativity.

The episodes in Figure 5 are darkenings of the dream signal itself (endogenous), NOT the external news
events in Supplementary Figure S2 — the two analyses are deliberately kept distinct.

Outputs -> 60-results/showcase/pub/ :
    fig5_csd.{png,pdf}
    figS2_event_study.{png,pdf}

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 USE_TF=0 python3 \
        analyses/2026-07-22-04-figures-supp.py
"""
from __future__ import annotations

import hashlib
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from psychohistory import config as C
from psychohistory.dreams.embed_cache import load_or_build
from psychohistory.utils.plotstyle import apply_style, PALETTE, panel_label, save_fig

SHOW = C.RESULTS / "showcase"
PUB = SHOW / "pub"
PUB.mkdir(parents=True, exist_ok=True)
ANALYSES = C.ROOT / "analyses"

WIN, RUNUP, SEED = 28, 42, 0
# 500 draws per generator = the 1,000-surrogate published-null budget reported in Section 3.5 and
# Supplementary Table S3. It was 400 (800 draws) while the manuscript quoted 1,000, so the plotted
# histogram and its on-image p did not match the caption above it.
N_SURR = 500


def _load_module(filename, name):
    """Import a committed analysis script by path (its filename is not a valid module name)."""
    spec = importlib.util.spec_from_file_location(name, ANALYSES / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # only defines functions; main() is __main__-guarded
    return mod


CD = _load_module("2026-07-19-19-collective-dynamics.py", "collective_dynamics")
ES = _load_module("2026-07-20-15-forward-event-study.py", "forward_event_study")


def _z(a):
    a = np.asarray(a, float)
    return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-9)


# ---- Figure 5: critical slowing down (F0033) --------------------------------------------------
def _csd_arrays(meta, col="nightmare_index"):
    """Reproduce the committed CSD pipeline (2026-07-19-19 `csd_test`) and expose the plot arrays."""
    x, mask = CD._daily_resid(meta, col)
    var, ac = CD._ews(x, WIN)
    # endogenous top-decile darkening anchors — identical to CD._pipeline_taus
    fwd = (pd.Series(x).shift(-14).rolling(14).mean() - pd.Series(x).rolling(7).mean()).values
    fin = np.isfinite(fwd)
    anc = CD._dedupe(np.where(fin & (fwd >= np.nanquantile(fwd, 0.90)))[0], RUNUP)
    obs, n_anc = CD._pipeline_taus(x, WIN, RUNUP)
    obs_tau = float(obs["variance"])
    # AR(1)+phase surrogates through the identical pipeline. The seed is the same stable-hash stream
    # the hardening script uses for this cell, and the generators run in the same order at the same
    # budget, so the p drawn on this figure is the p reported in Table S3 rather than a near miss.
    key = f"{SEED}|ar1_plus_phase (published)|{col}".encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))
    surr = []
    for gen in (CD._ar1_surrogate, CD._phase_surrogate):
        for _ in range(N_SURR):
            t, _ = CD._pipeline_taus(gen(x, rng), WIN, RUNUP)
            surr.append(t["variance"])
    surr = np.asarray(surr); surr = surr[np.isfinite(surr)]
    surr_p = float((1 + np.sum(surr >= obs_tau)) / (len(surr) + 1))
    # reconstruct the daily date grid (identical filtering to CD._daily_resid: lang==en, n>=20)
    sub = meta[meta.lang == "en"]
    sizes = sub.groupby(sub.date.dt.normalize())[col].size()
    sizes = sizes[sizes >= 20]
    dates = pd.date_range(sizes.index.min(), sizes.index.max(), freq="D").to_numpy()
    if len(dates) != len(var):        # safety: fall back to an integer index
        dates = np.arange(len(var))
    return dict(dates=dates, var=var, ac=ac, anc=anc, obs_tau=obs_tau, surr=surr,
                surr_mean=float(surr.mean()), surr_p=surr_p, n_anc=int(n_anc), n_days=int(mask.sum()))


def fig5_csd(meta):
    d = _csd_arrays(meta, "nightmare_index")
    dates, vz, az, anc = d["dates"], _z(d["var"]), _z(d["ac"]), d["anc"]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.5), gridspec_kw={"width_ratios": [2.4, 1]})

    # (a) rolling indicators + endogenous episodes + 42-day run-up shading
    ax[0].plot(dates, vz, color=PALETTE["blue"], lw=1.6, label="rolling variance (z)")
    ax[0].plot(dates, az, color=PALETTE["red"], lw=1.3, alpha=0.75, label="rolling AR(1) (z)")
    ax[0].axhline(0, color=PALETTE["grey"], lw=0.6)
    for a in anc:
        lo = max(0, a - RUNUP)
        ax[0].axvspan(dates[lo], dates[a], color=PALETTE["orange"], alpha=0.11, lw=0)
        ax[0].axvline(dates[a], color=PALETTE["orange"], lw=1.0)
    ax[0].set_ylabel("rolling indicator (z)"); ax[0].set_xlabel("date")
    ax[0].set_title("Early-warning indicators before mood-darkening episodes")
    h, lab = ax[0].get_legend_handles_labels()
    h.append(Patch(facecolor=PALETTE["orange"], alpha=0.35))
    lab.append(f"{len(anc)} darkening episodes (+42-d run-up)")
    ax[0].legend(h, lab, loc="upper left", fontsize=8)
    panel_label(ax[0], "a")

    # (b) observed variance run-up tau vs AR(1)/phase surrogates
    ax[1].hist(d["surr"], bins=30, color=PALETTE["grey"], alpha=0.85, edgecolor="white", linewidth=0.3)
    ax[1].axvline(d["obs_tau"], color=PALETTE["blue"], lw=2.4, label=f"observed τ = {d['obs_tau']:+.3f}")
    ax[1].axvline(d["surr_mean"], color=PALETTE["red"], lw=1.5, ls="--",
                  label=f"surrogate mean = {d['surr_mean']:+.3f}")
    ax[1].set_xlabel("variance run-up (Kendall τ)"); ax[1].set_ylabel("surrogate draws")
    ax[1].set_title(f"vs AR(1)/phase surrogates (p = {d['surr_p']:.3f})")
    ax[1].legend(loc="upper left", fontsize=8)
    panel_label(ax[1], "b")

    # Do not restate a claim here that the text has narrowed. The previous title asserted the
    # run-up held "beyond volatility-clustering surrogates" — the exact claim Section 2.5 retracts,
    # since AR(1)+phase surrogates do not preserve volatility clustering.
    fig.suptitle("Nightmare variance rises before the series' own mood-darkenings; significance is null-dependent",
                 fontsize=12, fontweight="bold", y=1.03)
    try:
        fig.autofmt_xdate(rotation=30)
    except Exception:
        pass
    fig.tight_layout()
    save_fig(fig, PUB / "fig5_csd")
    print(f"[fig5] CSD nightmare: obs_tau={d['obs_tau']:+.3f} surr_mean={d['surr_mean']:+.3f} "
          f"p={d['surr_p']:.3f} episodes={d['n_anc']} n_days={d['n_days']}", flush=True)


# ---- Supplementary Figure S2: forward event-study (F0050; NULL specificity control) -----------
def figS2_event_study():
    dl = ES.add_detrended(ES.daily("en"), ES.OUTCOMES + ES.NEUTRAL)
    evs = pd.DataFrame(ES.CURATED, columns=["date", "label", "category", "cohorts"]).assign(
        date=lambda x: pd.to_datetime(x.date))
    ev_en = evs[evs.category.isin(ES.THREAT) & evs.cohorts.apply(lambda c: "en" in c)]
    dts, _ = ES.space_events(ev_en["date"].tolist(), min_days=28)

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.7))
    # (a) event-aligned detrended affect (threat outcomes + neutral control)
    for oc, lab, col in [("negativity_mean_dt", "negativity", PALETTE["red"]),
                         ("fear_mean_dt", "fear", PALETTE["purple"]),
                         ("food_mean_dt", "food (neutral)", PALETTE["grey"])]:
        res = ES.aligned_curve(dl, dts, oc, min_dreams=15)
        if res is None:
            continue
        m, se = res
        xk = m.index.to_numpy()
        ax[0].plot(xk, m.to_numpy(), color=col, lw=1.7, label=lab)
        ax[0].fill_between(xk, (m - se).to_numpy(), (m + se).to_numpy(), color=col, alpha=0.15)
    ax[0].axvline(0, color=PALETTE["black"], lw=0.8); ax[0].axhline(0, color=PALETTE["black"], lw=0.5)
    ax[0].axvspan(1, 7, color=PALETTE["red"], alpha=0.06)
    ax[0].set_xlabel("days relative to shock"); ax[0].set_ylabel("Δ affect (detrended) vs pre-baseline")
    ax[0].set_title("Dreams around threat shocks (detrended)")
    ax[0].legend(loc="upper left", fontsize=8)
    panel_label(ax[0], "a")

    # (b) per-event Δ negativity (post − pre), red = threat / grey = other
    pe = ES.per_event(dl, evs, "negativity_mean_dt", min_dreams=15)
    cols = [PALETTE["red"] if c in ES.THREAT else PALETTE["grey"] for c in pe.category]
    ax[1].barh([l[:36] for l in pe.label], pe.effect.to_numpy(), color=cols)
    ax[1].axvline(0, color=PALETTE["black"], lw=0.6)
    ax[1].tick_params(axis="y", labelsize=7)
    ax[1].set_xlabel("Δ negativity (post − pre)")
    ax[1].set_title("Per-event Δ (red = threat)")
    panel_label(ax[1], "b")

    fig.suptitle("Specificity control: no adequately-powered event-locked response to curated global shocks (null)",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, PUB / "figS2_event_study")
    print("[figS2] event-study rendered (null specificity control)", flush=True)


def main():
    apply_style()
    meta, _ = load_or_build()
    fig5_csd(meta)
    figS2_event_study()
    print("[supp-figures] done ->", PUB)


if __name__ == "__main__":
    main()
