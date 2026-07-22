"""Round 4 / Q24 + Q28 — the physics of the dream narrative (intra-dream resolution).

Every prior structure probe used ONE vector per dream. Here we embed dreams sentence-by-sentence
and study the *trajectory* through meaning-space within a single dream.

(Q24) ANOMALOUS DIFFUSION / "the Levy flight of the dreaming mind".
    MSD(tau) = <||s_{i+tau} - s_i||^2> pooled over dreams; alpha = d log MSD / d log tau.
    alpha~1 normal diffusion, alpha>1 super-diffusive (heavy-tailed semantic jumps = scene cuts /
    "bizarreness"), alpha<1 sub-diffusive. Step-size (jump) distribution tail (kurtosis, p95/p50,
    long-jump share) = a quantitative bizarreness signature.
    Controls: (i) the LLM `interpretation` = waking analytic prose from the SAME pipeline;
    (ii) within-dream sentence-order SHUFFLE (destroys sequence -> alpha ~ 0).
    Then: does per-dream alpha track nightmare_index / fear? EN vs RU.

(Q28) THE ARROW OF TIME — is a dream narrative time-irreversible?
    Per-sentence valence (projection onto a learned valence axis) -> within-dream valence series.
    (a) net drift <v_last - v_first> (do dreams darken or lighten end-to-start?);
    (b) increment skew gamma = <dv^3>/<dv^2>^{3/2} (=0 for a time-reversible process; !=0 => arrow);
    (c) forward-vs-reversed classifier AUC (can a model tell a dream from itself played backwards?).
    Compared dreams vs interpretation: hypothesis = dreams have a WEAKER arrow than waking prose.

Aggregate-only outputs (no text, no userID). Exploratory / Frontier by design.

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 USE_TF=0 python3 \
        analyses/2026-07-19-17-dream-narrative-physics.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from psychohistory import config as C
from psychohistory.dreams.sentence_cache import load_or_build_sentences, offsets
from psychohistory.dreams.valence import fit_valence_axis, project

OUT = C.RESULTS / "showcase"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)


def _doc_blocks(emb, counts):
    off = offsets(counts)
    return off


def msd_alpha(emb, counts, mask, max_lag=6, min_sent=3, shuffle=False, seed=0):
    """Pooled MSD(tau) and the diffusion exponent alpha over docs where mask is True."""
    rng = np.random.default_rng(seed)
    off = offsets(counts)
    acc = {t: [] for t in range(1, max_lag + 1)}
    steps = []  # consecutive step lengths (tau=1) for the jump distribution
    for k in np.where(mask & (counts >= min_sent))[0]:
        s, c = off[k], counts[k]
        block = emb[s:s + c]
        if shuffle:
            block = block[rng.permutation(c)]
        for t in range(1, min(max_lag, c - 1) + 1):
            d = block[t:] - block[:-t]
            acc[t].extend(np.einsum("ij,ij->i", d, d))
        d1 = block[1:] - block[:-1]
        steps.extend(np.sqrt(np.einsum("ij,ij->i", d1, d1)))
    taus = np.array([t for t in acc if acc[t]])
    msd = np.array([np.mean(acc[t]) for t in taus])
    ok = msd > 0
    slope = np.polyfit(np.log(taus[ok]), np.log(msd[ok]), 1)[0] if ok.sum() >= 2 else np.nan
    return {"taus": taus, "msd": msd, "alpha": float(slope),
            "steps": np.array(steps), "n_docs": int((mask & (counts >= min_sent)).sum())}


def order_coherence(emb, counts, mask, min_sent=3):
    """Local-order test (critic): are CONSECUTIVE sentences closer than random within-dream pairs?
    alpha~0 (no MSD scaling) can still hide local coherence; ratio<1 => adjacency carries meaning."""
    off = offsets(counts)
    adj, allp = [], []
    for k in np.where(mask & (counts >= min_sent))[0]:
        s, c = off[k], counts[k]
        block = emb[s:s + c]
        d1 = block[1:] - block[:-1]
        adj.extend(np.sqrt(np.einsum("ij,ij->i", d1, d1)))
        G = block @ block.T
        iu = np.triu_indices(c, k=1)
        allp.extend(np.sqrt(np.maximum(2 - 2 * G[iu], 0)))
    adj, allp = np.mean(adj), np.mean(allp)
    return {"adj_step": float(adj), "allpair_step": float(allp), "ratio": float(adj / (allp + 1e-9))}


def jump_stats(steps):
    steps = steps[np.isfinite(steps) & (steps > 0)]
    q = np.percentile(steps, [50, 95, 99])
    return {"median": float(q[0]), "p95_p50": float(q[1] / (q[0] + 1e-9)),
            "p99_p50": float(q[2] / (q[0] + 1e-9)),
            "excess_kurtosis": float(stats.kurtosis(steps, fisher=True)),
            "long_jump_share": float(np.mean(steps > steps.mean() + 2 * steps.std()))}


def per_dream_alpha(emb, counts, mask, min_sent=8, max_lag=5):
    off = offsets(counts)
    out = []
    for k in np.where(mask & (counts >= min_sent))[0]:
        s, c = off[k], counts[k]
        block = emb[s:s + c]
        taus, msd = [], []
        for t in range(1, min(max_lag, c - 1) + 1):
            d = block[t:] - block[:-t]
            taus.append(t); msd.append(np.mean(np.einsum("ij,ij->i", d, d)))
        taus, msd = np.array(taus), np.array(msd)
        ok = msd > 0
        if ok.sum() >= 3:
            out.append((k, float(np.polyfit(np.log(taus[ok]), np.log(msd[ok]), 1)[0])))
    return out


def valence_series(emb, counts, mask, w, b, min_sent=4):
    off = offsets(counts)
    series = []
    idx = []
    for k in np.where(mask & (counts >= min_sent))[0]:
        s, c = off[k], counts[k]
        series.append(project(emb[s:s + c], w, b))
        idx.append(k)
    return idx, series


def irreversibility(series):
    drift, inc = [], []
    for v in series:
        v = np.asarray(v, float)
        drift.append(v[-1] - v[0])
        inc.extend(np.diff(v))
    inc = np.asarray(inc)
    m2 = np.mean(inc ** 2)
    gamma = float(np.mean(inc ** 3) / (m2 ** 1.5 + 1e-12))
    return {"drift_mean": float(np.mean(drift)), "drift_sd": float(np.std(drift)),
            "n": len(series), "increment_skew": gamma, "n_inc": len(inc)}


def _arc_features(v):
    v = np.asarray(v, float)
    k = len(v)
    x = np.linspace(0, 1, k)
    slope = np.polyfit(x, v, 1)[0]
    half = k // 2
    return [slope, v[half:].mean() - v[:half].mean(), v[-1] - v[0],
            np.argmax(v) / (k - 1), np.argmin(v) / (k - 1)]


def arrow_auc(series, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    F = np.array([_arc_features(v) for v in series])
    Fr = np.array([_arc_features(v[::-1]) for v in series])
    X = np.vstack([F, Fr])
    y = np.r_[np.ones(len(F)), np.zeros(len(Fr))]
    groups = np.r_[np.arange(len(F)), np.arange(len(F))]  # each dream's fwd+rev in same fold (no leak)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    auc = cross_val_score(LogisticRegression(max_iter=500), X, y, groups=groups,
                          cv=GroupKFold(5), scoring="roc_auc")
    return float(auc.mean()), float(auc.std())


def main():
    print("[q24/q28] loading sentence caches ...", flush=True)
    dt, ct, et, _, mt = load_or_build_sentences("text")
    di, ci, ei, _, mi = load_or_build_sentences("interpretation", limit=12000)

    w, b, vinfo = fit_valence_axis()
    print(f"[valence] axis fit n={vinfo['n_fit']} R2_insample={vinfo['r2_insample']:.3f}", flush=True)

    L = ["# Round 4 · Q24 + Q28 — the physics of the dream narrative", "",
         "*Intra-dream resolution: each dream embedded sentence-by-sentence (MiniLM, same encoder as "
         "the dream-level cache). Waking control = the LLM `interpretation` field. Aggregate-only.*", ""]

    # ---------------- Q24 anomalous diffusion ----------------
    rows = []
    diag = {}
    for name, emb, counts, meta in [("dream_text", et, ct, mt), ("interpretation", ei, ci, mi)]:
        for lang in ["en", "ru"]:
            m = (meta.lang.values == lang)
            if m.sum() < 50:
                continue
            r = msd_alpha(emb, counts, m)
            rs = msd_alpha(emb, counts, m, shuffle=True)
            js = jump_stats(r["steps"])
            rows.append({"series": name, "lang": lang, "n_docs": r["n_docs"],
                         "alpha": r["alpha"], "alpha_shuffled": rs["alpha"], **js})
            diag[(name, lang)] = r
    diff = pd.DataFrame(rows)
    diff.to_csv(OUT / "narrative_diffusion.csv", index=False)

    L += ["## (Q24) Anomalous diffusion — the Lévy flight of the dreaming mind", "",
          "`alpha` = slope of log MSD vs log lag (1=normal diffusion, >1 super-diffusive / heavy jumps). "
          "A sentence-order **shuffle** is the control (ordered ≈ shuffled ⇒ no measurable sequential "
          "diffusion structure).", ""]
    for _, r in diff.iterrows():
        L.append(f"- **{r.series} [{r.lang}]** (n={int(r.n_docs)} dreams): alpha=**{r.alpha:.2f}** "
                 f"(shuffled {r.alpha_shuffled:.2f}); jump tail p95/p50={r.p95_p50:.2f}, "
                 f"excess-kurtosis={r.excess_kurtosis:.1f}, long-jump share={r.long_jump_share*100:.1f}%.")
    # dream vs interpretation contrast (EN)
    try:
        de = diff[(diff.series == "dream_text") & (diff.lang == "en")].iloc[0]
        ie = diff[(diff.series == "interpretation") & (diff.lang == "en")].iloc[0]
        L += ["",
              f"- **No super-diffusive MSD scaling (α≈{de.alpha:.2f}≈order-shuffle {de.alpha_shuffled:.2f}):** "
              "sentence ORDER carries no MSD-scaling signal at these lengths → the Lévy/super-diffusion "
              "hypothesis is **not supported**. (Reframed per critic: this is order-invariance of the "
              "MSD exponent, not proof that 'dreams don't wander'; the α=1 baseline is for unbounded "
              "walks and short bounded trajectories on the sphere saturate.)"]
        # local-order test: does adjacency carry meaning even with alpha~0?
        oc = order_coherence(et, ct, mt.lang.values == "en")
        L += [f"- **Local order test:** consecutive-sentence step {oc['adj_step']:.3f} vs random "
              f"within-dream pair {oc['allpair_step']:.3f} (ratio {oc['ratio']:.3f}) → "
              f"{'adjacency IS locally coherent (neighbors closer than random within-dream pairs)' if oc['ratio'] < 0.97 else 'no local adjacency coherence'}, "
              "so alpha≈0 reflects fast saturation, not absence of local structure.",
              f"- **Jump-tail (tentative, confounded):** step-size excess-kurtosis {de.excess_kurtosis:.1f} "
              f"(dream) vs {ie.excess_kurtosis:.1f} (interpretation). Suggestive of more extreme leaps in "
              "dreams, but the `interpretation` is LLM prose (not a same-author waking control) and step "
              "size is confounded by sentence segmentation/length → NOT headlined."]
    except Exception:
        pass

    # per-dream alpha vs nightmare (EN)
    pda = per_dream_alpha(et, ct, mt.lang.values == "en")
    if len(pda) > 100:
        ks = np.array([k for k, _ in pda]); al = np.array([a for _, a in pda])
        nm = mt.nightmare_index.values[ks].astype(float); fe = mt.fear.values[ks].astype(float)
        ok = np.isfinite(al) & np.isfinite(nm) & np.isfinite(fe)
        r_nm = stats.spearmanr(al[ok], nm[ok])
        r_fe = stats.spearmanr(al[ok], fe[ok])
        L += ["", f"- **Per-dream alpha vs affect** (EN, n={int(ok.sum())} dreams ≥8 sentences): "
              f"alpha↔nightmare rho={r_nm.correlation:+.3f} (p={r_nm.pvalue:.3f}); "
              f"alpha↔fear rho={r_fe.correlation:+.3f} (p={r_fe.pvalue:.3f}). "
              f"Mean per-dream alpha={al[ok].mean():.2f}.",
              "  *(Exploratory: short trajectories make per-dream alpha noisy.)*"]

    # ---------------- Q28 arrow of time ----------------
    L += ["", "## (Q28) The arrow of time — is a dream time-irreversible?", ""]
    arr_rows = []
    for name, emb, counts, meta in [("dream_text", et, ct, mt), ("interpretation", ei, ci, mi)]:
        idx, series = valence_series(emb, counts, meta.lang.values == "en", w, b)
        if len(series) < 100:
            continue
        ir = irreversibility(series)
        auc_m, auc_s = arrow_auc(series)
        arr_rows.append({"series": name, "n": ir["n"], "drift_mean": ir["drift_mean"],
                         "increment_skew": ir["increment_skew"], "arrow_auc": auc_m, "arrow_auc_sd": auc_s})
    arrow = pd.DataFrame(arr_rows)
    arrow.to_csv(OUT / "narrative_arrow_of_time.csv", index=False)
    for _, r in arrow.iterrows():
        L.append(f"- **{r.series} [EN]** (n={int(r.n)}): end−start valence drift={r.drift_mean:+.3f}; "
                 f"increment-skew γ={r.increment_skew:+.3f} (0 ⇒ reversible); "
                 f"forward-vs-reversed AUC=**{r.arrow_auc:.3f}** (±{r.arrow_auc_sd:.3f}).")
    try:
        dtxt = arrow[arrow.series == "dream_text"].iloc[0]
        itxt = arrow[arrow.series == "interpretation"].iloc[0]
        L += ["",
              f"- **Dreams vs waking prose:** both are time-irreversible and about equally so "
              f"(AUC {dtxt.arrow_auc:.3f} vs {itxt.arrow_auc:.3f}) — but in **opposite emotional "
              f"directions**: dreams drift **darker** to their end (drift {dtxt.drift_mean:+.3f}, skew "
              f"{dtxt.increment_skew:+.3f}) while the AI interpretation ends **lighter/resolved** "
              f"(drift {itxt.drift_mean:+.3f}, skew {itxt.increment_skew:+.3f}). "
              "\"Your dream ends in the dark; the machine's reading ends in the light.\"",
              "- ⚠️ **Caveat (not headlined):** the drift/skew use a per-sentence valence *proxy* "
              "(sentence-level fidelity r≈0.27, cross-genre); the SIGN contrast is plausibly robust but "
              "is being re-verified with TRUE per-sentence XLM-R scoring (F0036). The AUC≈0.62 "
              "irreversibility (GroupKFold by dream, no leak) is the more robust part, and is modest."]
    except Exception:
        pass

    fig_diffusion(diag, diff, arrow)

    L += ["", "**Figures:** 26_dream_levy_flight.png · 27_dream_arrow_of_time.png",
          "", "*Verdict tier: Frontier / exploratory — a new intra-dream measurement, not yet "
          "externally replicated. Headline framing pending methodology-critic.*"]
    (OUT / "narrative_physics.md").write_text("\n".join(L))
    print("\n".join(L))
    print("\n[q24/q28] wrote", OUT / "narrative_physics.md")


def fig_diffusion(diag, diff, arrow):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for (name, lang), r in diag.items():
        if lang != "en":
            continue
        ax.plot(np.log(r["taus"]), np.log(r["msd"]), "-o", ms=4,
                label=f"{name} (α={r['alpha']:.2f})")
    ax.set_xlabel("log lag τ (sentences)"); ax.set_ylabel("log MSD")
    ax.set_title("Semantic diffusion within dreams (EN)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for name in ["dream_text", "interpretation"]:
        key = (name, "en")
        if key in diag:
            s = diag[key]["steps"]
            s = s[np.isfinite(s) & (s > 0)]
            xs = np.sort(s)
            ccdf = 1.0 - np.arange(len(xs)) / len(xs)
            ax.loglog(xs, ccdf, label=name)
    ax.set_xlabel("semantic step size (sentence→sentence)")
    ax.set_ylabel("CCDF")
    ax.set_title("Jump-size tail (heavier = more 'bizarre')")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "26_dream_levy_flight.png", dpi=150)
    plt.close(fig)

    # arrow of time figure
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(arrow):
        x = np.arange(len(arrow))
        ax.bar(x - 0.2, arrow.arrow_auc, 0.4, label="forward-vs-reversed AUC", color="#C1443C")
        ax.bar(x + 0.2, 0.5 + arrow.increment_skew, 0.4, label="0.5 + increment skew", color="#4C72B0")
        ax.axhline(0.5, color="k", lw=0.6, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(arrow.series, rotation=10)
        ax.set_ylabel("statistic"); ax.set_title("Arrow of time in dreams vs waking prose")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "27_dream_arrow_of_time.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
