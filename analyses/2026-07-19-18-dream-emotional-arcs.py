"""Round 4 / Q25 — the universal emotional arcs of dreams (Vonnegut/Reagan for dreams).

Reagan et al. (2016) showed novels reduce to ~6 emotional arc shapes. Nobody has asked whether
DREAMS have universal arcs. Here: per-sentence valence (projection onto the learned valence axis)
-> within-dream valence trajectory -> resample to fixed length -> SVD basis (the "shapes") +
KMeans prototypes. Then: is there a NIGHTMARE arc signature (where does dread peak?), and do the
same shapes appear in EN and RU?

Proxy honesty: the per-sentence valence is a projection, validated against the XLM-R scorer on a
held-out sentence subsample (report fidelity r). Aggregate-only.

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 USE_TF=0 python3 \
        analyses/2026-07-19-18-dream-emotional-arcs.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from psychohistory import config as C
from psychohistory.dreams.sentence_cache import load_or_build_sentences, offsets, split_sentences
from psychohistory.dreams.sentence_sentiment import load_or_build_sentence_sentiment
from psychohistory.dreams.valence import fit_valence_axis, project
from psychohistory.maps.embed import embed_texts

OUT = C.RESULTS / "showcase"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
L_ARC = 20  # resample length


def resample(v, L=L_ARC):
    x = np.linspace(0, 1, len(v))
    return np.interp(np.linspace(0, 1, L), x, v)


def build_arcs(sent, counts, mask, min_sent=5):
    """Arcs from TRUE per-sentence XLM-R sentiment (aligned to the sentence cache)."""
    off = offsets(counts)
    arcs, idx = [], []
    for k in np.where(mask & (counts >= min_sent))[0]:
        s, c = off[k], counts[k]
        v = sent[s:s + c]
        if not np.all(np.isfinite(v)):
            continue
        arcs.append(resample(v))
        idx.append(k)
    return np.array(arcs), np.array(idx)


def svd_modes(A):
    Ac = A - A.mean(1, keepdims=True)  # shape only (remove per-dream level)
    U, S, Vt = np.linalg.svd(Ac - Ac.mean(0), full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    return Vt, var, Ac


def cluster_arcs(Ac, k=6, seed=0):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init=5).fit(Ac)
    protos = np.array([Ac[km.labels_ == c].mean(0) for c in range(k)])
    sizes = np.bincount(km.labels_, minlength=k)
    return protos, sizes, km.labels_


def name_shape(p):
    a, m, z = p[:3].mean(), p[len(p)//2-1:len(p)//2+1].mean(), p[-3:].mean()
    if z - a > 0.15:
        return "rise (rags→riches)"
    if a - z > 0.15:
        return "fall (tragedy)"
    if m < a - 0.1 and m < z - 0.1:
        return "U / man-in-a-hole"
    if m > a + 0.1 and m > z + 0.1:
        return "∩ / Icarus"
    return "flat/other"


def validate_proxy(meta, w, b, n_dreams=350, seed=0):
    """Score a small sentence subsample with the real XLM-R model; correlate with the proxy."""
    from psychohistory.dreams.score import score_texts
    rng = np.random.default_rng(seed)
    raw = pd.read_csv(C.DREAMSEER_RAW, dtype=str, keep_default_na=False,
                      usecols=["documentID", "text"])
    tmap = dict(zip(raw.documentID, raw.text))
    sel = rng.choice(meta[meta.lang == "en"].documentID.values,
                     min(n_dreams, (meta.lang == "en").sum()), replace=False)
    sents = []
    for d in sel:
        sents.extend(split_sentences(tmap.get(d, "")))
    sents = [s for s in sents if len(s.split()) >= 2][:2500]
    emb = np.asarray(embed_texts(sents, batch=128), dtype=np.float32)
    proxy = project(emb, w, b)
    real = score_texts(sents, progress=False)["sentiment"].values
    r = stats.pearsonr(proxy, real)
    return {"n_sent": len(sents), "r": float(r[0]), "p": float(r[1])}


def main():
    print("[q25] loading sentence cache + sentiment ...", flush=True)
    dt, ct, et, _, mt = load_or_build_sentences("text")
    _d, _c, sent = load_or_build_sentence_sentiment("text", min_sent=5)
    w, b, vinfo = fit_valence_axis()

    L = ["# Round 4 · Q25 — the universal emotional arcs of dreams", "",
         "*TRUE per-sentence XLM-R sentiment → within-dream trajectory → SVD basis + KMeans shapes "
         "(Reagan et al. method). Aggregate-only.*", ""]

    # proxy fidelity (why we scored sentences directly rather than projecting)
    try:
        val = validate_proxy(mt, w, b)
        L += [f"- **Why true scoring:** the cheap valence *proxy* (dream-level axis) transfers poorly to "
              f"short sentences (proxy vs XLM-R r={val['r']:.2f}, n={val['n_sent']}), so arcs use the "
              "real per-sentence scorer.", ""]
    except Exception as e:
        L += [f"- (proxy validation skipped: {e})", ""]

    for lang in ["en", "ru"]:
        A, idx = build_arcs(sent, ct, mt.lang.values == lang)
        if len(A) < 200:
            L += [f"## {lang.upper()}: too few multi-sentence dreams (n={len(A)})", ""]
            continue
        Vt, var, Ac = svd_modes(A)
        protos, sizes, labels = cluster_arcs(Ac, k=6)
        # nightmare signature (NaN-safe)
        nm = mt.nightmare_index.values[idx].astype(float)
        fin = np.isfinite(nm)
        A, Ac, nm = A[fin], Ac[fin], nm[fin]
        hi = Ac[nm >= np.nanquantile(nm, 0.8)].mean(0)
        lo = Ac[nm <= np.nanquantile(nm, 0.2)].mean(0)
        endslope = A[:, -3:].mean(1) - A[:, :3].mean(1)
        r_end = stats.spearmanr(endslope, nm)
        minpos = np.array([np.argmin(a) for a in A]) / (L_ARC - 1)
        r_minpos = stats.spearmanr(minpos, nm)

        if lang == "en":
            fig_arcs(Vt, var, protos, sizes, hi, lo)
            pd.DataFrame(Vt[:4].T, columns=[f"mode{i+1}" for i in range(4)]).to_csv(
                OUT / "arcs_svd_modes_en.csv", index=False)
            pd.DataFrame(protos.T, columns=[f"cluster{i+1}" for i in range(6)]).to_csv(
                OUT / "arcs_prototypes_en.csv", index=False)

        L += [f"## {lang.upper()} — {len(A):,} dreams ≥5 sentences",
              f"- **Arc basis:** top-4 SVD shape-modes explain "
              f"{var[0]*100:.0f}/{var[:2].sum()*100:.0f}/{var[:3].sum()*100:.0f}/{var[:4].sum()*100:.0f}% "
              "of shape variance (cumulative).",
              "- **Six prototype shapes (KMeans):** " +
              "; ".join(f"{name_shape(protos[c])} ({sizes[c]})" for c in np.argsort(sizes)[::-1]),
              f"- **Nightmare signature:** end−start valence vs nightmare rho={r_end.correlation:+.3f} "
              f"(p={r_end.pvalue:.1e}); valence-trough position vs nightmare rho={r_minpos.correlation:+.3f} "
              f"(p={r_minpos.pvalue:.1e}).",
              f"  → high-nightmare dreams {'end darker (descending arc)' if r_end.correlation < -0.03 else 'do not systematically descend'}; "
              f"the dread trough sits {'later' if r_minpos.correlation>0.03 else 'earlier/uniformly'} in the dream.", ""]

    L += ["**Figures:** 33_dream_emotional_arcs.png",
          "", "*Verdict tier: Frontier / exploratory pending methodology-critic.*"]
    (OUT / "dream_emotional_arcs.md").write_text("\n".join(L))
    print("\n".join(L))
    print("\n[q25] wrote", OUT / "dream_emotional_arcs.md")


def fig_arcs(Vt, var, protos, sizes, hi, lo):
    x = np.linspace(0, 1, L_ARC)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    for i in range(3):
        ax.plot(x, Vt[i], label=f"mode {i+1} ({var[i]*100:.0f}%)")
    ax.axhline(0, color="k", lw=0.5); ax.set_title("Dream arc basis (SVD shape-modes, EN)")
    ax.set_xlabel("normalized dream time"); ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for c in np.argsort(sizes)[::-1]:
        ax.plot(x, protos[c], label=f"{name_shape(protos[c])} ({sizes[c]})")
    ax.axhline(0, color="k", lw=0.5); ax.set_title("Six prototype dream arcs (KMeans)")
    ax.set_xlabel("normalized dream time"); ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    ax.plot(x, hi, color="#C1443C", label="high-nightmare dreams")
    ax.plot(x, lo, color="#4C72B0", label="low-nightmare dreams")
    ax.axhline(0, color="k", lw=0.5); ax.set_title("Nightmare arc signature")
    ax.set_xlabel("normalized dream time"); ax.set_ylabel("valence (shape)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "33_dream_emotional_arcs.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
