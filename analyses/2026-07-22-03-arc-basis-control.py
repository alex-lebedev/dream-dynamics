"""Is the cross-corpus emotional-arc structure dream-specific, or a generic property of smooth
resampled valence curves? Three proper controls (answering the "did one control wrongly kill it?").

For each corpus we build within-dream valence arcs (proxy valence, resampled to 20 pts, per-dream
level removed), then compare REAL arcs to ORDER-SHUFFLED arcs (sentence order permuted per dream,
averaged over many shuffles):

  (1) LOW-DIMENSIONALITY: top-4 cumulative SVD variance-explained (of the residual around the mean
      shape). If real >> shuffle, the low-D basis reflects genuine temporal structure, not just the
      smoothness of resampled curves.
  (2) CROSS-CORPUS ALIGNMENT: mean off-diagonal |cos| of SVD mode-1 between dream corpora, REAL vs
      SHUFFLED. If real > shuffle, the shared basis is dream-specific; if real ~= shuffle, generic.
  (3) MEAN ARC (the ARROW): end-start of the population MEAN arc, REAL vs SHUFFLE. Shuffling should
      flatten a genuine descending arrow -> isolates the dream-specific signal (= the arrow of time).

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 USE_TF=0 python3 \
        analyses/2026-07-22-03-arc-basis-control.py
"""
from __future__ import annotations

import glob

import numpy as np

from psychohistory import config as C
from psychohistory.dreams.valence import fit_valence_axis, project

L_ARC = 20
MIN_SENT = 5
N_SHUF = 30
DREAM = ["DreamSeer_EN", "DreamSeer_RU", "DreamBank", "Reddit_rDreams", "SDDb"]


def _resample(v):
    return np.interp(np.linspace(0, 1, L_ARC), np.linspace(0, 1, len(v)), v)


def _offsets(counts):
    return np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)


def arcs_from(vals, counts, shuffle=False, seed=0):
    off = _offsets(counts)
    rng = np.random.default_rng(seed)
    out = []
    for k in np.where(counts >= MIN_SENT)[0]:
        v = vals[off[k]:off[k] + counts[k]]
        if shuffle:
            v = rng.permutation(v)
        out.append(_resample(v))
    return np.array(out)


def basis(A):
    """Return (top1_var, top4_var, mode1, mean_arc). mean_arc = population mean of level-removed arcs."""
    Ac = A - A.mean(1, keepdims=True)          # remove per-dream level
    mean_arc = Ac.mean(0)                        # the average arc SHAPE (the arrow)
    Acc = Ac - mean_arc                          # residual around the mean shape
    S = np.linalg.svd(Acc, compute_uv=False)
    _, _, Vt = np.linalg.svd(Acc, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    return float(var[0]), float(var[:4].sum()), Vt[0], mean_arc


def _cos(a, b):
    return abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))


def main():
    w, b, _ = fit_valence_axis()
    caches = {}
    for f in glob.glob(str(C.INTERIM / "emb_sent_ext_*.npz")):
        name = f.split("emb_sent_ext_")[1].rsplit(".npz", 1)[0].rstrip(".")
        caches[name] = f

    real_m1, real_marc = {}, {}
    m1_sh_all, marc_sh_all = {}, {}     # per corpus: (N_SHUF, L) arrays (NO vector averaging — sign-arbitrary)
    print(f"{'corpus':16} | top1/top4 var REAL | top4 var SHUF | mean-arc end-start REAL | SHUF | self-shuf |cos|")
    print("-" * 112)
    rows = {}
    for name in DREAM + ["DreamSeer_interp"]:
        f = caches.get(name)
        if not f:
            continue
        z = np.load(f, allow_pickle=True)
        emb, counts = z["emb"], z["counts"]
        vals = project(emb, w, b)
        A = arcs_from(vals, counts)
        t1, t4, m1, marc = basis(A)
        t4_sh, es_sh, m1_list, marc_list = [], [], [], []
        for s in range(N_SHUF):
            As = arcs_from(vals, counts, shuffle=True, seed=s)
            _, t4s, m1s, marcs = basis(As)
            t4_sh.append(t4s); es_sh.append(marcs[-3:].mean() - marcs[:3].mean())
            m1_list.append(m1s); marc_list.append(marcs)
        es_real = marc[-3:].mean() - marc[:3].mean()
        real_m1[name] = m1; real_marc[name] = marc
        m1_sh_all[name] = np.array(m1_list); marc_sh_all[name] = np.array(marc_list)
        # self-shuffle |cos|: per-seed |cos(real m1, shuffled m1)| then average (sign-safe via abs)
        self_cos = float(np.mean([_cos(m1, m1s) for m1s in m1_list]))
        rows[name] = dict(t1=t1, t4=t4, t4_sh=np.mean(t4_sh), es_real=es_real, es_sh=np.mean(es_sh),
                          self_cos=self_cos)
        print(f"{name:16} |   {t1:.0%} / {t4:.0%}       |    {np.mean(t4_sh):.0%}       |"
              f"      {es_real:+.3f}           |{np.mean(es_sh):+.3f}|    {self_cos:.2f}")

    dn = [n for n in DREAM if n in real_m1]

    def offdiag_real(md):
        vv = [_cos(md[a], md[bn]) for i, a in enumerate(dn) for bn in dn[i + 1:]]
        return float(np.mean(vv)), float(np.min(vv))

    def offdiag_shuf_perseed(md_all):
        """Mean over seeds of the off-diagonal |cos| between corpora's shuffled vectors (sign-safe)."""
        per = []
        for s in range(N_SHUF):
            vv = [_cos(md_all[a][s], md_all[bn][s]) for i, a in enumerate(dn) for bn in dn[i + 1:]]
            per.append(np.mean(vv))
        return float(np.mean(per)), float(np.min([min(_cos(md_all[a][s], md_all[bn][s])
                    for i, a in enumerate(dn) for bn in dn[i + 1:]) for s in range(N_SHUF)]))

    rm1, rm1min = offdiag_real(real_m1)
    sm1, sm1min = offdiag_shuf_perseed(m1_sh_all)
    rma, rmamin = offdiag_real(real_marc)
    sma, smamin = offdiag_shuf_perseed(marc_sh_all)

    print("\n== CROSS-CORPUS |cos| between dream corpora (sign-safe, shuffle per-seed) ==")
    print(f"  VARIANCE mode-1 : REAL {rm1:.3f} (min {rm1min:.3f})  vs  SHUFFLED {sm1:.3f} (min {sm1min:.3f})  Δ={rm1-sm1:+.3f}")
    print(f"  MEAN ARC (arrow): REAL {rma:.3f} (min {rmamin:.3f})  vs  SHUFFLED {sma:.3f} (min {smamin:.3f})  Δ={rma-sma:+.3f}")

    print("\n== VERDICT LOGIC ==")
    lowd = np.mean([rows[n]['t4'] - rows[n]['t4_sh'] for n in dn])
    arrow = np.mean([rows[n]['es_real'] for n in dn]); arrow_sh = np.mean([rows[n]['es_sh'] for n in dn])
    print(f"  (1) low-D: mean top-4 var REAL {np.mean([rows[n]['t4'] for n in dn]):.0%} vs SHUF "
          f"{np.mean([rows[n]['t4_sh'] for n in dn]):.0%}  (Δ={lowd:+.1%})  [modest -> low-D is mostly generic]")
    print(f"  (2) cross-corpus VARIANCE mode-1: REAL {rm1:.2f} vs SHUF {sm1:.2f}  "
          f"[{'DREAM-SPECIFIC' if rm1-sm1 > 0.10 else 'generic'}]")
    print(f"  (3) cross-corpus MEAN-ARC shape: REAL {rma:.2f} vs SHUF {sma:.2f}  "
          f"[{'DREAM-SPECIFIC' if rma-sma > 0.10 else 'generic'}]")
    print(f"  (4) mean-arc end-start (arrow): REAL {arrow:+.3f} vs SHUF {arrow_sh:+.3f}  "
          f"[{'DREAM-SPECIFIC' if abs(arrow)-abs(arrow_sh) > 0.02 else 'generic'}]")


if __name__ == "__main__":
    main()
