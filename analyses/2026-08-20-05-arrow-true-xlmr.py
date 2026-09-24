"""The arrow of time under the primary instrument, against waking-narrative controls.

Two gaps decide how much the emotional arrow is worth, and both are measurement gaps rather
than analysis gaps.

1. INSTRUMENT. The cross-corpus arrow was measured with a scalable embedding-valence proxy: a
   ridge axis fit on DOCUMENT-level XLM-R sentiment and then applied to sentence embeddings. Its
   agreement with true per-sentence XLM-R is only r~0.27. The direction of the arrow therefore
   rests on an instrument that was never validated at the unit where it is read. This scores
   EVERY sentence of EVERY corpus with the real model.

2. COMPARATOR. The only non-dream corpus in the published run is the platform's own LLM
   interpretations - machine prose, not waking human narrative. If ordinary first-person
   accounts of real events also darken toward their endings, the finding is about narration and
   not about dreaming. Three controls are added, bracketing dream reports on narrativity and
   person: r/confession posts (first-person, real events, waking), Gutenberg fiction paragraphs
   (deliberate narrative), Wikipedia openings (coherent, non-narrative).

Everything else is held fixed: the same sentence splitter, the same >=4-sentence filter, the
same five arc features, the same grouped logistic classifier, and author-clustered inference
throughout (GroupKFold by author, cluster bootstrap over authors, author-level sign-flip
permutation, one-random-document-per-author). Awakening arms are re-run under the true
instrument, because the published awakening audit also used the proxy.

Writes aggregates only. The sentiment cache stores per-document sentence counts and per-sentence
scores - no text, no author key (docs/ETHICS.md S1).

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 \
        python3 analyses/2026-08-20-05-arrow-true-xlmr.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from psychohistory import config as C
from psychohistory.dreams.arrow import (AWAKEN, MIN_SENT_ARROW, TEXT_CORPORA, arrow_auc,
                                        drift_stats, one_per_author)
from psychohistory.dreams.sentence_cache import split_sentences

OUT = C.RESULTS / "showcase"
OUT.mkdir(parents=True, exist_ok=True)

MAXDOC = int(os.environ.get("MAXDOC", 6000))       # documents scored per corpus
MAX_SENT = int(os.environ.get("MAX_SENT", 25))     # sentence cap per document (arc protocol)
B_PERM = int(os.environ.get("B_PERM", 2000))
B_BOOT = int(os.environ.get("B_BOOT", 4000))

# Arms that test whether the arrow is an artifact of reports ending at the moment of waking.
ARMS = [
    ("full", None),
    ("no-wake-sentences", "nowake"),
    ("drop-last-sentence", "droplast"),
    ("interior (drop first+last)", "interior"),
    ("reports NOT ending at awakening", "notwakeend"),
    ("reports ending at awakening", "wakeend"),
]


def documents(name):
    """Split, filter and cap one corpus; returns (list of sentence lists, author array)."""
    _stem, _kind, loader = TEXT_CORPORA[name]
    texts, authors = loader()
    sents, auth = [], []
    for t, a in zip(texts, authors):
        ss = split_sentences(t, min_words=2)[:MAX_SENT]
        if len(ss) < MIN_SENT_ARROW:
            continue
        sents.append(ss)
        auth.append(a)
        if len(sents) >= MAXDOC:
            break
    return sents, np.array(auth, dtype=object)


def valences(name, sents):
    """True per-sentence XLM-R valence for one corpus, cached by corpus stem."""
    from psychohistory.dreams.fast_sentiment import score_sentences
    stem = TEXT_CORPORA[name][0]
    cache = C.INTERIM / f"true_xlmr_arrow_{stem}.npz"
    counts = np.array([len(s) for s in sents], np.int32)
    if cache.exists():
        z = np.load(cache)
        if len(z["counts"]) == len(counts) and bool((z["counts"] == counts).all()):
            print(f"  [cache] {cache.name}", flush=True)
            return [z["vals"][o:o + c] for o, c in zip(_offsets(counts), counts)]
        print(f"  [cache] {cache.name} stale -> rescore", flush=True)
    flat = [s for ss in sents for s in ss]
    print(f"  [xlmr] scoring {len(flat):,} sentences from {len(sents):,} documents", flush=True)
    vals = score_sentences(flat)
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    np.savez(cache, counts=counts, vals=vals)
    return [vals[o:o + c] for o, c in zip(_offsets(counts), counts)]


def _offsets(counts):
    return np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)


def arm(sents, vals, auth, how, min_sent=MIN_SENT_ARROW):
    """Apply an awakening arm, returning the surviving (trajectories, authors)."""
    ser, keep = [], []
    for ss, v, a in zip(sents, vals, auth):
        wake = np.array([bool(AWAKEN.search(s)) for s in ss])
        if how == "nowake":
            v = v[~wake]
        elif how == "droplast":
            v = v[:-1]
        elif how == "interior":
            v = v[1:-1]
        elif how == "notwakeend":
            if wake[-1]:
                continue
        elif how == "wakeend":
            if not wake[-1]:
                continue
        if len(v) >= min_sent:
            ser.append(np.asarray(v, float))
            keep.append(a)
    return ser, np.array(keep, dtype=object)


def analyze(name, kind, sents, vals, auth):
    rows = []
    wake_end = float(np.mean([bool(AWAKEN.search(ss[-1])) for ss in sents]))
    wake_any = float(np.mean([any(AWAKEN.search(s) for s in ss) for ss in sents]))
    print(f"\n[{name}] docs={len(sents):,} authors={len(np.unique(auth)):,} "
          f"sentences={sum(len(s) for s in sents):,} | ending at awakening={wake_end:.1%} "
          f"| containing one={wake_any:.1%}", flush=True)
    for label, how in ARMS:
        ser, g = arm(sents, vals, auth, how)
        if len(ser) < 200:
            print(f"  {label:34} n={len(ser)} -> skip", flush=True)
            continue
        auc, p_auc, nA = arrow_auc(ser, g, B=B_PERM)
        st = drift_stats(ser, g, B=B_BOOT)
        opa = one_per_author(ser, g)
        rows.append({"corpus": name, "kind": kind, "arm": label,
                     "n_docs": len(ser), "n_authors": nA, "auc_author": auc, "p_auc": p_auc,
                     "frac_ending_awake": wake_end, "frac_any_awake": wake_any,
                     "drift_1pa": opa["drift_1pa"], **st})
        print(f"  {label:34} n={len(ser):6,} A={nA:6,} AUC={auc:.3f} (p={p_auc:.4f}) "
              f"drift={st['drift']:+.4f}[{st['drift_lo']:+.4f},{st['drift_hi']:+.4f}] "
              f"auth={st['drift_auth']:+.4f} p_cl={st['p_cluster']:.4f} "
              f"1/auth={opa['drift_1pa']:+.4f} neg={st['frac_authors_neg']:.1%}", flush=True)
    return rows


def main():
    rows = []
    for name, (_stem, kind, _l) in TEXT_CORPORA.items():
        try:
            sents, auth = documents(name)
            if len(sents) < 500:
                print(f"[{name}] only {len(sents)} documents -> skip", flush=True)
                continue
            vals = valences(name, sents)
            rows += analyze(name, kind, sents, vals, auth)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "arrow_true_xlmr.csv", index=False)
    _write_md(df)
    print(f"\n[true-xlmr] wrote {OUT / 'arrow_true_xlmr.md'}")


def _write_md(df):
    L = ["# The emotional arrow under true per-sentence XLM-R, with waking-narrative controls", "",
         "*Every sentence of every corpus scored with the study's primary sentiment instrument "
         "(`cardiffnlp/twitter-xlm-roberta-base-sentiment`, `sentiment = P(pos) - P(neg)`), "
         "replacing the embedding-valence proxy whose sentence-level agreement with this model is "
         "only r~0.27. Inference is author-clustered throughout: GroupKFold by author for the "
         f"forward-versus-reversed AUC, {B_PERM:,} author-grouped label-flip permutations, a "
         f"{B_BOOT:,}-draw cluster bootstrap over authors for the drift interval, an author-level "
         "sign-flip permutation, and a one-random-document-per-author estimate.*", "",
         "`drift` is dream-level mean end-minus-start valence; `auth` weights every author "
         "equally; `1/auth` keeps a single random document per author. `neg` is the fraction of "
         "authors whose own mean drift is negative.", ""]
    for kind in ["dream", "waking narrative", "written narrative", "non-narrative", "waking"]:
        sub = df[df.kind == kind]
        if not len(sub):
            continue
        L += [f"## {kind}", "",
              "| corpus | arm | n | authors | AUC | p | drift [95% CI] | author-weighted | "
              "1/author | p (author) | authors darkening |",
              "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for _, r in sub.iterrows():
            L.append(f"| {r.corpus} | {r.arm} | {int(r.n_docs):,} | {int(r.n_authors):,} | "
                     f"**{r.auc_author:.3f}** | {r.p_auc:.4f} | "
                     f"**{r.drift:+.4f}** [{r.drift_lo:+.4f}, {r.drift_hi:+.4f}] | "
                     f"{r.drift_auth:+.4f} | {r.drift_1pa:+.4f} | {r.p_cluster:.4f} | "
                     f"{r.frac_authors_neg:.1%} |")
        L.append("")
    full = df[df.arm == "full"]
    L += ["## Read-out", ""]
    for _, r in full.iterrows():
        L.append(f"- **{r.corpus}** ({r.kind}): AUC {r.auc_author:.3f}, drift "
                 f"{r.drift:+.4f} [{r.drift_lo:+.4f}, {r.drift_hi:+.4f}].")
    L += ["", "The decisive comparisons are (i) dream corpora against the waking first-person "
          "narrative control, which asks whether the descent belongs to dreaming or to narration, "
          "and (ii) the `full` versus trimming arms, which ask whether it is manufactured by "
          "reports ending at the moment of waking.", "",
          "*Aggregate-only: no text, author key or per-document score is written.*"]
    (OUT / "arrow_true_xlmr.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
