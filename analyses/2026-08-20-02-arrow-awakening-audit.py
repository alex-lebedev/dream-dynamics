"""Is the emotional arrow an AWAKENING artifact? A direct audit of the report's ending.

The strongest alternative explanation of "dream reports end darker than they begin" is not about
dreaming at all. Emotionally intense or threatening content promotes awakening and recall, so a
report may end dark simply because the dream *ended* — the negative event is what woke the person
and terminated the narrative. A second, related account is narrative: writers put the climax last.
Neither requires any time asymmetry in dreaming.

This audit strips the endings and asks whether the arrow survives.

  full            : all sentences (baseline)
  no-wake-sent    : sentences mentioning waking/awakening removed (EN + RU markers)
  drop-last       : final sentence of every report removed
  drop-last-2     : final two sentences removed
  interior        : first AND last sentence removed (pure interior gradient)
  no-wake-ending  : reports whose FINAL sentence mentions awakening are excluded entirely
  wake-ending only: the complement (reports that do end at awakening) — expected to be strongest
  no-nightmare    : top-decile text-threat reports excluded

Every arm is re-estimated with AUTHOR-clustered inference (cluster bootstrap CI + author-level
sign-flip p), on all five dream corpus-language samples plus the LLM-interpretation comparator,
under the embedding-axis proxy, and additionally on Dreamseer EN/RU under TRUE per-sentence
XLM-R. If the arrow persists with the endings deleted, the awakening account cannot carry it.

Aggregate-only outputs.

    cd psychohistory && PYTHONPATH=src PYTHONNOUSERSITE=1 USE_TF=0 \
        /opt/anaconda3/bin/python3 analyses/2026-08-20-02-arrow-awakening-audit.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from psychohistory import config as C
from psychohistory.dreams.arrow import (AWAKEN, CORPORA, MIN_SENT_ARROW, arrow_auc,
                                        dreamseer_true_xlmr, drift_stats, load_corpus,
                                        proxy_records)
from psychohistory.dreams.valence import fit_valence_axis, project

OUT = C.RESULTS / "showcase"
OUT.mkdir(parents=True, exist_ok=True)
B_PERM = int(os.environ.get("B_PERM", 2000))
B_BOOT = int(os.environ.get("B_BOOT", 2000))
MIN_KEEP = 3        # a trajectory needs >=3 points after trimming to have a slope worth reading
# All TRIMMING arms are computed on the same report set (>=6 sentences) so that differences
# between arms reflect the deleted sentences and not a change in which reports qualify.
MIN_SENT_AUDIT = 6


# ---- the arms: each maps a record -> a trimmed value vector (or None to drop the record) ------
def _wake_mask(sents):
    return np.array([bool(AWAKEN.search(s)) for s in sents])


def arm_full(r):
    return r["vals"]


def arm_no_wake_sent(r):
    m = ~_wake_mask(r["sents"])
    return r["vals"][m]


def arm_drop_last(r):
    return r["vals"][:-1]


def arm_drop_last2(r):
    return r["vals"][:-2]


def arm_interior(r):
    return r["vals"][1:-1]


def arm_no_wake_ending(r):
    return None if _wake_mask(r["sents"])[-1] else r["vals"]


def arm_wake_ending(r):
    return r["vals"] if _wake_mask(r["sents"])[-1] else None


ARMS = [("full", arm_full),
        ("no-wake-sentences", arm_no_wake_sent),
        ("drop-last-sentence", arm_drop_last),
        ("drop-last-2-sentences", arm_drop_last2),
        ("interior (drop first+last)", arm_interior),
        ("reports NOT ending at awakening", arm_no_wake_ending),
        ("reports ending at awakening", arm_wake_ending)]


def run_arm(name, proxy, arm_name, recs, fn, thr_cut=None):
    ser, auth = [], []
    for r in recs:
        if thr_cut is not None and r.get("threat", 0.0) >= thr_cut:
            continue
        v = fn(r)
        if v is None or len(v) < MIN_KEEP:
            continue
        ser.append(np.asarray(v, float)); auth.append(r["author"])
    if len(ser) < 100:
        return None
    auth = np.array(auth, dtype=object)
    auc, p, n_auth = arrow_auc(ser, auth, B=B_PERM)
    ds = drift_stats(ser, auth, B=B_BOOT)
    row = {"corpus": name, "proxy": proxy, "arm": arm_name, "n_dreams": ds["n_dreams"],
           "n_authors": n_auth, "auc": auc, "auc_p": p,
           **{k: v for k, v in ds.items() if k not in ("n_dreams", "n_authors")}}
    print(f"  {arm_name:34} n={ds['n_dreams']:6,} A={n_auth:6,} AUC={auc:.3f} "
          f"drift={ds['drift_auth']:+.4f}[{ds['drift_auth_lo']:+.4f},{ds['drift_auth_hi']:+.4f}] "
          f"p={ds['p_cluster']:.4f}", flush=True)
    return row


def audit(name, proxy, recs, rows):
    recs = [r for r in recs if len(r["vals"]) >= MIN_SENT_AUDIT]
    if len(recs) < 200:
        print(f"[{name} / {proxy}] too few >={MIN_SENT_AUDIT}-sentence reports -> skip", flush=True)
        return
    print(f"\n[{name} / {proxy}] n_records={len(recs):,} (>={MIN_SENT_AUDIT} sentences)", flush=True)
    wake_end = np.mean([bool(AWAKEN.search(r["sents"][-1])) for r in recs])
    wake_any = np.mean([any(AWAKEN.search(s) for s in r["sents"]) for r in recs])
    print(f"  ending at awakening: {wake_end:.1%} | containing an awakening sentence: "
          f"{wake_any:.1%}", flush=True)
    for arm_name, fn in ARMS:
        r = run_arm(name, proxy, arm_name, recs, fn)
        if r:
            r["wake_end_rate"] = wake_end
            rows.append(r)
    if "threat" in recs[0]:
        cut = float(np.quantile([r["threat"] for r in recs], 0.9))
        r = run_arm(name, proxy, "no-nightmare (drop top-decile threat)", recs, arm_full, cut)
        if r:
            r["wake_end_rate"] = wake_end
            rows.append(r)


def main():
    w, b, vinfo = fit_valence_axis()
    print(f"[valence] axis fit n={vinfo['n_fit']} R2={vinfo['r2_insample']:.3f}", flush=True)
    rows = []
    for name in CORPORA:
        try:
            cp = load_corpus(name)
        except Exception as e:
            print(f"[{name}] SKIP {e}", flush=True); continue
        audit(name, "embed_axis", proxy_records(cp, project(cp.emb, w, b)), rows)
        del cp
    for lang in ("en", "ru"):
        try:
            recs = dreamseer_true_xlmr(lang)
        except Exception as e:
            print(f"[true-xlmr {lang}] SKIP {e}", flush=True); continue
        if len(recs) >= 200:
            audit(f"DreamSeer {lang.upper()}", "true_xlmr", recs, rows)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "arrow_awakening_audit.csv", index=False)
    _write_md(df)
    print("\n[awakening-audit] wrote", OUT / "arrow_awakening_audit.md")


def _write_md(df):
    L = ["# Is the emotional arrow an awakening artifact?", "",
         "*Threatening content promotes awakening and recall, so a dream report may end dark "
         "because the dream ended — not because dreaming is time-asymmetric. A narrative account "
         "(the climax goes last) predicts the same thing. This audit deletes the endings and "
         "re-estimates the arrow. All arms use AUTHOR-clustered inference: the CI is a cluster "
         "bootstrap over authors and the p-value is an author-level sign-flip permutation "
         f"(B={B_BOOT:,}); AUC permutation B={B_PERM:,}.*", "",
         "Drift is reported as the **author-weighted** mean end−start valence (equal weight per "
         "author), which is the honest effect size when one dreamer can supply 1,853 reports.", "",
         f"All arms are computed on reports with **>={MIN_SENT_AUDIT} sentences**, so that "
         "differences between arms reflect the deleted sentences rather than a change in which "
         "reports qualify. The `full` row is therefore the matched baseline for this table and is "
         "not identical to the published whole-corpus estimate.", ""]
    for (corpus, proxy), g in df.groupby(["corpus", "proxy"], sort=False):
        we = g.wake_end_rate.iloc[0]
        L += [f"## {corpus} — {proxy}",
              f"*{we:.1%} of reports end on a sentence mentioning awakening.*", "",
              "| arm | n dreams | n authors | AUC | author-weighted drift [95% CI] | p (author-level) |",
              "|---|--:|--:|--:|--:|--:|"]
        for _, r in g.iterrows():
            L.append(f"| {r.arm} | {int(r.n_dreams):,} | {int(r.n_authors):,} | {r.auc:.3f} | "
                     f"{r.drift_auth:+.4f} [{r.drift_auth_lo:+.4f},{r.drift_auth_hi:+.4f}] | "
                     f"{r.p_cluster:.4f} |")
        L.append("")
    # headline summary: does the arrow survive the two hardest arms?
    hard = df[df.arm.isin(["drop-last-sentence", "interior (drop first+last)",
                           "reports NOT ending at awakening", "no-wake-sentences"])]
    dream = hard[~hard.corpus.str.contains("interp")]
    L += ["## Read-out",
          f"- Across the four ending-removal arms and all dream rows, the author-weighted drift "
          f"stays negative in {int((dream.drift_auth < 0).sum())}/{len(dream)} cases.",
          "- If the arrow were manufactured by reports terminating at a frightening awakening, "
          "deleting the final sentence (or every awakening sentence, or every report that ends at "
          "one) should abolish it. Compare the `full` row with `drop-last-sentence`, `interior` "
          "and `reports NOT ending at awakening` in each block above.",
          "- `reports ending at awakening` is expected to be the most negative arm; that "
          "gradient is itself informative about how much of the effect the ending contributes.",
          "", "*Aggregate-only. No author key, user ID or dream text is written by this script.*"]
    (OUT / "arrow_awakening_audit.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
