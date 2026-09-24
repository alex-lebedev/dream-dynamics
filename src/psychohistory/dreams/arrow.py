"""Author-aware re-analysis of the intra-dream emotional ARROW OF TIME.

`analyses/2026-07-22-01-arrow-arcs-external.py` established the arrow (forward-vs-reversed
AUC + end-start darkening) across five corpus-language samples, but treated every dream as an
independent unit and grouped its cross-validation by DREAM. Two corpora make that indefensible:
DreamBank's 15k reports come from 50 dreamers and one SDDb participant contributes 30% of that
sample, so dream-level inference badly overstates precision.

This module re-opens the (de-identified) per-sentence embedding caches that script wrote and
re-attaches, IN MEMORY ONLY, the per-document author key and the sentence text, by replaying
each corpus loader with byte-identical filters and the identical sentence splitter. Alignment is
not assumed: `load_corpus` asserts the replayed per-document sentence counts equal the cached
`counts` vector element-wise, which pins every cached embedding row to its sentence.

Nothing here writes text, author keys or user IDs to disk (docs/ETHICS.md S1); callers emit
aggregates only.

    from psychohistory.dreams.arrow import load_corpus, arrow_auc, drift_stats
"""
from __future__ import annotations

import glob
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config as C
from .sentence_cache import split_sentences

# ---- constants copied from analyses/2026-07-22-01 (comparability is the point) ---------------
MAXDOC = 15000
MIN_SENT_ARROW = 4
L_ARC = 20

# Awakening / end-of-sleep markers, EN + RU. Used to test whether the descending arrow is an
# artifact of reports terminating at the moment of waking (threat -> arousal -> recall).
AWAKEN = re.compile(
    r"(\bwoke\b|\bawoke\b|\bawaken(ed|ing)?\b|\bwoken\b|\bwake up\b|\bwaking up\b|"
    r"\bopened my eyes\b|\balarm went off\b|"
    r"просну|пробуди|разбуди|просыпа|очнул|пробужден|открыл[а]? глаза)", re.IGNORECASE)

# Same threat lexicon as the published run, for the nightmare-exclusion arm.
THREAT = re.compile(
    r"\b(chas(e|ed|ing)|attack|kill(ed|ing)?|murder|death|dead|die[ds]?|dying|fall(ing)?|fell|"
    r"monster|demon|ghost|zombie|scream(ed|ing)?|blood(y)?|afraid|fear(ful)?|terror|terrified|"
    r"panic|trapp?ed|escap(e|ing)|drown(ed|ing)?|war|gun|knife|weapon|snake|spider|threat|danger|"
    r"nightmare|lost|paralyz(ed|e)|suffocat|hunt(ed|ing)?|explos|fire|flee(ing)?|fled)\b",
    re.IGNORECASE)


# ---- corpus loaders: (texts, authors), filters identical to the published run ------------------
def _dreamseer(lang):
    lv = pd.read_parquet(C.DREAMS_OUT / "dreamseer_dream_level.parquet",
                         columns=["documentID", "date", "lang", "userID"])
    lv["date"] = pd.to_datetime(lv.date)
    lv = lv[(lv.lang == lang) & (lv.date >= "2024-03-01")]
    raw = pd.read_csv(C.DREAMSEER_RAW, dtype=str, keep_default_na=False,
                      usecols=["documentID", "text"])
    d = lv.merge(raw, on="documentID")
    keep = [(t, u) for t, u in zip(d.text, d.userID) if isinstance(t, str) and len(t) >= 40]
    return [x[0] for x in keep], [x[1] for x in keep]


def _dreamseer_interp():
    raw = pd.read_csv(C.DREAMSEER_RAW, dtype=str, keep_default_na=False,
                      usecols=["documentID", "interpretation"])
    # the published loader read `interpretation` in file order with no user join; the dream-level
    # table supplies the author key for the same documentIDs.
    lv = pd.read_parquet(C.DREAMS_OUT / "dreamseer_dream_level.parquet",
                         columns=["documentID", "userID"])
    umap = dict(zip(lv.documentID, lv.userID))
    keep = [(t, umap.get(d, f"__unknown_{i}"))
            for i, (d, t) in enumerate(zip(raw.documentID, raw.interpretation))
            if isinstance(t, str) and len(t) >= 60]
    return [x[0] for x in keep], [x[1] for x in keep]


def _dreambank():
    db = pd.read_csv(C.DREAMBANK_RAW, usecols=["content", "dreamer"], dtype=str,
                     keep_default_na=False)
    keep = [(t, u) for t, u in zip(db.content, db.dreamer) if isinstance(t, str) and len(t) >= 60]
    return [x[0] for x in keep], [x[1] for x in keep]


def _reddit():
    r = pd.read_csv(C.RAW / "mallett" / "r-dreams.csv", usecols=["selftext", "author"],
                    dtype=str, keep_default_na=False)
    keep = [(t, u) for t, u in zip(r.selftext, r.author) if isinstance(t, str) and len(t) >= 100]
    return [x[0] for x in keep], [x[1] for x in keep]


def _sddb():
    f = sorted(glob.glob(str(C.RAW / "sddb" / "*.csv")))[0]
    s = pd.read_csv(f, usecols=["Dream Text", "Participant ID"], dtype=str, keep_default_na=False)
    keep = [(t, u) for t, u in zip(s["Dream Text"], s["Participant ID"])
            if isinstance(t, str) and len(t) >= 60]
    return [x[0] for x in keep], [x[1] for x in keep]


def _baseline(name, min_chars=300):
    """Non-dream natural-language control corpora (scripts/fetch_language_baselines.py).

    The arrow analysis has no waking-text comparator: if ordinary first-person narratives also
    darken toward their endings, the effect belongs to narration rather than to dreaming. These
    supply that test. `author` is a document id for Reddit/Wikipedia and a book id for Gutenberg,
    so author-clustered inference is meaningful for fiction and conservative elsewhere.
    """
    f = C.EXTERNAL / "language_baselines" / f"{name}.csv"
    d = pd.read_csv(f, dtype=str, keep_default_na=False)
    keep = [(t, a) for t, a in zip(d.text, d.author)
            if isinstance(t, str) and len(t) >= min_chars]
    return [x[0] for x in keep], [x[1] for x in keep]


# display name -> (cache stem, kind, loader)
CORPORA = {
    "DreamSeer EN":      ("DreamSeer_EN", "dream", lambda: _dreamseer("en")),
    "DreamSeer RU":      ("DreamSeer_RU", "dream", lambda: _dreamseer("ru")),
    "DreamBank":         ("DreamBank", "dream", _dreambank),
    "Reddit r/Dreams":   ("Reddit_rDreams", "dream", _reddit),
    "SDDb":              ("SDDb", "dream", _sddb),
    "DreamSeer interp.": ("DreamSeer_interp.", "machine prose", _dreamseer_interp),
}

# Everything the true-per-sentence run scores from raw text: the six cached corpora above plus
# the three non-dream controls, which have no embedding cache because they are new here.
TEXT_CORPORA = dict(CORPORA, **{
    "r/confession (waking narrative)":
        ("confession", "waking narrative", lambda: _baseline("personal-narrative")),
    "Gutenberg fiction":
        ("fiction", "written narrative", lambda: _baseline("fiction")),
    "Wikipedia openings":
        ("wikipedia", "non-narrative", lambda: _baseline("encyclopedic")),
})


@dataclass
class Corpus:
    name: str
    kind: str
    emb: np.ndarray              # (S, 384) per-sentence embeddings, doc-contiguous
    counts: np.ndarray           # (D,) sentences per doc
    vader: np.ndarray            # (S,) per-sentence VADER compound
    threat: np.ndarray           # (D,) text-derived threat rate
    authors: np.ndarray          # (D,) author key (in memory only)
    sents: list = field(default_factory=list)   # (D,) list of sentence lists

    @property
    def offsets(self):
        return np.concatenate([[0], np.cumsum(self.counts)[:-1]]).astype(np.int64)


def load_corpus(name: str) -> Corpus:
    """Load the cached per-sentence arrays and re-attach author + sentence text by replay.

    Raises if the replay does not reproduce the cached per-document sentence counts exactly,
    which is what licenses indexing the cached embeddings by sentence.
    """
    stem, kind, loader = CORPORA[name]
    z = np.load(C.INTERIM / f"emb_sent_ext_{stem}.npz", allow_pickle=True)
    counts = z["counts"]
    texts, authors = loader()
    rep_counts, rep_auth, rep_sents = [], [], []
    for t, a in zip(texts, authors):
        ss = split_sentences(t, min_words=2)
        if len(ss) < MIN_SENT_ARROW:
            continue
        rep_counts.append(len(ss)); rep_auth.append(a); rep_sents.append(ss)
        if len(rep_counts) >= MAXDOC:
            break
    rep_counts = np.array(rep_counts, np.int32)
    if len(rep_counts) != len(counts) or not bool((rep_counts == counts).all()):
        raise RuntimeError(f"[{name}] replay does not match cache "
                           f"(cache n={len(counts)}, replay n={len(rep_counts)})")
    return Corpus(name=name, kind=kind, emb=z["emb"], counts=counts, vader=z["vader"],
                  threat=z["threat"], authors=np.array(rep_auth, dtype=object), sents=rep_sents)


# ---- arrow statistics: definitions identical to analyses/2026-07-22-01 ------------------------
def arc_features(v):
    v = np.asarray(v, float); k = len(v); x = np.linspace(0, 1, k)
    slope = np.polyfit(x, v, 1)[0]; half = k // 2
    return [slope, v[half:].mean() - v[:half].mean(), v[-1] - v[0],
            np.argmax(v) / (k - 1), np.argmin(v) / (k - 1)]


def series(vals, counts, min_sent=MIN_SENT_ARROW, keep=None):
    """Per-doc value trajectories. `keep` is an optional per-sentence boolean mask, applied
    before the length filter (used by the awakening-artifact arms)."""
    off = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    ser, idx = [], []
    for k in range(len(counts)):
        v = vals[off[k]:off[k] + counts[k]]
        if keep is not None:
            v = v[keep[off[k]:off[k] + counts[k]]]
        if len(v) >= min_sent:
            ser.append(np.asarray(v, float)); idx.append(k)
    return ser, np.array(idx, dtype=np.int64)


def arrow_auc(ser, groups, B: int = 2000, seed: int = 0, folds: int = 5):
    """Forward-vs-reversed AUC, grouped by `groups` (pass author keys for cluster-safe CV).

    Headline = GroupKFold mean AUC on true labels; p = per-pair label-flip permutation on a
    single grouped 80/20 split, where the split is also by group. Identical statistic to the
    published run; only the grouping variable changes (dream -> author).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.metrics import roc_auc_score
    n = len(ser)
    g = np.asarray(groups)
    if n < 25 or len(np.unique(g)) < 2:
        return float("nan"), float("nan"), 0
    F = np.array([arc_features(v) for v in ser])
    Fr = np.array([arc_features(np.asarray(v)[::-1]) for v in ser])
    X = np.vstack([F, Fr]); y = np.r_[np.ones(n), np.zeros(n)]
    gg = np.r_[g, g]
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    k = int(min(folds, len(np.unique(g))))
    auc_cv = float(cross_val_score(LogisticRegression(max_iter=400), X, y, groups=gg,
                                   cv=GroupKFold(k), scoring="roc_auc").mean())
    # group-wise 80/20 split so no author appears in both train and test
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    tr_g = set(ug[rng.random(len(ug)) < 0.8].tolist())
    tr_d = np.array([x in tr_g for x in g])
    tr = np.r_[tr_d, tr_d]; te = ~tr
    if te.sum() < 10 or tr.sum() < 10 or len(np.unique(y[te])) < 2:
        return auc_cv, float("nan"), len(ug)

    def split_auc(yy):
        lr = LogisticRegression(max_iter=400).fit(X[tr], yy[tr])
        return roc_auc_score(yy[te], lr.predict_proba(X[te])[:, 1])

    if B <= 0:                      # AUC only (used for the dream-grouped reproduction check)
        return auc_cv, float("nan"), len(ug)
    obs = split_auc(y)
    null = np.empty(B)
    for i in range(B):
        z = rng.integers(0, 2, n)
        null[i] = split_auc(np.r_[z, 1 - z].astype(float))
    p = float((1 + (null >= obs).sum()) / (B + 1))
    return auc_cv, p, len(ug)


def drift_stats(ser, groups, B: int = 4000, seed: int = 0):
    """End-minus-start drift with DREAM-level and AUTHOR-CLUSTERED inference.

    - `drift`      : dream-level mean (the published estimate)
    - `drift_ci`   : cluster bootstrap over authors (resample authors with replacement)
    - `drift_auth` : mean of per-author means (equal weight per author) + its cluster CI
    - `p_cluster`  : sign-flip permutation applied at the AUTHOR level (all of an author's
                     dreams flip together), the correct exchangeability unit here
    """
    d = np.array([v[-1] - v[0] for v in ser], float)
    g = np.asarray(groups)
    ug, inv = np.unique(g, return_inverse=True)
    nA = len(ug)
    per_auth = np.array([d[inv == i].mean() for i in range(nA)])
    rng = np.random.default_rng(seed)
    # cluster bootstrap: resample authors, pool their dreams
    idx_by_auth = [np.where(inv == i)[0] for i in range(nA)]
    boot_d, boot_a = np.empty(B), np.empty(B)
    for b in range(B):
        pick = rng.integers(0, nA, nA)
        rows = np.concatenate([idx_by_auth[i] for i in pick])
        boot_d[b] = d[rows].mean()
        boot_a[b] = per_auth[pick].mean()
    # author-level sign-flip permutation
    obs = float(per_auth.mean())
    null = np.array([float((per_auth * rng.choice([-1, 1], nA)).mean()) for _ in range(B)])
    p_cluster = float((1 + (np.abs(null) >= abs(obs)).sum()) / (B + 1))
    return {"n_dreams": len(d), "n_authors": int(nA),
            "drift": float(d.mean()),
            "drift_lo": float(np.percentile(boot_d, 2.5)),
            "drift_hi": float(np.percentile(boot_d, 97.5)),
            "drift_auth": obs,
            "drift_auth_lo": float(np.percentile(boot_a, 2.5)),
            "drift_auth_hi": float(np.percentile(boot_a, 97.5)),
            "p_cluster": p_cluster,
            "frac_authors_neg": float((per_auth < 0).mean())}


def dreamseer_true_xlmr(lang: str, min_sent: int = MIN_SENT_ARROW):
    """Records carrying TRUE per-sentence XLM-R valence for the stratified Dreamseer subset.

    The published proxy correlates only r~.27 with true per-sentence XLM-R, so the arrow's *sign*
    ultimately rests on this instrument. `sent_sentiment_text_ms5` scores a language-stratified,
    length-capped subset (<=3,000 docs/language, 5..25 sentences); everything else is NaN and
    excluded here. Returns list of {sents, vals, author}.
    """
    from .sentence_cache import _raw_field, load_or_build_sentences, offsets
    from .sentence_sentiment import load_or_build_sentence_sentiment
    docs, counts, sent = load_or_build_sentence_sentiment("text", min_sent=5)
    _d, _c, _e, _nw, meta = load_or_build_sentences("text")
    fmap = _raw_field(docs, "text")
    off = offsets(counts)
    langs = meta.lang.to_numpy(); users = meta.userID.to_numpy()
    recs = []
    for k in np.where((langs == lang) & (counts >= min_sent))[0]:
        v = sent[off[k]:off[k] + counts[k]]
        if not np.isfinite(v).all():
            continue
        ss = split_sentences(fmap.get(docs[k], ""), min_words=2)
        if len(ss) != int(counts[k]):
            raise RuntimeError(f"[true-xlmr] sentence replay misaligned at doc {k}")
        recs.append({"sents": ss, "vals": v.astype(float), "author": users[k]})
    return recs


def proxy_records(cp: Corpus, vals: np.ndarray, min_sent: int = MIN_SENT_ARROW):
    """Uniform record view of a cached corpus under a per-sentence value vector."""
    off = cp.offsets
    recs = []
    for k in range(len(cp.counts)):
        if cp.counts[k] < min_sent:
            continue
        recs.append({"sents": cp.sents[k],
                     "vals": np.asarray(vals[off[k]:off[k] + cp.counts[k]], float),
                     "author": cp.authors[k], "threat": float(cp.threat[k])})
    return recs


def one_per_author(ser, groups, R: int = 200, seed: int = 0):
    """Mean drift when exactly ONE randomly chosen dream per author is kept, averaged over R
    draws. The most conservative reading of the corpus: n = number of authors."""
    d = np.array([v[-1] - v[0] for v in ser], float)
    g = np.asarray(groups)
    ug, inv = np.unique(g, return_inverse=True)
    idx_by_auth = [np.where(inv == i)[0] for i in range(len(ug))]
    rng = np.random.default_rng(seed)
    means = np.empty(R)
    for r in range(R):
        rows = np.array([a[rng.integers(0, len(a))] for a in idx_by_auth])
        means[r] = d[rows].mean()
    return {"drift_1pa": float(means.mean()),
            "drift_1pa_lo": float(np.percentile(means, 2.5)),
            "drift_1pa_hi": float(np.percentile(means, 97.5)),
            "n_authors": int(len(ug))}
