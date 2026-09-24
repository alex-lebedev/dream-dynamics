"""Maps of the collective dream-space: theme/symbol co-occurrence networks, an embedding
map with clusters, and dynamic (over-time) theme prevalence.

Inputs: DreamSeer LLM-extracted `themes` (abstract) + `detected_symbolism` (concrete),
merged with the consistent model sentiment. Exports GraphML/JSON (for Gephi/Cytoscape/web)
+ node tables + figures. All outputs are aggregate (no dream text).
"""
from __future__ import annotations
import json
import re
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from ..config import RAW, DREAMS_OUT

_QUOTED = re.compile(r'"([^"]+)"')


def parse_list(x: str):
    if not isinstance(x, str) or not x.strip():
        return []
    items = _QUOTED.findall(x)
    if not items:
        items = [p for p in x.split(",")]
    return [i.strip().lower() for i in items if i.strip()]


def load_dream_tags() -> pd.DataFrame:
    """Per-dream: date, lang, model sentiment, themes[list], symbols[list]."""
    raw = pd.read_csv(RAW / "dreamseer" / "dreamseer_data.csv", dtype=str, keep_default_na=False,
                      usecols=["documentID", "themes", "detected_symbolism", "X", "Y", "Z"])
    sent = pd.read_parquet(DREAMS_OUT / "dreamseer_sentiment.parquet")  # documentID,date,lang,sentiment
    df = sent.merge(raw, on="documentID", how="left")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df.date.notna()].copy()
    for c in ["X", "Y", "Z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["themes"] = df["themes"].fillna("").map(parse_list)
    df["symbols"] = df["detected_symbolism"].fillna("").map(parse_list)
    return df


def node_stats(df, field, top_n):
    """Frequency, mean sentiment, and monthly-prevalence trend slope per top node
    (vectorised via explode)."""
    freq = Counter(t for lst in df[field] for t in lst)
    top = [t for t, _ in freq.most_common(top_n)]
    tset = set(top)
    ex = df[["date", "sentiment", field]].explode(field)
    ex = ex[ex[field].isin(tset)].copy()
    ex["month"] = ex.date.dt.to_period("M")
    sent = ex.groupby(field)["sentiment"].mean().to_dict()
    permonth = df.assign(month=df.date.dt.to_period("M")).groupby("month").size()
    good = set(permonth[permonth >= 20].index)
    cnt = ex[ex.month.isin(good)].groupby(["month", field]).size().rename("c").reset_index()
    cnt["prev"] = cnt["c"] / cnt["month"].map(permonth)
    trend = {}
    for t, sub in cnt.groupby(field):
        sub = sub.sort_values("month")
        trend[t] = float(np.polyfit(np.arange(len(sub)), sub["prev"], 1)[0]) if len(sub) >= 6 else 0.0
    return top, freq, sent, {t: trend.get(t, 0.0) for t in top}


def build_network(df, field, top_n=120, min_cooccur=8):
    import networkx as nx
    top, freq, sent, trend = node_stats(df, field, top_n)
    tset = set(top)
    edges = Counter()
    for lst in df[field]:
        items = sorted({t for t in lst if t in tset})
        for a, b in combinations(items, 2):
            edges[(a, b)] += 1
    G = nx.Graph()
    for t in top:
        s = sent[t]
        G.add_node(t, freq=int(freq[t]), sentiment=float(0 if s != s else s), trend=float(trend[t]))
    for (a, b), w in edges.items():
        if w >= min_cooccur:
            G.add_edge(a, b, weight=int(w))
    from networkx.algorithms.community import greedy_modularity_communities
    try:
        comms = list(greedy_modularity_communities(G, weight="weight"))
    except Exception:
        comms = []
    for i, c in enumerate(comms):
        for n in c:
            G.nodes[n]["community"] = int(i)
    for n in G.nodes:
        G.nodes[n].setdefault("community", -1)
    return G, comms


def export_network(G, name, outdir):
    import networkx as nx
    outdir.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(outdir / f"{name}.graphml"))
    (outdir / f"{name}.json").write_text(json.dumps(nx.node_link_data(G)))
    rows = [{"node": n, **G.nodes[n], "degree": int(G.degree(n))} for n in G.nodes]
    pd.DataFrame(rows).sort_values("freq", ascending=False).to_csv(outdir / f"{name}_nodes.csv", index=False)


def build_dreamspace(df, n_clusters=16, seed=0):
    """2-D map (uses the app's X,Y embedding) + KMeans clusters on X,Y,Z."""
    from sklearn.cluster import KMeans
    d = df.dropna(subset=["X", "Y", "Z"]).copy()
    for c in ["X", "Y", "Z"]:
        d[c] = d[c].astype(float)
    d["cluster"] = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(d[["X", "Y", "Z"]])
    return d


def label_clusters(d, field="themes"):
    rows = []
    for c, g in d.groupby("cluster"):
        toks = Counter(t for lst in g[field] for t in lst)
        syms = Counter(t for lst in g["symbols"] for t in lst)
        rows.append({"cluster": int(c), "size": len(g), "sentiment": round(g.sentiment.mean(), 3),
                     "en_share": round((g.lang == "en").mean(), 2),
                     "top_themes": ", ".join(w for w, _ in toks.most_common(8)),
                     "top_symbols": ", ".join(w for w, _ in syms.most_common(6))})
    return pd.DataFrame(rows).sort_values("size", ascending=False)
