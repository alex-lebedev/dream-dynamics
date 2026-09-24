"""Download NON-DREAM natural-language baseline corpora for the geometry controls.

The intrinsic-dimension and percolation results compare dream embeddings against synthetic
nulls only (an isotropic Gaussian and a variance-preserving feature shuffle). Those nulls show
that the geometry is not distance concentration, but they cannot show that it is a property of
*dreams*: modern sentence encoders impose their own low-dimensional, anisotropic structure on
any coherent prose. The missing control is ordinary human language put through the identical
pipeline.

This fetches three human registers, chosen to bracket dream reports:

  personal-narrative : r/confession posts — first-person accounts of real personal events;
                       the closest waking analogue to a dream report in person, tense and
                       register (HF `SocialGrep/one-million-reddit-confessions`, CC-BY-4.0).
  fiction            : Project Gutenberg paragraphs — deliberate narrative prose, capped per
                       book so no single author dominates (public domain).
  encyclopedic       : Wikipedia article openings — coherent but non-narrative, third person;
                       an anchor for how much of the geometry is "narrative" at all (CC-BY-SA).

Writes plain CSVs to `10-data/external/language_baselines/` (gitignored). Data cards go to
`10-data/manifests/`. Nothing here touches dream data.

    cd psychohistory && python3 scripts/fetch_language_baselines.py
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("10-data/external/language_baselines")
UA = {"User-Agent": "psychohistory-research/1.0 (academic geometry control; contact via repo)"}
TARGET = 12000          # rows per corpus (>= the 5,000 percolation + 8,000 TwoNN samples)
MIN_CHARS = 300         # dreams are truncated at 256 chars; require the baseline to fill it too


def _get(url, timeout=60, retries=4):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if a == retries - 1:
                raise
            print(f"    retry {a+1} ({type(e).__name__}) {e}", flush=True)
            time.sleep(2 + 3 * a)


_BAD = re.compile(r"^\s*(\[removed\]|\[deleted\])\s*$", re.I)


def fetch_personal_narrative():
    """r/confession posts: first-person accounts of real events.

    Pulled from the dataset's converted parquet shard rather than the paged rows API, which
    rate-limits (HTTP 429) long before 12,000 usable rows.
    """
    import pandas as pd

    ds = "SocialGrep/one-million-reddit-confessions"
    u = "https://datasets-server.huggingface.co/parquet?dataset=" + urllib.parse.quote(ds)
    shards = sorted(json.loads(_get(u))["parquet_files"], key=lambda s: s["size"])
    shard = shards[0]
    local = OUT / "_confessions.parquet"
    if not local.exists():
        print(f"[personal] downloading {shard['filename']} ({shard['size']/1e6:.0f}MB)", flush=True)
        local.write_bytes(_get(shard["url"], timeout=900))
    d = pd.read_parquet(local, columns=["selftext", "id", "subreddit.name"])
    print(f"[personal] shard rows={len(d):,}", flush=True)
    rng = random.Random(0)
    order = list(range(len(d)))
    rng.shuffle(order)
    seen, out = set(), []
    st = d["selftext"].to_numpy(); ids = d["id"].to_numpy(); sub = d["subreddit.name"].to_numpy()
    for i in order:
        if len(out) >= TARGET:
            break
        t = st[i].strip() if isinstance(st[i], str) else ""
        if len(t) < MIN_CHARS or _BAD.match(t):
            continue
        key = t[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append({"text": t, "author": str(ids[i]), "source": str(sub[i])})
    print(f"[personal] kept {len(out):,}", flush=True)
    return out


def fetch_fiction():
    """Project Gutenberg paragraphs, capped per book so no author dominates."""
    out, page, books = [], 1, []
    while len(books) < 90 and page <= 6:
        d = json.loads(_get(f"https://gutendex.com/books?languages=en&topic=fiction&page={page}"))
        books += [b for b in d.get("results", []) if b.get("formats")]
        page += 1
    print(f"[fiction] {len(books)} candidate books", flush=True)
    per_book = max(40, TARGET // max(1, len(books)) + 40)
    for b in books:
        if len(out) >= TARGET:
            break
        url = (b["formats"].get("text/plain; charset=utf-8")
               or b["formats"].get("text/plain")
               or b["formats"].get("text/plain; charset=us-ascii"))
        if not url:
            continue
        try:
            raw = _get(url, timeout=90).decode("utf-8", "ignore")
        except Exception as e:
            print(f"    book {b['id']} failed: {e}", flush=True); continue
        # strip Gutenberg front/back matter
        m = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw, re.S | re.I)
        if m:
            raw = raw[m.end():]
        m = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", raw, re.S | re.I)
        if m:
            raw = raw[:m.start()]
        paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", raw)]
        keep = [p for p in paras if len(p) >= MIN_CHARS and p.count(" ") > 30]
        for p in keep[:per_book]:
            out.append({"text": p, "author": f"pg{b['id']}",
                        "source": (b.get("title") or "")[:80]})
        print(f"    [fiction] book {b['id']} +{len(keep[:per_book])} -> {len(out):,}", flush=True)
    return out[:TARGET]


def fetch_encyclopedic():
    """Wikipedia lead sections.

    Read from the 2023-11-01 English snapshot's converted parquet, not the live
    `generator=random` API: that endpoint returns HTTP 429 within a few hundred articles, and a
    frozen snapshot is in any case the more reproducible choice.
    """
    import pandas as pd

    ds, cfg = "wikimedia/wikipedia", "20231101.en"
    u = "https://datasets-server.huggingface.co/parquet?dataset=" + urllib.parse.quote(ds)
    fs = [f for f in json.loads(_get(u))["parquet_files"] if f["config"] == cfg]
    shard = sorted(fs, key=lambda s: s["size"])[0]
    local = OUT / "_wikipedia.parquet"
    if not local.exists():
        print(f"[encyclopedic] downloading {shard['filename']} ({shard['size']/1e6:.0f}MB)",
              flush=True)
        local.write_bytes(_get(shard["url"], timeout=1800))
    d = pd.read_parquet(local, columns=["id", "title", "text"])
    print(f"[encyclopedic] shard rows={len(d):,}", flush=True)
    rng = random.Random(0)
    order = list(range(len(d)))
    rng.shuffle(order)
    ids, ti, tx = d["id"].to_numpy(), d["title"].to_numpy(), d["text"].to_numpy()
    out = []
    for i in order:
        if len(out) >= TARGET:
            break
        body = tx[i] if isinstance(tx[i], str) else ""
        lead = []                                   # accumulate lead paragraphs, stop at a header
        for p in re.split(r"\n\s*\n", body):
            p = re.sub(r"\s+", " ", p).strip()
            if not p or p.endswith(("=", ":")) or p.count(" ") < 8:
                continue
            lead.append(p)
            if sum(len(x) for x in lead) >= 400 or len(lead) >= 3:
                break
        t = " ".join(lead)
        if len(t) < MIN_CHARS:
            continue
        out.append({"text": t[:4000], "author": f"wp{ids[i]}", "source": str(ti[i])[:80]})
        if len(out) % 2000 == 0:
            print(f"    [encyclopedic] {len(out):,}/{TARGET}", flush=True)
    return out[:TARGET]


CORPORA = {"personal-narrative": fetch_personal_narrative,
           "fiction": fetch_fiction,
           "encyclopedic": fetch_encyclopedic}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    want = sys.argv[1:] or list(CORPORA)
    for name in want:
        f = OUT / f"{name}.csv"
        if f.exists():
            print(f"[{name}] exists ({sum(1 for _ in f.open())-1:,} rows) -> skip", flush=True)
            continue
        print(f"\n=== fetching {name} ===", flush=True)
        rows = CORPORA[name]()
        with f.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["text", "author", "source"])
            wr.writeheader(); wr.writerows(rows)
        print(f"[{name}] wrote {len(rows):,} rows -> {f}", flush=True)


if __name__ == "__main__":
    main()
