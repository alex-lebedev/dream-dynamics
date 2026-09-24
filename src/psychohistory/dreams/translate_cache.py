"""Translate non-English DreamSeer dreams -> English (MarianMT, CPU) and cache, resumably.

Enables pooling all languages in ONE coherent measurement space and EN<->non-EN
cross-verification (the crown lesson: the scorer is affect-miscalibrated across languages, so
translate-then-measure is the coherent option). Greedy MarianMT on CPU (MPS lacks the int64
cumsum used by generate()).

Output: 10-data/interim/ds_ru2en.csv  (documentID, text_en). Translated dream text is DERIVED
from raw dream text -> S3-sensitive: it lives in interim/ (GITIGNORED) and is NEVER committed
(docs/ETHICS.md). Resumable: skips documentIDs already present, appends in flushes.

    PYTHONPATH=src PYTHONNOUSERSITE=1 python3 -m psychohistory.dreams.translate_cache
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys

sys.path[:] = [p for p in sys.path if "/.local/lib/" not in p]

import pandas as pd  # noqa: E402

from .. import config as C  # noqa: E402

OUT = C.INTERIM / "ds_ru2en.csv"
MODEL = "Helsinki-NLP/opus-mt-ru-en"


def load_targets(start: str = "2024-03-01", min_chars: int = 20, lang: str = "ru") -> pd.DataFrame:
    lv = pd.read_parquet(C.DREAMS_OUT / "dreamseer_dream_level.parquet",
                         columns=["documentID", "date", "lang", "textlen"])
    lv["date"] = pd.to_datetime(lv["date"])
    lv = lv[(lv.date >= start) & (lv.textlen >= min_chars) & (lv.lang == lang)]
    raw = pd.read_csv(C.DREAMSEER_RAW, dtype=str, keep_default_na=False,
                      usecols=["documentID", "text"])
    m = dict(zip(raw.documentID, raw.text))
    lv["text"] = lv.documentID.map(m)
    lv = lv[lv.text.str.len() >= min_chars]
    return lv[["documentID", "text"]].reset_index(drop=True)


def main(batch: int = 32, maxlen: int = 220, flush_every: int = 64):
    from transformers import pipeline
    C.INTERIM.mkdir(parents=True, exist_ok=True)
    tgt = load_targets()
    done = set(pd.read_csv(OUT, dtype=str).documentID) if OUT.exists() else set()
    todo = tgt[~tgt.documentID.isin(done)].reset_index(drop=True)
    print(f"[translate] {len(tgt)} RU dreams; {len(done)} cached; {len(todo)} to do", flush=True)
    if len(todo) == 0:
        print("[translate] nothing to do ->", OUT, flush=True)
        return
    tp = pipeline("translation", model=MODEL, device=-1, truncation=True, max_length=256)
    rows = []
    texts, ids = todo.text.tolist(), todo.documentID.tolist()
    for i in range(0, len(todo), batch):
        chunk = [t[:maxlen] for t in texts[i:i + batch]]
        outs = tp(chunk, num_beams=1, max_length=200, no_repeat_ngram_size=3)
        rows.extend({"documentID": d, "text_en": o["translation_text"]}
                    for d, o in zip(ids[i:i + batch], outs))
        if len(rows) >= flush_every or i + batch >= len(todo):
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            rows = []
            print(f"[translate] {min(i + batch, len(todo))}/{len(todo)} flushed", flush=True)
    print("[translate] done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
