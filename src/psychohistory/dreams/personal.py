"""Parse the personal dream journal (N=1) — markdown entries with YAML frontmatter — into a
tidy table. Source: 10-data/raw/personal_dreams/entries/ (copied from the scdnb second brain).
Sensitive (S3): outputs stay gitignored; only derived aggregates are shared.
"""
from __future__ import annotations
import re
from pathlib import Path

import pandas as pd
import yaml

from ..config import RAW
from ..utils.lang import detect_language

PERSONAL_DIR = RAW / "personal_dreams" / "entries"
_FM = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)
_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _narrative(body: str) -> str:
    """Text under '## Narrative (raw recall)' up to the next header; fallback to body."""
    m = re.search(r"##\s*Narrative[^\n]*\n(.*?)(?:\n##\s|\Z)", body, re.S)
    text = m.group(1) if m else body
    text = re.sub(r"^#.*$", "", text, flags=re.M)   # strip any header lines
    return text.strip()


def load_personal(directory=PERSONAL_DIR) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(directory).glob("*.md")):
        raw = p.read_text(encoding="utf-8", errors="replace")
        m = _FM.match(raw)
        fm, body = {}, raw
        if m:
            try:
                fm = yaml.safe_load(m.group(1)) or {}
            except Exception:
                fm = {}
            body = m.group(2)
        text = _narrative(body)
        if not text:
            continue
        d = fm.get("created") or (_DATE.search(p.name).group(1) if _DATE.search(p.name) else None)
        date = pd.to_datetime(str(d), errors="coerce")
        tags = fm.get("tags") or []
        symbols = fm.get("symbols") or []
        lang = "ru" if ("russian" in tags) else detect_language(text)
        rows.append({"doc_id": p.stem, "date": date, "title": fm.get("title", ""),
                     "text": text, "lang": lang, "lucid": fm.get("lucid"),
                     "mood_on_waking": fm.get("mood_on_waking"),
                     "symbols": symbols, "tags": [t for t in tags if t not in ("dream", "imported")],
                     "textlen": len(text), "words": len(text.split())})
    df = pd.DataFrame(rows)
    df = df[df.date.notna()].sort_values("date").reset_index(drop=True)
    return df
