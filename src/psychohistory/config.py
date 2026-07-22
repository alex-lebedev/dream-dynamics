"""Canonical paths and feature-column definitions (single source of truth)."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/psychohistory/config.py -> repo root

DATA = ROOT / "10-data"
RAW = DATA / "raw"
EXTERNAL = DATA / "external"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MANIFESTS = DATA / "manifests"

DREAMS_OUT = PROCESSED / "dreams"
EVENTS_DIR = PROCESSED / "events"
CELESTIAL_DIR = PROCESSED / "celestial"
SIGNALS_DIR = PROCESSED / "signals"

RESULTS = ROOT / "60-results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
WIKI = ROOT / "50-wiki"

DREAMSEER_RAW = RAW / "dreamseer" / "dreamseer_data.csv"
DREAMBANK_RAW = RAW / "dreambank" / "dreambank_data.csv"

# --- DreamSeer feature columns (per-dream, 0-1) ---
EMOTIONS = ["fear", "anger", "sadness", "disgust", "joy", "trust", "anticipation", "surprise"]
# valence-negative / valence-positive split (surprise & anticipation are ambiguous -> excluded from valence index)
NEG_EMOTIONS = ["fear", "anger", "sadness", "disgust"]
POS_EMOTIONS = ["joy", "trust"]
TAGS = [
    "danger", "weapon", "family", "flying", "animal", "health", "swimming",
    "place_house", "place_nature", "indoors", "outdoors", "romantic", "social",
    "friend", "stranger", "situation_childhood", "game", "music", "food",
    "object_inanimate", "object_animate",
]
DERIVED = ["negativity", "nightmare_index"]
LANGS = ["en", "ru", "other"]

APP_START = "2023-01-01"  # DreamSeer data before this is spurious (26 rows)
