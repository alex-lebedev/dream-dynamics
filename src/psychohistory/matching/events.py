"""Load the pre-built event cohorts into a unified, analysis-ready events table.

Only the `*_core` tables are used (the shipped `*_control_matches` were matched on the
celestial spine and are NOT reused for dream-outcome studies — see docs/METHODS.md §3A).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from ..config import EVENTS_DIR

# family -> core parquet filename
CORES = {
    "earthquake": "historical_events_core.parquet",
    "aviation": "historical_events_aviation_core.parquet",
    "terror": "historical_events_terror_attacks_core.parquet",
    "coup": "historical_events_political_shocks_core.parquet",
    "conflict_onset": "historical_events_conflict_onsets_core.parquet",
}

_KEEP = ["event_id", "event_family", "event_subtype", "place_name",
         "country_iso3166_1_alpha3", "latitude", "longitude", "magnitude",
         "magnitude_type"]


def load_events(families=None, start=None, end=None, min_magnitude=None,
                countries=None) -> pd.DataFrame:
    """Return a unified events frame: date, family, subtype, country_iso3, severity, ...

    Args:
        families: subset of CORES keys (default all present).
        start/end: ISO date bounds (inclusive) — restrict to the analysis window.
        min_magnitude: severity screen (nkill for terror, Richter for quakes, etc.).
        countries: iso3 list to keep (crude salience/relevance screen).
    """
    families = families or list(CORES)
    frames = []
    for fam in families:
        p = Path(EVENTS_DIR) / CORES[fam]
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        cols = [c for c in _KEEP if c in df.columns]
        sub = df[cols].copy()
        sub["date"] = pd.to_datetime(df["calendar_date_start"], errors="coerce")
        if "terror_nkill" in df.columns:
            sub["nkill"] = pd.to_numeric(df["terror_nkill"], errors="coerce")
        sub["family"] = fam
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    ev = pd.concat(frames, ignore_index=True)
    ev = ev[ev["date"].notna()]
    if start:
        ev = ev[ev["date"] >= pd.Timestamp(start)]
    if end:
        ev = ev[ev["date"] <= pd.Timestamp(end)]
    if min_magnitude is not None:
        ev = ev[ev["magnitude"].fillna(0) >= min_magnitude]
    if countries:
        ev = ev[ev["country_iso3166_1_alpha3"].isin(countries)]
    return ev.sort_values("date").reset_index(drop=True)


def summary(ev: pd.DataFrame) -> pd.DataFrame:
    if ev.empty:
        return ev
    return (ev.assign(year=ev.date.dt.year)
              .groupby(["family", "year"]).size().rename("n").reset_index())
