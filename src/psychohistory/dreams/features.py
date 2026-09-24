"""DreamSeer: load -> clean -> language-tag -> per-dream features -> daily/weekly aggregates.

All logic lives here; the CLI in `build_features.py` just orchestrates and writes outputs.
Aggregates are de-identified (no userID, no text) and safe to track in git.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..config import (APP_START, DERIVED, EMOTIONS, NEG_EMOTIONS, POS_EMOTIONS, TAGS)
from ..utils.lang import detect_language

FEATURES = EMOTIONS + TAGS + DERIVED
_NUMERIC = EMOTIONS + TAGS + ["X", "Y", "Z", "rating"]


def load_dreamseer(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    for c in _NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=False)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop_duplicates(subset=["documentID"])
    df = df[df["createdAt"].notna()].copy()
    upper = pd.Timestamp.today().normalize() + pd.Timedelta(days=2)
    df = df[(df["createdAt"] >= pd.Timestamp(APP_START)) & (df["createdAt"] <= upper)]
    df["date"] = df["createdAt"].dt.normalize()
    df.attrs["dropped"] = n0 - len(df)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lang"] = df["text"].map(detect_language)
    txt = df["text"].fillna("")
    df["textlen"] = txt.str.len()
    df["words"] = txt.str.split().map(len)
    df["negativity"] = df[NEG_EMOTIONS].mean(axis=1) - df[POS_EMOTIONS].mean(axis=1)
    df["nightmare_index"] = df[["fear", "danger", "sadness"]].mean(axis=1) - df["joy"]
    return df


def _agg(group: pd.DataFrame) -> pd.Series:
    out = {}
    for c in FEATURES + ["textlen", "words"]:
        out[f"{c}_mean"] = group[c].mean()
    out["n_dreams"] = len(group)
    out["n_users"] = group["userID"].nunique()
    return pd.Series(out)


def aggregate_daily(df: pd.DataFrame, by_lang: bool = True) -> pd.DataFrame:
    frames = []
    allg = df.groupby("date", group_keys=False).apply(_agg).reset_index()
    allg["lang"] = "all"
    frames.append(allg)
    if by_lang:
        for lang, sub in df.groupby("lang"):
            if lang == "unknown":
                continue
            fg = sub.groupby("date", group_keys=False).apply(_agg).reset_index()
            fg["lang"] = lang
            frames.append(fg)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    cols = ["lang", "date", "n_dreams", "n_users"] + [c for c in out.columns if c.endswith("_mean")]
    return out[cols].sort_values(["lang", "date"]).reset_index(drop=True)


def aggregate_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["week"] = d["date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)
    mean_cols = [c for c in d.columns if c.endswith("_mean")]

    def wavg(g: pd.DataFrame) -> pd.Series:
        w = g["n_dreams"].to_numpy()
        out = {c: np.average(g[c], weights=w) for c in mean_cols}
        out["n_dreams"] = int(g["n_dreams"].sum())
        out["n_userdays"] = int(g["n_users"].sum())  # NB: user-days, not distinct users
        return pd.Series(out)

    out = d.groupby(["lang", "week"], group_keys=False).apply(wavg).reset_index()
    return out.sort_values(["lang", "week"]).reset_index(drop=True)
