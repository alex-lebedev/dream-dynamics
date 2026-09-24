"""Person-adjusted feature-network estimators used by the descriptive network gallery.

Everything here operates on de-identified vectors in memory and returns aggregate
matrices or scalars. Contributor identifiers never leave the caller's process.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .. import config as C


@dataclass(frozen=True)
class StatePanel:
    """Dream-level rows with within-person (leave-one-out) feature deviations."""

    frame: pd.DataFrame
    state: np.ndarray
    level: np.ndarray
    feature_names: tuple[str, ...]


def load_state_panel(
    *,
    lang: str = "en",
    start: str = "2024-03-01",
    end: str = "2026-07-10",
    min_user_reports: int = 3,
    features: list[str] | None = None,
) -> StatePanel:
    """Load dream-level features and subtract each contributor's leave-one-out mean.

    The leave-one-out baseline keeps the focal dream out of its own comparator, so a
    prolific contributor cannot shrink their own deviations.
    """
    names = tuple(features or (C.EMOTIONS + C.TAGS))
    frame = pd.read_parquet(
        C.DREAMS_OUT / "dreamseer_dream_level.parquet",
        columns=["documentID", "userID", "date", "lang", "words", "nightmare_index", *names],
    )
    frame = frame[
        frame["lang"].eq(lang)
        & frame["date"].ge(pd.Timestamp(start))
        & frame["date"].lt(pd.Timestamp(end))
    ].dropna(subset=list(names)).reset_index(drop=True)

    codes, _ = pd.factorize(frame["userID"], sort=False)
    counts = np.bincount(codes).astype(float)
    keep = counts[codes] >= min_user_reports
    frame = frame.loc[keep].reset_index(drop=True)

    codes, _ = pd.factorize(frame["userID"], sort=False)
    level = frame[list(names)].to_numpy(float)
    totals = np.zeros((codes.max() + 1, level.shape[1]))
    np.add.at(totals, codes, level)
    counts = np.bincount(codes).astype(float)[:, None]
    baseline = (totals[codes] - level) / (counts[codes] - 1.0)
    # `person` is an opaque within-run integer, kept so contributor counts can be checked
    # against the disclosure floor without carrying an identifier.
    return StatePanel(
        frame=frame.drop(columns=["userID"]).assign(person=codes),
        state=level - baseline,
        level=level,
        feature_names=names,
    )


def spearman_matrix(x: np.ndarray) -> np.ndarray:
    """Rank-correlation matrix with a unit diagonal and no NaNs."""
    if len(x) < 5:
        return np.full((x.shape[1], x.shape[1]), np.nan)
    result = stats.spearmanr(x, axis=0)
    # SciPy renamed this result field from `correlation` to `statistic`.
    corr = np.asarray(result.statistic if hasattr(result, "statistic") else result.correlation,
                      float)
    corr = np.atleast_2d(corr)
    np.fill_diagonal(corr, 1.0)
    return np.nan_to_num(corr)


def network_stats(corr: np.ndarray) -> dict[str, float]:
    """Threshold-free integration statistics of a correlation network."""
    iu = np.triu_indices_from(corr, k=1)
    eigenvalues = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    total = eigenvalues.sum()
    return {
        "mean_abs_rho": float(np.abs(corr[iu]).mean()),
        "lambda1_share": float(eigenvalues.max() / total) if total > 0 else np.nan,
        "participation_ratio": float(total**2 / np.square(eigenvalues).sum())
        if total > 0
        else np.nan,
    }


def top_edges(corr: np.ndarray, density: float = 0.12) -> list[tuple[int, int]]:
    """Fixed-density edge set selected on absolute correlation."""
    iu = np.triu_indices_from(corr, k=1)
    score = np.abs(corr[iu])
    keep = max(1, int(round(density * len(score))))
    chosen = np.argpartition(score, -keep)[-keep:]
    order = chosen[np.argsort(score[chosen])]
    return [(int(iu[0][j]), int(iu[1][j])) for j in order]


def window_masks(
    dates: pd.Series, event_dates: list[pd.Timestamp], windows: dict[str, tuple[int, int]]
) -> dict[str, np.ndarray]:
    """Event-locked boolean masks pooled over every event in a cohort."""
    day = dates.dt.normalize().to_numpy("datetime64[D]")
    out = {}
    for name, (lo, hi) in windows.items():
        mask = np.zeros(len(day), bool)
        for event in event_dates:
            anchor = np.datetime64(pd.Timestamp(event).date(), "D")
            mask |= (day >= anchor + lo) & (day <= anchor + hi)
        out[name] = mask
    return out


def placebo_anchors(
    eligible: np.ndarray,
    real: list[pd.Timestamp],
    n_events: int,
    rng: np.random.Generator,
    *,
    exclusion_days: int = 30,
) -> list[pd.Timestamp]:
    """Draw pseudo-event dates far from every real shock, from the observed window."""
    real_days = np.array([np.datetime64(pd.Timestamp(d).date(), "D") for d in real])
    gaps = np.abs(eligible[:, None] - real_days[None, :]).astype("timedelta64[D]").astype(int)
    pool = eligible[(gaps >= exclusion_days).all(axis=1)]
    picks = rng.choice(pool, size=min(n_events, len(pool)), replace=False)
    return [pd.Timestamp(p) for p in picks]
