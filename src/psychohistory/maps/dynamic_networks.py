"""Dynamic, person-adjusted network estimators for dream populations.

The estimators in this module operate on de-identified vectors in memory. They
return aggregate graph statistics only; user identifiers and node-level edges
must not be written to release paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DynamicPanel:
    """One row per contributor-period with past-only state vectors."""

    frame: pd.DataFrame
    semantic_state: np.ndarray
    semantic_level: np.ndarray
    semantic_raw_level: np.ndarray
    feature_state: np.ndarray
    feature_level: np.ndarray
    feature_names: tuple[str, ...]


def _unit_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(
        x,
        norm,
        out=np.full_like(x, np.nan, dtype=float),
        where=np.isfinite(norm) & (norm > 1e-12),
    )


def _past_only_states(
    frame: pd.DataFrame,
    semantic_level: np.ndarray,
    feature_level: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Difference each contributor-period from their previous eligible period."""
    semantic_state = np.full_like(semantic_level, np.nan, dtype=float)
    feature_state = np.full_like(feature_level, np.nan, dtype=float)
    if len(frame) > 1:
        user = frame["user"].to_numpy()
        quarter = frame["quarter"].to_numpy()
        if np.any(user[1:] < user[:-1]) or np.any(
            (user[1:] == user[:-1]) & (quarter[1:] < quarter[:-1])
        ):
            raise ValueError("panel rows must remain sorted by user and quarter")
        current = np.flatnonzero(user[1:] == user[:-1]) + 1
        previous = current - 1
        semantic_state[current] = semantic_level[current] - semantic_level[previous]
        feature_state[current] = feature_level[current] - feature_level[previous]
    return _unit_rows(semantic_state), feature_state


def build_quarter_panel(
    meta: pd.DataFrame,
    embeddings: np.ndarray,
    features: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
    *,
    language: str = "en",
    start: str = "2024-04-01",
    end: str = "2026-07-01",
    min_period_reports: int = 3,
    min_baseline_reports: int = 3,
    min_period_users: int = 20,
) -> DynamicPanel:
    """Build contributor-quarter levels and past-only period-to-period shifts.

    State at t is the focal quarter centroid minus the previous eligible quarter
    centroid. This avoids future leakage and the linear dependence induced by
    career-wide leave-period-out centering. ``min_baseline_reports`` is retained
    for API compatibility and must not exceed the focal-period threshold.
    """
    if len(meta) != len(embeddings) or len(meta) != len(features):
        raise ValueError("meta, embeddings and features must be row-aligned")
    if features.shape[1] != len(feature_names):
        raise ValueError("feature_names must match feature columns")
    if min_baseline_reports > min_period_reports:
        raise ValueError("past-period baseline uses the same report threshold as focal periods")

    date = pd.to_datetime(meta["date"])
    mask = (
        meta["lang"].eq(language).to_numpy()
        & date.ge(start).to_numpy()
        & date.lt(end).to_numpy()
    )
    pos = np.flatnonzero(mask)
    user_codes, user_labels = pd.factorize(meta.iloc[pos]["userID"], sort=True)
    first_date = (
        pd.DataFrame(
            {
                "user_id": meta.loc[meta["lang"].eq(language), "userID"],
                "date": date.loc[meta["lang"].eq(language)],
            }
        )
        .dropna()
        .groupby("user_id", sort=False)["date"]
        .min()
    )
    first_period_number = {
        code: int(first_date.loc[label].year * 4 + first_date.loc[label].quarter)
        for code, label in enumerate(user_labels)
    }
    work = pd.DataFrame(
        {
            "user": user_codes,
            "quarter": date.iloc[pos].dt.to_period("Q").astype(str).to_numpy(),
            "row": pos,
        }
    )
    grouped = work.groupby(["user", "quarter"], sort=True)["row"].agg(list)

    records: list[dict] = []
    sem_level: list[np.ndarray] = []
    feat_level: list[np.ndarray] = []
    for (user, quarter), rows in grouped.items():
        if len(rows) < min_period_reports:
            continue
        rows_arr = np.asarray(rows, dtype=int)

        q_sem = np.asarray(embeddings[rows_arr], float).mean(axis=0)
        q_feat = np.nanmean(np.asarray(features[rows_arr], float), axis=0)
        if not np.isfinite(q_feat).all():
            continue

        records.append(
            {
                "user": int(user),
                "quarter": quarter,
                "tenure_quarters": int(
                    int(quarter[:4]) * 4 + int(quarter[-1]) - first_period_number[user]
                ),
                "n_period": int(len(rows_arr)),
            }
        )
        sem_level.append(q_sem)
        feat_level.append(q_feat)

    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("no contributor-period cells met the eligibility rules")
    period_counts = frame["quarter"].value_counts()
    keep_periods = period_counts[period_counts >= min_period_users].index
    keep = frame["quarter"].isin(keep_periods).to_numpy()
    frame = frame.loc[keep].reset_index(drop=True)
    if frame.empty:
        raise ValueError("no periods met the minimum contributor count")
    semantic_raw_level = np.vstack(sem_level)[keep]
    feature_level = np.vstack(feat_level)[keep]
    semantic_state, feature_state = _past_only_states(
        frame, semantic_raw_level, feature_level
    )
    return DynamicPanel(
        frame=frame,
        semantic_state=semantic_state,
        semantic_level=_unit_rows(semantic_raw_level),
        semantic_raw_level=semantic_raw_level,
        feature_state=feature_state,
        feature_level=feature_level,
        feature_names=tuple(feature_names),
    )


def subset_panel(panel: DynamicPanel, mask: np.ndarray | pd.Series) -> DynamicPanel:
    """Row-subset a panel while preserving alignment of every vector block."""
    keep = np.asarray(mask, bool)
    if len(keep) != len(panel.frame):
        raise ValueError("mask length must match panel rows")
    frame = panel.frame.loc[keep].reset_index(drop=True)
    semantic_raw_level = panel.semantic_raw_level[keep]
    feature_level = panel.feature_level[keep]
    semantic_state, feature_state = _past_only_states(
        frame, semantic_raw_level, feature_level
    )
    return DynamicPanel(
        frame=frame,
        semantic_state=semantic_state,
        semantic_level=_unit_rows(semantic_raw_level),
        semantic_raw_level=semantic_raw_level,
        feature_state=feature_state,
        feature_level=feature_level,
        feature_names=panel.feature_names,
    )


def shuffle_states_within_user(panel: DynamicPanel, rng: np.random.Generator) -> DynamicPanel:
    """Shuffle complete period blocks within user and recompute past-only states."""
    order = np.arange(len(panel.frame))
    source = order.copy()
    for idx in panel.frame.groupby("user", sort=False).indices.values():
        idx = np.asarray(idx, dtype=int)
        source[idx] = rng.permutation(idx)
    frame = panel.frame.copy()
    frame["n_period"] = panel.frame.loc[source, "n_period"].to_numpy()
    semantic_raw_level = panel.semantic_raw_level[source]
    feature_level = panel.feature_level[source]
    semantic_state, feature_state = _past_only_states(
        frame, semantic_raw_level, feature_level
    )
    return DynamicPanel(
        frame=frame,
        semantic_state=semantic_state,
        semantic_level=_unit_rows(semantic_raw_level),
        semantic_raw_level=semantic_raw_level,
        feature_state=feature_state,
        feature_level=feature_level,
        feature_names=panel.feature_names,
    )


def directional_order_by_period(panel: DynamicPanel) -> pd.DataFrame:
    """Kuramoto-like alignment of person-adjusted semantic movement."""
    rows = []
    for quarter, idx in panel.frame.groupby("quarter", sort=True).indices.items():
        x = panel.semantic_state[np.asarray(idx)]
        x = x[np.isfinite(x).all(axis=1)]
        if len(x) < 2:
            continue
        resultant = float(np.linalg.norm(x.mean(axis=0)))
        n = len(x)
        mean_pair = float((n * resultant**2 - 1.0) / (n - 1)) if n > 1 else np.nan
        rows.append(
            {
                "quarter": quarter,
                "n_users": n,
                "resultant_length": resultant,
                "mean_pair_cosine": mean_pair,
            }
        )
    return pd.DataFrame(rows)


def _neighbor_sets(x: np.ndarray, users: np.ndarray, k: int) -> dict[int, set[int]]:
    similarity = x @ x.T
    np.fill_diagonal(similarity, -np.inf)
    k_eff = min(k, len(x) - 1)
    if k_eff < 1:
        return {int(u): set() for u in users}
    nn = np.argpartition(similarity, -k_eff, axis=1)[:, -k_eff:]
    return {int(users[i]): set(map(int, users[nn[i]])) for i in range(len(users))}


def neighbor_overlap_by_lag(
    panel: DynamicPanel, *, k: int = 5, state: bool = False, lag: int = 1
) -> pd.DataFrame:
    """Top-k relational-neighborhood overlap at a fixed quarter lag."""
    vectors = panel.semantic_state if state else panel.semantic_level
    quarters = sorted(panel.frame["quarter"].unique())
    rows = []
    for q0, q1 in zip(quarters[:-lag], quarters[lag:]):
        a = panel.frame.index[panel.frame["quarter"].eq(q0)].to_numpy()
        b = panel.frame.index[panel.frame["quarter"].eq(q1)].to_numpy()
        common = np.intersect1d(
            panel.frame.loc[a, "user"].to_numpy(),
            panel.frame.loc[b, "user"].to_numpy(),
        )
        map0 = panel.frame.loc[a].reset_index().set_index("user")["index"]
        map1 = panel.frame.loc[b].reset_index().set_index("user")["index"]
        idx0 = map0.loc[common].to_numpy()
        idx1 = map1.loc[common].to_numpy()
        valid = np.isfinite(vectors[idx0]).all(axis=1) & np.isfinite(
            vectors[idx1]
        ).all(axis=1)
        common, idx0, idx1 = common[valid], idx0[valid], idx1[valid]
        if len(common) <= k + 1:
            continue
        n0 = _neighbor_sets(vectors[idx0], common, k)
        n1 = _neighbor_sets(vectors[idx1], common, k)
        overlap = np.mean([len(n0[int(u)] & n1[int(u)]) / min(k, len(common) - 1) for u in common])
        edge0 = {(int(u), int(v)) for u in common for v in n0[int(u)]}
        edge1 = {(int(u), int(v)) for u in common for v in n1[int(u)]}
        union = edge0 | edge1
        rows.append(
            {
                "from_quarter": q0,
                "to_quarter": q1,
                "lag": lag,
                "n_common_users": int(len(common)),
                "neighbor_overlap": float(overlap),
                "chance_overlap": float(k / (len(common) - 1)),
                "edge_jaccard": float(len(edge0 & edge1) / len(union)) if union else np.nan,
            }
        )
    return pd.DataFrame(rows)


def adjacent_neighbor_overlap(
    panel: DynamicPanel, *, k: int = 5, state: bool = False
) -> pd.DataFrame:
    """Backward-compatible lag-1 neighbor overlap."""
    return neighbor_overlap_by_lag(panel, k=k, state=state, lag=1)


def identity_shuffle_neighbor_null(
    panel: DynamicPanel,
    *,
    n_perm: int = 500,
    k: int = 5,
    seed: int = 20260822,
) -> tuple[float, np.ndarray]:
    """Test trait-neighbor persistence against independently relabeled identities."""
    observed_frame = neighbor_overlap_by_lag(panel, k=k, state=False, lag=1)
    observed = _weighted_mean(
        observed_frame, "neighbor_overlap", "n_common_users"
    )
    quarters = sorted(panel.frame["quarter"].unique())
    rng = np.random.default_rng(seed)
    null = np.full(n_perm, np.nan)
    for b in range(n_perm):
        values = []
        weights = []
        for q0, q1 in zip(quarters[:-1], quarters[1:]):
            a = panel.frame.index[panel.frame["quarter"].eq(q0)].to_numpy()
            c = panel.frame.index[panel.frame["quarter"].eq(q1)].to_numpy()
            common = np.intersect1d(
                panel.frame.loc[a, "user"].to_numpy(),
                panel.frame.loc[c, "user"].to_numpy(),
            )
            if len(common) <= k + 1:
                continue
            map0 = panel.frame.loc[a].reset_index().set_index("user")["index"]
            map1 = panel.frame.loc[c].reset_index().set_index("user")["index"]
            idx0 = map0.loc[common].to_numpy()
            idx1 = map1.loc[common].to_numpy()
            n0 = _neighbor_sets(panel.semantic_level[idx0], common, k)
            relabeled = rng.permutation(common)
            n1 = _neighbor_sets(panel.semantic_level[idx1], relabeled, k)
            overlap = np.mean(
                [len(n0[int(u)] & n1[int(u)]) / k for u in common]
            )
            values.append(overlap)
            weights.append(len(common))
        null[b] = np.average(values, weights=weights)
    return observed, null


def _rank_corr_matrix(x: np.ndarray) -> np.ndarray:
    corr = stats.spearmanr(x, axis=0).statistic
    corr = np.asarray(corr, float)
    np.fill_diagonal(corr, 1.0)
    return np.nan_to_num(corr)


def _top_edge_set(corr: np.ndarray, density: float) -> set[tuple[int, int]]:
    iu = np.triu_indices_from(corr, k=1)
    score = np.abs(corr[iu])
    keep = max(1, int(round(density * len(score))))
    chosen = np.argpartition(score, -keep)[-keep:]
    return {(int(iu[0][j]), int(iu[1][j])) for j in chosen}


def theme_networks_by_period(panel: DynamicPanel, *, density: float = 0.15) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fixed-density networks of co-moving, person-adjusted feature axes."""
    summaries = []
    matrices: dict[str, np.ndarray] = {}
    for quarter, idx in panel.frame.groupby("quarter", sort=True).indices.items():
        x = panel.feature_state[np.asarray(idx)]
        x = x[np.isfinite(x).all(axis=1)]
        if len(x) < 3:
            continue
        corr = _rank_corr_matrix(x)
        matrices[quarter] = corr
        edges = _top_edge_set(corr, density)
        graph = nx.Graph()
        graph.add_nodes_from(range(corr.shape[0]))
        graph.add_weighted_edges_from((i, j, abs(float(corr[i, j]))) for i, j in edges)
        communities = nx.community.louvain_communities(graph, weight="weight", seed=0)
        summaries.append(
            {
                "quarter": quarter,
                "n_users": len(x),
                "n_edges": len(edges),
                "modularity": float(nx.community.modularity(graph, communities, weight="weight")),
                "n_communities": len(communities),
            }
        )
    return pd.DataFrame(summaries), matrices


def adjacent_theme_similarity(
    matrices: dict[str, np.ndarray],
    *,
    density: float = 0.15,
    period_sizes: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Threshold-free matrix persistence plus fixed-density edge persistence."""
    quarters = sorted(matrices)
    rows = []
    for q0, q1 in zip(quarters[:-1], quarters[1:]):
        a, b = matrices[q0], matrices[q1]
        iu = np.triu_indices_from(a, k=1)
        rho = stats.spearmanr(a[iu], b[iu]).statistic
        e0, e1 = _top_edge_set(a, density), _top_edge_set(b, density)
        rows.append(
            {
                "from_quarter": q0,
                "to_quarter": q1,
                "n_users": (
                    min(period_sizes[q0], period_sizes[q1])
                    if period_sizes is not None
                    else 1
                ),
                "matrix_spearman": float(rho),
                "edge_jaccard": float(len(e0 & e1) / len(e0 | e1)),
            }
        )
    return pd.DataFrame(rows)


def bh_fdr(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values in input order."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _weighted_mean(frame: pd.DataFrame, value: str, weight: str) -> float:
    valid = frame[value].notna() & frame[weight].gt(0)
    return float(np.average(frame.loc[valid, value], weights=frame.loc[valid, weight]))


def _order_mean(frame: pd.DataFrame) -> float:
    weighted = frame.assign(pair_weight=frame["n_users"] * (frame["n_users"] - 1) / 2)
    return _weighted_mean(weighted, "mean_pair_cosine", "pair_weight")


def permutation_summary(
    panel: DynamicPanel,
    *,
    n_perm: int = 500,
    k: int = 5,
    density: float = 0.15,
    seed: int = 20260822,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate predeclared dynamic-network statistics against chronology shuffles."""
    observed_order = directional_order_by_period(panel)
    observed_trait = adjacent_neighbor_overlap(panel, k=k, state=False)
    observed_state = adjacent_neighbor_overlap(panel, k=k, state=True)
    observed_theme_summary, observed_theme_matrices = theme_networks_by_period(
        panel, density=density
    )
    period_sizes = observed_theme_summary.set_index("quarter")["n_users"].to_dict()
    observed_theme = adjacent_theme_similarity(
        observed_theme_matrices, density=density, period_sizes=period_sizes
    )
    observed_trait_lags = {
        lag: neighbor_overlap_by_lag(panel, k=k, state=False, lag=lag)
        for lag in (1, 2, 4)
    }

    observed = {
        "semantic_order": _order_mean(observed_order),
        "trait_temporal_persistence": _weighted_mean(
            observed_trait, "neighbor_overlap", "n_common_users"
        ),
        "state_neighbor_persistence": _weighted_mean(
            observed_state, "neighbor_overlap", "n_common_users"
        ),
        "theme_matrix_persistence": _weighted_mean(
            observed_theme, "matrix_spearman", "n_users"
        ),
        "theme_edge_persistence": _weighted_mean(
            observed_theme, "edge_jaccard", "n_users"
        ),
    }
    null = {name: np.full(n_perm, np.nan) for name in observed}
    trait_lag_null = {lag: np.full(n_perm, np.nan) for lag in (1, 2, 4)}
    order_period_null = {
        q: np.full(n_perm, np.nan) for q in observed_order["quarter"].astype(str)
    }
    trait_period_null = {
        f"{row.from_quarter}->{row.to_quarter}": np.full(n_perm, np.nan)
        for row in observed_trait.itertuples(index=False)
    }
    state_period_null = {
        f"{row.from_quarter}->{row.to_quarter}": np.full(n_perm, np.nan)
        for row in observed_state.itertuples(index=False)
    }
    theme_matrix_period_null = {
        f"{row.from_quarter}->{row.to_quarter}": np.full(n_perm, np.nan)
        for row in observed_theme.itertuples(index=False)
    }
    theme_edge_period_null = {
        key: np.full(n_perm, np.nan) for key in theme_matrix_period_null
    }
    rng = np.random.default_rng(seed)
    for b in range(n_perm):
        shuffled = shuffle_states_within_user(panel, rng)
        order = directional_order_by_period(shuffled)
        trait = adjacent_neighbor_overlap(shuffled, k=k, state=False)
        state = adjacent_neighbor_overlap(shuffled, k=k, state=True)
        theme_summary, theme_matrices = theme_networks_by_period(shuffled, density=density)
        shuffled_sizes = theme_summary.set_index("quarter")["n_users"].to_dict()
        theme = adjacent_theme_similarity(
            theme_matrices, density=density, period_sizes=shuffled_sizes
        )
        null["semantic_order"][b] = _order_mean(order)
        for row in order.itertuples(index=False):
            order_period_null[str(row.quarter)][b] = row.mean_pair_cosine
        for row in trait.itertuples(index=False):
            trait_period_null[f"{row.from_quarter}->{row.to_quarter}"][
                b
            ] = row.neighbor_overlap
        for row in state.itertuples(index=False):
            state_period_null[f"{row.from_quarter}->{row.to_quarter}"][
                b
            ] = row.neighbor_overlap
        for row in theme.itertuples(index=False):
            key = f"{row.from_quarter}->{row.to_quarter}"
            theme_matrix_period_null[key][b] = row.matrix_spearman
            theme_edge_period_null[key][b] = row.edge_jaccard
        null["trait_temporal_persistence"][b] = _weighted_mean(
            trait, "neighbor_overlap", "n_common_users"
        )
        for lag in (1, 2, 4):
            lag_frame = neighbor_overlap_by_lag(
                shuffled, k=k, state=False, lag=lag
            )
            trait_lag_null[lag][b] = _weighted_mean(
                lag_frame, "neighbor_overlap", "n_common_users"
            )
        null["state_neighbor_persistence"][b] = _weighted_mean(
            state, "neighbor_overlap", "n_common_users"
        )
        null["theme_matrix_persistence"][b] = _weighted_mean(
            theme, "matrix_spearman", "n_users"
        )
        null["theme_edge_persistence"][b] = _weighted_mean(
            theme, "edge_jaccard", "n_users"
        )

    rows = []
    for name, value in observed.items():
        values = null[name]
        rows.append(
            {
                "statistic": name,
                "observed": value,
                "null_mean": float(np.nanmean(values)),
                "null_q025": float(np.nanquantile(values, 0.025)),
                "null_q975": float(np.nanquantile(values, 0.975)),
                "p_upper": float((1 + np.sum(values >= value)) / (len(values) + 1)),
                "p_lower": float((1 + np.sum(values <= value)) / (len(values) + 1)),
                "n_perm": n_perm,
            }
        )
    result = pd.DataFrame(rows)
    result["p_two_sided"] = np.minimum(1.0, 2 * np.minimum(result["p_upper"], result["p_lower"]))
    result["q_bh"] = bh_fdr(result["p_two_sided"].to_numpy())
    details = {
        "order": observed_order,
        "trait_overlap": observed_trait,
        "state_overlap": observed_state,
        "theme_overlap": observed_theme,
        "theme_matrices": observed_theme_matrices,
        "null": null,
        "order_period_null": order_period_null,
        "trait_period_null": trait_period_null,
        "state_period_null": state_period_null,
        "theme_matrix_period_null": theme_matrix_period_null,
        "theme_edge_period_null": theme_edge_period_null,
        "trait_lags": observed_trait_lags,
        "trait_lag_null": trait_lag_null,
    }
    return result, details


def permutation_relational_summary(
    panel: DynamicPanel,
    *,
    n_perm: int = 500,
    k: int = 5,
    seed: int = 20260822,
) -> pd.DataFrame:
    """Fast sensitivity gate for the three semantic/dreamer-network statistics."""
    observed_order = directional_order_by_period(panel)
    observed_trait = adjacent_neighbor_overlap(panel, k=k, state=False)
    observed_state = adjacent_neighbor_overlap(panel, k=k, state=True)
    observed = {
        "semantic_order": _order_mean(observed_order),
        "trait_temporal_persistence": _weighted_mean(
            observed_trait, "neighbor_overlap", "n_common_users"
        ),
        "state_neighbor_persistence": _weighted_mean(
            observed_state, "neighbor_overlap", "n_common_users"
        ),
    }
    null = {name: np.full(n_perm, np.nan) for name in observed}
    rng = np.random.default_rng(seed)
    for b in range(n_perm):
        shuffled = shuffle_states_within_user(panel, rng)
        null["semantic_order"][b] = _order_mean(directional_order_by_period(shuffled))
        trait = adjacent_neighbor_overlap(shuffled, k=k, state=False)
        state = adjacent_neighbor_overlap(shuffled, k=k, state=True)
        null["trait_temporal_persistence"][b] = _weighted_mean(
            trait, "neighbor_overlap", "n_common_users"
        )
        null["state_neighbor_persistence"][b] = _weighted_mean(
            state, "neighbor_overlap", "n_common_users"
        )

    rows = []
    for name, value in observed.items():
        values = null[name]
        p_upper = float((1 + np.sum(values >= value)) / (len(values) + 1))
        p_lower = float((1 + np.sum(values <= value)) / (len(values) + 1))
        rows.append(
            {
                "statistic": name,
                "observed": value,
                "null_mean": float(np.nanmean(values)),
                "null_q025": float(np.nanquantile(values, 0.025)),
                "null_q975": float(np.nanquantile(values, 0.975)),
                "p_upper": p_upper,
                "p_lower": p_lower,
                "n_perm": n_perm,
                "p_two_sided": min(1.0, 2 * min(p_upper, p_lower)),
            }
        )
    result = pd.DataFrame(rows)
    result["q_bh"] = bh_fdr(result["p_two_sided"].to_numpy())
    return result


def strongest_theme_edge_changes(
    matrices: dict[str, np.ndarray], feature_names: tuple[str, ...], *, top: int = 12
) -> pd.DataFrame:
    """Descriptive largest early-to-late changes after the global network test."""
    quarters = sorted(matrices)
    early = np.mean([matrices[q] for q in quarters[:3]], axis=0)
    late = np.mean([matrices[q] for q in quarters[-3:]], axis=0)
    iu = np.triu_indices_from(early, k=1)
    rows = [
        {
            "feature_a": feature_names[i],
            "feature_b": feature_names[j],
            "early_rho": float(early[i, j]),
            "late_rho": float(late[i, j]),
            "delta_rho": float(late[i, j] - early[i, j]),
        }
        for i, j in zip(*iu)
    ]
    frame = pd.DataFrame(rows)
    return frame.reindex(frame["delta_rho"].abs().sort_values(ascending=False).index).head(top)
