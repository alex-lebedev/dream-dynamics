"""Geometry estimators for embedded corpora: intrinsic dimension and percolation.

Lifted verbatim (behaviour-preserving) from the scripts that produced the published numbers, so
that any new corpus can be pushed through the *identical* pipeline:

  two_nn_dim      : analyses/2026-07-22-02-figures-publication.py  (TwoNN, Facco et al. 2017)
  percolation     : analyses/2026-07-19-19-collective-dynamics.py  (cosine-threshold graph)

Collected here because the geometry claims now need to be compared against corpora other than
Dreamseer (ordinary human language, other dream platforms), and a re-implementation would make
those comparisons uninterpretable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

THETAS = np.round(np.arange(0.02, 0.90, 0.03), 2)


def two_nn_dim(X, sample: int = 8000, seed: int = 0) -> float:
    """TwoNN intrinsic dimension from the ratio of 2nd- to 1st-nearest-neighbour distances."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    if len(X) > sample:
        X = X[rng.choice(len(X), sample, replace=False)]
    tree = cKDTree(X)
    dd, _ = tree.query(X, k=3)
    r1, r2 = dd[:, 1], dd[:, 2]
    ok = (r1 > 0)
    mu = np.sort(r2[ok] / r1[ok])
    Femp = np.arange(1, len(mu) + 1) / len(mu)
    m = Femp < 0.9
    d = np.polyfit(np.log(mu[m]), -np.log(1 - Femp[m]), 1)[0]
    return float(d)


def percolation_once(X, thetas=THETAS) -> pd.DataFrame:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    S = X @ X.T
    np.fill_diagonal(S, 0.0)
    N = len(X)
    rows = []
    for th in thetas:
        ncomp, labels = connected_components(csr_matrix(S > th), directed=False)
        sizes = np.bincount(labels)
        rows.append({"theta": float(th), "giant_frac": float(sizes.max() / N),
                     "n_components": int(ncomp), "mean_degree": float((S > th).sum() / N),
                     "n_singletons": int((sizes == 1).sum())})
    return pd.DataFrame(rows)


def percolation(emb, n: int = 5000, seeds=(0, 1, 2), shuffled: bool = False,
                thetas=THETAS) -> pd.DataFrame:
    """Multi-seed mean percolation curve over `n` sampled rows (optionally feature-shuffled)."""
    parts = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        idx = rng.choice(len(emb), min(n, len(emb)), replace=False)
        X = np.asarray(emb[idx], dtype=np.float32)
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        if shuffled:
            X = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        parts.append(percolation_once(X, thetas).set_index("theta"))
    return (sum(parts) / len(parts)).reset_index()


def feature_shuffle(X, seed: int = 1):
    """Variance-preserving null: permute each embedding coordinate independently."""
    rng = np.random.default_rng(seed)
    S = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
    return S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-9)


def gaussian_ref(n, d, seed: int = 1):
    """Isotropic-Gaussian reference on the unit sphere."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, d))
    return G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)


def participation_ratio(X) -> float:
    """Effective number of variance directions: (sum l)^2 / sum l^2 over PCA eigenvalues."""
    Xc = np.asarray(X, float) - np.asarray(X, float).mean(0)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum())
