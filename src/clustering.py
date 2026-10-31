from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture


def n_parameters(c: int, d: int, covariance_type: str) -> int:
    cov = {"spherical": c, "diag": c * d, "full": c * d * (d + 1) // 2, "tied": d * (d + 1) // 2}[covariance_type]
    return c * d + (c - 1) + cov


def bic_curves(X: np.ndarray, c_range: Tuple[int, int], cov_types: List[str], random_state: int = 42) -> pd.DataFrame:
    rows = []
    for cov in cov_types:
        for c in range(c_range[0], c_range[1] + 1):
            gmm = GaussianMixture(n_components=c, covariance_type=cov, random_state=random_state).fit(X)
            rows.append({"covariance": cov, "C": c, "BIC": gmm.bic(X), "AIC": gmm.aic(X),
                         "k": n_parameters(c, X.shape[1], cov), "converged": gmm.converged_})
    return pd.DataFrame(rows)


def best_by_bic(curves: pd.DataFrame) -> Tuple[int, str]:
    row = curves.loc[curves["BIC"].idxmin()]
    return int(row["C"]), str(row["covariance"])


def ari_matrix(X: np.ndarray, c: int, cov: str, n_init: int, seeds: int = 10) -> np.ndarray:
    labels = [GaussianMixture(c, covariance_type=cov, n_init=n_init, random_state=s).fit_predict(X)
              for s in range(seeds)]
    m = np.eye(seeds)
    for i in range(seeds):
        for j in range(i + 1, seeds):
            m[i, j] = m[j, i] = adjusted_rand_score(labels[i], labels[j])
    return m


def mean_offdiag(m: np.ndarray) -> float:
    return float(m[np.triu_indices_from(m, k=1)].mean())


def null_data(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    return np.column_stack([rng.permutation(col) for col in X.T])


def fit_final(X: np.ndarray, c: int, cov: str, random_state: int = 42) -> GaussianMixture:
    return GaussianMixture(c, covariance_type=cov, n_init=10, random_state=random_state).fit(X)


def responsibilities_entropy(gamma: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(gamma > 0, gamma * np.log(gamma), 0.0)
    return -terms.sum(axis=1)


def bootstrap_stability(X: np.ndarray, reference_labels: np.ndarray, c: int, cov: str,
                        n_resamples: int = 20, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    for b in range(n_resamples):
        idx = np.unique(rng.integers(0, len(X), len(X)))
        labels = GaussianMixture(c, covariance_type=cov, n_init=5, random_state=b).fit_predict(X[idx])
        rows.append({"resample": b, "n_unique": len(idx), "ARI_vs_final": adjusted_rand_score(reference_labels[idx], labels)})
    return pd.DataFrame(rows)
