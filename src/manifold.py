from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, trustworthiness


def subsample(n: int, max_points: int, random_state: int = 42) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n, size=min(n, max_points), replace=False))


def projections(X: np.ndarray, n_neighbors: int = 10, perplexity: int = 30, random_state: int = 42) -> Dict[str, np.ndarray]:
    return {
        "PCA": PCA(n_components=2, random_state=random_state).fit_transform(X),
        "LLE": LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, random_state=random_state).fit_transform(X),
        "Isomap": Isomap(n_neighbors=n_neighbors, n_components=2).fit_transform(X),
        "t-SNE": TSNE(n_components=2, init="pca", perplexity=perplexity, random_state=random_state).fit_transform(X),
    }


def tsne_perplexities(X: np.ndarray, perplexities, random_state: int = 42) -> Dict[int, np.ndarray]:
    return {p: TSNE(n_components=2, init="pca", perplexity=p, random_state=random_state).fit_transform(X)
            for p in perplexities}


def trust_table(X: np.ndarray, embeddings: Dict[str, np.ndarray], n_neighbors: int = 10) -> pd.DataFrame:
    rows = [{"проєкція": name, "trustworthiness": trustworthiness(X, Z, n_neighbors=n_neighbors)}
            for name, Z in embeddings.items()]
    return pd.DataFrame(rows).sort_values("trustworthiness", ascending=False, ignore_index=True)
