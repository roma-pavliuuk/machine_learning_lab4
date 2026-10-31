from __future__ import annotations

from typing import Dict, Set

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity


def scott_bandwidth(X: np.ndarray) -> float:
    n, d = X.shape
    return float(n ** (-1.0 / (d + 4)) * X.std(axis=0).mean())


def select_bandwidth(X: np.ndarray, factors=(0.2, 5.0), points: int = 25, folds: int = 5,
                     random_state: int = 42) -> GridSearchCV:
    h0 = scott_bandwidth(X)
    grid = np.logspace(np.log10(factors[0] * h0), np.log10(factors[1] * h0), points)
    search = GridSearchCV(KernelDensity(kernel="gaussian"), {"bandwidth": grid},
                          cv=KFold(folds, shuffle=True, random_state=random_state))
    search.fit(X)
    search.h_scott_ = h0
    return search


def lowest_share(scores: np.ndarray, share: float) -> Set[int]:
    k = int(round(share * len(scores)))
    return set(np.argsort(scores)[:k].tolist())


def isolation_scores(X: np.ndarray, contamination: float, random_state: int = 42) -> np.ndarray:
    return IsolationForest(contamination=contamination, random_state=random_state).fit(X).score_samples(X)


def jaccard(a: Set[int], b: Set[int]) -> float:
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def jaccard_table(sets: Dict[str, Set[int]]) -> pd.DataFrame:
    names = list(sets)
    m = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            m.loc[a, b] = jaccard(sets[a], sets[b])
    return m
