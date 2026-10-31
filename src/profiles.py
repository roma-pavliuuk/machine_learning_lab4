from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def profile_matrix(X_raw: pd.DataFrame, labels: np.ndarray, numeric_cols: List[str],
                   categorical_cols: List[str]) -> pd.DataFrame:
    num = X_raw[numeric_cols].astype(float)
    cat = pd.get_dummies(X_raw[categorical_cols].astype("string").fillna("Missing"), prefix_sep="=").astype(float)
    rows = []
    med, q1, q3 = num.median(), num.quantile(0.25), num.quantile(0.75)
    iqr = q3 - q1
    for k in np.unique(labels):
        mask = labels == k
        for col in numeric_cols:
            if iqr[col] > 0:
                rows.append({"cluster": k, "feature": col, "kind": "z",
                             "value": float((num.loc[mask, col].median() - med[col]) / iqr[col]),
                             "cluster_stat": float(num.loc[mask, col].median()), "global_stat": float(med[col])})
            else:
                nz_c, nz_g = (num.loc[mask, col] != 0).mean(), (num[col] != 0).mean()
                rows.append({"cluster": k, "feature": col, "kind": "share_nonzero",
                             "value": float(nz_c - nz_g), "cluster_stat": float(nz_c), "global_stat": float(nz_g)})
        for col in cat.columns:
            sh_c, sh_g = cat.loc[mask, col].mean(), cat[col].mean()
            rows.append({"cluster": k, "feature": col, "kind": "share",
                         "value": float(sh_c - sh_g), "cluster_stat": float(sh_c), "global_stat": float(sh_g)})
        for col in numeric_cols:
            if num[col].isna().any():
                m_c, m_g = num.loc[mask, col].isna().mean(), num[col].isna().mean()
                rows.append({"cluster": k, "feature": f"{col} (пропуск)", "kind": "share_missing",
                             "value": float(m_c - m_g), "cluster_stat": float(m_c), "global_stat": float(m_g)})
    return pd.DataFrame(rows)


def top_features(profiles: pd.DataFrame, k: int, top: int = 5) -> pd.DataFrame:
    sub = profiles[profiles["cluster"] == k].copy()
    sub["abs"] = sub["value"].abs()
    return sub.sort_values("abs", ascending=False).head(top).drop(columns="abs")


def risk_sets(log_density: np.ndarray, entropy: np.ndarray, share: float) -> Dict[str, set]:
    k = int(round(share * len(log_density)))
    return {"low_density": set(np.argsort(log_density)[:k].tolist()),
            "high_entropy": set(np.argsort(-entropy)[:k].tolist())}
