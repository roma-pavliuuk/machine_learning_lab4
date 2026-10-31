from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.preprocessor import DynamicPipelineBuilder


@dataclass
class FeatureSpaces:
    prep: object
    scaler: StandardScaler
    pca_wide: PCA
    pca_kde: PCA
    feature_names: np.ndarray

    @classmethod
    def fit(cls, config_path, X_ref_raw: pd.DataFrame, cfg: Dict) -> "FeatureSpaces":
        sp = cfg["spaces"]
        prep = DynamicPipelineBuilder(str(config_path)).build()
        Xp = prep.fit_transform(X_ref_raw)
        scaler = StandardScaler().fit(Xp)
        Xs = scaler.transform(Xp)
        pca_wide = PCA(n_components=sp["wide_components"], random_state=sp["random_state"]).fit(Xs)
        pca_kde = PCA(n_components=sp["kde_components"], random_state=sp["random_state"]).fit(Xs)
        return cls(prep, scaler, pca_wide, pca_kde, np.asarray(prep.get_feature_names_out()))

    def transform(self, X_raw: pd.DataFrame) -> Dict[str, np.ndarray]:
        Xs = self.scaler.transform(self.prep.transform(X_raw))
        return {"scaled": Xs, "wide": self.pca_wide.transform(Xs), "kde": self.pca_kde.transform(Xs)}

    def variance_table(self) -> pd.DataFrame:
        rows = []
        for name, pca in (("wide", self.pca_wide), ("kde", self.pca_kde)):
            rows.append({"простір": name, "розмірність": pca.n_components_,
                         "пояснена дисперсія": float(pca.explained_variance_ratio_.sum()),
                         "вхідна розмірність": pca.n_features_in_})
        return pd.DataFrame(rows)
