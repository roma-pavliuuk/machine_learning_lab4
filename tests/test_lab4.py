import json
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from src.clustering import bic_curves, best_by_bic, n_parameters, null_data, responsibilities_entropy
from src.data import CONFIG, ROOT, reference_id
from src.density import jaccard, scott_bandwidth, select_bandwidth
from src.profiles import profile_matrix, risk_sets
from src.spaces import FeatureSpaces

NOTEBOOK = ROOT / "notebooks" / "lab4_unsupervised.ipynb"


def test_reference_md5_matches_pointer(cfg):
    ref = reference_id(cfg)
    assert ref["md5"] == cfg["reference"]["expected_md5"]


def test_target_sealed_and_not_a_feature(cfg, reference):
    X_raw, y = reference
    assert cfg["target"] not in X_raw.columns and y.name == cfg["target"]
    features = {c for cols in cfg["features"].values() for c in cols}
    assert cfg["target"] not in features
    assert not (features & set(cfg["protected_attributes"]))


def test_no_target_dependent_steps_in_policy(cfg):
    policy = json.dumps(cfg["transform_policy"])
    assert "woe" not in policy and "target" not in policy


def test_spaces_fit_only_on_reference(cfg, reference):
    X_raw, _ = reference
    fs = FeatureSpaces.fit(CONFIG, X_raw, cfg)
    mean_before, comp_before = fs.scaler.mean_.copy(), fs.pca_wide.components_.copy()
    other = X_raw.sample(frac=1.0, random_state=1).reset_index(drop=True)
    other["credit_amount"] = other["credit_amount"] * 10
    fs.transform(other)
    assert np.array_equal(mean_before, fs.scaler.mean_) and np.array_equal(comp_before, fs.pca_wide.components_)
    assert fs.pca_kde.n_components_ <= 5


def test_n_parameters_formula():
    assert n_parameters(3, 2, "spherical") == 3 * 2 + 2 + 3
    assert n_parameters(3, 2, "diag") == 3 * 2 + 2 + 6
    assert n_parameters(3, 2, "full") == 3 * 2 + 2 + 9


def test_control_experiment_blobs():
    X, _ = make_blobs(n_samples=1000, centers=3, random_state=42)
    assert best_by_bic(bic_curves(X, (1, 8), ["full"])) == (3, "full")


def test_null_data_keeps_marginals():
    X = np.random.default_rng(0).normal(size=(200, 3)) * [1, 5, 10]
    Xn = null_data(X)
    assert np.allclose(np.sort(X, axis=0), np.sort(Xn, axis=0))
    assert not np.allclose(X, Xn)


def test_bandwidth_search_uses_cv_grid():
    X = np.random.default_rng(0).normal(size=(300, 3))
    search = select_bandwidth(X, factors=(0.2, 5.0), points=10, folds=5)
    grid = search.cv_results_["param_bandwidth"].astype(float)
    assert grid.min() == pytest.approx(0.2 * search.h_scott_) and grid.max() == pytest.approx(5 * search.h_scott_)
    assert search.h_scott_ == pytest.approx(scott_bandwidth(X))


def test_entropy_bounds():
    g = np.array([[1.0, 0.0], [0.5, 0.5]])
    H = responsibilities_entropy(g)
    assert H[0] == 0 and H[1] == pytest.approx(np.log(2))


def test_profiles_z_formula_and_binary_rule():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 100.0, 101.0], "b": [0, 0, 0, 1, 1, 1], "c": list("xxyyzz")})
    labels = np.array([0, 0, 0, 1, 1, 1])
    prof = profile_matrix(X, labels, ["a", "b"], ["c"])
    row = prof[(prof.cluster == 1) & (prof.feature == "a")].iloc[0]
    med, iqr = X["a"].median(), X["a"].quantile(0.75) - X["a"].quantile(0.25)
    assert row.value == pytest.approx((X.loc[labels == 1, "a"].median() - med) / iqr)
    assert prof[(prof.feature == "c=z") & (prof.cluster == 1)].iloc[0].value == pytest.approx(2 / 3 - 2 / 6)


def test_risk_sets_and_jaccard():
    R = risk_sets(np.array([5.0, 1.0, 3.0, 4.0]), np.array([0.1, 0.0, 0.9, 0.2]), 0.25)
    assert R == {"low_density": {1}, "high_entropy": {2}} and jaccard({1}, {2}) == 0


def test_notebook_opens_envelope_last():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]
    envelope = next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and "Запечатаний конверт" in "".join(c["source"]) and "## 5" in "".join(c["source"]))
    for i, c in enumerate(cells[:envelope]):
        src = "".join(c["source"])
        if c["cell_type"] == "code":
            uses = re.findall(r"Y_sealed(?!\s*=)", src)
            assert not uses or "load_reference" in src, f"клітинка {i} використовує Y до конверта"
    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    assert "SEED = 42" in code and "random_state=0" not in code.replace("random_state=s", "").replace("random_state=b", "")
