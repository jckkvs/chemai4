"""
tests/test_linear_tree_extra.py

linear_tree.py のカバレッジ改善テスト。
LinearTreeRegressor, LinearTreeClassifier,
LinearForestRegressor, LinearForestClassifier,
LinearBoostRegressor, LinearBoostClassifier を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression, Lasso

from backend.models.linear_tree import (
    LinearTreeRegressor,
    LinearTreeClassifier,
    LinearForestRegressor,
    LinearForestClassifier,
    LinearBoostRegressor,
    LinearBoostClassifier,
    _Node,
    _to_numpy,
    _fit_linear,
    _predict_linear,
    _mse,
    _gini,
)


# ============================================================
# ユーティリティ関数
# ============================================================

class TestUtilities:
    def test_to_numpy_array(self):
        X = np.array([[1, 2], [3, 4]])
        result = _to_numpy(X)
        assert isinstance(result, np.ndarray)

    def test_to_numpy_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _to_numpy(df)
        assert isinstance(result, np.ndarray)
        assert result.dtype in (np.float64, np.float32)

    def test_fit_linear_success(self):
        X = np.random.randn(20, 3)
        y = X[:, 0] * 2 + 1
        model = _fit_linear(Ridge(), X, y)
        assert model is not None

    def test_fit_linear_empty(self):
        X = np.zeros((0, 3))
        y = np.zeros(0)
        model = _fit_linear(Ridge(), X, y)
        assert model is None

    def test_predict_linear(self):
        X = np.random.randn(20, 3)
        y = X[:, 0] * 2
        m = _fit_linear(Ridge(), X, y)
        preds = _predict_linear(m, X)
        assert len(preds) == 20

    def test_mse(self):
        y = np.array([1.0, 2.0, 3.0])
        result = _mse(y)
        assert result > 0

    def test_mse_empty(self):
        assert _mse(np.array([])) == 0.0

    def test_gini(self):
        y = np.array([0, 0, 1, 1])
        result = _gini(y, 2)
        assert 0 <= result <= 1

    def test_gini_pure(self):
        y = np.array([1, 1, 1])
        result = _gini(y, 2)
        assert result < 0.01


class TestNode:
    def test_is_leaf(self):
        n = _Node()
        assert n.is_leaf is True

    def test_is_not_leaf(self):
        n = _Node(left=_Node(), right=_Node())
        assert n.is_leaf is False


# ============================================================
# LinearTreeRegressor
# ============================================================

class TestLinearTreeRegressor:
    def test_fit_predict(self):
        rng = np.random.RandomState(42)
        X = rng.randn(60, 3)
        y = X[:, 0] * 3 + X[:, 1] * 2 + rng.randn(60) * 0.1
        model = LinearTreeRegressor(max_depth=3, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 60

    def test_with_dataframe(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.randn(40, 3), columns=["a", "b", "c"])
        y = df["a"] * 2
        model = LinearTreeRegressor(max_depth=2)
        model.fit(df, y)
        preds = model.predict(df)
        assert len(preds) == 40

    def test_custom_estimator(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        y = X[:, 0] * 2
        model = LinearTreeRegressor(base_estimator=Lasso(alpha=0.1), max_depth=2)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 40

    def test_n_leaves(self):
        rng = np.random.RandomState(42)
        X = rng.randn(60, 3)
        y = X[:, 0]
        model = LinearTreeRegressor(max_depth=2)
        model.fit(X, y)
        assert model.n_leaves_ >= 1

    def test_max_features_sqrt(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 10)
        y = X[:, 0]
        model = LinearTreeRegressor(max_depth=2, max_features="sqrt")
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 40


# ============================================================
# LinearTreeClassifier
# ============================================================

class TestLinearTreeClassifier:
    def test_fit_predict(self):
        rng = np.random.RandomState(42)
        X = rng.randn(60, 3)
        y = (X[:, 0] > 0).astype(int)
        model = LinearTreeClassifier(max_depth=2, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 60
        assert set(preds).issubset({0, 1})

    def test_predict_proba(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        y = (X[:, 0] > 0).astype(int)
        model = LinearTreeClassifier(max_depth=2, random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (40, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=0.1)

    def test_multiclass(self):
        rng = np.random.RandomState(42)
        X = rng.randn(90, 3)
        y = np.repeat([0, 1, 2], 30)
        model = LinearTreeClassifier(max_depth=2, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 90


# ============================================================
# LinearForestRegressor
# ============================================================

class TestLinearForestRegressor:
    def test_fit_predict(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        y = X[:, 0] * 2 + rng.randn(40) * 0.1
        model = LinearForestRegressor(n_estimators=5, max_depth=2, n_jobs=1)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 40

    def test_no_bootstrap(self):
        rng = np.random.RandomState(42)
        X = rng.randn(30, 3)
        y = X[:, 0]
        model = LinearForestRegressor(n_estimators=3, bootstrap=False, n_jobs=1)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 30


# ============================================================
# LinearForestClassifier
# ============================================================

class TestLinearForestClassifier:
    def test_fit_predict(self):
        rng = np.random.RandomState(42)
        X = rng.randn(60, 3)
        y = (X[:, 0] > 0).astype(int)
        model = LinearForestClassifier(n_estimators=5, max_depth=2, n_jobs=1)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 60

    def test_predict_proba(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        y = (X[:, 0] > 0).astype(int)
        model = LinearForestClassifier(n_estimators=3, max_depth=2, n_jobs=1)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape[0] == 40
        assert proba.shape[1] == 2


# ============================================================
# LinearBoostRegressor
# ============================================================

class TestLinearBoostRegressor:
    def test_fit_predict(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        y = X[:, 0] * 3 + rng.randn(40) * 0.1
        model = LinearBoostRegressor(n_estimators=5, max_depth=2, learning_rate=0.1)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 40

    def test_subsample(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        y = X[:, 0]
        model = LinearBoostRegressor(n_estimators=3, subsample=0.8)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 40


# ============================================================
# LinearBoostClassifier
# ============================================================

class TestLinearBoostClassifier:
    def test_fit_predict_binary(self):
        rng = np.random.RandomState(42)
        X = rng.randn(60, 3)
        y = (X[:, 0] > 0).astype(int)
        model = LinearBoostClassifier(n_estimators=5, max_depth=2, learning_rate=0.1)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 60

    def test_predict_proba_binary(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        y = (X[:, 0] > 0).astype(int)
        model = LinearBoostClassifier(n_estimators=3, max_depth=2)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (40, 2)
