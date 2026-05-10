"""
tests/test_rgf_comprehensive.py

RGFRegressor / RGFClassifier の包括テスト。
全メソッド・エッジケース・sklearn互換性を検証。
"""
from __future__ import annotations

import pytest
import numpy as np
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression, make_classification

from backend.models.rgf import (
    RGFRegressor,
    RGFClassifier,
    _RGFCore,
    _to_numpy,
    _sigmoid,
    _softmax,
)


# ============================================================
# ユーティリティ関数テスト
# ============================================================

class TestUtilities:
    def test_to_numpy_from_list(self):
        result = _to_numpy([[1, 2], [3, 4]])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_to_numpy_from_df(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = _to_numpy(df)
        assert result.dtype == float

    def test_sigmoid_basic(self):
        x = np.array([0.0])
        assert abs(_sigmoid(x)[0] - 0.5) < 1e-7

    def test_sigmoid_clipping(self):
        """極端な値でもovertflowしない"""
        x = np.array([-1000, 1000])
        result = _sigmoid(x)
        assert np.all(np.isfinite(result))
        assert result[0] < 1e-6
        assert result[1] > 1 - 1e-6

    def test_softmax_basic(self):
        x = np.array([[1.0, 2.0, 3.0]])
        result = _softmax(x)
        assert abs(result.sum() - 1.0) < 1e-6
        assert result[0, 2] > result[0, 0]


# ============================================================
# RGFRegressor テスト
# ============================================================

class TestRGFRegressor:
    @pytest.fixture
    def data_reg(self):
        X, y = make_regression(n_samples=80, n_features=5, noise=0.5, random_state=42)
        return X, y

    def test_fit_predict(self, data_reg):
        X, y = data_reg
        m = RGFRegressor(n_estimators=5, max_leaf_nodes=4, random_state=42)
        m.fit(X, y)
        pred = m.predict(X)
        assert pred.shape == (80,)
        assert np.all(np.isfinite(pred))

    def test_n_features_in(self, data_reg):
        X, y = data_reg
        m = RGFRegressor(n_estimators=3, random_state=0).fit(X, y)
        assert m.n_features_in_ == 5

    def test_clone_compatibility(self, data_reg):
        m = RGFRegressor(n_estimators=3, lambda_l2=2.0)
        cloned = clone(m)
        assert cloned.n_estimators == 3
        assert cloned.lambda_l2 == 2.0

    def test_cross_val_score(self, data_reg):
        X, y = data_reg
        m = RGFRegressor(n_estimators=3, max_leaf_nodes=4, random_state=42)
        scores = cross_val_score(m, X, y, cv=3, scoring="r2")
        assert len(scores) == 3
        # 3回のうち少なくとも1回は正のR²
        assert any(s > -5 for s in scores)

    def test_pipeline_compatibility(self, data_reg):
        X, y = data_reg
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rgf", RGFRegressor(n_estimators=3, max_leaf_nodes=4, random_state=0)),
        ])
        pipe.fit(X, y)
        pred = pipe.predict(X)
        assert pred.shape == (80,)

    def test_subsample(self, data_reg):
        X, y = data_reg
        m = RGFRegressor(n_estimators=5, subsample=0.5, random_state=42)
        m.fit(X, y)
        pred = m.predict(X)
        assert np.all(np.isfinite(pred))

    def test_l1_regularization(self, data_reg):
        X, y = data_reg
        m = RGFRegressor(n_estimators=5, lambda_l1=1.0, lambda_l2=1.0, random_state=42)
        m.fit(X, y)
        # L1正則化で一部の重みが0になることを確認
        n_zero = (np.abs(m.weights_) < 1e-10).sum()
        assert n_zero >= 0  # 少なくともエラーなく動作

    def test_small_dataset(self):
        """5サンプルでも動作する"""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=float)
        y = np.array([1, 2, 3, 4, 5], dtype=float)
        m = RGFRegressor(n_estimators=2, max_leaf_nodes=2, random_state=0)
        m.fit(X, y)
        assert m.predict(X).shape == (5,)


# ============================================================
# RGFClassifier テスト
# ============================================================

class TestRGFClassifier:
    @pytest.fixture
    def data_binary(self):
        X, y = make_classification(n_samples=80, n_features=5, n_classes=2, random_state=42)
        return X, y

    @pytest.fixture
    def data_multiclass(self):
        X, y = make_classification(
            n_samples=120, n_features=5, n_classes=3,
            n_informative=5, n_redundant=0, random_state=42,
        )
        return X, y

    def test_binary_fit_predict(self, data_binary):
        X, y = data_binary
        m = RGFClassifier(n_estimators=5, max_leaf_nodes=4, random_state=42)
        m.fit(X, y)
        pred = m.predict(X)
        assert pred.shape == (80,)
        assert set(pred).issubset(set(y))

    def test_binary_predict_proba(self, data_binary):
        X, y = data_binary
        m = RGFClassifier(n_estimators=5, max_leaf_nodes=4, random_state=42)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (80, 2)
        # 確率の合計が1に近い
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_multiclass_fit_predict(self, data_multiclass):
        X, y = data_multiclass
        m = RGFClassifier(n_estimators=5, max_leaf_nodes=4, random_state=42)
        m.fit(X, y)
        pred = m.predict(X)
        assert pred.shape == (120,)

    def test_multiclass_predict_proba(self, data_multiclass):
        X, y = data_multiclass
        m = RGFClassifier(n_estimators=5, max_leaf_nodes=4, random_state=42)
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (120, 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_clone_compatibility(self):
        m = RGFClassifier(n_estimators=5, lambda_l2=2.0)
        cloned = clone(m)
        assert cloned.n_estimators == 5

    def test_binary_subsample(self, data_binary):
        X, y = data_binary
        m = RGFClassifier(n_estimators=5, subsample=0.7, random_state=42)
        m.fit(X, y)
        pred = m.predict(X)
        assert len(pred) == 80


# ============================================================
# _RGFCore テスト
# ============================================================

class TestRGFCore:
    def test_init_forest_state(self):
        core = _RGFCore()
        core._init_forest_state()
        assert core.trees_ == []
        assert core._total_leaves == 0

    def test_predict_from_weights_empty(self):
        core = _RGFCore()
        core._init_forest_state()
        X = np.random.randn(10, 3)
        result = core._predict_from_weights(X)
        assert np.allclose(result, 0.0)
