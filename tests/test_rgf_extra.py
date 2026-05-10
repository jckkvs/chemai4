"""
tests/test_rgf_extra.py

rgf.py のカバレッジ改善テスト。
RGFRegressor, RGFClassifier, ユーティリティ関数を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.models.rgf import (
    RGFRegressor,
    RGFClassifier,
    _to_numpy,
    _sigmoid,
    _softmax,
)


# ============================================================
# ユーティリティ
# ============================================================

class TestRGFUtilities:
    def test_to_numpy_array(self):
        X = np.array([[1, 2], [3, 4]])
        result = _to_numpy(X)
        assert result.dtype == np.float64

    def test_to_numpy_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _to_numpy(df)
        assert isinstance(result, np.ndarray)

    def test_sigmoid(self):
        x = np.array([0, 1, -1, 100, -100])
        s = _sigmoid(x)
        assert np.allclose(s[0], 0.5)
        assert s[3] > 0.99
        assert s[4] < 0.01

    def test_softmax(self):
        x = np.array([[1, 2, 3], [1, 1, 1]])
        s = _softmax(x)
        np.testing.assert_allclose(s.sum(axis=1), 1.0, atol=1e-6)


# ============================================================
# RGFRegressor
# ============================================================

class TestRGFRegressor:
    def test_fit_predict(self):
        rng = np.random.RandomState(42)
        X = rng.randn(60, 3)
        y = X[:, 0] * 3 + X[:, 1] * 2 + rng.randn(60) * 0.1
        model = RGFRegressor(n_estimators=5, max_leaf_nodes=8, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 60

    def test_with_l1(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        y = X[:, 0]
        model = RGFRegressor(n_estimators=3, lambda_l1=0.1, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 40

    def test_subsample(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        y = X[:, 0]
        model = RGFRegressor(n_estimators=3, subsample=0.7, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 40

    def test_with_dataframe(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.randn(30, 3), columns=["a", "b", "c"])
        y = df["a"] * 2
        model = RGFRegressor(n_estimators=3, random_state=42)
        model.fit(df, y)
        preds = model.predict(df)
        assert len(preds) == 30


# ============================================================
# RGFClassifier
# ============================================================

class TestRGFClassifier:
    def test_binary_fit_predict(self):
        rng = np.random.RandomState(42)
        X = rng.randn(60, 3)
        y = (X[:, 0] > 0).astype(int)
        model = RGFClassifier(n_estimators=5, max_leaf_nodes=8, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 60
        assert set(preds).issubset({0, 1})

    def test_binary_predict_proba(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        y = (X[:, 0] > 0).astype(int)
        model = RGFClassifier(n_estimators=3, random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (40, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_multiclass(self):
        rng = np.random.RandomState(42)
        X = rng.randn(90, 3)
        y = np.repeat([0, 1, 2], 30)
        model = RGFClassifier(n_estimators=3, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 90

    def test_multiclass_proba(self):
        rng = np.random.RandomState(42)
        X = rng.randn(60, 3)
        y = np.repeat([0, 1, 2], 20)
        model = RGFClassifier(n_estimators=3, random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (60, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_with_l1_subsample(self):
        rng = np.random.RandomState(42)
        X = rng.randn(60, 3)
        y = (X[:, 0] > 0).astype(int)
        model = RGFClassifier(
            n_estimators=3, lambda_l1=0.05, subsample=0.8, random_state=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == 60
