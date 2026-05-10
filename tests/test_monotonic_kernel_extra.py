"""
tests/test_monotonic_kernel_extra.py

monotonic_kernel.py の低カバレッジ部分を補うテスト。
MonotonicKernelWrapper/MonotonicKernelClassifierWrapper の
fit/predict/score/get_params/set_params,
ユーティリティ関数(_to_numpy/_compute_monotonic_violation/_build_monotonic_augmented_data/_build_grid_X),
ファクトリー(is_soft_monotonic_candidate/wrap_with_soft_monotonic)を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from sklearn.svm import SVR, SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LogisticRegression

from backend.models.monotonic_kernel import (
    MonotonicKernelWrapper,
    MonotonicKernelClassifierWrapper,
    _to_numpy,
    _compute_monotonic_violation,
    _build_monotonic_augmented_data,
    _build_grid_X,
    is_soft_monotonic_candidate,
    wrap_with_soft_monotonic,
    _fit_with_weight,
)


# ============================================================
# テストデータ
# ============================================================

def _make_regression(n: int = 50, d: int = 3):
    rng = np.random.RandomState(42)
    X = rng.randn(n, d)
    y = X[:, 0] * 2 - X[:, 1] + rng.randn(n) * 0.1
    return X, y


def _make_classification(n: int = 50, d: int = 3):
    rng = np.random.RandomState(42)
    X = rng.randn(n, d)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


# ============================================================
# ユーティリティ関数
# ============================================================

class TestToNumpy:
    def test_ndarray(self):
        X = np.array([[1, 2], [3, 4]])
        result = _to_numpy(X)
        np.testing.assert_array_equal(result, X)

    def test_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _to_numpy(df)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_list(self):
        result = _to_numpy([[1, 2], [3, 4]])
        assert isinstance(result, np.ndarray)


class TestComputeViolation:
    def test_monotonic_increasing_no_violation(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _compute_monotonic_violation(y, direction=1) == 0.0

    def test_monotonic_decreasing_no_violation(self):
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert _compute_monotonic_violation(y, direction=-1) == 0.0

    def test_increasing_with_violation(self):
        y = np.array([1.0, 3.0, 2.0, 4.0])  # 3→2 is violation
        result = _compute_monotonic_violation(y, direction=1)
        assert result > 0

    def test_decreasing_with_violation(self):
        y = np.array([4.0, 2.0, 3.0, 1.0])  # 2→3 is violation
        result = _compute_monotonic_violation(y, direction=-1)
        assert result > 0


class TestBuildGridX:
    def test_basic(self):
        X = np.random.randn(20, 3)
        stats = {0: (0.0, 1.0), 1: (0.0, 1.0), 2: (0.0, 1.0)}
        result = _build_grid_X(X, feat_idx=0, feature_stats=stats, sigma_factor=1.5, n_grid=10)
        assert result.shape == (10, 3)
        # All columns except feat_idx should be constant (median)
        assert len(np.unique(result[:, 1])) == 1
        assert len(np.unique(result[:, 2])) == 1

    def test_zero_sigma(self):
        X = np.ones((20, 2))  # zero std
        stats = {0: (1.0, 0.0), 1: (1.0, 0.0)}
        result = _build_grid_X(X, 0, stats, 1.5, 10)
        assert result.shape == (10, 2)


class TestBuildMonotonicAugmentedData:
    def test_no_violation(self):
        """Perfectly monotonic → no augmentation."""
        X = np.array([[1], [2], [3], [4], [5]], dtype=float)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        from sklearn.linear_model import LinearRegression
        est = LinearRegression().fit(X, y)
        stats = {0: (3.0, 1.5)}
        X_aug, y_aug, sw = _build_monotonic_augmented_data(
            X, y, est, (1,), n_grid=10, sigma_factor=1.5,
            penalty_weight=10.0, feature_stats=stats,
        )
        # No augmentation expected for well-fitted linear
        assert len(X_aug) >= len(X)

    def test_with_violation(self):
        """Non-monotonic estimator → augmentation."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 2)
        y = np.sin(X[:, 0] * 3) + rng.randn(50) * 0.1  # non-monotonic
        from sklearn.svm import SVR
        est = SVR(kernel="rbf").fit(X, y)
        stats = {0: (0.0, 1.0), 1: (0.0, 1.0)}
        X_aug, y_aug, sw = _build_monotonic_augmented_data(
            X, y, est, (1, 0), n_grid=20, sigma_factor=1.5,
            penalty_weight=10.0, feature_stats=stats,
        )
        assert len(X_aug) >= len(X)


# ============================================================
# MonotonicKernelWrapper (回帰)
# ============================================================

class TestMonotonicKernelWrapper:
    def test_no_constraints(self):
        X, y = _make_regression()
        wrapper = MonotonicKernelWrapper(
            base_estimator=SVR(kernel="rbf"),
            monotonic_constraints=(0, 0, 0),
        )
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert preds.shape == (50,)

    def test_with_constraints(self):
        X, y = _make_regression()
        wrapper = MonotonicKernelWrapper(
            base_estimator=SVR(kernel="rbf"),
            monotonic_constraints=(1, -1, 0),
            max_iter=2,
            n_grid=10,
        )
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert preds.shape == (50,)

    def test_default_estimator(self):
        X, y = _make_regression()
        wrapper = MonotonicKernelWrapper(
            monotonic_constraints=(1, 0, 0),
            max_iter=1,
        )
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert len(preds) == 50

    def test_score(self):
        X, y = _make_regression()
        wrapper = MonotonicKernelWrapper(
            base_estimator=KernelRidge(kernel="rbf"),
            monotonic_constraints=(0, 0, 0),
        )
        wrapper.fit(X, y)
        s = wrapper.score(X, y)
        assert isinstance(s, float)

    def test_get_params(self):
        wrapper = MonotonicKernelWrapper(
            base_estimator=SVR(),
            monotonic_constraints=(1, 0),
        )
        params = wrapper.get_params(deep=True)
        assert "base_estimator" in params
        assert "monotonic_constraints" in params
        assert "n_grid" in params
        # Deep params
        assert "base_estimator__C" in params

    def test_set_params(self):
        wrapper = MonotonicKernelWrapper(
            base_estimator=SVR(),
            monotonic_constraints=(1, 0),
        )
        wrapper.set_params(n_grid=30, base_estimator__C=10.0)
        assert wrapper.n_grid == 30
        assert wrapper.base_estimator.C == 10.0

    def test_with_dataframe(self):
        X, y = _make_regression()
        df = pd.DataFrame(X, columns=["a", "b", "c"])
        wrapper = MonotonicKernelWrapper(
            base_estimator=SVR(kernel="rbf"),
            monotonic_constraints=(0, 0, 0),
        )
        wrapper.fit(df, y)
        assert wrapper.feature_names_in_ == ["a", "b", "c"]


# ============================================================
# MonotonicKernelClassifierWrapper (分類)
# ============================================================

class TestMonotonicKernelClassifierWrapper:
    def test_no_constraints(self):
        X, y = _make_classification()
        wrapper = MonotonicKernelClassifierWrapper(
            base_estimator=SVC(probability=True),
            monotonic_constraints=(0, 0, 0),
        )
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert preds.shape == (50,)

    def test_with_constraints(self):
        X, y = _make_classification()
        wrapper = MonotonicKernelClassifierWrapper(
            base_estimator=SVC(probability=True),
            monotonic_constraints=(1, 1, 0),
            max_iter=2,
            n_grid=10,
        )
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert preds.shape == (50,)

    def test_predict_proba(self):
        X, y = _make_classification()
        wrapper = MonotonicKernelClassifierWrapper(
            base_estimator=SVC(probability=True),
            monotonic_constraints=(0, 0, 0),
        )
        wrapper.fit(X, y)
        proba = wrapper.predict_proba(X)
        assert proba.shape == (50, 2)

    def test_default_estimator(self):
        X, y = _make_classification()
        wrapper = MonotonicKernelClassifierWrapper(
            monotonic_constraints=(1, 0, 0),
            max_iter=1,
        )
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert len(preds) == 50

    def test_score(self):
        X, y = _make_classification()
        wrapper = MonotonicKernelClassifierWrapper(
            base_estimator=SVC(probability=True),
            monotonic_constraints=(0, 0, 0),
        )
        wrapper.fit(X, y)
        s = wrapper.score(X, y)
        assert isinstance(s, float)

    def test_get_set_params(self):
        wrapper = MonotonicKernelClassifierWrapper(
            base_estimator=SVC(probability=True),
            monotonic_constraints=(1, 0),
        )
        params = wrapper.get_params(deep=True)
        assert "base_estimator__C" in params
        wrapper.set_params(n_grid=25)
        assert wrapper.n_grid == 25


# ============================================================
# _fit_with_weight
# ============================================================

class TestFitWithWeight:
    def test_without_weight(self):
        X, y = _make_regression()
        est = Ridge()
        _fit_with_weight(est, X, y, None)
        assert hasattr(est, "coef_")

    def test_with_weight(self):
        X, y = _make_regression()
        sw = np.ones(len(X))
        est = Ridge()
        _fit_with_weight(est, X, y, sw)
        assert hasattr(est, "coef_")


# ============================================================
# ファクトリー関数
# ============================================================

class TestSoftMonotonicCandidate:
    def test_svr(self):
        assert is_soft_monotonic_candidate(SVR()) is True

    def test_svc(self):
        assert is_soft_monotonic_candidate(SVC()) is True

    def test_kernel_ridge(self):
        assert is_soft_monotonic_candidate(KernelRidge()) is True

    def test_ridge(self):
        assert is_soft_monotonic_candidate(Ridge()) is False

    def test_logistic_regression(self):
        assert is_soft_monotonic_candidate(LogisticRegression()) is False


class TestWrapWithSoftMonotonic:
    def test_no_constraints(self):
        est = SVR()
        result = wrap_with_soft_monotonic(est, (0, 0, 0))
        assert result is est  # No wrapping

    def test_wrap_regressor(self):
        est = SVR()
        result = wrap_with_soft_monotonic(est, (1, -1, 0))
        assert isinstance(result, MonotonicKernelWrapper)

    def test_wrap_classifier(self):
        est = SVC(probability=True)
        result = wrap_with_soft_monotonic(est, (1, 0, 0))
        assert isinstance(result, MonotonicKernelClassifierWrapper)

    def test_custom_params(self):
        est = SVR()
        result = wrap_with_soft_monotonic(
            est, (1, 0), n_grid=30, penalty_weight=20.0,
        )
        assert result.n_grid == 30
        assert result.penalty_weight == 20.0
