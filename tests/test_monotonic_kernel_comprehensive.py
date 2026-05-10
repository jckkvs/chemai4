"""
tests/test_monotonic_kernel_comprehensive.py

MonotonicKernelWrapper / MonotonicKernelClassifierWrapper /
is_soft_monotonic_candidate / wrap_with_soft_monotonic の包括テスト。
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.svm import SVR, SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.base import clone
from sklearn.datasets import make_regression, make_classification

from backend.models.monotonic_kernel import (
    MonotonicKernelWrapper,
    MonotonicKernelClassifierWrapper,
    is_soft_monotonic_candidate,
    wrap_with_soft_monotonic,
    _compute_monotonic_violation,
    _build_grid_X,
    _fit_with_weight,
)


# ============================================================
# ユーティリティテスト
# ============================================================

class TestComputeMonotonicViolation:
    def test_monotonic_increasing(self):
        y = np.array([1, 2, 3, 4, 5])
        assert _compute_monotonic_violation(y, direction=1) == 0.0

    def test_monotonic_decreasing(self):
        y = np.array([5, 4, 3, 2, 1])
        assert _compute_monotonic_violation(y, direction=-1) == 0.0

    def test_violation_increasing(self):
        y = np.array([1, 3, 2, 4])  # 3→2 violates increasing
        assert _compute_monotonic_violation(y, direction=1) > 0

    def test_violation_decreasing(self):
        y = np.array([4, 2, 3, 1])  # 2→3 violates decreasing
        assert _compute_monotonic_violation(y, direction=-1) > 0


class TestBuildGridX:
    def test_basic(self):
        X = np.random.randn(20, 3)
        stats = {i: (float(np.mean(X[:, i])), float(np.std(X[:, i]))) for i in range(3)}
        grid = _build_grid_X(X, 0, stats, sigma_factor=1.5, n_grid=10)
        assert grid.shape == (10, 3)

    def test_zero_std(self):
        X = np.ones((10, 2))
        stats = {0: (1.0, 0.0), 1: (1.0, 0.0)}
        grid = _build_grid_X(X, 0, stats, sigma_factor=1.5, n_grid=5)
        assert grid.shape == (5, 2)


class TestFitWithWeight:
    def test_with_weight(self):
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        w = np.ones(20)
        est = Ridge()
        _fit_with_weight(est, X, y, w)
        assert hasattr(est, "coef_")

    def test_without_weight(self):
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        est = Ridge()
        _fit_with_weight(est, X, y, None)
        assert hasattr(est, "coef_")


# ============================================================
# is_soft_monotonic_candidate テスト
# ============================================================

class TestIsSoftMonotonicCandidate:
    def test_svr(self):
        assert is_soft_monotonic_candidate(SVR()) is True

    def test_kernel_ridge(self):
        assert is_soft_monotonic_candidate(KernelRidge()) is True

    def test_ridge_no(self):
        assert is_soft_monotonic_candidate(Ridge()) is False

    def test_svc(self):
        assert is_soft_monotonic_candidate(SVC()) is True


# ============================================================
# MonotonicKernelWrapper テスト
# ============================================================

class TestMonotonicKernelWrapper:
    @pytest.fixture
    def reg_data(self):
        X, y = make_regression(n_samples=30, n_features=3, noise=0.1, random_state=42)
        return X, y

    def test_no_constraints(self, reg_data):
        X, y = reg_data
        w = MonotonicKernelWrapper(
            base_estimator=SVR(),
            monotonic_constraints=(0, 0, 0),
            max_iter=1,
        )
        w.fit(X, y)
        pred = w.predict(X)
        assert pred.shape == (30,)

    def test_with_constraints(self, reg_data):
        X, y = reg_data
        w = MonotonicKernelWrapper(
            base_estimator=SVR(),
            monotonic_constraints=(1, 0, -1),
            max_iter=2,
            n_grid=10,
        )
        w.fit(X, y)
        pred = w.predict(X)
        assert pred.shape == (30,)

    def test_default_estimator(self, reg_data):
        X, y = reg_data
        w = MonotonicKernelWrapper(
            monotonic_constraints=(1, 0, 0),
            max_iter=1,
        )
        w.fit(X, y)
        assert w.predict(X).shape == (30,)

    def test_score(self, reg_data):
        X, y = reg_data
        w = MonotonicKernelWrapper(base_estimator=SVR(), max_iter=1)
        w.fit(X, y)
        score = w.score(X, y)
        assert isinstance(score, float)

    def test_get_set_params(self):
        w = MonotonicKernelWrapper(
            base_estimator=SVR(C=2.0),
            n_grid=15,
        )
        params = w.get_params(deep=True)
        assert params["n_grid"] == 15
        assert "base_estimator__C" in params

        w.set_params(n_grid=25)
        assert w.n_grid == 25

    def test_clone(self):
        w = MonotonicKernelWrapper(base_estimator=SVR(), n_grid=10)
        cloned = clone(w)
        assert cloned.n_grid == 10


# ============================================================
# MonotonicKernelClassifierWrapper テスト
# ============================================================

class TestMonotonicKernelClassifierWrapper:
    @pytest.fixture
    def cls_data(self):
        X, y = make_classification(n_samples=40, n_features=5, n_informative=3,
                                    n_redundant=1, random_state=42)
        return X, y

    def test_no_constraints(self, cls_data):
        X, y = cls_data
        w = MonotonicKernelClassifierWrapper(
            base_estimator=SVC(probability=True),
            monotonic_constraints=(0, 0, 0, 0, 0),
            max_iter=1,
        )
        w.fit(X, y)
        pred = w.predict(X)
        assert pred.shape == (40,)

    def test_with_constraints(self, cls_data):
        X, y = cls_data
        w = MonotonicKernelClassifierWrapper(
            base_estimator=SVC(probability=True),
            monotonic_constraints=(1, 0, 0, 0, 0),
            max_iter=1,
            n_grid=5,
        )
        w.fit(X, y)
        proba = w.predict_proba(X)
        assert proba.shape == (40, 2)

    def test_default_estimator(self, cls_data):
        X, y = cls_data
        w = MonotonicKernelClassifierWrapper(
            monotonic_constraints=(1, 0, 0, 0, 0),
            max_iter=1,
        )
        w.fit(X, y)
        assert w.predict(X).shape == (40,)

    def test_score(self, cls_data):
        X, y = cls_data
        w = MonotonicKernelClassifierWrapper(
            base_estimator=SVC(probability=True), max_iter=1
        )
        w.fit(X, y)
        score = w.score(X, y)
        assert isinstance(score, float)


# ============================================================
# wrap_with_soft_monotonic テスト
# ============================================================

class TestWrapWithSoftMonotonic:
    def test_no_constraints_returns_original(self):
        est = SVR()
        result = wrap_with_soft_monotonic(est, (0, 0, 0))
        assert result is est

    def test_regressor(self):
        est = SVR()
        result = wrap_with_soft_monotonic(est, (1, 0, -1))
        assert isinstance(result, MonotonicKernelWrapper)

    def test_classifier(self):
        est = SVC(probability=True)
        result = wrap_with_soft_monotonic(est, (1, 0, 0))
        assert isinstance(result, MonotonicKernelClassifierWrapper)
