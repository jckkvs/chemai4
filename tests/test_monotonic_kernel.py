# -*- coding: utf-8 -*-
"""
tests/test_monotonic_kernel.py

monotonic_kernel.py（ソフト単調性制約ラッパー）のユニットテスト。

カバー対象:
  - MonotonicKernelWrapper (回帰)
  - MonotonicKernelClassifierWrapper (分類)
  - _compute_monotonic_violation ヘルパー
  - wrap_with_soft_monotonic ファクトリー
  - is_soft_monotonic_candidate 判定
  - get_params / set_params (sklearn clone互換)
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.svm import SVR, SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.base import clone

from backend.models.monotonic_kernel import (
    MonotonicKernelWrapper,
    MonotonicKernelClassifierWrapper,
    _compute_monotonic_violation,
    _build_grid_X,
    wrap_with_soft_monotonic,
    is_soft_monotonic_candidate,
)


# ═══════════════════════════════════════════════════════════════════
# テストデータ
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def monotonic_regression_data():
    """正相関が明確な回帰データ（x→y増加）"""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 10, (80, 3))
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(0, 0.5, 80)
    return X, y


@pytest.fixture(scope="module")
def classification_data():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


# ═══════════════════════════════════════════════════════════════════
# _compute_monotonic_violation
# ═══════════════════════════════════════════════════════════════════

class TestComputeMonotonicViolation:

    def test_perfect_increasing(self):
        """完全単調増加 → 違反0"""
        y_grid = np.array([1.0, 2.0, 3.0, 4.0])
        assert _compute_monotonic_violation(y_grid, direction=1) == 0.0

    def test_perfect_decreasing(self):
        """完全単調減少 → 違反0"""
        y_grid = np.array([4.0, 3.0, 2.0, 1.0])
        assert _compute_monotonic_violation(y_grid, direction=-1) == 0.0

    def test_violation_increasing(self):
        """増加制約に違反"""
        y_grid = np.array([1.0, 3.0, 2.0, 4.0])  # 3→2で違反
        v = _compute_monotonic_violation(y_grid, direction=1)
        assert v > 0

    def test_violation_decreasing(self):
        """減少制約に違反"""
        y_grid = np.array([4.0, 2.0, 3.0, 1.0])  # 2→3で違反
        v = _compute_monotonic_violation(y_grid, direction=-1)
        assert v > 0

    def test_constant_no_violation(self):
        """一定値 → 違反0"""
        y_grid = np.array([2.0, 2.0, 2.0])
        assert _compute_monotonic_violation(y_grid, direction=1) == 0.0
        assert _compute_monotonic_violation(y_grid, direction=-1) == 0.0


# ═══════════════════════════════════════════════════════════════════
# MonotonicKernelWrapper (回帰)
# ═══════════════════════════════════════════════════════════════════

class TestMonotonicKernelWrapper:

    def test_fit_predict_no_constraints(self, monotonic_regression_data):
        """制約なしで普通にfit/predict"""
        X, y = monotonic_regression_data
        wrapper = MonotonicKernelWrapper(
            base_estimator=SVR(kernel="rbf"),
            monotonic_constraints=(0, 0, 0),
        )
        wrapper.fit(X, y)
        pred = wrapper.predict(X)
        assert pred.shape == (len(X),)

    def test_fit_predict_with_constraints(self, monotonic_regression_data):
        """増加制約付きfit/predict"""
        X, y = monotonic_regression_data
        wrapper = MonotonicKernelWrapper(
            base_estimator=SVR(kernel="rbf"),
            monotonic_constraints=(1, -1, 0),
            max_iter=2,
            n_grid=10,
        )
        wrapper.fit(X, y)
        pred = wrapper.predict(X)
        assert pred.shape == (len(X),)

    def test_default_base_estimator(self, monotonic_regression_data):
        """base_estimator=Noneの場合はSVRがデフォルト"""
        X, y = monotonic_regression_data
        wrapper = MonotonicKernelWrapper(monotonic_constraints=(1, 0, 0))
        wrapper.fit(X, y)
        assert hasattr(wrapper, "estimator_")

    def test_kernel_ridge(self, monotonic_regression_data):
        """KernelRidgeでも動作する"""
        X, y = monotonic_regression_data
        wrapper = MonotonicKernelWrapper(
            base_estimator=KernelRidge(kernel="rbf"),
            monotonic_constraints=(1, 0, 0),
            max_iter=1,
        )
        wrapper.fit(X, y)
        pred = wrapper.predict(X)
        assert pred.shape == (len(X),)

    def test_score_method(self, monotonic_regression_data):
        """scoreメソッドが動作する"""
        X, y = monotonic_regression_data
        wrapper = MonotonicKernelWrapper(
            base_estimator=SVR(), monotonic_constraints=(0, 0, 0)
        )
        wrapper.fit(X, y)
        score = wrapper.score(X, y)
        assert isinstance(score, float)

    def test_get_params(self):
        wrapper = MonotonicKernelWrapper(
            base_estimator=SVR(kernel="linear"),
            monotonic_constraints=(1, -1),
            n_grid=30,
        )
        params = wrapper.get_params(deep=True)
        assert params["n_grid"] == 30
        assert params["monotonic_constraints"] == (1, -1)
        assert "base_estimator__kernel" in params

    def test_set_params(self):
        wrapper = MonotonicKernelWrapper(
            base_estimator=SVR(),
            monotonic_constraints=(1, 0),
        )
        wrapper.set_params(n_grid=50)
        assert wrapper.n_grid == 50

    def test_clone_compatibility(self):
        """sklearn clone()互換"""
        wrapper = MonotonicKernelWrapper(
            base_estimator=SVR(),
            monotonic_constraints=(1, -1, 0),
            n_grid=25,
        )
        cloned = clone(wrapper)
        assert cloned.n_grid == 25
        assert cloned.monotonic_constraints == (1, -1, 0)

    def test_n_features_in(self, monotonic_regression_data):
        X, y = monotonic_regression_data
        wrapper = MonotonicKernelWrapper(
            base_estimator=SVR(), monotonic_constraints=(0, 0, 0)
        )
        wrapper.fit(X, y)
        assert wrapper.n_features_in_ == 3

    def test_violation_recorded(self, monotonic_regression_data):
        """制約付きの場合、violation値が記録される"""
        X, y = monotonic_regression_data
        wrapper = MonotonicKernelWrapper(
            base_estimator=SVR(),
            monotonic_constraints=(1, 0, 0),
            max_iter=1,
        )
        wrapper.fit(X, y)
        assert hasattr(wrapper, "monotonic_violation_")


# ═══════════════════════════════════════════════════════════════════
# MonotonicKernelClassifierWrapper
# ═══════════════════════════════════════════════════════════════════

class TestMonotonicKernelClassifierWrapper:

    def test_fit_predict_no_constraints(self, classification_data):
        X, y = classification_data
        wrapper = MonotonicKernelClassifierWrapper(
            base_estimator=SVC(probability=True),
            monotonic_constraints=(0, 0, 0),
        )
        wrapper.fit(X, y)
        pred = wrapper.predict(X)
        assert pred.shape == (len(X),)
        assert set(pred).issubset({0, 1})

    def test_predict_proba(self, classification_data):
        X, y = classification_data
        wrapper = MonotonicKernelClassifierWrapper(
            base_estimator=SVC(probability=True),
            monotonic_constraints=(0, 0, 0),
        )
        wrapper.fit(X, y)
        proba = wrapper.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_with_constraints(self, classification_data):
        X, y = classification_data
        wrapper = MonotonicKernelClassifierWrapper(
            base_estimator=SVC(probability=True),
            monotonic_constraints=(1, 0, 0),
            max_iter=1, n_grid=10,
        )
        wrapper.fit(X, y)
        pred = wrapper.predict(X)
        assert len(pred) == len(X)

    def test_default_base_estimator(self, classification_data):
        X, y = classification_data
        wrapper = MonotonicKernelClassifierWrapper(
            monotonic_constraints=(0, 0, 0)
        )
        wrapper.fit(X, y)
        assert hasattr(wrapper, "classes_")


# ═══════════════════════════════════════════════════════════════════
# wrap_with_soft_monotonic / is_soft_monotonic_candidate
# ═══════════════════════════════════════════════════════════════════

class TestWrapFactory:

    def test_svr_is_candidate(self):
        assert is_soft_monotonic_candidate(SVR()) is True

    def test_kernel_ridge_is_candidate(self):
        assert is_soft_monotonic_candidate(KernelRidge()) is True

    def test_svc_is_candidate(self):
        assert is_soft_monotonic_candidate(SVC()) is True

    def test_ridge_not_candidate(self):
        assert is_soft_monotonic_candidate(Ridge()) is False

    def test_wrap_regressor(self):
        wrapped = wrap_with_soft_monotonic(SVR(), (1, -1, 0))
        assert isinstance(wrapped, MonotonicKernelWrapper)

    def test_wrap_classifier(self):
        wrapped = wrap_with_soft_monotonic(SVC(probability=True), (1, 0))
        assert isinstance(wrapped, MonotonicKernelClassifierWrapper)

    def test_no_constraint_returns_original(self):
        model = SVR()
        result = wrap_with_soft_monotonic(model, (0, 0, 0))
        assert result is model  # 同じオブジェクト

    def test_wrap_empty_constraints_returns_original(self):
        model = SVR()
        result = wrap_with_soft_monotonic(model, ())
        assert result is model


# ═══════════════════════════════════════════════════════════════════
# _build_grid_X ヘルパー
# ═══════════════════════════════════════════════════════════════════

class TestBuildGridX:

    def test_grid_shape(self):
        X_ref = np.random.default_rng(42).normal(size=(50, 3))
        stats = {i: (float(np.mean(X_ref[:, i])), float(np.std(X_ref[:, i]))) for i in range(3)}
        X_grid = _build_grid_X(X_ref, feat_idx=0, feature_stats=stats,
                               sigma_factor=1.5, n_grid=20)
        assert X_grid.shape == (20, 3)

    def test_grid_varies_only_target_feature(self):
        X_ref = np.random.default_rng(42).normal(size=(50, 3))
        stats = {i: (float(np.mean(X_ref[:, i])), float(np.std(X_ref[:, i]))) for i in range(3)}
        X_grid = _build_grid_X(X_ref, feat_idx=1, feature_stats=stats,
                               sigma_factor=1.5, n_grid=10)
        # 特徴量0と2は全行同じ値（中央値）
        assert len(np.unique(X_grid[:, 0])) == 1
        assert len(np.unique(X_grid[:, 2])) == 1
        # 特徴量1は変動する
        assert len(np.unique(X_grid[:, 1])) == 10
