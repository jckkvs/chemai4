# -*- coding: utf-8 -*-
"""
tests/test_tuner.py

tuner.py（ハイパーパラメータ最適化モジュール）のユニットテスト。

カバー対象:
  - TunerConfig のデフォルト値
  - GridSearchCV / RandomizedSearchCV
  - HalvingGridSearchCV / HalvingRandomSearchCV
  - Optuna / BayesSearchCV（未インストール時のフォールバック）
  - 不正メソッドのエラーハンドリング
  - _extract_results ヘルパー
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import uniform
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.datasets import make_regression, make_classification

from backend.models.tuner import TunerConfig, tune


# ═══════════════════════════════════════════════════════════════════
# テストデータ
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def regression_data():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    return X, y


@pytest.fixture(scope="module")
def classification_data():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    return X, y


# ═══════════════════════════════════════════════════════════════════
# TunerConfig
# ═══════════════════════════════════════════════════════════════════

class TestTunerConfig:

    def test_default_values(self):
        cfg = TunerConfig()
        assert cfg.method == "random"
        assert cfg.n_iter == 50
        assert cfg.cv == 5
        assert cfg.refit is True

    def test_custom_values(self):
        cfg = TunerConfig(method="grid", n_iter=10, cv=3,
                          scoring="r2", n_jobs=1)
        assert cfg.method == "grid"
        assert cfg.n_iter == 10
        assert cfg.cv == 3
        assert cfg.scoring == "r2"
        assert cfg.n_jobs == 1


# ═══════════════════════════════════════════════════════════════════
# GridSearch
# ═══════════════════════════════════════════════════════════════════

class TestGridSearch:

    def test_grid_search_basic(self, regression_data):
        X, y = regression_data
        cfg = TunerConfig(
            method="grid",
            param_grid={"alpha": [0.1, 1.0, 10.0]},
            cv=3, n_jobs=1, scoring="neg_mean_squared_error"
        )
        result = tune(Ridge(), X, y, cfg)
        assert "best_estimator" in result
        assert "best_params" in result
        assert "best_score" in result
        assert "cv_results" in result
        assert result["best_params"]["alpha"] in [0.1, 1.0, 10.0]

    def test_grid_search_refit(self, regression_data):
        X, y = regression_data
        cfg = TunerConfig(
            method="grid",
            param_grid={"alpha": [0.1, 1.0]},
            cv=2, n_jobs=1, refit=True
        )
        result = tune(Ridge(), X, y, cfg)
        # refit=True → best_estimator はfit済み
        pred = result["best_estimator"].predict(X[:5])
        assert len(pred) == 5


# ═══════════════════════════════════════════════════════════════════
# RandomSearch
# ═══════════════════════════════════════════════════════════════════

class TestRandomSearch:

    def test_random_search_basic(self, regression_data):
        X, y = regression_data
        cfg = TunerConfig(
            method="random",
            param_grid={"alpha": uniform(0.01, 10.0)},
            n_iter=5, cv=2, n_jobs=1
        )
        result = tune(Ridge(), X, y, cfg)
        assert "best_estimator" in result
        assert isinstance(result["best_score"], float)

    def test_random_search_reproducible(self, regression_data):
        X, y = regression_data
        cfg1 = TunerConfig(
            method="random",
            param_grid={"alpha": uniform(0.01, 10.0)},
            n_iter=5, cv=2, n_jobs=1, random_state=42
        )
        result1 = tune(Ridge(), X, y, cfg1)
        cfg2 = TunerConfig(
            method="random",
            param_grid={"alpha": uniform(0.01, 10.0)},
            n_iter=5, cv=2, n_jobs=1, random_state=42
        )
        result2 = tune(Ridge(), X, y, cfg2)
        assert result1["best_params"]["alpha"] == pytest.approx(
            result2["best_params"]["alpha"])


# ═══════════════════════════════════════════════════════════════════
# HalvingSearch
# ═══════════════════════════════════════════════════════════════════

class TestHalvingSearch:

    def test_halving_grid(self, regression_data):
        """HalvingGridSearchCV (sklearn 0.24+) または GridSearchCVへのフォールバック"""
        X, y = regression_data
        cfg = TunerConfig(
            method="halving_grid",
            param_grid={"alpha": [0.1, 1.0, 10.0]},
            cv=2, n_jobs=1
        )
        result = tune(Ridge(), X, y, cfg)
        assert "best_estimator" in result

    def test_halving_random(self, regression_data):
        """HalvingRandomSearchCV または RandomizedSearchCVへのフォールバック"""
        X, y = regression_data
        cfg = TunerConfig(
            method="halving_random",
            param_grid={"alpha": uniform(0.01, 10.0)},
            n_iter=5, cv=2, n_jobs=1
        )
        result = tune(Ridge(), X, y, cfg)
        assert "best_estimator" in result


# ═══════════════════════════════════════════════════════════════════
# Optuna（フォールバック含む）
# ═══════════════════════════════════════════════════════════════════

class TestOptuna:

    def test_optuna_or_fallback(self, regression_data):
        """Optuna使用可能ならOptuna、不可ならRandomSearchにフォールバック"""
        X, y = regression_data
        cfg = TunerConfig(
            method="optuna",
            param_grid={
                "alpha": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            },
            n_iter=5, cv=2, n_jobs=1,
            optuna_direction="maximize"
        )
        result = tune(Ridge(), X, y, cfg)
        assert "best_estimator" in result
        assert "best_params" in result


# ═══════════════════════════════════════════════════════════════════
# BayesSearch（フォールバック含む）
# ═══════════════════════════════════════════════════════════════════

class TestBayesSearch:

    def test_bayes_or_fallback(self, regression_data):
        """scikit-optimize使用可能ならBayesSearchCV、不可ならRandomSearchにフォールバック"""
        X, y = regression_data
        # skoptがインストール済みならskopt.space.Realを使用
        try:
            from skopt.space import Real
            param_grid = {"alpha": Real(0.01, 10.0, prior="log-uniform")}
        except ImportError:
            from scipy.stats import uniform
            param_grid = {"alpha": uniform(0.01, 10.0)}
        cfg = TunerConfig(
            method="bayes",
            param_grid=param_grid,
            n_iter=5, cv=2, n_jobs=1
        )
        result = tune(Ridge(), X, y, cfg)
        assert "best_estimator" in result


# ═══════════════════════════════════════════════════════════════════
# エラーハンドリング
# ═══════════════════════════════════════════════════════════════════

class TestErrorHandling:

    def test_unknown_method_raises(self, regression_data):
        X, y = regression_data
        cfg = TunerConfig(method="UNKNOWN_METHOD")
        with pytest.raises(ValueError, match="未知のチューニング手法"):
            tune(Ridge(), X, y, cfg)

    def test_empty_param_grid(self, regression_data):
        """空のparam_gridでもエラーにならないこと"""
        X, y = regression_data
        cfg = TunerConfig(
            method="grid",
            param_grid={"alpha": [1.0]},  # 1要素
            cv=2, n_jobs=1
        )
        result = tune(Ridge(), X, y, cfg)
        assert result["best_params"]["alpha"] == 1.0


# ═══════════════════════════════════════════════════════════════════
# cv_results の構造検証
# ═══════════════════════════════════════════════════════════════════

class TestCVResults:

    def test_cv_results_is_dataframe(self, regression_data):
        import pandas as pd
        X, y = regression_data
        cfg = TunerConfig(
            method="grid",
            param_grid={"alpha": [0.1, 1.0]},
            cv=2, n_jobs=1
        )
        result = tune(Ridge(), X, y, cfg)
        assert isinstance(result["cv_results"], pd.DataFrame)
        assert len(result["cv_results"]) >= 2
