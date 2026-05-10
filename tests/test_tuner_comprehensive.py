"""
tests/test_tuner_comprehensive.py

Tuner モジュールの包括テスト。
TunerConfig, tune(), 各手法（Grid/Random/Halving/Optuna/Bayes）、
フォールバック挙動、エッジケースを検証。
"""
from __future__ import annotations

import pytest
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.datasets import make_regression, make_classification

from backend.models.tuner import TunerConfig, tune, _convert_optuna_grid_for_random


# ============================================================
# TunerConfig テスト
# ============================================================

class TestTunerConfig:
    def test_defaults(self):
        cfg = TunerConfig()
        assert cfg.method == "random"
        assert cfg.n_iter == 50
        assert cfg.cv == 5
        assert cfg.refit is True

    def test_custom(self):
        cfg = TunerConfig(
            method="grid",
            param_grid={"alpha": [0.1, 1.0]},
            cv=3,
            n_iter=10,
        )
        assert cfg.method == "grid"
        assert cfg.cv == 3


# ============================================================
# _convert_optuna_grid_for_random テスト
# ============================================================

class TestConvertOptunaGrid:
    def test_float_grid(self):
        grid = {"alpha": {"type": "float", "low": 0.01, "high": 10.0}}
        result = _convert_optuna_grid_for_random(grid)
        assert "alpha" in result

    def test_float_log_grid(self):
        grid = {"alpha": {"type": "float", "low": 0.01, "high": 10.0, "log": True}}
        result = _convert_optuna_grid_for_random(grid)
        assert "alpha" in result

    def test_int_grid(self):
        grid = {"n_estimators": {"type": "int", "low": 10, "high": 100}}
        result = _convert_optuna_grid_for_random(grid)
        assert "n_estimators" in result

    def test_categorical_grid(self):
        grid = {"solver": {"type": "categorical", "choices": ["svd", "cholesky"]}}
        result = _convert_optuna_grid_for_random(grid)
        assert result["solver"] == ["svd", "cholesky"]

    def test_passthrough_list(self):
        grid = {"alpha": [0.1, 1.0, 10.0]}
        result = _convert_optuna_grid_for_random(grid)
        assert result["alpha"] == [0.1, 1.0, 10.0]


# ============================================================
# tune() テスト - GridSearchCV
# ============================================================

class TestTuneGrid:
    @pytest.fixture
    def reg_data(self):
        return make_regression(n_samples=50, n_features=3, noise=0.1, random_state=42)

    def test_grid_search(self, reg_data):
        X, y = reg_data
        model = Ridge()
        cfg = TunerConfig(
            method="grid",
            param_grid={"alpha": [0.1, 1.0, 10.0]},
            cv=3,
            scoring="neg_root_mean_squared_error",
        )
        result = tune(model, X, y, cfg)
        assert "best_estimator" in result
        assert "best_params" in result
        assert "best_score" in result
        assert "cv_results" in result
        assert result["best_params"]["alpha"] in [0.1, 1.0, 10.0]

    def test_random_search(self, reg_data):
        X, y = reg_data
        model = Ridge()
        cfg = TunerConfig(
            method="random",
            param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            cv=3,
            n_iter=3,
        )
        result = tune(model, X, y, cfg)
        assert result["best_params"]["alpha"] in [0.01, 0.1, 1.0, 10.0, 100.0]


# ============================================================
# tune() テスト - Unknown method
# ============================================================

class TestTuneUnknown:
    def test_unknown_method(self):
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        model = Ridge()
        cfg = TunerConfig(method="unknown_method")
        with pytest.raises(ValueError, match="未知のチューニング手法"):
            tune(model, X, y, cfg)


# ============================================================
# tune() テスト - Halving
# ============================================================

class TestTuneHalving:
    @pytest.fixture
    def reg_data(self):
        return make_regression(n_samples=80, n_features=3, noise=0.1, random_state=42)

    def test_halving_grid(self, reg_data):
        X, y = reg_data
        model = Ridge()
        cfg = TunerConfig(
            method="halving_grid",
            param_grid={"alpha": [0.01, 0.1, 1.0, 10.0]},
            cv=3,
        )
        # HalvingGridSearchCVが利用可能でもフォールバックでもエラーなく動作
        result = tune(model, X, y, cfg)
        assert "best_estimator" in result

    def test_halving_random(self, reg_data):
        X, y = reg_data
        model = Ridge()
        cfg = TunerConfig(
            method="halving_random",
            param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            cv=3,
            n_iter=3,
        )
        result = tune(model, X, y, cfg)
        assert "best_estimator" in result


# ============================================================
# tune() テスト - Optuna (optional)
# ============================================================

class TestTuneOptuna:
    @pytest.fixture
    def reg_data(self):
        return make_regression(n_samples=50, n_features=3, noise=0.1, random_state=42)

    def test_optuna_or_fallback(self, reg_data):
        """Optunaインストール済みならOptuna、なければRandomSearchにフォールバック"""
        X, y = reg_data
        model = Ridge()
        cfg = TunerConfig(
            method="optuna",
            param_grid={
                "alpha": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            },
            cv=3,
            n_iter=3,
        )
        result = tune(model, X, y, cfg)
        assert "best_estimator" in result
        assert "best_score" in result


# ============================================================
# tune() テスト - Bayes (optional)
# ============================================================

class TestTuneBayes:
    @pytest.fixture
    def reg_data(self):
        return make_regression(n_samples=50, n_features=3, noise=0.1, random_state=42)

    def test_bayes_or_fallback(self, reg_data):
        """scikit-optimizeインストール済みならBayes、なければRandomSearchにフォールバック"""
        X, y = reg_data
        model = Ridge()
        cfg = TunerConfig(
            method="bayes",
            param_grid={"alpha": [0.01, 0.1, 1.0, 10.0]},
            cv=3,
            n_iter=3,
        )
        result = tune(model, X, y, cfg)
        assert "best_estimator" in result
