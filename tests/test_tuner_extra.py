"""
tests/test_tuner_extra.py

tuner.py の低カバレッジ部分を補うテスト。
_convert_optuna_grid_for_random, tune() の全メソッド分岐,
TunerConfig デフォルト、フォールバックロジックを網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from backend.models.tuner import (
    TunerConfig,
    tune,
    _convert_optuna_grid_for_random,
    _run_grid,
    _run_random,
    _extract_results,
)


# ============================================================
# テストデータ
# ============================================================

def _make_regression(n: int = 60, p: int = 4):
    rng = np.random.RandomState(42)
    X = rng.randn(n, p)
    y = X[:, 0] * 2 + rng.randn(n) * 0.1
    return X, y


# ============================================================
# TunerConfig
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
            method="grid", n_iter=10, cv=3,
            scoring="r2", n_jobs=1,
        )
        assert cfg.method == "grid"
        assert cfg.cv == 3


# ============================================================
# _convert_optuna_grid_for_random
# ============================================================

class TestConvertOptunaGrid:
    def test_float_param(self):
        grid = {"alpha": {"type": "float", "low": 0.01, "high": 10.0}}
        result = _convert_optuna_grid_for_random(grid)
        assert "alpha" in result
        # Should be a scipy.stats distribution
        assert hasattr(result["alpha"], "rvs")

    def test_float_log(self):
        grid = {"lr": {"type": "float", "low": 0.001, "high": 1.0, "log": True}}
        result = _convert_optuna_grid_for_random(grid)
        assert "lr" in result
        assert hasattr(result["lr"], "rvs")

    def test_int_param(self):
        grid = {"n_estimators": {"type": "int", "low": 10, "high": 100}}
        result = _convert_optuna_grid_for_random(grid)
        assert "n_estimators" in result
        assert hasattr(result["n_estimators"], "rvs")

    def test_categorical_param(self):
        grid = {"solver": {"type": "categorical", "choices": ["lbfgs", "sgd"]}}
        result = _convert_optuna_grid_for_random(grid)
        assert result["solver"] == ["lbfgs", "sgd"]

    def test_unknown_type(self):
        grid = {"x": {"type": "unknown", "low": 0, "high": 1}}
        result = _convert_optuna_grid_for_random(grid)
        assert "x" in result

    def test_list_passthrough(self):
        grid = {"alpha": [0.1, 1.0, 10.0]}
        result = _convert_optuna_grid_for_random(grid)
        assert result["alpha"] == [0.1, 1.0, 10.0]

    def test_mixed_grid(self):
        grid = {
            "alpha": {"type": "float", "low": 0.01, "high": 10.0},
            "n_estimators": {"type": "int", "low": 10, "high": 200},
            "solver": {"type": "categorical", "choices": ["svd", "lsqr"]},
            "tol": [1e-3, 1e-4, 1e-5],
        }
        result = _convert_optuna_grid_for_random(grid)
        assert len(result) == 4

    def test_empty_grid(self):
        result = _convert_optuna_grid_for_random({})
        assert result == {}


# ============================================================
# tune() — Grid
# ============================================================

class TestTuneGrid:
    def test_grid_search(self):
        X, y = _make_regression()
        model = Ridge()
        cfg = TunerConfig(
            method="grid",
            param_grid={"alpha": [0.1, 1.0, 10.0]},
            cv=2,
            n_jobs=1,
        )
        result = tune(model, X, y, cfg)
        assert "best_estimator" in result
        assert "best_params" in result
        assert "best_score" in result
        assert "cv_results" in result
        assert result["best_params"]["alpha"] in [0.1, 1.0, 10.0]


# ============================================================
# tune() — Random
# ============================================================

class TestTuneRandom:
    def test_random_search(self):
        X, y = _make_regression()
        model = Ridge()
        cfg = TunerConfig(
            method="random",
            param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            n_iter=5,
            cv=2,
            n_jobs=1,
        )
        result = tune(model, X, y, cfg)
        assert "best_estimator" in result
        assert isinstance(result["best_score"], float)


# ============================================================
# tune() — Halving
# ============================================================

class TestTuneHalving:
    def test_halving_grid(self):
        X, y = _make_regression()
        model = Ridge()
        cfg = TunerConfig(
            method="halving_grid",
            param_grid={"alpha": [0.1, 1.0, 10.0]},
            cv=2,
            n_jobs=1,
        )
        result = tune(model, X, y, cfg)
        assert "best_estimator" in result

    def test_halving_random(self):
        X, y = _make_regression()
        model = Ridge()
        cfg = TunerConfig(
            method="halving_random",
            param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            n_iter=3,
            cv=2,
            n_jobs=1,
        )
        result = tune(model, X, y, cfg)
        assert "best_estimator" in result


# ============================================================
# tune() — Optuna (with fallback)
# ============================================================

class TestTuneOptuna:
    def test_optuna_or_fallback(self):
        """Optuna available → Optuna, otherwise Random fallback."""
        X, y = _make_regression()
        model = Ridge()
        cfg = TunerConfig(
            method="optuna",
            param_grid={
                "alpha": {"type": "float", "low": 0.01, "high": 10.0},
            },
            n_iter=3,
            cv=2,
            n_jobs=1,
        )
        result = tune(model, X, y, cfg)
        assert "best_estimator" in result
        assert "best_params" in result


# ============================================================
# tune() — Bayes (with fallback)
# ============================================================

class TestTuneBayes:
    def test_bayes_or_fallback(self):
        X, y = _make_regression()
        model = Ridge()
        cfg = TunerConfig(
            method="bayes",
            param_grid={"alpha": [0.01, 0.1, 1.0, 10.0]},
            n_iter=3,
            cv=2,
            n_jobs=1,
        )
        result = tune(model, X, y, cfg)
        assert "best_estimator" in result


# ============================================================
# tune() — 未知メソッド
# ============================================================

class TestTuneUnknown:
    def test_unknown_method_raises(self):
        X, y = _make_regression()
        model = Ridge()
        cfg = TunerConfig(method="nonexistent")
        with pytest.raises(ValueError, match="未知のチューニング手法"):
            tune(model, X, y, cfg)
