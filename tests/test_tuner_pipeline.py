# -*- coding: utf-8 -*-
"""
tests/test_tuner_pipeline.py

tuner.py の Pipeline プレフィックス自動変換テスト。

テストID:
    T-TUN01: _detect_pipeline_step_name
    T-TUN02: _prefix_param_grid
    T-TUN03: _strip_prefix_from_params
    T-TUN04: tune() with Pipeline (GridSearchCV)
    T-TUN05: tune() with bare estimator (GridSearchCV)
    T-TUN06: tune() with Pipeline (RandomizedSearchCV)
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.models.tuner import (
    TunerConfig,
    tune,
    _detect_pipeline_step_name,
    _prefix_param_grid,
    _strip_prefix_from_params,
)


# ============================================================
# T-TUN01: Pipeline ステップ名検出
# ============================================================

class TestDetectPipelineStepName:
    def test_pipeline_with_estimator_step(self):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("estimator", Ridge()),
        ])
        assert _detect_pipeline_step_name(pipe) == "estimator"

    def test_pipeline_with_model_step(self):
        pipe = Pipeline([
            ("preprocess", StandardScaler()),
            ("model", Lasso()),
        ])
        assert _detect_pipeline_step_name(pipe) == "model"

    def test_bare_estimator(self):
        assert _detect_pipeline_step_name(Ridge()) is None

    def test_single_step_pipeline(self):
        pipe = Pipeline([("clf", Ridge())])
        assert _detect_pipeline_step_name(pipe) == "clf"


# ============================================================
# T-TUN02: param_grid プレフィックス
# ============================================================

class TestPrefixParamGrid:
    def test_basic_prefix(self):
        grid = {"alpha": [0.1, 1.0], "fit_intercept": [True, False]}
        prefixed = _prefix_param_grid(grid, "estimator")
        assert "estimator__alpha" in prefixed
        assert "estimator__fit_intercept" in prefixed
        assert prefixed["estimator__alpha"] == [0.1, 1.0]

    def test_already_prefixed(self):
        """既にプレフィックス付きのキーは二重付与しない"""
        grid = {"estimator__alpha": [0.1, 1.0]}
        prefixed = _prefix_param_grid(grid, "estimator")
        assert "estimator__alpha" in prefixed
        assert "estimator__estimator__alpha" not in prefixed

    def test_nested_key_preserved(self):
        """他の__を含むキーもskip"""
        grid = {"model__subsystem__lr": [0.01, 0.1]}
        prefixed = _prefix_param_grid(grid, "estimator")
        assert "model__subsystem__lr" in prefixed

    def test_empty_grid(self):
        assert _prefix_param_grid({}, "estimator") == {}


# ============================================================
# T-TUN03: プレフィックス除去
# ============================================================

class TestStripPrefix:
    def test_basic_strip(self):
        params = {"estimator__alpha": 0.5, "estimator__fit_intercept": True}
        stripped = _strip_prefix_from_params(params, "estimator")
        assert stripped == {"alpha": 0.5, "fit_intercept": True}

    def test_no_prefix(self):
        params = {"alpha": 0.5}
        stripped = _strip_prefix_from_params(params, "estimator")
        assert stripped == {"alpha": 0.5}

    def test_mixed(self):
        params = {"estimator__alpha": 0.5, "other_param": 42}
        stripped = _strip_prefix_from_params(params, "estimator")
        assert stripped == {"alpha": 0.5, "other_param": 42}


# ============================================================
# T-TUN04: Pipeline + GridSearchCV
# ============================================================

class TestTuneWithPipeline:
    @pytest.fixture
    def data(self):
        X, y = make_regression(n_samples=50, n_features=5, random_state=42)
        return X, y

    def test_grid_with_pipeline(self, data):
        """Pipeline(StandardScaler, Ridge) でGridSearchがプレフィックス付きで動作"""
        X, y = data
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("estimator", Ridge()),
        ])
        config = TunerConfig(
            method="grid",
            param_grid={"alpha": [0.1, 1.0, 10.0]},  # プレフィックスなしで渡す
            cv=3,
            scoring="neg_mean_squared_error",
        )
        result = tune(pipe, X, y, config)

        assert "best_params" in result
        assert "best_score" in result
        assert "best_estimator" in result
        # best_paramsにはプレフィックスなしの "alpha" が返る
        assert "alpha" in result["best_params"]
        assert "estimator__alpha" not in result["best_params"]
        assert result["best_params"]["alpha"] in [0.1, 1.0, 10.0]

    def test_grid_with_bare_estimator(self, data):
        """直接estimatorの場合もそのまま動作"""
        X, y = data
        config = TunerConfig(
            method="grid",
            param_grid={"alpha": [0.1, 1.0, 10.0]},
            cv=3,
            scoring="neg_mean_squared_error",
        )
        result = tune(Ridge(), X, y, config)
        assert "alpha" in result["best_params"]

    def test_random_with_pipeline(self, data):
        """Pipeline + RandomizedSearchCV"""
        X, y = data
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("estimator", Ridge()),
        ])
        config = TunerConfig(
            method="random",
            param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            n_iter=3,
            cv=3,
            scoring="neg_mean_squared_error",
        )
        result = tune(pipe, X, y, config)
        assert "alpha" in result["best_params"]
        assert "estimator__alpha" not in result["best_params"]

    def test_multiple_params_pipeline(self, data):
        """複数パラメータのPipeline GridSearch"""
        X, y = data
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("estimator", Ridge()),
        ])
        config = TunerConfig(
            method="grid",
            param_grid={
                "alpha": [0.1, 1.0],
                "fit_intercept": [True, False],
            },
            cv=3,
            scoring="neg_mean_squared_error",
        )
        result = tune(pipe, X, y, config)
        assert "alpha" in result["best_params"]
        assert "fit_intercept" in result["best_params"]
