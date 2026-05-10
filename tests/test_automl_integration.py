# -*- coding: utf-8 -*-
"""
tests/test_automl_integration.py

AutoMLEngine の統合テスト。
ダミーデータでのE2Eパイプライン（型判定→前処理→CV→最良モデル選択）を検証。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.models.automl import AutoMLEngine, AutoMLResult


# ═══════════════════════════════════════════════════════════════════
# テストデータ
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def regression_df():
    """回帰用のシンプルなDataFrame"""
    np.random.seed(42)
    n = 80
    return pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.exponential(2, n),
        "cat": np.random.choice(["A", "B", "C"], n),
        "target": np.random.randn(n),
    })


@pytest.fixture
def classification_df():
    """分類用のシンプルなDataFrame"""
    np.random.seed(42)
    n = 80
    return pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "target": np.random.randint(0, 2, n),
    })


# ═══════════════════════════════════════════════════════════════════
# AutoMLEngine 初期化テスト
# ═══════════════════════════════════════════════════════════════════

class TestAutoMLInit:

    def test_default_init(self):
        engine = AutoMLEngine()
        assert engine.task == "auto"
        assert engine.cv_folds >= 2

    def test_custom_init(self):
        engine = AutoMLEngine(task="regression", cv_folds=3,
                              model_keys=["ridge", "lasso"])
        assert engine.task == "regression"
        assert engine.cv_folds == 3


# ═══════════════════════════════════════════════════════════════════
# AutoMLEngine.run E2E（回帰）
# ═══════════════════════════════════════════════════════════════════

class TestAutoMLRegression:

    def test_basic_run(self, regression_df):
        """基本的なAutoML回帰実行"""
        engine = AutoMLEngine(
            task="regression", cv_folds=2,
            model_keys=["ridge", "lasso"]
        )
        result = engine.run(regression_df, target_col="target")
        assert isinstance(result, AutoMLResult)
        assert result.task == "regression"
        assert result.best_model_key in ["ridge", "lasso"]
        assert result.best_pipeline is not None
        assert isinstance(result.best_score, float)

    def test_model_scores_populated(self, regression_df):
        engine = AutoMLEngine(
            task="regression", cv_folds=2,
            model_keys=["ridge"]
        )
        result = engine.run(regression_df, target_col="target")
        assert "ridge" in result.model_scores
        assert isinstance(result.model_scores["ridge"], float)

    def test_elapsed_time(self, regression_df):
        engine = AutoMLEngine(
            task="regression", cv_folds=2,
            model_keys=["ridge"]
        )
        result = engine.run(regression_df, target_col="target")
        assert result.elapsed_seconds > 0

    def test_progress_callback(self, regression_df):
        """コールバックが呼ばれること"""
        calls = []
        def cb(step, total, msg):
            calls.append((step, total, msg))

        engine = AutoMLEngine(
            task="regression", cv_folds=2,
            model_keys=["ridge"],
            progress_callback=cb
        )
        engine.run(regression_df, target_col="target")
        assert len(calls) > 0


# ═══════════════════════════════════════════════════════════════════
# AutoMLEngine.run E2E（分類）
# ═══════════════════════════════════════════════════════════════════

class TestAutoMLClassification:

    def test_basic_classification(self, classification_df):
        engine = AutoMLEngine(
            task="classification", cv_folds=2,
            model_keys=["dt_c"]
        )
        result = engine.run(classification_df, target_col="target")
        assert result.task == "classification"

    def test_auto_task_detection(self, classification_df):
        """task='auto' で分類が自動検出される"""
        engine = AutoMLEngine(
            task="auto", cv_folds=2,
            model_keys=["dt_c"]
        )
        result = engine.run(classification_df, target_col="target")
        assert result.task == "classification"


# ═══════════════════════════════════════════════════════════════════
# AutoMLResult
# ═══════════════════════════════════════════════════════════════════

class TestAutoMLResult:

    def test_result_fields(self, regression_df):
        engine = AutoMLEngine(
            task="regression", cv_folds=2,
            model_keys=["ridge"]
        )
        result = engine.run(regression_df, target_col="target")
        assert hasattr(result, "detection_result")
        assert hasattr(result, "model_details")
        assert hasattr(result, "warnings")
        assert isinstance(result.warnings, list)
