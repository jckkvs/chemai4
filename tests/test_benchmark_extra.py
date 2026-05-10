"""
tests/test_benchmark_extra.py

benchmark.py のカバレッジ改善テスト。
ModelScore, BenchmarkResult, evaluate_regression, evaluate_classification,
benchmark_models を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression

from backend.data.benchmark import (
    ModelScore,
    BenchmarkResult,
    evaluate_regression,
    evaluate_classification,
    benchmark_models,
)


# ============================================================
# ModelScore
# ============================================================

class TestModelScore:
    def test_to_dict_regression(self):
        ms = ModelScore(model_key="ridge", task="regression",
                        rmse=0.5, mae=0.3, r2=0.9)
        d = ms.to_dict()
        assert d["rmse"] == 0.5
        assert d["r2"] == 0.9
        assert "accuracy" not in d  # None fields excluded

    def test_to_dict_classification(self):
        ms = ModelScore(model_key="lr", task="classification",
                        accuracy=0.95, f1_weighted=0.94)
        d = ms.to_dict()
        assert d["accuracy"] == 0.95
        assert "rmse" not in d


# ============================================================
# BenchmarkResult
# ============================================================

class TestBenchmarkResult:
    def test_to_dataframe(self):
        s1 = ModelScore(model_key="m1", task="regression", r2=0.9)
        s2 = ModelScore(model_key="m2", task="regression", r2=0.85)
        br = BenchmarkResult(task="regression", scores=[s1, s2])
        df = br.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_best_regression(self):
        s1 = ModelScore(model_key="m1", task="regression", r2=0.9)
        s2 = ModelScore(model_key="m2", task="regression", r2=0.95)
        br = BenchmarkResult(task="regression", scores=[s1, s2])
        best = br.best
        assert best is not None
        assert best.model_key == "m2"

    def test_best_classification(self):
        s1 = ModelScore(model_key="m1", task="classification", f1_weighted=0.8)
        s2 = ModelScore(model_key="m2", task="classification", f1_weighted=0.9)
        br = BenchmarkResult(task="classification", scores=[s1, s2])
        best = br.best
        assert best is not None
        assert best.model_key == "m2"

    def test_best_empty(self):
        br = BenchmarkResult(task="regression")
        assert br.best is None


# ============================================================
# evaluate_regression
# ============================================================

class TestEvaluateRegression:
    def test_basic(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.2, 2.9, 3.8])
        ms = evaluate_regression(y_true, y_pred, model_key="test")
        assert ms.task == "regression"
        assert ms.rmse is not None
        assert ms.mae is not None
        assert ms.r2 is not None
        assert ms.r2 > 0.9

    def test_with_cv(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        ms = evaluate_regression(y_true, y_pred, cv_mean=0.95, cv_std=0.02)
        assert ms.cv_mean == 0.95
        assert ms.cv_std == 0.02


# ============================================================
# evaluate_classification
# ============================================================

class TestEvaluateClassification:
    def test_basic(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0])
        ms = evaluate_classification(y_true, y_pred, model_key="test")
        assert ms.task == "classification"
        assert ms.accuracy is not None
        assert ms.f1_weighted is not None

    def test_with_proba(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.8, 0.2]])
        ms = evaluate_classification(y_true, y_pred, y_prob)
        assert ms.roc_auc is not None

    def test_multiclass_proba(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ])
        ms = evaluate_classification(y_true, y_pred, y_prob)
        assert ms.roc_auc is not None


# ============================================================
# benchmark_models
# ============================================================

class TestBenchmarkModels:
    def test_regression(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        y = X[:, 0] * 2 + rng.randn(40) * 0.1
        models = {"ridge": Ridge()}
        result = benchmark_models(
            models, X[:30], y[:30], X[30:], y[30:],
            task="regression",
        )
        assert len(result.scores) == 1
        assert result.scores[0].r2 is not None

    def test_classification(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        y = (X[:, 0] > 0).astype(int)
        models = {"lr": LogisticRegression()}
        result = benchmark_models(
            models, X[:30], y[:30], X[30:], y[30:],
            task="classification",
        )
        assert len(result.scores) == 1
        assert result.scores[0].accuracy is not None

    def test_multiple_models(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        y = X[:, 0] * 2
        models = {"r1": Ridge(alpha=0.1), "r2": Ridge(alpha=1.0)}
        result = benchmark_models(
            models, X[:30], y[:30], X[30:], y[30:],
        )
        assert len(result.scores) == 2
        best = result.best
        assert best is not None
