"""
tests/test_benchmark.py

backend/data/benchmark.py のユニットテスト。
evaluate_regression, evaluate_classification, compute_learning_curve,
benchmark_models を検証する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from backend.data.benchmark import (
    BenchmarkResult,
    ModelScore,
    benchmark_models,
    compute_learning_curve,
    evaluate_classification,
    evaluate_regression,
)


# ============================================================
# フィクスチャ
# ============================================================

@pytest.fixture
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(0)
    X = np.random.randn(100, 4)
    y = X[:, 0] * 2 + X[:, 1] + np.random.randn(100) * 0.1
    return X, y


@pytest.fixture
def cls_data() -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(0)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


# ============================================================
# T-BM-001: ModelScore
# ============================================================

class TestModelScore:
    """T-BM-001: ModelScore データクラスのテスト。"""

    def test_to_dict_filters_none(self) -> None:
        """Noneフィールドがto_dict()で除外されること。(T-BM-001-01)"""
        s = ModelScore(model_key="lr", task="regression", rmse=0.5, r2=0.9)
        d = s.to_dict()
        assert "mae" not in d
        assert d["rmse"] == 0.5
        assert d["r2"] == 0.9

    def test_task_field(self) -> None:
        """taskフィールドが正しく設定されること。(T-BM-001-02)"""
        s = ModelScore(model_key="lr", task="classification", accuracy=0.95)
        assert s.task == "classification"


# ============================================================
# T-BM-002: BenchmarkResult
# ============================================================

class TestBenchmarkResult:
    """T-BM-002: BenchmarkResult データクラスのテスト。"""

    def test_to_dataframe(self) -> None:
        """to_dataframe()がDataFrameを返すこと。(T-BM-002-01)"""
        br = BenchmarkResult(task="regression")
        br.scores = [
            ModelScore(model_key="a", task="regression", r2=0.9),
            ModelScore(model_key="b", task="regression", r2=0.7),
        ]
        df = br.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "model_key"
        assert len(df) == 2

    def test_best_regression(self) -> None:
        """回帰タスクでbest()がR²最大モデルを返すこと。(T-BM-002-02)"""
        br = BenchmarkResult(task="regression")
        br.scores = [
            ModelScore(model_key="a", task="regression", r2=0.7),
            ModelScore(model_key="b", task="regression", r2=0.95),
        ]
        best = br.best
        assert best is not None
        assert best.model_key == "b"

    def test_best_classification(self) -> None:
        """分類タスクでbest()がF1最大モデルを返すこと。(T-BM-002-03)"""
        br = BenchmarkResult(task="classification")
        br.scores = [
            ModelScore(model_key="a", task="classification", f1_weighted=0.8),
            ModelScore(model_key="b", task="classification", f1_weighted=0.92),
        ]
        best = br.best
        assert best is not None
        assert best.model_key == "b"

    def test_best_empty(self) -> None:
        """スコアが空のときbest()がNoneを返すこと。(T-BM-002-04)"""
        br = BenchmarkResult(task="regression")
        assert br.best is None


# ============================================================
# T-BM-003: evaluate_regression
# ============================================================

class TestEvaluateRegression:
    """T-BM-003: evaluate_regression のテスト。"""

    def test_returns_model_score(self, reg_data: tuple) -> None:
        """ModelScoreが返ること。(T-BM-003-01)"""
        X, y = reg_data
        y_pred = y + np.random.randn(len(y)) * 0.5
        score = evaluate_regression(y, y_pred, model_key="test")
        assert isinstance(score, ModelScore)
        assert score.task == "regression"

    def test_rmse_positive(self, reg_data: tuple) -> None:
        """RMSEが正の値であること。(T-BM-003-02)"""
        X, y = reg_data
        y_pred = y + 0.5
        score = evaluate_regression(y, y_pred)
        assert score.rmse is not None
        assert score.rmse > 0

    def test_r2_perfect(self, reg_data: tuple) -> None:
        """完全予測のときR²が1.0になること。(T-BM-003-03)"""
        X, y = reg_data
        score = evaluate_regression(y, y.copy())
        assert score.r2 is not None
        assert abs(score.r2 - 1.0) < 1e-8

    def test_cv_fields(self, reg_data: tuple) -> None:
        """cv_mean/cv_stdが正しく格納されること。(T-BM-003-04)"""
        X, y = reg_data
        score = evaluate_regression(y, y, cv_mean=0.85, cv_std=0.05)
        assert score.cv_mean == 0.85
        assert score.cv_std == 0.05

    def test_mae_positive(self, reg_data: tuple) -> None:
        """MAEが正の値であること。(T-BM-003-05)"""
        X, y = reg_data
        y_pred = y + 1.0
        score = evaluate_regression(y, y_pred)
        assert score.mae is not None
        assert score.mae > 0


# ============================================================
# T-BM-004: evaluate_classification
# ============================================================

class TestEvaluateClassification:
    """T-BM-004: evaluate_classification のテスト。"""

    def test_returns_model_score(self, cls_data: tuple) -> None:
        """ModelScoreが返ること。(T-BM-004-01)"""
        X, y = cls_data
        score = evaluate_classification(y, y)
        assert isinstance(score, ModelScore)
        assert score.task == "classification"

    def test_perfect_accuracy(self, cls_data: tuple) -> None:
        """完全予測でaccuracy=1.0になること。(T-BM-004-02)"""
        X, y = cls_data
        score = evaluate_classification(y, y)
        assert score.accuracy == pytest.approx(1.0)

    def test_roc_auc_with_proba(self, cls_data: tuple) -> None:
        """確率配列が渡されたときROC-AUCが計算されること。(T-BM-004-03)"""
        X, y = cls_data
        model = LogisticRegression(max_iter=200, solver="liblinear")
        model.fit(X, y)
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
        score = evaluate_classification(y, y_pred, y_prob=y_prob)
        assert score.roc_auc is not None
        assert 0 <= score.roc_auc <= 1

    def test_no_prob_roc_none(self, cls_data: tuple) -> None:
        """確率配列なしのときroc_aucがNoneになること。(T-BM-004-04)"""
        X, y = cls_data
        score = evaluate_classification(y, y, y_prob=None)
        assert score.roc_auc is None


# ============================================================
# T-BM-005: compute_learning_curve
# ============================================================

class TestComputeLearningCurve:
    """T-BM-005: compute_learning_curve のテスト。"""

    def test_returns_dict_with_keys(self, reg_data: tuple) -> None:
        """必要なキーを持つdictが返ること。(T-BM-005-01)"""
        X, y = reg_data
        model = LinearRegression()
        lc = compute_learning_curve(model, X, y, scoring="r2", cv=3, n_points=4)
        for key in ["train_sizes", "train_scores_mean", "val_scores_mean"]:
            assert key in lc

    def test_train_sizes_length(self, reg_data: tuple) -> None:
        """n_pointsの数だけtrain_sizesが返ること。(T-BM-005-02)"""
        X, y = reg_data
        model = LinearRegression()
        lc = compute_learning_curve(model, X, y, scoring="r2", cv=3, n_points=5)
        assert len(lc["train_sizes"]) == 5

    def test_scores_shape_match(self, reg_data: tuple) -> None:
        """train_scores_meanとval_scores_meanが同じ長さであること。(T-BM-005-03)"""
        X, y = reg_data
        model = LinearRegression()
        lc = compute_learning_curve(model, X, y, scoring="r2", cv=3, n_points=4)
        assert len(lc["train_scores_mean"]) == len(lc["val_scores_mean"])


# ============================================================
# T-BM-006: benchmark_models
# ============================================================

class TestBenchmarkModels:
    """T-BM-006: benchmark_models のテスト。"""

    def test_regression(self, reg_data: tuple) -> None:
        """回帰ベンチマークが正常実行できること。(T-BM-006-01)"""
        X, y = reg_data
        n = len(y)
        X_train, y_train = X[:80], y[:80]
        X_test, y_test = X[80:], y[80:]
        models = {
            "lr": LinearRegression(),
            "dt": DecisionTreeRegressor(random_state=0),
        }
        result = benchmark_models(models, X_train, y_train, X_test, y_test, task="regression")
        assert isinstance(result, BenchmarkResult)
        assert len(result.scores) == 2

    def test_classification(self, cls_data: tuple) -> None:
        """分類ベンチマークが正常実行できること。(T-BM-006-02)"""
        X, y = cls_data
        X_train, y_train = X[:80], y[:80]
        X_test, y_test = X[80:], y[80:]
        models = {
            "lr": LogisticRegression(max_iter=200, solver="liblinear"),
            "dt": DecisionTreeClassifier(random_state=0),
        }
        result = benchmark_models(
            models, X_train, y_train, X_test, y_test,
            task="classification"
        )
        assert len(result.scores) == 2
        assert all(s.accuracy is not None for s in result.scores)

    def test_best_model(self, reg_data: tuple) -> None:
        """benchmark_models後にbest()が正常に動作すること。(T-BM-006-03)"""
        X, y = reg_data
        X_train, y_train = X[:80], y[:80]
        X_test, y_test = X[80:], y[80:]
        models = {"lr": LinearRegression(), "dt": DecisionTreeRegressor(random_state=0)}
        result = benchmark_models(models, X_train, y_train, X_test, y_test, task="regression")
        best = result.best
        assert best is not None
        assert best.model_key in ["lr", "dt"]
