"""
backend/data/benchmark.py

モデルのベンチマーク・評価モジュール。
複数モデルを比較し、詳細な評価指標を生成する。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import learning_curve

logger = logging.getLogger(__name__)


# ============================================================
# データクラス
# ============================================================

@dataclass
class ModelScore:
    """1モデルの評価結果。"""
    model_key: str
    task: str  # "regression" | "classification"
    # 共通
    train_time: float = 0.0
    # 回帰指標
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None
    # 分類指標
    accuracy: float | None = None
    f1_weighted: float | None = None
    roc_auc: float | None = None
    # CV指標（オプション）
    cv_mean: float | None = None
    cv_std: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class BenchmarkResult:
    """複数モデルのベンチマーク結果。"""
    task: str
    scores: list[ModelScore] = field(default_factory=list)
    scoring: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        """スコア一覧をDataFrameで返す。"""
        rows = [s.to_dict() for s in self.scores]
        return pd.DataFrame(rows).set_index("model_key")

    @property
    def best(self) -> ModelScore | None:
        """最良スコアのModelScoreを返す（回帰: R²最大 / 分類: F1最大）."""
        if not self.scores:
            return None
        if self.task == "regression":
            valid = [s for s in self.scores if s.r2 is not None]
            return max(valid, key=lambda s: s.r2) if valid else None  # type: ignore[arg-type]
        else:
            valid = [s for s in self.scores if s.f1_weighted is not None]
            return max(valid, key=lambda s: s.f1_weighted) if valid else None  # type: ignore[arg-type]


# ============================================================
# 評価関数
# ============================================================

def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_key: str = "model",
    train_time: float = 0.0,
    cv_mean: float | None = None,
    cv_std: float | None = None,
) -> ModelScore:
    """
    回帰タスクの評価指標を計算して ModelScore を返す。

    Implements: §3.7 評価モジュール

    Args:
        y_true: 実測値
        y_pred: 予測値
        model_key: モデル識別子
        train_time: 学習時間 (秒)
        cv_mean: CV平均スコア
        cv_std: CV標準偏差

    Returns:
        ModelScore インスタンス
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    logger.info(f"[evaluate_regression] {model_key}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    return ModelScore(
        model_key=model_key, task="regression",
        train_time=train_time,
        rmse=rmse, mae=mae, r2=r2,
        cv_mean=cv_mean, cv_std=cv_std,
    )


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    model_key: str = "model",
    train_time: float = 0.0,
    cv_mean: float | None = None,
    cv_std: float | None = None,
) -> ModelScore:
    """
    分類タスクの評価指標を計算して ModelScore を返す。

    Implements: §3.7 評価モジュール

    Args:
        y_true: 実測クラス
        y_pred: 予測クラス
        y_prob: クラス確率（ROC-AUC計算用、任意）
        model_key: モデル識別子
        train_time: 学習時間 (秒)

    Returns:
        ModelScore インスタンス
    """
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    roc = None
    if y_prob is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                roc = float(roc_auc_score(y_true, prob))
            else:
                roc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        except Exception as e:
            logger.debug(f"ROC-AUC calculation failed: {e}")
            roc = None
    logger.info(f"[evaluate_classification] {model_key}: Acc={acc:.4f}, F1={f1:.4f}")
    return ModelScore(
        model_key=model_key, task="classification",
        train_time=train_time,
        accuracy=acc, f1_weighted=f1, roc_auc=roc,
        cv_mean=cv_mean, cv_std=cv_std,
    )


def compute_learning_curve(
    estimator: BaseEstimator,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    scoring: str = "r2",
    cv: int = 5,
    n_points: int = 8,
    n_jobs: int = -1,
) -> dict[str, np.ndarray]:
    """
    学習曲線データを計算する。

    Args:
        estimator: sklearn互換モデル
        X: 特徴量
        y: 目的変数
        scoring: スコアリング指標
        cv: CV分割数
        n_points: 学習曲線の点数
        n_jobs: 並列数

    Returns:
        {train_sizes, train_scores_mean, train_scores_std,
         val_scores_mean, val_scores_std}
    """
    len(y) if hasattr(y, "__len__") else X.shape[0]
    train_sizes_frac = np.linspace(0.1, 1.0, n_points)

    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=train_sizes_frac,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        error_score="raise",
    )
    return {
        "train_sizes": train_sizes,
        "train_scores_mean": train_scores.mean(axis=1),
        "train_scores_std": train_scores.std(axis=1),
        "val_scores_mean": val_scores.mean(axis=1),
        "val_scores_std": val_scores.std(axis=1),
    }


def benchmark_models(
    models: dict[str, BaseEstimator],
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    task: str = "regression",
    scoring: str = "r2",
) -> BenchmarkResult:
    """
    複数のモデルをベンチマーク比較する。

    Args:
        models: {モデル名: fitされたestimator} の辞書
        X_train, y_train: 学習データ
        X_test, y_test: テストデータ
        task: "regression" | "classification"
        scoring: スコアリング指標 (ログ用)

    Returns:
        BenchmarkResult
    """
    result = BenchmarkResult(task=task, scoring=scoring)

    for key, model in models.items():
        t0 = time.time()
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            logger.warning(f"[benchmark] {key} fit error: {e}")
            continue
        elapsed = time.time() - t0

        y_pred = model.predict(X_test)
        y_prob = None
        if task == "classification" and hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
            except Exception as e:
                logger.debug(f"predict_proba failed for {key}: {e}")
                y_prob = None

        if task == "regression":
            score = evaluate_regression(
                np.asarray(y_test), y_pred,
                model_key=key, train_time=elapsed,
            )
        else:
            score = evaluate_classification(
                np.asarray(y_test), y_pred, y_prob,
                model_key=key, train_time=elapsed,
            )
        result.scores.append(score)
        logger.info(f"[benchmark] {key} done in {elapsed:.2f}s")

    return result
