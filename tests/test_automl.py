# -*- coding: utf-8 -*-
"""
tests/test_automl.py

AutoML エンジンのテストスイート。
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from backend.models.automl import AutoMLEngine, AutoMLResult
from backend.pipeline.column_selector import ColumnMeta

@pytest.fixture
def df_regression():
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(50) * 0.1
    df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    df["target"] = y
    return df

@pytest.fixture
def df_classification():
    np.random.seed(42)
    X = np.random.randn(50, 3)
    score = X[:, 0] * 2 + X[:, 1] - X[:, 2]
    y = (score > 0).astype(int)
    df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    df["target"] = y
    df["group"] = np.repeat(np.arange(10), 5)
    return df

class TestAutoMLEngine:
    def test_run_regression(self, df_regression):
        engine = AutoMLEngine(
            task="regression",
            cv_folds=2,
            model_keys=["ridge", "lasso"],
            timeout_seconds=60
        )
        
        result = engine.run(df_regression, target_col="target")
        assert isinstance(result, AutoMLResult)
        assert result.task == "regression"
        assert result.best_model_key in ["ridge", "lasso"]
        assert "ridge" in result.model_scores
        assert "lasso" in result.model_scores
        assert result.processed_X is not None
        assert result.oof_predictions is not None


    def test_run_classification_group_kfold(self, df_classification):
        engine = AutoMLEngine(
            task="classification",
            cv_folds=3,
            cv_key="group_kfold",
            model_keys=["logistic"],
            timeout_seconds=60
        )
        
        result = engine.run(
            df_classification, 
            target_col="target", 
            group_col="group"
        )
        assert result.task == "classification"
        assert result.best_model_key == "logistic"
        assert result.oof_predictions is not None
        assert "group" not in result.X_train.columns

    @patch("backend.models.automl.SmilesDescriptorTransformer")
    def test_run_with_smiles(self, mock_transformer_cls, df_regression):
        # SMILESを含める
        df = df_regression.copy()
        df["smiles"] = ["C"] * 50
        
        # モック設定: clone()時の再帰エラーを防ぐため設定
        from sklearn.preprocessing import FunctionTransformer
        mock_instance = FunctionTransformer(func=lambda x: df_regression.drop(columns=["target"]))
        mock_transformer_cls.return_value = mock_instance
        
        engine = AutoMLEngine(
            task="regression",
            cv_folds=2,
            model_keys=["ridge"]
        )
        
        result = engine.run(df, target_col="target", smiles_col="smiles")
        assert result.best_model_key == "ridge"
        assert result.processed_X is not None

    def test_run_with_monotonic_constraints(self, df_regression):
        # 単調性制約を明示的に指定
        constraints_dict = {"f1": 1, "f2": -1}
        engine = AutoMLEngine(
            task="regression",
            cv_folds=2,
            model_keys=["ridge", "lgbm"],
            monotonic_constraints_dict=constraints_dict,
        )
        result = engine.run(df_regression, target_col="target")
        
        assert result.best_model_key in ["ridge", "lgbm"]

    def test_auto_task_inference(self, df_classification):
        engine = AutoMLEngine(
            task="auto",
            cv_folds=2,
            model_keys=["dt_c"]
        )
        result = engine.run(df_classification, target_col="target")
        assert result.task == "classification"
        assert result.best_model_key == "dt_c"

    def test_all_models_fail(self, df_regression):
        engine = AutoMLEngine(
            task="regression",
            cv_folds=2,
            model_keys=["non_existent_model"]
        )
        with pytest.raises(RuntimeError, match="全モデルの学習に失敗しました"):
            engine.run(df_regression, target_col="target")

    def test_no_features_left(self, df_regression):
        engine = AutoMLEngine(
            task="regression",
            cv_folds=2,
            model_keys=["ridge"]
        )
        
        # 特徴量がゼロの状態で学習を実行（エラーになるか）
        df = df_regression[["target"]]
        with pytest.raises(ValueError, match="学習に使用できる特徴量がありません"):
            engine.run(df, target_col="target")
