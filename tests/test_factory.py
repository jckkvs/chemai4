# -*- coding: utf-8 -*-
"""
tests/test_factory.py

factory.py（モデルファクトリー）のユニットテスト。

カバー対象:
  - get_model: 全レジストリキーのインスタンス生成
  - list_models: タスク/タグフィルタリング
  - get_default_automl_models: デフォルトモデルリスト
  - get_model_registry: レジストリ取得
  - エラーハンドリング（未知キー、利用不可ライブラリ）
"""
from __future__ import annotations

import pytest
from sklearn.base import BaseEstimator

from backend.models.factory import (
    get_model,
    list_models,
    get_default_automl_models,
    get_model_registry,
)


# ═══════════════════════════════════════════════════════════════════
# get_model テスト
# ═══════════════════════════════════════════════════════════════════

class TestGetModel:

    @pytest.mark.parametrize("key", [
        "linear", "ridge", "ridge_cv", "lasso", "lasso_cv",
        "elasticnet", "elasticnet_cv", "bayesian_ridge", "ard",
        "huber", "theilsen", "ransac",
        "svr_rbf", "svr_linear", "knn", "dt", "rf", "et",
        "gbm", "hgbm", "adaboost", "bagging", "pls",
    ])
    def test_regression_models(self, key):
        """全回帰モデルがインスタンス化できること"""
        model = get_model(key, task="regression")
        assert isinstance(model, BaseEstimator)

    @pytest.mark.parametrize("key", [
        "knn_c", "dt_c", "rf_c", "et_c",
        "gbm_c", "hgbm_c", "adaboost_c", "bagging_c",
        "gnb", "bnb",
    ])
    def test_classification_models(self, key):
        """全分類モデルがインスタンス化できること"""
        model = get_model(key, task="classification")
        assert isinstance(model, BaseEstimator)

    def test_override_params(self):
        model = get_model("ridge", task="regression", alpha=5.0)
        assert model.alpha == 5.0

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="未知のモデルキー"):
            get_model("nonexistent_model", task="regression")

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="未知のタスク"):
            get_model("ridge", task="unknown_task")


# ═══════════════════════════════════════════════════════════════════
# list_models テスト
# ═══════════════════════════════════════════════════════════════════

class TestListModels:

    def test_regression_list_not_empty(self):
        models = list_models(task="regression")
        assert len(models) > 10

    def test_classification_list_not_empty(self):
        models = list_models(task="classification")
        assert len(models) > 10

    def test_filter_by_tag(self):
        models = list_models(task="regression", tags=["linear"])
        for m in models:
            assert "linear" in m["tags"]

    def test_filter_ensemble(self):
        models = list_models(task="regression", tags=["ensemble"])
        for m in models:
            assert "ensemble" in m["tags"]

    def test_model_entry_has_required_fields(self):
        for m in list_models(task="regression"):
            assert "key" in m
            assert "name" in m
            assert "tags" in m
            assert "available" in m

    def test_available_only_filter(self):
        all_models = list_models(task="regression", available_only=False)
        avail_models = list_models(task="regression", available_only=True)
        assert len(all_models) >= len(avail_models)


# ═══════════════════════════════════════════════════════════════════
# get_default_automl_models テスト
# ═══════════════════════════════════════════════════════════════════

class TestGetDefaultAutoMLModels:

    def test_regression_defaults(self):
        defaults = get_default_automl_models(task="regression")
        assert isinstance(defaults, list)
        assert len(defaults) >= 3
        # ridge_cvが含まれるはず
        assert "ridge_cv" in defaults

    def test_classification_defaults(self):
        defaults = get_default_automl_models(task="classification")
        assert isinstance(defaults, list)
        assert len(defaults) >= 3

    def test_all_defaults_are_instantiable(self):
        for key in get_default_automl_models("regression"):
            model = get_model(key, task="regression")
            assert model is not None


# ═══════════════════════════════════════════════════════════════════
# get_model_registry テスト
# ═══════════════════════════════════════════════════════════════════

class TestGetModelRegistry:

    def test_regression_registry(self):
        reg = get_model_registry(task="regression")
        assert isinstance(reg, dict)
        assert "ridge" in reg

    def test_classification_registry(self):
        reg = get_model_registry(task="classification")
        assert isinstance(reg, dict)
        assert "dt_c" in reg

    def test_registry_entries_have_name(self):
        for key, entry in get_model_registry("regression").items():
            assert "name" in entry, f"key={key} missing 'name'"


# ═══════════════════════════════════════════════════════════════════
# オプショナルモデル（XGBoost / LightGBM 等）
# ═══════════════════════════════════════════════════════════════════

class TestOptionalModels:

    def test_xgb_if_available(self):
        try:
            model = get_model("xgb", task="regression")
            assert model is not None
        except ValueError:
            pytest.skip("xgboost not installed")

    def test_lgbm_if_available(self):
        try:
            model = get_model("lgbm", task="regression")
            assert model is not None
        except ValueError:
            pytest.skip("lightgbm not installed")

    def test_catboost_if_available(self):
        try:
            model = get_model("catboost", task="regression")
            assert model is not None
        except ValueError:
            pytest.skip("catboost not installed")

    def test_lineartree_if_available(self):
        try:
            model = get_model("lineartree", task="regression")
            assert model is not None
        except ValueError:
            pytest.skip("linear-tree not available")

    def test_rgf_if_available(self):
        try:
            model = get_model("rgf", task="regression")
            assert model is not None
        except ValueError:
            pytest.skip("rgf not available")
