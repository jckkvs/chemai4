"""
tests/test_factory_comprehensive.py

factory.py の包括テスト。
get_model / list_models / get_default_automl_models / get_model_registry /
_get_registry を網羅。sklearn標準モデル + フルスクラッチモデルをテスト。
"""
from __future__ import annotations

import pytest

from backend.models.factory import (
    get_model,
    list_models,
    get_default_automl_models,
    get_model_registry,
    _get_registry,
)


class TestGetModel:
    def test_ridge(self):
        m = get_model("ridge", task="regression")
        assert hasattr(m, "fit")
        assert hasattr(m, "predict")

    def test_rf(self):
        m = get_model("rf", task="regression")
        assert hasattr(m, "fit")

    def test_rf_classifier(self):
        m = get_model("rf_c", task="classification")
        assert hasattr(m, "predict_proba")

    def test_logistic(self):
        m = get_model("logistic", task="classification")
        assert hasattr(m, "fit")

    def test_svr(self):
        m = get_model("svr_rbf", task="regression")
        assert hasattr(m, "fit")

    def test_knn(self):
        m = get_model("knn", task="regression")
        assert hasattr(m, "fit")

    def test_dt(self):
        m = get_model("dt", task="regression")
        assert hasattr(m, "fit")

    def test_gbm(self):
        m = get_model("gbm", task="regression")
        assert hasattr(m, "fit")

    def test_hgbm(self):
        m = get_model("hgbm", task="regression")
        assert hasattr(m, "fit")

    def test_mlp(self):
        m = get_model("mlp", task="regression")
        assert hasattr(m, "fit")

    def test_gp(self):
        m = get_model("gp", task="regression")
        assert hasattr(m, "fit")

    def test_pls(self):
        m = get_model("pls", task="regression")
        assert hasattr(m, "fit")

    def test_override_params(self):
        m = get_model("ridge", task="regression", alpha=0.5)
        assert m.alpha == 0.5

    def test_unknown_key(self):
        with pytest.raises(ValueError, match="未知"):
            get_model("unknown_model", task="regression")

    def test_unknown_task(self):
        with pytest.raises(ValueError, match="未知のタスク"):
            get_model("ridge", task="unknown_task")


class TestListModels:
    def test_regression(self):
        models = list_models(task="regression")
        assert len(models) > 10
        keys = [m["key"] for m in models]
        assert "ridge" in keys

    def test_classification(self):
        models = list_models(task="classification")
        assert len(models) > 5
        keys = [m["key"] for m in models]
        assert "logistic" in keys

    def test_with_tags(self):
        models = list_models(task="regression", tags=["linear"])
        assert all(
            "linear" in m["tags"] for m in models
        )

    def test_available_only(self):
        all_models = list_models(task="regression", available_only=False)
        avail_models = list_models(task="regression", available_only=True)
        assert len(all_models) >= len(avail_models)


class TestDefaultAutoml:
    def test_regression(self):
        defaults = get_default_automl_models("regression")
        assert isinstance(defaults, list)
        assert len(defaults) > 0

    def test_classification(self):
        defaults = get_default_automl_models("classification")
        assert isinstance(defaults, list)
        assert len(defaults) > 0


class TestGetModelRegistry:
    def test_regression(self):
        reg = get_model_registry("regression")
        assert isinstance(reg, dict)
        assert "ridge" in reg

    def test_classification(self):
        reg = get_model_registry("classification")
        assert isinstance(reg, dict)
        assert "logistic" in reg


class TestGetRegistry:
    def test_regression(self):
        reg = _get_registry("regression")
        assert "ridge" in reg

    def test_classification(self):
        reg = _get_registry("classification")
        assert "logistic" in reg

    def test_unknown(self):
        with pytest.raises(ValueError, match="未知のタスク"):
            _get_registry("unknown")
