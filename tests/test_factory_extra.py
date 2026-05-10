"""
tests/test_factory_extra.py

factory.py のカバレッジ改善テスト。
get_model, list_models, get_default_automl_models, _get_registry を網羅。
"""
from __future__ import annotations

import pytest
from sklearn.linear_model import Ridge, Lasso

from backend.models.factory import (
    get_model,
    list_models,
    get_default_automl_models,
    _get_registry,
    _REGRESSION_REGISTRY,
    _CLASSIFICATION_REGISTRY,
)


# ============================================================
# _get_registry
# ============================================================

class TestGetRegistry:
    def test_regression(self):
        reg = _get_registry("regression")
        assert "ridge" in reg
        assert "rf" in reg

    def test_classification(self):
        reg = _get_registry("classification")
        assert "logistic" in reg
        assert "rf_c" in reg

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="未知のタスク"):
            _get_registry("unsupervised")


# ============================================================
# get_model
# ============================================================

class TestGetModel:
    def test_linear_regression(self):
        model = get_model("linear")
        assert hasattr(model, "fit")

    def test_ridge(self):
        model = get_model("ridge", alpha=2.0)
        assert model.alpha == 2.0

    def test_lasso(self):
        model = get_model("lasso")
        assert hasattr(model, "fit")

    def test_rf(self):
        model = get_model("rf")
        assert hasattr(model, "fit")

    def test_svr(self):
        model = get_model("svr_rbf")
        assert model.kernel == "rbf"

    def test_knn(self):
        model = get_model("knn", n_neighbors=3)
        assert model.n_neighbors == 3

    def test_classifier(self):
        model = get_model("logistic", task="classification")
        assert hasattr(model, "predict_proba")

    def test_unknown_key(self):
        with pytest.raises(ValueError, match="未知のモデルキー"):
            get_model("totally_nonexistent_model")

    def test_regression_default(self):
        """All always-available regression models can be instantiated."""
        for key, entry in _REGRESSION_REGISTRY.items():
            if entry.get("available", True):
                model = get_model(key)
                assert hasattr(model, "fit"), f"{key} has no fit method"

    def test_classification_default(self):
        """All always-available classification models can be instantiated."""
        for key, entry in _CLASSIFICATION_REGISTRY.items():
            if entry.get("available", True):
                model = get_model(key, task="classification")
                assert hasattr(model, "fit"), f"{key} has no fit method"


# ============================================================
# list_models
# ============================================================

class TestListModels:
    def test_regression_list(self):
        models = list_models("regression")
        assert len(models) > 0
        assert all("key" in m for m in models)

    def test_classification_list(self):
        models = list_models("classification")
        assert len(models) > 0

    def test_available_only(self):
        available = list_models("regression", available_only=True)
        all_models = list_models("regression", available_only=False)
        assert len(all_models) >= len(available)

    def test_filter_by_tags(self):
        linear = list_models("regression", tags=["linear"])
        assert all("linear" in m["tags"] for m in linear)

    def test_filter_ensemble(self):
        ensemble = list_models("regression", tags=["ensemble"])
        assert len(ensemble) > 0


# ============================================================
# get_default_automl_models
# ============================================================

class TestDefaultAutoMLModels:
    def test_regression_defaults(self):
        defaults = get_default_automl_models("regression")
        assert len(defaults) > 0
        assert "rf" in defaults

    def test_classification_defaults(self):
        defaults = get_default_automl_models("classification")
        assert len(defaults) > 0
        assert "rf_c" in defaults
