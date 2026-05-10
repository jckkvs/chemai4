"""
tests/test_feature_selector_comprehensive.py

FeatureSelector / FeatureSelectorConfig の包括テスト。
Lasso/Ridge/RF/KBest/Percentile/パススルー/フォールバックを網羅。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression, make_classification

from backend.pipeline.feature_selector import FeatureSelectorConfig, FeatureSelector


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=60, n_features=10, n_informative=5, random_state=42)
    return X, y


@pytest.fixture
def cls_data():
    X, y = make_classification(n_samples=60, n_features=10, n_informative=5,
                                n_redundant=0, random_state=42)
    return X, y


class TestFeatureSelectorConfig:
    def test_defaults(self):
        cfg = FeatureSelectorConfig()
        assert cfg.method == "none"
        assert cfg.task == "regression"

    def test_custom(self):
        cfg = FeatureSelectorConfig(method="lasso", task="regression")
        assert cfg.method == "lasso"


class TestFeatureSelectorNone:
    def test_passthrough(self, reg_data):
        X, y = reg_data
        sel = FeatureSelector(FeatureSelectorConfig(method="none"))
        sel.fit(X, y)
        result = sel.transform(X)
        assert result.shape == X.shape

    def test_feature_names_passthrough(self):
        df = pd.DataFrame(np.random.randn(20, 5), columns=[f"f{i}" for i in range(5)])
        sel = FeatureSelector(FeatureSelectorConfig(method="none"))
        sel.fit(df)
        names = sel.get_feature_names_out()
        assert list(names) == [f"f{i}" for i in range(5)]


class TestFeatureSelectorLasso:
    def test_lasso(self, reg_data):
        X, y = reg_data
        cfg = FeatureSelectorConfig(method="lasso", task="regression")
        sel = FeatureSelector(config=cfg)
        sel.fit(X, y)
        result = sel.transform(X)
        assert result.shape[1] <= X.shape[1]
        assert sel.support_mask is not None


class TestFeatureSelectorRidge:
    def test_ridge(self, reg_data):
        X, y = reg_data
        cfg = FeatureSelectorConfig(method="ridge", task="regression")
        sel = FeatureSelector(config=cfg)
        sel.fit(X, y)
        result = sel.transform(X)
        assert result.shape[1] <= X.shape[1]


class TestFeatureSelectorRF:
    def test_rfr(self, reg_data):
        X, y = reg_data
        cfg = FeatureSelectorConfig(method="rfr", task="regression")
        sel = FeatureSelector(config=cfg)
        sel.fit(X, y)
        result = sel.transform(X)
        assert result.shape[1] <= X.shape[1]

    def test_rfc(self, cls_data):
        X, y = cls_data
        cfg = FeatureSelectorConfig(method="rfc", task="classification")
        sel = FeatureSelector(config=cfg)
        sel.fit(X, y)
        result = sel.transform(X)
        assert result.shape[1] <= X.shape[1]


class TestFeatureSelectorKBest:
    def test_select_kbest(self, reg_data):
        X, y = reg_data
        cfg = FeatureSelectorConfig(method="select_kbest", k=5, task="regression")
        sel = FeatureSelector(config=cfg)
        sel.fit(X, y)
        result = sel.transform(X)
        assert result.shape[1] == 5

    def test_select_kbest_classification(self, cls_data):
        X, y = cls_data
        cfg = FeatureSelectorConfig(
            method="select_kbest", k=5, task="classification",
            score_func="f_regression"  # should auto-correct to f_classif
        )
        sel = FeatureSelector(config=cfg)
        sel.fit(X, y)
        assert sel.transform(X).shape[1] == 5


class TestFeatureSelectorPercentile:
    def test_select_percentile(self, reg_data):
        X, y = reg_data
        cfg = FeatureSelectorConfig(method="select_percentile", percentile=50)
        sel = FeatureSelector(config=cfg)
        sel.fit(X, y)
        result = sel.transform(X)
        assert result.shape[1] == 5  # 50% of 10

    def test_feature_names_out(self, reg_data):
        X, y = reg_data
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        cfg = FeatureSelectorConfig(method="select_percentile", percentile=30)
        sel = FeatureSelector(config=cfg)
        sel.fit(df, y)
        names = sel.get_feature_names_out()
        assert len(names) == 3  # 30% of 10


class TestFeatureSelectorUnknown:
    def test_unknown_method_fallback(self, reg_data):
        X, y = reg_data
        cfg = FeatureSelectorConfig(method="nonexistent_method")
        sel = FeatureSelector(config=cfg)
        sel.fit(X, y)
        # Falls back to RF SelectFromModel
        result = sel.transform(X)
        assert result.shape[1] <= X.shape[1]
