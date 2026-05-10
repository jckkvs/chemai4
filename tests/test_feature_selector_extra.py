"""
tests/test_feature_selector_extra.py

feature_selector.py の低カバレッジ部分を補うテスト。
主に XGB, Lasso, Ridge, Percentile, KBest, フォールバック、
未知メソッド、分類タスク、get_feature_names_out を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.pipeline.feature_selector import (
    FeatureSelector,
    FeatureSelectorConfig,
)


# ============================================================
# テストデータ
# ============================================================

def _make_data(n: int = 100, p: int = 10, task: str = "regression"):
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
    if task == "regression":
        y = X["f0"] * 2 + rng.randn(n) * 0.1
    else:
        y = (X["f0"] > 0).astype(int)
    return X, y


# ============================================================
# method=none (パススルー)
# ============================================================

class TestFeatureSelectorNone:
    def test_none_passthrough(self):
        X, y = _make_data()
        cfg = FeatureSelectorConfig(method="none")
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape == X.shape

    def test_none_feature_names(self):
        X, y = _make_data()
        fs = FeatureSelector(config=FeatureSelectorConfig(method="none"))
        fs.fit(X, y)
        names = fs.get_feature_names_out()
        assert list(names) == list(X.columns)


# ============================================================
# method=lasso
# ============================================================

class TestLasso:
    def test_lasso_regression(self):
        X, y = _make_data()
        cfg = FeatureSelectorConfig(method="lasso", task="regression")
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] <= X.shape[1]
        assert fs.support_mask is not None


# ============================================================
# method=ridge
# ============================================================

class TestRidge:
    def test_ridge_regression(self):
        X, y = _make_data()
        cfg = FeatureSelectorConfig(method="ridge", task="regression")
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] <= X.shape[1]


# ============================================================
# method=rfr / rfc
# ============================================================

class TestRandomForest:
    def test_rfr(self):
        X, y = _make_data()
        cfg = FeatureSelectorConfig(method="rfr", task="regression")
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] <= X.shape[1]

    def test_rfc(self):
        X, y = _make_data(task="classification")
        cfg = FeatureSelectorConfig(method="rfc", task="classification")
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] <= X.shape[1]


# ============================================================
# method=xgb
# ============================================================

class TestXGBoost:
    def test_xgb_regression(self):
        X, y = _make_data()
        cfg = FeatureSelectorConfig(method="xgb", task="regression")
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] <= X.shape[1]

    def test_xgb_classification(self):
        X, y = _make_data(task="classification")
        cfg = FeatureSelectorConfig(method="xgb", task="classification")
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] <= X.shape[1]


# ============================================================
# method=select_percentile
# ============================================================

class TestSelectPercentile:
    def test_percentile_regression(self):
        X, y = _make_data()
        cfg = FeatureSelectorConfig(
            method="select_percentile", task="regression",
            percentile=50, score_func="f_regression",
        )
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] == 5  # 50% of 10

    def test_percentile_classification(self):
        X, y = _make_data(task="classification")
        cfg = FeatureSelectorConfig(
            method="select_percentile", task="classification",
            percentile=30, score_func="f_regression",  # auto-correct to f_classif
        )
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] == 3  # 30% of 10

    def test_percentile_mutual_info(self):
        X, y = _make_data(task="classification")
        cfg = FeatureSelectorConfig(
            method="select_percentile", task="classification",
            percentile=50, score_func="mutual_info_regression",
        )
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] == 5


# ============================================================
# method=select_kbest
# ============================================================

class TestSelectKBest:
    def test_kbest_regression(self):
        X, y = _make_data()
        cfg = FeatureSelectorConfig(
            method="select_kbest", task="regression", k=3,
        )
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] == 3

    def test_kbest_unknown_score_func(self):
        X, y = _make_data()
        cfg = FeatureSelectorConfig(
            method="select_kbest", task="regression", k=5,
            score_func="nonexistent_func",
        )
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] == 5


# ============================================================
# method=select_from_model (custom)
# ============================================================

class TestSelectFromModel:
    def test_sfm_with_estimator_key(self):
        X, y = _make_data()
        cfg = FeatureSelectorConfig(
            method="select_from_model", task="regression",
            estimator_key="rf",
        )
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] <= X.shape[1]

    def test_sfm_without_estimator_key(self):
        X, y = _make_data()
        cfg = FeatureSelectorConfig(
            method="select_from_model", task="regression",
        )
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] <= X.shape[1]

    def test_sfm_invalid_estimator_key(self):
        X, y = _make_data()
        cfg = FeatureSelectorConfig(
            method="select_from_model", task="regression",
            estimator_key="totally_invalid_model",
        )
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] <= X.shape[1]


# ============================================================
# 未知メソッド (フォールバック)
# ============================================================

class TestUnknownMethod:
    def test_unknown_method_uses_rf(self):
        X, y = _make_data()
        cfg = FeatureSelectorConfig(method="unknown_method")
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] <= X.shape[1]


# ============================================================
# ndarray入力 (DataFrame以外)
# ============================================================

class TestNdarrayInput:
    def test_ndarray_input(self):
        X, y = _make_data()
        cfg = FeatureSelectorConfig(method="lasso", task="regression")
        fs = FeatureSelector(config=cfg)
        fs.fit(X.values, y)
        X_out = fs.transform(X.values)
        assert X_out.shape[1] <= X.shape[1]
        names = fs.get_feature_names_out()
        # ndarray入力の場合はx0, x1, ...になる
        assert all(n.startswith("x") for n in names)


# ============================================================
# max_features
# ============================================================

class TestMaxFeatures:
    def test_max_features_limits(self):
        X, y = _make_data()
        cfg = FeatureSelectorConfig(
            method="rfr", task="regression", max_features=3,
        )
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] <= 3


# ============================================================
# ReliefF (skip if unavailable)
# ============================================================

class TestReliefF:
    def test_relieff_or_fallback(self):
        X, y = _make_data()
        cfg = FeatureSelectorConfig(method="relieff", task="regression")
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] <= X.shape[1]


# ============================================================
# Boruta (skip if unavailable)
# ============================================================

class TestBoruta:
    def test_boruta_or_fallback(self):
        X, y = _make_data()
        cfg = FeatureSelectorConfig(method="boruta", task="regression")
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] <= X.shape[1]


# ============================================================
# Genetic (skip if unavailable)
# ============================================================

class TestGenetic:
    def test_genetic_or_fallback(self):
        X, y = _make_data()
        cfg = FeatureSelectorConfig(method="genetic", task="regression")
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] <= X.shape[1]


# ============================================================
# GroupLasso (skip if unavailable)
# ============================================================

class TestGroupLasso:
    def test_group_lasso_or_fallback(self):
        X, y = _make_data()
        cfg = FeatureSelectorConfig(method="group_lasso", task="regression")
        fs = FeatureSelector(config=cfg)
        fs.fit(X, y)
        X_out = fs.transform(X)
        assert X_out.shape[1] <= X.shape[1]
