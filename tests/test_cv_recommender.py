# -*- coding: utf-8 -*-
"""
tests/test_cv_recommender.py

cv_recommender.py の網羅的テストスイート。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.utils.cv_recommender import (
    CVRecommendation,
    recommend_cv_strategy,
    _detect_timeseries,
    _detect_groups,
    _detect_imbalance,
    _assess_sample_size,
    _recommend_n_splits,
    _recommend_ts_splits,
)

# ────────────────────────────────────────────────────────────
# 1. ヘルパー関数のテスト
# ────────────────────────────────────────────────────────────
class TestRecommendSplits:
    def test_recommend_n_splits(self):
        assert _recommend_n_splits(15000) == 10
        assert _recommend_n_splits(1500) == 5
        assert _recommend_n_splits(250) == 5
        assert _recommend_n_splits(80) == 3
        assert _recommend_n_splits(10) == 2

    def test_recommend_ts_splits(self):
        assert _recommend_ts_splits(600) == 5
        assert _recommend_ts_splits(250) == 4
        assert _recommend_ts_splits(80) == 3
        assert _recommend_ts_splits(10) == 2

# ────────────────────────────────────────────────────────────
# 2. _detect_timeseries のテスト
# ────────────────────────────────────────────────────────────
class TestDetectTimeseries:
    def test_explicit_metadata(self):
        X = pd.DataFrame({"A": [1, 2], "time": [3, 4]})
        meta = {"time_col": "time"}
        res = _detect_timeseries(X, meta)
        assert res["is_timeseries"]
        assert res["detected_column"] == "time"
        assert res["confidence"] == 0.95

    def test_column_pattern(self):
        X = pd.DataFrame({"A": [1, 2], "created_at": [3, 4]})
        res = _detect_timeseries(X, {})
        assert res["is_timeseries"]
        assert "created_at" in res["detected_column"].lower()

    def test_monotonic_increasing(self):
        # CVが小さい単調増加列
        vals = np.linspace(0, 100, 20)
        X = pd.DataFrame({"A": vals})
        res = _detect_timeseries(X, {})
        assert res["is_timeseries"]
        assert res["detected_column"] == "A"
        assert res["confidence"] == 0.70

    def test_not_timeseries(self):
        X = pd.DataFrame({"A": np.random.randn(20)})
        res = _detect_timeseries(X, {})
        assert not res["is_timeseries"]

# ────────────────────────────────────────────────────────────
# 3. _detect_groups のテスト
# ────────────────────────────────────────────────────────────
class TestDetectGroups:
    def test_explicit_group_col(self):
        X = pd.DataFrame({"g": [1, 1, 2, 2], "v": [1, 2, 3, 4]})
        res = _detect_groups(X, pd.Series([0, 1, 0, 1]), {"group_col": "g"})
        assert res["has_groups"]
        assert res["n_groups"] == 2
        assert res["confidence"] == 0.95

    def test_explicit_groups_array(self):
        X = pd.DataFrame({"v": [1, 2, 3, 4]})
        res = _detect_groups(X, pd.Series([0, 1, 0, 1]), {"groups": [1, 1, 2, 2]})
        assert res["has_groups"]
        assert res["n_groups"] == 2
        assert res["confidence"] == 0.90
        
    def test_leakage_groups(self):
        X = pd.DataFrame({"v": [1, 2, 3, 4]})
        res = _detect_groups(X, pd.Series([0, 1, 0, 1]), {"leakage_group_labels": [1, 1, 2, 2]})
        assert res["has_groups"]
        assert res["confidence"] == 0.85

    def test_rbf_similarity(self):
        # 高い類似度を持つペアを意図的に作成
        X_arr = np.concatenate([
            np.tile([1.0, 1.0], (10, 1)),
            np.tile([-1.0, -1.0], (10, 1)),
            np.tile([5.0, 5.0], (5, 1))
        ])
        X = pd.DataFrame(X_arr, columns=["f1", "f2"])
        y = pd.Series([0]*25)
        res = _detect_groups(X, y, {})
        assert res["has_groups"]
        assert res["n_groups"] >= 2
        assert res["confidence"] == 0.65

    def test_no_groups(self):
        X = pd.DataFrame(np.random.randn(50, 5))
        y = pd.Series(np.random.randn(50))
        res = _detect_groups(X, y, {})
        assert not res["has_groups"]

# ────────────────────────────────────────────────────────────
# 4. _detect_imbalance のテスト
# ────────────────────────────────────────────────────────────
class TestDetectImbalance:
    def test_highly_imbalanced(self):
        y = pd.Series([0]*100 + [1]*5)
        res = _detect_imbalance(y)
        assert res["is_imbalanced"]
        assert res["imbalance_ratio"] > 10
        assert res["confidence"] == 0.90

    def test_moderately_imbalanced(self):
        y = pd.Series([0]*50 + [1]*10)
        res = _detect_imbalance(y)
        assert res["is_imbalanced"]
        assert 3 < res["imbalance_ratio"] <= 10
        assert res["confidence"] == 0.75

    def test_few_samples_in_minority(self):
        y = pd.Series([0]*20 + [1]*8)  # ratio < 3 but min_count < 10
        res = _detect_imbalance(y)
        assert res["is_imbalanced"]
        assert res["confidence"] == 0.70

    def test_balanced(self):
        y = pd.Series([0]*50 + [1]*50)
        res = _detect_imbalance(y)
        assert not res["is_imbalanced"]

    def test_single_class(self):
        y = pd.Series([0]*50)
        res = _detect_imbalance(y)
        assert not res["is_imbalanced"]
        assert "1クラス" in res["reason"]

# ────────────────────────────────────────────────────────────
# 5. _assess_sample_size のテスト
# ────────────────────────────────────────────────────────────
class TestAssessSampleSize:
    def test_very_small(self):
        res = _assess_sample_size(15, 5)
        assert res["is_small"]
        assert res["category"] == "very_small"

    def test_small(self):
        res = _assess_sample_size(40, 5)
        assert res["is_small"]
        assert res["category"] == "small"

    def test_high_dim(self):
        res = _assess_sample_size(100, 80)
        assert res["is_small"]
        assert res["category"] == "high_dim"

    def test_normal(self):
        res = _assess_sample_size(200, 10)
        assert not res["is_small"]

# ────────────────────────────────────────────────────────────
# 6. recommend_cv_strategy メインAPIのテスト
# ────────────────────────────────────────────────────────────
class TestRecommendCvStrategy:
    def test_timeseries_priority(self):
        # 時系列でもありクラス不均衡でもある場合、時系列が優先される
        X = pd.DataFrame({"time": np.arange(100), "feat": np.random.randn(100)})
        y = pd.Series([0]*90 + [1]*10)
        res = recommend_cv_strategy(X, y, {"task_type": "classification"})
        assert res.recommended_cv == "timeseries"

    def test_groups_priority(self):
        # グループでもあり小サンプル(<20)の場合、グループが優先
        X = pd.DataFrame({"v": np.random.randn(15)})
        y = pd.Series(np.random.randn(15))
        groups = np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4])
        res = recommend_cv_strategy(X, y, {"groups": groups})
        # n_groups = 5 なので logo が推奨される
        assert res.recommended_cv == "logo"

    def test_groups_group_kfold(self):
        X = pd.DataFrame({"v": np.random.randn(40)})
        y = pd.Series(np.random.randn(40))
        groups = np.repeat(np.arange(10), 4) # 10 groups
        res = recommend_cv_strategy(X, y, {"groups": groups})
        assert res.recommended_cv == "group_kfold"

    def test_imbalance_stratified(self):
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series([0]*95 + [1]*5)
        res = recommend_cv_strategy(X, y, {"task_type": "classification"})
        assert res.recommended_cv == "stratified_kfold"

    def test_small_sample_loo(self):
        X = pd.DataFrame(np.random.randn(15, 5))
        y = pd.Series(np.random.randn(15))
        res = recommend_cv_strategy(X, y, {"task_type": "regression"})
        assert res.recommended_cv == "loo"

    def test_small_sample_repeated(self):
        X = pd.DataFrame(np.random.randn(30, 5))
        y = pd.Series(np.random.randn(30))
        res = recommend_cv_strategy(X, y, {"task_type": "regression"})
        assert res.recommended_cv == "repeated_kfold"

    def test_default_regression(self):
        X = pd.DataFrame(np.random.randn(500, 5))
        y = pd.Series(np.random.randn(500))
        res = recommend_cv_strategy(X, y, {"task_type": "regression"})
        assert res.recommended_cv == "kfold"
        assert res.recommended_params["n_splits"] == 5

    def test_default_classification(self):
        X = pd.DataFrame(np.random.randn(500, 5))
        y = pd.Series([0]*250 + [1]*250)
        res = recommend_cv_strategy(X, y, {"task_type": "classification"})
        assert res.recommended_cv == "stratified_kfold"

    def test_numpy_input(self):
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        res = recommend_cv_strategy(X, y)
        assert res.recommended_cv == "kfold"
