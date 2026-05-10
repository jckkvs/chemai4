# -*- coding: utf-8 -*-
"""
tests/test_cv_walkforward.py

cv_manager.py の WalkForwardSplit / CVConfig / get_cv / list_cv_methods /
run_cross_validation の包括テスト。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge, LogisticRegression

from backend.models.cv_manager import (
    WalkForwardSplit,
    CVConfig,
    get_cv,
    list_cv_methods,
    run_cross_validation,
)


# ═══════════════════════════════════════════════════════════════════
# WalkForwardSplit
# ═══════════════════════════════════════════════════════════════════

class TestWalkForwardSplit:

    def test_basic_split(self):
        X = np.arange(50).reshape(-1, 1)
        splitter = WalkForwardSplit(n_splits=3, min_train_size=10, gap=0)
        splits = list(splitter.split(X))
        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

    def test_train_before_test(self):
        """学習期間がテスト期間の前にあること（時系列順序）"""
        X = np.arange(100).reshape(-1, 1)
        splitter = WalkForwardSplit(n_splits=5, min_train_size=20, gap=0)
        for train_idx, test_idx in splitter.split(X):
            assert train_idx.max() < test_idx.min()

    def test_gap_parameter(self):
        """gapが正しく適用されること"""
        X = np.arange(100).reshape(-1, 1)
        splitter = WalkForwardSplit(n_splits=3, min_train_size=20, gap=5)
        for train_idx, test_idx in splitter.split(X):
            assert test_idx.min() - train_idx.max() >= 5

    def test_expanding_window(self):
        """学習データが拡大していくこと"""
        X = np.arange(100).reshape(-1, 1)
        splitter = WalkForwardSplit(n_splits=4, min_train_size=10, gap=0)
        splits = list(splitter.split(X))
        train_sizes = [len(s[0]) for s in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]

    def test_too_few_samples_raises(self):
        X = np.arange(5).reshape(-1, 1)
        splitter = WalkForwardSplit(n_splits=10, min_train_size=3, gap=2)
        with pytest.raises(ValueError, match="データ数"):
            list(splitter.split(X))

    def test_get_n_splits(self):
        splitter = WalkForwardSplit(n_splits=7)
        assert splitter.get_n_splits() == 7

    def test_dataframe_input(self):
        df = pd.DataFrame({"a": range(50), "b": range(50)})
        splitter = WalkForwardSplit(n_splits=3, min_train_size=10)
        splits = list(splitter.split(df))
        assert len(splits) == 3


# ═══════════════════════════════════════════════════════════════════
# CVConfig / get_cv
# ═══════════════════════════════════════════════════════════════════

class TestCVConfig:

    def test_default_config(self):
        cfg = CVConfig()
        assert cfg.cv_key == "stratified_kfold"
        assert cfg.n_splits == 5

    def test_kfold(self):
        cfg = CVConfig(cv_key="kfold", n_splits=10)
        cv = get_cv(cfg)
        assert cv.get_n_splits() == 10

    def test_stratified_kfold(self):
        cfg = CVConfig(cv_key="stratified_kfold", n_splits=5)
        cv = get_cv(cfg)
        assert cv.get_n_splits() == 5

    def test_loo(self):
        cfg = CVConfig(cv_key="loo")
        cv = get_cv(cfg)
        assert hasattr(cv, "split")

    def test_timeseries(self):
        cfg = CVConfig(cv_key="timeseries", n_splits=3)
        cv = get_cv(cfg)
        assert cv.get_n_splits() == 3

    def test_walk_forward(self):
        cfg = CVConfig(cv_key="walk_forward", n_splits=4)
        cv = get_cv(cfg)
        assert isinstance(cv, WalkForwardSplit)

    def test_shuffle_split(self):
        cfg = CVConfig(cv_key="shuffle_split", n_splits=5)
        cv = get_cv(cfg)
        assert cv.get_n_splits() == 5

    def test_extra_params(self):
        # random_state未設定でshuffle=Falseを明示
        cfg = CVConfig(cv_key="kfold", n_splits=5,
                       extra_params={"shuffle": False, "random_state": None})
        cv = get_cv(cfg)
        assert cv.shuffle is False

    def test_predefined_split_without_test_fold_raises(self):
        cfg = CVConfig(cv_key="predefined")
        with pytest.raises(ValueError, match="test_fold"):
            get_cv(cfg)

    def test_invalid_cv_key(self):
        cfg = CVConfig(cv_key="nonexistent_cv_method")
        with pytest.raises(ValueError):
            get_cv(cfg)


# ═══════════════════════════════════════════════════════════════════
# list_cv_methods
# ═══════════════════════════════════════════════════════════════════

class TestListCVMethods:

    def test_regression_methods(self):
        methods = list_cv_methods(task="regression")
        assert len(methods) > 0
        # 回帰ではstratified系は含まれない
        for m in methods:
            if "Stratified" in m["name"]:
                assert False, f"回帰タスクにStratifiedが含まれている: {m['name']}"

    def test_classification_methods(self):
        methods = list_cv_methods(task="classification")
        assert len(methods) > 0

    def test_group_filter(self):
        methods = list_cv_methods(requires_groups=True)
        for m in methods:
            assert m["requires_groups"] is True

    def test_no_group_filter(self):
        methods = list_cv_methods(requires_groups=False)
        for m in methods:
            assert m["requires_groups"] is False

    def test_methods_have_required_fields(self):
        for m in list_cv_methods():
            assert "key" in m
            assert "name" in m
            assert "description" in m


# ═══════════════════════════════════════════════════════════════════
# run_cross_validation
# ═══════════════════════════════════════════════════════════════════

class TestRunCrossValidation:

    @pytest.fixture
    def regression_data(self):
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] * 2 + X[:, 1] + np.random.randn(100) * 0.1
        return X, y

    def test_kfold_regression(self, regression_data):
        X, y = regression_data
        cfg = CVConfig(cv_key="kfold", n_splits=3)
        result = run_cross_validation(
            Ridge(), X, y, cfg, scoring="r2", n_jobs=1
        )
        assert "test_r2" in result
        assert "mean_test_score" in result
        assert len(result["test_r2"]) == 3

    def test_timeseries_regression(self, regression_data):
        X, y = regression_data
        cfg = CVConfig(cv_key="timeseries", n_splits=3)
        result = run_cross_validation(
            Ridge(), X, y, cfg, scoring="neg_mean_squared_error", n_jobs=1
        )
        assert "mean_test_score" in result

    def test_walk_forward_regression(self, regression_data):
        X, y = regression_data
        cfg = CVConfig(cv_key="walk_forward", n_splits=3,
                       extra_params={"min_train_size": 20})
        result = run_cross_validation(
            Ridge(), X, y, cfg, scoring="r2", n_jobs=1
        )
        assert "mean_test_score" in result

    def test_return_train_score(self, regression_data):
        X, y = regression_data
        cfg = CVConfig(cv_key="kfold", n_splits=3)
        result = run_cross_validation(
            Ridge(), X, y, cfg, scoring="r2", n_jobs=1,
            return_train_score=True
        )
        assert "train_r2" in result or "train_score" in result
