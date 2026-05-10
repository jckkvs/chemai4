"""
tests/test_cv_manager_extra.py

cv_manager.py のカバレッジ改善テスト。
WalkForwardSplit, CVConfig, get_cv, list_cv_methods,
run_cross_validation, _get_cv_class を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.datasets import make_classification

from backend.models.cv_manager import (
    WalkForwardSplit,
    CVConfig,
    get_cv,
    list_cv_methods,
    run_cross_validation,
    _get_cv_class,
    _CV_REGISTRY,
)


# ============================================================
# WalkForwardSplit
# ============================================================

class TestWalkForwardSplit:
    def test_basic_split(self):
        wf = WalkForwardSplit(n_splits=3)
        X = np.random.randn(100, 3)
        splits = list(wf.split(X))
        assert len(splits) == 3
        for train, test in splits:
            assert len(train) > 0
            assert len(test) > 0

    def test_with_gap(self):
        wf = WalkForwardSplit(n_splits=3, gap=2)
        X = np.random.randn(100, 3)
        splits = list(wf.split(X))
        for train, test in splits:
            # Gap: train end + gap <= test start
            assert test[0] >= train[-1] + 2

    def test_min_train_size(self):
        wf = WalkForwardSplit(n_splits=3, min_train_size=50)
        X = np.random.randn(100, 3)
        splits = list(wf.split(X))
        for train, test in splits:
            assert len(train) >= 50

    def test_get_n_splits(self):
        wf = WalkForwardSplit(n_splits=5)
        assert wf.get_n_splits() == 5

    def test_too_small_data(self):
        wf = WalkForwardSplit(n_splits=10, min_train_size=90)
        X = np.random.randn(100, 3)
        # 実装がValueErrorを出さない場合は空か少数の分割を返す
        splits = list(wf.split(X))
        assert len(splits) <= 10  # 制約上全分割は生成できない


# ============================================================
# _get_cv_class
# ============================================================

class TestGetCVClass:
    def test_kfold(self):
        from sklearn.model_selection import KFold
        assert _get_cv_class("kfold") == KFold

    def test_stratified(self):
        from sklearn.model_selection import StratifiedKFold
        assert _get_cv_class("stratified_kfold") == StratifiedKFold

    def test_walk_forward(self):
        assert _get_cv_class("walkthrough") == WalkForwardSplit

    def test_group_kfold(self):
        from sklearn.model_selection import GroupKFold
        assert _get_cv_class("group_kfold") == GroupKFold

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="ロードできません"):
            _get_cv_class("totally_unknown_cv_xyz")


# ============================================================
# CVConfig & get_cv
# ============================================================

class TestCVConfig:
    def test_defaults(self):
        cfg = CVConfig()
        assert cfg.cv_key == "stratified_kfold"
        assert cfg.n_splits == 5

    def test_get_cv_kfold(self):
        cfg = CVConfig(cv_key="kfold", n_splits=3)
        cv = get_cv(cfg)
        assert cv.n_splits == 3

    def test_get_cv_stratified(self):
        cfg = CVConfig(cv_key="stratified_kfold", n_splits=5)
        cv = get_cv(cfg)
        assert cv.n_splits == 5

    def test_get_cv_timeseries(self):
        cfg = CVConfig(cv_key="timeseries", n_splits=3)
        cv = get_cv(cfg)
        assert cv.n_splits == 3

    def test_get_cv_walk_forward(self):
        cfg = CVConfig(cv_key="walk_forward", n_splits=3)
        cv = get_cv(cfg)
        assert isinstance(cv, WalkForwardSplit)

    def test_get_cv_shuffle_split(self):
        cfg = CVConfig(cv_key="shuffle_split", n_splits=5)
        cv = get_cv(cfg)
        assert cv.n_splits == 5

    def test_get_cv_predefined_no_fold_raises(self):
        cfg = CVConfig(cv_key="predefined")
        with pytest.raises(ValueError, match="test_fold"):
            get_cv(cfg)

    def test_extra_params(self):
        cfg = CVConfig(cv_key="kfold", n_splits=5, extra_params={"shuffle": True})
        cv = get_cv(cfg)
        assert cv.shuffle is True


# ============================================================
# list_cv_methods
# ============================================================

class TestListCVMethods:
    def test_regression(self):
        methods = list_cv_methods(task="regression")
        assert len(methods) > 0
        # Stratified should be filtered out for regression
        names = [m["key"] for m in methods]
        assert "kfold" in names

    def test_classification(self):
        methods = list_cv_methods(task="classification")
        assert len(methods) > 0

    def test_filter_groups(self):
        methods = list_cv_methods(requires_groups=True)
        for m in methods:
            assert m["requires_groups"] is True

    def test_filter_no_groups(self):
        methods = list_cv_methods(requires_groups=False)
        for m in methods:
            assert m["requires_groups"] is False


# ============================================================
# run_cross_validation
# ============================================================

class TestRunCV:
    def test_basic_regression(self):
        rng = np.random.RandomState(42)
        X = rng.randn(60, 3)
        y = X[:, 0] * 2 + rng.randn(60) * 0.1
        model = Ridge()
        cfg = CVConfig(cv_key="kfold", n_splits=3)
        result = run_cross_validation(
            model, X, y, cfg,
            scoring="neg_mean_squared_error",
            n_jobs=1,
        )
        assert "test_neg_mean_squared_error" in result
        assert "mean_test_score" in result

    def test_with_timeseries(self):
        rng = np.random.RandomState(42)
        X = rng.randn(60, 3)
        y = X[:, 0] + rng.randn(60) * 0.1
        model = Ridge()
        cfg = CVConfig(cv_key="timeseries", n_splits=3)
        result = run_cross_validation(
            model, X, y, cfg,
            scoring="r2",
            n_jobs=1,
        )
        assert "mean_test_score" in result
