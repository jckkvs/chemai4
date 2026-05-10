"""
tests/test_cv_manager_comprehensive.py

CVManager モジュールの包括テスト。
WalkForwardSplit, CVConfig, get_cv, list_cv_methods, run_cross_validation を網羅。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.datasets import make_regression, make_classification

from backend.models.cv_manager import (
    WalkForwardSplit,
    CVConfig,
    get_cv,
    list_cv_methods,
    run_cross_validation,
    _CV_REGISTRY,
)


# ============================================================
# WalkForwardSplit テスト
# ============================================================

class TestWalkForwardSplit:
    def test_basic_split(self):
        X = np.arange(50).reshape(-1, 1)
        splitter = WalkForwardSplit(n_splits=3)
        splits = list(splitter.split(X))
        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # テストは常に学習後
            assert train_idx[-1] < test_idx[0] or splitter.gap > 0

    def test_with_gap(self):
        X = np.arange(100).reshape(-1, 1)
        splitter = WalkForwardSplit(n_splits=3, gap=5)
        splits = list(splitter.split(X))
        for train_idx, test_idx in splits:
            # gapがあるのでtrain_endとtest_startの間に間隔
            assert test_idx[0] - train_idx[-1] > 1

    def test_min_train_size(self):
        X = np.arange(50).reshape(-1, 1)
        splitter = WalkForwardSplit(n_splits=3, min_train_size=20)
        splits = list(splitter.split(X))
        for train_idx, _ in splits:
            assert len(train_idx) >= 20

    def test_get_n_splits(self):
        splitter = WalkForwardSplit(n_splits=7)
        assert splitter.get_n_splits() == 7

    def test_too_few_samples(self):
        X = np.arange(3).reshape(-1, 1)
        splitter = WalkForwardSplit(n_splits=5)
        with pytest.raises(ValueError, match="データ数"):
            list(splitter.split(X))

    def test_with_dataframe(self):
        df = pd.DataFrame({"a": range(50)})
        splitter = WalkForwardSplit(n_splits=3)
        splits = list(splitter.split(df))
        assert len(splits) == 3


# ============================================================
# CVConfig テスト
# ============================================================

class TestCVConfig:
    def test_defaults(self):
        cfg = CVConfig()
        assert cfg.cv_key == "stratified_kfold"
        assert cfg.n_splits == 5

    def test_custom(self):
        cfg = CVConfig(cv_key="kfold", n_splits=10)
        assert cfg.n_splits == 10


# ============================================================
# get_cv テスト
# ============================================================

class TestGetCV:
    def test_kfold(self):
        cv = get_cv(CVConfig(cv_key="kfold", n_splits=3))
        assert cv.n_splits == 3

    def test_stratified_kfold(self):
        cv = get_cv(CVConfig(cv_key="stratified_kfold", n_splits=5))
        assert cv.n_splits == 5

    def test_loo(self):
        cv = get_cv(CVConfig(cv_key="loo"))
        assert cv is not None

    def test_walk_forward(self):
        cv = get_cv(CVConfig(cv_key="walk_forward", n_splits=3))
        assert isinstance(cv, WalkForwardSplit)

    def test_timeseries(self):
        cv = get_cv(CVConfig(cv_key="timeseries", n_splits=4))
        assert cv.n_splits == 4

    def test_shuffle_split(self):
        cv = get_cv(CVConfig(cv_key="shuffle_split", n_splits=3))
        assert cv is not None

    def test_repeated_kfold(self):
        cv = get_cv(CVConfig(cv_key="repeated_kfold", n_splits=3))
        assert cv is not None

    def test_predefined_split_without_test_fold_raises(self):
        with pytest.raises(ValueError, match="test_fold"):
            get_cv(CVConfig(cv_key="predefined"))

    def test_predefined_split_with_test_fold(self):
        cv = get_cv(CVConfig(
            cv_key="predefined",
            extra_params={"test_fold": [0, 0, 1, 1, 2, 2]}
        ))
        assert cv is not None

    def test_unknown_cv_raises(self):
        with pytest.raises(ValueError):
            get_cv(CVConfig(cv_key="NonExistentCV"))

    def test_extra_params(self):
        cv = get_cv(CVConfig(
            cv_key="kfold", n_splits=10,
            extra_params={"shuffle": True}
        ))
        assert cv.n_splits == 10

    def test_all_registered_keys(self):
        """全レジストリキーがエラーなくインスタンス化可能"""
        skip_keys = {"predefined"}  # test_fold必須
        for key in _CV_REGISTRY:
            if key in skip_keys:
                continue
            try:
                cv = get_cv(CVConfig(cv_key=key, n_splits=3))
                assert cv is not None, f"key={key} returns None"
            except Exception as e:
                pytest.fail(f"key={key} raised {e}")


# ============================================================
# list_cv_methods テスト
# ============================================================

class TestListCVMethods:
    def test_regression(self):
        methods = list_cv_methods(task="regression")
        assert len(methods) > 0
        keys = [m["key"] for m in methods]
        assert "kfold" in keys
        # 分類限定のものは含まれない
        for m in methods:
            entry = _CV_REGISTRY[m["key"]]
            assert not entry.get("requires_classification")

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
# run_cross_validation テスト
# ============================================================

class TestRunCrossValidation:
    @pytest.fixture
    def reg_data(self):
        return make_regression(n_samples=50, n_features=3, noise=0.1, random_state=42)

    def test_basic_cv(self, reg_data):
        X, y = reg_data
        config = CVConfig(cv_key="kfold", n_splits=3)
        result = run_cross_validation(
            Ridge(), X, y, config, scoring="r2"
        )
        assert "test_r2" in result
        assert "mean_test_score" in result
        assert len(result["test_r2"]) == 3

    def test_timeseries_cv(self, reg_data):
        X, y = reg_data
        config = CVConfig(cv_key="timeseries", n_splits=3)
        result = run_cross_validation(
            Ridge(), X, y, config, scoring="neg_root_mean_squared_error"
        )
        assert "mean_test_score" in result

    def test_walk_forward_cv(self, reg_data):
        X, y = reg_data
        config = CVConfig(cv_key="walk_forward", n_splits=3)
        result = run_cross_validation(
            Ridge(), X, y, config, scoring="r2", n_jobs=1
        )
        assert "mean_test_score" in result
