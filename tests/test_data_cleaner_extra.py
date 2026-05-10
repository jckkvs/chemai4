"""
tests/test_data_cleaner_extra.py

data_cleaner.py のカバレッジ改善テスト。
CleaningAction, drop_columns, drop_rows_with_missing, remove_constant_columns,
clip_outliers, remove_duplicates, preview_missing_impact, preview_outlier_impact,
get_cleaning_summary を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.data.data_cleaner import (
    CleaningAction,
    drop_columns,
    drop_rows_with_missing,
    remove_constant_columns,
    clip_outliers,
    remove_duplicates,
    preview_missing_impact,
    preview_outlier_impact,
    get_cleaning_summary,
)


# ============================================================
# CleaningAction
# ============================================================

class TestCleaningAction:
    def test_rows_removed(self):
        a = CleaningAction("test", "test", rows_before=100, rows_after=80,
                           cols_before=5, cols_after=5)
        assert a.rows_removed == 20

    def test_cols_removed(self):
        a = CleaningAction("test", "test", rows_before=50, rows_after=50,
                           cols_before=10, cols_after=7)
        assert a.cols_removed == 3


# ============================================================
# drop_columns
# ============================================================

class TestDropColumns:
    def test_basic(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        result, action = drop_columns(df, ["b"])
        assert "b" not in result.columns
        assert action.cols_removed == 1

    def test_multiple(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        result, action = drop_columns(df, ["b", "d"])
        assert list(result.columns) == ["a", "c"]

    def test_empty_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="指定されていません"):
            drop_columns(df, [])

    def test_nonexistent_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="存在しません"):
            drop_columns(df, ["zzz"])


# ============================================================
# drop_rows_with_missing
# ============================================================

class TestDropRowsWithMissing:
    def test_basic(self):
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, np.nan, 6]})
        result, action = drop_rows_with_missing(df, threshold=0.5)
        assert len(result) < len(df)

    def test_threshold_zero(self):
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6]})
        result, action = drop_rows_with_missing(df, threshold=0.0)
        assert len(result) == 2  # Only rows with no NaN

    def test_invalid_threshold(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="0.0〜1.0"):
            drop_rows_with_missing(df, threshold=1.5)

    def test_with_subset(self):
        df = pd.DataFrame({"a": [1, np.nan], "b": [np.nan, np.nan]})
        result, action = drop_rows_with_missing(df, threshold=0.0, subset=["a"])
        assert len(result) == 1

    def test_no_check_cols(self):
        df = pd.DataFrame({"a": [1, 2]})
        result, action = drop_rows_with_missing(df, subset=["nonexistent"])
        assert len(result) == 2  # No change


# ============================================================
# remove_constant_columns
# ============================================================

class TestRemoveConstantColumns:
    def test_with_constants(self):
        df = pd.DataFrame({"vary": [1, 2, 3], "const": [5, 5, 5]})
        result, action = remove_constant_columns(df)
        assert "const" not in result.columns
        assert action.cols_removed == 1

    def test_no_constants(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result, action = remove_constant_columns(df)
        assert result.shape == df.shape


# ============================================================
# clip_outliers
# ============================================================

class TestClipOutliers:
    def test_basic(self):
        rng = np.random.RandomState(42)
        vals = np.concatenate([rng.randn(95), [100, -100, 200, -200, 500]])
        df = pd.DataFrame({"x": vals})
        result, action = clip_outliers(df)
        assert result["x"].max() < 500
        assert action.action_type == "clip_outliers"

    def test_with_columns(self):
        df = pd.DataFrame({"a": [1, 2, 100], "b": [3, 4, 5]})
        result, action = clip_outliers(df, columns=["a"])
        assert action.action_type == "clip_outliers"

    def test_invalid_multiplier(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="正の数"):
            clip_outliers(df, iqr_multiplier=0)


# ============================================================
# remove_duplicates
# ============================================================

class TestRemoveDuplicates:
    def test_basic(self):
        df = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": [10, 10, 20, 20, 30]})
        result, action = remove_duplicates(df)
        assert len(result) == 3

    def test_keep_last(self):
        df = pd.DataFrame({"a": [1, 1, 2]})
        result, action = remove_duplicates(df, keep="last")
        assert len(result) == 2

    def test_subset(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
        result, action = remove_duplicates(df, subset=["a"])
        assert len(result) == 2


# ============================================================
# プレビュー関数
# ============================================================

class TestPreviewFunctions:
    def test_preview_missing(self):
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [np.nan, np.nan, 6]})
        n = preview_missing_impact(df, threshold=0.5)
        assert isinstance(n, int)
        assert n >= 0

    def test_preview_missing_zero(self):
        df = pd.DataFrame({"a": [1, np.nan, 3]})
        n = preview_missing_impact(df, threshold=0.0)
        assert n == 1

    def test_preview_outlier(self):
        vals = np.concatenate([np.zeros(95), [100, -100]])
        df = pd.DataFrame({"x": vals})
        result = preview_outlier_impact(df)
        assert isinstance(result, dict)

    def test_get_cleaning_summary(self):
        df = pd.DataFrame({
            "x": [1, 1, 2],
            "const": [5, 5, 5],
            "na_all": [np.nan, np.nan, np.nan],
        })
        summary = get_cleaning_summary(df)
        assert summary["n_const_cols"] >= 1
        assert "const" in summary["const_cols"]
        assert summary["n_all_missing_cols"] >= 1
