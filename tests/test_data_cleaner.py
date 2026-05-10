"""
tests/test_data_cleaner.py

backend/data/data_cleaner.py のユニットテスト。
全クリーニング関数の正常系・異常系・エッジケースをテスト。

テストID対応:
  T-CLEAN-001: drop_columns
  T-CLEAN-002: drop_rows_with_missing
  T-CLEAN-003: remove_constant_columns
  T-CLEAN-004: clip_outliers
  T-CLEAN-005: remove_duplicates
  T-CLEAN-006: preview 関数群
  T-CLEAN-007: get_cleaning_summary
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.data.data_cleaner import (
    CleaningAction,
    clip_outliers,
    drop_columns,
    drop_rows_with_missing,
    get_cleaning_summary,
    preview_missing_impact,
    preview_outlier_impact,
    remove_constant_columns,
    remove_duplicates,
)


# ============================================================
# フィクスチャ
# ============================================================

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """標準テスト用DataFrame。"""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "num_a": np.random.randn(n),
        "num_b": np.random.randn(n) * 10 + 5,
        "cat": np.random.choice(["A", "B", "C"], n),
        "const": [1.0] * n,
        "target": np.random.randn(n),
        "missing_heavy": [None] * 30 + list(np.random.randn(20)),
    })


@pytest.fixture
def outlier_df() -> pd.DataFrame:
    """外れ値を含むDataFrame。"""
    np.random.seed(0)
    values = np.concatenate([np.random.randn(95), [100.0, -100.0, 200.0, -200.0, 50.0]])
    return pd.DataFrame({
        "x": values,
        "y": np.random.randn(100),
        "z": [1.0] * 100,  # 定数 (IQR=0)
    })


@pytest.fixture
def dup_df() -> pd.DataFrame:
    """重複行を含むDataFrame。"""
    return pd.DataFrame({
        "a": [1, 2, 3, 1, 2, 3, 4],
        "b": [10, 20, 30, 10, 20, 30, 40],
    })


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """空のDataFrame。"""
    return pd.DataFrame()


@pytest.fixture
def all_missing_df() -> pd.DataFrame:
    """全欠損列を含むDataFrame。"""
    return pd.DataFrame({
        "ok": [1, 2, 3, 4, 5],
        "all_nan": [None, None, None, None, None],
        "half_nan": [None, None, None, 4, 5],
    })


# ============================================================
# T-CLEAN-001: drop_columns
# ============================================================

class TestDropColumns:
    """T-CLEAN-001: 列除外のテスト。"""

    def test_basic_drop(self, sample_df: pd.DataFrame) -> None:
        """T-CLEAN-001-01: 基本的な列除外。"""
        result, action = drop_columns(sample_df, ["const", "cat"])
        assert "const" not in result.columns
        assert "cat" not in result.columns
        assert "num_a" in result.columns
        assert action.action_type == "drop_columns"
        assert action.cols_removed == 2
        assert action.rows_removed == 0
        assert len(result) == len(sample_df)

    def test_drop_single_column(self, sample_df: pd.DataFrame) -> None:
        """T-CLEAN-001-02: 1列のみ除外。"""
        result, action = drop_columns(sample_df, ["target"])
        assert result.shape[1] == sample_df.shape[1] - 1
        assert action.cols_before == sample_df.shape[1]
        assert action.cols_after == sample_df.shape[1] - 1

    def test_drop_nonexistent_column_raises(self, sample_df: pd.DataFrame) -> None:
        """T-CLEAN-001-03: 存在しない列のみ指定でValueError。"""
        with pytest.raises(ValueError, match="存在しません"):
            drop_columns(sample_df, ["nonexistent_col"])

    def test_drop_empty_list_raises(self, sample_df: pd.DataFrame) -> None:
        """T-CLEAN-001-04: 空リストでValueError。"""
        with pytest.raises(ValueError, match="指定されていません"):
            drop_columns(sample_df, [])

    def test_drop_partial_nonexistent(self, sample_df: pd.DataFrame) -> None:
        """T-CLEAN-001-05: 一部存在しない列は無視して既存列のみ除外。"""
        result, action = drop_columns(sample_df, ["num_a", "fake_col"])
        assert "num_a" not in result.columns
        assert action.cols_removed == 1

    def test_action_details(self, sample_df: pd.DataFrame) -> None:
        """T-CLEAN-001-06: CleaningActionのdetailsに除外列名が含まれる。"""
        _, action = drop_columns(sample_df, ["const"])
        assert "dropped_columns" in action.details
        assert "const" in action.details["dropped_columns"]

    def test_action_timestamp_format(self, sample_df: pd.DataFrame) -> None:
        """T-CLEAN-001-07: タイムスタンプがISO形式。"""
        _, action = drop_columns(sample_df, ["const"])
        assert "T" in action.timestamp  # ISO 8601


# ============================================================
# T-CLEAN-002: drop_rows_with_missing
# ============================================================

class TestDropRowsWithMissing:
    """T-CLEAN-002: 欠損行削除のテスト。"""

    def test_basic_drop_any_missing(self, sample_df: pd.DataFrame) -> None:
        """T-CLEAN-002-01: threshold=0.0 で欠損のある行を全て削除。"""
        result, action = drop_rows_with_missing(sample_df, threshold=0.0)
        assert result.isna().sum().sum() == 0
        assert action.rows_before == len(sample_df)
        assert action.rows_after <= len(sample_df)
        assert action.action_type == "drop_missing_rows"

    def test_threshold_50_percent(self, all_missing_df: pd.DataFrame) -> None:
        """T-CLEAN-002-02: 閾値50%での削除。"""
        result, action = drop_rows_with_missing(all_missing_df, threshold=0.5)
        # all_nan列とhalf_nan列の両方がNaNの行 (行0,1,2: 2/3=66.7% >= 50%)
        assert action.rows_removed > 0

    def test_threshold_100_removes_none(self, sample_df: pd.DataFrame) -> None:
        """T-CLEAN-002-03: threshold=1.0 では全欠損行のみ削除。"""
        result, action = drop_rows_with_missing(sample_df, threshold=1.0)
        # 全列欠損の行はないはず
        assert len(result) == len(sample_df)

    def test_invalid_threshold_raises(self, sample_df: pd.DataFrame) -> None:
        """T-CLEAN-002-04: 不正な閾値でValueError。"""
        with pytest.raises(ValueError, match="0.0〜1.0"):
            drop_rows_with_missing(sample_df, threshold=1.5)
        with pytest.raises(ValueError, match="0.0〜1.0"):
            drop_rows_with_missing(sample_df, threshold=-0.1)

    def test_subset_columns(self, sample_df: pd.DataFrame) -> None:
        """T-CLEAN-002-05: subset指定で特定列のみチェック。"""
        result, action = drop_rows_with_missing(
            sample_df, threshold=0.0, subset=["num_a", "num_b"]
        )
        # num_a, num_bに欠損はないはず
        assert len(result) == len(sample_df)

    def test_subset_with_missing(self, all_missing_df: pd.DataFrame) -> None:
        """T-CLEAN-002-06: subset指定で欠損列をチェック。"""
        result, action = drop_rows_with_missing(
            all_missing_df, threshold=0.0, subset=["all_nan"]
        )
        assert len(result) == 0  # all_nan列は全欠損

    def test_empty_subset(self, sample_df: pd.DataFrame) -> None:
        """T-CLEAN-002-07: 空のsubsetでは変更なし。"""
        result, action = drop_rows_with_missing(
            sample_df, threshold=0.0, subset=["nonexistent"]
        )
        assert len(result) == len(sample_df)

    def test_reset_index(self, sample_df: pd.DataFrame) -> None:
        """T-CLEAN-002-08: 結果のインデックスがリセットされている。"""
        result, _ = drop_rows_with_missing(sample_df, threshold=0.0)
        assert list(result.index) == list(range(len(result)))


# ============================================================
# T-CLEAN-003: remove_constant_columns
# ============================================================

class TestRemoveConstantColumns:
    """T-CLEAN-003: 定数列除去のテスト。"""

    def test_basic_remove(self, sample_df: pd.DataFrame) -> None:
        """T-CLEAN-003-01: 定数列 'const' が除去される。"""
        result, action = remove_constant_columns(sample_df)
        assert "const" not in result.columns
        assert action.cols_removed >= 1
        assert action.action_type == "remove_constant_columns"

    def test_no_constant_columns(self) -> None:
        """T-CLEAN-003-02: 定数列がない場合は変更なし。"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result, action = remove_constant_columns(df)
        assert result.shape == df.shape
        assert action.cols_removed == 0

    def test_all_constant(self) -> None:
        """T-CLEAN-003-03: 全列が定数の場合。"""
        df = pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]})
        result, action = remove_constant_columns(df)
        assert result.shape[1] == 0
        assert action.cols_removed == 2

    def test_all_nan_is_constant(self, all_missing_df: pd.DataFrame) -> None:
        """T-CLEAN-003-04: 全欠損列も定数列として除去。"""
        result, action = remove_constant_columns(all_missing_df)
        assert "all_nan" not in result.columns

    def test_details_has_list(self, sample_df: pd.DataFrame) -> None:
        """T-CLEAN-003-05: detailsに除去された列名リスト。"""
        _, action = remove_constant_columns(sample_df)
        assert "constant_columns" in action.details
        assert "const" in action.details["constant_columns"]


# ============================================================
# T-CLEAN-004: clip_outliers
# ============================================================

class TestClipOutliers:
    """T-CLEAN-004: 外れ値クリッピングのテスト。"""

    def test_basic_clip(self, outlier_df: pd.DataFrame) -> None:
        """T-CLEAN-004-01: 基本的な外れ値クリッピング。"""
        result, action = clip_outliers(outlier_df, iqr_multiplier=1.5)
        assert action.action_type == "clip_outliers"
        assert action.details["total_clipped"] > 0
        # 行列数は変わらない
        assert result.shape == outlier_df.shape

    def test_no_outliers(self) -> None:
        """T-CLEAN-004-02: 外れ値がない場合は変更なし。"""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        result, action = clip_outliers(df, iqr_multiplier=3.0)
        assert action.details["total_clipped"] == 0
        pd.testing.assert_frame_equal(result, df)

    def test_iqr_zero_skipped(self, outlier_df: pd.DataFrame) -> None:
        """T-CLEAN-004-03: IQR=0の列（定数列）はスキップ。"""
        _, action = clip_outliers(outlier_df, iqr_multiplier=1.5)
        assert "z" not in action.details["clipped_per_column"]

    def test_column_subset(self, outlier_df: pd.DataFrame) -> None:
        """T-CLEAN-004-04: 特定列のみ対象。"""
        result, action = clip_outliers(outlier_df, columns=["x"])
        clipped = action.details["clipped_per_column"]
        assert "y" not in clipped
        # x列の外れ値がクリップされている
        assert result["x"].max() < outlier_df["x"].max()

    def test_invalid_multiplier_raises(self, outlier_df: pd.DataFrame) -> None:
        """T-CLEAN-004-05: 不正なIQR倍率でValueError。"""
        with pytest.raises(ValueError, match="正の数"):
            clip_outliers(outlier_df, iqr_multiplier=0)
        with pytest.raises(ValueError, match="正の数"):
            clip_outliers(outlier_df, iqr_multiplier=-1.0)

    def test_larger_multiplier_clips_less(self, outlier_df: pd.DataFrame) -> None:
        """T-CLEAN-004-06: 大きい倍率ほどクリップ数が少ない。"""
        _, action_15 = clip_outliers(outlier_df, iqr_multiplier=1.5)
        _, action_30 = clip_outliers(outlier_df, iqr_multiplier=3.0)
        assert action_15.details["total_clipped"] >= action_30.details["total_clipped"]

    def test_values_within_bounds(self, outlier_df: pd.DataFrame) -> None:
        """T-CLEAN-004-07: クリップ後の値がIQR範囲内。"""
        result, _ = clip_outliers(outlier_df, iqr_multiplier=1.5, columns=["x"])
        q1 = outlier_df["x"].quantile(0.25)
        q3 = outlier_df["x"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        assert result["x"].min() >= lower - 1e-10
        assert result["x"].max() <= upper + 1e-10


# ============================================================
# T-CLEAN-005: remove_duplicates
# ============================================================

class TestRemoveDuplicates:
    """T-CLEAN-005: 重複行除去のテスト。"""

    def test_basic_remove(self, dup_df: pd.DataFrame) -> None:
        """T-CLEAN-005-01: 基本的な重複行除去。"""
        result, action = remove_duplicates(dup_df)
        assert len(result) == 4  # 7行 - 3重複
        assert action.action_type == "remove_duplicates"
        assert action.rows_removed == 3

    def test_no_duplicates(self, sample_df: pd.DataFrame) -> None:
        """T-CLEAN-005-02: 重複なしの場合は変更なし。"""
        result, action = remove_duplicates(sample_df)
        assert len(result) == len(sample_df)
        assert action.rows_removed == 0

    def test_keep_last(self, dup_df: pd.DataFrame) -> None:
        """T-CLEAN-005-03: keep='last' で最後の行を保持。"""
        result, action = remove_duplicates(dup_df, keep="last")
        assert len(result) == 4
        assert action.details["keep"] == "last"

    def test_subset(self, dup_df: pd.DataFrame) -> None:
        """T-CLEAN-005-04: subset指定で特定列のみで重複判定。"""
        result, action = remove_duplicates(dup_df, subset=["a"])
        assert len(result) == 4  # a列の値 1,2,3,4 の4ユニーク

    def test_reset_index(self, dup_df: pd.DataFrame) -> None:
        """T-CLEAN-005-05: 結果のインデックスがリセットされている。"""
        result, _ = remove_duplicates(dup_df)
        assert list(result.index) == list(range(len(result)))


# ============================================================
# T-CLEAN-006: preview 関数群
# ============================================================

class TestPreviewFunctions:
    """T-CLEAN-006: プレビュー関数のテスト。"""

    def test_preview_missing_basic(self, sample_df: pd.DataFrame) -> None:
        """T-CLEAN-006-01: 欠損行プレビュー基本。"""
        n = preview_missing_impact(sample_df, threshold=0.0)
        assert isinstance(n, int)
        assert n >= 0

    def test_preview_missing_with_threshold(self, all_missing_df: pd.DataFrame) -> None:
        """T-CLEAN-006-02: 全欠損DFのプレビュー。"""
        n = preview_missing_impact(all_missing_df, threshold=0.5)
        assert n > 0

    def test_preview_missing_matches_actual(self, sample_df: pd.DataFrame) -> None:
        """T-CLEAN-006-03: プレビュー結果が実際の削除結果と一致。"""
        threshold = 0.0
        preview_n = preview_missing_impact(sample_df, threshold=threshold)
        _, action = drop_rows_with_missing(sample_df, threshold=threshold)
        assert preview_n == action.rows_removed

    def test_preview_outlier_basic(self, outlier_df: pd.DataFrame) -> None:
        """T-CLEAN-006-04: 外れ値プレビュー基本。"""
        result = preview_outlier_impact(outlier_df, iqr_multiplier=1.5)
        assert isinstance(result, dict)
        assert "x" in result
        assert result["x"] > 0

    def test_preview_outlier_matches_actual(self, outlier_df: pd.DataFrame) -> None:
        """T-CLEAN-006-05: 外れ値プレビューが実際のクリップ結果と一致。"""
        preview = preview_outlier_impact(outlier_df, iqr_multiplier=1.5)
        _, action = clip_outliers(outlier_df, iqr_multiplier=1.5)
        for col, n in preview.items():
            assert n == action.details["clipped_per_column"].get(col, 0)


# ============================================================
# T-CLEAN-007: get_cleaning_summary
# ============================================================

class TestGetCleaningSummary:
    """T-CLEAN-007: クリーニングサマリーのテスト。"""

    def test_basic_summary(self, sample_df: pd.DataFrame) -> None:
        """T-CLEAN-007-01: 基本サマリー。"""
        summary = get_cleaning_summary(sample_df)
        assert "n_const_cols" in summary
        assert "n_dup_rows" in summary
        assert "total_missing_rate" in summary
        assert summary["n_const_cols"] >= 1  # 'const' 列

    def test_all_missing_summary(self, all_missing_df: pd.DataFrame) -> None:
        """T-CLEAN-007-02: 全欠損列のサマリー。"""
        summary = get_cleaning_summary(all_missing_df)
        assert summary["n_all_missing_cols"] == 1  # all_nan
        assert "all_nan" in summary["all_missing_cols"]

    def test_dup_summary(self, dup_df: pd.DataFrame) -> None:
        """T-CLEAN-007-03: 重複行のサマリー。"""
        summary = get_cleaning_summary(dup_df)
        assert summary["n_dup_rows"] == 3


# ============================================================
# エッジケース
# ============================================================

class TestEdgeCases:
    """エッジケースのテスト。"""

    def test_empty_df_constant_cols(self, empty_df: pd.DataFrame) -> None:
        """空DataFrameで定数列除去。"""
        result, action = remove_constant_columns(empty_df)
        assert result.shape == (0, 0)
        assert action.cols_removed == 0

    def test_empty_df_duplicates(self, empty_df: pd.DataFrame) -> None:
        """空DataFrameで重複除去。"""
        result, action = remove_duplicates(empty_df)
        assert len(result) == 0

    def test_single_row_df(self) -> None:
        """1行DFの各操作。"""
        df = pd.DataFrame({"a": [1], "b": [2]})
        r1, action = remove_constant_columns(df)
        # 1行のDFでは全列のnunique==1なので全て定数列として除去される
        assert r1.shape == (1, 0)
        assert action.cols_removed == 2

    def test_cleaning_action_properties(self) -> None:
        """CleaningActionのプロパティ。"""
        action = CleaningAction(
            action_type="test",
            description="test",
            rows_before=100,
            rows_after=90,
            cols_before=10,
            cols_after=8,
        )
        assert action.rows_removed == 10
        assert action.cols_removed == 2
