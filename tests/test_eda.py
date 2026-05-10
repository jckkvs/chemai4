"""
tests/test_eda.py

backend/data/eda.py のユニットテスト。
compute_column_stats, summarize_dataframe, compute_correlation,
detect_outliers, compute_distribution, analyze_target をテストする。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.data.eda import (
    compute_column_stats,
    summarize_dataframe,
    compute_correlation,
    detect_outliers,
    compute_distribution,
    analyze_target,
)


# ============================================================
# フィクスチャ
# ============================================================

@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """数値・カテゴリ・欠損を含む混合DataFrame。"""
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "num_a": np.random.randn(n),
        "num_b": np.random.randn(n) * 10,
        "cat": np.random.choice(["X", "Y", "Z"], n),
        "target": np.random.randn(n),
        "null_col": [None] * 10 + list(np.random.randn(50)),
    })


@pytest.fixture
def outlier_df() -> pd.DataFrame:
    """外れ値を含むDataFrame。"""
    np.random.seed(0)
    values = np.concatenate([np.random.randn(95), [100.0, -100.0, 200.0, -200.0, 50.0]])
    return pd.DataFrame({"x": values, "y": np.random.randn(100)})


# ============================================================
# compute_column_stats
# ============================================================

class TestComputeColumnStats:
    """T-EDA-001: 列統計計算のテスト。"""

    def test_returns_list_length(self, mixed_df: pd.DataFrame) -> None:
        """全列数分のColumnStatsが返ること。(T-EDA-001-01)"""
        stats = compute_column_stats(mixed_df)
        assert len(stats) == len(mixed_df.columns)

    def test_numeric_col_has_mean(self, mixed_df: pd.DataFrame) -> None:
        """数値列にmeanが設定されること。(T-EDA-001-02)"""
        stats = compute_column_stats(mixed_df)
        num_stat = next(s for s in stats if s.name == "num_a")
        assert num_stat.mean is not None
        assert num_stat.std is not None

    def test_categorical_col_has_top_values(self, mixed_df: pd.DataFrame) -> None:
        """カテゴリ列にtop_valuesが設定されること。(T-EDA-001-03)"""
        stats = compute_column_stats(mixed_df)
        cat_stat = next(s for s in stats if s.name == "cat")
        assert len(cat_stat.top_values) > 0

    def test_null_rate_correctness(self, mixed_df: pd.DataFrame) -> None:
        """null_rateが正しく計算されること。(T-EDA-001-04)"""
        stats = compute_column_stats(mixed_df)
        null_stat = next(s for s in stats if s.name == "null_col")
        assert null_stat.n_null == 10
        assert abs(null_stat.null_rate - 10 / 60) < 1e-6

    def test_dtype_field(self, mixed_df: pd.DataFrame) -> None:
        """dtypeフィールドが文字列で返ること。(T-EDA-001-05)"""
        stats = compute_column_stats(mixed_df)
        for s in stats:
            assert isinstance(s.dtype, str)


# ============================================================
# summarize_dataframe
# ============================================================

class TestSummarizeDataframe:
    """T-EDA-002: DataFrame全体サマリーのテスト。"""

    def test_required_keys(self, mixed_df: pd.DataFrame) -> None:
        """必須キーが全て含まれること。(T-EDA-002-01)"""
        summary = summarize_dataframe(mixed_df)
        for key in ["n_rows", "n_cols", "n_numeric", "n_categorical",
                    "total_null_rate", "n_duplicates", "memory_mb"]:
            assert key in summary

    def test_shape_correctness(self, mixed_df: pd.DataFrame) -> None:
        """n_rows / n_cols が正しいこと。(T-EDA-002-02)"""
        summary = summarize_dataframe(mixed_df)
        assert summary["n_rows"] == 60
        assert summary["n_cols"] == 5

    def test_n_numeric_count(self, mixed_df: pd.DataFrame) -> None:
        """n_numericが正しく数値列をカウントすること。(T-EDA-002-03)"""
        summary = summarize_dataframe(mixed_df)
        # num_a, num_b, target, null_col(float) = 4
        assert summary["n_numeric"] == 4

    def test_null_rate_range(self, mixed_df: pd.DataFrame) -> None:
        """total_null_rateが0〜1の範囲であること。(T-EDA-002-04)"""
        summary = summarize_dataframe(mixed_df)
        assert 0.0 <= summary["total_null_rate"] <= 1.0

    def test_memory_mb_positive(self, mixed_df: pd.DataFrame) -> None:
        """memory_mbが正の値であること。(T-EDA-002-05)"""
        summary = summarize_dataframe(mixed_df)
        assert summary["memory_mb"] > 0


# ============================================================
# compute_correlation
# ============================================================

class TestComputeCorrelation:
    """T-EDA-003: 相関計算のテスト。"""

    def test_pearson_matrix_shape(self, mixed_df: pd.DataFrame) -> None:
        """Pearson相関行列が正方行列であること。(T-EDA-003-01)"""
        corr = compute_correlation(mixed_df)
        n_num = mixed_df.select_dtypes(include="number").shape[1]
        assert corr.shape == (n_num, n_num)

    def test_diagonal_is_one(self, mixed_df: pd.DataFrame) -> None:
        """対角成分が1.0であること。(T-EDA-003-02)"""
        corr = compute_correlation(mixed_df)
        diag = np.diag(corr.values)
        np.testing.assert_array_almost_equal(diag, np.ones(len(diag)))

    def test_spearman_method(self, mixed_df: pd.DataFrame) -> None:
        """Spearman相関が計算できること。(T-EDA-003-03)"""
        corr = compute_correlation(mixed_df, method="spearman")
        assert corr.shape[0] == corr.shape[1]

    def test_target_col_filter(self, mixed_df: pd.DataFrame) -> None:
        """target_col指定時に1列のDataFrameが返ること。(T-EDA-003-04)"""
        corr = compute_correlation(mixed_df, target_col="target")
        assert corr.shape[1] == 1

    def test_few_columns_raises(self) -> None:
        """数値列が1列以下でValueErrorが上がること。(T-EDA-003-05)"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        with pytest.raises(ValueError):
            compute_correlation(df)


# ============================================================
# detect_outliers
# ============================================================

class TestDetectOutliers:
    """T-EDA-004: 外れ値検出のテスト。"""

    def test_iqr_detects_extreme_values(self, outlier_df: pd.DataFrame) -> None:
        """IQR法で極端な外れ値を検出できること。(T-EDA-004-01)"""
        results = detect_outliers(outlier_df, method="iqr")
        x_result = next(r for r in results if r.col == "x")
        assert x_result.n_outliers >= 4  # 100, -100, 200, -200

    def test_zscore_method(self, outlier_df: pd.DataFrame) -> None:
        """Zscore法でも外れ値が検出できること。(T-EDA-004-02)"""
        results = detect_outliers(outlier_df, method="zscore", z_threshold=3.0)
        assert any(r.n_outliers > 0 for r in results)

    def test_modified_zscore_method(self, outlier_df: pd.DataFrame) -> None:
        """Modified Zscore法でも外れ値が検出できること。(T-EDA-004-03)"""
        results = detect_outliers(outlier_df, method="modified_zscore")
        assert any(r.n_outliers > 0 for r in results)

    def test_outlier_rate_range(self, outlier_df: pd.DataFrame) -> None:
        """outlier_rateが0〜1の範囲であること。(T-EDA-004-04)"""
        results = detect_outliers(outlier_df, method="iqr")
        for r in results:
            assert 0.0 <= r.outlier_rate <= 1.0

    def test_unknown_method_raises(self, outlier_df: pd.DataFrame) -> None:
        """未知の手法でValueErrorが上がること。(T-EDA-004-05)"""
        with pytest.raises(ValueError, match="未知の外れ値"):
            detect_outliers(outlier_df, method="unknown")

    def test_col_filter(self, outlier_df: pd.DataFrame) -> None:
        """cols指定で特定列のみを対象とすること。(T-EDA-004-06)"""
        results = detect_outliers(outlier_df, method="iqr", cols=["x"])
        assert all(r.col == "x" for r in results)


# ============================================================
# compute_distribution
# ============================================================

class TestComputeDistribution:
    """T-EDA-005: 分布計算のテスト。"""

    def test_numeric_returns_histogram(self, mixed_df: pd.DataFrame) -> None:
        """数値列でcounts/bin_edgesが返ること。(T-EDA-005-01)"""
        result = compute_distribution(mixed_df["num_a"], bins=20)
        assert "counts" in result
        assert "bin_edges" in result
        assert len(result["counts"]) == 20

    def test_categorical_returns_categories(self, mixed_df: pd.DataFrame) -> None:
        """カテゴリ列でcategories/countsが返ること。(T-EDA-005-02)"""
        result = compute_distribution(mixed_df["cat"])
        assert "categories" in result
        assert "counts" in result

    def test_sum_of_counts_equals_non_null(self, mixed_df: pd.DataFrame) -> None:
        """histogramのcounts合計が非null数と一致すること。(T-EDA-005-03)"""
        result = compute_distribution(mixed_df["num_a"], bins=10)
        assert sum(result["counts"]) == int(mixed_df["num_a"].notna().sum())


# ============================================================
# analyze_target
# ============================================================

class TestAnalyzeTarget:
    """T-EDA-006: 目的変数分析のテスト。"""

    def test_regression_keys(self, mixed_df: pd.DataFrame) -> None:
        """回帰タスクでmean/std/min/maxが返ること。(T-EDA-006-01)"""
        result = analyze_target(mixed_df, "target", task="regression")
        for key in ["mean", "std", "min", "max", "p50", "skewness"]:
            assert key in result

    def test_classification_keys(self, mixed_df: pd.DataFrame) -> None:
        """分類タスクでclass_countsが返ること。(T-EDA-006-02)"""
        result = analyze_target(mixed_df, "cat", task="classification")
        assert "class_counts" in result
        assert "class_balance" in result

    def test_auto_detection_regression(self, mixed_df: pd.DataFrame) -> None:
        """task='auto'で浮動小数列が回帰判定されること。(T-EDA-006-03)"""
        result = analyze_target(mixed_df, "num_a", task="auto")
        assert result["task"] == "regression"

    def test_missing_col_raises(self, mixed_df: pd.DataFrame) -> None:
        """存在しない列名でValueErrorが上がること。(T-EDA-006-04)"""
        with pytest.raises(ValueError, match="目的変数列"):
            analyze_target(mixed_df, "nonexistent")

    def test_null_rate(self, mixed_df: pd.DataFrame) -> None:
        """null_rateが正しく返ること。(T-EDA-006-05)"""
        result = analyze_target(mixed_df, "null_col", task="regression")
        assert result["n_null"] == 10
        assert abs(result["null_rate"] - 10 / 60) < 1e-4
