"""
tests/test_eda_extra.py

eda.py のカバレッジ改善テスト。
compute_column_stats, summarize_dataframe, compute_correlation,
detect_outliers, compute_distribution, analyze_target を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.data.eda import (
    ColumnStats,
    compute_column_stats,
    summarize_dataframe,
    compute_correlation,
    OutlierResult,
    detect_outliers,
    compute_distribution,
    analyze_target,
)


# ============================================================
# テストデータ
# ============================================================

def _make_df():
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "num1": rng.randn(100),
        "num2": rng.randn(100) * 10 + 5,
        "cat1": rng.choice(["A", "B", "C"], 100),
        "int1": rng.randint(0, 50, 100),
    })


# ============================================================
# compute_column_stats
# ============================================================

class TestComputeColumnStats:
    def test_numeric_stats(self):
        df = _make_df()
        stats = compute_column_stats(df)
        assert len(stats) == 4
        num_stat = [s for s in stats if s.name == "num1"][0]
        assert num_stat.mean is not None
        assert num_stat.std is not None
        assert num_stat.p25 is not None

    def test_categorical_stats(self):
        df = _make_df()
        stats = compute_column_stats(df)
        cat_stat = [s for s in stats if s.name == "cat1"][0]
        assert cat_stat.mean is None
        assert len(cat_stat.top_values) > 0

    def test_null_stats(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan, 5.0]})
        stats = compute_column_stats(df)
        assert stats[0].n_null == 2
        assert stats[0].null_rate == 0.4

    def test_single_value(self):
        df = pd.DataFrame({"x": [5.0]})
        stats = compute_column_stats(df)
        assert stats[0].std == 0.0


# ============================================================
# summarize_dataframe
# ============================================================

class TestSummarizeDataframe:
    def test_basic(self):
        df = _make_df()
        summary = summarize_dataframe(df)
        assert summary["n_rows"] == 100
        assert summary["n_cols"] == 4
        assert summary["n_numeric"] >= 2
        assert summary["n_categorical"] >= 1
        assert "memory_mb" in summary

    def test_with_duplicates(self):
        df = pd.DataFrame({"x": [1, 1, 2, 2, 3]})
        summary = summarize_dataframe(df)
        assert summary["n_duplicates"] >= 1


# ============================================================
# compute_correlation
# ============================================================

class TestComputeCorrelation:
    def test_pearson(self):
        df = _make_df()
        corr = compute_correlation(df, method="pearson")
        assert corr.shape[0] == corr.shape[1]
        assert corr.shape[0] >= 2

    def test_spearman(self):
        df = _make_df()
        corr = compute_correlation(df, method="spearman")
        assert corr.shape[0] >= 2

    def test_with_target(self):
        df = _make_df()
        corr = compute_correlation(df, target_col="num1")
        assert corr.shape[1] == 1
        assert "num1" not in corr.index

    def test_too_few_columns(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="2列"):
            compute_correlation(df)


# ============================================================
# detect_outliers
# ============================================================

class TestDetectOutliers:
    def test_iqr(self):
        df = _make_df()
        results = detect_outliers(df, method="iqr")
        assert len(results) >= 2
        for r in results:
            assert r.method == "iqr"
            assert r.lower_bound is not None

    def test_zscore(self):
        df = _make_df()
        results = detect_outliers(df, method="zscore")
        assert len(results) >= 2

    def test_modified_zscore(self):
        df = _make_df()
        results = detect_outliers(df, method="modified_zscore")
        assert len(results) >= 2

    def test_specific_cols(self):
        df = _make_df()
        results = detect_outliers(df, cols=["num1"])
        assert len(results) == 1
        assert results[0].col == "num1"

    def test_unknown_method(self):
        df = _make_df()
        with pytest.raises(ValueError, match="未知"):
            detect_outliers(df, method="unknown")


# ============================================================
# compute_distribution
# ============================================================

class TestComputeDistribution:
    def test_numeric(self):
        s = pd.Series(np.random.randn(100))
        result = compute_distribution(s, bins=20)
        assert "counts" in result
        assert "bin_edges" in result
        assert len(result["counts"]) == 20

    def test_categorical(self):
        s = pd.Series(["A", "B", "C", "A", "B"])
        result = compute_distribution(s)
        assert "categories" in result
        assert "counts" in result


# ============================================================
# analyze_target
# ============================================================

class TestAnalyzeTarget:
    def test_regression(self):
        df = pd.DataFrame({"y": np.random.randn(50), "x": np.random.randn(50)})
        result = analyze_target(df, "y", task="regression")
        assert result["task"] == "regression"
        assert "mean" in result
        assert "std" in result

    def test_classification(self):
        df = pd.DataFrame({
            "label": np.random.choice([0, 1, 2], 50),
            "x": np.random.randn(50),
        })
        result = analyze_target(df, "label", task="classification")
        assert result["task"] == "classification"
        assert "class_counts" in result

    def test_auto_float(self):
        df = pd.DataFrame({"y": np.random.randn(50)})
        result = analyze_target(df, "y", task="auto")
        assert result["task"] == "regression"

    def test_auto_categorical(self):
        df = pd.DataFrame({"y": np.random.choice(["A", "B"], 50)})
        result = analyze_target(df, "y", task="auto")
        assert result["task"] == "classification"

    def test_missing_col(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="存在しません"):
            analyze_target(df, "y_nonexistent")
