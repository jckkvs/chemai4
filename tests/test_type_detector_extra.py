"""
tests/test_type_detector_extra.py

type_detector.py のカバレッジ改善テスト。
ColumnType, ColumnInfo, DetectionResult, TypeDetector を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.data.type_detector import (
    ColumnType,
    ColumnInfo,
    DetectionResult,
    TypeDetector,
)


# ============================================================
# ColumnInfo
# ============================================================

class TestColumnInfo:
    def test_is_numeric(self):
        info = ColumnInfo(name="x", col_type=ColumnType.NUMERIC_NORMAL, n_unique=10, null_rate=0.0)
        assert info.is_numeric is True
        assert info.is_categorical is False

    def test_is_categorical(self):
        info = ColumnInfo(name="x", col_type=ColumnType.BINARY, n_unique=2, null_rate=0.0)
        assert info.is_categorical is True
        assert info.is_numeric is False

    def test_numeric_log(self):
        info = ColumnInfo(name="x", col_type=ColumnType.NUMERIC_LOG, n_unique=50, null_rate=0.0)
        assert info.is_numeric is True

    def test_categorical_high(self):
        info = ColumnInfo(name="x", col_type=ColumnType.CATEGORY_HIGH, n_unique=500, null_rate=0.0)
        assert info.is_categorical is True


# ============================================================
# TypeDetector
# ============================================================

class TestTypeDetector:
    def test_numeric_normal(self):
        """正規分布に近い数値列"""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"normal": rng.randn(100)})
        det = TypeDetector()
        result = det.detect(df)
        info = result.column_info["normal"]
        assert info.is_numeric

    def test_numeric_log(self):
        """右裾重い正値列 → NUMERIC_LOG"""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"exp_val": np.exp(rng.randn(100) * 2)})
        det = TypeDetector(skewness_threshold=1.0)
        result = det.detect(df)
        info = result.column_info["exp_val"]
        assert info.col_type in (ColumnType.NUMERIC_LOG, ColumnType.NUMERIC_POWER)

    def test_binary_numeric(self):
        """0/1の数値バイナリ"""
        df = pd.DataFrame({"flag": [0, 1, 0, 1, 1]})
        det = TypeDetector()
        result = det.detect(df)
        info = result.column_info["flag"]
        assert info.col_type == ColumnType.BINARY

    def test_binary_string(self):
        """文字列バイナリ"""
        df = pd.DataFrame({"yn": ["yes", "no", "yes", "no"]})
        det = TypeDetector()
        result = det.detect(df)
        info = result.column_info["yn"]
        assert info.col_type == ColumnType.BINARY

    def test_category_low(self):
        """低カーディナリティカテゴリ"""
        df = pd.DataFrame({"color": ["red", "green", "blue", "red", "green"]})
        det = TypeDetector()
        result = det.detect(df)
        info = result.column_info["color"]
        assert info.col_type == ColumnType.CATEGORY_LOW

    def test_category_high(self):
        """高カーディナリティカテゴリ"""
        many = [f"id_{i}" for i in range(100)]
        df = pd.DataFrame({"uid": many})
        det = TypeDetector(cardinality_threshold=20)
        result = det.detect(df)
        info = result.column_info["uid"]
        assert info.col_type == ColumnType.CATEGORY_HIGH

    def test_constant(self):
        """定数列"""
        df = pd.DataFrame({"c": [42] * 10})
        det = TypeDetector()
        result = det.detect(df)
        assert "c" in result.constant_columns

    def test_datetime(self):
        """datetime型"""
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])})
        det = TypeDetector()
        result = det.detect(df)
        assert "ts" in result.datetime_columns

    def test_text_long(self):
        """長いテキスト（多様な値）"""
        df = pd.DataFrame({"desc": [f"This is a long text entry number {i} " * 5 for i in range(30)]})
        det = TypeDetector()
        result = det.detect(df)
        assert "desc" in result.text_columns

    def test_smiles_by_name(self):
        """列名にSMILESヒントがある場合"""
        df = pd.DataFrame({"smiles": ["CCO", "CC", "c1ccccc1"]})
        det = TypeDetector()
        result = det.detect(df)
        assert "smiles" in result.smiles_columns

    def test_periodic_user_specified(self):
        """ユーザー指定の周期変数"""
        df = pd.DataFrame({"angle": [0, 90, 180, 270, 360]})
        det = TypeDetector(periodic_cols=["angle"])
        result = det.detect(df)
        assert result.column_info["angle"].col_type == ColumnType.PERIODIC

    def test_null_rate(self):
        """欠損率"""
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan, 5.0]})
        det = TypeDetector()
        result = det.detect(df)
        assert result.column_info["x"].null_rate == 0.4


# ============================================================
# DetectionResult
# ============================================================

class TestDetectionResult:
    def _make_result(self):
        det = TypeDetector()
        df = pd.DataFrame({
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat": ["a", "b", "c", "a", "b"],
            "flag": [0, 1, 0, 1, 0],
            "const": [5, 5, 5, 5, 5],
        })
        return det.detect(df)

    def test_numeric_columns(self):
        result = self._make_result()
        assert "num" in result.numeric_columns

    def test_categorical_columns(self):
        result = self._make_result()
        cats = result.categorical_columns
        assert "cat" in cats or "flag" in cats

    def test_binary_columns(self):
        result = self._make_result()
        assert "flag" in result.binary_columns

    def test_constant_columns(self):
        result = self._make_result()
        assert "const" in result.constant_columns

    def test_summary_table(self):
        result = self._make_result()
        table = result.summary_table()
        assert isinstance(table, pd.DataFrame)
        assert "列名" in table.columns
        assert len(table) == 4

    def test_ignored_columns(self):
        result = self._make_result()
        ignored = result.ignored_columns
        assert "const" in ignored

    def test_get_columns_by_type(self):
        result = self._make_result()
        consts = result.get_columns_by_type(ColumnType.CONSTANT)
        assert "const" in consts
