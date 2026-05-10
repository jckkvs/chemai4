"""
tests/test_type_detector_comprehensive.py

TypeDetector / ColumnInfo / DetectionResult の包括テスト。
全列タイプ（BINARY/CATEGORY/NUMERIC/SMILES/DATETIME/TEXT/CONSTANT/PERIODIC）の
判定ロジックとエッジケースを網羅。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.data.type_detector import (
    TypeDetector,
    ColumnType,
    ColumnInfo,
    DetectionResult,
)


# ============================================================
# ColumnInfo テスト
# ============================================================

class TestColumnInfo:
    def test_is_numeric(self):
        for ct in [ColumnType.NUMERIC_NORMAL, ColumnType.NUMERIC_LOG,
                    ColumnType.NUMERIC_POWER, ColumnType.NUMERIC_OUTLIER]:
            info = ColumnInfo(name="x", col_type=ct, n_unique=10, null_rate=0.0)
            assert info.is_numeric

    def test_is_not_numeric(self):
        for ct in [ColumnType.BINARY, ColumnType.CATEGORY_LOW, ColumnType.SMILES]:
            info = ColumnInfo(name="x", col_type=ct, n_unique=2, null_rate=0.0)
            assert not info.is_numeric

    def test_is_categorical(self):
        for ct in [ColumnType.BINARY, ColumnType.CATEGORY_LOW, ColumnType.CATEGORY_HIGH]:
            info = ColumnInfo(name="x", col_type=ct, n_unique=3, null_rate=0.0)
            assert info.is_categorical

    def test_is_not_categorical(self):
        info = ColumnInfo(name="x", col_type=ColumnType.NUMERIC_NORMAL, n_unique=100, null_rate=0.0)
        assert not info.is_categorical


# ============================================================
# TypeDetector テスト
# ============================================================

class TestTypeDetector:
    @pytest.fixture
    def detector(self):
        return TypeDetector()

    def test_constant_column(self, detector):
        df = pd.DataFrame({"const": [1, 1, 1, 1, 1]})
        result = detector.detect(df)
        assert result.column_info["const"].col_type == ColumnType.CONSTANT
        assert "const" in result.constant_columns

    def test_binary_numeric(self, detector):
        df = pd.DataFrame({"bin": [0, 1, 0, 1, 0, 1, 0, 1]})
        result = detector.detect(df)
        assert result.column_info["bin"].col_type == ColumnType.BINARY

    def test_binary_string(self, detector):
        df = pd.DataFrame({"bin": ["yes", "no", "yes", "no", "yes", "no"]})
        result = detector.detect(df)
        assert result.column_info["bin"].col_type == ColumnType.BINARY

    def test_category_low(self, detector):
        df = pd.DataFrame({"cat": ["a", "b", "c", "a", "b", "c", "a", "b"]})
        result = detector.detect(df)
        assert result.column_info["cat"].col_type == ColumnType.CATEGORY_LOW

    def test_category_high(self, detector):
        """ユニーク数がcardinality_thresholdを超える場合"""
        det = TypeDetector(cardinality_threshold=3)
        df = pd.DataFrame({"cat": ["a", "b", "c", "d", "e", "f", "g", "h"]})
        result = det.detect(df)
        assert result.column_info["cat"].col_type == ColumnType.CATEGORY_HIGH

    def test_numeric_normal(self, detector):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"num": rng.randn(100)})
        result = detector.detect(df)
        info = result.column_info["num"]
        assert info.is_numeric
        # 正規分布データは NUMERIC_NORMAL のはず
        assert info.col_type == ColumnType.NUMERIC_NORMAL

    def test_numeric_log_skewed(self, detector):
        """右裾重い正値データ → NUMERIC_LOG"""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"num": np.exp(rng.randn(100) * 2)})
        result = detector.detect(df)
        info = result.column_info["num"]
        assert info.col_type in {ColumnType.NUMERIC_LOG, ColumnType.NUMERIC_POWER}

    def test_datetime_column(self, detector):
        df = pd.DataFrame({"dt": pd.date_range("2020-01-01", periods=5)})
        result = detector.detect(df)
        assert result.column_info["dt"].col_type == ColumnType.DATETIME
        assert "dt" in result.datetime_columns

    def test_datetime_string(self, detector):
        df = pd.DataFrame({"dt": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]})
        result = detector.detect(df)
        assert result.column_info["dt"].col_type == ColumnType.DATETIME

    def test_text_column(self, detector):
        texts = [f"This is a very long text about topic {i} " * 10 for i in range(10)]
        df = pd.DataFrame({"text": texts})
        result = detector.detect(df)
        assert result.column_info["text"].col_type == ColumnType.TEXT
        assert "text" in result.text_columns

    def test_smiles_by_name(self, detector):
        df = pd.DataFrame({"SMILES": ["CCO", "c1ccccc1", "CC(=O)O", "C1CCCCC1", "CCN"]})
        result = detector.detect(df)
        assert result.column_info["SMILES"].col_type == ColumnType.SMILES
        assert "SMILES" in result.smiles_columns

    def test_periodic_column(self):
        det = TypeDetector(periodic_cols=["angle"])
        df = pd.DataFrame({"angle": [0, 90, 180, 270, 360]})
        result = det.detect(df)
        assert result.column_info["angle"].col_type == ColumnType.PERIODIC

    def test_null_handling(self, detector):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan, 5.0]})
        result = detector.detect(df)
        info = result.column_info["x"]
        assert info.null_rate == pytest.approx(0.4)


# ============================================================
# DetectionResult テスト
# ============================================================

class TestDetectionResult:
    def test_summary_table(self):
        detector = TypeDetector()
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "num": rng.randn(20),
            "cat": ["a", "b"] * 10,
            "const": [1] * 20,
        })
        result = detector.detect(df)
        table = result.summary_table()
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 3
        assert "列名" in table.columns

    def test_numeric_columns(self):
        detector = TypeDetector()
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"num": rng.randn(20), "cat": ["a", "b"] * 10})
        result = detector.detect(df)
        assert "num" in result.numeric_columns
        assert "cat" not in result.numeric_columns

    def test_categorical_columns(self):
        detector = TypeDetector()
        df = pd.DataFrame({"cat": ["a", "b", "c"] * 5, "num": range(15)})
        result = detector.detect(df)
        assert "cat" in result.categorical_columns

    def test_ignored_columns(self):
        detector = TypeDetector()
        df = pd.DataFrame({"const": [1] * 10, "num": range(10)})
        result = detector.detect(df)
        assert "const" in result.ignored_columns

    def test_get_columns_by_type(self):
        detector = TypeDetector()
        df = pd.DataFrame({"a": [0, 1, 0, 1], "b": range(4)})
        result = detector.detect(df)
        binary_cols = result.get_columns_by_type(ColumnType.BINARY)
        assert isinstance(binary_cols, list)
