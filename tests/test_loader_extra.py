"""
tests/test_loader_extra.py

loader.py のカバレッジ改善テスト。
load_file, load_from_bytes, save_dataframe, get_supported_extensions を網羅。
"""
from __future__ import annotations

import os
import pytest
import tempfile
import numpy as np
import pandas as pd

from backend.data.loader import (
    load_file,
    load_from_bytes,
    save_dataframe,
    get_supported_extensions,
    _SUPPORTED_EXTENSIONS,
)


# ============================================================
# テスト用ファイル作成ヘルパー
# ============================================================

def _write_csv(path, df):
    df.to_csv(path, index=False)


def _write_tsv(path, df):
    df.to_csv(path, index=False, sep="\t")


def _write_json(path, df):
    df.to_json(path, orient="records")


# ============================================================
# load_file
# ============================================================

class TestLoadFile:
    def test_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        p = tmp_path / "test.csv"
        _write_csv(p, df)
        result = load_file(p)
        assert result.shape == (3, 2)
        assert list(result.columns) == ["a", "b"]

    def test_tsv(self, tmp_path):
        df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
        p = tmp_path / "test.tsv"
        _write_tsv(p, df)
        result = load_file(p)
        assert result.shape == (2, 2)

    def test_json(self, tmp_path):
        df = pd.DataFrame({"n": [1, 2]})
        p = tmp_path / "data.json"
        _write_json(p, df)
        result = load_file(p)
        assert len(result) == 2

    def test_parquet(self, tmp_path):
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
        p = tmp_path / "data.parquet"
        df.to_parquet(p, index=False)
        result = load_file(p)
        assert result.shape == (3, 1)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="見つかりません"):
            load_file("/nonexistent/path/test.csv")

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "test.xyz"
        p.write_text("data")
        with pytest.raises(ValueError, match="未対応"):
            load_file(p)


# ============================================================
# load_from_bytes
# ============================================================

class TestLoadFromBytes:
    def test_csv_bytes(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        content = df.to_csv(index=False).encode()
        result = load_from_bytes(content, "test.csv")
        assert result.shape == (2, 2)

    def test_json_bytes(self):
        df = pd.DataFrame({"x": [10]})
        content = df.to_json(orient="records").encode()
        result = load_from_bytes(content, "data.json")
        assert len(result) == 1

    def test_unsupported_bytes(self):
        with pytest.raises(ValueError, match="未対応"):
            load_from_bytes(b"data", "test.xyz")


# ============================================================
# save_dataframe
# ============================================================

class TestSaveDataframe:
    def test_save_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        p = tmp_path / "out.csv"
        result = save_dataframe(df, p)
        assert result.exists()
        reloaded = pd.read_csv(p)
        assert len(reloaded) == 2

    def test_save_tsv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2]})
        p = tmp_path / "out.tsv"
        result = save_dataframe(df, p)
        assert result.exists()

    def test_save_json(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        p = tmp_path / "out.json"
        save_dataframe(df, p)
        assert p.exists()

    def test_save_parquet(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        p = tmp_path / "out.parquet"
        save_dataframe(df, p)
        assert p.exists()

    def test_save_unsupported(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        p = tmp_path / "out.xyz"
        with pytest.raises(ValueError, match="未対応"):
            save_dataframe(df, p, fmt="xyz")


# ============================================================
# get_supported_extensions
# ============================================================

class TestSupportedExtensions:
    def test_returns_list(self):
        exts = get_supported_extensions()
        assert isinstance(exts, list)
        assert ".csv" in exts
        assert ".xlsx" in exts

    def test_matches_internal(self):
        assert set(get_supported_extensions()) == set(_SUPPORTED_EXTENSIONS.keys())
