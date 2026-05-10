"""
tests/test_benchmark_datasets_extra.py

benchmark_datasets.py のカバレッジ改善テスト。
list_benchmark_datasets, load_benchmark, BENCHMARK_URLS を網羅。
ネットワークアクセスを避けるため、load_benchmark はモック使用。
"""
from __future__ import annotations

import io
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd

from backend.data.benchmark_datasets import (
    list_benchmark_datasets,
    load_benchmark,
    BENCHMARK_URLS,
    _db_cache,
)


# ============================================================
# list_benchmark_datasets
# ============================================================

class TestListBenchmarkDatasets:
    def test_returns_list(self):
        datasets = list_benchmark_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) >= 3

    def test_has_required_keys(self):
        datasets = list_benchmark_datasets()
        for ds in datasets:
            assert "id" in ds
            assert "name" in ds
            assert "description" in ds
            assert "target" in ds

    def test_known_ids(self):
        datasets = list_benchmark_datasets()
        ids = {d["id"] for d in datasets}
        assert "esol" in ids
        assert "freesolv" in ids
        assert "lipophilicity" in ids


# ============================================================
# BENCHMARK_URLS
# ============================================================

class TestBenchmarkURLs:
    def test_keys(self):
        assert "esol" in BENCHMARK_URLS
        assert "freesolv" in BENCHMARK_URLS
        assert "lipophilicity" in BENCHMARK_URLS

    def test_urls_are_strings(self):
        for key, url in BENCHMARK_URLS.items():
            assert isinstance(url, str)
            assert url.startswith("http")


# ============================================================
# load_benchmark
# ============================================================

class TestLoadBenchmark:
    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="未知"):
            load_benchmark("nonexistent")

    def test_load_from_cache(self):
        """キャッシュ済みデータを返す"""
        dummy_df = pd.DataFrame({"smiles": ["CCO", "CC"], "target": [1.0, 2.0]})
        _db_cache["esol"] = dummy_df
        try:
            result = load_benchmark("esol")
            assert len(result) == 2
            assert list(result.columns) == ["smiles", "target"]
            # copy が返されること
            assert result is not dummy_df
        finally:
            _db_cache.pop("esol", None)

    def test_load_with_mock_download(self):
        """ネットワークアクセスをモックしてダウンロードをテスト"""
        csv_content = b"smiles,target\nCCO,1.0\nCC,2.0\nO,0.5\n"
        mock_response = MagicMock()
        mock_response.read.return_value = csv_content
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        # キャッシュをクリア
        _db_cache.pop("esol", None)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = load_benchmark("esol")
            assert len(result) == 3
            assert "smiles" in result.columns
        
        # クリーンアップ
        _db_cache.pop("esol", None)

    def test_download_failure(self):
        """ダウンロード失敗時のエラー処理"""
        _db_cache.pop("freesolv", None)
        
        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            with pytest.raises(RuntimeError, match="失敗"):
                load_benchmark("freesolv")
