"""
tests/test_cosmo_adapter.py

backend/chem/cosmo_adapter.py のユニットテスト。

openCOSMO-RS がインストール済みでもテスト可能な設計。
"""
from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from backend.chem.cosmo_adapter import CosmoAdapter, _COSMO_DESCRIPTORS


class TestCosmoAdapterProperties:
    """プロパティの基本テスト。"""

    def test_name(self):
        adapter = CosmoAdapter()
        assert adapter.name == "cosmo_rs"

    def test_description(self):
        adapter = CosmoAdapter()
        assert "COSMO" in adapter.description

    def test_get_descriptor_names(self):
        adapter = CosmoAdapter()
        names = adapter.get_descriptor_names()
        assert "mu_comb" in names
        assert "mu_res" in names
        assert "ln_gamma" in names

    def test_get_descriptors_metadata(self):
        adapter = CosmoAdapter()
        meta = adapter.get_descriptors_metadata()
        assert len(meta) == 3
        assert meta[0].name == "mu_comb"
        assert meta[0].is_count is False

    def test_parameterization_default(self):
        adapter = CosmoAdapter()
        assert adapter._par == "default_turbomole"

    def test_parameterization_custom(self):
        adapter = CosmoAdapter(parameterization="custom_par")
        assert adapter._par == "custom_par"


class TestCosmoAdapterAvailability:
    """is_available のテスト。"""

    def test_is_available_returns_bool(self):
        adapter = CosmoAdapter()
        available = adapter.is_available()
        assert isinstance(available, bool)


class TestCosmoAdapterComputeNoCosmiFiles:
    """cosmi_files が与えられない場合のテスト。"""

    def test_no_cosmi_files_returns_nan(self):
        """cosmi_files 未指定 → 全NaN"""
        mock_module = MagicMock()
        adapter = CosmoAdapter()

        with patch.object(adapter, "_require_available"):
            with patch.dict(sys.modules, {"opencosmorspy": mock_module}):
                result = adapter.compute(["CCO", "CC"])
                assert result.descriptors.shape == (2, 3)
                assert result.descriptors.isna().all().all()
                assert result.failed_indices == [0, 1]
                assert result.adapter_name == "cosmo_rs"

    def test_empty_cosmi_files(self):
        """空リスト → 全NaN"""
        mock_module = MagicMock()
        adapter = CosmoAdapter()

        with patch.object(adapter, "_require_available"):
            with patch.dict(sys.modules, {"opencosmorspy": mock_module}):
                result = adapter.compute(["CCO"], cosmi_files=[])
                assert result.descriptors.shape == (1, 3)
                assert result.descriptors.isna().all().all()


class TestCosmoAdapterComputeWithCosmiFiles:
    """cosmi_files が与えられる場合のテスト。"""

    def _make_mock_module(self, crs_instance):
        """COSMORSクラスを含むモックモジュールを作成。"""
        mock_module = MagicMock()
        mock_module.COSMORS = MagicMock(return_value=crs_instance)
        return mock_module

    def test_missing_file_returns_nan(self):
        """存在しないCOSMIファイル → NaN"""
        mock_crs = MagicMock()
        mock_module = self._make_mock_module(mock_crs)
        adapter = CosmoAdapter()

        with patch.object(adapter, "_require_available"):
            with patch.dict(sys.modules, {"opencosmorspy": mock_module}):
                result = adapter.compute(
                    ["CCO"],
                    cosmi_files=["/nonexistent/path_9999.cosmi"]
                )
                assert result.descriptors.shape == (1, 3)
                assert result.descriptors.isna().all().all()

    def test_fewer_cosmi_files_than_smiles(self):
        """COSMIファイル数 < SMILES数 → 足りない分はNaN"""
        mock_crs = MagicMock()
        mock_module = self._make_mock_module(mock_crs)
        adapter = CosmoAdapter()

        with patch.object(adapter, "_require_available"):
            with patch.dict(sys.modules, {"opencosmorspy": mock_module}):
                result = adapter.compute(
                    ["CCO", "CC", "CCC"],
                    cosmi_files=["/nonexistent_9999.cosmi"]
                )
                assert result.descriptors.shape == (3, 3)
                assert result.descriptors.isna().all().all()

    def test_successful_calculation_with_dict_result(self):
        """COSMORS計算成功時（dict結果）"""
        mock_crs = MagicMock()
        mock_crs.calculate.return_value = {
            "mu_comb": [1.5],
            "mu_res": [2.3],
            "ln_gamma": [0.5],
        }
        mock_module = self._make_mock_module(mock_crs)
        adapter = CosmoAdapter()

        with tempfile.NamedTemporaryFile(suffix=".cosmi", delete=False) as f:
            tmp_path = f.name
        try:
            with patch.object(adapter, "_require_available"):
                with patch.dict(sys.modules, {"opencosmorspy": mock_module}):
                    result = adapter.compute(
                        ["CCO"],
                        cosmi_files=[tmp_path]
                    )
                    assert result.descriptors.shape == (1, 3)
                    assert result.descriptors["mu_comb"].iloc[0] == pytest.approx(1.5)
                    assert result.descriptors["mu_res"].iloc[0] == pytest.approx(2.3)
                    assert result.descriptors["ln_gamma"].iloc[0] == pytest.approx(0.5)
        finally:
            os.unlink(tmp_path)

    def test_calculation_exception_returns_nan(self):
        """計算で例外 → NaN"""
        mock_crs = MagicMock()
        mock_crs.calculate.side_effect = RuntimeError("計算エラー")
        mock_module = self._make_mock_module(mock_crs)
        adapter = CosmoAdapter()

        with tempfile.NamedTemporaryFile(suffix=".cosmi", delete=False) as f:
            tmp_path = f.name
        try:
            with patch.object(adapter, "_require_available"):
                with patch.dict(sys.modules, {"opencosmorspy": mock_module}):
                    result = adapter.compute(
                        ["CCO"],
                        cosmi_files=[tmp_path]
                    )
                    assert result.descriptors.shape == (1, 3)
                    assert result.descriptors.isna().all().all()
        finally:
            os.unlink(tmp_path)

    def test_non_dict_result_returns_nan(self):
        """COSMORS が dict 以外を返す → NaN"""
        mock_crs = MagicMock()
        mock_crs.calculate.return_value = "not_a_dict"
        mock_module = self._make_mock_module(mock_crs)
        adapter = CosmoAdapter()

        with tempfile.NamedTemporaryFile(suffix=".cosmi", delete=False) as f:
            tmp_path = f.name
        try:
            with patch.object(adapter, "_require_available"):
                with patch.dict(sys.modules, {"opencosmorspy": mock_module}):
                    result = adapter.compute(
                        ["CCO"],
                        cosmi_files=[tmp_path]
                    )
                    assert result.descriptors.shape == (1, 3)
                    assert result.descriptors.isna().all().all()
        finally:
            os.unlink(tmp_path)


class TestCosmoDescriptorConstants:
    """記述子定数の整合性。"""

    def test_descriptor_keys(self):
        assert set(_COSMO_DESCRIPTORS.keys()) == {"mu_comb", "mu_res", "ln_gamma"}

    def test_descriptor_values_are_japanese(self):
        for desc in _COSMO_DESCRIPTORS.values():
            assert isinstance(desc, str)
            assert len(desc) > 0
"""
Implements: F-CHEM-COSMO-001
"""
