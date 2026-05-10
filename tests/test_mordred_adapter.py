# -*- coding: utf-8 -*-
"""
tests/test_mordred_adapter.py

mordred_adapter.py（Mordred記述子計算アダプタ）のユニットテスト。

カバー対象:
  - MordredAdapter初期化、name/description
  - is_available() 判定
  - compute() — 正常ケース、無効SMILES、混合入力
  - get_descriptor_names() / get_descriptors_metadata()
  - SELECTED_DESCRIPTORS の整合性
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.chem.mordred_adapter import MordredAdapter


# ═══════════════════════════════════════════════════════════════════
# 基本属性テスト
# ═══════════════════════════════════════════════════════════════════

class TestMordredAdapterProperties:

    def test_name(self):
        adapter = MordredAdapter()
        assert adapter.name == "mordred"

    def test_description(self):
        adapter = MordredAdapter()
        assert "Mordred" in adapter.description

    def test_is_available(self):
        adapter = MordredAdapter()
        result = adapter.is_available()
        assert isinstance(result, bool)

    def test_selected_descriptors_not_empty(self):
        assert len(MordredAdapter.SELECTED_DESCRIPTORS) > 50

    def test_default_selected_only_true(self):
        adapter = MordredAdapter()
        assert adapter.selected_only is True


# ═══════════════════════════════════════════════════════════════════
# get_descriptor_names / get_descriptors_metadata
# ═══════════════════════════════════════════════════════════════════

class TestDescriptorNames:

    def test_get_descriptor_names_selected(self):
        adapter = MordredAdapter(selected_only=True)
        names = adapter.get_descriptor_names()
        assert names == MordredAdapter.SELECTED_DESCRIPTORS

    def test_get_descriptor_names_all(self):
        adapter = MordredAdapter(selected_only=False)
        if not adapter.is_available():
            pytest.skip("mordred not installed")
        try:
            names = adapter.get_descriptor_names()
            assert len(names) > len(MordredAdapter.SELECTED_DESCRIPTORS)
        except ImportError:
            pytest.skip("mordred runtime error (numpy incompatibility)")

    def test_get_descriptors_metadata(self):
        adapter = MordredAdapter()
        if not adapter.is_available():
            pytest.skip("mordred not installed")
        metadata = adapter.get_descriptors_metadata()
        assert len(metadata) > 0
        # メタデータの各エントリにnameがある
        for m in metadata:
            assert hasattr(m, "name")
            assert hasattr(m, "meaning")


# ═══════════════════════════════════════════════════════════════════
# compute テスト
# ═══════════════════════════════════════════════════════════════════

class TestMordredCompute:

    @pytest.fixture(autouse=True)
    def _skip_if_unavailable(self):
        adapter = MordredAdapter()
        if not adapter.is_available():
            pytest.skip("mordred/rdkit not installed")
        # mordred 1.2.0 + numpy 2.x 互換性チェック
        try:
            from mordred import Calculator, descriptors
            calc = Calculator(descriptors, ignore_3D=True)
            from rdkit import Chem
            mol = Chem.MolFromSmiles("C")
            calc.pandas([mol])
        except Exception as e:
            pytest.skip(f"mordred runtime error (numpy incompatibility): {e}")

    def test_basic_compute(self):
        adapter = MordredAdapter()
        result = adapter.compute(["CCO", "c1ccccc1", "CCC"])
        assert result.descriptors.shape[0] == 3
        assert result.descriptors.shape[1] > 0
        assert result.adapter_name == "mordred"

    def test_invalid_smiles(self):
        adapter = MordredAdapter()
        result = adapter.compute(["CCO", "INVALID_SMILES", "CCC"])
        assert result.descriptors.shape[0] == 3
        assert 1 in result.failed_indices

    def test_empty_smiles(self):
        adapter = MordredAdapter()
        result = adapter.compute(["CCO", "", "CCC"])
        assert result.descriptors.shape[0] == 3
        assert 1 in result.failed_indices

    def test_all_invalid(self):
        adapter = MordredAdapter()
        result = adapter.compute(["INVALID1", "INVALID2"])
        assert result.descriptors.shape[0] == 2
        assert len(result.failed_indices) == 2

    def test_selected_only_limits_columns(self):
        adapter = MordredAdapter(selected_only=True)
        result = adapter.compute(["CCO"])
        # 選択モードでは列数がSELECTED_DESCRIPTORSの数以下
        assert result.descriptors.shape[1] <= len(MordredAdapter.SELECTED_DESCRIPTORS)

    def test_full_descriptors(self):
        adapter = MordredAdapter(selected_only=False)
        result = adapter.compute(["CCO"])
        # 全記述子モードでは列数が大幅に多い
        assert result.descriptors.shape[1] > len(MordredAdapter.SELECTED_DESCRIPTORS)

    def test_no_inf_values(self):
        adapter = MordredAdapter()
        result = adapter.compute(["CCO", "c1ccccc1"])
        assert not np.any(np.isinf(result.descriptors.values))

    def test_metadata_in_result(self):
        adapter = MordredAdapter()
        result = adapter.compute(["CCO"])
        assert "use_3d" in result.metadata
        assert "n_descriptors" in result.metadata
        assert "selected_only" in result.metadata

    def test_smiles_list_preserved(self):
        smiles = ["CCO", "c1ccccc1"]
        adapter = MordredAdapter()
        result = adapter.compute(smiles)
        assert result.smiles_list == smiles
