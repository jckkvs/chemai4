# -*- coding: utf-8 -*-
"""
tests/test_psmiles_adapter.py

PSmilesAdapter（ポリマーSMILES → 記述子）のユニットテスト。
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

requires_rdkit = pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit未インストール")


@requires_rdkit
class TestPSmilesAdapter:

    @pytest.fixture(autouse=True)
    def _adapter(self):
        from backend.chem.psmiles_adapter import PSmilesAdapter
        self.adp = PSmilesAdapter()

    def test_is_available(self):
        assert self.adp.is_available() is True

    def test_name(self):
        assert self.adp.name == "psmiles"

    def test_description(self):
        assert "ポリマー" in self.adp.description or "PSMILES" in self.adp.description

    # ── is_psmiles 判定 ──

    def test_is_psmiles_true(self):
        assert self.adp.is_psmiles("[*]CC[*]") is True
        assert self.adp.is_psmiles("*CC*") is True

    def test_is_psmiles_false(self):
        assert self.adp.is_psmiles("CCO") is False
        assert self.adp.is_psmiles("c1ccccc1") is False

    def test_is_psmiles_non_string(self):
        assert self.adp.is_psmiles(None) is False
        assert self.adp.is_psmiles(123) is False

    # ── fallback_process_psmiles ──

    def test_fallback_simple_psmiles(self):
        """[*]CC[*] → [CH3]CC[CH3] で有効なMolが返る"""
        mol = self.adp._fallback_process_psmiles("[*]CC[*]")
        assert mol is not None

    def test_fallback_isotope_psmiles(self):
        """[*:1]CC[*:2] → [CH3]CC[CH3]"""
        mol = self.adp._fallback_process_psmiles("[*:1]CC[*:2]")
        assert mol is not None

    def test_fallback_star_only(self):
        """*CC* 形式"""
        mol = self.adp._fallback_process_psmiles("*CC*")
        assert mol is not None

    # ── compute ──

    def test_compute_psmiles_basic(self):
        """基本的なPSMILES計算"""
        result = self.adp.compute(["[*]CC[*]", "[*]c1ccccc1[*]"])
        assert result.descriptors.shape[0] == 2
        assert "PSMILES_MonomerWt" in result.descriptors.columns
        assert result.success_rate >= 0.5

    def test_compute_has_expected_columns(self):
        """期待するカラムが存在"""
        result = self.adp.compute(["[*]CCO[*]"])
        cols = result.descriptors.columns.tolist()
        assert "PSMILES_MonomerWt" in cols
        assert "PSMILES_NumHDonors" in cols
        assert "PSMILES_NumHAcceptors" in cols
        assert "PSMILES_MolLogP" in cols

    def test_compute_morgan_fp_present(self):
        """Morgan FP列が生成される"""
        result = self.adp.compute(["[*]c1ccccc1[*]"])
        morgan_cols = [c for c in result.descriptors.columns if "Morgan" in c]
        assert len(morgan_cols) > 0

    def test_compute_empty_list(self):
        result = self.adp.compute([])
        assert result.descriptors.shape[0] == 0

    def test_compute_invalid_input(self):
        """不正入力は失敗インデックスに記録"""
        result = self.adp.compute(["", None, 123])
        assert len(result.failed_indices) >= 2

    def test_compute_mixed_valid_invalid(self):
        """有効・無効混在"""
        result = self.adp.compute(["[*]CC[*]", "INVALID###", "[*]CCO[*]"])
        assert result.descriptors.shape[0] == 3
        assert len(result.failed_indices) >= 1

    def test_compute_regular_smiles_treated_as_psmiles(self):
        """通常SMILESも処理される（*なしでもMol生成可能なため）"""
        result = self.adp.compute(["CCO"])
        assert result.descriptors.shape[0] == 1

    def test_adapter_name_in_result(self):
        result = self.adp.compute(["[*]CC[*]"])
        assert result.adapter_name == "psmiles"

    def test_metadata_has_psmiles_lib(self):
        result = self.adp.compute(["[*]CC[*]"])
        assert "has_psmiles_lib" in result.metadata
