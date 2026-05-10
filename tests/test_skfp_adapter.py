# -*- coding: utf-8 -*-
"""
tests/test_skfp_adapter.py

scikit-fingerprints アダプタ (SkfpAdapter) のユニットテスト。
"""
from __future__ import annotations

import pytest

TEST_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCCCCC"]
INVALID_SMILES = ["invalid_smi", "XXX"]


class TestSkfpAdapter:
    """scikit-fingerprints アダプタのテスト"""

    def test_name(self):
        from backend.chem.skfp_adapter import SkfpAdapter
        assert SkfpAdapter().name == "skfp"

    def test_is_available_returns_bool(self):
        from backend.chem.skfp_adapter import SkfpAdapter
        assert isinstance(SkfpAdapter().is_available(), bool)

    def test_description_not_empty(self):
        from backend.chem.skfp_adapter import SkfpAdapter
        assert len(SkfpAdapter().description) > 0

    def test_get_descriptors_metadata(self):
        from backend.chem.skfp_adapter import SkfpAdapter
        meta = SkfpAdapter().get_descriptors_metadata()
        assert isinstance(meta, list)
        assert len(meta) > 0

    @pytest.mark.skipif(
        not __import__("backend.chem.skfp_adapter", fromlist=["SkfpAdapter"]).SkfpAdapter().is_available(),
        reason="scikit-fingerprints not installed"
    )
    def test_compute_returns_descriptor_result(self):
        from backend.chem.skfp_adapter import SkfpAdapter
        adapter = SkfpAdapter(fp_types=["ECFP"])
        result = adapter.compute(TEST_SMILES)
        assert result.descriptors.shape[0] == len(TEST_SMILES)
        assert result.adapter_name == "skfp"

    @pytest.mark.skipif(
        not __import__("backend.chem.skfp_adapter", fromlist=["SkfpAdapter"]).SkfpAdapter().is_available(),
        reason="scikit-fingerprints not installed"
    )
    def test_compute_with_invalid_smiles(self):
        from backend.chem.skfp_adapter import SkfpAdapter
        adapter = SkfpAdapter(fp_types=["ECFP"])
        result = adapter.compute(INVALID_SMILES)
        assert result.descriptors.shape[0] == len(INVALID_SMILES)
        assert len(result.failed_indices) > 0

    def test_custom_fp_types(self):
        from backend.chem.skfp_adapter import SkfpAdapter
        adapter = SkfpAdapter(fp_types=["ECFP", "MACCS"])
        assert adapter._fp_types == ["ECFP", "MACCS"]

    def test_custom_fp_configs(self):
        from backend.chem.skfp_adapter import SkfpAdapter
        adapter = SkfpAdapter(fp_configs={"ECFP": {"radius": 3}})
        assert adapter._fp_configs["ECFP"]["radius"] == 3
