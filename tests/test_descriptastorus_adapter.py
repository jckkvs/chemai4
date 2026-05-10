# -*- coding: utf-8 -*-
"""
tests/test_descriptastorus_adapter.py

DescriptaStorus アダプタ (DescriptaStorusAdapter) のユニットテスト。
"""
from __future__ import annotations

import pytest

TEST_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCCCCC"]


class TestDescriptaStorusAdapter:
    """DescriptaStorus アダプタのテスト"""

    def test_name(self):
        from backend.chem.descriptastorus_adapter import DescriptaStorusAdapter
        assert DescriptaStorusAdapter().name == "descriptastorus"

    def test_is_available_returns_bool(self):
        from backend.chem.descriptastorus_adapter import DescriptaStorusAdapter
        assert isinstance(DescriptaStorusAdapter().is_available(), bool)

    def test_description_not_empty(self):
        from backend.chem.descriptastorus_adapter import DescriptaStorusAdapter
        assert len(DescriptaStorusAdapter().description) > 0

    def test_init_descriptor_type(self):
        from backend.chem.descriptastorus_adapter import DescriptaStorusAdapter
        adapter = DescriptaStorusAdapter(descriptor_type="rdkit2dnormalized")
        assert adapter._descriptor_type == "rdkit2dnormalized"

    def test_get_descriptors_metadata(self):
        from backend.chem.descriptastorus_adapter import DescriptaStorusAdapter
        meta = DescriptaStorusAdapter().get_descriptors_metadata()
        assert len(meta) > 0

    @pytest.mark.skipif(
        not __import__("backend.chem.descriptastorus_adapter", fromlist=["DescriptaStorusAdapter"]).DescriptaStorusAdapter().is_available(),
        reason="descriptastorus not installed"
    )
    def test_compute_basic(self):
        from backend.chem.descriptastorus_adapter import DescriptaStorusAdapter
        adapter = DescriptaStorusAdapter()
        result = adapter.compute(TEST_SMILES)
        assert result.descriptors.shape[0] == len(TEST_SMILES)
        assert result.adapter_name == "descriptastorus"
