# -*- coding: utf-8 -*-
"""
tests/test_chemprop_adapter.py

Chemprop アダプタ (ChempropAdapter) のユニットテスト。
"""
from __future__ import annotations

import pytest

TEST_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCCCCC"]


class TestChempropAdapter:
    """Chemprop (D-MPNN) アダプタのテスト"""

    def test_name(self):
        from backend.chem.chemprop_adapter import ChempropAdapter
        assert ChempropAdapter().name == "chemprop"

    def test_is_available_returns_bool(self):
        from backend.chem.chemprop_adapter import ChempropAdapter
        assert isinstance(ChempropAdapter().is_available(), bool)

    def test_description_not_empty(self):
        from backend.chem.chemprop_adapter import ChempropAdapter
        assert len(ChempropAdapter().description) > 0

    def test_get_descriptors_metadata(self):
        from backend.chem.chemprop_adapter import ChempropAdapter
        meta = ChempropAdapter().get_descriptors_metadata()
        assert len(meta) > 0

    def test_init_params(self):
        from backend.chem.chemprop_adapter import ChempropAdapter
        adapter = ChempropAdapter(features_dim=128)
        assert adapter._features_dim == 128

    @pytest.mark.skipif(
        not __import__("backend.chem.chemprop_adapter", fromlist=["ChempropAdapter"]).ChempropAdapter().is_available(),
        reason="chemprop not installed"
    )
    def test_compute_basic(self):
        from backend.chem.chemprop_adapter import ChempropAdapter
        adapter = ChempropAdapter()
        result = adapter.compute(TEST_SMILES)
        assert result.descriptors.shape[0] == len(TEST_SMILES)
        assert result.adapter_name == "chemprop"
