# -*- coding: utf-8 -*-
"""
tests/test_mol2vec_adapter.py

Mol2Vec アダプタ (Mol2VecAdapter) のユニットテスト。
"""
from __future__ import annotations

import pytest

TEST_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCCCCC"]


class TestMol2VecAdapter:
    """Mol2Vec アダプタのテスト"""

    def test_name(self):
        from backend.chem.mol2vec_adapter import Mol2VecAdapter
        assert Mol2VecAdapter().name == "mol2vec"

    def test_is_available_returns_bool(self):
        from backend.chem.mol2vec_adapter import Mol2VecAdapter
        assert isinstance(Mol2VecAdapter().is_available(), bool)

    def test_description_not_empty(self):
        from backend.chem.mol2vec_adapter import Mol2VecAdapter
        assert len(Mol2VecAdapter().description) > 0

    def test_get_descriptors_metadata(self):
        from backend.chem.mol2vec_adapter import Mol2VecAdapter
        meta = Mol2VecAdapter().get_descriptors_metadata()
        assert len(meta) > 0

    def test_init_params(self):
        from backend.chem.mol2vec_adapter import Mol2VecAdapter
        adapter = Mol2VecAdapter(radius=2)
        assert adapter._radius == 2

    @pytest.mark.skipif(
        not __import__("backend.chem.mol2vec_adapter", fromlist=["Mol2VecAdapter"]).Mol2VecAdapter().is_available(),
        reason="mol2vec not installed"
    )
    def test_compute_returns_correct_shape(self):
        from backend.chem.mol2vec_adapter import Mol2VecAdapter
        adapter = Mol2VecAdapter()
        result = adapter.compute(TEST_SMILES)
        assert result.descriptors.shape[0] == len(TEST_SMILES)
        assert result.adapter_name == "mol2vec"
