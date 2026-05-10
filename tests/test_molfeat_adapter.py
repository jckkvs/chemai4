# -*- coding: utf-8 -*-
"""
tests/test_molfeat_adapter.py

Molfeat アダプタ (MolfeatAdapter) のユニットテスト。
"""
from __future__ import annotations

import pytest

TEST_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCCCCC"]


class TestMolfeatAdapter:
    """Molfeat アダプタのテスト"""

    def test_name(self):
        from backend.chem.molfeat_adapter import MolfeatAdapter
        assert MolfeatAdapter().name == "molfeat"

    def test_is_available_returns_bool(self):
        from backend.chem.molfeat_adapter import MolfeatAdapter
        assert isinstance(MolfeatAdapter().is_available(), bool)

    def test_description_not_empty(self):
        from backend.chem.molfeat_adapter import MolfeatAdapter
        assert len(MolfeatAdapter().description) > 0

    def test_get_descriptors_metadata(self):
        from backend.chem.molfeat_adapter import MolfeatAdapter
        meta = MolfeatAdapter().get_descriptors_metadata()
        assert len(meta) > 0

    def test_init_calculator_type(self):
        from backend.chem.molfeat_adapter import MolfeatAdapter
        adapter = MolfeatAdapter(calculator_type="maccs")
        assert adapter._calculator_type == "maccs"

    @pytest.mark.skipif(
        not __import__("backend.chem.molfeat_adapter", fromlist=["MolfeatAdapter"]).MolfeatAdapter().is_available(),
        reason="molfeat not installed"
    )
    def test_compute_ecfp(self):
        from backend.chem.molfeat_adapter import MolfeatAdapter
        adapter = MolfeatAdapter(calculator_type="ecfp")
        result = adapter.compute(TEST_SMILES)
        assert result.descriptors.shape[0] == len(TEST_SMILES)
        assert result.adapter_name == "molfeat"
