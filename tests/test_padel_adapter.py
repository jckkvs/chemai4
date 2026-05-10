# -*- coding: utf-8 -*-
"""
tests/test_padel_adapter.py

PaDEL-Descriptor アダプタ (PaDELAdapter) のユニットテスト。
"""
from __future__ import annotations

import pytest


class TestPaDELAdapter:
    """PaDEL-Descriptor アダプタのテスト"""

    def test_name(self):
        from backend.chem.padel_adapter import PaDELAdapter
        assert PaDELAdapter().name == "padel"

    def test_is_available_returns_bool(self):
        from backend.chem.padel_adapter import PaDELAdapter
        assert isinstance(PaDELAdapter().is_available(), bool)

    def test_description_not_empty(self):
        from backend.chem.padel_adapter import PaDELAdapter
        assert len(PaDELAdapter().description) > 0

    def test_get_descriptors_metadata(self):
        from backend.chem.padel_adapter import PaDELAdapter
        meta = PaDELAdapter().get_descriptors_metadata()
        assert len(meta) > 0

    def test_init_params(self):
        from backend.chem.padel_adapter import PaDELAdapter
        adapter = PaDELAdapter(compute_fingerprints=True, timeout=60)
        assert adapter._compute_fp is True
        assert adapter._timeout == 60

    @pytest.mark.skipif(
        not __import__("backend.chem.padel_adapter", fromlist=["PaDELAdapter"]).PaDELAdapter().is_available(),
        reason="padelpy not installed"
    )
    def test_compute_basic(self):
        from backend.chem.padel_adapter import PaDELAdapter
        adapter = PaDELAdapter(timeout=30)
        result = adapter.compute(["CCO"])
        assert result.descriptors.shape[0] == 1
        assert result.adapter_name == "padel"
