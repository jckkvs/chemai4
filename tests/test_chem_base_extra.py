"""
tests/test_chem_base_extra.py

backend/chem/base.py のカバレッジ改善テスト。
DescriptorResult, DescriptorMetadata, BaseChemAdapter を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.chem.base import (
    DescriptorResult,
    DescriptorMetadata,
    BaseChemAdapter,
)


# ============================================================
# DescriptorResult
# ============================================================

class TestDescriptorResult:
    def test_basic(self):
        df = pd.DataFrame({"d1": [1.0, 2.0], "d2": [3.0, 4.0]})
        dr = DescriptorResult(
            descriptors=df,
            smiles_list=["CCO", "CC"],
            failed_indices=[],
            adapter_name="test",
        )
        assert dr.n_descriptors == 2
        assert dr.success_rate == 1.0

    def test_with_failures(self):
        df = pd.DataFrame({"d1": [1.0, np.nan, 3.0]})
        dr = DescriptorResult(
            descriptors=df,
            smiles_list=["CCO", "INVALID", "CC"],
            failed_indices=[1],
            adapter_name="test",
        )
        assert dr.success_rate == pytest.approx(2 / 3)
        assert dr.n_descriptors == 1

    def test_empty_smiles_list(self):
        df = pd.DataFrame()
        dr = DescriptorResult(
            descriptors=df,
            smiles_list=[],
            failed_indices=[],
            adapter_name="test",
        )
        assert dr.success_rate == 0.0

    def test_metadata(self):
        df = pd.DataFrame({"d1": [1.0]})
        dr = DescriptorResult(
            descriptors=df,
            smiles_list=["CCO"],
            failed_indices=[],
            adapter_name="test",
            metadata={"key": "value"},
        )
        assert dr.metadata["key"] == "value"


# ============================================================
# DescriptorMetadata
# ============================================================

class TestDescriptorMetadata:
    def test_basic(self):
        dm = DescriptorMetadata(
            name="MW", meaning="分子量", is_count=False,
        )
        assert dm.name == "MW"
        assert dm.is_binary is False

    def test_binary(self):
        dm = DescriptorMetadata(
            name="has_ring", meaning="環の有無", is_count=False, is_binary=True,
        )
        assert dm.is_binary is True

    def test_with_description(self):
        dm = DescriptorMetadata(
            name="TPSA", meaning="極性表面積",
            is_count=False, description="topological polar surface area",
        )
        assert dm.description == "topological polar surface area"


# ============================================================
# BaseChemAdapter (具象テスト用サブクラス)
# ============================================================

class _DummyAdapter(BaseChemAdapter):
    @property
    def name(self): return "dummy"
    @property
    def description(self): return "テスト用ダミー"
    def is_available(self): return True
    def compute(self, smiles_list, **kwargs):
        rows = [{"d1": 1.0} for _ in smiles_list]
        return DescriptorResult(
            descriptors=pd.DataFrame(rows),
            smiles_list=smiles_list,
            failed_indices=[],
            adapter_name=self.name,
        )
    def get_descriptors_metadata(self):
        return [DescriptorMetadata("d1", "テスト記述子", False)]


class _UnavailableAdapter(BaseChemAdapter):
    @property
    def name(self): return "unavailable"
    @property
    def description(self): return "利用不可テスト"
    def is_available(self): return False
    def compute(self, smiles_list, **kwargs):
        self._require_available()


class TestBaseChemAdapter:
    def test_repr_available(self):
        adapter = _DummyAdapter()
        r = repr(adapter)
        assert "dummy" in r
        assert "available" in r

    def test_repr_unavailable(self):
        adapter = _UnavailableAdapter()
        r = repr(adapter)
        assert "unavailable" in r

    def test_get_descriptor_names(self):
        adapter = _DummyAdapter()
        names = adapter.get_descriptor_names()
        assert names == ["d1"]

    def test_get_descriptor_names_empty(self):
        adapter = _UnavailableAdapter()
        names = adapter.get_descriptor_names()
        assert names == []

    def test_require_available_raises(self):
        adapter = _UnavailableAdapter()
        with pytest.raises(RuntimeError, match="インストールされていません"):
            adapter._require_available()

    def test_require_available_passes(self):
        adapter = _DummyAdapter()
        adapter._require_available()  # Should not raise

    def test_compute(self):
        adapter = _DummyAdapter()
        result = adapter.compute(["CCO", "CC"])
        assert len(result.descriptors) == 2
