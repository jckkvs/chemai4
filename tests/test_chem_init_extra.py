"""
tests/test_chem_init_extra.py

backend/chem/__init__.py のカバレッジ改善テスト。
_make_unavailable_adapter, ADAPTER_REGISTRY, get_available_adapters,
__all__, 安全importメカニズムを網羅。
"""
from __future__ import annotations

import pytest

from backend.chem import (
    BaseChemAdapter,
    DescriptorResult,
    ADAPTER_REGISTRY,
    get_available_adapters,
    _make_unavailable_adapter,
)


# ============================================================
# _make_unavailable_adapter
# ============================================================

class TestMakeUnavailableAdapter:
    def test_creates_class(self):
        FakeAdapter = _make_unavailable_adapter("FakeAdapter")
        assert FakeAdapter.__name__ == "FakeAdapter"

    def test_is_not_available(self):
        FakeAdapter = _make_unavailable_adapter("FakeAdapter")
        instance = FakeAdapter()
        assert instance.is_available() is False

    def test_compute_raises(self):
        FakeAdapter = _make_unavailable_adapter("FakeAdapter")
        instance = FakeAdapter()
        with pytest.raises(RuntimeError, match="利用できません"):
            instance.compute(["CCO"])

    def test_accepts_kwargs(self):
        FakeAdapter = _make_unavailable_adapter("FakeAdapter")
        # kwargs を受け入れてもエラーにならない
        instance = FakeAdapter(some_param="value")
        assert instance.is_available() is False


# ============================================================
# ADAPTER_REGISTRY
# ============================================================

class TestAdapterRegistry:
    def test_is_dict(self):
        assert isinstance(ADAPTER_REGISTRY, dict)

    def test_has_known_keys(self):
        assert "RDKit" in ADAPTER_REGISTRY
        assert "XTB" in ADAPTER_REGISTRY
        assert "Mordred" in ADAPTER_REGISTRY

    def test_all_values_have_is_available(self):
        """全アダプタークラスが is_available メソッドを持つ"""
        for name, cls in ADAPTER_REGISTRY.items():
            instance = cls()
            assert hasattr(instance, "is_available"), f"{name} lacks is_available"

    def test_registry_size(self):
        # 14アダプター登録済み
        assert len(ADAPTER_REGISTRY) >= 10


# ============================================================
# get_available_adapters
# ============================================================

class TestGetAvailableAdapters:
    def test_returns_dict(self):
        result = get_available_adapters()
        assert isinstance(result, dict)

    def test_only_available(self):
        """返されるアダプターはすべて is_available() == True"""
        result = get_available_adapters()
        for name, cls in result.items():
            instance = cls()
            assert instance.is_available(), f"{name} should be available"

    def test_subset_of_registry(self):
        """get_available_adapters の結果はレジストリのサブセット"""
        available = get_available_adapters()
        for name in available:
            assert name in ADAPTER_REGISTRY


# ============================================================
# __all__ エクスポートの検証
# ============================================================

class TestAllExports:
    def test_base_classes_exported(self):
        from backend import chem
        assert hasattr(chem, "BaseChemAdapter")
        assert hasattr(chem, "DescriptorResult")

    def test_registry_exported(self):
        from backend import chem
        assert hasattr(chem, "ADAPTER_REGISTRY")
        assert hasattr(chem, "get_available_adapters")

    def test_all_adapters_exist(self):
        from backend import chem
        for adapter_name in [
            "RDKitAdapter", "XTBAdapter", "MordredAdapter",
            "GroupContribAdapter", "MolAIAdapter", "UMAAdapter",
        ]:
            assert hasattr(chem, adapter_name), f"{adapter_name} not exported"
