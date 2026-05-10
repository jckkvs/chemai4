# -*- coding: utf-8 -*-
"""
tests/test_chem_init.py

backend/chem/__init__.py の安全import確認テスト。
全アダプタが未インストール環境でもエラーなくimportできることを検証。
"""
from __future__ import annotations


class TestChemInitSafeImport:
    """backend.chem.__init__.py の安全import確認"""

    def test_all_adapters_importable(self):
        """全アダプタが __init__.py から import できる"""
        from backend.chem import (
            RDKitAdapter,
            XTBAdapter,
            CosmoAdapter,
            UniPkaAdapter,
            GroupContribAdapter,
            MordredAdapter,
            MolAIAdapter,
            UMAAdapter,
            SkfpAdapter,
            PaDELAdapter,
            DescriptaStorusAdapter,
            Mol2VecAdapter,
            MolfeatAdapter,
            ChempropAdapter,
        )
        for cls in [RDKitAdapter, XTBAdapter, CosmoAdapter, UniPkaAdapter,
                    GroupContribAdapter, MordredAdapter, MolAIAdapter, UMAAdapter,
                    SkfpAdapter, PaDELAdapter, DescriptaStorusAdapter,
                    Mol2VecAdapter, MolfeatAdapter, ChempropAdapter]:
            obj = cls()
            assert isinstance(obj.is_available(), bool)

    def test_all_exported(self):
        """全アダプタ名が __all__ に含まれる"""
        from backend.chem import __all__
        expected = [
            "BaseChemAdapter", "DescriptorResult",
            "RDKitAdapter", "XTBAdapter", "CosmoAdapter",
            "UniPkaAdapter", "GroupContribAdapter", "MordredAdapter",
            "MolAIAdapter", "UMAAdapter",
            "SkfpAdapter", "PaDELAdapter", "DescriptaStorusAdapter",
            "Mol2VecAdapter", "MolfeatAdapter", "ChempropAdapter",
        ]
        for name in expected:
            assert name in __all__, f"{name} が __all__ にありません"
