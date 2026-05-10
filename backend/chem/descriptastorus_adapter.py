# -*- coding: utf-8 -*-
"""
backend/chem/descriptastorus_adapter.py

DescriptaStorus アダプタ。
Merck が開発した高速記述子計算ライブラリ。
RDKit ベースの記述子を高速かつ一貫性のある形で計算。

参考: descriptastorus (https://github.com/bp-kelley/descriptastorus)
  pip install descriptastorus
"""
from __future__ import annotations

import logging

import pandas as pd

from backend.chem.base import BaseChemAdapter, DescriptorMetadata, DescriptorResult

logger = logging.getLogger(__name__)


class DescriptaStorusAdapter(BaseChemAdapter):
    """
    DescriptaStorus アダプタ。

    Merck 開発の高速記述子計算。RDKit2D 記述子セットを
    標準化された形で高速計算する。
    """

    def __init__(self, descriptor_type: str = "rdkit2d"):
        """
        Args:
            descriptor_type: "rdkit2d" | "rdkit2dnormalized" | "morgan3counts"
        """
        self._descriptor_type = descriptor_type

    @property
    def name(self) -> str:
        return "descriptastorus"

    @property
    def description(self) -> str:
        return (
            "DescriptaStorus: Merck開発の高速記述子計算。"
            "RDKit2D記述子の標準化済みセット。"
            "pip install descriptastorus"
        )

    def is_available(self) -> bool:
        try:
            from descriptastorus.descriptors import rdDescriptors  # noqa: F401
            return True
        except ImportError:
            return False

    def compute(self, smiles_list: list[str], **kwargs) -> DescriptorResult:
        self._require_available()
        logger.info(f"📋 DescriptaStorus 記述子計算を開始します（{len(smiles_list)}分子）...")
        from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors

        if "normalized" in self._descriptor_type.lower():
            generator = rdNormalizedDescriptors.RDKit2DNormalized()
        else:
            generator = rdDescriptors.RDKit2D()

        rows = []
        failed_indices = []
        col_names = None

        for i, smi in enumerate(smiles_list):
            try:
                results = generator.process(smi)
                if results[0]:  # 成功フラグ
                    values = results[1:]
                    if col_names is None:
                        col_names = [f"DS_{generator.columns[j]}" for j in range(len(values))]
                    rows.append(dict(zip(col_names, values)))
                else:
                    failed_indices.append(i)
                    rows.append({})
            except Exception as e:
                logger.warning(f"DescriptaStorus: SMILES '{smi}' でエラー: {e}")
                failed_indices.append(i)
                rows.append({})

        descriptors = pd.DataFrame(rows)

        return DescriptorResult(
            descriptors=descriptors,
            smiles_list=smiles_list,
            failed_indices=failed_indices,
            adapter_name=self.name,
        )

    def get_descriptors_metadata(self) -> list[DescriptorMetadata]:
        """DescriptaStorusで計算される全記述子のメタデータを返す。"""
        try:
            if self.is_available():
                from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
                if "normalized" in self._descriptor_type.lower():
                    gen = rdNormalizedDescriptors.RDKit2DNormalized()
                else:
                    gen = rdDescriptors.RDKit2D()
                col_names = gen.columns
                meta = []
                for col in col_names:
                    meta.append(DescriptorMetadata(
                        name=f"DS_{col}",
                        meaning=f"DescriptaStorus (Merck) 標準化RDKit2D記述子: {col}",
                        is_count=col.startswith("fr_") or col.startswith("Num") or col.startswith("n"),
                    ))
                return meta
        except Exception:
            pass
        # フォールバック
        return [
            DescriptorMetadata(name="DS_MolWt", meaning="分子量 (DescriptaStorus標準化)", is_count=False),
            DescriptorMetadata(name="DS_TPSA", meaning="位相的極性表面積 (DescriptaStorus標準化)", is_count=False),
            DescriptorMetadata(name="DS_MolLogP", meaning="LogP (DescriptaStorus標準化)", is_count=False),
        ]
