# -*- coding: utf-8 -*-
"""
backend/chem/padel_adapter.py

PaDEL-Descriptor アダプタ。
1800+ 分子記述子と 10 種フィンガープリントを計算する。
Java ベースの PaDEL-Descriptor を Python 経由で利用。

参考: padelpy (https://github.com/ecrl/padelpy)
  pip install padelpy
"""
from __future__ import annotations

import logging

import pandas as pd

from backend.chem.base import BaseChemAdapter, DescriptorMetadata, DescriptorResult

logger = logging.getLogger(__name__)


class PaDELAdapter(BaseChemAdapter):
    """
    PaDEL-Descriptor アダプタ。

    1D/2D/3D 記述子と10種のフィンガープリント（MACCS, PubChem, SubstructureFP等）を計算。
    Java が必要（padelpy が内蔵のJARファイルを使用）。
    """

    def __init__(self, compute_fingerprints: bool = False, timeout: int = 120):
        """
        Args:
            compute_fingerprints: True でフィンガープリントも計算
            timeout: 1分子あたりのタイムアウト（秒）
        """
        self._compute_fp = compute_fingerprints
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "padel"

    @property
    def description(self) -> str:
        return (
            "PaDEL-Descriptor: 1800+ 分子記述子 + 10種フィンガープリント。"
            "pip install padelpy (Java Runtime 必要)"
        )

    def is_available(self) -> bool:
        try:
            import padelpy  # noqa: F401
            return True
        except ImportError:
            return False

    def compute(self, smiles_list: list[str], **kwargs) -> DescriptorResult:
        self._require_available()
        logger.info(f"📦 PaDEL記述子計算を開始します（{len(smiles_list)}分子）...")
        from padelpy import from_smiles

        rows = []
        failed_indices = []

        for i, smi in enumerate(smiles_list):
            try:
                desc = from_smiles(smi, fingerprints=self._compute_fp, timeout=self._timeout)
                # desc は OrderedDict {name: value}
                row = {}
                for k, v in desc.items():
                    try:
                        row[f"PaDEL_{k}"] = float(v) if v not in ("", "Infinity", "-Infinity") else float("nan")
                    except (ValueError, TypeError):
                        row[f"PaDEL_{k}"] = float("nan")
                rows.append(row)
            except Exception as e:
                logger.warning(f"PaDEL: SMILES '{smi}' でエラー: {e}")
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
        """PaDEL記述子のメタデータを返す。"""
        _PADEL_KNOWN = {
            "MW": "分子量 (Da)",
            "nAtom": "全原子数",
            "nHeavyAtom": "重原子数",
            "nBonds": "全結合数",
            "nRotB": "回転可能結合数。分子の柔軟性",
            "TPSA": "位相的極性表面積 (Å²)。膜透過性の指標",
            "ALogP": "Ghose-Crippen LogP。疎水性の指標",
            "ALogp2": "ALogPの2乗",
            "AMR": "Ghose-Crippen屈折率",
            "nHBAcc": "水素結合受容体数",
            "nHBDon": "水素結合供与体数",
            "nRing": "環構造の数",
            "nAromRing": "芳香環の数",
            "nHeteroRing": "ヘテロ環の数",
            "TopoPSA": "位相的極性表面積 (2D版)",
            "WTPT-1": "Weighted Path 1。1結合経路の加重値",
            "WTPT-2": "Weighted Path 2。2結合経路の加重値",
            "WTPT-3": "Weighted Path 3。3結合経路の加重値",
            "WTPT-4": "Weighted Path 4。4結合経路の加重値",
            "WTPT-5": "Weighted Path 5。5結合経路の加重値",
            "BCUTw-1h": "BCUT 重み最高固有値。分子の原子分布パターン",
            "BCUTw-1l": "BCUT 重み最低固有値",
            "BCUTc-1h": "BCUT 電荷最高固有値。電荷分布パターン",
            "BCUTc-1l": "BCUT 電荷最低固有値",
            "BCUTp-1h": "BCUT 分極率最高固有値",
            "BCUTp-1l": "BCUT 分極率最低固有値",
            "VP-0": "Van der Waals体積 0次",
            "VP-1": "Van der Waals体積 1次",
            "VP-2": "Van der Waals体積 2次",
        }
        try:
            if self.is_available():
                from padelpy import from_smiles
                desc = from_smiles("C", fingerprints=self._compute_fp, timeout=30)
                meta = []
                for k in desc.keys():
                    padel_name = f"PaDEL_{k}"
                    known_meaning = _PADEL_KNOWN.get(k, "")
                    meaning = f"PaDEL: {known_meaning}" if known_meaning else f"PaDEL 分子記述子: {k}"
                    is_count = k.startswith("n") or k.startswith("Num")
                    meta.append(DescriptorMetadata(name=padel_name, meaning=meaning, is_count=is_count))
                return meta
        except Exception:
            pass
        return [
            DescriptorMetadata(name="PaDEL_MW", meaning="分子量 (PaDEL)", is_count=False),
            DescriptorMetadata(name="PaDEL_nAtom", meaning="原子数 (PaDEL)", is_count=True),
            DescriptorMetadata(name="PaDEL_TPSA", meaning="極性表面積 (PaDEL)", is_count=False),
        ]
