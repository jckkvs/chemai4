"""
backend/chem/unipka_adapter.py

Uni-pKa (dptech-corp) の非公式 Python ラッパー `unipka` を用いた
pKa・LogD 記述子計算アダプター。

インストール: pip install unipka
依存: torch>2.3.0, rdkit, numpy, pandas, requests

API: unipka.UnipKa クラス
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from backend.chem.base import BaseChemAdapter, DescriptorMetadata, DescriptorResult

logger = logging.getLogger(__name__)

_UNIPKA_DESCRIPTORS: dict[str, str] = {
    "pKa_acidic":        "最も酸性の強い官能基のマクロ pKa（Uni-pKa）",
    "pKa_basic":         "最も塩基性の強い官能基のマクロ pKa（Uni-pKa）",
    "LogD_7_4":          "pH 7.4 における分配係数 LogD（Uni-pKa）",
    "SolvationEnergy":   "溶媒和自由エネルギー [kcal/mol]（Uni-pKa）",
}


class UniPkaAdapter(BaseChemAdapter):
    """Uni-pKa (dptech-corp) の Python ラッパー unipka を使った pKa 記述子アダプター。"""

    def __init__(self, batch_size: int = 32, remove_hs: bool = False) -> None:
        self._batch_size = batch_size
        self._remove_hs = remove_hs
        self._model: Any = None

    @property
    def name(self) -> str:
        return "unipka"

    @property
    def description(self) -> str:
        return "Uni-pKa による高精度 pKa / LogD / 溶媒和エネルギー予測"

    def is_available(self) -> bool:
        try:
            from unipka import UnipKa  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_model(self):
        if self._model is None:
            from unipka import UnipKa
            self._model = UnipKa(batch_size=self._batch_size, remove_hs=self._remove_hs)
        return self._model

    def compute(self, smiles_list: list[str], **kwargs: Any) -> DescriptorResult:
        self._require_available()
        logger.info(f"🧪 UniPka pKa予測を開始します（{len(smiles_list)}分子）...")
        model = self._get_model()

        records: list[dict[str, float]] = []
        for smi in smiles_list:
            row: dict[str, float] = {}
            try:
                row["pKa_acidic"] = float(model.get_acidic_macro_pka(smi))
            except Exception:
                row["pKa_acidic"] = float("nan")
            try:
                row["pKa_basic"] = float(model.get_basic_macro_pka(smi))
            except Exception:
                row["pKa_basic"] = float("nan")
            try:
                row["LogD_7_4"] = float(model.get_logd(smi, pH=7.4))
            except Exception:
                row["LogD_7_4"] = float("nan")
            try:
                from unipka import get_solvation_energy
                row["SolvationEnergy"] = float(get_solvation_energy(smi))
            except Exception:
                row["SolvationEnergy"] = float("nan")
            records.append(row)

        df = pd.DataFrame(records, columns=list(_UNIPKA_DESCRIPTORS.keys()))
        return DescriptorResult(
            descriptors=df,
            smiles_list=smiles_list,
            failed_indices=[],
            adapter_name=self.name,
        )

    def get_descriptor_names(self) -> list[str]:
        return list(_UNIPKA_DESCRIPTORS.keys())

    def get_descriptors_metadata(self) -> list[DescriptorMetadata]:
        return [
            DescriptorMetadata(name=k, meaning=v, is_count=False)
            for k, v in _UNIPKA_DESCRIPTORS.items()
        ]

