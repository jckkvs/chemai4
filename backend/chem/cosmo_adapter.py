"""
backend/chem/cosmo_adapter.py

openCOSMO-RS (https://github.com/TUHH-TVT/openCOSMO-RS_py) を用いた
COSMO-RS 理論に基づく熱力学的記述子の計算アダプター。

インストール:
  pip install git+https://github.com/TUHH-TVT/openCOSMO-RS_py.git

重要: openCOSMO-RS は量子化学計算で生成した σ-profile ファイル（.cosmi / .sigma）
が入力として必要です。SMILES → COSMI は XTB + COSMO計算が別途必要です。
本アダプターでは、kwargs["cosmi_files"] (list[str]) でCOSMIファイルパスのリストを
渡すことで計算を実行します。ファイルがない場合は NaN を返します。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.chem.base import BaseChemAdapter, DescriptorMetadata, DescriptorResult

logger = logging.getLogger(__name__)

_COSMO_DESCRIPTORS: dict[str, str] = {
    "mu_comb":    "組合せ項の化学ポテンシャル [kcal/mol]（openCOSMO-RS）",
    "mu_res":     "残余項の化学ポテンシャル [kcal/mol]（openCOSMO-RS）",
    "ln_gamma":   "活量係数の自然対数 ln(γ)（openCOSMO-RS、純溶媒基準）",
}


class CosmoAdapter(BaseChemAdapter):
    """
    openCOSMO-RS を用いた COSMO-RS 記述子アダプター。

    使い方:
        adapter.compute(smiles_list, cosmi_files=["mol1.cosmi", "mol2.cosmi"])

    cosmi_files が与えられない場合や COSMI ファイルが不足する場合は NaN を返す。
    """

    def __init__(self, parameterization: str = "default_turbomole") -> None:
        self._par = parameterization

    @property
    def name(self) -> str:
        return "cosmo_rs"

    @property
    def description(self) -> str:
        return "openCOSMO-RS による化学ポテンシャル・活量係数の計算（σ-profileファイル必要）"

    def is_available(self) -> bool:
        try:
            from opencosmorspy import COSMORS  # noqa: F401
            return True
        except ImportError:
            return False

    def compute(self, smiles_list: list[str], **kwargs: Any) -> DescriptorResult:
        self._require_available()
        logger.info(f"🌐 COSMO-RS 計算を開始します（{len(smiles_list)}分子）...")
        from opencosmorspy import COSMORS

        cosmi_files: list[str] | None = kwargs.get("cosmi_files")
        n = len(smiles_list)
        nan_row = {k: float("nan") for k in _COSMO_DESCRIPTORS}

        if not cosmi_files:
            logger.warning(
                "CosmoAdapter: cosmi_files が指定されていません。"
                "kwargs['cosmi_files'] に COSMI/sigma ファイルのリストを渡してください。"
            )
            df = pd.DataFrame([nan_row] * n)
            return DescriptorResult(
                descriptors=df,
                smiles_list=smiles_list,
                failed_indices=list(range(n)),
                adapter_name=self.name,
            )

        records: list[dict[str, float]] = []
        for i, smi in enumerate(smiles_list):
            if i >= len(cosmi_files) or not Path(cosmi_files[i]).exists():
                records.append(nan_row.copy())
                continue
            try:
                crs = COSMORS(par=self._par)
                # 溶媒 = 自分自身（純液体）として化学ポテンシャル計算
                crs.add_molecule([cosmi_files[i]])
                crs.add_job(x=[1.0], T=298.15, refst="pure_component")
                result = crs.calculate()
                # result は辞書 / DataFrame を返すことが多い
                if isinstance(result, dict):
                    row = {
                        "mu_comb":  float(result.get("mu_comb", [float("nan")])[0]),
                        "mu_res":   float(result.get("mu_res",  [float("nan")])[0]),
                        "ln_gamma": float(result.get("ln_gamma",[float("nan")])[0]),
                    }
                else:
                    row = nan_row.copy()
                records.append(row)
            except Exception as e:
                logger.warning(f"COSMO-RS 計算失敗 ({smi}): {e}")
                records.append(nan_row.copy())

        df = pd.DataFrame(records, columns=list(_COSMO_DESCRIPTORS.keys()))
        failed = [i for i, r in enumerate(records) if any(np.isnan(v) for v in r.values())]
        return DescriptorResult(
            descriptors=df,
            smiles_list=smiles_list,
            failed_indices=failed,
            adapter_name=self.name,
        )

    def get_descriptor_names(self) -> list[str]:
        return list(_COSMO_DESCRIPTORS.keys())

    def get_descriptors_metadata(self) -> list[DescriptorMetadata]:
        return [
            DescriptorMetadata(name=k, meaning=v, is_count=False)
            for k, v in _COSMO_DESCRIPTORS.items()
        ]

