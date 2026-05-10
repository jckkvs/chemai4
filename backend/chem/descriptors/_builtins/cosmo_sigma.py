"""
COSMO-RS σプロファイル記述子

COSMOtherm / OpenCOSMO-RS による溶媒和記述子。
σプロファイル、σポテンシャル、溶解度パラメータ等。
利用には COSMOtherm または openCOSMO-RS が必要。
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import logging

DESCRIPTOR_NAME = "COSMO-RSσプロファイル"
DESCRIPTOR_CATEGORY = "溶媒和"
DESCRIPTOR_ENGINE = "COSMO-RS"
DESCRIPTOR_DESCRIPTION = (
    "COSMO-RSによるσプロファイル・σポテンシャルベースの記述子。"
    "溶解度予測、混合物の相平衡計算に使用。"
    "利用にはCOSMOthermまたはopenCOSMO-RSが必要。"
)
MULTI_DESCRIPTOR = True

logger = logging.getLogger(__name__)


def compute(smiles_list: list[str]) -> "pd.DataFrame":
    import pandas as pd
    try:
        from backend.chem.cosmo_adapter import CosmoAdapter
        adapter = CosmoAdapter()
        if not adapter.is_available():
            logger.info("COSMO-RSが未インストールです。スキップします。")
            return pd.DataFrame(index=range(len(smiles_list)))
        result = adapter.compute(smiles_list)
        return result.descriptors.reset_index(drop=True)
    except Exception as e:
        logger.warning(f"COSMO-RS計算エラー: {e}")
        return pd.DataFrame(index=range(len(smiles_list)))
