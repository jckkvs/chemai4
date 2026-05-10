from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

"""
基団寄与法記述子

Joback法・Benson法等の基団寄与法による熱力学物性推算。
融点・沸点・臨界物性・生成ギブズエネルギーの推定。
"""
import logging

DESCRIPTOR_NAME = "基団寄与法"
DESCRIPTOR_CATEGORY = "熱力学"
DESCRIPTOR_ENGINE = "GroupContrib"
DESCRIPTOR_DESCRIPTION = (
    "Joback法・Benson法による基団寄与法推算。"
    "Tb(沸点), Tm(融点), Tc(臨界温度), Pc(臨界圧力), "
    "ΔGf(生成ギブズエネルギー)等の熱力学物性を構造から推算。"
)
MULTI_DESCRIPTOR = True

logger = logging.getLogger(__name__)


def compute(smiles_list: list[str]) -> "pd.DataFrame":
    import pandas as pd
    try:
        from backend.chem.group_contrib_adapter import GroupContribAdapter
        adapter = GroupContribAdapter()
        if not adapter.is_available():
            logger.info("基団寄与法アダプタが利用不可です。スキップします。")
            return pd.DataFrame(index=range(len(smiles_list)))
        result = adapter.compute(smiles_list)
        return result.descriptors.reset_index(drop=True)
    except Exception as e:
        logger.warning(f"基団寄与法計算エラー: {e}")
        return pd.DataFrame(index=range(len(smiles_list)))
