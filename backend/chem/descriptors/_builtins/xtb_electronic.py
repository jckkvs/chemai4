from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

"""
XTB 量子化学記述子

GFN2-xTB による半経験的量子化学計算。
HOMO/LUMO エネルギー、双極子モーメント、分極率など。
利用には xtb コマンドラインツールのインストールが必要。
"""
import logging

DESCRIPTOR_NAME = "XTB量子化学"
DESCRIPTOR_CATEGORY = "量子化学"
DESCRIPTOR_ENGINE = "XTB"
DESCRIPTOR_DESCRIPTION = (
    "GFN2-xTBによるHOMO/LUMOエネルギー、ギャップ、双極子モーメント、"
    "分極率、全エネルギー等。反応性・光学特性の予測に有用。"
    "利用にはxtb実行ファイルが必要（conda install xtb）。"
)
MULTI_DESCRIPTOR = True

logger = logging.getLogger(__name__)


def compute(smiles_list: list[str]) -> "pd.DataFrame":
    import pandas as pd
    try:
        from backend.chem.xtb_adapter import XTBAdapter
        adapter = XTBAdapter()
        if not adapter.is_available():
            logger.info("XTBが未インストールです。スキップします。")
            return pd.DataFrame(index=range(len(smiles_list)))
        result = adapter.compute(smiles_list)
        return result.descriptors.reset_index(drop=True)
    except Exception as e:
        logger.warning(f"XTB計算エラー: {e}")
        return pd.DataFrame(index=range(len(smiles_list)))
