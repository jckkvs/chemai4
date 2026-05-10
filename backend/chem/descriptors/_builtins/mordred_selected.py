from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

"""
Mordred 厳選記述子

Mordredライブラリによる2D記述子計算。
RDKitでカバーされない追加の物理化学記述子を提供。
"""
import logging

DESCRIPTOR_NAME = "Mordred厳選"
DESCRIPTOR_CATEGORY = "物理化学"
DESCRIPTOR_ENGINE = "Mordred"
DESCRIPTOR_DESCRIPTION = (
    "Mordredライブラリの厳選2D記述子。ABCIndex, AcidBase, "
    "TopoPSA, WienerIndex 等、RDKit非搭載の記述子を補完。"
    "利用にはmordred-communityパッケージが必要。"
)
MULTI_DESCRIPTOR = True

logger = logging.getLogger(__name__)


def compute(smiles_list: list[str]) -> "pd.DataFrame":
    import pandas as pd
    try:
        from mordred import Calculator, descriptors
        from rdkit import Chem
    except ImportError:
        logger.warning("Mordredが未インストールです。空のDataFrameを返します。")
        return pd.DataFrame(index=range(len(smiles_list)))

    calc = Calculator(descriptors, ignore_3D=True)
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        mols.append(mol)

    try:
        df = calc.pandas(mols, quiet=True)
        # 数値変換: Mordredはmissing等を返すことがある
        df = df.apply(pd.to_numeric, errors="coerce")
        # 全欠損列を除去
        df = df.dropna(axis=1, how="all")
        return df.reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Mordred計算エラー: {e}")
        return pd.DataFrame(index=range(len(smiles_list)))
