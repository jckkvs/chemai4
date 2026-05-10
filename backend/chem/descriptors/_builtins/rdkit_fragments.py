from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

"""
RDKit 官能基フラグメントカウント記述子

fr_系の85種の官能基（アルコール、アミン、カルボニル、ハロゲン等）の
出現回数をカウント。SAR解析・反応性予測・毒性予測に有用。
"""
import numpy as np

DESCRIPTOR_NAME = "RDKit官能基カウント"
DESCRIPTOR_CATEGORY = "官能基"
DESCRIPTOR_ENGINE = "RDKit"
DESCRIPTOR_DESCRIPTION = (
    "fr_Al_OH, fr_ester, fr_halogen, fr_benzene 等85種の官能基フラグメントカウント。"
    "構造活性相関(SAR)解析、反応部位の予測、毒性アラート検出に使用。"
)
MULTI_DESCRIPTOR = True


def compute(smiles_list: list[str]) -> "pd.DataFrame":
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    fr_descs = [(name, fn) for name, fn in Descriptors.descList if name.startswith("fr_")]
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        row = {}
        for dname, fn in fr_descs:
            if mol:
                try:
                    v = fn(mol)
                    row[dname] = float(v) if v is not None and np.isfinite(float(v)) else np.nan
                except Exception:
                    row[dname] = np.nan
            else:
                row[dname] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)
