from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

"""
RDKit トポロジー記述子

環構造、回転可能結合、水素結合ドナー/アクセプター数など、
分子の立体的柔軟性・結晶性・薬品性に関連する構造的特徴。
"""
import numpy as np

DESCRIPTOR_NAME = "RDKitトポロジー"
DESCRIPTOR_CATEGORY = "トポロジー"
DESCRIPTOR_ENGINE = "RDKit"
DESCRIPTOR_DESCRIPTION = (
    "環数(芳香族/脂肪族/飽和)、回転可能結合数、水素結合ドナー/アクセプター数、"
    "ヘテロ環数、スピロ/橋頭原子数、アミド結合数など。Lipinski Rule-of-5の評価にも使用。"
)
MULTI_DESCRIPTOR = True

_TOPO_DESCS = [
    "NumRotatableBonds", "RingCount", "NumAromaticRings", "NumAliphaticRings",
    "NumSaturatedRings", "NumHeterocycles", "NumAromaticHeterocycles",
    "NumAromaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticCarbocycles",
    "NumSaturatedHeterocycles", "NumSaturatedCarbocycles",
    "NumHAcceptors", "NumHDonors", "NHOHCount", "NOCount",
    "NumAmideBonds", "NumBridgeheadAtoms", "NumSpiroAtoms",
]


def compute(smiles_list: list[str]) -> "pd.DataFrame":
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    desc_fns = {name: fn for name, fn in Descriptors.descList if name in _TOPO_DESCS}
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        row = {}
        for dname in _TOPO_DESCS:
            fn = desc_fns.get(dname)
            if fn and mol:
                try:
                    v = fn(mol)
                    row[dname] = float(v) if v is not None and np.isfinite(float(v)) else np.nan
                except Exception:
                    row[dname] = np.nan
            else:
                row[dname] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)
