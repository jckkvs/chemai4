from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

"""
RDKit 物理化学記述子

分子量、LogP、極性表面積(TPSA)、モル屈折率(MolMR)など、
溶解性・膜透過性・分配に直結する基本的な物理化学パラメータ。
"""
import numpy as np

DESCRIPTOR_NAME = "RDKit物理化学"
DESCRIPTOR_CATEGORY = "物理化学"
DESCRIPTOR_ENGINE = "RDKit"
DESCRIPTOR_DESCRIPTION = (
    "MolWt, LogP, TPSA, MolMR, QED 等の基本物理化学記述子。"
    "溶解性予測・膜透過性評価・薬品適性スクリーニングに有用。"
)
MULTI_DESCRIPTOR = True

_TARGET_DESCS = [
    "MolWt", "HeavyAtomMolWt", "ExactMolWt", "MolLogP", "TPSA",
    "MolMR", "LabuteASA", "qed", "HeavyAtomCount", "FractionCSP3",
]


def compute(smiles_list: list[str]) -> "pd.DataFrame":
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    desc_fns = {name: fn for name, fn in Descriptors.descList if name in _TARGET_DESCS}
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        row = {}
        for dname in _TARGET_DESCS:
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
