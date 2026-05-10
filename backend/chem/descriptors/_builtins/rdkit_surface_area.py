from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

"""
RDKit 表面積分布記述子 (VSA)

部分電荷(PEOE)・屈折率(SMR)・LogP・EState値ごとに、
分子表面積をビン分割した記述子群。
溶媒和エネルギー・タンパク質結合の予測に有用。
"""
import numpy as np

DESCRIPTOR_NAME = "RDKit表面積分布"
DESCRIPTOR_CATEGORY = "表面積分布"
DESCRIPTOR_ENGINE = "RDKit"
DESCRIPTOR_DESCRIPTION = (
    "PEOE_VSA(14種), SMR_VSA(10種), SlogP_VSA(12種), "
    "EState_VSA(11種), VSA_EState(10種) の計57記述子。"
    "分子表面の物性分布を詳細にプロファイリング。"
)
MULTI_DESCRIPTOR = True

_VSA_PREFIXES = ["PEOE_VSA", "SMR_VSA", "SlogP_VSA", "EState_VSA", "VSA_EState"]


def compute(smiles_list: list[str]) -> "pd.DataFrame":
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    # VSA系の記述子を自動収集
    vsa_descs = [
        (name, fn) for name, fn in Descriptors.descList
        if any(name.startswith(prefix) for prefix in _VSA_PREFIXES)
    ]
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        row = {}
        for dname, fn in vsa_descs:
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
