from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

"""
RDKit グラフ理論的指標

Chi(結合性指数), Kappa(形状指数), Balaban J, Bertz複雑度,
Hall-Kier Alpha, Ipc(情報含量)等のトポロジカル指標。
分子形状の数学的記述・QSPR回帰分析に有用。
"""
import numpy as np

DESCRIPTOR_NAME = "RDKitグラフ指標"
DESCRIPTOR_CATEGORY = "グラフ理論"
DESCRIPTOR_ENGINE = "RDKit"
DESCRIPTOR_DESCRIPTION = (
    "Chi0〜Chi4(結合性指数), Kappa1〜3(形状指数), BalabanJ, BertzCT, "
    "HallKierAlpha, Ipc, AvgIpc, BCUT2D 等のグラフ理論的指標。"
    "分子の直線性・分岐度・対称性・複雑さを定量化。"
)
MULTI_DESCRIPTOR = True

_GRAPH_DESCS = [
    "BalabanJ", "BertzCT", "HallKierAlpha",
    "Kappa1", "Kappa2", "Kappa3",
    "Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v",
    "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v",
    "Ipc", "AvgIpc",
    "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW",
    "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3",
]


def compute(smiles_list: list[str]) -> "pd.DataFrame":
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    desc_fns = {name: fn for name, fn in Descriptors.descList if name in _GRAPH_DESCS}
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        row = {}
        for dname in _GRAPH_DESCS:
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
