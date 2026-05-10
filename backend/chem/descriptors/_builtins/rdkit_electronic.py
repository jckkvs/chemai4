from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

"""
RDKit 電子状態記述子

EState指標およびGasteiger部分電荷統計量。
反応性予測・電子リッチ/プア部位の定量化に使用。
"""
import numpy as np

DESCRIPTOR_NAME = "RDKit電子状態"
DESCRIPTOR_CATEGORY = "電子状態"
DESCRIPTOR_ENGINE = "RDKit"
DESCRIPTOR_DESCRIPTION = (
    "MaxAbsEStateIndex, Gasteiger部分電荷(max/min/range/std/abs_mean)等。"
    "反応部位の予測、求電子/求核反応性の評価に有用。"
)
MULTI_DESCRIPTOR = True

_ESTATE_DESCS = [
    "MaxAbsEStateIndex", "MaxEStateIndex", "MinAbsEStateIndex", "MinEStateIndex",
    "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge",
    "NumRadicalElectrons", "NumValenceElectrons",
]


def compute(smiles_list: list[str]) -> "pd.DataFrame":
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdPartialCharges

    desc_fns = {name: fn for name, fn in Descriptors.descList if name in _ESTATE_DESCS}
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        row = {}
        for dname in _ESTATE_DESCS:
            fn = desc_fns.get(dname)
            if fn and mol:
                try:
                    v = fn(mol)
                    row[dname] = float(v) if v is not None and np.isfinite(float(v)) else np.nan
                except Exception:
                    row[dname] = np.nan
            else:
                row[dname] = np.nan

        # Gasteiger部分電荷統計量
        if mol:
            try:
                mol_h = Chem.AddHs(mol)
                rdPartialCharges.ComputeGasteigerCharges(mol_h)
                charges = [
                    float(mol_h.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge"))
                    for i in range(mol_h.GetNumAtoms())
                ]
                charges = [q for q in charges if np.isfinite(q)]
                if charges:
                    ca = np.array(charges)
                    row["gasteiger_q_max"] = float(np.max(ca))
                    row["gasteiger_q_min"] = float(np.min(ca))
                    row["gasteiger_q_range"] = float(np.max(ca) - np.min(ca))
                    row["gasteiger_q_std"] = float(np.std(ca))
                    row["gasteiger_q_abs_mean"] = float(np.mean(np.abs(ca)))
                else:
                    for k in ["gasteiger_q_max", "gasteiger_q_min", "gasteiger_q_range",
                              "gasteiger_q_std", "gasteiger_q_abs_mean"]:
                        row[k] = np.nan
            except Exception:
                for k in ["gasteiger_q_max", "gasteiger_q_min", "gasteiger_q_range",
                          "gasteiger_q_std", "gasteiger_q_abs_mean"]:
                    row[k] = np.nan
        else:
            for k in ["gasteiger_q_max", "gasteiger_q_min", "gasteiger_q_range",
                      "gasteiger_q_std", "gasteiger_q_abs_mean"]:
                row[k] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)
