from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

"""
RDKit フィンガープリント記述子

Morgan (ECFP4), RDKit トポロジカル, MACCS Keys。
化合物の構造類似性検索・クラスタリング・QSAR/QSPRモデルに広く使用。
"""

DESCRIPTOR_NAME = "RDKitフィンガープリント"
DESCRIPTOR_CATEGORY = "フィンガープリント"
DESCRIPTOR_ENGINE = "RDKit"
DESCRIPTOR_DESCRIPTION = (
    "Morgan FP (ECFP4) + RDKit FP。"
    "化合物の部分構造パターンをビットベクトルで表現。"
    "類似性検索・仮想スクリーニング・QSARモデルの標準的な入力特徴量。"
)
MULTI_DESCRIPTOR = True

# ── 設定可能パラメータ（UIで動的に生成される） ──
DESCRIPTOR_PARAMS = {
    "morgan_radius": {
        "type": "choice",
        "default": 2,
        "choices": [1, 2, 3, 4],
        "description": "Morgan FPの半径（1=ECFP2, 2=ECFP4, 3=ECFP6）",
    },
    "morgan_bits": {
        "type": "choice",
        "default": 2048,
        "choices": [512, 1024, 2048, 4096],
        "description": "Morgan FPのビット数",
    },
    "rdkit_fp_bits": {
        "type": "choice",
        "default": 2048,
        "choices": [512, 1024, 2048, 4096],
        "description": "RDKit FPのビット数",
    },
    "include_maccs": {
        "type": "bool",
        "default": False,
        "description": "MACCS Keys (166bit) を含める",
    },
}


def compute(smiles_list: list[str], **kwargs) -> "pd.DataFrame":
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys

    morgan_radius = kwargs.get("morgan_radius", DESCRIPTOR_PARAMS["morgan_radius"]["default"])
    morgan_bits = kwargs.get("morgan_bits", DESCRIPTOR_PARAMS["morgan_bits"]["default"])
    rdkit_fp_bits = kwargs.get("rdkit_fp_bits", DESCRIPTOR_PARAMS["rdkit_fp_bits"]["default"])
    include_maccs = kwargs.get("include_maccs", DESCRIPTOR_PARAMS["include_maccs"]["default"])

    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        row = {}

        if mol:
            # Morgan FP
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, morgan_radius, nBits=morgan_bits)
                for j, bit in enumerate(fp):
                    row[f"Morgan_r{morgan_radius}_{j}"] = float(bit)
            except Exception:
                for j in range(morgan_bits):
                    row[f"Morgan_r{morgan_radius}_{j}"] = 0.0

            # RDKit トポロジカル FP
            try:
                rdkit_fp = Chem.RDKFingerprint(mol, fpSize=rdkit_fp_bits)
                for j, bit in enumerate(rdkit_fp):
                    row[f"RDKitFP_{j}"] = float(bit)
            except Exception:
                for j in range(rdkit_fp_bits):
                    row[f"RDKitFP_{j}"] = 0.0

            # MACCS Keys
            if include_maccs:
                try:
                    maccs = MACCSkeys.GenMACCSKeys(mol)
                    for j, bit in enumerate(maccs):
                        row[f"MACCS_{j}"] = float(bit)
                except Exception:
                    for j in range(167):
                        row[f"MACCS_{j}"] = 0.0
        else:
            for j in range(morgan_bits):
                row[f"Morgan_r{morgan_radius}_{j}"] = 0.0
            for j in range(rdkit_fp_bits):
                row[f"RDKitFP_{j}"] = 0.0
            if include_maccs:
                for j in range(167):
                    row[f"MACCS_{j}"] = 0.0

        rows.append(row)
    return pd.DataFrame(rows)
