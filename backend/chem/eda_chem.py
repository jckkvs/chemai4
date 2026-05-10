"""
backend/chem/eda_chem.py

化学構造に特化した探索的データ分析（EDA）機能を提供するモジュール。
MACCSキー等のフィンガープリントを用いた分子間類似度（Tanimoto係数）計算などを含む。
"""
from typing import Any, List, Dict
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_tanimoto_similarity(smiles_list: List[str]) -> pd.DataFrame:
    """
    SMILES文字列のリストから、分子間のTanimoto類似度行列を計算する。

    Args:
        smiles_list: 解析対象のSMILES文字列のリスト

    Returns:
        分子間の類似度行列のDataFrame。インデックスとカラムはSMILES文字列。
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import MACCSkeys
        from rdkit import DataStructs
    except ImportError:
        logger.error("RDKitがインストールされていません。")
        return pd.DataFrame()

    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    valid_indices = [i for i, m in enumerate(mols) if m is not None]
    valid_mols = [mols[i] for i in valid_indices]
    valid_smiles = [smiles_list[i] for i in valid_indices]

    if not valid_mols:
        return pd.DataFrame()

    fps = [MACCSkeys.GenMACCSKeys(m) for m in valid_mols]
    n = len(fps)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    return pd.DataFrame(similarity_matrix, index=valid_smiles, columns=valid_smiles)

def get_chem_eda_summary(smiles_list: List[str]) -> Dict[str, Any]:
    """
    化学構造のサマリー情報を生成する。

    Args:
        smiles_list: SMILES文字列のリスト

    Returns:
        有効な分子数等のサマリー情報を含む辞書
    """
    try:
        from rdkit import Chem
    except ImportError:
        return {"error": "RDKit is required."}

    valid_mols = [Chem.MolFromSmiles(smi) for smi in smiles_list if Chem.MolFromSmiles(smi) is not None]
    invalid_count = len(smiles_list) - len(valid_mols)

    return {
        "total_input": len(smiles_list),
        "valid_molecules": len(valid_mols),
        "invalid_smiles": invalid_count,
        "valid_ratio": len(valid_mols) / len(smiles_list) if smiles_list else 0.0
    }
