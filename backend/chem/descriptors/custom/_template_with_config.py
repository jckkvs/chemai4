"""
=============================================
カスタム記述子テンプレート（設定値付き版）
=============================================

設定値（パラメータ）を受け取る記述子テンプレート。
ユーザーが計算条件を変更できる柔軟な記述子を作成する際に使用。

ルール:
  1. DESCRIPTOR_NAME, DESCRIPTOR_CATEGORY, DESCRIPTOR_ENGINE を定義
  2. compute(smiles_list, **kwargs) を実装
  3. 戻り値は list[float] または pd.DataFrame
  4. MULTI_DESCRIPTOR = True にすると DataFrame を返せる

作成したファイルは backend/chem/descriptors/custom/ に保存してください。
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


# ─── 必須メタデータ ─────────────────────────
DESCRIPTOR_NAME = "MyConfigurableDescriptor"
DESCRIPTOR_CATEGORY = "カスタム"
DESCRIPTOR_ENGINE = "Custom"
DESCRIPTOR_DESCRIPTION = "設定値を受け取るカスタム記述子の例"

# 複数記述子を返す場合は True に設定
MULTI_DESCRIPTOR = True

# ─── デフォルト設定値 ────────────────────────
DEFAULT_RADIUS = 2
DEFAULT_N_BITS = 1024


def compute(smiles_list: list[str], **kwargs) -> "pd.DataFrame":
    """
    SMILESリストと設定値を受け取り、記述子のDataFrameを返す。

    Args:
        smiles_list: SMILES文字列のリスト
        **kwargs: オプション設定
            - radius (int): Morganフィンガープリントの半径（デフォルト: 2）
            - n_bits (int): ビット数（デフォルト: 1024）

    Returns:
        pd.DataFrame: 各SMILESに対する記述子値
    """
    import pandas as pd

    radius = kwargs.get("radius", DEFAULT_RADIUS)
    n_bits = kwargs.get("n_bits", DEFAULT_N_BITS)

    rows = []
    for smi in smiles_list:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append({f"CustomFP_{j}": 0.0 for j in range(n_bits)})
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            row = {f"CustomFP_{j}": float(bit) for j, bit in enumerate(fp)}
            rows.append(row)
        except Exception:
            rows.append({f"CustomFP_{j}": 0.0 for j in range(n_bits)})
    return pd.DataFrame(rows)
