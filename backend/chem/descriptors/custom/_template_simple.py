"""
=============================================
カスタム記述子テンプレート（シンプル版）
=============================================

このファイルをコピーして、独自の記述子を作成してください。

ルール:
  1. DESCRIPTOR_NAME, DESCRIPTOR_CATEGORY, DESCRIPTOR_ENGINE を定義
  2. compute(smiles_list) を実装
  3. 戻り値は list[float] （各SMILESに対して1個の値）
  4. エラー時は None を返す（アプリが自動でスキップ）
  
作成したファイルは backend/chem/descriptors/custom/ に保存してください。
ファイル名の先頭に _ をつけるとスキャンされません。
"""

# ─── 必須メタデータ ─────────────────────────
DESCRIPTOR_NAME = "MyCustomDescriptor"
DESCRIPTOR_CATEGORY = "カスタム"         # 例: "物理化学", "電子状態", "トポロジー"
DESCRIPTOR_ENGINE = "Custom"             # 例: "RDKit", "XTB", "Custom"
DESCRIPTOR_DESCRIPTION = "ここに記述子の説明を書く"


def compute(smiles_list: list[str]) -> list[float | None]:
    """
    SMILESのリストを受け取り、各SMILESに対する記述子値のリストを返す。

    Args:
        smiles_list: SMILES文字列のリスト

    Returns:
        float値のリスト（計算失敗時は None）
    """
    results = []
    for smi in smiles_list:
        try:
            # ── ここに計算ロジックを書く ──
            # 例: 分子量を計算
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                results.append(None)
                continue
            value = Descriptors.MolWt(mol)
            results.append(float(value))
        except Exception:
            results.append(None)
    return results
