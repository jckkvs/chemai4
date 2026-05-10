# ChemAI カスタム記述子プラグイン — テストサンプル（成功例）
# 生成プロンプト: "RDKitを使って分子の親油性（LogP）と水溶性関連記述子を計算してください"
#
# ▼ このファイルが「正常に読み込まれた場合」の動作
#   → SMILES特徴量タブ > カスタムプラグイン > 登録済み一覧に表示される
#   → 「解析開始」時に自動で計算される

DESCRIPTOR_NAME = "LogP_Aqueous"
DESCRIPTOR_CATEGORY = "親油性・水溶性"
DESCRIPTOR_ENGINE = "RDKit"
DESCRIPTOR_DESCRIPTION = (
    "RDKit による LogP（Crippen法）、TPSA、水素結合ドナー/アクセプター数を計算します。"
    "Lipinski の経口吸収性ルール（Rule of 5）に関連する記述子セットです。"
)
MULTI_DESCRIPTOR = True  # 複数の値を返す（pd.DataFrame）


def compute(smiles_list: list[str]):
    """
    SMILESリストから親油性・水溶性関連記述子を計算する。

    Returns:
        pd.DataFrame: LogP, TPSA, HBD, HBA の4列
    """
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    rows = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append({"LogP": None, "TPSA": None, "HBD": None, "HBA": None})
                continue
            rows.append({
                "LogP":  round(Descriptors.MolLogP(mol), 4),
                "TPSA":  round(Descriptors.TPSA(mol), 4),
                "HBD":   rdMolDescriptors.CalcNumHBD(mol),
                "HBA":   rdMolDescriptors.CalcNumHBA(mol),
            })
        except Exception:
            rows.append({"LogP": None, "TPSA": None, "HBD": None, "HBA": None})

    return pd.DataFrame(rows)
