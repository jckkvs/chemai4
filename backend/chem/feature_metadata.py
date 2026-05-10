from typing import Dict, List, Optional


class FeatureMetadataRegistry:
    """SMILES記述子エンジンの特徴量メタデータを管理"""

    def __init__(self):
        self._registry: Dict[str, Dict] = {}

    def register_engine(self, engine_name: str, feature_prefixes: List[str],
                       feature_descriptions: Optional[Dict[str, str]] = None):
        """エンジンの特徴量プレフィックス・説明を登録"""
        self._registry[engine_name] = {
            "prefixes": feature_prefixes,
            "descriptions": feature_descriptions or {},
            "default_constraints": {
                "direction": "unknown",
                "linearity": False,
                "strength": 0.5,
                "sigma_range": 3.0
            }
        }

    def get_default_constraints(self, feature_name: str) -> Dict:
        """特徴量名からデフォルト制約設定を取得"""
        for engine, config in self._registry.items():
            if any(feature_name.lower().startswith(p.lower()) for p in config["prefixes"]):
                return config["default_constraints"].copy()
        # 未知の特徴量用のデフォルト
        return {"direction": "none", "linearity": False, "strength": 0.0, "sigma_range": 3.0}

    def export_for_frontend(self) -> Dict[str, List[str]]:
        """フロントエンド用のプレフィックスマップを出力"""
        return {engine: config["prefixes"] for engine, config in self._registry.items()}


# 初期化と登録
feature_metadata = FeatureMetadataRegistry()

# RDKit - 2D記述子
feature_metadata.register_engine(
    "rdkit",
    ["rdkit_", "rdkit_2d", "molwt", "logp", "tpsa", "num_hbd", "num_hba", "fp_", "balaban", "bertz", "maccs_keys"],
    feature_descriptions={"molwt": "分子量", "logp": "オクタノール/水分配係数", "tpsa": "極性表面積"}
)

# Mordred - 包括的記述子
feature_metadata.register_engine(
    "mordred",
    ["mordred_", "ABC", "ATS", "BCUT", "GETAWAY", "MOR", "WHIM"],
)

# MolFeat - 機械学習特化記述子
feature_metadata.register_engine(
    "molfeat",
    ["molfeat_"],
)

# xTB - 量子化学計算特徴量
feature_metadata.register_engine(
    "xtb",
    ["xtb_", "xtb_ml_", "xtb_opt", "xtb_sp", "xtb_ml_derived", "3D_"],
)

# Cosmo - COSMO-RS 溶媒和特徴量
feature_metadata.register_engine(
    "cosmo",
    ["cosmo_rs", "cosmo_"],
)

# Chemprop - グラフニューラルネットワーク特徴量
feature_metadata.register_engine(
    "chemprop",
    ["chemprop_"],
)

# PaDEL - PaDEL記述子
feature_metadata.register_engine(
    "padel",
    ["padel_"],
)

# SKFP - Scikit-learn fingerprint
feature_metadata.register_engine(
    "skfp",
    ["FCFP", "skfp_"],
)

# P-SMILES - SMILES-based特徴量
feature_metadata.register_engine(
    "psmiles",
    ["psmiles_"],
)

# Mol2Vec - 言語モデル特徴量
feature_metadata.register_engine(
    "mol2vec",
    ["mol2vec_"],
)

# MolAI - AI特徴量
feature_metadata.register_engine(
    "molai",
    ["molai_"],
)

# UMA - ユニバーサルモデル特徴量
feature_metadata.register_engine(
    "uma",
    ["uma_"],
)

# UniPka - pKa特徴量
feature_metadata.register_engine(
    "unipka",
    ["unipka_"],
)

# Conformer Ensemble - 立体配座アンサンブル特徴量
feature_metadata.register_engine(
    "conformer_ensemble",
    ["ens_"],
)

# Group Contribution - 基準寄与特徴量
feature_metadata.register_engine(
    "group_contrib",
    ["gc_", "group_"],
)

# DescriptaStorus - 記述子ライブラリ
feature_metadata.register_engine(
    "descriptastorus",
    ["dstorus_"],
)

# Morgan fingerprints
feature_metadata.register_engine(
    "morgan",
    ["morgan_fp", "morgan_"],
)
