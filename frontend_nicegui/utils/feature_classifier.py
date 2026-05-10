from typing import List, Dict
from enum import Enum

class FeatureSource(Enum):
    RAW = "raw"              # CSV等から読み込んだ生データ
    SMILES_DERIVED = "smiles_derived"  # SMILESから生成された記述子
    ENGINEERED = "engineered"  # 交互作用項・多項式特徴量など

class FeatureClassifier:
    """特徴量の種別・所属セットを判定・管理するユーティリティ"""
    
    # 既知のSMILES記述子エンジンと特徴量プレフィックスのマッピング
    ENGINE_PREFIX_MAP = {
        "rdkit": ["rdkit_", "molwt", "logp", "tpsa", "num_hbd", "num_hba", "fp_"],
        "mordred": ["mordred_", "ABC", "ATS", "BCUT", "GETAWAY"],  # Mordred記述子コード
        "gfn2_xtb": ["xtb_", "homo", "lumo", "dipole", "mulliken_"],
        "unipka": ["unipka_", "pka_", "logd_", "solv_"],
        "cosmo": ["cosmo_", "sigma_", "dG_solv"],
        "molai": ["molai_pca_", "latent_"],
        "group_contrib": ["gc_", "contrib_"],
    }
    
    @classmethod
    def classify_feature(cls, feature_name: str, known_sources: Dict[str, List[str]] = None) -> Dict[str, str]:
        """特徴量名から種別・エンジン・セット名を判定"""
        name_lower = feature_name.lower()
        
        # 1. 既知ソースからの明示的マッピングチェック
        if known_sources:
            for engine, prefixes in known_sources.items():
                if any(name_lower.startswith(p.lower()) or p.lower() in name_lower for p in prefixes):
                    return {
                        "source": FeatureSource.SMILES_DERIVED.value,
                        "engine": engine,
                        "set_name": f"{engine.upper()} Features"
                    }
        
        # 2. 静的マッピングによるフォールバック判定
        for engine, prefixes in cls.ENGINE_PREFIX_MAP.items():
            if any(name_lower.startswith(p.lower()) or p.lower() in name_lower for p in prefixes):
                return {
                    "source": FeatureSource.SMILES_DERIVED.value,
                    "engine": engine,
                    "set_name": f"{engine.upper()} Features"
                }
        
        # 3. 工程特徴量（交互作用・多項式）の検出
        if any(op in feature_name for op in ["_x_", "_sq", "_poly", ":"]):
            return {
                "source": FeatureSource.ENGINEERED.value,
                "engine": "pipeline",
                "set_name": "Engineered Features"
            }
        
        # 4. 默认: 生データ由来
        return {
            "source": FeatureSource.RAW.value,
            "engine": None,
            "set_name": "Raw Input Features"
        }
    
    @classmethod
    def group_features_by_set(cls, feature_names: List[str], 
                             known_sources: Dict[str, List[str]] = None) -> Dict[str, List[Dict]]:
        """特徴量リストをセット別にグルーピング"""
        grouped = {}
        for feat in feature_names:
            meta = cls.classify_feature(feat, known_sources)
            set_name = meta["set_name"]
            if set_name not in grouped:
                grouped[set_name] = {
                    "engine": meta["engine"],
                    "source": meta["source"],
                    "features": []
                }
            grouped[set_name]["features"].append({
                "name": feat,
                "metadata": meta
            })
        return grouped
