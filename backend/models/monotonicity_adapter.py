import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def apply_monotonicity_constraints(
    estimator: Any,
    pipeline: Any,
    constraints_dict: Dict[str, Dict[str, Any]]
) -> Any:
    """
    変数単位の制約辞書を、選択モデルのネイティブ形式へ自動マッピングして適用する。
    """
    if not constraints_dict:
        return estimator

    try:
        feature_order = list(pipeline.get_feature_names_out())
    except AttributeError:
        logger.warning("Pipelineがget_feature_names_out()をサポートしていません。制約適用をスキップします。")
        return estimator

    def _get_orig_name(s4_name: str) -> str:
        return s4_name.split("__", 1)[1] if "__" in s4_name else s4_name

    # 制約配列の構築
    # 方向性のマッピング: increasing=1, decreasing=-1, その他(unknown/none)=0
    constraint_values = []
    applied_count = 0
    
    for feat in feature_order:
        orig_feat = _get_orig_name(feat)
        orig_feat_base = orig_feat.split("_")[0] if "_" in orig_feat and orig_feat not in constraints_dict else orig_feat
        
        cfg = constraints_dict.get(orig_feat, {})
        if not cfg:
            cfg = constraints_dict.get(orig_feat_base, {}) # Fallback for one-hot encoded categories
            
        direction = cfg.get("direction", "none")
        
        if direction == "increasing":
            val = 1
        elif direction == "decreasing":
            val = -1
        else:
            val = 0 # "unknown" や "none" は制約なしとして処理
            
        constraint_values.append(val)
        if val != 0:
            applied_count += 1

    if applied_count == 0:
        return estimator

    # モデル別適用
    model_name = estimator.__class__.__name__.lower()

    if "lgbm" in model_name or "lightgbm" in model_name:
        estimator.set_params(monotone_constraints=constraint_values)
        logger.info(f"LightGBM: {applied_count}変数に単調性制約を適用しました")
        
    elif "xgb" in model_name or "xgboost" in model_name:
        estimator.set_params(monotone_constraints=str(tuple(constraint_values)))
        logger.info(f"XGBoost: {applied_count}変数に単調性制約を適用しました")
        
    elif "catboost" in model_name:
        active_dict = {f: v for f, v in zip(feature_order, constraint_values) if v != 0}
        estimator.set_params(monotone_constraints=active_dict)
        logger.info(f"CatBoost: {len(active_dict)}変数に単調性制約を適用しました")
        
    elif "histgradient" in model_name or "hist" in model_name:
        estimator.set_params(monotonic_cst=constraint_values)
        logger.info(f"HistGradientBoosting: {applied_count}変数に単調性制約を適用しました")
        
    else:
        logger.info(f"{model_name} は単調性制約にネイティブ対応していません。設定は保持されますが適用されません。")

    return estimator
