"""
backend/pipeline/constraints/validator.py

制約とEstimatorの互換性を検証する機能。
"""
from typing import List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ConstraintValidator:
    """
    パイプラインに設定された制約条件が、設定されたEstimatorでサポートされているかを検証・判定する。
    """
    
    # 単調性制約をネイティブサポートしているよく知られたアルゴリズム
    MONOTONIC_SUPPORTED_ESTIMATORS = [
        "LGBMRegressor", "LGBMClassifier",
        "XGBRegressor", "XGBClassifier",
        "CatBoostRegressor", "CatBoostClassifier",
        "HistGradientBoostingRegressor", "HistGradientBoostingClassifier"
    ]
    
    @classmethod
    def check_compatibility(cls, estimator: Any, constraints_list: List[Any]) -> Tuple[List[Any], List[str]]:
        """
        Estimatorと制約の互換性をチェックする。
        互換性のないものは除外し、警告メッセージのリストを返す。
        
        Returns:
            (有効な制約のリスト, 警告メッセージのリスト)
        """
        valid_constraints = []
        warnings = []
        
        # クラス名から判定、またはラッピングされたベース推定器を展開して判定
        estimator_name = cls._get_estimator_name(estimator)
        
        for c in constraints_list:
            c_type = type(c).__name__
            
            if c_type == "MonotonicConstraint":
                if estimator_name not in cls.MONOTONIC_SUPPORTED_ESTIMATORS:
                    msg = f"MonotonicConstraint は {estimator_name} ではネイティブサポートされていません。この制約はスキップされます。"
                    logger.warning(msg)
                    warnings.append(msg)
                else:
                    valid_constraints.append(c)
                    
            elif c_type == "LinearityConstraint":
                # 今後のロジック追加用（特定のペナルティ対応判定など）
                valid_constraints.append(c)
                
            elif c_type == "GroupConstraint":
                # Group Lassoなどの判定
                valid_constraints.append(c)
                
            else:
                # 未知の制約
                valid_constraints.append(c)
                
        return valid_constraints, warnings

    @classmethod
    def _get_estimator_name(cls, estimator: Any) -> str:
        # もし Pipeline などでラップされているなら展開するなどの処理を入れる
        # ここでは単にクラス名を返す
        
        if hasattr(estimator, "estimator"):
            # Meta-estimator の場合
            return type(estimator.estimator).__name__
            
        return type(estimator).__name__
