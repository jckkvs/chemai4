"""
backend/trees/composer.py

決定木・フォレスト系の「トッピング」可能統合クラス。
ユーザーはベースアルゴリズムを選び、任意の機能を組み合わせてカスタムアンサンブルを構築可能。
"""
from typing import Dict
from .base import TreeEnsemble
from .validator import OptionValidator

class TreeComposer:
    """
    決定木・フォレスト系の機能を柔軟に設定するコンポーズクラス。
    """
    
    BASE_ALGORITHMS = {
        "random_forest",
        "extra_trees",
        "rotation_forest",
        "regularized_greedy",
        "sigmoid_tree",
        "linear_tree",
        "rulefit",
        "sporf",
        "cart",
        "bernoulli_forest",
    }
    
    AVAILABLE_FEATURES = {
        "regularization": {
            "type": "l1", # default as placeholder
            "params": {"alpha": float, "l1_ratio": float}
        },
        "variable_usage_penalty": {
            "type": "none",
            "params": {"enabled": bool, "penalty_strength": float}
        },
        "leaf_weight_regularization": {
            "type": "none",
            "params": {"enabled": bool, "min_leaf_weight": float}
        },
        "monotonic_constraints": {
            "type": "none",
            "params": {"constraints": dict}
        },
        "linearity_promotion": {
            "type": "none",
            "params": {"features": dict, "method": str}
        },
        "oblique_splits": {
            "type": "none",
            "params": {"enabled": bool, "max_combinations": int}
        },
        "random_rotation": {
            "type": "none",
            "params": {"enabled": bool, "n_pca_components": int}
        },
        "sigmoid_output": {
            "type": "none",
            "params": {"enabled": bool, "temperature": float}
        },
    }
    
    def __init__(self, base_algorithm: str):
        if base_algorithm not in self.BASE_ALGORITHMS:
            raise ValueError(f"Unknown base algorithm: {base_algorithm}")
        self.base = base_algorithm
        self.features: Dict[str, dict] = {}
        self._validator = OptionValidator()
    
    def add_feature(self, feature_name: str, **params) -> "TreeComposer":
        if feature_name not in self.AVAILABLE_FEATURES:
            raise ValueError(f"Unknown feature: {feature_name}")
            
        conflicts = self._validator.check_conflicts(
            current_features=self.features,
            new_feature=feature_name,
            new_params=params
        )
        
        if conflicts:
            import warnings
            warnings.warn(
                f"組み合わせの矛盾: {conflicts} → 自動調整します。",
                UserWarning
            )
            params = self._validator.resolve_conflicts(feature_name, params, conflicts)
            
        self.features[feature_name] = {
            "enabled": True,
            **params
        }
        return self
        
    def remove_feature(self, feature_name: str) -> "TreeComposer":
        self.features.pop(feature_name, None)
        return self
        
    def build(self) -> TreeEnsemble:
        from .implementations import get_implementation_class
        ImplClass = get_implementation_class(self.base)
        return ImplClass(base_config={"algorithm": self.base}, features=self.features)
        
    def get_ui_schema(self) -> dict:
        return {
            "base_algorithm": {
                "type": "string",
                "enum": list(self.BASE_ALGORITHMS),
                "default": self.base,
                "ui:widget": "radio"
            },
            "features": {
                "type": "object",
                "properties": {
                    name: {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean", "default": False},
                            # 簡略化のため個別の厳密な型展開は省略
                        },
                        "required": ["enabled"],
                        "ui:collapsible": True,
                        "ui:group": self._get_feature_group(name)
                    }
                    for name in self.AVAILABLE_FEATURES.keys()
                },
                "ui:layout": "tabs",
                "ui:searchable": True
            }
        }
        
    def _map_type(self, py_type) -> str:
        mapping = {int: "integer", float: "number", str: "string", bool: "boolean", list: "array", dict: "object"}
        return mapping.get(py_type, "string")
        
    def _get_feature_group(self, feature_name: str) -> str:
        groups = {
            "regularization": ["regularization", "variable_usage_penalty", "leaf_weight_regularization"],
            "constraints": ["monotonic_constraints", "linearity_promotion"],
            "splits": ["oblique_splits", "random_rotation"],
            "output": ["sigmoid_output"]
        }
        for group, members in groups.items():
            if feature_name in members:
                return group
        return "other"
