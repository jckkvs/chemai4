"""
backend/pipeline/constraints/constraints.py

- 単調性制約 (MonotonicConstraint)
- 線形性制約 (LinearityConstraint)
- グループ制約 (GroupConstraint)

など、特定のEstimatorに注入する制約メタ情報を保持し、
動的UI表示用のSchemaを生成するクラス群。
"""
from typing import Dict, List

class BaseConstraint:
    def get_ui_schema(self) -> dict:
        return {}

class MonotonicConstraint(BaseConstraint):
    def __init__(self, constraints: Dict[str, str] = None, **kwargs):
        """
        constraints: {"feature_name": "increasing" | "decreasing" | "none"}
        """
        self.constraints = constraints or {}

    def get_ui_schema(self) -> dict:
        return {
            "type": "object",
            "title": "単調性制約",
            "properties": {
                "constraints": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "string",
                        "enum": ["increasing", "decreasing", "none"]
                    }
                }
            }
        }

class LinearityConstraint(BaseConstraint):
    def __init__(self, features: Dict[str, float] = None, method: str = "penalty", **kwargs):
        """
        features: {"feature_name": strength_of_linearity (0.0 to 1.0)}
        method: "penalty" | "projection"
        """
        self.features = features or {}
        self.method = method

    def get_ui_schema(self) -> dict:
        return {
            "type": "object",
            "title": "線形性制約",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["penalty", "projection"],
                    "default": "penalty"
                },
                "features": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                }
            }
        }

class GroupConstraint(BaseConstraint):
    def __init__(self, groups: Dict[str, List[str]] = None, **kwargs):
        """
        groups: {"group_name": ["feature1", "feature2", ...]}
        """
        self.groups = groups or {}

    def get_ui_schema(self) -> dict:
        return {
            "type": "object",
            "title": "グループ化制約",
            "properties": {
                "groups": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        }
