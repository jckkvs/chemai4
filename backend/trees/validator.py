"""
backend/trees/validator.py
"""
from typing import Dict

class OptionValidator:
    """
    追加しようとしているトッピング機能が、ベースアルゴリズムや既存トッピングと衝突しないか検証する。
    """
    def check_conflicts(self, current_features: Dict[str, dict], new_feature: str, new_params: dict) -> list:
        # デモ実装: 特別な衝突ロジックは実装せず空リストを返す
        conflicts = []
        return conflicts

    def resolve_conflicts(self, feature_name: str, params: dict, conflicts: list) -> dict:
        return params
