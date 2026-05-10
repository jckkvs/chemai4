"""
backend/pipeline/stages/04_feature_selection.py

特徴量選択の手法を提供するステージ。
"""
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression

class SelectorFactory:
    @staticmethod
    def create(method: str, task: str = "auto", **kwargs):
        """
        method: "select_from_model", "lasso", "rfr"
        task: "regression", "classification", "auto"
        """
        # 単純化のため、taskに基づいてモデルを分岐する実装
        is_classification = (task == "classification")
        
        if method == "select_from_model" or method == "rfr":
            if is_classification:
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            return SelectFromModel(estimator)
            
        elif method == "lasso":
            if is_classification:
                estimator = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)
            else:
                estimator = Lasso(alpha=0.1, random_state=42)
            return SelectFromModel(estimator)
            
        return "passthrough"

    @staticmethod
    def get_ui_schema() -> dict:
        return {
            "type": "object",
            "title": "特徴量選択",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["select_from_model", "lasso", "rfr", "none"],
                    "default": "none"
                }
            }
        }
