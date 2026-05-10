"""
backend/pipeline/stages/01_column_control.py

使用する説明変数を制御（選択・除外など）するためのパイプラインステージ。
"""
from typing import List, Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    指定された条件に基づいてDataFrameの一部列を抽出する。
    """
    def __init__(self, mode: str = "all", columns: Optional[List[str]] = None, start: Optional[str] = None, end: Optional[str] = None):
        """
        mode: "all", "include", "exclude", "range"
        """
        self.mode = mode
        self.columns = columns or []
        self.start = start
        self.end = end
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            # To handle numpy array if mistakenly passed
            X = pd.DataFrame(X)
            
        if self.mode == "all":
            return X
        elif self.mode == "include":
            valid_cols = [c for c in self.columns if c in X.columns]
            return X[valid_cols]
        elif self.mode == "exclude":
            valid_cols = [c for c in X.columns if c not in self.columns]
            return X[valid_cols]
        elif self.mode == "range":
            cols = list(X.columns)
            try:
                start_idx = cols.index(self.start) if self.start else 0
                end_idx = cols.index(self.end) if self.end else len(cols)
                # Ensure start <= end
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx
                return X.iloc[:, start_idx:end_idx+1]
            except ValueError:
                # If start/end not found, return all
                return X
        return X

    def get_ui_schema(self) -> dict:
        return {
            "type": "object",
            "title": "列選択制御",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["all", "include", "exclude", "range"],
                    "default": "all"
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "start": {"type": "string"},
                "end": {"type": "string"}
            }
        }
