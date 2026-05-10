"""
backend/pipeline/stages/02_preprocessing.py

列ごとに型を自動判定し、それぞれ適した欠損値補完手法やスケーリングを割り当てる。
sklearnのColumnTransformerを利用する。
"""
from typing import Dict, List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

class AutoTypeDetector:
    @staticmethod
    def build_transformer(X_columns: List[str], auto_detect: bool, configs: Dict[str, dict]) -> ColumnTransformer:
        """
        与えられた設定に基づいて ColumnTransformer を構築する。
        （※現状はダミー実装寄りで、すべての列を numeric 扱いにするなど簡略化している場合があります。
          実際のアプリケーションでは df や X_columns ベースの型チェックを行います）
        """
        # （ここではデモのため、すべての列を数値とみなして numeric_config を適用する擬似実装）
        num_cfg = configs.get("numeric", {})
        
        imputer_strategy = num_cfg.get("imputer", "median")
        scaler_type = num_cfg.get("scaler", "standard")
        
        steps = []
        if imputer_strategy:
            steps.append(("imputer", SimpleImputer(strategy=imputer_strategy)))
            
        if scaler_type == "standard":
            steps.append(("scaler", StandardScaler()))
        elif scaler_type == "minmax":
            steps.append(("scaler", MinMaxScaler()))
        elif scaler_type == "robust":
            steps.append(("scaler", RobustScaler()))
        elif scaler_type == "quantile_uniform":
            steps.append(("scaler", QuantileTransformer(output_distribution="uniform")))
        elif scaler_type == "quantile_normal":
            steps.append(("scaler", QuantileTransformer(output_distribution="normal")))
            
        num_pipeline = Pipeline(steps)
        
        return ColumnTransformer([
            ("num", num_pipeline, X_columns)
        ], remainder="drop")

class PreprocessorFactory:
    pass
