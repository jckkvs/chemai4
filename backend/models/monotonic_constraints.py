import numpy as np
import pandas as pd
from typing import Dict, List

CONSTRAINT_STRENGTH_MAP = {
    "strong": {"threshold": 0.9, "enforcement": "hard"},   # 0.9〜1.0: 必須遵守
    "medium": {"threshold": 0.5, "enforcement": "soft"},   # 0.5〜0.9: ペナルティ強化
    "weak": {"threshold": 0.1, "enforcement": "soft"},     # 0.1〜0.5: 軽微なペナルティ
    "none": {"threshold": 0.0, "enforcement": "none"},     # 0.0〜0.1: 制約なし
}

def get_enforcement_mode(strength: float) -> str:
    """強度値から強制モードを判定"""
    for mode, config in CONSTRAINT_STRENGTH_MAP.items():
        if float(strength) >= config["threshold"]:
            return config["enforcement"]
    return "none"

class ConstraintRangeCalculator:
    """制約適用範囲（±nσ）の計算エンジン"""
    
    @staticmethod
    def compute_feature_stats(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, dict]:
        """各特徴量の統計量を計算"""
        stats = {}
        for col in feature_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                clean = df[col].dropna()
                if len(clean) > 0:
                    stats[col] = {
                        "mean": float(clean.mean()),
                        "std": float(clean.std()),
                        "min": float(clean.min()),
                        "max": float(clean.max()),
                        "q01": float(clean.quantile(0.01)),
                        "q99": float(clean.quantile(0.99)),
                    }
        return stats
    
    @staticmethod
    def get_constraint_bounds(stats: dict, sigma: float) -> tuple:
        """±nσ範囲の境界値を計算（nが負または極端に大きい場合は訓練範囲にフォールバック）"""
        if sigma < 0 or abs(sigma) > 10:
            # 実質制約なし：訓練データの範囲を使用
            return stats["min"], stats["max"]
        
        lower = stats["mean"] - abs(sigma) * stats["std"]
        upper = stats["mean"] + abs(sigma) * stats["std"]
        # 現実的な範囲にクリップ（分位点で制限）
        lower = max(lower, stats.get("q01", stats["min"]))
        upper = min(upper, stats.get("q99", stats["max"]))
        return lower, upper

class ConstraintToModelParams:
    """制約設定を各モデルのパラメータ形式に変換"""
    
    @staticmethod
    def to_lightgbm_params(constraints: Dict[str, dict], feature_names: List[str], 
                          feature_stats: Dict[str, dict]) -> dict:
        """LightGBM用 monotone_constraints パラメータ生成"""
        # 強い制約（strength >= 0.9）のみ変換
        constraint_array = []
        for feat in feature_names:
            c = constraints.get(feat, {})
            if c.get("strength", 0) < 0.9:
                constraint_array.append(0)  # 制約なし
                continue
            
            direction = c.get("direction", "none")
            c.get("sigma_range", 3.0)
            
            if direction == "increasing":
                constraint_array.append(1)
            elif direction == "decreasing":
                constraint_array.append(-1)
            else:
                constraint_array.append(0)
        
        return {"monotone_constraints": constraint_array}

    @staticmethod
    def to_xgboost_params(constraints: Dict[str, dict], feature_names: List[str], 
                          feature_stats: Dict[str, dict]) -> dict:
        """XGBoost用 monotone_constraints パラメータ生成"""
        return ConstraintToModelParams.to_lightgbm_params(constraints, feature_names, feature_stats)
    
    @staticmethod
    def to_penalty_term(constraints: Dict[str, dict], predictions: np.ndarray, 
                       X: pd.DataFrame, feature_stats: Dict[str, dict], 
                       lambda_reg: float = 0.1) -> float:
        """弱い制約・線形性用のペナルティ項計算（カスタム損失関数用）"""
        penalty = 0.0
        for feat, c in constraints.items():
            if feat not in X.columns:
                continue
            
            strength = c.get("strength", 0)
            if strength <= 0:
                continue  # 制約なし
            
            direction = c.get("direction", "none")
            linearity = c.get("linearity", False)
            sigma = c.get("sigma_range", 3.0)
            
            # 適用範囲内のサンプルのみ対象
            lower, upper = ConstraintRangeCalculator.get_constraint_bounds(feature_stats.get(feat, {"min": -np.inf, "max": np.inf, "mean": 0, "std": 1}), sigma)
            mask = (X[feat] >= lower) & (X[feat] <= upper)
            if not mask.any():
                continue
            
            x_vals = X.loc[mask, feat].values
            y_pred = predictions[mask]
            
            # 単調性ペナルティ：隣接サンプル間の勾配符号違反を罰する
            if direction != "none" and not linearity:
                sorted_idx = np.argsort(x_vals)
                x_sorted = x_vals[sorted_idx]
                y_sorted = y_pred[sorted_idx]
                gradients = np.diff(y_sorted) / (np.diff(x_sorted) + 1e-8)
                
                if direction == "increasing":
                    violations = np.sum(np.maximum(0, -gradients) ** 2)
                else:  # decreasing
                    violations = np.sum(np.maximum(0, gradients) ** 2)
                penalty += strength * lambda_reg * violations
            
            # 線形性ペナルティ：予測値と線形回帰からの残差を罰する
            if linearity:
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression().fit(x_vals.reshape(-1, 1), y_pred)
                residuals = y_pred - lr.predict(x_vals.reshape(-1, 1))
                penalty += strength * lambda_reg * np.mean(residuals ** 2)
        
        return penalty
