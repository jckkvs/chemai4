import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def extract_regression_coefficients(model, feature_names: list, 
                                    X_original: np.ndarray,
                                    X_scaled: np.ndarray = None,
                                    scaler: StandardScaler = None) -> pd.DataFrame:
    """
    回帰係数を標準化前・後の両方で抽出
    
    Parameters
    ----------
    model : 学習済み線形モデル
        coef_ と intercept_ を持つモデル
    feature_names : list
        特徴量名リスト
    X_original : np.ndarray
        標準化前の元データ（n_samples × n_features）
    X_scaled : np.ndarray, optional
        標準化後のデータ（モデル学習に使用）
    scaler : StandardScaler, optional
        標準化オブジェクト（fit済み）
    
    Returns
    -------
    pd.DataFrame
        係数情報を含むDataFrame（標準化前/後の両方を含む）
    """
    coef_scaled = model.coef_.flatten() if hasattr(model.coef_, 'flatten') else model.coef_
    intercept = model.intercept_ if np.isscalar(model.intercept_) else model.intercept_[0]
    
    results = []
    for i, name in enumerate(feature_names):
        row = {
            "feature": name,
            "coef_scaled": coef_scaled[i],  # 標準化後（比較用）
            "coef_original": None,           # 標準化前（解釈用）
            "abs_coef_scaled": abs(coef_scaled[i]),
        }
        results.append(row)
    
    df_coef = pd.DataFrame(results)
    
    # 標準化前の係数を計算（可能なら）
    if scaler is not None and X_scaled is not None and hasattr(scaler, 'scale_'):
        # 標準化: x_scaled = (x - mean) / std
        # 係数の変換: beta_original = beta_scaled / std
        stds = scaler.scale_
        means = scaler.mean_
        
        df_coef["coef_original"] = coef_scaled / stds
        # 切片の調整: intercept_original = intercept_scaled - sum(beta_scaled * mean / std)
        intercept_original = intercept - np.sum(coef_scaled * means / stds)
        # return a dict or similar for intercept if needed, but for now we assign to df or keep it separate
        # df_coef does not have columns for intercept intuitively, so we will not store intercept here directly for all rows
        # But for compatibility with user code, let's keep it (it will be repeated for every row)
        df_coef["intercept_original"] = intercept_original
        df_coef["intercept_scaled"] = intercept
    else:
        df_coef["coef_original"] = coef_scaled
        df_coef["intercept_original"] = intercept
        df_coef["intercept_scaled"] = intercept
    
    # 重要度でソート
    df_coef = df_coef.sort_values("abs_coef_scaled", ascending=False).reset_index(drop=True)
    
    return df_coef
