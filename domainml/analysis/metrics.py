"""
Implements: F-205
制約充足度スコア計算器。Kendall Tauに基づくペア毎違反判定。
"""
import numpy as np
from scipy.stats import kendalltau

def constraint_satisfaction_score(X, y_pred, feature_idx=0, constraint_type='increasing'):
    """
    Implements: F-205
    指定された特徴量(Xのfeature_idx列)に対する予測値y_predの単調性制約の充足割合を計算する。
    1.0 ならば完全に制約を満たしており、0.0 ならば完全に逆転している。
    """
    if X.ndim == 1:
        x_feat = X
    else:
        x_feat = X[:, feature_idx]
        
    sort_idx = np.argsort(x_feat)
    x_sorted = x_feat[sort_idx]
    y_sorted = y_pred[sort_idx]
    
    # 完全にフラットな場合などのNaN対策
    if np.all(y_sorted == y_sorted[0]):
        return 1.0
        
    tau, _ = kendalltau(x_sorted, y_sorted)
    if np.isnan(tau):
        tau = 0.0
        
    if constraint_type == 'increasing':
        score = (tau + 1.0) / 2.0
    elif constraint_type == 'decreasing':
        score = (-tau + 1.0) / 2.0
    else:
        raise ValueError(f"Invalid constraint type: {constraint_type}")
        
    return float(score)
