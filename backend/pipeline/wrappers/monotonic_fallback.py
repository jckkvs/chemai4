"""
backend/pipeline/wrappers/monotonic_fallback.py
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.isotonic import IsotonicRegression

class MonotonicFallbackWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator, monotonic_map: dict):
        self.base_estimator = base_estimator
        self.monotonic_map = monotonic_map
        self.fitted_iso_ = {}

    def fit(self, X, y):
        self.model_ = clone(self.base_estimator).fit(X, y)
        X = np.array(X)
        for idx, dir_ in self.monotonic_map.items():
            if dir_ in ("increasing", "decreasing"):
                ir = IsotonicRegression(increasing=(dir_ == "increasing"), out_of_bounds="clip")
                ir.fit(X[:, int(idx)], y)
                self.fitted_iso_[int(idx)] = ir
        return self

    def predict(self, X):
        X = np.array(X)
        y_base = self.model_.predict(X)
        y_adj = y_base.copy()
        
        for idx, ir in self.fitted_iso_.items():
            y_iso = ir.predict(X[:, idx])
            # 単調性逸脱を補正（簡易版）
            mask = (y_iso > y_adj) if ir.increasing else (y_iso < y_adj)
            y_adj[mask] = y_iso[mask]
            
        return y_adj
