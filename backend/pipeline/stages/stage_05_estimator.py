"""
backend/pipeline/stages/05_estimator.py

最終推定器をラップし、単調性・線形性などの制約を注入する。
"""
from typing import Dict, List, Union
from sklearn.base import BaseEstimator, clone

class EstimatorWrapper(BaseEstimator):
    def __init__(
        self,
        estimator: Union[str, BaseEstimator],
        task: str = "auto",
        monotonic: Dict[str, str] = None,
        linearity: Dict[str, float] = None,
        groups: Dict[str, List[str]] = None,
        **estimator_params
    ):
        self.estimator = estimator
        self.task = task
        self.monotonic = monotonic or {}
        self.linearity = linearity or {}
        self.groups = groups or {}
        self.estimator_params = estimator_params
        self._fitted_estimator = None
        
    def inject_monotonic(self, constraint_obj):
        if hasattr(constraint_obj, "constraints"):
            self.monotonic.update(constraint_obj.constraints)
            
    def inject_linearity(self, constraint_obj):
        if hasattr(constraint_obj, "features"):
            self.linearity.update(constraint_obj.features)
            
    def inject_groups(self, constraint_obj):
        if hasattr(constraint_obj, "groups"):
            self.groups.update(constraint_obj.groups)

    def _resolve_estimator(self):
        # 文字列による指定の解決などはここで実装する
        # このデモでは単純なクローン
        if isinstance(self.estimator, BaseEstimator):
            est = clone(self.estimator)
            est.set_params(**self.estimator_params)
            
            self._needs_monotonic_fallback = False
            # 単調性制約のサポート判定
            if self.monotonic:
                from ..constraints.validator import ConstraintValidator
                est_name = type(est).__name__
                if est_name in ConstraintValidator.MONOTONIC_SUPPORTED_ESTIMATORS:
                    # LightGBM等の仕様に合わせて変換などを本来は行う
                    if hasattr(est, "set_params"):
                        est.set_params(monotone_constraints=self.monotonic)
                else:
                    # フォールバックが必要
                    self._needs_monotonic_fallback = True
                    
            return est
        pass

    def fit(self, X, y):
        self._fitted_estimator = self._resolve_estimator()
        self._fitted_estimator.fit(X, y)
        
        if getattr(self, "_needs_monotonic_fallback", False):
            from ..wrappers.monotonic_fallback import MonotonicFallbackWrapper
            self._fitted_estimator = MonotonicFallbackWrapper(self._fitted_estimator, self.monotonic)
            # Re-fit the wrapper (it will re-clone the built estimator and fit again, plus Isotonic logic)
            # or we can pass a pre-fitted estimator if we tweak wrapper's fit
            self._fitted_estimator.fit(X, y)
            
        return self
        
    def predict(self, X):
        return self._fitted_estimator.predict(X)

    def get_ui_schema(self) -> dict:
        return {
            "type": "object",
            "title": "Estimator設定",
            "properties": {
                "estimator": {
                    "type": "string",
                    "description": "Estimatorの選択",
                }
            }
        }
