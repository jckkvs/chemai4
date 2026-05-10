"""
Implements: F-210
Bootstrapサンプリングを用いた不確実性推計 (UncertaintyEstimator)
ダミー実装・passは一切排除し、Joblibによる並列実行をサポート。
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import resample
from joblib import Parallel, delayed

class UncertaintyEstimator(BaseEstimator, RegressorMixin):
    """
    Implements: F-210
    BaseEstimatorをラップし、Bootstrap法を用いて予測の不確実性(標準偏差や信頼区間)を推計する。
    制約が競合しやすい外挿領域において不確実性が高まることを定量化するために使用する。
    """
    def __init__(self, estimator, n_bootstraps=50, n_jobs=-1, random_state=None):
        self.estimator = estimator
        self.n_bootstraps = n_bootstraps
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _fit_single_model(self, X, y, seed):
        # DataFrame/Series array compatibility
        if hasattr(X, 'iloc'):
            X_arr, y_arr = X.values, getattr(y, 'values', y)
        else:
            X_arr, y_arr = X, y
            
        X_resampled, y_resampled = resample(X_arr, y_arr, random_state=seed)
        model = clone(self.estimator)
        model.fit(X_resampled, y_resampled)
        return model

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        # Fix possible issue of RandomState mapping by picking deterministic seeds
        seeds = rng.randint(0, 100000, size=self.n_bootstraps)
        
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_model)(X, y, seed) for seed in seeds
        )
        self.is_fitted_ = True
        return self

    def predict(self, X, return_std=False):
        if not hasattr(self, 'is_fitted_'):
            from sklearn.exceptions import NotFittedError
            raise NotFittedError("This UncertaintyEstimator instance is not fitted yet.")
            
        all_preds = np.array([model.predict(X) for model in self.estimators_])
        mean_preds = np.mean(all_preds, axis=0)
        
        if return_std:
            std_preds = np.std(all_preds, axis=0)
            return mean_preds, std_preds
        return mean_preds
