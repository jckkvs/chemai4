"""
backend/trees/base.py
"""
from sklearn.base import BaseEstimator

class TreeEnsemble(BaseEstimator):
    """
    あらゆるタイプの決定木・フォレストアルゴリズムを表す基底クラス。
    """
    def __init__(self, base_config: dict, features: dict):
        self.base_config = base_config
        self.features = features
