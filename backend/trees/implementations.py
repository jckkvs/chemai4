"""
backend/trees/implementations.py
"""
from typing import Type
from .base import TreeEnsemble

class DefaultEnsembleImplementation(TreeEnsemble):
    def fit(self, X, y=None):
        return self
        
    def predict(self, X):
        return [0]*len(X)

def get_implementation_class(base_algorithm: str) -> Type[TreeEnsemble]:
    """
    指定された文字列に基づいて実際の実装クラスを返す。
    ダミーとしてデフォルト実装を返す。
    """
    return DefaultEnsembleImplementation
