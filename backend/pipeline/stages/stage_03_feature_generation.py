"""
backend/pipeline/stages/03_feature_generation.py

多項式や交互作用など、特徴量を生成するステージ。
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures

class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, generator_type: str, **kwargs):
        self.generator_type = generator_type
        self.kwargs = kwargs
        self._transformer = None
        
    def fit(self, X, y=None):
        if self.generator_type == "polynomial":
            degree = self.kwargs.get("degree", 2)
            self._transformer = PolynomialFeatures(degree=degree, include_bias=False)
            self._transformer.fit(X, y)
        elif self.generator_type == "interaction":
            self._transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            self._transformer.fit(X, y)
        return self
        
    def transform(self, X):
        if self._transformer is not None:
            return self._transformer.transform(X)
        return X

    def get_ui_schema(self) -> dict:
        return {
            "type": "object",
            "title": "特徴量生成",
            "properties": {
                "generator_type": {
                    "type": "string",
                    "enum": ["polynomial", "interaction", "custom"],
                    "default": "polynomial"
                },
                "degree": {
                    "type": "integer",
                    "default": 2
                }
            }
        }
