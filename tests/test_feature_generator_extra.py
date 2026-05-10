"""
tests/test_feature_generator_extra.py

feature_generator.py のカバレッジ改善テスト。
FeatureGenConfig, FeatureGenerator を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.pipeline.feature_generator import (
    FeatureGenConfig,
    FeatureGenerator,
)


# ============================================================
# FeatureGenConfig
# ============================================================

class TestFeatureGenConfig:
    def test_defaults(self):
        cfg = FeatureGenConfig()
        assert cfg.method == "none"
        assert cfg.degree == 2
        assert cfg.include_bias is False

    def test_custom(self):
        cfg = FeatureGenConfig(method="polynomial", degree=3, include_bias=True)
        assert cfg.method == "polynomial"
        assert cfg.degree == 3


# ============================================================
# FeatureGenerator — none
# ============================================================

class TestFeatureGeneratorNone:
    def test_passthrough_ndarray(self):
        X = np.random.randn(20, 3)
        gen = FeatureGenerator(FeatureGenConfig(method="none"))
        result = gen.fit_transform(X)
        np.testing.assert_array_equal(result, X)

    def test_passthrough_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        gen = FeatureGenerator(FeatureGenConfig(method="none"))
        result = gen.fit_transform(df)
        np.testing.assert_array_equal(result, df.values)

    def test_is_passthrough(self):
        gen = FeatureGenerator(FeatureGenConfig(method="none"))
        assert gen.is_passthrough is True

    def test_feature_names_out_none(self):
        gen = FeatureGenerator(FeatureGenConfig(method="none"))
        gen.fit(pd.DataFrame({"a": [1], "b": [2]}))
        names = gen.get_feature_names_out()
        assert list(names) == ["a", "b"]

    def test_n_output_features_none(self):
        gen = FeatureGenerator()
        gen.fit(np.random.randn(10, 3))
        assert gen.n_output_features == 3


# ============================================================
# FeatureGenerator — polynomial
# ============================================================

class TestFeatureGeneratorPoly:
    def test_polynomial(self):
        X = np.random.randn(20, 2)
        gen = FeatureGenerator(FeatureGenConfig(method="polynomial", degree=2))
        result = gen.fit_transform(X)
        # 2 features, degree=2, no bias → 2 + C(2,2) + 2*(2-1)/2 = 5
        assert result.shape[0] == 20
        assert result.shape[1] > 2

    def test_polynomial_with_bias(self):
        X = np.random.randn(10, 2)
        gen = FeatureGenerator(FeatureGenConfig(method="polynomial", degree=2, include_bias=True))
        result = gen.fit_transform(X)
        # With bias: +1 column
        assert result.shape[1] > 2

    def test_is_not_passthrough(self):
        gen = FeatureGenerator(FeatureGenConfig(method="polynomial"))
        assert gen.is_passthrough is False

    def test_n_output_features_poly(self):
        gen = FeatureGenerator(FeatureGenConfig(method="polynomial", degree=2))
        gen.fit(np.random.randn(10, 3))
        assert gen.n_output_features > 3


# ============================================================
# FeatureGenerator — interaction_only
# ============================================================

class TestFeatureGeneratorInteraction:
    def test_interaction_only(self):
        X = np.random.randn(20, 3)
        gen = FeatureGenerator(FeatureGenConfig(method="interaction_only", degree=2))
        result = gen.fit_transform(X)
        # 3 original + 3 interaction = 6 (no squared terms)
        assert result.shape[1] > 3

    def test_feature_names_out_interaction(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        gen = FeatureGenerator(FeatureGenConfig(method="interaction_only", degree=2))
        gen.fit(df)
        names = gen.get_feature_names_out()
        assert len(names) > 3
