"""
tests/test_preprocessor_extra.py

preprocessor.py のカバレッジ改善テスト。
LogTransformer, SinCosTransformer, PreprocessConfig, Preprocessor, build_full_pipeline を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.data.preprocessor import (
    LogTransformer,
    SinCosTransformer,
    PreprocessConfig,
    Preprocessor,
    build_full_pipeline,
)
from backend.data.type_detector import TypeDetector


# ============================================================
# LogTransformer
# ============================================================

class TestLogTransformer:
    def test_transform(self):
        X = np.array([[1.0], [2.0], [10.0]])
        lt = LogTransformer(offset=1.0)
        result = lt.fit_transform(X)
        assert result.shape == X.shape
        assert result[0, 0] > 0

    def test_inverse(self):
        X = np.array([[1.0], [2.0], [10.0]])
        lt = LogTransformer(offset=1.0)
        transformed = lt.fit_transform(X)
        recovered = lt.inverse_transform(transformed)
        np.testing.assert_allclose(recovered, X, atol=1e-6)

    def test_offset_zero(self):
        X = np.array([[1.0], [2.0]])
        lt = LogTransformer(offset=0.0)
        result = lt.fit_transform(X)
        assert result.shape == X.shape


# ============================================================
# SinCosTransformer
# ============================================================

class TestSinCosTransformer:
    def test_default_period(self):
        X = np.array([[0], [np.pi / 2], [np.pi]])
        sc = SinCosTransformer()
        result = sc.fit_transform(X)
        assert result.shape == (3, 2)

    def test_degree_period(self):
        X = np.array([[0], [90], [180], [360]])
        sc = SinCosTransformer(period=360)
        result = sc.fit_transform(X)
        assert result.shape == (4, 2)
        np.testing.assert_allclose(result[0, 0], 0, atol=1e-10)  # sin(0)

    def test_feature_names(self):
        sc = SinCosTransformer()
        names = sc.get_feature_names_out()
        assert list(names) == ["sin", "cos"]


# ============================================================
# PreprocessConfig
# ============================================================

class TestPreprocessConfig:
    def test_defaults(self):
        cfg = PreprocessConfig()
        assert cfg.numeric_scaler == "auto"
        assert cfg.cat_low_encoder == "onehot"
        assert cfg.numeric_imputer == "mean"
        assert cfg.add_missing_indicator is True

    def test_custom(self):
        cfg = PreprocessConfig(
            numeric_scaler="robust",
            cat_low_encoder="ordinal",
            numeric_imputer="knn",
        )
        assert cfg.numeric_scaler == "robust"


# ============================================================
# Preprocessor
# ============================================================

class TestPreprocessor:
    def _make_detection_result(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "normal": rng.randn(20),
            "cat": ["A", "B", "C", "A"] * 5,
            "flag": [0, 1] * 10,
            "const": [1] * 20,
        })
        det = TypeDetector()
        return det.detect(df)

    def test_build(self):
        dr = self._make_detection_result()
        pp = Preprocessor()
        ct = pp.build(dr)
        assert ct is not None
        assert len(ct.transformers) >= 1

    def test_build_with_target(self):
        dr = self._make_detection_result()
        pp = Preprocessor()
        ct = pp.build(dr, target_col="normal")
        assert ct is not None

    def test_transformer_property_error(self):
        pp = Preprocessor()
        with pytest.raises(RuntimeError, match="build"):
            _ = pp.transformer

    def test_transformer_property(self):
        dr = self._make_detection_result()
        pp = Preprocessor()
        pp.build(dr)
        ct = pp.transformer
        assert ct is not None

    def test_different_scalers(self):
        dr = self._make_detection_result()
        for scaler in ["standard", "minmax", "robust", "none"]:
            cfg = PreprocessConfig(numeric_scaler=scaler)
            pp = Preprocessor(cfg)
            ct = pp.build(dr)
            assert ct is not None

    def test_knn_imputer(self):
        dr = self._make_detection_result()
        cfg = PreprocessConfig(numeric_imputer="knn")
        pp = Preprocessor(cfg)
        ct = pp.build(dr)
        assert ct is not None


# ============================================================
# build_full_pipeline
# ============================================================

class TestBuildFullPipeline:
    def test_basic(self):
        from sklearn.linear_model import Ridge
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"x": rng.randn(20), "y": rng.randn(20)})
        det = TypeDetector()
        dr = det.detect(df)
        pipe = build_full_pipeline(dr, Ridge(), target_col="y")
        assert len(pipe.steps) == 2
        assert pipe.steps[0][0] == "preprocess"
        assert pipe.steps[1][0] == "model"
