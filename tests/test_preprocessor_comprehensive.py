"""
tests/test_preprocessor_comprehensive.py

preprocessor.py の包括テスト。
LogTransformer / SinCosTransformer / PreprocessConfig / Preprocessor.build /
build_full_pipeline を網羅。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from backend.data.preprocessor import (
    LogTransformer,
    SinCosTransformer,
    PreprocessConfig,
    Preprocessor,
    build_full_pipeline,
)
from backend.data.type_detector import TypeDetector


@pytest.fixture
def mixed_df():
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "num1": rng.randn(30),
        "num2": rng.randn(30) * 10 + 50,
        "cat": np.random.choice(["a", "b", "c"], 30),
        "bin": np.random.choice([0, 1], 30),
        "target": rng.randn(30),
    })


@pytest.fixture
def detection_result(mixed_df):
    detector = TypeDetector()
    return detector.detect(mixed_df)


class TestLogTransformer:
    def test_transform(self):
        X = np.array([[1, 2], [3, 4]])
        lt = LogTransformer()
        result = lt.transform(X)
        assert result.shape == X.shape
        assert np.all(np.isfinite(result))

    def test_inverse(self):
        X = np.array([[1, 2], [3, 4]], dtype=float)
        lt = LogTransformer()
        transformed = lt.transform(X)
        recovered = lt.inverse_transform(transformed)
        np.testing.assert_allclose(recovered, X, atol=1e-10)

    def test_fit_returns_self(self):
        lt = LogTransformer()
        result = lt.fit(np.array([[1, 2]]))
        assert result is lt


class TestSinCosTransformer:
    def test_transform(self):
        X = np.array([[0], [np.pi / 2], [np.pi]])
        sc = SinCosTransformer(period=2 * np.pi)
        result = sc.transform(X)
        assert result.shape == (3, 2)

    def test_feature_names(self):
        sc = SinCosTransformer()
        names = sc.get_feature_names_out()
        assert list(names) == ["sin", "cos"]

    def test_period_360(self):
        X = np.array([[0], [90], [180], [360]])
        sc = SinCosTransformer(period=360)
        result = sc.transform(X)
        assert np.abs(result[0, 0]) < 1e-10  # sin(0) == 0
        assert np.abs(result[1, 0] - 1.0) < 1e-10  # sin(90°) == 1


class TestPreprocessConfig:
    def test_defaults(self):
        cfg = PreprocessConfig()
        assert cfg.numeric_scaler == "auto"
        assert cfg.cat_low_encoder == "onehot"

    def test_custom(self):
        cfg = PreprocessConfig(numeric_scaler="robust")
        assert cfg.numeric_scaler == "robust"


class TestPreprocessor:
    def test_build(self, detection_result):
        pp = Preprocessor()
        ct = pp.build(detection_result, target_col="target")
        assert ct is not None

    def test_transformer_property(self, detection_result):
        pp = Preprocessor()
        pp.build(detection_result, target_col="target")
        ct = pp.transformer
        assert ct is not None

    def test_transformer_before_build(self):
        pp = Preprocessor()
        with pytest.raises(RuntimeError, match="build"):
            _ = pp.transformer

    def test_fit_transform(self, detection_result, mixed_df):
        pp = Preprocessor()
        ct = pp.build(detection_result, target_col="target")
        X = mixed_df.drop(columns=["target"])
        ct.fit(X)
        result = ct.transform(X)
        assert result.shape[0] == 30

    def test_custom_scaler(self, detection_result):
        cfg = PreprocessConfig(numeric_scaler="minmax")
        pp = Preprocessor(config=cfg)
        ct = pp.build(detection_result, target_col="target")
        assert ct is not None

    def test_knn_imputer(self, detection_result):
        cfg = PreprocessConfig(numeric_imputer="knn")
        pp = Preprocessor(config=cfg)
        ct = pp.build(detection_result, target_col="target")
        assert ct is not None


class TestBuildFullPipeline:
    def test_basic(self, detection_result, mixed_df):
        model = Ridge()
        pipe = build_full_pipeline(detection_result, model, target_col="target")
        X = mixed_df.drop(columns=["target"])
        y = mixed_df["target"]
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (30,)
