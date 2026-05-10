"""
tests/test_feature_engineer_extra.py

feature_engineer.py のカバレッジ改善テスト。
InteractionTransformer, GroupAggTransformer, DatetimeFeatureExtractor,
LagRollingTransformer, FeatureEngineeringConfig, build_feature_engineering_pipeline を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.data.feature_engineer import (
    InteractionTransformer,
    GroupAggTransformer,
    DatetimeFeatureExtractor,
    LagRollingTransformer,
    FeatureEngineeringConfig,
    build_feature_engineering_pipeline,
)


# ============================================================
# InteractionTransformer
# ============================================================

class TestInteractionTransformer:
    def test_fit_transform_ndarray(self):
        X = np.random.randn(20, 3)
        t = InteractionTransformer(degree=2)
        result = t.fit_transform(X)
        assert result.shape[0] == 20
        assert result.shape[1] > 3

    def test_fit_transform_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        t = InteractionTransformer(degree=2)
        result = t.fit_transform(df)
        assert result.shape[0] == 3

    def test_feature_names_out(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        t = InteractionTransformer(degree=2, interaction_only=True)
        t.fit(df)
        names = t.get_feature_names_out()
        assert len(names) > 2

    def test_not_fitted(self):
        t = InteractionTransformer()
        names = t.get_feature_names_out()
        assert len(names) == 0


# ============================================================
# GroupAggTransformer
# ============================================================

class TestGroupAggTransformer:
    def test_basic(self):
        df = pd.DataFrame({
            "grp": ["A", "A", "B", "B"],
            "val1": [1, 2, 3, 4],
            "val2": [10, 20, 30, 40],
        })
        t = GroupAggTransformer(group_col="grp", agg_cols=["val1", "val2"])
        result = t.fit_transform(df)
        assert result.shape[0] == 4
        assert result.shape[1] > 0

    def test_feature_names(self):
        df = pd.DataFrame({"g": ["A", "B"], "v": [1, 2]})
        t = GroupAggTransformer(group_col="g", agg_cols=["v"], agg_funcs=["mean"])
        t.fit(df)
        names = t.get_feature_names_out()
        assert len(names) >= 1

    def test_missing_group_col(self):
        df = pd.DataFrame({"x": [1, 2]})
        t = GroupAggTransformer(group_col="nonexist", agg_cols=["x"])
        with pytest.raises(ValueError, match="存在しません"):
            t.fit(df)


# ============================================================
# DatetimeFeatureExtractor
# ============================================================

class TestDatetimeFeatureExtractor:
    def test_basic(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({"ts": dates})
        t = DatetimeFeatureExtractor(add_cyclic=True)
        result = t.fit_transform(df)
        assert result.shape[0] == 5
        assert result.shape[1] > 5  # Many components + cyclic

    def test_no_cyclic(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="h")
        df = pd.DataFrame({"ts": dates})
        t = DatetimeFeatureExtractor(add_cyclic=False)
        result = t.fit_transform(df)
        assert result.shape[0] == 3

    def test_specific_components(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame({"ts": dates})
        t = DatetimeFeatureExtractor(components=["year", "month"], add_cyclic=False)
        t.fit(df)
        names = t.get_feature_names_out()
        assert "year" in names
        assert "month" in names

    def test_ndarray_input(self):
        """ndarray入力をDataFrame経由で変換"""
        dates = pd.DataFrame({"dt": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])})
        t = DatetimeFeatureExtractor(components=["year", "month"], add_cyclic=False)
        result = t.fit_transform(dates)
        assert result.shape[0] == 3


# ============================================================
# LagRollingTransformer
# ============================================================

class TestLagRollingTransformer:
    def test_basic_ndarray(self):
        X = np.random.randn(20, 2)
        t = LagRollingTransformer(lags=[1, 2], windows=[3], agg_funcs=["mean"])
        result = t.fit_transform(X)
        assert result.shape[0] == 20
        # 2 features × (2 lags + 1 window × 1 func) = 6
        assert result.shape[1] == 6

    def test_basic_dataframe(self):
        df = pd.DataFrame({"a": np.arange(10), "b": np.arange(10, 20)})
        t = LagRollingTransformer(lags=[1], windows=[3], agg_funcs=["mean", "std"])
        result = t.fit_transform(df)
        assert result.shape[0] == 10
        # 2 features × (1 lag + 1 window × 2 funcs) = 6
        assert result.shape[1] == 6

    def test_feature_names(self):
        X = np.random.randn(10, 2)
        t = LagRollingTransformer(lags=[1], windows=[3], agg_funcs=["mean"])
        t.fit(X)
        names = t.get_feature_names_out()
        assert len(names) == 4  # 2 × (1 lag + 1 rolling)


# ============================================================
# FeatureEngineeringConfig & build_pipeline
# ============================================================

class TestFeatureEngineeringConfig:
    def test_defaults(self):
        cfg = FeatureEngineeringConfig()
        assert cfg.add_interactions is False
        assert cfg.add_cyclic is True

    def test_build_empty_pipeline(self):
        cfg = FeatureEngineeringConfig()
        steps = build_feature_engineering_pipeline(cfg)
        assert len(steps) == 0

    def test_build_with_interactions(self):
        cfg = FeatureEngineeringConfig(add_interactions=True)
        steps = build_feature_engineering_pipeline(cfg)
        assert len(steps) == 1
        assert steps[0][0] == "interactions"

    def test_build_with_datetime(self):
        cfg = FeatureEngineeringConfig(datetime_cols=["ts"])
        steps = build_feature_engineering_pipeline(cfg)
        assert len(steps) == 1
        assert steps[0][0] == "datetime_features"

    def test_build_with_lag_rolling(self):
        cfg = FeatureEngineeringConfig(lag_rolling_cols=["val"])
        steps = build_feature_engineering_pipeline(cfg)
        assert len(steps) == 1
        assert steps[0][0] == "lag_rolling"

    def test_build_with_group_agg(self):
        cfg = FeatureEngineeringConfig(
            group_agg_configs=[{"group_col": "g", "agg_cols": ["v"]}]
        )
        steps = build_feature_engineering_pipeline(cfg)
        assert len(steps) == 1
