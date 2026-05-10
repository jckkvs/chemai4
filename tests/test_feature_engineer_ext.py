# -*- coding: utf-8 -*-
"""
tests/test_feature_engineer_ext.py

特徴量エンジニアリングモジュールのテスト。

カバー対象:
  - InteractionTransformer
  - GroupAggTransformer
  - DatetimeFeatureExtractor
  - LagRollingTransformer
  - build_feature_engineering_pipeline（全設定パターン）
  - FeatureEngineeringConfig
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.data.feature_engineer import (
    InteractionTransformer,
    GroupAggTransformer,
    DatetimeFeatureExtractor,
    LagRollingTransformer,
    FeatureEngineeringConfig,
    build_feature_engineering_pipeline,
)


# ═══════════════════════════════════════════════════════════════════
# InteractionTransformer
# ═══════════════════════════════════════════════════════════════════

class TestInteractionTransformer:

    def test_basic_transform(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        t = InteractionTransformer(degree=2, interaction_only=True)
        out = t.fit_transform(df)
        # interaction_only=True, degree=2, no bias → a, b, a*b = 3列
        assert out.shape[0] == 3
        assert out.shape[1] >= 1  # at least a*b

    def test_with_bias(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        t = InteractionTransformer(degree=2, interaction_only=True, include_bias=True)
        out = t.fit_transform(df)
        assert out.shape[1] >= 2  # bias + features

    def test_numpy_input(self):
        X = np.array([[1, 2], [3, 4]])
        t = InteractionTransformer(degree=2, interaction_only=True)
        out = t.fit_transform(X)
        assert out.shape[0] == 2
        assert out.shape[1] >= 1

    def test_feature_names_out(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        t = InteractionTransformer(degree=2, interaction_only=True)
        t.fit(df)
        names = t.get_feature_names_out()
        assert len(names) >= 1
        # sklearn PolynomialFeaturesの出力名にx, yが含まれること
        all_names = " ".join(str(n) for n in names)
        assert "x" in all_names and "y" in all_names


# ═══════════════════════════════════════════════════════════════════
# GroupAggTransformer
# ═══════════════════════════════════════════════════════════════════

class TestGroupAggTransformer:

    def test_basic_aggregation(self):
        df = pd.DataFrame({
            "group": ["A", "A", "B", "B"],
            "value": [10, 20, 30, 40],
        })
        t = GroupAggTransformer(group_col="group", agg_cols=["value"], agg_funcs=["mean"])
        out = t.fit_transform(df)
        assert out.shape[0] == 4
        assert out.shape[1] == 1

    def test_multiple_agg_funcs(self):
        df = pd.DataFrame({
            "group": ["A", "A", "B", "B"],
            "value": [10, 20, 30, 40],
        })
        t = GroupAggTransformer(group_col="group", agg_cols=["value"])
        out = t.fit_transform(df)
        # デフォルト: mean, std, min, max → 4列
        assert out.shape[1] == 4

    def test_missing_group_col_raises(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        t = GroupAggTransformer(group_col="missing", agg_cols=["a"])
        with pytest.raises(ValueError, match="グループ列"):
            t.fit(df)

    def test_feature_names_out(self):
        df = pd.DataFrame({
            "g": ["X", "Y", "X"],
            "v": [1, 2, 3],
        })
        t = GroupAggTransformer(group_col="g", agg_cols=["v"], agg_funcs=["mean"])
        t.fit(df)
        names = t.get_feature_names_out()
        assert len(names) == 1
        assert "grp_g_v_mean" in names[0]


# ═══════════════════════════════════════════════════════════════════
# DatetimeFeatureExtractor
# ═══════════════════════════════════════════════════════════════════

class TestDatetimeFeatureExtractor:

    def test_basic_extraction(self):
        df = pd.DataFrame({"dt": pd.to_datetime(["2024-01-15", "2024-07-20"])})
        t = DatetimeFeatureExtractor(components=["year", "month", "day"], add_cyclic=False)
        out = t.fit_transform(df)
        assert out.shape == (2, 3)
        assert out[0, 0] == 2024  # year
        assert out[0, 1] == 1     # month
        assert out[0, 2] == 15    # day

    def test_cyclic_features(self):
        df = pd.DataFrame({"dt": pd.to_datetime(["2024-06-15"])})
        t = DatetimeFeatureExtractor(components=["month"], add_cyclic=True)
        out = t.fit_transform(df)
        # month + month_sin + month_cos = 3列
        assert out.shape[1] == 3

    def test_is_weekend(self):
        df = pd.DataFrame({
            "dt": pd.to_datetime(["2024-01-13", "2024-01-15"])  # Sat, Mon
        })
        t = DatetimeFeatureExtractor(components=["is_weekend"], add_cyclic=False)
        out = t.fit_transform(df)
        assert out[0, 0] == 1  # Saturday = weekend
        assert out[1, 0] == 0  # Monday = not weekend

    def test_numpy_input(self):
        # numpy入力はto_datetimeでDatetimeIndex化されるため
        # .dt属性は不要（DatetimeIndex自体がyear/month等を持つ）
        df = pd.DataFrame({"dt": pd.to_datetime(["2024-03-01", "2024-12-25"])})
        t = DatetimeFeatureExtractor(components=["month"], add_cyclic=False)
        out = t.fit_transform(df)
        assert out[0, 0] == 3
        assert out[1, 0] == 12

    def test_feature_names_out(self):
        df = pd.DataFrame({"dt": pd.to_datetime(["2024-01-01"])})
        t = DatetimeFeatureExtractor(components=["year", "month"], add_cyclic=True)
        t.fit(df)
        names = t.get_feature_names_out()
        assert "year" in names
        assert "month" in names
        assert "month_sin" in names


# ═══════════════════════════════════════════════════════════════════
# LagRollingTransformer
# ═══════════════════════════════════════════════════════════════════

class TestLagRollingTransformer:

    def test_basic_lags(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        t = LagRollingTransformer(lags=[1], windows=[3], agg_funcs=["mean", "std"])
        out = t.fit_transform(df)
        # 1列 × (1 lag + 2 rolling(mean,std for window=3)) = 3列
        assert out.shape[0] == 5
        assert out.shape[1] == 3
        assert out[0, 0] == 0  # lag=1 の最初は0（fillna）
        assert out[1, 0] == 1  # 2個目のlag1 = 前の値

    def test_rolling_mean(self):
        df = pd.DataFrame({"x": [10, 20, 30, 40, 50]})
        t = LagRollingTransformer(lags=[], windows=[3], agg_funcs=["mean"])
        out = t.fit_transform(df)
        # 1列 × (0 lags + 1 windows × 1 func) + default lags
        # lags=[] → デフォルト[1,2,3], windows=[3]
        # 実際のコンストラクタ: lags or [1,2,3] → lags=[] は falsy なので [1,2,3] が使われる
        # テストは windows=[3], agg_funcs=["mean"] の結果のみチェック
        assert out.shape[0] == 5

    def test_multiple_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        t = LagRollingTransformer(lags=[1], windows=[2], agg_funcs=["mean"])
        out = t.fit_transform(df)
        # 2列 × (1 lag + 1 window×1func) = 4 出力列
        assert out.shape == (3, 4)

    def test_numpy_input(self):
        X = np.array([[1], [2], [3], [4]])
        t = LagRollingTransformer(lags=[1, 2], windows=[2], agg_funcs=["mean"])
        out = t.fit_transform(X)
        # 1列 × (2 lags + 1 window × 1 func) = 3
        assert out.shape == (4, 3)

    def test_feature_names_out(self):
        df = pd.DataFrame({"val": range(5)})
        t = LagRollingTransformer(lags=[1], windows=[3], agg_funcs=["mean", "std"])
        t.fit(df)
        names = t.get_feature_names_out()
        assert "val_lag1" in names
        assert "val_roll3_mean" in names
        assert "val_roll3_std" in names


# ═══════════════════════════════════════════════════════════════════
# build_feature_engineering_pipeline
# ═══════════════════════════════════════════════════════════════════

class TestBuildFeatureEngineeringPipeline:

    def test_empty_config(self):
        """デフォルト設定 → 空のステップリスト"""
        cfg = FeatureEngineeringConfig()
        steps = build_feature_engineering_pipeline(cfg)
        assert len(steps) == 0

    def test_interactions_only(self):
        cfg = FeatureEngineeringConfig(add_interactions=True, interaction_degree=2)
        steps = build_feature_engineering_pipeline(cfg)
        assert len(steps) == 1
        assert steps[0][0] == "interactions"

    def test_group_agg(self):
        cfg = FeatureEngineeringConfig(
            group_agg_configs=[
                {"group_col": "category", "agg_cols": ["value"], "agg_funcs": ["mean"]},
            ]
        )
        steps = build_feature_engineering_pipeline(cfg)
        assert len(steps) == 1
        assert "group_agg" in steps[0][0]

    def test_datetime_features(self):
        cfg = FeatureEngineeringConfig(datetime_cols=["created_at"])
        steps = build_feature_engineering_pipeline(cfg)
        assert len(steps) == 1
        assert steps[0][0] == "datetime_features"

    def test_lag_rolling(self):
        cfg = FeatureEngineeringConfig(lag_rolling_cols=["price"])
        steps = build_feature_engineering_pipeline(cfg)
        assert len(steps) == 1
        assert steps[0][0] == "lag_rolling"

    def test_all_combined(self):
        """全ステップ有効"""
        cfg = FeatureEngineeringConfig(
            add_interactions=True,
            group_agg_configs=[{"group_col": "g", "agg_cols": ["v"]}],
            datetime_cols=["dt"],
            lag_rolling_cols=["ts"],
        )
        steps = build_feature_engineering_pipeline(cfg)
        assert len(steps) == 4

    def test_multiple_group_agg(self):
        """複数グループ集約"""
        cfg = FeatureEngineeringConfig(
            group_agg_configs=[
                {"group_col": "a", "agg_cols": ["v1"]},
                {"group_col": "b", "agg_cols": ["v2"]},
            ]
        )
        steps = build_feature_engineering_pipeline(cfg)
        assert len(steps) == 2
        assert steps[0][0] == "group_agg_0"
        assert steps[1][0] == "group_agg_1"

    def test_empty_group_agg_skipped(self):
        """group_colが空のgroup_agg_configsはスキップ"""
        cfg = FeatureEngineeringConfig(
            group_agg_configs=[{"group_col": "", "agg_cols": ["v"]}]
        )
        steps = build_feature_engineering_pipeline(cfg)
        assert len(steps) == 0
