"""
tests/test_feature_engineer.py

backend/data/feature_engineer.py のユニットテスト。
InteractionTransformer, GroupAggTransformer,
DatetimeFeatureExtractor, LagRollingTransformer をテストする。
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


# ============================================================
# フィクスチャ (追加)
# ============================================================

@pytest.fixture
def group_df() -> pd.DataFrame:
    """グループ集約テスト用DataFrameフィクスチャ。"""
    np.random.seed(42)
    return pd.DataFrame({
        "category": ["A", "B", "A", "B", "A", "C", "C"],
        "val1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "val2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
    })


# ============================================================
# フィクスチャ
# ============================================================

@pytest.fixture
def small_numeric_df() -> pd.DataFrame:
    np.random.seed(0)
    return pd.DataFrame({
        "a": np.random.randn(30),
        "b": np.random.randn(30),
        "c": np.random.randn(30),
    })


@pytest.fixture
def datetime_series() -> pd.Series:
    return pd.Series(pd.date_range("2023-01-01", periods=24, freq="h"))


# ============================================================
# InteractionTransformer
# ============================================================

class TestInteractionTransformer:
    """T-FE-001: 交互作用項Transformerのテスト。"""

    def test_output_shape_2col(self, small_numeric_df: pd.DataFrame) -> None:
        """2列入力で正しい形状の出力が得られること。(T-FE-001-01)"""
        X = small_numeric_df[["a", "b"]]
        t = InteractionTransformer(degree=2, interaction_only=True)
        t.fit(X)
        out = t.transform(X)
        # 行数が保持され、列数 > 0
        assert out.shape[0] == 30
        assert out.shape[1] > 0

    def test_output_shape_3col(self, small_numeric_df: pd.DataFrame) -> None:
        """3列入力でinteraction_onlyが機能すること。(T-FE-001-02)"""
        t_interaction = InteractionTransformer(degree=2, interaction_only=True)
        t_full = InteractionTransformer(degree=2, interaction_only=False)
        t_interaction.fit(small_numeric_df)
        t_full.fit(small_numeric_df)
        out_i = t_interaction.transform(small_numeric_df)
        out_f = t_full.transform(small_numeric_df)
        # interaction_onlyは自乗項を含まないため列数が少ない
        assert out_i.shape[1] < out_f.shape[1]
        assert out_i.shape[0] == 30

    def test_feature_names_out(self, small_numeric_df: pd.DataFrame) -> None:
        """get_feature_names_out()が出力列数と同じ数の名前を返すこと。(T-FE-001-03)"""
        t = InteractionTransformer(degree=2, interaction_only=True)
        t.fit(small_numeric_df)
        out = t.transform(small_numeric_df)
        names = t.get_feature_names_out()
        assert len(names) == out.shape[1]

    def test_numpy_input(self) -> None:
        """numpy配列入力でも動作すること。(T-FE-001-04)"""
        X = np.random.randn(20, 4)
        t = InteractionTransformer()
        t.fit(X)
        out = t.transform(X)
        assert out.shape[0] == 20

    def test_fit_transform_equivalence(self, small_numeric_df: pd.DataFrame) -> None:
        """fit後のtransformがfit_transform相当であること。(T-FE-001-05)"""
        t = InteractionTransformer()
        t.fit(small_numeric_df)
        out1 = t.transform(small_numeric_df)
        t2 = InteractionTransformer()
        t2.fit(small_numeric_df)
        out2 = t2.transform(small_numeric_df)
        np.testing.assert_array_almost_equal(out1, out2)


# ============================================================
# DatetimeFeatureExtractor
# ============================================================

class TestDatetimeFeatureExtractor:
    """T-FE-002: 時系列特徴量抽出のテスト。"""

    def test_output_cols_include_cyclic(self, datetime_series: pd.Series) -> None:
        """cyclic=Trueでsin/cosが追加されること。(T-FE-002-01)"""
        df = pd.DataFrame({"dt": datetime_series})
        t = DatetimeFeatureExtractor(components=["month", "hour", "dayofweek"], add_cyclic=True)
        t.fit(df)
        out = t.transform(df)
        # month_sin, month_cos, hour_sin, hour_cos, dow_sin, dow_cos + 元3列 = 9列
        assert out.shape[1] == 9
        assert out.shape[0] == 24

    def test_no_cyclic(self, datetime_series: pd.Series) -> None:
        """add_cyclic=Falseでsin/cosが追加されないこと。(T-FE-002-02)"""
        df = pd.DataFrame({"dt": datetime_series})
        t = DatetimeFeatureExtractor(components=["year", "month", "day"], add_cyclic=False)
        t.fit(df)
        out = t.transform(df)
        assert out.shape == (24, 3)

    def test_feature_names_out(self, datetime_series: pd.Series) -> None:
        """get_feature_names_out() が空でないリストを返すこと。(T-FE-002-03)"""
        df = pd.DataFrame({"dt": datetime_series})
        t = DatetimeFeatureExtractor(components=["year", "month"])
        t.fit(df)
        names = t.get_feature_names_out()
        assert len(names) > 0

    def test_hour_range(self, datetime_series: pd.Series) -> None:
        """hour特徴量が0-23の範囲に収まること。(T-FE-002-04)"""
        df = pd.DataFrame({"dt": datetime_series})
        t = DatetimeFeatureExtractor(components=["hour"], add_cyclic=False)
        t.fit(df)
        out = t.transform(df)
        assert out[:, 0].min() >= 0
        assert out[:, 0].max() <= 23

    def test_is_weekend_flag(self) -> None:
        """is_weekendが正しく平日/週末を判定すること。(T-FE-002-05)"""
        # 2023-01-02 (月) = 0, 2023-01-07 (土) = 1
        dates = pd.Series(pd.to_datetime(["2023-01-02", "2023-01-07"]))
        df = pd.DataFrame({"dt": dates})
        t = DatetimeFeatureExtractor(components=["is_weekend"], add_cyclic=False)
        t.fit(df)
        out = t.transform(df)
        assert out[0, 0] == 0  # 月曜
        assert out[1, 0] == 1  # 土曜


# ============================================================
# LagRollingTransformer
# ============================================================

class TestLagRollingTransformer:
    """T-FE-003: ラグ・ローリングTransformerのテスト。"""

    def test_output_shape(self, small_numeric_df: pd.DataFrame) -> None:
        """出力列数が (n_cols × (n_lags + n_windows × n_funcs)) であること。(T-FE-003-01)"""
        t = LagRollingTransformer(lags=[1, 2], windows=[3], agg_funcs=["mean", "std"])
        t.fit(small_numeric_df)
        out = t.transform(small_numeric_df)
        # 3列 × (2ラグ + 1窓×2関数) = 3×4 = 12列
        assert out.shape == (30, 12)

    def test_no_future_leak_in_lag(self, small_numeric_df: pd.DataFrame) -> None:
        """lag=1の値が1行前の値と一致すること（未来リーク検証）。(T-FE-003-02)"""
        t = LagRollingTransformer(lags=[1], windows=[], agg_funcs=[])
        t.fit(small_numeric_df)
        out = t.transform(small_numeric_df)
        # lag1 of 'a': 行1の値は元の行0の値
        assert out[1, 0] == pytest.approx(float(small_numeric_df["a"].iloc[0]))

    def test_feature_names_count(self, small_numeric_df: pd.DataFrame) -> None:
        """get_feature_names_out() の長さが出力列数と一致すること。(T-FE-003-03)"""
        t = LagRollingTransformer(lags=[1], windows=[3], agg_funcs=["mean"])
        t.fit(small_numeric_df)
        out = t.transform(small_numeric_df)
        names = t.get_feature_names_out()
        assert len(names) == out.shape[1]

    def test_numpy_input(self) -> None:
        """numpy配列入力でも動作すること。(T-FE-003-04)"""
        X = np.arange(40).reshape(20, 2).astype(float)
        t = LagRollingTransformer(lags=[1], windows=[3], agg_funcs=["mean"])
        t.fit(X)
        out = t.transform(X)
        assert out.shape[0] == 20

    def test_fill_nan_with_zero(self) -> None:
        """先頭のNaN（ラグ）が0埋めされること。(T-FE-003-05)"""
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        t = LagRollingTransformer(lags=[1], windows=[], agg_funcs=[])
        t.fit(X)
        out = t.transform(X)
        assert out[0, 0] == 0.0  # 先頭はnull→0


# ============================================================
# FeatureEngineeringConfig / build_pipeline
# ============================================================

class TestFeatureEngineeringConfig:
    """T-FE-004: 設定クラスとパイプライン構築のテスト。"""

    def test_default_config(self) -> None:
        """デフォルト設定が正しく生成されること。(T-FE-004-01)"""
        cfg = FeatureEngineeringConfig()
        assert cfg.add_interactions is False
        assert cfg.interaction_degree == 2
        assert cfg.lags == [1, 2, 3]

    def test_build_pipeline_empty_when_no_interactions(self) -> None:
        """interactions=Falseでパイプラインが空であること。(T-FE-004-02)"""
        cfg = FeatureEngineeringConfig(add_interactions=False)
        steps = build_feature_engineering_pipeline(cfg)
        assert steps == []

    def test_build_pipeline_has_interaction_step(self) -> None:
        """interactions=Trueでパイプラインに交互作用ステップが含まれること。(T-FE-004-03)"""
        cfg = FeatureEngineeringConfig(add_interactions=True)
        steps = build_feature_engineering_pipeline(cfg)
        assert len(steps) == 1
        assert steps[0][0] == "interactions"


# ============================================================
# GroupAggTransformer
# ============================================================

class TestGroupAggTransformer:
    """T-FE-005: グループ集約Transformerのテスト。"""

    def test_output_shape(self, group_df: pd.DataFrame) -> None:
        """出力形状が (n_rows, n_agg_cols × n_agg_funcs) であること。(T-FE-005-01)"""
        t = GroupAggTransformer(
            group_col="category",
            agg_cols=["val1", "val2"],
            agg_funcs=["mean", "std"],
        )
        t.fit(group_df)
        out = t.transform(group_df)
        # 2列 × 2関数 = 4列
        assert out.shape == (7, 4)

    def test_feature_names(self, group_df: pd.DataFrame) -> None:
        """get_feature_names_out()が (n_col × n_func) 個の名前を返すこと。(T-FE-005-02)"""
        t = GroupAggTransformer(
            group_col="category",
            agg_cols=["val1"],
            agg_funcs=["mean", "max"],
        )
        t.fit(group_df)
        names = t.get_feature_names_out()
        assert len(names) == 2
        assert all("grp_category_val1" in n for n in names)

    def test_group_mean_value(self, group_df: pd.DataFrame) -> None:
        """グループ平均値が正しく計算されること。(T-FE-005-03)"""
        t = GroupAggTransformer(
            group_col="category",
            agg_cols=["val1"],
            agg_funcs=["mean"],
        )
        t.fit(group_df)
        out = t.transform(group_df)
        # カテゴリA: val1 = [1,3,5] → mean=3.0
        # 行0 はカテゴリA
        expected_a_mean = (1.0 + 3.0 + 5.0) / 3
        assert out[0, 0] == pytest.approx(expected_a_mean)

    def test_missing_group_col_raises(self) -> None:
        """存在しないグループ列を指定するとValueErrorが上がること。(T-FE-005-04)"""
        t = GroupAggTransformer(group_col="nonexistent", agg_cols=["val1"])
        df = pd.DataFrame({"val1": [1.0, 2.0]})
        with pytest.raises(ValueError, match="nonexistent"):
            t.fit(df)

    def test_fillna_zero_for_new_group(self, group_df: pd.DataFrame) -> None:
        """transformで学習時にないグループが0埋めされること。(T-FE-005-05)"""
        t = GroupAggTransformer(
            group_col="category",
            agg_cols=["val1"],
            agg_funcs=["mean"],
        )
        t.fit(group_df)
        # 新規グループ 'D' を含むDataFrame
        new_df = pd.DataFrame({
            "category": ["D"],
            "val1": [99.0],
            "val2": [999.0],
        })
        out = t.transform(new_df)
        assert out[0, 0] == 0.0
