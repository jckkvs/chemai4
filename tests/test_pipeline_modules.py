# -*- coding: utf-8 -*-
"""
tests/test_pipeline_modules.py

backend/pipeline 下の4モジュールのユニットテスト。
  - ColPreprocessor (col_preprocessor.py)
  - ColumnSelectorWrapper (column_selector.py)
  - FeatureGenerator (feature_generator.py)
  - FeatureSelector (feature_selector.py)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from backend.pipeline.col_preprocessor import ColPreprocessor, ColPreprocessConfig
from backend.pipeline.column_selector import ColumnSelectorWrapper, ColumnMeta
from backend.pipeline.feature_generator import FeatureGenerator, FeatureGenConfig
from backend.pipeline.feature_selector import FeatureSelector, FeatureSelectorConfig


# ═══════════════════════════════════════════════════════════════════
# テスト用フィクスチャ
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """数値・カテゴリ・バイナリ混合DataFrame。"""
    rng = np.random.RandomState(42)
    n = 50
    return pd.DataFrame({
        "num1": rng.randn(n),
        "num2": rng.rand(n) * 100,
        "cat_low": rng.choice(["A", "B", "C"], n),
        "cat_high": rng.choice([f"val_{i}" for i in range(30)], n),
        "binary": rng.choice([0, 1], n),
    })


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    """数値のみのDataFrame。"""
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "x1": rng.randn(30),
        "x2": rng.randn(30) * 10,
        "x3": rng.rand(30),
    })


@pytest.fixture
def regression_target(numeric_df) -> np.ndarray:
    return numeric_df["x1"].values * 2 + numeric_df["x2"].values * 0.5 + np.random.randn(30) * 0.1


# ═══════════════════════════════════════════════════════════════════
# ColPreprocessor テスト
# ═══════════════════════════════════════════════════════════════════

class TestColPreprocessor:

    def test_default_config_fit_transform(self, mixed_df):
        """デフォルト設定でfit/transformが動作すること。"""
        cp = ColPreprocessor()
        result = cp.fit_transform(mixed_df)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(mixed_df)
        assert result.shape[1] > 0

    def test_feature_names_out(self, mixed_df):
        """変換後の特徴量名が取得できること。"""
        cp = ColPreprocessor()
        cp.fit(mixed_df)
        names = cp.get_feature_names_out()
        assert len(names) > 0
        assert isinstance(names, np.ndarray)

    def test_column_transformer_property(self, mixed_df):
        """column_transformerプロパティがfit後に利用可能なこと。"""
        cp = ColPreprocessor()
        cp.fit(mixed_df)
        ct = cp.column_transformer
        assert ct is not None

    def test_column_transformer_raises_before_fit(self):
        """fit前のcolumn_transformer取得でRuntimeError。"""
        cp = ColPreprocessor()
        with pytest.raises(RuntimeError):
            _ = cp.column_transformer

    def test_transform_raises_before_fit(self, mixed_df):
        """fit前のtransformでRuntimeError。"""
        cp = ColPreprocessor()
        with pytest.raises(RuntimeError):
            cp.transform(mixed_df)

    def test_numeric_scaler_variants(self, mixed_df):
        """各スケーラーで動作すること。"""
        for scaler in ["standard", "minmax", "robust", "maxabs", "none"]:
            cfg = ColPreprocessConfig(numeric_scaler=scaler)
            cp = ColPreprocessor(cfg)
            result = cp.fit_transform(mixed_df)
            assert result.shape[0] == len(mixed_df)

    def test_numeric_imputer_variants(self):
        """各imputer戦略で動作すること。"""
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, np.nan, 40.0, 50.0],
        })
        for imputer in ["mean", "median", "knn", "constant"]:
            cfg = ColPreprocessConfig(numeric_imputer=imputer)
            cp = ColPreprocessor(cfg)
            result = cp.fit_transform(df)
            assert not np.isnan(result).any(), f"imputer={imputer} に NaN が残存"

    def test_override_types(self, mixed_df):
        """override_typesで列型を上書きできること。"""
        cfg = ColPreprocessConfig(
            override_types={"num1": "passthrough", "binary": "numeric"}
        )
        cp = ColPreprocessor(cfg)
        result = cp.fit_transform(mixed_df)
        assert result.shape[0] == len(mixed_df)

    def test_encoder_onehot(self, mixed_df):
        """OneHotEncoderで列展開されること。"""
        cfg = ColPreprocessConfig(cat_low_encoder="onehot")
        cp = ColPreprocessor(cfg)
        result = cp.fit_transform(mixed_df)
        # OneHotで列が増える
        assert result.shape[1] >= 5

    def test_encoder_ordinal(self, mixed_df):
        """OrdinalEncoderで動作すること。"""
        cfg = ColPreprocessConfig(cat_low_encoder="ordinal")
        cp = ColPreprocessor(cfg)
        result = cp.fit_transform(mixed_df)
        assert result.shape[0] == len(mixed_df)

    def test_ndarray_input(self):
        """numpy配列の入力でも動作すること。"""
        arr = np.random.randn(20, 3)
        cp = ColPreprocessor()
        result = cp.fit_transform(arr)
        assert result.shape[0] == 20

    def test_empty_df_raises(self):
        """空のDataFrameでValueError。"""
        cp = ColPreprocessor()
        with pytest.raises((ValueError, RuntimeError)):
            cp.fit_transform(pd.DataFrame())


# ═══════════════════════════════════════════════════════════════════
# ColumnSelectorWrapper テスト
# ═══════════════════════════════════════════════════════════════════

class TestColumnSelectorWrapper:

    def test_all_mode(self, numeric_df):
        """mode='all'で全列が返ること。"""
        cs = ColumnSelectorWrapper(mode="all")
        result = cs.fit_transform(numeric_df)
        assert list(result.columns) == list(numeric_df.columns)

    def test_include_mode(self, numeric_df):
        """mode='include'で指定列のみ返ること。"""
        cs = ColumnSelectorWrapper(mode="include", columns=["x1", "x3"])
        result = cs.fit_transform(numeric_df)
        assert list(result.columns) == ["x1", "x3"]

    def test_exclude_mode(self, numeric_df):
        """mode='exclude'で指定列が除外されること。"""
        cs = ColumnSelectorWrapper(mode="exclude", columns=["x2"])
        result = cs.fit_transform(numeric_df)
        assert "x2" not in result.columns
        assert "x1" in result.columns

    def test_include_col_range(self, numeric_df):
        """col_rangeで列範囲指定ができること。"""
        cs = ColumnSelectorWrapper(mode="include", col_range=(0, 2))
        result = cs.fit_transform(numeric_df)
        assert result.shape[1] == 2

    def test_invalid_mode_raises(self, numeric_df):
        """不正modeでValueError。"""
        cs = ColumnSelectorWrapper(mode="invalid")
        with pytest.raises(ValueError, match="未知"):
            cs.fit(numeric_df)

    def test_ndarray_raises(self):
        """numpy配列入力でTypeError。"""
        cs = ColumnSelectorWrapper()
        with pytest.raises(TypeError):
            cs.fit(np.array([[1, 2], [3, 4]]))

    def test_get_feature_names_out(self, numeric_df):
        """選択列名が返ること。"""
        cs = ColumnSelectorWrapper(mode="include", columns=["x1"])
        cs.fit(numeric_df)
        names = cs.get_feature_names_out()
        assert list(names) == ["x1"]

    def test_selected_columns_property(self, numeric_df):
        """selected_columnsプロパティが正しいこと。"""
        cs = ColumnSelectorWrapper(mode="include", columns=["x2", "x3"])
        cs.fit(numeric_df)
        assert cs.selected_columns == ["x2", "x3"]

    def test_column_meta(self):
        """ColumnMetaが正しく取得できること。"""
        meta = {
            "x1": ColumnMeta(monotonic=1, linearity="linear"),
            "x2": ColumnMeta(monotonic=-1),
        }
        cs = ColumnSelectorWrapper(mode="all", column_meta=meta)
        df = pd.DataFrame({"x1": [1], "x2": [2], "x3": [3]})
        cs.fit(df)

        assert cs.get_column_meta("x1").monotonic == 1
        assert cs.get_column_meta("x2").monotonic == -1
        assert cs.get_column_meta("x3").monotonic == 0  # デフォルト

    def test_monotonic_constraints(self):
        """get_monotonic_constraintsが正しいタプルを返すこと。"""
        meta = {"a": ColumnMeta(monotonic=1), "b": ColumnMeta(monotonic=-1)}
        cs = ColumnSelectorWrapper(mode="all", column_meta=meta)
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        cs.fit(df)
        constraints = cs.get_monotonic_constraints()
        assert constraints == (1, -1, 0)

    def test_groups_array(self):
        """get_groups_arrayが正しいリストを返すこと。"""
        meta = {
            "a": ColumnMeta(group="grp1"),
            "b": ColumnMeta(group="grp1"),
            "c": ColumnMeta(group="grp2"),
        }
        cs = ColumnSelectorWrapper(mode="all", column_meta=meta)
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        cs.fit(df)
        groups = cs.get_groups_array()
        assert groups == ["grp1", "grp1", "grp2"]

    def test_missing_column_warning(self, numeric_df):
        """存在しない列名を指定した場合にwarning。"""
        cs = ColumnSelectorWrapper(mode="include", columns=["x1", "nonexistent"])
        cs.fit(numeric_df)
        assert "x1" in cs.selected_columns
        assert "nonexistent" not in cs.selected_columns

    def test_zero_selection_raises(self, numeric_df):
        """選択列が0件になるとValueError。"""
        cs = ColumnSelectorWrapper(mode="exclude", columns=list(numeric_df.columns))
        with pytest.raises(ValueError, match="0件"):
            cs.fit(numeric_df)


# ═══════════════════════════════════════════════════════════════════
# FeatureGenerator テスト
# ═══════════════════════════════════════════════════════════════════

class TestFeatureGenerator:

    def test_none_passthrough(self, numeric_df):
        """method='none'でパススルー。"""
        fg = FeatureGenerator(FeatureGenConfig(method="none"))
        result = fg.fit_transform(numeric_df)
        np.testing.assert_array_equal(result, numeric_df.values)

    def test_polynomial(self, numeric_df):
        """polynomialで特徴量が増加すること。"""
        fg = FeatureGenerator(FeatureGenConfig(method="polynomial", degree=2))
        result = fg.fit_transform(numeric_df)
        # 3特徴 → polynomial(degree=2, no bias): 3 + 3 + 3 = 9
        assert result.shape[1] > numeric_df.shape[1]
        assert result.shape[0] == len(numeric_df)

    def test_interaction_only(self, numeric_df):
        """interaction_onlyで交互作用項のみ。"""
        fg = FeatureGenerator(FeatureGenConfig(method="interaction_only", degree=2))
        result = fg.fit_transform(numeric_df)
        assert result.shape[1] > numeric_df.shape[1]
        # interaction_only は polynomial より列数が少ない
        fg_poly = FeatureGenerator(FeatureGenConfig(method="polynomial", degree=2))
        result_poly = fg_poly.fit_transform(numeric_df)
        assert result.shape[1] <= result_poly.shape[1]

    def test_get_feature_names_out(self, numeric_df):
        """特徴量名が取得できること。"""
        fg = FeatureGenerator(FeatureGenConfig(method="polynomial", degree=2))
        fg.fit(numeric_df)
        names = fg.get_feature_names_out()
        assert len(names) > 0

    def test_is_passthrough(self):
        """is_passthroughプロパティ。"""
        fg_none = FeatureGenerator(FeatureGenConfig(method="none"))
        assert fg_none.is_passthrough is True

        fg_poly = FeatureGenerator(FeatureGenConfig(method="polynomial"))
        assert fg_poly.is_passthrough is False

    def test_n_output_features(self, numeric_df):
        """n_output_featuresがfit後に正しい値。"""
        fg = FeatureGenerator(FeatureGenConfig(method="polynomial", degree=2))
        fg.fit(numeric_df)
        assert fg.n_output_features > numeric_df.shape[1]

    def test_include_bias(self, numeric_df):
        """include_bias=Trueでバイアス列が追加されること。"""
        fg_no = FeatureGenerator(FeatureGenConfig(method="polynomial", degree=2, include_bias=False))
        fg_yes = FeatureGenerator(FeatureGenConfig(method="polynomial", degree=2, include_bias=True))
        r_no = fg_no.fit_transform(numeric_df)
        r_yes = fg_yes.fit_transform(numeric_df)
        assert r_yes.shape[1] == r_no.shape[1] + 1

    def test_ndarray_input(self):
        """numpy配列でも動作すること。"""
        arr = np.random.randn(20, 4)
        fg = FeatureGenerator(FeatureGenConfig(method="polynomial", degree=2))
        result = fg.fit_transform(arr)
        assert result.shape[0] == 20
        assert result.shape[1] > 4


# ═══════════════════════════════════════════════════════════════════
# FeatureSelector テスト
# ═══════════════════════════════════════════════════════════════════

class TestFeatureSelector:

    def test_none_passthrough(self, numeric_df, regression_target):
        """method='none'でパススルー。"""
        fs = FeatureSelector(FeatureSelectorConfig(method="none"))
        result = fs.fit_transform(numeric_df, regression_target)
        np.testing.assert_array_equal(result, numeric_df.values)

    def test_lasso_selection(self, numeric_df, regression_target):
        """Lasso特徴量選択が動作すること。"""
        fs = FeatureSelector(FeatureSelectorConfig(
            method="lasso", task="regression"
        ))
        result = fs.fit_transform(numeric_df, regression_target)
        assert result.shape[0] == len(numeric_df)
        assert result.shape[1] <= numeric_df.shape[1]

    def test_rfr_selection(self, numeric_df, regression_target):
        """RandomForestで特徴量選択。"""
        fs = FeatureSelector(FeatureSelectorConfig(
            method="rfr", task="regression"
        ))
        result = fs.fit_transform(numeric_df, regression_target)
        assert result.shape[0] == len(numeric_df)

    def test_select_kbest(self, numeric_df, regression_target):
        """SelectKBestで特徴量選択。"""
        fs = FeatureSelector(FeatureSelectorConfig(
            method="select_kbest", task="regression", k=2
        ))
        result = fs.fit_transform(numeric_df, regression_target)
        assert result.shape[1] == 2

    def test_select_percentile(self, numeric_df, regression_target):
        """SelectPercentileで特徴量選択。"""
        fs = FeatureSelector(FeatureSelectorConfig(
            method="select_percentile", task="regression", percentile=50
        ))
        result = fs.fit_transform(numeric_df, regression_target)
        assert result.shape[1] > 0
        assert result.shape[1] <= numeric_df.shape[1]

    def test_get_feature_names_out(self, numeric_df, regression_target):
        """選択後の特徴量名が取得できること。"""
        fs = FeatureSelector(FeatureSelectorConfig(
            method="select_kbest", task="regression", k=2
        ))
        fs.fit(numeric_df, regression_target)
        names = fs.get_feature_names_out()
        assert len(names) == 2

    def test_support_mask(self, numeric_df, regression_target):
        """support_maskプロパティが正しいサイズであること。"""
        fs = FeatureSelector(FeatureSelectorConfig(
            method="select_kbest", task="regression", k=2
        ))
        fs.fit(numeric_df, regression_target)
        mask = fs.support_mask
        assert mask is not None
        assert len(mask) == numeric_df.shape[1]
        assert mask.sum() == 2

    def test_unknown_method_fallback(self, numeric_df, regression_target):
        """未知メソッドでRandomForestフォールバックが動作すること。"""
        fs = FeatureSelector(FeatureSelectorConfig(method="unknown_method"))
        result = fs.fit_transform(numeric_df, regression_target)
        # RF フォールバック → 特徴量が選択される（全列以下）
        assert result.shape[0] == len(numeric_df)
        assert result.shape[1] <= numeric_df.shape[1]
        assert result.shape[1] > 0

    def test_classification_task(self):
        """分類タスクでの動作確認。"""
        rng = np.random.RandomState(42)
        X = pd.DataFrame({
            "f1": rng.randn(50),
            "f2": rng.randn(50),
            "f3": rng.randn(50),
        })
        y = (X["f1"] > 0).astype(int).values

        fs = FeatureSelector(FeatureSelectorConfig(
            method="select_kbest", task="classification", k=2
        ))
        result = fs.fit_transform(X, y)
        assert result.shape[1] == 2

    def test_ndarray_input(self, regression_target):
        """numpy配列入力でも動作すること。"""
        arr = np.random.randn(30, 3)
        fs = FeatureSelector(FeatureSelectorConfig(
            method="select_kbest", k=2
        ))
        result = fs.fit_transform(arr, regression_target)
        assert result.shape[1] == 2

    def test_ridge_selection(self, numeric_df, regression_target):
        """Ridge特徴量選択。"""
        fs = FeatureSelector(FeatureSelectorConfig(
            method="ridge", task="regression"
        ))
        result = fs.fit_transform(numeric_df, regression_target)
        assert result.shape[0] == len(numeric_df)
