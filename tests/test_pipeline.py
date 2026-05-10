"""
tests/test_pipeline.py

backend/pipeline/ パッケージの全コンポーネントに対するテスト。

テスト構成:
  T-001〜T-005: ColumnSelectorWrapper（全列/include/excludeモード）
  T-006〜T-011: ColPreprocessor（数値/カテゴリ/混在/上書き）
  T-012〜T-015: FeatureGenerator（none/polynomial/interaction_only）
  T-016〜T-023: FeatureSelector（各手法）
  T-024〜T-028: build_pipeline（回帰・分類 end-to-end）
  T-029〜T-031: apply_monotonic_constraints / extract_group_array
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import Ridge

# ---- テスト対象 ----
from backend.pipeline.column_selector import ColumnMeta, ColumnSelectorWrapper
from backend.pipeline.col_preprocessor import ColPreprocessConfig, ColPreprocessor
from backend.pipeline.feature_generator import FeatureGenConfig, FeatureGenerator
from backend.pipeline.feature_selector import FeatureSelectorConfig, FeatureSelector
from backend.pipeline.pipeline_builder import (
    PipelineConfig,
    build_pipeline,
    apply_monotonic_constraints,
    extract_group_array,
)


# ==============================================================
# フィクスチャ
# ==============================================================

@pytest.fixture
def simple_numeric_df() -> pd.DataFrame:
    """数値のみの小さな DataFrame。"""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "a": rng.normal(0, 1, 50),
        "b": rng.normal(5, 2, 50),
        "c": rng.exponential(1, 50),
    })


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """数値・カテゴリ・バイナリ混在 DataFrame。"""
    rng = np.random.default_rng(42)
    n = 80
    return pd.DataFrame({
        "num1": rng.normal(0, 1, n),
        "num2": rng.normal(5, 2, n),
        "cat_low": rng.choice(["A", "B", "C"], n),
        "binary": rng.choice([0, 1], n),
    })


@pytest.fixture
def regression_data():
    """sklearn の make_regression データセット。"""
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    return df, y


@pytest.fixture
def classification_data():
    """sklearn の make_classification データセット。"""
    X, y = make_classification(
        n_samples=100, n_features=6, n_informative=4, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    return df, y


# ==============================================================
# T-001〜T-005: ColumnSelectorWrapper
# ==============================================================

class TestColumnSelectorWrapper:

    def test_t001_all_mode(self, simple_numeric_df):
        """T-001: mode=all → 全列パススルー。"""
        sel = ColumnSelectorWrapper(mode="all")
        sel.fit(simple_numeric_df)
        out = sel.transform(simple_numeric_df)
        assert list(out.columns) == list(simple_numeric_df.columns)
        assert out.shape == simple_numeric_df.shape

    def test_t002_include_by_column_names(self, simple_numeric_df):
        """T-002: mode=include, columns指定 → 指定列のみ通過。"""
        sel = ColumnSelectorWrapper(mode="include", columns=["a", "b"])
        sel.fit(simple_numeric_df)
        out = sel.transform(simple_numeric_df)
        assert list(out.columns) == ["a", "b"]
        assert out.shape[1] == 2

    def test_t003_include_by_range(self, simple_numeric_df):
        """T-003: mode=include, col_range指定 → 範囲内列のみ通過。"""
        sel = ColumnSelectorWrapper(mode="include", col_range=(0, 2))
        sel.fit(simple_numeric_df)
        out = sel.transform(simple_numeric_df)
        assert out.shape[1] == 2

    def test_t004_exclude_mode(self, simple_numeric_df):
        """T-004: mode=exclude → 指定列を除外した残り列。"""
        sel = ColumnSelectorWrapper(mode="exclude", columns=["c"])
        sel.fit(simple_numeric_df)
        out = sel.transform(simple_numeric_df)
        assert "c" not in out.columns
        assert out.shape[1] == 2

    def test_t005_column_meta_monotonic(self, simple_numeric_df):
        """T-005: ColumnMeta の monotonic と group が正しく格納・取得できる。"""
        meta = {
            "a": ColumnMeta(monotonic=1, linearity="linear", group="g1"),
            "b": ColumnMeta(monotonic=-1, group="g1"),
            "c": ColumnMeta(monotonic=0, group=None),
        }
        sel = ColumnSelectorWrapper(mode="all", column_meta=meta)
        sel.fit(simple_numeric_df)
        constraints = sel.get_monotonic_constraints()
        assert constraints == (1, -1, 0)
        groups = sel.get_groups_array()
        assert groups == ["g1", "g1", None]

    def test_t005b_invalid_mode_raises(self, simple_numeric_df):
        """T-005b: 無効な mode は ValueError を送出。"""
        sel = ColumnSelectorWrapper(mode="invalid")
        with pytest.raises(ValueError, match="未知の mode"):
            sel.fit(simple_numeric_df)

    def test_t005c_include_nonexistent_column_warning(self, simple_numeric_df):
        """T-005c: include で存在しない列名はスキップされ警告が出る。"""
        sel = ColumnSelectorWrapper(mode="include", columns=["a", "NONEXISTENT"])
        sel.fit(simple_numeric_df)
        out = sel.transform(simple_numeric_df)
        assert list(out.columns) == ["a"]


# ==============================================================
# T-006〜T-011: ColPreprocessor
# ==============================================================

class TestColPreprocessor:

    def test_t006_numeric_standard_scaler(self, simple_numeric_df):
        """T-006: 数値列に StandardScaler を適用。"""
        cfg = ColPreprocessConfig(numeric_scaler="standard")
        pp = ColPreprocessor(config=cfg)
        pp.fit(simple_numeric_df)
        out = pp.transform(simple_numeric_df)
        assert out.shape[0] == len(simple_numeric_df)
        assert out.shape[1] >= 3

    @pytest.mark.parametrize("scaler", [
        "minmax", "robust", "maxabs",
        "power_yj", "quantile_normal", "quantile_uniform",
        "log", "none"
    ])
    def test_t007_all_scalers(self, simple_numeric_df, scaler):
        """T-007: 全スケーラーで fit/transform が正常終了。"""
        # power_bc は全正値が必要なのでスキップ
        if scaler == "power_bc":
            pytest.skip("power_bc は全正値が必要")
        cfg = ColPreprocessConfig(numeric_scaler=scaler)
        pp = ColPreprocessor(config=cfg)
        pp.fit(simple_numeric_df)
        out = pp.transform(simple_numeric_df)
        assert out.shape[0] == len(simple_numeric_df)

    def test_t008_mixed_data(self, mixed_df):
        """T-008: 数値・カテゴリ・バイナリ混在 DataFrame で動作。"""
        cfg = ColPreprocessConfig(cat_low_encoder="onehot")
        pp = ColPreprocessor(config=cfg)
        pp.fit(mixed_df)
        out = pp.transform(mixed_df)
        # onehot で列が増える
        assert out.shape[0] == len(mixed_df)
        assert out.shape[1] > mixed_df.shape[1]

    def test_t009_override_types(self, mixed_df):
        """T-009: override_types で自動判定を上書き（カテゴリ→passthrough）。"""
        cfg = ColPreprocessConfig(
            override_types={"cat_low": "passthrough"},
        )
        pp = ColPreprocessor(config=cfg)
        pp.fit(mixed_df)
        out = pp.transform(mixed_df)
        assert out is not None
        assert out.shape[0] == len(mixed_df)

    def test_t010_knn_imputer(self, simple_numeric_df):
        """T-010: KNN Imputer で欠損補間。"""
        df = simple_numeric_df.copy()
        df.loc[0, "a"] = np.nan
        cfg = ColPreprocessConfig(numeric_imputer="knn")
        pp = ColPreprocessor(config=cfg)
        pp.fit(df)
        out = pp.transform(df)
        assert not np.isnan(out).any()

    def test_t011_get_feature_names_out(self, mixed_df):
        """T-011: get_feature_names_out が動作する。"""
        pp = ColPreprocessor()
        pp.fit(mixed_df)
        names = pp.get_feature_names_out()
        assert len(names) > 0


# ==============================================================
# T-012〜T-015: FeatureGenerator
# ==============================================================

class TestFeatureGenerator:

    def test_t012_none_passthrough(self, regression_data):
        """T-012: method=none → パススルー（入力と同一形状）。"""
        df, _ = regression_data
        X = df.values
        fg = FeatureGenerator(FeatureGenConfig(method="none"))
        fg.fit(X)
        out = fg.transform(X)
        np.testing.assert_array_equal(out, X)

    def test_t013_polynomial(self, regression_data):
        """T-013: method=polynomial, degree=2 → 特徴量が増加。"""
        df, _ = regression_data
        X = df.values
        fg = FeatureGenerator(FeatureGenConfig(method="polynomial", degree=2))
        fg.fit(X)
        out = fg.transform(X)
        assert out.shape[0] == X.shape[0]
        assert out.shape[1] > X.shape[1]

    def test_t014_interaction_only(self, regression_data):
        """T-014: method=interaction_only → 交互作用項のみ（自乗項なし）。"""
        df, _ = regression_data
        X = df.values
        cfg = FeatureGenConfig(method="interaction_only", degree=2)
        fg = FeatureGenerator(cfg)
        fg.fit(X)
        out = fg.transform(X)
        assert out.shape[0] == X.shape[0]
        assert out.shape[1] > X.shape[1]

    def test_t015_feature_names_out(self, regression_data):
        """T-015: get_feature_names_out が文字列配列を返す。"""
        df, _ = regression_data
        fg = FeatureGenerator(FeatureGenConfig(method="polynomial", degree=2))
        fg.fit(df)
        names = fg.get_feature_names_out()
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)


# ==============================================================
# T-016〜T-023: FeatureSelector
# ==============================================================

class TestFeatureSelector:

    def test_t016_none_passthrough(self, regression_data):
        """T-016: method=none → パススルー。"""
        df, y = regression_data
        fs = FeatureSelector(FeatureSelectorConfig(method="none"))
        fs.fit(df, y)
        out = fs.transform(df)
        np.testing.assert_array_equal(out, df.values)

    def test_t017_lasso_regression(self, regression_data):
        """T-017: method=lasso, task=regression → 特徴量を削減。"""
        df, y = regression_data
        cfg = FeatureSelectorConfig(method="lasso", task="regression")
        fs = FeatureSelector(cfg)
        fs.fit(df.values, y)
        out = fs.transform(df.values)
        assert out.shape[0] == df.shape[0]
        assert out.shape[1] <= df.shape[1]

    def test_t018_ridge(self, regression_data):
        """T-018: method=ridge, task=regression → SelectFromModel(Ridge)。"""
        df, y = regression_data
        cfg = FeatureSelectorConfig(method="ridge", task="regression")
        fs = FeatureSelector(cfg)
        fs.fit(df.values, y)
        out = fs.transform(df.values)
        assert out.shape[0] == df.shape[0]

    def test_t019_rfr_regression(self, regression_data):
        """T-019: method=rfr, task=regression → RandomForest 重要度選択。"""
        df, y = regression_data
        cfg = FeatureSelectorConfig(method="rfr", task="regression", threshold="mean")
        fs = FeatureSelector(cfg)
        fs.fit(df.values, y)
        out = fs.transform(df.values)
        assert out.shape[0] == df.shape[0]

    def test_t020_rfc_classification(self, classification_data):
        """T-020: method=rfr（分類タスク）→ RandomForestClassifier を使用。"""
        df, y = classification_data
        cfg = FeatureSelectorConfig(method="rfr", task="classification")
        fs = FeatureSelector(cfg)
        fs.fit(df.values, y)
        out = fs.transform(df.values)
        assert out.shape[0] == df.shape[0]

    def test_t021_select_percentile(self, regression_data):
        """T-021: method=select_percentile → 上位 50% を選択。"""
        df, y = regression_data
        cfg = FeatureSelectorConfig(method="select_percentile", percentile=50)
        fs = FeatureSelector(cfg)
        fs.fit(df.values, y)
        out = fs.transform(df.values)
        assert out.shape[1] <= df.shape[1]

    def test_t022_select_kbest(self, regression_data):
        """T-022: method=select_kbest, k=3 → 上位 3 特徴量。"""
        df, y = regression_data
        cfg = FeatureSelectorConfig(method="select_kbest", k=3)
        fs = FeatureSelector(cfg)
        fs.fit(df.values, y)
        out = fs.transform(df.values)
        assert out.shape[1] == 3

    def test_t023_get_feature_names_out(self, regression_data):
        """T-023: get_feature_names_out が選択列名を返す。"""
        df, y = regression_data
        cfg = FeatureSelectorConfig(method="select_kbest", k=3)
        fs = FeatureSelector(cfg)
        fs.fit(df, y)
        names = fs.get_feature_names_out()
        assert len(names) == 3


# ==============================================================
# T-024〜T-028: build_pipeline（end-to-end）
# ==============================================================

class TestBuildPipeline:

    def test_t024_regression_default(self, regression_data):
        """T-024: デフォルト設定の回帰パイプラインで fit/predict。"""
        df, y = regression_data
        config = PipelineConfig(task="regression", estimator_key="rf")
        pipe = build_pipeline(config)
        pipe.fit(df, y)
        preds = pipe.predict(df)
        assert preds.shape == (len(y),)

    def test_t025_classification_default(self, classification_data):
        """T-025: デフォルト設定の分類パイプラインで fit/predict。"""
        df, y = classification_data
        config = PipelineConfig(task="classification", estimator_key="rf_c")
        pipe = build_pipeline(config)
        pipe.fit(df, y)
        preds = pipe.predict(df)
        assert preds.shape == (len(y),)

    def test_t026_with_feature_gen_and_selection(self, regression_data):
        """T-026: 特徴量生成（polynomial）+ 選択（select_kbest）の組み合わせ。"""
        df, y = regression_data
        config = PipelineConfig(
            task="regression",
            estimator_key="ridge",
            feature_gen_config=FeatureGenConfig(method="polynomial", degree=2),
            feature_sel_config=FeatureSelectorConfig(method="select_kbest", k=10),
        )
        pipe = build_pipeline(config)
        pipe.fit(df, y)
        preds = pipe.predict(df)
        assert preds.shape == (len(y),)

    def test_t027_exclude_columns(self, regression_data):
        """T-027: 入力列制御（exclude モード）でパイプラインが動作。"""
        df, y = regression_data
        df["noise"] = np.random.randn(len(df))
        config = PipelineConfig(
            col_select_mode="exclude",
            col_select_columns=["noise"],
            task="regression",
            estimator_key="ridge",
        )
        pipe = build_pipeline(config)
        pipe.fit(df, y)
        preds = pipe.predict(df)
        assert preds.shape == (len(y),)

    def test_t028_xgb_pipeline(self, regression_data):
        """T-028: XGBoost estimator でパイプラインが動作（XGBoost 利用可能時）。"""
        pytest.importorskip("xgboost")
        df, y = regression_data
        config = PipelineConfig(task="regression", estimator_key="xgb")
        pipe = build_pipeline(config)
        pipe.fit(df, y)
        preds = pipe.predict(df)
        assert preds.shape == (len(y),)


# ==============================================================
# T-029〜T-031: apply_monotonic_constraints / extract_group_array
# ==============================================================

class TestMonotonicAndGroups:

    def test_t029_apply_monotonic_non_support_no_error(self):
        """T-029: monotonic非対応estimatorでも例外なし（警告のみ）。"""
        estimator = Ridge()
        meta = {"f0": ColumnMeta(monotonic=1), "f1": ColumnMeta(monotonic=-1)}
        result = apply_monotonic_constraints(estimator, meta)
        # Ridge はそのまま返るか、MonotonicConstraintRegressorでラップされる
        assert "MonotonicConstraintRegressor" in type(result).__name__ or type(result).__name__ == "Ridge"

    def test_t030_apply_monotonic_xgb(self):
        """T-030: XGBoost estimatorで monotonic_constraints が設定される。"""
        pytest.importorskip("xgboost")
        from xgboost import XGBRegressor
        estimator = XGBRegressor()
        meta = {
            "f0": ColumnMeta(monotonic=1),
            "f1": ColumnMeta(monotonic=0),
            "f2": ColumnMeta(monotonic=-1),
        }
        result = apply_monotonic_constraints(estimator, meta)
        # get_params()でmonoton系キーに値が設定されていることを確認
        params = result.get_params()
        set_keys = [
            k for k in params
            if "monoton" in k.lower() and params[k] is not None and params[k] != 0
        ]
        assert len(set_keys) > 0, f"monotonic系パラメータが設定されていない。params={params}"

    def test_t031_extract_group_array(self):
        """T-031: ColumnMeta から整数グループ配列が正しく生成される。"""
        meta = {
            "f0": ColumnMeta(group="A"),
            "f1": ColumnMeta(group="A"),
            "f2": ColumnMeta(group="B"),
            "f3": ColumnMeta(group=None),
        }
        feature_names = ["f0", "f1", "f2", "f3"]
        arr = extract_group_array(meta, feature_names)
        assert arr is not None
        assert arr.shape == (4,)
        # f0, f1 は同グループ
        assert arr[0] == arr[1]
        # f2 は異グループ
        assert arr[2] != arr[0]
        # f3 は -1（グループなし）
        assert arr[3] == -1

    def test_t031b_all_none_groups_returns_none(self):
        """T-031b: 全列 group=None → None を返す。"""
        meta = {"f0": ColumnMeta(), "f1": ColumnMeta()}
        arr = extract_group_array(meta, ["f0", "f1"])
        assert arr is None
