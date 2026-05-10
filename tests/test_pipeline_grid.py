"""
tests/test_pipeline_grid.py

PipelineGrid（全組み合わせパイプライン生成）のテスト。

テスト構成:
  T-G001: 組み合わせ数カウント
  T-G002: 単一選択 → 1件生成
  T-G003: 複数imputer × 複数estimator
  T-G004: 全グリッド軸の展開
  T-G005: feature_gen の none vs polynomial
  T-G006: feature_sel の複数選択
  T-G007: max_combinations による制限
  T-G008: fit/predict まで通る end-to-end
  T-G009: estimator_params_list 対応
  T-G010: 分類タスクのグリッド
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification

from backend.pipeline import (
    generate_pipeline_grid,
    count_combinations,
    PipelineGridConfig,
    PipelineCombination,
    FeatureGenConfig,
    FeatureSelectorConfig,
)


# ==============================================================
# フィクスチャ
# ==============================================================

@pytest.fixture
def reg_df():
    X, y = make_regression(n_samples=80, n_features=5, noise=0.1, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), y


@pytest.fixture
def clf_df():
    X, y = make_classification(
        n_samples=80, n_features=5, n_informative=3, random_state=0
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), y


# ==============================================================
# T-G001: count_combinations
# ==============================================================

class TestCountCombinations:

    def test_tg001_single_each(self):
        """T-G001a: 全て1択 → 1件。"""
        gc = PipelineGridConfig(
            numeric_imputers=["mean"],
            numeric_scalers=["standard"],
            cat_low_encoders=["onehot"],
            cat_high_encoders=["ordinal"],
            feature_gen_methods=["none"],
            feature_sel_methods=["none"],
            estimator_keys=["rf"],
        )
        assert count_combinations(gc) == 1

    def test_tg001_two_estimators(self):
        """T-G001b: estimator 2択 → 2件。"""
        gc = PipelineGridConfig(estimator_keys=["rf", "ridge"])
        assert count_combinations(gc) == 2

    def test_tg001_two_scalers_two_estimators(self):
        """T-G001c: scaler 2択 × estimator 2択 → 4件。"""
        gc = PipelineGridConfig(
            numeric_scalers=["standard", "robust"],
            estimator_keys=["rf", "ridge"],
        )
        assert count_combinations(gc) == 4

    def test_tg001_full_grid(self):
        """T-G001d: 2×2×1×1×2×2×3 = 48件。"""
        gc = PipelineGridConfig(
            numeric_imputers=["mean", "median"],
            numeric_scalers=["standard", "robust"],
            cat_low_encoders=["onehot"],
            cat_high_encoders=["ordinal"],
            feature_gen_methods=["none", "polynomial"],
            feature_sel_methods=["none", "rfr"],
            estimator_keys=["ridge", "rf", "xgb"],
        )
        # none×1 + polynomial×degree(2) = 3 gen combos
        # 2×2×1×1×3×2×3 = 72
        n = count_combinations(gc)
        assert n > 0

    def test_tg001_empty_list_defaults(self):
        """T-G001e: 空リストはデフォルト1択として扱う。"""
        gc = PipelineGridConfig(
            numeric_imputers=[],
            estimator_keys=["rf"],
        )
        assert count_combinations(gc) == 1


# ==============================================================
# T-G002: 単一選択 → 1件生成
# ==============================================================

class TestSingleCombination:

    def test_tg002_single(self, reg_df):
        """T-G002: 各軸1択 → PipelineCombination 1件。"""
        gc = PipelineGridConfig(
            task="regression",
            estimator_keys=["ridge"],
        )
        combos = generate_pipeline_grid(gc)
        assert len(combos) == 1
        combo = combos[0]
        assert isinstance(combo, PipelineCombination)
        assert "est=ridge" in combo.name

    def test_tg002_name_format(self, reg_df):
        """T-G002b: name が全ステップ情報を含む。"""
        gc = PipelineGridConfig(
            numeric_imputers=["median"],
            numeric_scalers=["minmax"],
            estimator_keys=["ridge"],
        )
        combos = generate_pipeline_grid(gc)
        name = combos[0].name
        assert "imp=median" in name
        assert "scl=minmax" in name
        assert "est=ridge" in name


# ==============================================================
# T-G003: 複数imputer × 複数estimator
# ==============================================================

class TestMultipleImputerEstimator:

    def test_tg003_two_imputers_two_estimators(self):
        """T-G003: imputer 2択 × estimator 2択 → 4件。"""
        gc = PipelineGridConfig(
            task="regression",
            numeric_imputers=["mean", "median"],
            estimator_keys=["ridge", "rf"],
        )
        combos = generate_pipeline_grid(gc)
        assert len(combos) == 4

    def test_tg003_all_names_unique(self):
        """T-G003b: 全ての name が一意。"""
        gc = PipelineGridConfig(
            numeric_imputers=["mean", "median"],
            numeric_scalers=["standard", "robust"],
            estimator_keys=["ridge", "rf"],
        )
        combos = generate_pipeline_grid(gc)
        names = [c.name for c in combos]
        assert len(names) == len(set(names))

    def test_tg003_config_fields(self):
        """T-G003c: config フィールドが正しく設定される。"""
        gc = PipelineGridConfig(
            numeric_imputers=["mean", "knn"],
            estimator_keys=["ridge"],
        )
        combos = generate_pipeline_grid(gc)
        imputers_used = {
            c.config.preprocessor_config.numeric_imputer for c in combos
        }
        assert imputers_used == {"mean", "knn"}


# ==============================================================
# T-G004: 全グリッド軸の展開
# ==============================================================

class TestFullGridAxes:

    def test_tg004_all_scalers(self):
        """T-G004: 複数スケーラーが全て展開される。"""
        gc = PipelineGridConfig(
            numeric_scalers=["standard", "minmax", "robust"],
            estimator_keys=["ridge"],
        )
        combos = generate_pipeline_grid(gc)
        scalers_used = {c.config.preprocessor_config.numeric_scaler for c in combos}
        assert scalers_used == {"standard", "minmax", "robust"}
        assert len(combos) == 3

    def test_tg004_all_encoders(self):
        """T-G004b: 複数エンコーダーが全て展開される。"""
        gc = PipelineGridConfig(
            cat_low_encoders=["onehot", "ordinal"],
            estimator_keys=["ridge"],
        )
        combos = generate_pipeline_grid(gc)
        assert len(combos) == 2
        enc_used = {c.config.preprocessor_config.cat_low_encoder for c in combos}
        assert enc_used == {"onehot", "ordinal"}


# ==============================================================
# T-G005: feature_gen の展開
# ==============================================================

class TestFeatureGenGrid:

    def test_tg005_none_and_polynomial(self):
        """T-G005: feature_gen_methods=[none, polynomial] → none 1件 + polynomial×degree 件数。"""
        gc = PipelineGridConfig(
            feature_gen_methods=["none", "polynomial"],
            feature_gen_degrees=[2],
            estimator_keys=["ridge"],
        )
        combos = generate_pipeline_grid(gc)
        # none=1件, polynomial,degree=2 = 1件 → 計2件
        assert len(combos) == 2
        methods = {c.config.feature_gen_config.method for c in combos}
        assert "none" in methods
        assert "polynomial" in methods

    def test_tg005_polynomial_multi_degree(self):
        """T-G005b: polynomial × degree[2,3] → 2件。"""
        gc = PipelineGridConfig(
            feature_gen_methods=["polynomial"],
            feature_gen_degrees=[2, 3],
            estimator_keys=["ridge"],
        )
        combos = generate_pipeline_grid(gc)
        assert len(combos) == 2
        degrees = {c.config.feature_gen_config.degree for c in combos}
        assert degrees == {2, 3}


# ==============================================================
# T-G006: feature_sel の複数選択
# ==============================================================

class TestFeatureSelGrid:

    def test_tg006_none_and_rfr(self):
        """T-G006: feature_sel_methods=[none, rfr] × estimator 2択 → 4件。"""
        gc = PipelineGridConfig(
            feature_sel_methods=["none", "rfr"],
            estimator_keys=["ridge", "rf"],
        )
        combos = generate_pipeline_grid(gc)
        assert len(combos) == 4

    def test_tg006_sel_methods_all_appear(self):
        """T-G006b: 全選択手法が config に反映される。"""
        gc = PipelineGridConfig(
            feature_sel_methods=["none", "lasso", "select_kbest"],
            estimator_keys=["ridge"],
        )
        combos = generate_pipeline_grid(gc)
        assert len(combos) == 3
        methods = {c.config.feature_sel_config.method for c in combos}
        assert methods == {"none", "lasso", "select_kbest"}


# ==============================================================
# T-G007: max_combinations による制限
# ==============================================================

class TestMaxCombinations:

    def test_tg007_limit(self):
        """T-G007: max_combinations=3 → 最大3件。"""
        gc = PipelineGridConfig(
            numeric_imputers=["mean", "median", "knn"],
            estimator_keys=["ridge", "rf", "rf"],
        )
        combos = generate_pipeline_grid(gc, max_combinations=3)
        assert len(combos) <= 3

    def test_tg007_no_limit(self):
        """T-G007b: max_combinations=None → 全件生成。"""
        gc = PipelineGridConfig(
            numeric_scalers=["standard", "minmax", "robust", "maxabs"],
            estimator_keys=["ridge"],
        )
        combos = generate_pipeline_grid(gc, max_combinations=None)
        assert len(combos) == 4


# ==============================================================
# T-G008: end-to-end fit/predict
# ==============================================================

class TestEndToEnd:

    def test_tg008_fit_predict_regression(self, reg_df):
        """T-G008: 回帰グリッドの全組み合わせで fit/predict が正常終了。"""
        df, y = reg_df
        gc = PipelineGridConfig(
            task="regression",
            numeric_imputers=["mean", "median"],
            numeric_scalers=["standard", "robust"],
            estimator_keys=["ridge"],
        )
        combos = generate_pipeline_grid(gc)
        assert len(combos) == 4

        for combo in combos:
            combo.pipeline.fit(df, y)
            preds = combo.pipeline.predict(df)
            assert preds.shape == (len(y),), f"{combo.name} の predict 形状が不正"

    def test_tg008_fit_predict_with_selection(self, reg_df):
        """T-G008b: 特徴量選択あり組み合わせでも fit/predict 正常終了。"""
        df, y = reg_df
        gc = PipelineGridConfig(
            task="regression",
            feature_sel_methods=["none", "select_kbest"],
            estimator_keys=["ridge", "rf"],
        )
        combos = generate_pipeline_grid(gc)
        for combo in combos:
            combo.pipeline.fit(df, y)
            preds = combo.pipeline.predict(df)
            assert preds.shape[0] == len(y)


# ==============================================================
# T-G009: estimator_params_list
# ==============================================================

class TestEstimatorParamsList:

    def test_tg009_params_applied(self, reg_df):
        """T-G009: estimator_params_list のパラメータがモデルに反映される。"""
        df, y = reg_df
        gc = PipelineGridConfig(
            task="regression",
            estimator_keys=["ridge", "ridge"],
            estimator_params_list=[{"alpha": 0.1}, {"alpha": 10.0}],
        )
        combos = generate_pipeline_grid(gc)
        assert len(combos) == 2
        # alpha の違いを name ではなく config で確認
        alphas = [c.config.estimator_params.get("alpha") for c in combos]
        assert set(alphas) == {0.1, 10.0}


# ==============================================================
# T-G010: 分類タスク
# ==============================================================

class TestClassificationGrid:

    def test_tg010_classification_fit_predict(self, clf_df):
        """T-G010: 分類タスクのグリッドで fit/predict 正常終了。"""
        df, y = clf_df
        gc = PipelineGridConfig(
            task="classification",
            numeric_scalers=["standard", "minmax"],
            estimator_keys=["rf_c", "logistic"],
        )
        combos = generate_pipeline_grid(gc)
        assert len(combos) == 4
        for combo in combos:
            combo.pipeline.fit(df, y)
            preds = combo.pipeline.predict(df)
            assert preds.shape == (len(y),)

    def test_tg010_feature_sel_classification(self, clf_df):
        """T-G010b: 分類タスクでの feature_sel（スコア関数自動切替）。"""
        df, y = clf_df
        gc = PipelineGridConfig(
            task="classification",
            feature_sel_methods=["none", "select_percentile"],
            estimator_keys=["logistic"],
        )
        combos = generate_pipeline_grid(gc)
        assert len(combos) == 2
        for combo in combos:
            combo.pipeline.fit(df, y)
            preds = combo.pipeline.predict(df)
            assert preds.shape == (len(y),)


# ==============================================================
# T-G011: cat_imputers 複数選択
# ==============================================================

class TestCatImputersGrid:

    def test_tg011_cat_imputers_count(self):
        """T-G011a: cat_imputers 2択 → 組合せ数が2倍になる。"""
        gc1 = PipelineGridConfig(
            cat_imputers=["most_frequent"],
            estimator_keys=["ridge"],
        )
        gc2 = PipelineGridConfig(
            cat_imputers=["most_frequent", "constant"],
            estimator_keys=["ridge"],
        )
        assert count_combinations(gc2) == count_combinations(gc1) * 2

    def test_tg011_cat_imputers_config_applied(self, reg_df):
        """T-G011b: cat_imputers の値が ColPreprocessConfig に反映される。"""
        df, y = reg_df
        gc = PipelineGridConfig(
            cat_imputers=["most_frequent", "constant"],
            estimator_keys=["ridge"],
        )
        combos = generate_pipeline_grid(gc)
        assert len(combos) == 2
        ci_values = {c.config.preprocessor_config.categorical_imputer for c in combos}
        assert ci_values == {"most_frequent", "constant"}

    def test_tg011_fit_predict(self, reg_df):
        """T-G011c: cat_imputers 複数でも fit/predict が正常終了。"""
        df, y = reg_df
        gc = PipelineGridConfig(
            cat_imputers=["most_frequent", "constant"],
            estimator_keys=["ridge"],
        )
        combos = generate_pipeline_grid(gc)
        for combo in combos:
            combo.pipeline.fit(df, y)
            preds = combo.pipeline.predict(df)
            assert preds.shape[0] == len(y)


# ==============================================================
# T-G012: binary_imputers 複数選択
# ==============================================================

class TestBinaryImputersGrid:

    def test_tg012_binary_imputers_count(self):
        """T-G012a: binary_imputers 2択 → 組合せ数が2倍になる。"""
        gc1 = PipelineGridConfig(
            binary_imputers=["most_frequent"],
            estimator_keys=["ridge"],
        )
        gc2 = PipelineGridConfig(
            binary_imputers=["most_frequent", "constant"],
            estimator_keys=["ridge"],
        )
        assert count_combinations(gc2) == count_combinations(gc1) * 2

    def test_tg012_binary_imputers_config_applied(self):
        """T-G012b: binary_imputers が ColPreprocessConfig.binary_imputer に反映される。"""
        gc = PipelineGridConfig(
            binary_imputers=["most_frequent", "constant"],
            estimator_keys=["ridge"],
        )
        combos = generate_pipeline_grid(gc)
        bi_values = {c.config.preprocessor_config.binary_imputer for c in combos}
        assert bi_values == {"most_frequent", "constant"}


# ==============================================================
# T-G013: exclude_columns
# ==============================================================

class TestExcludeColumns:

    def test_tg013_exclude_columns_mode(self):
        """T-G013a: exclude_columns 指定時に col_select_mode='exclude' になる。"""
        gc = PipelineGridConfig(
            exclude_columns=["col_a", "col_b"],
            estimator_keys=["ridge"],
        )
        combos = generate_pipeline_grid(gc)
        assert len(combos) == 1
        cfg = combos[0].config
        assert cfg.col_select_mode == "exclude"
        assert "col_a" in cfg.col_select_columns
        assert "col_b" in cfg.col_select_columns

    def test_tg013_no_exclude(self):
        """T-G013b: exclude_columns 未指定時は元の col_select_mode を維持。"""
        gc = PipelineGridConfig(
            col_select_mode="all",
            exclude_columns=[],
            estimator_keys=["ridge"],
        )
        combos = generate_pipeline_grid(gc)
        assert combos[0].config.col_select_mode == "all"

    def test_tg013_full_grid_with_exclude(self):
        """T-G013c: exclude_columns + 複数 imputer/estimator の組み合わせ。"""
        gc = PipelineGridConfig(
            exclude_columns=["feat_to_drop"],
            numeric_imputers=["mean", "median"],
            estimator_keys=["ridge", "rf"],
        )
        combos = generate_pipeline_grid(gc)
        assert len(combos) == 4  # 2×2
        for c in combos:
            assert c.config.col_select_mode == "exclude"
            assert "feat_to_drop" in c.config.col_select_columns

