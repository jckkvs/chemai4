"""
tests/test_integration.py

エンド・ツー・エンド統合テスト。
データ読み込み → 型判定 → 前処理 → モデル学習 → 評価のフルパイプライン。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from backend.data.loader import load_from_bytes
from backend.data.type_detector import TypeDetector
from backend.data.preprocessor import Preprocessor, PreprocessConfig, build_full_pipeline
from backend.data.eda import compute_column_stats, summarize_dataframe, detect_outliers
from backend.data.dim_reduction import run_pca
from backend.data.benchmark import evaluate_regression, evaluate_classification


# ============================================================
# 簡易前処理ヘルパー (ColumnTransformerではなくDetect→build→fit)
# ============================================================

def _build_ct(df: pd.DataFrame, target_col: str | None = None):
    """TypeDetector + Preprocessor.build() で ColumnTransformer を返す。"""
    detector = TypeDetector()
    result = detector.detect(df)
    preprocessor = Preprocessor()
    ct = preprocessor.build(result, target_col=target_col)
    return ct


# ============================================================
# フィクスチャ
# ============================================================

@pytest.fixture
def reg_df() -> pd.DataFrame:
    """回帰タスク用サンプルDataFrame。"""
    np.random.seed(99)
    n = 200
    return pd.DataFrame({
        "feat_a": np.random.randn(n),
        "feat_b": np.random.uniform(0, 10, n),
        "feat_c": np.random.choice(["x", "y", "z"], n),
        "target": np.random.randn(n),
    })


@pytest.fixture
def cls_df() -> pd.DataFrame:
    """分類タスク用サンプルDataFrame。"""
    np.random.seed(0)
    n = 150
    return pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.choice(["cat", "dog"], n),
        "label": np.random.choice([0, 1], n),
    })


# ============================================================
# T-INT-001: データロード → 型判定
# ============================================================

class TestDataPipeline:
    """T-INT-001: データロード→型判定の統合テスト。"""

    def test_load_csv_from_bytes(self, reg_df: pd.DataFrame) -> None:
        """CSVバイト→DataFrameに正しくロードできること。(T-INT-001-01)"""
        csv_bytes = reg_df.to_csv(index=False).encode()
        loaded = load_from_bytes(csv_bytes, filename="test.csv")
        assert loaded is not None
        assert len(loaded) == 200
        assert "feat_a" in loaded.columns

    def test_type_detection(self, reg_df: pd.DataFrame) -> None:
        """TypeDetectorが全列を正しく検出すること。(T-INT-001-02)"""
        detector = TypeDetector()
        result = detector.detect(reg_df)
        assert len(result.column_info) == 4

    def test_column_transformer_builds(self, reg_df: pd.DataFrame) -> None:
        """ColumnTransformerが正常に構築できること。(T-INT-001-03)"""
        ct = _build_ct(reg_df, target_col="target")
        assert ct is not None

    def test_preprocess_produces_numpy(self, reg_df: pd.DataFrame) -> None:
        """ColumnTransformer.fit_transformがnumpy配列を返すこと。(T-INT-001-04)"""
        X = reg_df.drop(columns=["target"])
        ct = _build_ct(reg_df, target_col="target")
        X_proc = ct.fit_transform(X)
        assert isinstance(X_proc, np.ndarray)
        assert X_proc.shape[0] == 200
        assert X_proc.shape[1] > 0

    def test_no_data_leakage(self, reg_df: pd.DataFrame) -> None:
        """訓練・テストで列数が一致すること（データリーク検証）。(T-INT-001-05)"""
        train = reg_df.iloc[:160]
        test = reg_df.iloc[160:]
        ct = _build_ct(train, target_col="target")
        X_train = ct.fit_transform(train.drop(columns=["target"]))
        X_test = ct.transform(test.drop(columns=["target"]))
        assert X_train.shape[1] == X_test.shape[1]


# ============================================================
# T-INT-002: EDA → 次元削減
# ============================================================

class TestEDAPipeline:
    """T-INT-002: EDA→次元削減の統合テスト。"""

    def test_eda_summary(self, reg_df: pd.DataFrame) -> None:
        """summarize_dataframeがn_rows/n_colsキーを返すこと。(T-INT-002-01)"""
        summary = summarize_dataframe(reg_df)
        assert "n_rows" in summary
        assert summary["n_rows"] == 200

    def test_column_stats_returns_list(self, reg_df: pd.DataFrame) -> None:
        """compute_column_statsがColumnStatsのリストを返すこと。(T-INT-002-02)"""
        stats = compute_column_stats(reg_df)
        assert isinstance(stats, list)
        assert len(stats) == 4
        assert any(s.name == "feat_a" for s in stats)

    def test_pca_on_numeric(self, reg_df: pd.DataFrame) -> None:
        """数値列のみのDataFrameにPCAが適用できること。(T-INT-002-03)"""
        X = reg_df.select_dtypes(include="number").drop(columns=["target"])
        emb_df, evr = run_pca(X, n_components=2)
        assert emb_df.shape == (200, 2)
        assert len(evr) == 2
        assert evr.sum() <= 1.0 + 1e-9

    def test_outlier_detection(self, reg_df: pd.DataFrame) -> None:
        """detect_outliersがIQR法でOutlierResultリストを返すこと。(T-INT-002-04)"""
        results = detect_outliers(reg_df, method="iqr", cols=["feat_a"])
        assert isinstance(results, list)
        assert len(results) == 1
        assert hasattr(results[0], "n_outliers")


# ============================================================
# T-INT-003: 前処理 → モデル学習 → 評価
# ============================================================

class TestModelTrainingPipeline:
    """T-INT-003: 前処理→モデル学習→評価の統合テスト。"""

    def test_regression_end_to_end(self, reg_df: pd.DataFrame) -> None:
        """回帰パイプライン全体が正常終了してR²≥-1になること。(T-INT-003-01)"""
        train = reg_df.iloc[:160]
        test = reg_df.iloc[160:]
        # build_full_pipeline でパイプライン構築
        detector = TypeDetector()
        dr = detector.detect(train)
        pipeline = build_full_pipeline(dr, LinearRegression(), target_col="target")

        X_train = train.drop(columns=["target"])
        y_train = train["target"].values
        X_test = test.drop(columns=["target"])
        y_test = test["target"].values

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        score = evaluate_regression(y_test, y_pred, model_key="lr")
        assert score.r2 is not None
        assert score.r2 >= -1
        assert score.rmse is not None and score.rmse >= 0

    def test_classification_end_to_end(self, cls_df: pd.DataFrame) -> None:
        """分類パイプライン全体が正常終了してAccuracy∈[0,1]になること。(T-INT-003-02)"""
        train = cls_df.iloc[:120]
        test = cls_df.iloc[120:]
        detector = TypeDetector()
        dr = detector.detect(train)
        pipeline = build_full_pipeline(
            dr, DecisionTreeClassifier(random_state=42), target_col="label"
        )
        X_train = train.drop(columns=["label"])
        y_train = train["label"].values
        X_test = test.drop(columns=["label"])
        y_test = test["label"].values

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, 'predict_proba') else None

        score = evaluate_classification(y_test, y_pred, y_prob=y_prob, model_key="lr")
        assert score.accuracy is not None
        assert 0.0 <= score.accuracy <= 1.0

    def test_train_test_column_consistency(self, reg_df: pd.DataFrame) -> None:
        """訓練・テスト前処理後の列数が一致すること。(T-INT-003-03)"""
        train = reg_df.iloc[:160]
        test = reg_df.iloc[160:]
        ct = _build_ct(train, target_col="target")
        X_train = ct.fit_transform(train.drop(columns=["target"]))
        X_test = ct.transform(test.drop(columns=["target"]))
        assert X_train.shape[1] == X_test.shape[1]
