# -*- coding: utf-8 -*-
"""
tests/test_random_projection.py

JLRandomProjection, should_apply_random_projection, build_full_pipeline(JL-RP),
AutoMLEngine.run_multi_feature_sets の包括テスト。

テストID対応:
    T-RP-001: JLRandomProjection — 低次元入力で passthrough
    T-RP-002: JLRandomProjection — 高次元入力で射影適用
    T-RP-003: JLRandomProjection — summary() 出力
    T-RP-004: JLRandomProjection — get_feature_names_out
    T-RP-005: JLRandomProjection — method 自動選択 (auto)
    T-RP-006: JLRandomProjection — method 固定 (sparse / gaussian)
    T-RP-007: should_apply_random_projection ユーティリティ
    T-RP-008: build_full_pipeline — JL-RP ステップ挿入
    T-RP-009: build_full_pipeline — JL-RP ステップなし（デフォルト）
    T-RP-010: PreprocessConfig — random_projection フィールド
    T-RP-011: run_multi_feature_sets — normal / highdim 混在
    T-RP-012: JLRandomProjection — transform前にfit必須 (NotFittedError)
    T-RP-013: JLRandomProjection — sparse行列入力対応
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.utils.validation import check_is_fitted

from backend.data.random_projection import (
    JLRandomProjection,
    should_apply_random_projection,
)
from backend.data.preprocessor import (
    PreprocessConfig,
    build_full_pipeline,
)
from backend.data.type_detector import TypeDetector


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def low_dim_X() -> np.ndarray:
    """JL条件を満たさない低次元データ (50サンプル×5特徴量)。"""
    np.random.seed(42)
    return np.random.randn(50, 5)


@pytest.fixture
def high_dim_X() -> np.ndarray:
    """JL条件を満たす高次元データ (100サンプル×2000特徴量)。
    eps=0.5で jl_min_dim ≈ 22 なので 2000 >> 22 → 射影適用。
    """
    np.random.seed(42)
    return np.random.randn(100, 2000)


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """20列の数値特徴量を持つ回帰データ。"""
    np.random.seed(42)
    n = 80
    data = {f"feat_{i}": np.random.randn(n) for i in range(20)}
    data["target"] = np.random.randn(n) * 3 + 1
    return pd.DataFrame(data)


# ============================================================
# T-RP-001: JLRandomProjection — 低次元入力で passthrough
# ============================================================
class TestJLRandomProjectionPassthrough:
    """低次元入力では射影が適用されずpassthroughとなる。"""

    def test_passthrough_flag(self, low_dim_X: np.ndarray) -> None:
        """T-RP-001: n_features <= jl_min_dim ならば projection_active_ = False。"""
        rp = JLRandomProjection(eps=0.1)
        rp.fit(low_dim_X)

        assert rp.projection_active_ is False
        assert rp.projector_ is None
        assert rp.n_components_ == low_dim_X.shape[1]

    def test_passthrough_identity(self, low_dim_X: np.ndarray) -> None:
        """passthrough時はtransformが入力をそのまま返す。"""
        rp = JLRandomProjection(eps=0.1)
        rp.fit(low_dim_X)
        X_out = rp.transform(low_dim_X)

        np.testing.assert_array_equal(X_out, low_dim_X)
        assert X_out.shape == low_dim_X.shape


# ============================================================
# T-RP-002: JLRandomProjection — 高次元入力で射影適用
# ============================================================
class TestJLRandomProjectionActive:
    """高次元入力では射影が適用され次元削減される。"""

    def test_projection_active(self, high_dim_X: np.ndarray) -> None:
        """T-RP-002: n_features > jl_min_dim ならば projection_active_ = True。"""
        rp = JLRandomProjection(eps=0.5)
        rp.fit(high_dim_X)

        assert rp.projection_active_ is True
        assert rp.projector_ is not None
        assert rp.n_components_ < high_dim_X.shape[1]
        assert rp.n_components_ == rp.jl_min_dim_

    def test_dimension_reduction(self, high_dim_X: np.ndarray) -> None:
        """射影後の出力次元が n_components_ と一致する。"""
        rp = JLRandomProjection(eps=0.5)
        rp.fit(high_dim_X)
        X_out = rp.transform(high_dim_X)

        assert X_out.shape == (high_dim_X.shape[0], rp.n_components_)

    def test_deterministic_with_seed(self, high_dim_X: np.ndarray) -> None:
        """同じrandom_stateでは同じ結果が得られる。"""
        rp1 = JLRandomProjection(eps=0.5, random_state=42)
        rp1.fit(high_dim_X)
        out1 = rp1.transform(high_dim_X)

        rp2 = JLRandomProjection(eps=0.5, random_state=42)
        rp2.fit(high_dim_X)
        out2 = rp2.transform(high_dim_X)

        np.testing.assert_array_equal(out1, out2)


# ============================================================
# T-RP-003: summary()
# ============================================================
class TestSummary:
    def test_summary_passthrough(self, low_dim_X: np.ndarray) -> None:
        """T-RP-003a: passthroughの場合「不適用」文字列を返す。"""
        rp = JLRandomProjection(eps=0.1)
        rp.fit(low_dim_X)
        s = rp.summary()
        assert "不適用" in s

    def test_summary_active(self, high_dim_X: np.ndarray) -> None:
        """T-RP-003b: 射影適用時は次元数情報を含む。"""
        rp = JLRandomProjection(eps=0.5)
        rp.fit(high_dim_X)
        s = rp.summary()
        assert "→" in s
        assert str(rp.n_components_) in s


# ============================================================
# T-RP-004: get_feature_names_out
# ============================================================
class TestFeatureNamesOut:
    def test_passthrough_names(self, low_dim_X: np.ndarray) -> None:
        """T-RP-004a: passthrough時は入力特徴名をそのまま返す。"""
        rp = JLRandomProjection(eps=0.1)
        rp.fit(low_dim_X)
        names = rp.get_feature_names_out(["a", "b", "c", "d", "e"])
        np.testing.assert_array_equal(names, ["a", "b", "c", "d", "e"])

    def test_active_names(self, high_dim_X: np.ndarray) -> None:
        """T-RP-004b: 射影時は rp_0, rp_1, ... 形式。"""
        rp = JLRandomProjection(eps=0.5)
        rp.fit(high_dim_X)
        names = rp.get_feature_names_out()
        assert len(names) == rp.n_components_
        assert names[0] == "rp_0"


# ============================================================
# T-RP-005: method 自動選択
# ============================================================
class TestMethodAuto:
    def test_auto_selects_sparse_for_highdim(self) -> None:
        """T-RP-005: d > 1000 なら auto → sparse。"""
        rp = JLRandomProjection(method="auto")
        result = rp._resolve_method(1500)
        assert result == "sparse"

    def test_auto_selects_gaussian_for_lowdim(self) -> None:
        """T-RP-005: d <= 1000 なら auto → gaussian。"""
        rp = JLRandomProjection(method="auto")
        result = rp._resolve_method(500)
        assert result == "gaussian"


# ============================================================
# T-RP-006: method 固定
# ============================================================
class TestMethodFixed:
    def test_sparse_fixed(self, high_dim_X: np.ndarray) -> None:
        """T-RP-006a: method="sparse" で SparseRandomProjection が使われる。"""
        rp = JLRandomProjection(eps=0.5, method="sparse")
        rp.fit(high_dim_X)
        assert rp.projection_active_ is True
        from sklearn.random_projection import SparseRandomProjection
        assert isinstance(rp.projector_, SparseRandomProjection)

    def test_gaussian_fixed(self, high_dim_X: np.ndarray) -> None:
        """T-RP-006b: method="gaussian" で GaussianRandomProjection が使われる。"""
        rp = JLRandomProjection(eps=0.5, method="gaussian")
        rp.fit(high_dim_X)
        assert rp.projection_active_ is True
        from sklearn.random_projection import GaussianRandomProjection
        assert isinstance(rp.projector_, GaussianRandomProjection)


# ============================================================
# T-RP-007: should_apply_random_projection ユーティリティ
# ============================================================
class TestShouldApply:
    def test_should_not_apply_low_dim(self) -> None:
        """T-RP-007a: 低次元では False。"""
        apply, jl_min = should_apply_random_projection(
            n_features=5, n_samples=50, eps=0.1
        )
        assert apply is False
        assert isinstance(jl_min, int)
        assert jl_min > 0

    def test_should_apply_high_dim(self) -> None:
        """T-RP-007b: 高次元では True。"""
        apply, jl_min = should_apply_random_projection(
            n_features=2000, n_samples=100, eps=0.5
        )
        assert apply is True
        assert jl_min < 2000


# ============================================================
# T-RP-008: build_full_pipeline — JL-RP ステップ挿入
# ============================================================
class TestBuildPipelineWithJLRP:
    def test_jl_rp_step_present(self, regression_df: pd.DataFrame) -> None:
        """T-RP-008: random_projection_enable=True なら jl_rp ステップがある。"""
        detector = TypeDetector()
        dr = detector.detect(regression_df.drop(columns=["target"]))
        cfg = PreprocessConfig(
            random_projection_enable=True,
            random_projection_eps=0.1,
            random_projection_method="auto",
        )
        pipe = build_full_pipeline(dr, Ridge(), target_col="target", config=cfg)
        step_names = [name for name, _ in pipe.steps]
        assert "jl_rp" in step_names
        assert step_names.index("jl_rp") < step_names.index("model")

    def test_pipeline_fits(self, regression_df: pd.DataFrame) -> None:
        """JL-RPパイプラインが正常にfit/predictできる。"""
        detector = TypeDetector()
        X = regression_df.drop(columns=["target"])
        y = regression_df["target"].values
        dr = detector.detect(X)
        cfg = PreprocessConfig(
            random_projection_enable=True,
            random_projection_eps=0.1,
        )
        pipe = build_full_pipeline(dr, Ridge(), target_col="target", config=cfg)
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (len(y),)


# ============================================================
# T-RP-009: build_full_pipeline — JL-RP ステップなし
# ============================================================
class TestBuildPipelineWithoutJLRP:
    def test_no_jl_rp_step_by_default(self, regression_df: pd.DataFrame) -> None:
        """T-RP-009: デフォルト（random_projection_enable=False）ではjl_rpなし。"""
        detector = TypeDetector()
        dr = detector.detect(regression_df.drop(columns=["target"]))
        pipe = build_full_pipeline(dr, Ridge(), target_col="target")
        step_names = [name for name, _ in pipe.steps]
        assert "jl_rp" not in step_names


# ============================================================
# T-RP-010: PreprocessConfig — random_projection フィールド
# ============================================================
class TestPreprocessConfigRP:
    def test_default_disabled(self) -> None:
        """T-RP-010a: デフォルトは random_projection_enable=False。"""
        cfg = PreprocessConfig()
        assert cfg.random_projection_enable is False
        assert cfg.random_projection_eps == 0.1
        assert cfg.random_projection_method == "auto"

    def test_custom_config(self) -> None:
        """T-RP-010b: カスタム設定が適用される。"""
        cfg = PreprocessConfig(
            random_projection_enable=True,
            random_projection_eps=0.3,
            random_projection_method="sparse",
        )
        assert cfg.random_projection_enable is True
        assert cfg.random_projection_eps == 0.3
        assert cfg.random_projection_method == "sparse"


# ============================================================
# T-RP-011: run_multi_feature_sets — normal / highdim 混在
# ============================================================
class TestRunMultiFeatureSets:
    """AutoMLEngine.run_multi_feature_sets のテスト。"""

    def test_multi_sets_basic(self, regression_df: pd.DataFrame) -> None:
        """T-RP-011: normal と highdim の混在セットで実行可能。"""
        from backend.models.automl import AutoMLEngine

        engine = AutoMLEngine(
            task="regression",
            cv_folds=2,
            model_keys=["ridge"],
        )
        feature_sets = [
            {
                "id": "set1",
                "name": "通常セット",
                "descriptors": [],
                "pipeline": "normal",
            },
            {
                "id": "set2",
                "name": "高次元セット",
                "descriptors": [],
                "pipeline": "highdim",
                "rp_eps": 0.3,
            },
        ]
        results = engine.run_multi_feature_sets(
            df=regression_df,
            target_col="target",
            feature_sets=feature_sets,
        )

        assert len(results) == 2
        # 各結果の best_model_key が存在する
        for r in results:
            assert r.best_model_key == "ridge"
            assert r.best_score != 0.0

        # warningsにセット名とパイプラインタイプが含まれる
        assert any("__feature_set_name__:" in w for w in results[0].warnings)
        assert any("__feature_set_pipeline__:normal" in w for w in results[0].warnings)
        assert any("__feature_set_pipeline__:highdim" in w for w in results[1].warnings)


# ============================================================
# T-RP-012: NotFittedError
# ============================================================
class TestNotFittedError:
    def test_transform_before_fit(self) -> None:
        """T-RP-012: fit() 前の transform() で例外が発生する。"""
        rp = JLRandomProjection()
        X = np.random.randn(10, 5)
        with pytest.raises(Exception):  # sklearn NotFittedError
            rp.transform(X)


# ============================================================
# T-RP-013: sparse行列入力
# ============================================================
class TestSparseInput:
    def test_sparse_matrix_passthrough(self) -> None:
        """T-RP-013a: sparse行列を低次元で入力 → passthroughで密行列化。"""
        X_sparse = sparse.random(50, 5, density=0.3, format="csr")
        rp = JLRandomProjection(eps=0.1)
        rp.fit(X_sparse)
        X_out = rp.transform(X_sparse)
        assert isinstance(X_out, np.ndarray)
        assert X_out.shape == (50, 5)

    def test_sparse_matrix_projection(self) -> None:
        """T-RP-013b: sparse行列を高次元で入力 → 射影適用・密行列出力。"""
        X_sparse = sparse.random(100, 2000, density=0.1, format="csr")
        rp = JLRandomProjection(eps=0.5)
        rp.fit(X_sparse)
        X_out = rp.transform(X_sparse)
        assert isinstance(X_out, np.ndarray)
        assert X_out.shape[1] == rp.n_components_
        assert X_out.shape[1] < 2000
