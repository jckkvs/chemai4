# -*- coding: utf-8 -*-
"""
tests/test_sri.py

SRIDecomposer（SHAP SRI分解）のユニットテスト。

論文: Ittner et al. (2021) arXiv:2107.12436

Note: shapライブラリ自体がインストールされていない環境でも動作するよう、
ShapResultをモック的に構築してテストする。
"""
from __future__ import annotations

import numpy as np
import pytest

# shapライブラリ依存を回避するため、ShapResultを直接インポートせず
# ダミーのdataclassを定義する
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _MockShapResult:
    """SRI分解テスト用のモックShapResult"""
    shap_values: np.ndarray
    expected_value: float | np.ndarray
    feature_names: list[str]
    X_transformed: np.ndarray
    explainer_type: str = "mock"
    is_multiclass: bool = False
    shap_interaction_values: np.ndarray | None = None
    base_values: np.ndarray | None = None


# sri.py はshap_explainerからShapResultをインポートしているが、
# shap未インストールだと __init__.py 経由のインポートがエラーになる。
# モジュールを直接インポートしてみて、ダメならモックで回避する。
try:
    # interpretパッケージの__init__.pyが失敗するので直接sri.pyをインポート
    import importlib
    import sys

    # shap_explainer をダミーモジュールとして登録
    if "backend.interpret.shap_explainer" not in sys.modules:
        import types
        dummy_module = types.ModuleType("backend.interpret.shap_explainer")
        dummy_module.ShapResult = _MockShapResult  # type: ignore
        sys.modules["backend.interpret.shap_explainer"] = dummy_module
    if "backend.interpret" not in sys.modules:
        import types
        dummy_interpret = types.ModuleType("backend.interpret")
        sys.modules["backend.interpret"] = dummy_interpret

    from backend.interpret.sri import SRIDecomposer, SRIResult, select_features_by_independence
    SRI_AVAILABLE = True
except Exception as e:
    SRI_AVAILABLE = False
    _SRI_IMPORT_ERROR = str(e)

requires_sri = pytest.mark.skipif(not SRI_AVAILABLE, reason=f"SRI module import failed")


# ═══════════════════════════════════════════════════════════════════
# ヘルパー: ダミーShapResult
# ═══════════════════════════════════════════════════════════════════

def _make_shap_result(
    shap_values: np.ndarray,
    feature_names: list[str] | None = None,
) -> _MockShapResult:
    n, d = shap_values.shape
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(d)]
    return _MockShapResult(
        shap_values=shap_values,
        expected_value=0.0,
        feature_names=feature_names,
        X_transformed=np.zeros((n, d)),
    )


# ═══════════════════════════════════════════════════════════════════
# SRIDecomposer 基本テスト
# ═══════════════════════════════════════════════════════════════════

@requires_sri
class TestSRIDecomposer:

    def test_basic_decomposition(self):
        """3特徴量のSRI分解が正しい形状を返す"""
        shap_vals = np.random.default_rng(42).normal(size=(100, 3))
        sr = _make_shap_result(shap_vals, ["a", "b", "c"])
        decomposer = SRIDecomposer(center=True)
        result = decomposer.decompose(sr)
        assert isinstance(result, SRIResult)
        assert result.synergy_matrix.shape == (3, 3)
        assert result.redundancy_matrix.shape == (3, 3)
        assert result.independence_vec.shape == (3,)
        assert len(result.feature_names) == 3

    def test_symmetry(self):
        """Synergy/Redundancy行列は対称"""
        shap_vals = np.random.default_rng(42).normal(size=(50, 4))
        sr = _make_shap_result(shap_vals)
        result = SRIDecomposer().decompose(sr)
        np.testing.assert_allclose(result.synergy_matrix, result.synergy_matrix.T)
        np.testing.assert_allclose(result.redundancy_matrix, result.redundancy_matrix.T)

    def test_diagonal_zero(self):
        """対角要素は0"""
        shap_vals = np.random.default_rng(42).normal(size=(50, 3))
        sr = _make_shap_result(shap_vals)
        result = SRIDecomposer().decompose(sr)
        np.testing.assert_allclose(np.diag(result.synergy_matrix), 0)
        np.testing.assert_allclose(np.diag(result.redundancy_matrix), 0)

    def test_independence_nonnegative(self):
        """Independence値は非負（clip後）"""
        shap_vals = np.random.default_rng(42).normal(size=(50, 5))
        sr = _make_shap_result(shap_vals)
        result = SRIDecomposer().decompose(sr)
        assert np.all(result.independence_vec >= 0)

    def test_total_sri_tuple(self):
        """total_sriは3要素のタプル"""
        shap_vals = np.random.default_rng(42).normal(size=(30, 3))
        sr = _make_shap_result(shap_vals)
        result = SRIDecomposer().decompose(sr)
        assert len(result.total_sri) == 3
        syn, red, ind = result.total_sri
        assert syn >= 0
        assert red >= 0
        assert ind >= 0

    def test_center_false(self):
        """center=Falseでも動作する"""
        shap_vals = np.random.default_rng(42).normal(size=(30, 3))
        sr = _make_shap_result(shap_vals)
        result = SRIDecomposer(center=False).decompose(sr)
        assert result.synergy_matrix.shape == (3, 3)

    def test_correlated_features_high_redundancy(self):
        """相関の高い特徴量ペアはRedundancyが高い"""
        rng = np.random.default_rng(42)
        base = rng.normal(size=(100, 1))
        shap_vals = np.hstack([base, base + rng.normal(0, 0.01, (100, 1)),
                               rng.normal(size=(100, 1))])
        sr = _make_shap_result(shap_vals, ["corr1", "corr2", "indep"])
        result = SRIDecomposer().decompose(sr)
        assert abs(result.redundancy_matrix[0, 1]) > abs(result.redundancy_matrix[0, 2])

    def test_single_feature(self):
        """1特徴量のみ"""
        shap_vals = np.random.default_rng(42).normal(size=(20, 1))
        sr = _make_shap_result(shap_vals, ["only"])
        result = SRIDecomposer().decompose(sr)
        assert result.synergy_matrix.shape == (1, 1)
        assert result.independence_vec.shape == (1,)

    def test_3d_shap_multiclass(self):
        """マルチクラスSHAP (3D) はクラス0のみ使用"""
        shap_3d = np.random.default_rng(42).normal(size=(20, 3, 2))
        sr = _MockShapResult(
            shap_values=shap_3d, expected_value=0.0,
            feature_names=["a", "b", "c"],
            X_transformed=np.zeros((20, 3)),
        )
        result = SRIDecomposer().decompose(sr)
        assert result.synergy_matrix.shape == (3, 3)

    def test_invalid_1d_raises(self):
        """1次元SHAPはValueError"""
        shap_1d = np.array([1.0, 2.0, 3.0])
        sr = _MockShapResult(
            shap_values=shap_1d, expected_value=0.0,
            feature_names=["a"],
            X_transformed=np.zeros((3, 1)),
        )
        with pytest.raises(ValueError, match="2次元"):
            SRIDecomposer().decompose(sr)


# ═══════════════════════════════════════════════════════════════════
# SRIResult メソッド
# ═══════════════════════════════════════════════════════════════════

@requires_sri
class TestSRIResult:

    @pytest.fixture
    def sri_result(self):
        shap_vals = np.random.default_rng(42).normal(size=(100, 4))
        sr = _make_shap_result(shap_vals, ["a", "b", "c", "d"])
        return SRIDecomposer().decompose(sr)

    def test_summary_df(self, sri_result):
        df = sri_result.summary_df()
        assert "feature" in df.columns
        assert "synergy" in df.columns
        assert "redundancy" in df.columns
        assert "independence" in df.columns
        assert len(df) == 4

    def test_summary_df_normalized(self, sri_result):
        df = sri_result.summary_df()
        assert "synergy_norm" in df.columns
        assert "redundancy_norm" in df.columns
        assert "independence_norm" in df.columns

    def test_pairwise_df(self, sri_result):
        df = sri_result.pairwise_df()
        assert "feature_i" in df.columns
        assert "feature_j" in df.columns
        assert "synergy" in df.columns
        assert "redundancy" in df.columns
        assert len(df) == 6  # 4 features → C(4,2) = 6 pairs


# ═══════════════════════════════════════════════════════════════════
# select_features_by_independence
# ═══════════════════════════════════════════════════════════════════

@requires_sri
class TestSelectFeatures:

    @pytest.fixture
    def sri_result(self):
        shap_vals = np.random.default_rng(42).normal(size=(100, 5))
        sr = _make_shap_result(shap_vals, ["a", "b", "c", "d", "e"])
        return SRIDecomposer().decompose(sr)

    def test_top_n(self, sri_result):
        selected = select_features_by_independence(sri_result, top_n=3)
        assert len(selected) == 3

    def test_threshold(self, sri_result):
        selected = select_features_by_independence(sri_result, threshold=0.0)
        assert len(selected) >= 1

    def test_no_filter(self, sri_result):
        selected = select_features_by_independence(sri_result)
        assert len(selected) == 5
