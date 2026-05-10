"""
tests/test_sri_extra.py

sri.py のカバレッジ改善テスト。
SRIResult, SRIDecomposer, select_features_by_independence を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.interpret.sri import (
    SRIResult,
    SRIDecomposer,
    select_features_by_independence,
)
from backend.interpret.shap_explainer import ShapResult


def _make_shap_result(n: int = 50, d: int = 5, multiclass: bool = False):
    """テスト用のダミーShapResultを生成。"""
    rng = np.random.RandomState(42)
    if multiclass:
        sv = rng.randn(n, d, 3)
    else:
        sv = rng.randn(n, d)
    return ShapResult(
        shap_values=sv,
        expected_value=0.5,
        feature_names=[f"f{i}" for i in range(d)],
        X_transformed=rng.randn(n, d),
        explainer_type="tree",
        is_multiclass=multiclass,
    )


# ============================================================
# SRIDecomposer
# ============================================================

class TestSRIDecomposer:
    def test_basic_decompose(self):
        shap_r = _make_shap_result()
        decomp = SRIDecomposer(center=True)
        result = decomp.decompose(shap_r)
        assert isinstance(result, SRIResult)
        assert result.synergy_matrix.shape == (5, 5)
        assert result.redundancy_matrix.shape == (5, 5)
        assert len(result.independence_vec) == 5
        assert len(result.total_sri) == 3

    def test_no_center(self):
        shap_r = _make_shap_result()
        decomp = SRIDecomposer(center=False)
        result = decomp.decompose(shap_r)
        assert result.synergy_matrix.shape == (5, 5)

    def test_multiclass(self):
        """マルチクラスSHAPは3次元 → クラス0で分解。"""
        shap_r = _make_shap_result(multiclass=True)
        decomp = SRIDecomposer()
        result = decomp.decompose(shap_r)
        assert result.synergy_matrix.shape == (5, 5)

    def test_invalid_shape(self):
        sr = ShapResult(
            shap_values=np.array([1, 2, 3]),  # 1次元
            expected_value=0.0,
            feature_names=["a"],
            X_transformed=np.array([[1], [2], [3]]),
            explainer_type="tree",
        )
        decomp = SRIDecomposer()
        with pytest.raises(ValueError, match="2次元"):
            decomp.decompose(sr)


# ============================================================
# SRIResult
# ============================================================

class TestSRIResult:
    def _make_result(self):
        rng = np.random.RandomState(42)
        d = 4
        syn = rng.randn(d, d) * 0.1
        syn = (syn + syn.T) / 2
        np.fill_diagonal(syn, 0)
        red = np.abs(rng.randn(d, d)) * 0.05
        red = (red + red.T) / 2
        np.fill_diagonal(red, 0)
        ind = np.abs(rng.randn(d))
        return SRIResult(
            synergy_matrix=syn,
            redundancy_matrix=red,
            independence_vec=ind,
            feature_names=["a", "b", "c", "d"],
            total_sri=(float(np.abs(syn).sum()/2),
                       float(np.abs(red).sum()/2),
                       float(ind.sum())),
        )

    def test_summary_df(self):
        result = self._make_result()
        df = result.summary_df()
        assert isinstance(df, pd.DataFrame)
        assert "synergy" in df.columns
        assert "redundancy" in df.columns
        assert "independence" in df.columns
        assert len(df) == 4

    def test_pairwise_df(self):
        result = self._make_result()
        df = result.pairwise_df()
        assert isinstance(df, pd.DataFrame)
        assert "feature_i" in df.columns
        assert "synergy" in df.columns
        # 4C2 = 6 pairs
        assert len(df) == 6


# ============================================================
# select_features_by_independence
# ============================================================

class TestSelectFeatures:
    def _make_result(self):
        return SRIResult(
            synergy_matrix=np.zeros((4, 4)),
            redundancy_matrix=np.zeros((4, 4)),
            independence_vec=np.array([0.8, 0.2, 0.5, 0.1]),
            feature_names=["a", "b", "c", "d"],
            total_sri=(0.0, 0.0, 1.6),
        )

    def test_top_n(self):
        result = self._make_result()
        selected = select_features_by_independence(result, top_n=2)
        assert len(selected) == 2
        assert selected[0] == "a"  # highest independence

    def test_threshold(self):
        result = self._make_result()
        selected = select_features_by_independence(result, threshold=0.3)
        assert "a" in selected
        assert "c" in selected
        assert "b" not in selected

    def test_all(self):
        result = self._make_result()
        selected = select_features_by_independence(result)
        assert len(selected) == 4
