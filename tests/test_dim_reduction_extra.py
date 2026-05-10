"""
tests/test_dim_reduction_extra.py

dim_reduction.py のカバレッジ改善テスト。
DimReductionConfig, DimReducer (PCA/t-SNE), run_pca, run_tsne を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.data.dim_reduction import (
    DimReductionConfig,
    DimReducer,
    run_pca,
    run_tsne,
)


# ============================================================
# テストデータ
# ============================================================

def _make_data(n: int = 60, d: int = 5):
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        rng.randn(n, d),
        columns=[f"feat_{i}" for i in range(d)]
    )


# ============================================================
# DimReductionConfig
# ============================================================

class TestDimReductionConfig:
    def test_defaults(self):
        cfg = DimReductionConfig()
        assert cfg.method == "pca"
        assert cfg.n_components == 2
        assert cfg.scale is True


# ============================================================
# DimReducer — PCA
# ============================================================

class TestDimReducerPCA:
    def test_pca_fit_transform(self):
        df = _make_data()
        cfg = DimReductionConfig(method="pca", n_components=2)
        reducer = DimReducer(cfg)
        result = reducer.fit_transform(df)
        assert result.shape == (60, 2)

    def test_pca_explained_variance(self):
        df = _make_data()
        reducer = DimReducer(DimReductionConfig(method="pca", n_components=3))
        reducer.fit_transform(df)
        evr = reducer.explained_variance_ratio_
        assert evr is not None
        assert len(evr) == 3
        assert sum(evr) <= 1.0 + 1e-6

    def test_pca_loadings(self):
        df = _make_data()
        reducer = DimReducer(DimReductionConfig(method="pca", n_components=2))
        reducer.fit_transform(df)
        loadings = reducer.loadings_
        assert loadings is not None
        assert loadings.shape == (2, 5)

    def test_pca_reconstruction_error(self):
        df = _make_data()
        reducer = DimReducer(DimReductionConfig(method="pca", n_components=2))
        reducer.fit_transform(df)
        errors = reducer.reconstruction_error_
        assert errors is not None
        assert len(errors) == 60
        assert all(e >= 0 for e in errors)

    def test_pca_feature_names(self):
        df = _make_data()
        reducer = DimReducer(DimReductionConfig(method="pca", n_components=2))
        reducer.fit_transform(df)
        names = reducer.feature_names_in_
        assert len(names) == 5

    def test_get_feature_names_out(self):
        reducer = DimReducer(DimReductionConfig(method="pca", n_components=3))
        names = reducer.get_feature_names_out()
        assert len(names) == 3
        assert names[0] == "pca_0"

    def test_pca_no_scale(self):
        df = _make_data()
        cfg = DimReductionConfig(method="pca", n_components=2, scale=False)
        reducer = DimReducer(cfg)
        result = reducer.fit_transform(df)
        assert result.shape == (60, 2)

    def test_pca_whiten(self):
        df = _make_data()
        cfg = DimReductionConfig(method="pca", n_components=2, whiten=True)
        reducer = DimReducer(cfg)
        result = reducer.fit_transform(df)
        assert result.shape == (60, 2)

    def test_pca_ndarray_input(self):
        X = np.random.randn(40, 4)
        cfg = DimReductionConfig(method="pca", n_components=2)
        reducer = DimReducer(cfg)
        result = reducer.fit_transform(X)
        assert result.shape == (40, 2)


# ============================================================
# DimReducer — t-SNE
# ============================================================

class TestDimReducerTSNE:
    def test_tsne_fit_transform(self):
        df = _make_data(n=30)
        cfg = DimReductionConfig(method="tsne", n_components=2, perplexity=5.0)
        reducer = DimReducer(cfg)
        result = reducer.fit_transform(df)
        assert result.shape == (30, 2)


# ============================================================
# DimReducer — Unknown
# ============================================================

class TestDimReducerUnknown:
    def test_unknown_method(self):
        cfg = DimReductionConfig(method="unknown_dim")
        reducer = DimReducer(cfg)
        with pytest.raises(ValueError, match="未知"):
            reducer.fit(np.random.randn(20, 3))


# ============================================================
# 便利関数
# ============================================================

class TestConvenienceFunctions:
    def test_run_pca(self):
        df = _make_data()
        emb_df, evr = run_pca(df, n_components=2)
        assert isinstance(emb_df, pd.DataFrame)
        assert emb_df.shape == (60, 2)
        assert "PC1" in emb_df.columns
        assert len(evr) == 2

    def test_run_pca_with_target(self):
        df = _make_data()
        df["target"] = np.random.randn(60)
        emb_df, evr = run_pca(df, n_components=2, target_col="target")
        assert "target" not in emb_df.columns

    def test_run_tsne(self):
        df = _make_data(n=30)
        emb_df = run_tsne(df, n_components=2, perplexity=5.0)
        assert isinstance(emb_df, pd.DataFrame)
        assert emb_df.shape == (30, 2)
        assert "tSNE1" in emb_df.columns
