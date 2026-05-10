"""
tests/test_bo_visualizer.py

bo_visualizer.py（ベイズ最適化可視化モジュール）のユニットテスト。
カバレッジ0% → 80%+を目指す。
全4 public関数 + 内部ヘルパーを網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from backend.optim.bo_visualizer import (
    plot_pca_2d,
    plot_pca_3d,
    plot_pareto_front,
    plot_convergence,
    _is_pareto_efficient,
)


# ============================================================
# テストデータ
# ============================================================

def _random_X(n: int = 30, d: int = 5) -> np.ndarray:
    rng = np.random.RandomState(42)
    return rng.randn(n, d)


def _random_df(n: int = 30, d: int = 5) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame(rng.randn(n, d), columns=[f"feat_{i}" for i in range(d)])


# ============================================================
# PCA 2D
# ============================================================

class TestPlotPCA2D:
    def test_basic_ndarray(self):
        X = _random_X()
        fig, info = plot_pca_2d(X)
        assert isinstance(fig, go.Figure)
        assert "explained_variance" in info
        assert "cumulative" in info
        assert "components" in info

    def test_with_dataframe(self):
        df = _random_df()
        fig, info = plot_pca_2d(df)
        assert isinstance(fig, go.Figure)
        assert len(info["explained_variance"]) == 2

    def test_with_candidates(self):
        X = _random_X(30, 5)
        X_cand = _random_X(5, 5)
        fig, info = plot_pca_2d(X, X_candidates=X_cand)
        assert isinstance(fig, go.Figure)

    def test_with_y_values(self):
        X = _random_X()
        y = np.random.randn(30)
        fig, info = plot_pca_2d(X, y_existing=y)
        assert isinstance(fig, go.Figure)

    def test_without_y(self):
        X = _random_X()
        fig, info = plot_pca_2d(X, y_existing=None)
        assert isinstance(fig, go.Figure)

    def test_custom_feature_names(self):
        X = _random_X(30, 5)
        fig, info = plot_pca_2d(X, feature_names=["a", "b", "c", "d", "e"])
        assert isinstance(fig, go.Figure)

    def test_top_n_arrows(self):
        X = _random_X(30, 10)
        fig, info = plot_pca_2d(X, top_n_arrows=3)
        assert isinstance(fig, go.Figure)

    def test_full_with_all_options(self):
        X = _random_X(50, 8)
        X_cand = _random_X(5, 8)
        y = np.random.randn(50)
        fig, info = plot_pca_2d(
            X, X_candidates=X_cand, y_existing=y,
            feature_names=[f"x{i}" for i in range(8)],
            top_n_arrows=4,
        )
        assert isinstance(fig, go.Figure)
        assert len(info["cumulative"]) == 2


# ============================================================
# PCA 3D
# ============================================================

class TestPlotPCA3D:
    def test_basic(self):
        X = _random_X()
        fig, info = plot_pca_3d(X)
        assert isinstance(fig, go.Figure)
        assert len(info["explained_variance"]) == 3

    def test_with_candidates(self):
        X = _random_X(30, 5)
        X_cand = _random_X(5, 5)
        fig, info = plot_pca_3d(X, X_candidates=X_cand)
        assert isinstance(fig, go.Figure)

    def test_with_y(self):
        X = _random_X()
        y = np.random.randn(30)
        fig, info = plot_pca_3d(X, y_existing=y)
        assert isinstance(fig, go.Figure)

    def test_without_y(self):
        X = _random_X()
        fig, info = plot_pca_3d(X, y_existing=None)
        assert isinstance(fig, go.Figure)

    def test_with_dataframe(self):
        df = _random_df()
        fig, info = plot_pca_3d(df)
        assert isinstance(fig, go.Figure)

    def test_feature_names_and_arrows(self):
        X = _random_X(30, 8)
        fig, info = plot_pca_3d(
            X,
            feature_names=[f"var{i}" for i in range(8)],
            top_n_arrows=5,
        )
        assert isinstance(fig, go.Figure)

    def test_low_dim_input(self):
        """2次元データでもエラーなく動作する"""
        X = _random_X(30, 2)
        fig, info = plot_pca_3d(X)
        assert isinstance(fig, go.Figure)


# ============================================================
# パレートフロント
# ============================================================

class TestPlotParetoFront:
    def test_basic_minimize(self):
        rng = np.random.RandomState(42)
        Y = rng.randn(20, 2)
        fig = plot_pareto_front(Y)
        assert isinstance(fig, go.Figure)

    def test_with_objective_names(self):
        rng = np.random.RandomState(42)
        Y = rng.randn(20, 2)
        fig = plot_pareto_front(Y, objective_names=["Obj1", "Obj2"])
        assert isinstance(fig, go.Figure)

    def test_with_candidates(self):
        rng = np.random.RandomState(42)
        Y_ex = rng.randn(20, 2)
        Y_cand = rng.randn(5, 2)
        fig = plot_pareto_front(Y_ex, Y_candidates=Y_cand)
        assert isinstance(fig, go.Figure)

    def test_with_directions(self):
        rng = np.random.RandomState(42)
        Y = rng.randn(20, 2)
        fig = plot_pareto_front(Y, directions=["min", "max"])
        assert isinstance(fig, go.Figure)

    def test_with_dataframe(self):
        rng = np.random.RandomState(42)
        Y = pd.DataFrame(rng.randn(30, 2), columns=["A", "B"])
        fig = plot_pareto_front(Y)
        assert isinstance(fig, go.Figure)


# ============================================================
# パレート効率性
# ============================================================

class TestParetoEfficient:
    def test_basic_min(self):
        Y = np.array([[1, 2], [2, 1], [3, 3], [0.5, 0.5]])
        mask = _is_pareto_efficient(Y, ["min", "min"])
        # (0.5, 0.5) dominates all for min
        assert mask[3] is True or mask[3] == True

    def test_basic_max(self):
        Y = np.array([[1, 2], [2, 1], [3, 3], [0.5, 0.5]])
        mask = _is_pareto_efficient(Y, ["max", "max"])
        assert mask[2] is True or mask[2] == True  # (3, 3) dominates

    def test_no_domination(self):
        Y = np.array([[1, 3], [2, 2], [3, 1]])
        mask = _is_pareto_efficient(Y, ["min", "min"])
        assert np.all(mask)  # All are pareto optimal


# ============================================================
# 収束曲線
# ============================================================

class TestPlotConvergence:
    def test_minimize(self):
        y_hist = [5.0, 4.0, 4.5, 3.0, 3.5, 2.0]
        fig = plot_convergence(y_hist, objective="minimize")
        assert isinstance(fig, go.Figure)

    def test_maximize(self):
        y_hist = [1.0, 2.0, 1.5, 3.0, 2.5, 4.0]
        fig = plot_convergence(y_hist, objective="maximize")
        assert isinstance(fig, go.Figure)

    def test_single_point(self):
        fig = plot_convergence([3.14])
        assert isinstance(fig, go.Figure)

    def test_long_history(self):
        rng = np.random.RandomState(42)
        y_hist = rng.randn(200).tolist()
        fig = plot_convergence(y_hist)
        assert isinstance(fig, go.Figure)
