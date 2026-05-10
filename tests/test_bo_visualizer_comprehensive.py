"""
tests/test_bo_visualizer_comprehensive.py

bo_visualizer.py の包括テスト。
plot_pca_2d / plot_pca_3d / plot_pareto_front / plot_convergence /
_is_pareto_efficient を網羅。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.optim.bo_visualizer import (
    plot_pca_2d,
    plot_pca_3d,
    plot_pareto_front,
    plot_convergence,
    _is_pareto_efficient,
)


@pytest.fixture
def data_5d():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 5)
    y = X[:, 0] + rng.randn(30) * 0.1
    return X, y


class TestPlotPCA2D:
    def test_basic(self, data_5d):
        X, y = data_5d
        fig, info = plot_pca_2d(X, y_existing=y)
        assert "explained_variance" in info
        assert len(info["explained_variance"]) == 2

    def test_with_candidates(self, data_5d):
        X, y = data_5d
        cand = np.random.randn(5, 5)
        fig, info = plot_pca_2d(X, X_candidates=cand, y_existing=y)
        assert info["cumulative"][-1] <= 1.0

    def test_dataframe(self, data_5d):
        X, y = data_5d
        df = pd.DataFrame(X, columns=["a", "b", "c", "d", "e"])
        fig, info = plot_pca_2d(df)
        assert len(info["explained_variance"]) == 2

    def test_no_y(self, data_5d):
        X, _ = data_5d
        fig, info = plot_pca_2d(X)
        assert fig is not None


class TestPlotPCA3D:
    def test_basic(self, data_5d):
        X, y = data_5d
        fig, info = plot_pca_3d(X, y_existing=y)
        assert len(info["explained_variance"]) == 3

    def test_with_candidates(self, data_5d):
        X, y = data_5d
        cand = np.random.randn(5, 5)
        fig, info = plot_pca_3d(X, X_candidates=cand)
        assert fig is not None


class TestParetoFront:
    def test_basic(self):
        Y = np.array([[1, 5], [2, 3], [3, 1], [4, 4], [5, 2]])
        fig = plot_pareto_front(Y)
        assert fig is not None

    def test_with_candidates(self):
        Y_ex = np.array([[1, 5], [3, 1]])
        Y_cand = np.array([[2, 2]])
        fig = plot_pareto_front(Y_ex, Y_candidates=Y_cand)
        assert fig is not None

    def test_custom_directions(self):
        Y = np.array([[1, 5], [2, 3], [3, 1]])
        fig = plot_pareto_front(Y, directions=["max", "min"])
        assert fig is not None


class TestParetoEfficient:
    def test_basic(self):
        Y = np.array([[1, 5], [2, 3], [3, 1], [4, 4]])
        mask = _is_pareto_efficient(Y, ["min", "min"])
        assert mask[0]  # (1,5) dominates nothing but not dominated by (3,1)
        assert mask[2]  # (3,1) is pareto optimal

    def test_all_min(self):
        Y = np.array([[1, 1], [2, 2], [3, 3]])
        mask = _is_pareto_efficient(Y, ["min", "min"])
        assert mask[0]  # (1,1) dominates all
        assert not mask[2]  # (3,3) is dominated


class TestConvergence:
    def test_minimize(self):
        y_hist = [5.0, 3.0, 4.0, 2.0, 1.0]
        fig = plot_convergence(y_hist, objective="minimize")
        assert fig is not None

    def test_maximize(self):
        y_hist = [1.0, 3.0, 2.0, 5.0, 4.0]
        fig = plot_convergence(y_hist, objective="maximize")
        assert fig is not None
