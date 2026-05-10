"""
tests/test_bayesian_optimizer_extra.py

bayesian_optimizer.py の低カバレッジ部分を補うテスト。
BOConfig, BayesianOptimizer の fit/suggest/predict,
獲得関数(EI/PI/UCB/PTR), バッチ戦略(single/kriging_believer/doe_then_bo/bo_then_doe),
多目的(ParEGO), ヘルパー(_maximin_select等)を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.optim.bayesian_optimizer import (
    BOConfig,
    BayesianOptimizer,
)


# ============================================================
# テストデータ
# ============================================================

def _make_data(n: int = 30, d: int = 3):
    """基本的な回帰データ"""
    rng = np.random.RandomState(42)
    X = rng.randn(n, d)
    y = X[:, 0] ** 2 + 0.5 * X[:, 1] + rng.randn(n) * 0.1
    return X, y


def _make_candidates(n: int = 100, d: int = 3):
    rng = np.random.RandomState(123)
    return rng.randn(n, d)


# ============================================================
# BOConfig
# ============================================================

class TestBOConfig:
    def test_defaults(self):
        cfg = BOConfig()
        assert cfg.objective == "minimize"
        assert cfg.acquisition == "ei"
        assert cfg.n_candidates == 5

    def test_custom(self):
        cfg = BOConfig(
            objective="maximize",
            acquisition="ucb",
            kappa=3.0,
            n_candidates=10,
        )
        assert cfg.objective == "maximize"
        assert cfg.kappa == 3.0


# ============================================================
# BayesianOptimizer 基本テスト
# ============================================================

class TestBayesianOptimizerBasic:
    def test_fit_and_predict(self):
        X, y = _make_data()
        bo = BayesianOptimizer()
        bo.fit(X, y)
        X_new = _make_candidates(10, 3)
        mu, sigma = bo.predict(X_new)
        assert mu.shape == (10,)
        assert sigma.shape == (10,)

    def test_fit_with_dataframe(self):
        X, y = _make_data()
        df = pd.DataFrame(X, columns=["a", "b", "c"])
        bo = BayesianOptimizer()
        bo.fit(df, y)
        mu, sigma = bo.predict(df.iloc[:5])
        assert mu.shape == (5,)

    def test_predict_before_fit_raises(self):
        bo = BayesianOptimizer()
        with pytest.raises(RuntimeError, match="fit"):
            bo.predict(np.random.randn(5, 3))

    def test_suggest_before_fit_raises(self):
        bo = BayesianOptimizer()
        with pytest.raises(RuntimeError, match="fit"):
            bo.suggest(np.random.randn(10, 3))

    def test_get_gp_info_before_fit(self):
        bo = BayesianOptimizer()
        assert bo.get_gp_info() == {}

    def test_get_gp_info_after_fit(self):
        X, y = _make_data()
        bo = BayesianOptimizer()
        bo.fit(X, y)
        info = bo.get_gp_info()
        assert "kernel" in info
        assert "log_marginal_likelihood" in info
        assert "n_train" in info


# ============================================================
# カーネル構築
# ============================================================

class TestKernelBuilding:
    def test_default_kernel(self):
        bo = BayesianOptimizer(BOConfig(kernel_type="default"))
        X, y = _make_data()
        bo.fit(X, y)
        assert bo._gp is not None

    def test_matern_kernel(self):
        bo = BayesianOptimizer(BOConfig(kernel_type="matern", matern_nu=1.5))
        X, y = _make_data()
        bo.fit(X, y)
        assert bo._gp is not None

    def test_dotproduct_kernel(self):
        bo = BayesianOptimizer(BOConfig(kernel_type="dotproduct"))
        X, y = _make_data()
        bo.fit(X, y)
        assert bo._gp is not None


# ============================================================
# 獲得関数
# ============================================================

class TestAcquisitionFunctions:
    def test_ei_minimize(self):
        X, y = _make_data()
        bo = BayesianOptimizer(BOConfig(objective="minimize", acquisition="ei"))
        bo.fit(X, y)
        cands = _make_candidates(50, 3)
        result = bo.suggest(cands, n=3)
        assert len(result) == 3

    def test_ei_maximize(self):
        X, y = _make_data()
        bo = BayesianOptimizer(BOConfig(objective="maximize", acquisition="ei"))
        bo.fit(X, y)
        cands = _make_candidates(50, 3)
        result = bo.suggest(cands, n=3)
        assert len(result) == 3

    def test_pi(self):
        X, y = _make_data()
        bo = BayesianOptimizer(BOConfig(acquisition="pi"))
        bo.fit(X, y)
        cands = _make_candidates(50, 3)
        result = bo.suggest(cands, n=3)
        assert len(result) == 3

    def test_pi_maximize(self):
        X, y = _make_data()
        bo = BayesianOptimizer(BOConfig(objective="maximize", acquisition="pi"))
        bo.fit(X, y)
        cands = _make_candidates(50, 3)
        result = bo.suggest(cands, n=3)
        assert len(result) == 3

    def test_ucb_minimize(self):
        X, y = _make_data()
        bo = BayesianOptimizer(BOConfig(acquisition="ucb"))
        bo.fit(X, y)
        cands = _make_candidates(50, 3)
        result = bo.suggest(cands, n=3)
        assert len(result) == 3

    def test_ucb_maximize(self):
        X, y = _make_data()
        bo = BayesianOptimizer(BOConfig(objective="maximize", acquisition="ucb"))
        bo.fit(X, y)
        cands = _make_candidates(50, 3)
        result = bo.suggest(cands, n=3)
        assert len(result) == 3

    def test_ptr(self):
        X, y = _make_data()
        bo = BayesianOptimizer(BOConfig(
            acquisition="ptr",
            target_lo=-0.5,
            target_hi=0.5,
        ))
        bo.fit(X, y)
        cands = _make_candidates(50, 3)
        result = bo.suggest(cands, n=3)
        assert len(result) == 3

    def test_ptr_no_range_raises(self):
        X, y = _make_data()
        bo = BayesianOptimizer(BOConfig(acquisition="ptr"))
        bo.fit(X, y)
        cands = _make_candidates(50, 3)
        with pytest.raises(ValueError, match="target_lo"):
            bo.suggest(cands, n=3)

    def test_unknown_acquisition_raises(self):
        X, y = _make_data()
        bo = BayesianOptimizer(BOConfig(acquisition="unknown"))
        bo.fit(X, y)
        cands = _make_candidates(50, 3)
        with pytest.raises(ValueError, match="不明な獲得関数"):
            bo.suggest(cands, n=3)


# ============================================================
# バッチ戦略
# ============================================================

class TestBatchStrategies:
    def test_single(self):
        X, y = _make_data()
        bo = BayesianOptimizer(BOConfig(batch_strategy="single", n_candidates=3))
        bo.fit(X, y)
        cands = _make_candidates(50, 3)
        result = bo.suggest(cands)
        assert len(result) == 3

    def test_kriging_believer(self):
        X, y = _make_data()
        bo = BayesianOptimizer(BOConfig(batch_strategy="kriging_believer", n_candidates=3))
        bo.fit(X, y)
        cands = _make_candidates(50, 3)
        result = bo.suggest(cands)
        assert len(result) == 3

    def test_doe_then_bo(self):
        X, y = _make_data()
        bo = BayesianOptimizer(BOConfig(batch_strategy="doe_then_bo", n_candidates=3))
        bo.fit(X, y)
        cands = _make_candidates(50, 3)
        result = bo.suggest(cands)
        assert len(result) == 3

    def test_bo_then_doe(self):
        X, y = _make_data()
        bo = BayesianOptimizer(BOConfig(batch_strategy="bo_then_doe", n_candidates=3))
        bo.fit(X, y)
        cands = _make_candidates(50, 3)
        result = bo.suggest(cands)
        assert len(result) == 3

    def test_n_candidates_one(self):
        X, y = _make_data()
        bo = BayesianOptimizer(BOConfig(n_candidates=1))
        bo.fit(X, y)
        cands = _make_candidates(50, 3)
        result = bo.suggest(cands)
        assert len(result) == 1

    def test_suggest_with_dataframe(self):
        X, y = _make_data()
        bo = BayesianOptimizer(BOConfig(n_candidates=3))
        bo.fit(X, y)
        df_cand = pd.DataFrame(_make_candidates(50, 3), columns=["a", "b", "c"])
        result = bo.suggest(df_cand)
        assert isinstance(result, pd.DataFrame)
        assert "_acq_value" in result.columns
        assert "_rank" in result.columns
        assert len(result) == 3


# ============================================================
# 多目的 (ParEGO)
# ============================================================

class TestMultiObjective:
    def test_parego_basic(self):
        rng = np.random.RandomState(42)
        X = rng.randn(30, 3)
        # 2目的
        Y = np.column_stack([
            X[:, 0] ** 2 + rng.randn(30) * 0.1,
            -X[:, 1] + rng.randn(30) * 0.1,
        ])
        cfg = BOConfig(
            multi_objective=True,
            objective_columns=["obj1", "obj2"],
            objective_directions=["min", "max"],
            n_candidates=3,
        )
        bo = BayesianOptimizer(cfg)
        bo.fit(X, Y)
        cands = _make_candidates(50, 3)
        result = bo.suggest(cands)
        assert len(result) == 3

    def test_multi_predict(self):
        rng = np.random.RandomState(42)
        X = rng.randn(30, 3)
        Y = np.column_stack([
            X[:, 0] + rng.randn(30) * 0.1,
            X[:, 1] + rng.randn(30) * 0.1,
        ])
        cfg = BOConfig(multi_objective=True, n_candidates=3)
        bo = BayesianOptimizer(cfg)
        bo.fit(X, Y)
        mu, sigma = bo.predict(X[:5])
        assert mu.shape == (5, 2)
        assert sigma.shape == (5, 2)

    def test_multi_gp_info(self):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 2)
        Y = np.column_stack([X[:, 0], X[:, 1]])
        cfg = BOConfig(multi_objective=True)
        bo = BayesianOptimizer(cfg)
        bo.fit(X, Y)
        info = bo.get_gp_info()
        assert "n_objectives" in info
        assert info["n_objectives"] == 2


# ============================================================
# ヘルパー
# ============================================================

class TestMaximinSelect:
    def test_basic(self):
        X = np.array([
            [0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5],
        ])
        result = BayesianOptimizer._maximin_select(X, 3)
        assert len(result) == 3

    def test_n_greater_than_length(self):
        X = np.array([[0, 0], [1, 1]])
        result = BayesianOptimizer._maximin_select(X, 10)
        assert len(result) == 2
