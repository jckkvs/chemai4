"""
tests/test_bayesian_optimizer_comprehensive.py

BayesianOptimizer / BOConfig の包括テスト。
全獲得関数(EI/PI/UCB/PTR)、全バッチ戦略(single/KB/doe_then_bo/bo_then_doe)、
多目的(ParEGO)、predict、get_gp_info を網羅。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.optim.bayesian_optimizer import BOConfig, BayesianOptimizer


@pytest.fixture
def reg_data():
    rng = np.random.RandomState(42)
    X = rng.randn(20, 3)
    y = X[:, 0] * 2 + X[:, 1] - X[:, 2] + rng.randn(20) * 0.1
    return X, y


@pytest.fixture
def candidates():
    rng = np.random.RandomState(0)
    return rng.randn(50, 3)


class TestBOConfig:
    def test_defaults(self):
        cfg = BOConfig()
        assert cfg.objective == "minimize"
        assert cfg.acquisition == "ei"

    def test_custom(self):
        cfg = BOConfig(objective="maximize", acquisition="ucb", kappa=3.0)
        assert cfg.kappa == 3.0


class TestBayesianOptimizerFit:
    def test_fit_basic(self, reg_data):
        X, y = reg_data
        bo = BayesianOptimizer()
        bo.fit(X, y)
        assert bo._is_fitted

    def test_fit_dataframe(self, reg_data):
        X, y = reg_data
        df = pd.DataFrame(X, columns=["a", "b", "c"])
        bo = BayesianOptimizer()
        bo.fit(df, y)
        assert bo._is_fitted


class TestAcquisitionFunctions:
    def test_ei_minimize(self, reg_data, candidates):
        X, y = reg_data
        bo = BayesianOptimizer(BOConfig(acquisition="ei", objective="minimize"))
        bo.fit(X, y)
        result = bo.suggest(candidates, n=3)
        assert len(result) == 3

    def test_ei_maximize(self, reg_data, candidates):
        X, y = reg_data
        bo = BayesianOptimizer(BOConfig(acquisition="ei", objective="maximize"))
        bo.fit(X, y)
        result = bo.suggest(candidates, n=3)
        assert len(result) == 3

    def test_pi(self, reg_data, candidates):
        X, y = reg_data
        bo = BayesianOptimizer(BOConfig(acquisition="pi"))
        bo.fit(X, y)
        result = bo.suggest(candidates, n=3)
        assert len(result) == 3

    def test_ucb(self, reg_data, candidates):
        X, y = reg_data
        bo = BayesianOptimizer(BOConfig(acquisition="ucb"))
        bo.fit(X, y)
        result = bo.suggest(candidates, n=3)
        assert len(result) == 3

    def test_ptr(self, reg_data, candidates):
        X, y = reg_data
        bo = BayesianOptimizer(BOConfig(
            acquisition="ptr",
            objective="target_range",
            target_lo=-1.0,
            target_hi=1.0,
        ))
        bo.fit(X, y)
        result = bo.suggest(candidates, n=3)
        assert len(result) == 3


class TestBatchStrategies:
    def test_single(self, reg_data, candidates):
        X, y = reg_data
        bo = BayesianOptimizer(BOConfig(batch_strategy="single"))
        bo.fit(X, y)
        result = bo.suggest(candidates, n=3)
        assert len(result) == 3

    def test_kriging_believer(self, reg_data, candidates):
        X, y = reg_data
        bo = BayesianOptimizer(BOConfig(batch_strategy="kriging_believer"))
        bo.fit(X, y)
        result = bo.suggest(candidates, n=3)
        assert len(result) == 3

    def test_doe_then_bo(self, reg_data, candidates):
        X, y = reg_data
        bo = BayesianOptimizer(BOConfig(batch_strategy="doe_then_bo"))
        bo.fit(X, y)
        result = bo.suggest(candidates, n=3)
        assert len(result) == 3

    def test_bo_then_doe(self, reg_data, candidates):
        X, y = reg_data
        bo = BayesianOptimizer(BOConfig(batch_strategy="bo_then_doe"))
        bo.fit(X, y)
        result = bo.suggest(candidates, n=3)
        assert len(result) == 3


class TestPredict:
    def test_predict(self, reg_data, candidates):
        X, y = reg_data
        bo = BayesianOptimizer()
        bo.fit(X, y)
        mu, sigma = bo.predict(candidates)
        assert mu.shape == (50,)
        assert sigma.shape == (50,)

    def test_predict_before_fit(self, candidates):
        bo = BayesianOptimizer()
        with pytest.raises(RuntimeError, match="fit"):
            bo.predict(candidates)


class TestKernelTypes:
    def test_matern(self, reg_data, candidates):
        X, y = reg_data
        bo = BayesianOptimizer(BOConfig(kernel_type="matern"))
        bo.fit(X, y)
        result = bo.suggest(candidates, n=2)
        assert len(result) == 2

    def test_dotproduct(self, reg_data, candidates):
        X, y = reg_data
        bo = BayesianOptimizer(BOConfig(kernel_type="dotproduct"))
        bo.fit(X, y)
        result = bo.suggest(candidates, n=2)
        assert len(result) == 2


class TestGPInfo:
    def test_gp_info(self, reg_data):
        X, y = reg_data
        bo = BayesianOptimizer()
        bo.fit(X, y)
        info = bo.get_gp_info()
        assert "kernel" in info
        assert "n_train" in info

    def test_gp_info_before_fit(self):
        bo = BayesianOptimizer()
        info = bo.get_gp_info()
        assert info == {}


class TestSuggestDataFrame:
    def test_suggest_returns_dataframe(self, reg_data, candidates):
        X, y = reg_data
        df = pd.DataFrame(candidates, columns=["a", "b", "c"])
        bo = BayesianOptimizer()
        bo.fit(X, y)
        result = bo.suggest(df, n=5)
        assert isinstance(result, pd.DataFrame)
        assert "_acq_value" in result.columns
        assert "_rank" in result.columns

    def test_suggest_before_fit(self, candidates):
        bo = BayesianOptimizer()
        with pytest.raises(RuntimeError, match="fit"):
            bo.suggest(candidates)


class TestMaximinSelect:
    def test_basic(self):
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
        idx = BayesianOptimizer._maximin_select(X, 3)
        assert len(idx) == 3

    def test_n_ge_len(self):
        X = np.array([[0, 0], [1, 1]])
        idx = BayesianOptimizer._maximin_select(X, 5)
        assert len(idx) == 2


class TestMultiObjective:
    def test_parego(self, candidates):
        rng = np.random.RandomState(42)
        X = rng.randn(20, 3)
        Y = np.column_stack([X[:, 0] + rng.randn(20) * 0.1,
                              -X[:, 1] + rng.randn(20) * 0.1])
        cfg = BOConfig(
            multi_objective=True,
            objective_directions=["min", "min"],
        )
        bo = BayesianOptimizer(cfg)
        bo.fit(X, Y)
        result = bo.suggest(candidates, n=3)
        assert len(result) == 3
