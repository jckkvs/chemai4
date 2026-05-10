# -*- coding: utf-8 -*-
"""
tests/test_forward_inverse_combinations.py

順解析（Forward Analysis）× 逆解析（Inverse Analysis）の組み合わせテスト。

テスト方針:
    - 順解析: 複数モデル × 前処理 × CV 手法 × 特徴量生成の直積
    - 逆解析: 順解析結果のpredict_fnを使い、5手法 × 3目標モード × 制約組み合わせ
    - E2E: 順解析 → 逆解析 → 候補検証の一連パイプライン

テストID対応:
    T-FI01: 順解析モデル×CV組み合わせ
    T-FI02: 順解析モデル×前処理組み合わせ
    T-FI03: 逆解析手法×目標モード組み合わせ
    T-FI04: 逆解析手法×制約組み合わせ
    T-FI05: 順→逆E2Eパイプライン
    T-FI06: 固定変数・非アクティブ変数のエッジケース
    T-FI07: 高次元逆解析
    T-FI08: 逆解析ベイズ最適化×獲得関数組み合わせ
    T-FI09: 遺伝的アルゴリズムパラメータ組み合わせ
    T-FI10: ディリクレ分布パラメータ組み合わせ
    T-FI11: 制約型組み合わせ統合テスト
    T-FI12: スコア計算境界テスト
    T-FI13: 順解析→ベイズ最適化→制約フィルタE2E
    T-FI14: CVManager統合テスト
    T-FI15: AutoML高度テスト
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.models.automl import AutoMLEngine, AutoMLResult
from backend.models.factory import get_model, list_models, get_default_automl_models
from backend.optim.inverse_optimizer import (
    InverseConfig,
    InverseResult,
    run_inverse_optimization,
    _build_full_df,
    _score_predictions,
    _build_result_df,
    _optimize_random,
    _optimize_grid,
    _optimize_bayesian,
    _optimize_ga,
    _optimize_dirichlet,
    _sbx_crossover,
)
from backend.optim.constraints import (
    Constraint,
    RangeConstraint,
    SumConstraint,
    InequalityConstraint,
    AtLeastNConstraint,
    AtLeastOneConstraint,
    CustomConstraint,
    apply_constraints,
)
from backend.optim.bayesian_optimizer import BayesianOptimizer, BOConfig
from backend.models.cv_manager import (
    CVConfig,
    get_cv,
    list_cv_methods,
    run_cross_validation,
    WalkForwardSplit,
)
from sklearn.base import BaseEstimator


# ============================================================
# テストデータ生成
# ============================================================

def _make_regression_df(n: int = 80, n_features: int = 5, seed: int = 42) -> pd.DataFrame:
    """回帰テスト用DataFrame"""
    rng = np.random.RandomState(seed)
    data = {}
    for i in range(n_features):
        data[f"x{i}"] = rng.randn(n)
    data["target"] = sum(data[f"x{i}"] * (i + 1) for i in range(n_features)) + rng.randn(n) * 0.1
    return pd.DataFrame(data)


def _make_classification_df(n: int = 100, n_features: int = 4, seed: int = 42) -> pd.DataFrame:
    """分類テスト用DataFrame"""
    rng = np.random.RandomState(seed)
    data = {}
    for i in range(n_features):
        data[f"x{i}"] = rng.randn(n)
    score = sum(data[f"x{i}"] for i in range(n_features))
    data["target"] = (score > np.median(score)).astype(int)
    return pd.DataFrame(data)


def _simple_predict_fn(X: pd.DataFrame) -> np.ndarray:
    """y = -(x0-0.3)^2 - (x1-0.5)^2 : 最大化→(0.3, 0.5)が最適"""
    vals = X.values
    return -((vals[:, 0] - 0.3) ** 2) - ((vals[:, 1] - 0.5) ** 2)


def _linear_predict_fn(X: pd.DataFrame) -> np.ndarray:
    """y = 2*x0 + 3*x1 - x2 : 線形モデル"""
    vals = X.values
    result = np.zeros(len(vals))
    if vals.shape[1] >= 1:
        result += 2.0 * vals[:, 0]
    if vals.shape[1] >= 2:
        result += 3.0 * vals[:, 1]
    if vals.shape[1] >= 3:
        result -= 1.0 * vals[:, 2]
    return result


def _composition_predict_fn(X: pd.DataFrame) -> np.ndarray:
    """組成系: y = 0.5A + 0.3B + 0.2C"""
    vals = X.values
    return vals[:, 0] * 0.5 + vals[:, 1] * 0.3 + vals[:, 2] * 0.2


# ============================================================
# T-FI01: 順解析モデル × CV 組み合わせ
# ============================================================

class TestForwardModelCVCombinations:
    """順解析: モデル × CV手法のすべての組み合わせで実行可能か"""

    MODELS = ["ridge", "lasso", "rf", "et", "knn", "dt", "svr_linear", "hgbm"]
    CV_KEYS = ["kfold", "repeated_kfold", "shuffle_split"]

    @pytest.fixture
    def reg_df(self):
        return _make_regression_df(n=60, n_features=3)

    @pytest.mark.parametrize("model_key", MODELS)
    @pytest.mark.parametrize("cv_key", CV_KEYS)
    def test_model_cv_combination(self, reg_df, model_key, cv_key):
        """各モデル × CV組み合わせが正常に完走すること"""
        engine = AutoMLEngine(
            task="regression",
            cv_folds=3,
            cv_key=cv_key,
            model_keys=[model_key],
        )
        result = engine.run(reg_df, target_col="target")
        assert isinstance(result, AutoMLResult)
        assert result.best_model_key == model_key
        assert isinstance(result.best_score, float)
        assert result.best_pipeline is not None

    @pytest.mark.parametrize("model_key", ["dt_c", "rf_c", "knn_c", "hgbm_c"])
    def test_classification_models(self, model_key):
        """分類モデルも正常に完走すること"""
        df = _make_classification_df(n=60, n_features=3)
        engine = AutoMLEngine(
            task="classification",
            cv_folds=2,
            model_keys=[model_key],
        )
        result = engine.run(df, target_col="target")
        assert result.task == "classification"


# ============================================================
# T-FI02: 順解析モデル × 前処理組み合わせ
# ============================================================

class TestForwardModelPreprocessCombinations:
    """順解析: モデル × 前処理設定の組み合わせ"""

    MODELS = ["ridge", "rf"]

    @pytest.fixture
    def df_with_outliers(self):
        rng = np.random.RandomState(42)
        n = 60
        data = {
            "x0": rng.randn(n),
            "x1": rng.exponential(2, n),
            "target": rng.randn(n),
        }
        data["x0"][0] = 100.0
        data["x1"][1] = 500.0
        return pd.DataFrame(data)

    @pytest.mark.parametrize("model_key", MODELS)
    def test_model_with_outlier_data(self, df_with_outliers, model_key):
        """外れ値を含むデータでも正常完走"""
        engine = AutoMLEngine(
            task="regression",
            cv_folds=2,
            model_keys=[model_key],
        )
        result = engine.run(df_with_outliers, target_col="target")
        assert isinstance(result, AutoMLResult)
        assert result.best_pipeline is not None


# ============================================================
# T-FI03: 逆解析手法 × 目標モード組み合わせ
# ============================================================

class TestInverseMethodTargetCombinations:
    """逆解析: 5手法 × 3目標モードの全組み合わせ"""

    CONSTRAINTS_2D = {
        "x0": {"min": 0.0, "max": 1.0, "active": True},
        "x1": {"min": 0.0, "max": 1.0, "active": True},
    }
    CONSTRAINTS_3D = {
        "A": {"min": 0.0, "max": 0.8, "active": True},
        "B": {"min": 0.0, "max": 0.8, "active": True},
        "C": {"min": 0.0, "max": 0.8, "active": True},
    }

    def _get_method_params(self, method: str) -> dict:
        base = {"seed": 42}
        if method == "random":
            return {**base, "n_samples": 200}
        elif method == "grid":
            return {**base, "n_points": 8}
        elif method == "bayesian":
            return {**base, "n_trials": 15, "acq_func": "EI"}
        elif method == "ga":
            return {**base, "pop_size": 10, "n_generations": 10,
                    "mutation_rate": 0.1, "crossover_rate": 0.8}
        elif method == "dirichlet":
            return {**base, "n_samples_per_round": 100, "n_rounds": 5,
                    "top_k": 20, "concentration": 10.0, "total_sum": 1.0}
        return base

    @pytest.mark.parametrize("method", ["random", "grid", "bayesian", "ga"])
    @pytest.mark.parametrize("target_mode", ["maximize", "minimize", "range"])
    def test_method_target_combination_2d(self, method, target_mode):
        """2D: 各手法×目標モード"""
        config = InverseConfig(
            method=method,
            target_mode=target_mode,
            target_min=-0.1 if target_mode == "range" else None,
            target_max=0.0 if target_mode == "range" else None,
            constraints=self.CONSTRAINTS_2D,
            method_params=self._get_method_params(method),
        )
        result = run_inverse_optimization(
            _simple_predict_fn, ["x0", "x1"], config,
        )
        assert isinstance(result, InverseResult)
        assert result.method == method
        assert len(result.candidates) > 0
        assert result.n_evaluated > 0
        assert result.elapsed_seconds >= 0

    @pytest.mark.parametrize("target_mode", ["maximize", "minimize", "range"])
    def test_dirichlet_target_modes(self, target_mode):
        """ディリクレ: 3目標モードの動作"""
        config = InverseConfig(
            method="dirichlet",
            target_mode=target_mode,
            target_min=0.1 if target_mode == "range" else None,
            target_max=0.3 if target_mode == "range" else None,
            constraints=self.CONSTRAINTS_3D,
            method_params=self._get_method_params("dirichlet"),
        )
        result = run_inverse_optimization(
            _composition_predict_fn, ["A", "B", "C"], config,
        )
        assert result.method == "dirichlet"
        assert len(result.candidates) > 0


# ============================================================
# T-FI04: 逆解析手法 × 制約組み合わせ
# ============================================================

class TestInverseMethodConstraintCombinations:
    """逆解析: 手法 × 制約タイプの組み合わせ"""

    def test_random_with_fixed_vars(self):
        """ランダム: 一部変数を固定"""
        config = InverseConfig(
            method="random",
            target_mode="maximize",
            constraints={
                "x0": {"min": 0.0, "max": 1.0, "active": True},
                "x1": {"fixed": True, "fixed_val": 0.5, "active": True},
            },
            method_params={"n_samples": 200, "seed": 42},
        )
        result = run_inverse_optimization(
            _simple_predict_fn, ["x0", "x1"], config,
        )
        assert len(result.candidates) > 0

    def test_grid_with_inactive_vars(self):
        """グリッド: 非アクティブ変数"""
        config = InverseConfig(
            method="grid",
            target_mode="maximize",
            constraints={
                "x0": {"min": 0.0, "max": 1.0, "active": True},
                "x1": {"min": 0.0, "max": 1.0, "active": False},
            },
            method_params={"n_points": 10},
        )
        result = run_inverse_optimization(
            _simple_predict_fn, ["x0", "x1"], config,
        )
        assert result.n_evaluated == 10

    def test_bayesian_narrow_bounds(self):
        """ベイズ: 非常に狭い探索範囲"""
        config = InverseConfig(
            method="bayesian",
            target_mode="maximize",
            constraints={
                "x0": {"min": 0.25, "max": 0.35, "active": True},
                "x1": {"min": 0.45, "max": 0.55, "active": True},
            },
            method_params={"n_trials": 15, "seed": 42, "acq_func": "EI"},
        )
        result = run_inverse_optimization(
            _simple_predict_fn, ["x0", "x1"], config,
        )
        best = result.candidates.iloc[0]
        assert 0.25 <= best["x0"] <= 0.35
        assert 0.45 <= best["x1"] <= 0.55

    @pytest.mark.parametrize("method", ["random", "ga"])
    def test_method_with_3_vars_1_fixed(self, method):
        """3変数のうち1つを固定"""
        config = InverseConfig(
            method=method,
            target_mode="maximize",
            constraints={
                "x0": {"min": 0.0, "max": 1.0, "active": True},
                "x1": {"min": 0.0, "max": 1.0, "active": True},
                "x2": {"fixed": True, "fixed_val": 0.0, "active": True},
            },
            method_params={"n_samples": 100, "seed": 42} if method == "random"
            else {"pop_size": 10, "n_generations": 5, "seed": 42},
        )
        result = run_inverse_optimization(
            _linear_predict_fn, ["x0", "x1", "x2"], config,
        )
        assert result.n_evaluated > 0

    def test_dirichlet_with_tight_bounds(self):
        """ディリクレ: 厳しいbounds"""
        config = InverseConfig(
            method="dirichlet",
            target_mode="maximize",
            constraints={
                "A": {"min": 0.3, "max": 0.4, "active": True},
                "B": {"min": 0.3, "max": 0.4, "active": True},
                "C": {"min": 0.2, "max": 0.3, "active": True},
            },
            method_params={
                "n_samples_per_round": 100, "n_rounds": 3,
                "top_k": 10, "concentration": 5.0,
                "total_sum": 1.0, "seed": 42,
            },
        )
        result = run_inverse_optimization(
            _composition_predict_fn, ["A", "B", "C"], config,
        )
        assert result.n_evaluated > 0


# ============================================================
# T-FI05: 順解析 → 逆解析 E2Eパイプライン
# ============================================================

class TestForwardToInverseE2E:
    """順解析の学習結果を逆解析の予測関数として使うE2Eテスト"""

    @pytest.fixture
    def trained_model(self):
        df = _make_regression_df(n=60, n_features=3, seed=42)
        engine = AutoMLEngine(
            task="regression", cv_folds=2, model_keys=["ridge"],
        )
        result = engine.run(df, target_col="target")
        feature_names = [c for c in df.columns if c != "target"]

        def predict_fn(X: pd.DataFrame) -> np.ndarray:
            return result.best_pipeline.predict(X[feature_names])

        return predict_fn, feature_names, result

    @pytest.mark.parametrize("method", ["random", "grid", "bayesian", "ga"])
    def test_e2e_various_methods(self, trained_model, method):
        """順解析モデル → 各逆解析手法"""
        predict_fn, feature_names, _ = trained_model
        constraints = {
            col: {"min": -3.0, "max": 3.0, "active": True}
            for col in feature_names
        }

        params = {"seed": 42}
        if method == "random":
            params["n_samples"] = 100
        elif method == "grid":
            params["n_points"] = 5
        elif method == "bayesian":
            params["n_trials"] = 15
        elif method == "ga":
            params.update({"pop_size": 10, "n_generations": 5})

        config = InverseConfig(
            method=method,
            target_mode="maximize",
            constraints=constraints,
            method_params=params,
        )
        result = run_inverse_optimization(
            predict_fn, feature_names, config,
        )
        assert isinstance(result, InverseResult)
        assert len(result.candidates) > 0
        assert "predicted" in result.candidates.columns
        assert "score" in result.candidates.columns

    def test_e2e_multiple_models_then_inverse(self):
        """複数モデル比較 → 最良モデルで逆解析"""
        df = _make_regression_df(n=80, n_features=3, seed=42)
        engine = AutoMLEngine(
            task="regression", cv_folds=2,
            model_keys=["ridge", "rf", "dt"],
        )
        result = engine.run(df, target_col="target")
        feature_names = [c for c in df.columns if c != "target"]

        def predict_fn(X: pd.DataFrame) -> np.ndarray:
            return result.best_pipeline.predict(X[feature_names])

        config = InverseConfig(
            method="random",
            target_mode="maximize",
            constraints={c: {"min": -3.0, "max": 3.0, "active": True}
                         for c in feature_names},
            method_params={"n_samples": 100, "seed": 42},
        )
        inv_result = run_inverse_optimization(predict_fn, feature_names, config)
        assert len(inv_result.candidates) > 0
        assert not np.isnan(inv_result.best_predicted)


# ============================================================
# T-FI06: 固定変数・非アクティブ変数のエッジケース
# ============================================================

class TestEdgeCasesDetailed:

    def test_all_but_one_fixed(self):
        """1変数のみ探索"""
        config = InverseConfig(
            method="random",
            target_mode="maximize",
            constraints={
                "x0": {"min": 0.0, "max": 1.0, "active": True},
                "x1": {"fixed": True, "fixed_val": 0.5, "active": True},
            },
            method_params={"n_samples": 50, "seed": 42},
        )
        result = run_inverse_optimization(_simple_predict_fn, ["x0", "x1"], config)
        assert result.n_evaluated == 50

    def test_missing_constraints_defaults(self):
        """制約未指定→デフォルト(0,1)"""
        config = InverseConfig(
            method="random",
            target_mode="maximize",
            constraints={},
            method_params={"n_samples": 50, "seed": 42},
        )
        result = run_inverse_optimization(_simple_predict_fn, ["x0", "x1"], config)
        assert result.n_evaluated == 50

    def test_progress_callback_called(self):
        """進捗コールバック呼び出し確認"""
        calls = []

        def cb(step, total, msg):
            calls.append((step, total, msg))

        config = InverseConfig(
            method="random",
            target_mode="maximize",
            constraints={
                "x0": {"min": 0.0, "max": 1.0, "active": True},
                "x1": {"min": 0.0, "max": 1.0, "active": True},
            },
            method_params={"n_samples": 50, "seed": 42},
        )
        run_inverse_optimization(
            _simple_predict_fn, ["x0", "x1"], config,
            progress_callback=cb,
        )
        assert len(calls) >= 1


# ============================================================
# T-FI07: 高次元逆解析
# ============================================================

class TestHighDimensionalInverse:

    def _make_high_dim_fn(self, n_dims: int):
        weights = np.arange(1, n_dims + 1, dtype=float)

        def predict_fn(X: pd.DataFrame) -> np.ndarray:
            return X.values @ weights

        return predict_fn

    @pytest.mark.parametrize("method", ["random", "ga"])
    def test_10d_optimization(self, method):
        n_dims = 10
        feature_names = [f"x{i}" for i in range(n_dims)]
        predict_fn = self._make_high_dim_fn(n_dims)
        constraints = {f"x{i}": {"min": 0.0, "max": 1.0, "active": True} for i in range(n_dims)}

        params = {"seed": 42}
        if method == "random":
            params["n_samples"] = 200
        elif method == "ga":
            params.update({"pop_size": 20, "n_generations": 10})

        config = InverseConfig(
            method=method, target_mode="maximize",
            constraints=constraints, method_params=params,
        )
        result = run_inverse_optimization(predict_fn, feature_names, config)
        assert result.n_evaluated > 0
        assert len(result.candidates) > 0


# ============================================================
# T-FI08: ベイズ最適化 × 獲得関数組み合わせ
# ============================================================

class TestBayesianAcquisitionCombinations:

    @pytest.mark.parametrize("acq_func", ["EI", "PI", "UCB"])
    @pytest.mark.parametrize("objective", ["maximize", "minimize"])
    def test_acq_objective_combination(self, acq_func, objective):
        config = InverseConfig(
            method="bayesian",
            target_mode=objective,
            constraints={
                "x0": {"min": 0.0, "max": 1.0, "active": True},
                "x1": {"min": 0.0, "max": 1.0, "active": True},
            },
            method_params={"n_trials": 12, "seed": 42, "acq_func": acq_func},
        )
        result = run_inverse_optimization(_simple_predict_fn, ["x0", "x1"], config)
        assert result.method == "bayesian"
        assert result.n_evaluated >= 12

    def test_kernel_types(self):
        rng = np.random.RandomState(42)
        X = rng.uniform(0, 1, size=(20, 2))
        y = (X ** 2).sum(axis=1)
        X_cand = rng.uniform(0, 1, size=(50, 2))

        for kernel_type in ["default", "matern", "dotproduct"]:
            bo = BayesianOptimizer(BOConfig(
                objective="minimize", kernel_type=kernel_type,
                batch_strategy="single", n_candidates=3,
            ))
            bo.fit(X, y)
            result = bo.suggest(X_cand, n=3)
            assert len(result) == 3

    def test_batch_strategies(self):
        rng = np.random.RandomState(42)
        X = rng.uniform(0, 1, size=(20, 2))
        y = (X ** 2).sum(axis=1)
        X_cand = rng.uniform(0, 1, size=(50, 2))

        for strategy in ["single", "kriging_believer", "doe_then_bo", "bo_then_doe"]:
            bo = BayesianOptimizer(BOConfig(
                objective="minimize", batch_strategy=strategy, n_candidates=3,
            ))
            bo.fit(X, y)
            result = bo.suggest(X_cand, n=3)
            assert len(result) == 3

    def test_ptr_target_range(self):
        rng = np.random.RandomState(42)
        X = rng.uniform(0, 10, size=(20, 2))
        y = X.sum(axis=1)
        X_cand = rng.uniform(0, 10, size=(50, 2))

        bo = BayesianOptimizer(BOConfig(
            acquisition="ptr", target_lo=8.0, target_hi=12.0, n_candidates=3,
        ))
        bo.fit(X, y)
        result = bo.suggest(X_cand, n=3)
        assert len(result) == 3

    def test_multi_objective_parego(self):
        rng = np.random.RandomState(42)
        X = rng.uniform(0, 10, size=(25, 2))
        Y = np.column_stack([(X[:, 0] - 5) ** 2, (X[:, 1] - 3) ** 2])
        X_cand = rng.uniform(0, 10, size=(50, 2))

        bo = BayesianOptimizer(BOConfig(
            multi_objective=True, objective_directions=["min", "min"], n_candidates=5,
        ))
        bo.fit(X, Y)
        result = bo.suggest(X_cand, n=5)
        assert len(result) == 5

    def test_multi_objective_max_direction(self):
        rng = np.random.RandomState(42)
        X = rng.uniform(0, 10, size=(25, 2))
        Y = np.column_stack([X[:, 0], X[:, 1]])
        X_cand = rng.uniform(0, 10, size=(50, 2))

        bo = BayesianOptimizer(BOConfig(
            multi_objective=True, objective_directions=["max", "max"], n_candidates=3,
        ))
        bo.fit(X, Y)
        result = bo.suggest(X_cand, n=3)
        assert len(result) == 3


# ============================================================
# T-FI09: GAパラメータ組み合わせ
# ============================================================

class TestGAParameterCombinations:

    @pytest.mark.parametrize("pop_size", [10, 30])
    @pytest.mark.parametrize("mutation_rate", [0.01, 0.2])
    @pytest.mark.parametrize("crossover_rate", [0.5, 0.9])
    def test_ga_params(self, pop_size, mutation_rate, crossover_rate):
        config = InverseConfig(
            method="ga", target_mode="maximize",
            constraints={
                "x0": {"min": 0.0, "max": 1.0, "active": True},
                "x1": {"min": 0.0, "max": 1.0, "active": True},
            },
            method_params={
                "pop_size": pop_size, "n_generations": 5,
                "mutation_rate": mutation_rate, "crossover_rate": crossover_rate,
                "seed": 42,
            },
        )
        result = run_inverse_optimization(_simple_predict_fn, ["x0", "x1"], config)
        assert result.n_evaluated >= pop_size * 5

    def test_sbx_crossover_direct(self):
        rng = np.random.RandomState(42)
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        p1 = np.array([0.3, 0.7])
        p2 = np.array([0.8, 0.2])

        c1, c2 = _sbx_crossover(p1, p2, lo, hi, rng, eta=20.0)
        assert np.all(c1 >= lo) and np.all(c1 <= hi)
        assert np.all(c2 >= lo) and np.all(c2 <= hi)

    def test_sbx_crossover_identical_parents(self):
        rng = np.random.RandomState(42)
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        p = np.array([0.5, 0.5])

        c1, c2 = _sbx_crossover(p, p.copy(), lo, hi, rng)
        np.testing.assert_allclose(c1, p, atol=1e-10)
        np.testing.assert_allclose(c2, p, atol=1e-10)


# ============================================================
# T-FI10: ディリクレ分布パラメータ組み合わせ
# ============================================================

class TestDirichletParameterCombinations:

    @pytest.mark.parametrize("concentration", [1.0, 5.0, 20.0])
    @pytest.mark.parametrize("total_sum", [1.0, 100.0])
    def test_concentration_total_sum(self, concentration, total_sum):
        constraints = {
            "A": {"min": 0.0, "max": 0.8 * total_sum, "active": True},
            "B": {"min": 0.0, "max": 0.8 * total_sum, "active": True},
            "C": {"min": 0.0, "max": 0.8 * total_sum, "active": True},
        }

        def predict_fn(X):
            vals = X.values / total_sum
            return vals[:, 0] * 0.5 + vals[:, 1] * 0.3 + vals[:, 2] * 0.2

        config = InverseConfig(
            method="dirichlet", target_mode="maximize",
            constraints=constraints,
            method_params={
                "n_samples_per_round": 50, "n_rounds": 3,
                "top_k": 10, "concentration": concentration,
                "total_sum": total_sum, "seed": 42,
            },
        )
        result = run_inverse_optimization(predict_fn, ["A", "B", "C"], config)
        assert result.n_evaluated > 0


# ============================================================
# T-FI11: 制約型組み合わせ統合テスト
# ============================================================

class TestConstraintCombinations:

    def _make_df(self, n=100):
        rng = np.random.RandomState(42)
        return pd.DataFrame({
            "A": rng.uniform(0, 50, n),
            "B": rng.uniform(0, 50, n),
            "C": rng.uniform(0, 50, n),
        })

    def test_range_plus_sum(self):
        df = self._make_df()
        constraints = [
            RangeConstraint("A", lo=10, hi=40),
            SumConstraint(columns=["A", "B"], target=50, tolerance=5.0),
        ]
        filtered, report = apply_constraints(df, constraints)
        assert report["after"] <= report["before"]
        assert len(report["details"]) == 2

    def test_range_plus_inequality_plus_atleast(self):
        df = self._make_df()
        constraints = [
            RangeConstraint("A", lo=5, hi=45),
            InequalityConstraint({"A": 1.0, "B": -1.0}, rhs=0, operator="ge"),
            AtLeastNConstraint(columns=["A", "B", "C"], min_count=2, threshold=10),
        ]
        filtered, report = apply_constraints(df, constraints)
        for _, row in filtered.iterrows():
            assert 5 <= row["A"] <= 45
            assert row["A"] >= row["B"]
            count_above = sum(1 for v in [row["A"], row["B"], row["C"]] if v > 10)
            assert count_above >= 2

    def test_all_constraint_types(self):
        df = self._make_df(200)
        constraints = [
            RangeConstraint("A", lo=5, hi=45),
            SumConstraint(columns=["A", "B", "C"], target=80, tolerance=20),
            InequalityConstraint({"A": 1.0}, rhs=10, operator="ge"),
            AtLeastNConstraint(columns=["A", "B"], min_count=1, threshold=15),
            CustomConstraint("A + B > 30"),
        ]
        filtered, report = apply_constraints(df, constraints)
        assert report["after"] <= report["before"]
        assert len(report["details"]) == 5

    def test_constraint_is_satisfied_single_row(self):
        row = pd.Series({"A": 20.0, "B": 30.0, "C": 10.0})
        assert RangeConstraint("A", lo=10, hi=30).is_satisfied(row)
        assert not RangeConstraint("A", lo=25, hi=30).is_satisfied(row)
        assert SumConstraint(["A", "B"], target=50).is_satisfied(row)
        assert not SumConstraint(["A", "B"], target=40).is_satisfied(row)

    def test_constraint_describe(self):
        assert "≤" in RangeConstraint("A", lo=5, hi=10).describe()
        assert "A + B" in SumConstraint(["A", "B"], target=100).describe()
        assert "A" in InequalityConstraint({"A": 1.0}, rhs=10, operator="ge").describe()
        assert "少なくとも" in AtLeastNConstraint(["A"], min_count=1).describe()
        assert "カスタム" in CustomConstraint("A > 0").describe()

    def test_custom_constraint_error_returns_false(self):
        c = CustomConstraint("undefined_variable > 0")
        row = pd.Series({"A": 10.0})
        assert not c.is_satisfied(row)

    def test_inequality_operators(self):
        df = pd.DataFrame({"A": [10, 20, 30]})
        assert InequalityConstraint({"A": 1.0}, rhs=25, operator="le").mask(df).sum() == 2
        assert InequalityConstraint({"A": 1.0}, rhs=25, operator="ge").mask(df).sum() == 1
        assert InequalityConstraint({"A": 1.0}, rhs=20, operator="lt").mask(df).sum() == 1
        assert InequalityConstraint({"A": 1.0}, rhs=20, operator="gt").mask(df).sum() == 1

    def test_atleast_alias(self):
        assert AtLeastOneConstraint is AtLeastNConstraint


# ============================================================
# T-FI12: スコア計算境界テスト
# ============================================================

class TestScoreCalculation:

    def test_maximize_score(self):
        config = InverseConfig(target_mode="maximize")
        preds = np.array([1.0, 2.0, 3.0])
        scores = _score_predictions(preds, config)
        np.testing.assert_array_equal(scores, preds)

    def test_minimize_score(self):
        config = InverseConfig(target_mode="minimize")
        preds = np.array([1.0, 2.0, 3.0])
        scores = _score_predictions(preds, config)
        np.testing.assert_array_equal(scores, -preds)

    def test_range_score_center(self):
        config = InverseConfig(target_mode="range", target_min=0.0, target_max=10.0)
        preds = np.array([5.0, 0.0, 10.0, 50.0])
        scores = _score_predictions(preds, config)
        assert scores[0] > scores[1]
        assert scores[0] > scores[3]

    def test_range_score_gaussian_shape(self):
        config = InverseConfig(target_mode="range", target_min=4.0, target_max=6.0)
        preds = np.linspace(0, 10, 100)
        scores = _score_predictions(preds, config)
        center_idx = 50
        assert scores[center_idx] == scores.max()


# ============================================================
# T-FI13: 順解析→BO→制約フィルタ E2E
# ============================================================

class TestFullPipelineE2E:

    def test_full_pipeline_with_constraints(self):
        df = _make_regression_df(n=60, n_features=3, seed=42)
        engine = AutoMLEngine(task="regression", cv_folds=2, model_keys=["ridge"])
        result = engine.run(df, target_col="target")
        feature_names = [c for c in df.columns if c != "target"]

        def predict_fn(X):
            return result.best_pipeline.predict(X[feature_names])

        config = InverseConfig(
            method="random", target_mode="maximize",
            constraints={c: {"min": -2.0, "max": 2.0, "active": True} for c in feature_names},
            method_params={"n_samples": 200, "seed": 42},
        )
        inv_result = run_inverse_optimization(predict_fn, feature_names, config)

        # 制約フィルタ
        df_cands = inv_result.candidates.copy()
        cand_cols = [c for c in feature_names if c in df_cands.columns]
        if cand_cols:
            constraints = [RangeConstraint(cand_cols[0], lo=-1.0, hi=1.0)]
            if set(cand_cols).issubset(df_cands.columns):
                cols_for_filter = [c for c in cand_cols + ["predicted", "score"] if c in df_cands.columns]
                filtered, report = apply_constraints(df_cands[cols_for_filter], constraints)
                assert report["after"] <= report["before"]

    def test_build_full_df(self):
        X_search = np.array([[0.1, 0.2], [0.3, 0.4]])
        df = _build_full_df(X_search, ["x0", "x1"], {"x2": 0.5}, ["x0", "x1", "x2"])
        assert list(df.columns) == ["x0", "x1", "x2"]
        assert len(df) == 2
        np.testing.assert_array_equal(df["x2"].values, [0.5, 0.5])

    def test_build_full_df_missing_col_fills_zero(self):
        X_search = np.array([[0.1], [0.2]])
        df = _build_full_df(X_search, ["x0"], {}, ["x0", "x1"])
        np.testing.assert_array_equal(df["x1"].values, [0.0, 0.0])

    def test_build_result_df(self):
        candidates = np.array([[0.3, 0.5], [0.1, 0.9]])
        result_df = _build_result_df(
            candidates, ["x0", "x1"], {}, ["x0", "x1"],
            _simple_predict_fn, InverseConfig(target_mode="maximize"), top_n=2,
        )
        assert "rank" in result_df.columns
        assert "predicted" in result_df.columns
        assert "score" in result_df.columns
        assert result_df["rank"].iloc[0] == 1
        assert result_df["score"].iloc[0] >= result_df["score"].iloc[1]


# ============================================================
# T-FI14: CVManager統合テスト
# ============================================================

class TestCVManagerCombinations:
    """CVManagerの全手法組み合わせテスト"""

    @pytest.fixture
    def reg_data(self):
        rng = np.random.RandomState(42)
        X = rng.randn(60, 3)
        y = X[:, 0] * 2 + rng.randn(60) * 0.1
        return X, y

    @pytest.mark.parametrize("cv_key", [
        "kfold", "stratified_kfold", "loo", "lpo",
        "repeated_kfold", "shuffle_split",
        "timeseries", "walk_forward",
    ])
    def test_get_cv_all_types(self, cv_key):
        """全CV手法がインスタンス化できること"""
        extra = {}
        if cv_key == "lpo":
            extra["p"] = 2
        config = CVConfig(cv_key=cv_key, n_splits=3, extra_params=extra)
        cv = get_cv(config)
        assert cv is not None

    @pytest.mark.parametrize("cv_key", ["kfold", "shuffle_split", "timeseries"])
    def test_run_cv_with_ridge(self, reg_data, cv_key):
        """Ridge × 各CV手法でCVが完走すること"""
        X, y = reg_data
        from sklearn.linear_model import Ridge
        model = Ridge()
        config = CVConfig(cv_key=cv_key, n_splits=3)
        result = run_cross_validation(
            model, X, y, config, scoring="neg_mean_squared_error",
        )
        assert "mean_test_score" in result
        assert isinstance(result["mean_test_score"], float)

    def test_list_cv_methods_regression(self):
        methods = list_cv_methods(task="regression")
        keys = [m["key"] for m in methods]
        assert "kfold" in keys
        assert "timeseries" in keys

    def test_list_cv_methods_classification(self):
        methods = list_cv_methods(task="classification")
        keys = [m["key"] for m in methods]
        assert "stratified_kfold" in keys

    def test_list_cv_methods_groups_filter(self):
        methods = list_cv_methods(requires_groups=True)
        for m in methods:
            assert m["requires_groups"]

    def test_walk_forward_split(self):
        """WalkForwardSplit直接テスト"""
        wf = WalkForwardSplit(n_splits=3, gap=0)
        X = np.arange(30).reshape(-1, 1)
        splits = list(wf.split(X))
        assert len(splits) == 3
        for train, test in splits:
            assert len(test) > 0
            assert train[-1] < test[0]  # 学習が常にテストの前

    def test_walk_forward_with_gap(self):
        wf = WalkForwardSplit(n_splits=3, gap=2)
        X = np.arange(40).reshape(-1, 1)
        splits = list(wf.split(X))
        assert len(splits) >= 1
        for train, test in splits:
            assert test[0] - train[-1] >= 2

    def test_walk_forward_too_small_raises(self):
        wf = WalkForwardSplit(n_splits=10)
        X = np.arange(5).reshape(-1, 1)
        with pytest.raises(ValueError, match="データ数"):
            list(wf.split(X))

    def test_walk_forward_get_n_splits(self):
        wf = WalkForwardSplit(n_splits=7)
        assert wf.get_n_splits() == 7

    def test_cv_config_with_unknown_key_uses_get_cv_class(self):
        """_CV_REGISTRYにないキー → _get_cv_class経由でロード"""
        config = CVConfig(cv_key="KFold", n_splits=3)
        cv = get_cv(config)
        assert cv is not None

    def test_cv_invalid_key_raises(self):
        from backend.models.cv_manager import _get_cv_class
        with pytest.raises(ValueError, match="ロードできませんでした"):
            _get_cv_class("nonexistent_cv_class")


# ============================================================
# T-FI15: AutoML高度テスト
# ============================================================

class TestAutoMLAdvanced:

    def test_auto_task_detection_regression(self):
        df = _make_regression_df(n=60, n_features=3)
        engine = AutoMLEngine(task="auto", cv_folds=2, model_keys=["ridge"])
        result = engine.run(df, target_col="target")
        assert result.task == "regression"

    def test_auto_task_detection_classification(self):
        df = _make_classification_df(n=60, n_features=3)
        engine = AutoMLEngine(task="auto", cv_folds=2, model_keys=["dt_c"])
        result = engine.run(df, target_col="target")
        assert result.task == "classification"

    def test_model_details_populated(self):
        df = _make_regression_df(n=60, n_features=3)
        engine = AutoMLEngine(
            task="regression", cv_folds=2,
            model_keys=["ridge", "lasso"],
        )
        result = engine.run(df, target_col="target")
        for key in ["ridge", "lasso"]:
            assert key in result.model_details
            detail = result.model_details[key]
            assert "mean" in detail
            assert "std" in detail
            assert "fit_time" in detail

    def test_elapsed_time(self):
        df = _make_regression_df(n=40, n_features=2)
        engine = AutoMLEngine(
            task="regression", cv_folds=2, model_keys=["ridge"],
        )
        result = engine.run(df, target_col="target")
        assert result.elapsed_seconds > 0

    def test_progress_callback(self):
        calls = []
        df = _make_regression_df(n=40, n_features=2)
        engine = AutoMLEngine(
            task="regression", cv_folds=2, model_keys=["ridge"],
            progress_callback=lambda s, t, m: calls.append((s, t, m)),
        )
        engine.run(df, target_col="target")
        assert len(calls) >= 4

    def test_oof_predictions(self):
        df = _make_regression_df(n=60, n_features=3)
        engine = AutoMLEngine(task="regression", cv_folds=2, model_keys=["ridge"])
        result = engine.run(df, target_col="target")
        if result.oof_predictions is not None:
            assert len(result.oof_predictions) == len(df)

    def test_processed_X(self):
        df = _make_regression_df(n=60, n_features=3)
        engine = AutoMLEngine(task="regression", cv_folds=2, model_keys=["ridge"])
        result = engine.run(df, target_col="target")
        assert result.processed_X is not None
        assert result.processed_X.shape[0] == len(df)

    def test_too_few_rows_raises(self):
        df = _make_regression_df(n=5, n_features=2)
        engine = AutoMLEngine(task="regression", cv_folds=2, model_keys=["ridge"])
        with pytest.raises(ValueError, match="データが少なすぎ"):
            engine.run(df, target_col="target")

    def test_missing_target_col_raises(self):
        df = _make_regression_df(n=40, n_features=2)
        engine = AutoMLEngine(task="regression", cv_folds=2, model_keys=["ridge"])
        with pytest.raises(ValueError, match="存在しません"):
            engine.run(df, target_col="nonexistent_column")

    def test_run_multi_feature_sets(self):
        df = _make_regression_df(n=60, n_features=5)
        engine = AutoMLEngine(
            task="regression", cv_folds=2, model_keys=["ridge"],
        )
        feature_sets = [
            {"id": "set1", "name": "First3", "descriptors": ["x0", "x1", "x2"], "pipeline": "normal"},
            {"id": "set2", "name": "Last3", "descriptors": ["x2", "x3", "x4"], "pipeline": "normal"},
        ]
        results = engine.run_multi_feature_sets(
            df, target_col="target", feature_sets=feature_sets,
        )
        assert len(results) == 2
        for r in results:
            assert isinstance(r, AutoMLResult)


# ============================================================
# BO 追加カバレッジ
# ============================================================

class TestBOAdditionalCoverage:

    def test_get_gp_info_not_fitted(self):
        bo = BayesianOptimizer()
        assert bo.get_gp_info() == {}

    def test_get_gp_info_multi_objective(self):
        rng = np.random.RandomState(42)
        X = rng.uniform(0, 1, (20, 2))
        Y = np.column_stack([X[:, 0], X[:, 1]])
        bo = BayesianOptimizer(BOConfig(multi_objective=True))
        bo.fit(X, Y)
        info = bo.get_gp_info()
        assert info["n_objectives"] == 2

    def test_predict_multi_objective(self):
        rng = np.random.RandomState(42)
        X = rng.uniform(0, 1, (20, 2))
        Y = np.column_stack([X[:, 0], X[:, 1]])
        bo = BayesianOptimizer(BOConfig(multi_objective=True))
        bo.fit(X, Y)
        mu, sigma = bo.predict(X[:3])
        assert mu.shape == (3, 2)
        assert sigma.shape == (3, 2)

    def test_maximin_select(self):
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
        idx = BayesianOptimizer._maximin_select(X, 3)
        assert len(idx) == 3

    def test_maximin_select_n_exceeds(self):
        X = np.array([[0, 0], [1, 1]])
        idx = BayesianOptimizer._maximin_select(X, 10)
        assert len(idx) == 2

    def test_unknown_acquisition_raises(self):
        rng = np.random.RandomState(42)
        X = rng.uniform(0, 1, (20, 2))
        y = X.sum(axis=1)
        bo = BayesianOptimizer(BOConfig(acquisition="unknown_acq"))
        bo.fit(X, y)
        with pytest.raises(ValueError, match="不明な獲得関数"):
            bo.suggest(rng.uniform(0, 1, (10, 2)), n=3)

    def test_ptr_missing_bounds_raises(self):
        rng = np.random.RandomState(42)
        X = rng.uniform(0, 1, (20, 2))
        y = X.sum(axis=1)
        bo = BayesianOptimizer(BOConfig(
            acquisition="ptr", target_lo=None, target_hi=None,
        ))
        bo.fit(X, y)
        with pytest.raises(ValueError, match="target_lo.*target_hi"):
            bo.suggest(rng.uniform(0, 1, (10, 2)), n=3)

    def test_not_fitted_suggest_raises(self):
        bo = BayesianOptimizer()
        with pytest.raises(RuntimeError, match="fit"):
            bo.suggest(np.array([[1, 2]]))

    def test_not_fitted_predict_raises(self):
        bo = BayesianOptimizer()
        with pytest.raises(RuntimeError, match="fit"):
            bo.predict(np.array([[1, 2]]))

    def test_suggest_with_dataframe(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.uniform(0, 1, (20, 2)), columns=["a", "b"])
        y = X["a"] + X["b"]
        X_cand = pd.DataFrame(rng.uniform(0, 1, (50, 2)), columns=["a", "b"])

        bo = BayesianOptimizer(BOConfig(n_candidates=3))
        bo.fit(X, y)
        result = bo.suggest(X_cand, n=3)
        assert isinstance(result, pd.DataFrame)
        assert "_acq_value" in result.columns
        assert "_rank" in result.columns


# ============================================================
# Factory網羅テスト: 全モデルfit/predict
# ============================================================

class TestFactoryComprehensive:

    @pytest.fixture
    def simple_data(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        y = X[:, 0] * 2 + X[:, 1] * 3 + rng.randn(40) * 0.1
        return X, y

    @pytest.mark.parametrize("key", [
        "linear", "ridge", "ridge_cv", "lasso", "lasso_cv",
        "elasticnet", "bayesian_ridge", "ard",
        "huber", "theilsen",
        "svr_rbf", "svr_linear", "knn", "dt", "rf", "et",
        "gbm", "hgbm", "adaboost", "bagging", "pls",
    ])
    def test_fit_predict_regression(self, simple_data, key):
        X, y = simple_data
        model = get_model(key, task="regression")
        model.fit(X, y)
        preds = model.predict(X[:5])
        assert preds.shape == (5,)
        assert not np.any(np.isnan(preds))

    @pytest.fixture
    def cls_data(self):
        rng = np.random.RandomState(42)
        X = rng.randn(40, 3)
        y = (X[:, 0] > 0).astype(int)
        return X, y

    @pytest.mark.parametrize("key", [
        "knn_c", "dt_c", "rf_c", "et_c",
        "gbm_c", "hgbm_c", "adaboost_c", "bagging_c",
        "gnb", "bnb",
    ])
    def test_fit_predict_classification(self, cls_data, key):
        X, y = cls_data
        model = get_model(key, task="classification")
        model.fit(X, y)
        preds = model.predict(X[:5])
        assert preds.shape == (5,)

    def test_ransac_regression(self, simple_data):
        X, y = simple_data
        y_outlier = y.copy()
        y_outlier[0] = 1000.0
        model = get_model("ransac", task="regression")
        model.fit(X, y_outlier)
        preds = model.predict(X[:5])
        assert preds.shape == (5,)

    def test_list_models_count(self):
        assert len(list_models(task="regression")) >= 15
        assert len(list_models(task="classification")) >= 8

    def test_default_automl_models(self):
        defaults = get_default_automl_models(task="regression")
        assert len(defaults) >= 3
        for key in defaults:
            model = get_model(key, task="regression")
            assert isinstance(model, BaseEstimator)
