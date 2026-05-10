# -*- coding: utf-8 -*-
"""
tests/test_inverse_optimizer.py

逆解析エンジンのテスト — 5手法(random/grid/bayesian/ga/dirichlet)
"""
import numpy as np
import pandas as pd
import pytest

from backend.optim.inverse_optimizer import (
    InverseConfig,
    InverseResult,
    run_inverse_optimization,
)


# ── テスト用の予測関数 ──
def _simple_predict_fn(X: pd.DataFrame) -> np.ndarray:
    """簡単な二乗和モデル: y = -(x1-0.3)^2 - (x2-0.5)^2（最大化→(0.3,0.5)が最適）"""
    vals = X.values
    return -((vals[:, 0] - 0.3) ** 2) - ((vals[:, 1] - 0.5) ** 2)


def _composition_predict_fn(X: pd.DataFrame) -> np.ndarray:
    """組成系: y = x1*0.5 + x2*0.3 + x3*0.2（合計1のとき最大→x1=1が理想だがboundsで制限）"""
    vals = X.values
    return vals[:, 0] * 0.5 + vals[:, 1] * 0.3 + vals[:, 2] * 0.2


FEATURE_NAMES_2D = ["x1", "x2"]
FEATURE_NAMES_3D = ["A", "B", "C"]

BOUNDS_2D = {"x1": (0.0, 1.0), "x2": (0.0, 1.0)}
BOUNDS_3D = {"A": (0.0, 0.8), "B": (0.0, 0.8), "C": (0.0, 0.8)}


# ═══════════ ランダムサンプリング ═══════════
class TestRandomOptimizer:
    def test_basic_random(self):
        config = InverseConfig(
            method="random",
            target_mode="maximize",
            constraints={
                "x1": {"min": 0.0, "max": 1.0, "active": True},
                "x2": {"min": 0.0, "max": 1.0, "active": True},
            },
            method_params={"n_samples": 500, "seed": 42},
        )
        result = run_inverse_optimization(
            _simple_predict_fn, FEATURE_NAMES_2D, config,
        )
        assert isinstance(result, InverseResult)
        assert result.method == "random"
        assert result.n_evaluated == 500
        assert len(result.candidates) > 0
        # 最良候補は(0.3, 0.5)付近のはず
        best = result.candidates.iloc[0]
        assert abs(best["x1"] - 0.3) < 0.15
        assert abs(best["x2"] - 0.5) < 0.15

    def test_minimize_mode(self):
        config = InverseConfig(
            method="random",
            target_mode="minimize",
            constraints={
                "x1": {"min": 0.0, "max": 1.0, "active": True},
                "x2": {"min": 0.0, "max": 1.0, "active": True},
            },
            method_params={"n_samples": 500, "seed": 42},
        )
        result = run_inverse_optimization(
            _simple_predict_fn, FEATURE_NAMES_2D, config,
        )
        # minimize = -(x1-0.3)^2の符号反転 → 端点付近が最良
        best = result.candidates.iloc[0]
        assert result.best_predicted is not None


# ═══════════ グリッドサーチ ═══════════
class TestGridOptimizer:
    def test_basic_grid(self):
        config = InverseConfig(
            method="grid",
            target_mode="maximize",
            constraints={
                "x1": {"min": 0.0, "max": 1.0, "active": True},
                "x2": {"min": 0.0, "max": 1.0, "active": True},
            },
            method_params={"n_points": 20},
        )
        result = run_inverse_optimization(
            _simple_predict_fn, FEATURE_NAMES_2D, config,
        )
        assert result.method == "grid"
        assert result.n_evaluated == 20 * 20  # 400
        best = result.candidates.iloc[0]
        assert abs(best["x1"] - 0.3) < 0.1
        assert abs(best["x2"] - 0.5) < 0.1


# ═══════════ ベイズ最適化 ═══════════
class TestBayesianOptimizer:
    def test_basic_bayesian(self):
        config = InverseConfig(
            method="bayesian",
            target_mode="maximize",
            constraints={
                "x1": {"min": 0.0, "max": 1.0, "active": True},
                "x2": {"min": 0.0, "max": 1.0, "active": True},
            },
            method_params={"n_trials": 30, "seed": 42, "acq_func": "EI"},
        )
        result = run_inverse_optimization(
            _simple_predict_fn, FEATURE_NAMES_2D, config,
        )
        assert result.method == "bayesian"
        assert result.n_evaluated >= 30
        best = result.candidates.iloc[0]
        assert abs(best["x1"] - 0.3) < 0.2
        assert abs(best["x2"] - 0.5) < 0.2


# ═══════════ 遺伝的アルゴリズム ═══════════
class TestGAOptimizer:
    def test_basic_ga(self):
        config = InverseConfig(
            method="ga",
            target_mode="maximize",
            constraints={
                "x1": {"min": 0.0, "max": 1.0, "active": True},
                "x2": {"min": 0.0, "max": 1.0, "active": True},
            },
            method_params={
                "pop_size": 20,
                "n_generations": 30,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "seed": 42,
            },
        )
        result = run_inverse_optimization(
            _simple_predict_fn, FEATURE_NAMES_2D, config,
        )
        assert result.method == "ga"
        assert result.n_evaluated >= 20 * 30
        best = result.candidates.iloc[0]
        assert abs(best["x1"] - 0.3) < 0.15
        assert abs(best["x2"] - 0.5) < 0.15


# ═══════════ ディリクレ分布最適化 ═══════════
class TestDirichletOptimizer:
    """ディリクレ分布α更新型の組成系最適化テスト。

    参考:
        Ferguson (1973) ディリクレ過程
        Minka (2000) ディリクレ分布推定
    """

    def test_basic_dirichlet_composition(self):
        """基本ケース: 3成分組成系、合計=1"""
        config = InverseConfig(
            method="dirichlet",
            target_mode="maximize",
            constraints={
                "A": {"min": 0.0, "max": 0.8, "active": True},
                "B": {"min": 0.0, "max": 0.8, "active": True},
                "C": {"min": 0.0, "max": 0.8, "active": True},
            },
            method_params={
                "n_samples_per_round": 200,
                "n_rounds": 10,
                "top_k": 30,
                "concentration": 10.0,
                "total_sum": 1.0,
                "seed": 42,
            },
        )
        result = run_inverse_optimization(
            _composition_predict_fn, FEATURE_NAMES_3D, config,
        )
        assert result.method == "dirichlet"
        assert result.n_evaluated >= 200 * 10
        assert len(result.candidates) > 0

        # 最良候補の合計はtotal_sum (≈1.0) 付近のはず
        best = result.candidates.iloc[0]
        total = best.get("A", 0) + best.get("B", 0) + best.get("C", 0)
        assert abs(total - 1.0) < 0.05, f"合計={total} (期待: ≈1.0)"

        # Aが大きい方がスコアが高い (y = 0.5A + 0.3B + 0.2C)
        assert best["A"] > best["C"], "Aの係数が最大なのでA>Cのはず"

    def test_dirichlet_total_sum_100(self):
        """wt%（合計100）ケース"""
        def predict_wt(X):
            vals = X.values / 100.0  # 0-1に正規化して計算
            return vals[:, 0] * 0.5 + vals[:, 1] * 0.3 + vals[:, 2] * 0.2

        config = InverseConfig(
            method="dirichlet",
            target_mode="maximize",
            constraints={
                "A": {"min": 0.0, "max": 80.0, "active": True},
                "B": {"min": 0.0, "max": 80.0, "active": True},
                "C": {"min": 0.0, "max": 80.0, "active": True},
            },
            method_params={
                "n_samples_per_round": 100,
                "n_rounds": 5,
                "top_k": 20,
                "concentration": 10.0,
                "total_sum": 100.0,
                "seed": 42,
            },
        )
        result = run_inverse_optimization(
            predict_wt, FEATURE_NAMES_3D, config,
        )
        best = result.candidates.iloc[0]
        total = best.get("A", 0) + best.get("B", 0) + best.get("C", 0)
        assert abs(total - 100.0) < 5.0, f"合計={total} (期待: ≈100.0)"

    def test_dirichlet_alpha_convergence(self):
        """αが上位候補に収束すること"""
        config = InverseConfig(
            method="dirichlet",
            target_mode="maximize",
            constraints={
                "A": {"min": 0.0, "max": 0.8, "active": True},
                "B": {"min": 0.0, "max": 0.8, "active": True},
                "C": {"min": 0.0, "max": 0.8, "active": True},
            },
            method_params={
                "n_samples_per_round": 300,
                "n_rounds": 15,
                "top_k": 30,
                "concentration": 15.0,
                "total_sum": 1.0,
                "seed": 123,
            },
        )
        result = run_inverse_optimization(
            _composition_predict_fn, FEATURE_NAMES_3D, config,
        )
        # 多くのラウンドを重ねると最良候補のスコアが改善するはず
        assert result.best_predicted > 0.3, (
            f"ディリクレ収束で最良予測値{result.best_predicted}が0.3超のはず"
        )


# ═══════════ エッジケース ═══════════
class TestEdgeCases:
    def test_no_search_cols_raises(self):
        """全変数が固定の場合はエラー"""
        config = InverseConfig(
            method="random",
            target_mode="maximize",
            constraints={
                "x1": {"fixed": True, "fixed_val": 0.5, "active": True},
                "x2": {"fixed": True, "fixed_val": 0.5, "active": True},
            },
        )
        with pytest.raises(ValueError, match="探索対象の変数がありません"):
            run_inverse_optimization(
                _simple_predict_fn, FEATURE_NAMES_2D, config,
            )

    def test_range_mode(self):
        """range目標モード"""
        config = InverseConfig(
            method="random",
            target_mode="range",
            target_min=-0.1,
            target_max=0.0,
            constraints={
                "x1": {"min": 0.0, "max": 1.0, "active": True},
                "x2": {"min": 0.0, "max": 1.0, "active": True},
            },
            method_params={"n_samples": 500, "seed": 42},
        )
        result = run_inverse_optimization(
            _simple_predict_fn, FEATURE_NAMES_2D, config,
        )
        # rangeモードのスコアはガウシアン型 → 範囲中心に近い予測値の候補が上位
        assert len(result.candidates) > 0

    def test_unknown_method_raises(self):
        """不明な手法はエラー"""
        config = InverseConfig(method="unknown")
        config.constraints = {"x1": {"min": 0, "max": 1, "active": True}}
        with pytest.raises(ValueError, match="未対応の最適化手法"):
            run_inverse_optimization(
                _simple_predict_fn, ["x1"], config,
            )
