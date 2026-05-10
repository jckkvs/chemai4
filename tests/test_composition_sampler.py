"""
tests/test_composition_sampler.py

DirichletSampler のユニットテスト。

テスト対象: backend/optim/composition_sampler.py
    T-DIR01: sample() — 基本サンプリング + 合計制約
    T-DIR02: constrained_sample() — 範囲制約付きサンプリング
    T-DIR03: update_alpha() — α更新の収束性
    T-DIR04: get_concentration_summary() — サマリー出力
    T-DIR05: reset_alpha() — リセット動作
    T-DIR06: エッジケース（k=2, 大量サンプル, 厳しい制約）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.optim.composition_sampler import DirichletSampler


# ============================================================
# T-DIR01: 基本サンプリング
# ============================================================
class TestDirichletBasicSample:
    """sample() の基本動作テスト。"""

    def test_sample_returns_correct_shape(self):
        """サンプル数と列数が要求通りであること。"""
        sampler = DirichletSampler(columns=["A", "B", "C"], target_sum=100.0)
        df = sampler.sample(50)
        assert df.shape == (50, 3)
        assert list(df.columns) == ["A", "B", "C"]

    def test_sample_sum_equals_target(self):
        """各行の合計がtarget_sumに等しいこと。"""
        sampler = DirichletSampler(columns=["A", "B", "C"], target_sum=100.0)
        df = sampler.sample(100)
        row_sums = df.sum(axis=1)
        np.testing.assert_allclose(row_sums, 100.0, atol=1e-10)

    def test_sample_all_positive(self):
        """全値が正であること（ディリクレ分布はsimplex上）。"""
        sampler = DirichletSampler(columns=["X", "Y"], target_sum=1.0)
        df = sampler.sample(200)
        assert (df > 0).all().all()

    def test_sample_custom_target_sum(self):
        """target_sum=1.0 でも正しく動作すること。"""
        sampler = DirichletSampler(columns=["a", "b", "c", "d"], target_sum=1.0)
        df = sampler.sample(30)
        np.testing.assert_allclose(df.sum(axis=1), 1.0, atol=1e-10)

    def test_sample_reproducible_with_seed(self):
        """同じseedで同じ結果が得られること。"""
        s1 = DirichletSampler(columns=["A", "B"], seed=123)
        s2 = DirichletSampler(columns=["A", "B"], seed=123)
        df1 = s1.sample(10)
        df2 = s2.sample(10)
        pd.testing.assert_frame_equal(df1, df2)

    def test_sample_different_seeds_differ(self):
        """異なるseedで異なる結果。"""
        s1 = DirichletSampler(columns=["A", "B"], seed=1)
        s2 = DirichletSampler(columns=["A", "B"], seed=99)
        df1 = s1.sample(10)
        df2 = s2.sample(10)
        assert not df1.equals(df2)


# ============================================================
# T-DIR02: 制約付きサンプリング
# ============================================================
class TestDirichletConstrainedSample:
    """constrained_sample() のテスト。"""

    def test_min_constraint_satisfied(self):
        """min_values が守られること。"""
        sampler = DirichletSampler(
            columns=["A", "B", "C"],
            target_sum=100.0,
            min_values={"A": 10.0},
        )
        df = sampler.constrained_sample(50)
        assert len(df) == 50
        assert (df["A"] >= 10.0).all()

    def test_max_constraint_satisfied(self):
        """max_values が守られること。"""
        sampler = DirichletSampler(
            columns=["A", "B", "C"],
            target_sum=100.0,
            max_values={"B": 50.0},
        )
        df = sampler.constrained_sample(50)
        assert len(df) == 50
        assert (df["B"] <= 50.0).all()

    def test_combined_constraints(self):
        """min+max 複合制約。"""
        sampler = DirichletSampler(
            columns=["A", "B", "C"],
            target_sum=100.0,
            min_values={"A": 20.0, "B": 10.0},
            max_values={"A": 60.0, "C": 40.0},
        )
        df = sampler.constrained_sample(30)
        assert len(df) > 0
        assert (df["A"] >= 20.0).all()
        assert (df["A"] <= 60.0).all()
        assert (df["B"] >= 10.0).all()
        assert (df["C"] <= 40.0).all()
        np.testing.assert_allclose(df.sum(axis=1), 100.0, atol=1e-10)

    def test_impossible_constraint_returns_empty(self):
        """不可能な制約ではDataFrameが空（or少数）。"""
        sampler = DirichletSampler(
            columns=["A", "B"],
            target_sum=100.0,
            min_values={"A": 90.0, "B": 90.0},  # A+B=100なのに両方90以上は不可能
        )
        df = sampler.constrained_sample(10, max_attempts=5)
        assert len(df) == 0


# ============================================================
# T-DIR03: α更新
# ============================================================
class TestDirichletAlphaUpdate:
    """update_alpha() のテスト。"""

    def test_alpha_changes_after_update(self):
        """更新後にαが変化すること。"""
        sampler = DirichletSampler(columns=["A", "B", "C"], target_sum=100.0)
        alpha_before = sampler.alpha.copy()
        # A成分が多いサンプルを「良い」とフィードバック
        good = pd.DataFrame({"A": [70, 65, 75], "B": [20, 25, 15], "C": [10, 10, 10]})
        sampler.update_alpha(good)
        assert not np.allclose(sampler.alpha, alpha_before)

    def test_alpha_concentrates_on_good_region(self):
        """良いサンプルの成分にαが集中すること。"""
        sampler = DirichletSampler(columns=["A", "B", "C"], target_sum=100.0)
        # A成分が支配的なサンプルをフィードバック
        good = pd.DataFrame({
            "A": [80, 85, 90],
            "B": [15, 10, 5],
            "C": [5, 5, 5],
        })
        sampler.update_alpha(good, learning_rate=0.5)
        # Aのαが最大になるはず
        assert sampler.alpha[0] > sampler.alpha[1]
        assert sampler.alpha[0] > sampler.alpha[2]

    def test_update_count_increments(self):
        """更新回数がカウントされること。"""
        sampler = DirichletSampler(columns=["A", "B"], target_sum=1.0)
        assert sampler._update_count == 0
        good = pd.DataFrame({"A": [0.6], "B": [0.4]})
        sampler.update_alpha(good)
        assert sampler._update_count == 1
        sampler.update_alpha(good)
        assert sampler._update_count == 2

    def test_empty_good_samples_no_change(self):
        """空のgood_samplesでは更新されないこと。"""
        sampler = DirichletSampler(columns=["A", "B"], target_sum=1.0)
        alpha_before = sampler.alpha.copy()
        sampler.update_alpha(pd.DataFrame(columns=["A", "B"]))
        np.testing.assert_array_equal(sampler.alpha, alpha_before)

    def test_min_alpha_enforced(self):
        """αが下限以下にならないこと。"""
        sampler = DirichletSampler(
            columns=["A", "B"],
            target_sum=100.0,
            alpha=np.array([0.2, 0.2]),
        )
        good = pd.DataFrame({"A": [99], "B": [1]})
        sampler.update_alpha(good, min_alpha=0.05)
        assert (sampler.alpha >= 0.05).all()


# ============================================================
# T-DIR04: サマリー
# ============================================================
class TestDirichletSummary:
    """get_concentration_summary() のテスト。"""

    def test_summary_keys(self):
        """サマリーが必要なキーを含むこと。"""
        sampler = DirichletSampler(columns=["A", "B", "C"])
        summary = sampler.get_concentration_summary()
        assert "alpha" in summary
        assert "concentration" in summary
        assert "expected_proportions" in summary
        assert "update_count" in summary
        assert "entropy" in summary

    def test_expected_proportions_sum_to_target(self):
        """期待組成の合計がtarget_sumに等しいこと。"""
        sampler = DirichletSampler(columns=["A", "B", "C"], target_sum=100.0)
        summary = sampler.get_concentration_summary()
        total = sum(summary["expected_proportions"].values())
        assert abs(total - 100.0) < 1e-10


# ============================================================
# T-DIR05: リセット
# ============================================================
class TestDirichletReset:
    """reset_alpha() のテスト。"""

    def test_reset_to_uniform(self):
        """リセット後にα=1（均一）になること。"""
        sampler = DirichletSampler(columns=["A", "B", "C"])
        good = pd.DataFrame({"A": [70], "B": [20], "C": [10]})
        sampler.update_alpha(good)
        sampler.reset_alpha(uniform=True)
        np.testing.assert_array_equal(sampler.alpha, np.ones(3))
        assert sampler._update_count == 0

    def test_reset_to_initial(self):
        """リセット後に初期αに戻ること。"""
        init_alpha = np.array([2.0, 3.0, 1.0])
        sampler = DirichletSampler(columns=["A", "B", "C"], alpha=init_alpha.copy())
        good = pd.DataFrame({"A": [70], "B": [20], "C": [10]})
        sampler.update_alpha(good)
        sampler.reset_alpha(uniform=False)
        np.testing.assert_array_equal(sampler.alpha, init_alpha)


# ============================================================
# T-DIR06: エッジケース
# ============================================================
class TestDirichletEdgeCases:
    """エッジケースのテスト。"""

    def test_two_components(self):
        """2成分でも正しく動作すること。"""
        sampler = DirichletSampler(columns=["A", "B"], target_sum=100.0)
        df = sampler.sample(20)
        assert df.shape == (20, 2)
        np.testing.assert_allclose(df.sum(axis=1), 100.0, atol=1e-10)

    def test_many_components(self):
        """10成分でも正しく動作すること。"""
        cols = [f"C{i}" for i in range(10)]
        sampler = DirichletSampler(columns=cols, target_sum=100.0)
        df = sampler.sample(50)
        assert df.shape == (50, 10)
        np.testing.assert_allclose(df.sum(axis=1), 100.0, atol=1e-10)

    def test_invalid_alpha_raises(self):
        """不正なα（負の値）でValueErrorが出ること。"""
        with pytest.raises(ValueError, match="正の実数"):
            DirichletSampler(columns=["A", "B"], alpha=np.array([-1.0, 1.0]))

    def test_alpha_length_mismatch_raises(self):
        """αの長さが列数と不一致でValueErrorが出ること。"""
        with pytest.raises(ValueError, match="不一致"):
            DirichletSampler(columns=["A", "B"], alpha=np.array([1.0, 1.0, 1.0]))

    def test_large_sample(self):
        """10000点でも合計制約が維持されること。"""
        sampler = DirichletSampler(columns=["A", "B", "C"], target_sum=100.0)
        df = sampler.sample(10000)
        np.testing.assert_allclose(df.sum(axis=1), 100.0, atol=1e-10)
        assert (df > 0).all().all()
