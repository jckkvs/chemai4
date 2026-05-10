"""
tests/test_optim_comprehensive.py

optim/constraints.py + optim/search_space.py の包括テスト。
全制約型（Range/Sum/Inequality/AtLeastN/Custom）+ apply_constraints と
Variable/SearchSpace/全生成メソッドを網羅。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.optim.constraints import (
    RangeConstraint,
    SumConstraint,
    InequalityConstraint,
    AtLeastNConstraint,
    CustomConstraint,
    apply_constraints,
)
from backend.optim.search_space import (
    VarType,
    Variable,
    SearchSpace,
)


# ============================================================
# Constraint テスト
# ============================================================

class TestRangeConstraint:
    def test_in_range(self):
        c = RangeConstraint(column="x", lo=0, hi=10)
        row = pd.Series({"x": 5})
        assert c.is_satisfied(row)

    def test_below(self):
        c = RangeConstraint(column="x", lo=0, hi=10)
        row = pd.Series({"x": -1})
        assert not c.is_satisfied(row)

    def test_above(self):
        c = RangeConstraint(column="x", lo=0, hi=10)
        row = pd.Series({"x": 11})
        assert not c.is_satisfied(row)

    def test_mask(self):
        c = RangeConstraint(column="x", lo=0, hi=5)
        df = pd.DataFrame({"x": [-1, 0, 3, 5, 6]})
        m = c.mask(df)
        assert m.tolist() == [False, True, True, True, False]

    def test_describe(self):
        c = RangeConstraint(column="x", lo=0, hi=10)
        assert "x" in c.describe()

    def test_lo_only(self):
        c = RangeConstraint(column="x", lo=5)
        assert c.is_satisfied(pd.Series({"x": 10}))
        assert not c.is_satisfied(pd.Series({"x": 3}))

    def test_hi_only(self):
        c = RangeConstraint(column="x", hi=5)
        assert c.is_satisfied(pd.Series({"x": 3}))
        assert not c.is_satisfied(pd.Series({"x": 10}))


class TestSumConstraint:
    def test_exact(self):
        c = SumConstraint(columns=["a", "b", "c"], target=100, tolerance=1e-6)
        row = pd.Series({"a": 30, "b": 30, "c": 40})
        assert c.is_satisfied(row)

    def test_not_exact(self):
        c = SumConstraint(columns=["a", "b"], target=100)
        row = pd.Series({"a": 30, "b": 30})
        assert not c.is_satisfied(row)

    def test_mask(self):
        c = SumConstraint(columns=["a", "b"], target=10, tolerance=0.5)
        df = pd.DataFrame({"a": [5, 9, 3], "b": [5, 1, 2]})
        m = c.mask(df)
        assert m.tolist() == [True, True, False]

    def test_describe(self):
        c = SumConstraint(columns=["a", "b"], target=100)
        assert "a" in c.describe()


class TestInequalityConstraint:
    def test_le(self):
        c = InequalityConstraint(coefficients={"x": 1, "y": 1}, rhs=10, operator="le")
        assert c.is_satisfied(pd.Series({"x": 3, "y": 5}))
        assert not c.is_satisfied(pd.Series({"x": 8, "y": 5}))

    def test_ge(self):
        c = InequalityConstraint(coefficients={"x": 1}, rhs=5, operator="ge")
        assert c.is_satisfied(pd.Series({"x": 10}))
        assert not c.is_satisfied(pd.Series({"x": 3}))

    def test_lt(self):
        c = InequalityConstraint(coefficients={"x": 1}, rhs=5, operator="lt")
        assert c.is_satisfied(pd.Series({"x": 4}))
        assert not c.is_satisfied(pd.Series({"x": 5}))

    def test_gt(self):
        c = InequalityConstraint(coefficients={"x": 1}, rhs=5, operator="gt")
        assert c.is_satisfied(pd.Series({"x": 6}))

    def test_mask(self):
        c = InequalityConstraint(coefficients={"x": 2, "y": 1}, rhs=10, operator="le")
        df = pd.DataFrame({"x": [1, 5, 3], "y": [1, 5, 5]})
        m = c.mask(df)
        assert m.tolist() == [True, False, False]

    def test_describe(self):
        c = InequalityConstraint(coefficients={"x": 1.0, "y": -1.0}, rhs=5, operator="le")
        desc = c.describe()
        assert "x" in desc


class TestAtLeastNConstraint:
    def test_satisfied(self):
        c = AtLeastNConstraint(columns=["a", "b", "c"], min_count=2, threshold=0)
        row = pd.Series({"a": 1, "b": 2, "c": 0})
        assert c.is_satisfied(row)

    def test_not_satisfied(self):
        c = AtLeastNConstraint(columns=["a", "b", "c"], min_count=3, threshold=0)
        row = pd.Series({"a": 1, "b": 0, "c": 0})
        assert not c.is_satisfied(row)

    def test_mask(self):
        c = AtLeastNConstraint(columns=["a", "b"], min_count=1, threshold=5)
        df = pd.DataFrame({"a": [0, 10, 3], "b": [0, 0, 6]})
        m = c.mask(df)
        assert m.tolist() == [False, True, True]


class TestCustomConstraint:
    def test_eval(self):
        c = CustomConstraint(expression="A * B <= 50")
        assert c.is_satisfied(pd.Series({"A": 5, "B": 5}))
        assert not c.is_satisfied(pd.Series({"A": 10, "B": 10}))

    def test_mask(self):
        c = CustomConstraint(expression="x > 3")
        df = pd.DataFrame({"x": [1, 5, 10]})
        m = c.mask(df)
        assert m.tolist() == [False, True, True]

    def test_describe(self):
        c = CustomConstraint(expression="A + B == 100")
        assert "カスタム" in c.describe()


class TestApplyConstraints:
    def test_multiple(self):
        df = pd.DataFrame({"x": [1, 5, 10, 15], "y": [4, 5, 0, 5]})
        constraints = [
            RangeConstraint(column="x", lo=0, hi=12),
            RangeConstraint(column="y", lo=1, hi=10),
        ]
        filtered, report = apply_constraints(df, constraints)
        assert report["before"] == 4
        assert len(filtered) == report["after"]
        assert report["removed"] == report["before"] - report["after"]

    def test_no_constraints(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        filtered, report = apply_constraints(df, [])
        assert len(filtered) == 3


# ============================================================
# Variable テスト
# ============================================================

class TestVariable:
    def test_continuous(self):
        v = Variable("x", VarType.CONTINUOUS, lo=0, hi=10)
        assert v.n_levels == 20
        vals = v.grid_values(10)
        assert len(vals) == 10

    def test_discrete(self):
        v = Variable("d", VarType.DISCRETE, lo=0, hi=10, step=2)
        vals = v.grid_values()
        assert all(v_i in [0, 2, 4, 6, 8, 10] for v_i in vals)

    def test_categorical(self):
        v = Variable("c", VarType.CATEGORICAL, categories=["a", "b", "c"])
        assert v.n_levels == 3
        vals = v.grid_values()
        assert list(vals) == ["a", "b", "c"]

    def test_validation_no_lo(self):
        with pytest.raises(ValueError, match="lo/hi"):
            Variable("x", VarType.CONTINUOUS, lo=None, hi=10)

    def test_validation_lo_gt_hi(self):
        with pytest.raises(ValueError, match="lo"):
            Variable("x", VarType.CONTINUOUS, lo=10, hi=0)

    def test_validation_discrete_no_step(self):
        with pytest.raises(ValueError, match="step"):
            Variable("x", VarType.DISCRETE, lo=0, hi=10)

    def test_validation_categorical_no_categories(self):
        with pytest.raises(ValueError, match="categories"):
            Variable("x", VarType.CATEGORICAL)


# ============================================================
# SearchSpace テスト
# ============================================================

class TestSearchSpace:
    @pytest.fixture
    def simple_space(self):
        return SearchSpace([
            Variable("x", VarType.CONTINUOUS, lo=0, hi=1),
            Variable("y", VarType.DISCRETE, lo=0, hi=10, step=5),
        ])

    def test_dim(self, simple_space):
        assert simple_space.dim == 2

    def test_names(self, simple_space):
        assert simple_space.names == ["x", "y"]

    def test_grid(self, simple_space):
        df = simple_space.generate_candidates(method="grid", n_per_dim=5)
        assert len(df) == 5 * 3  # 5 continuous x 3 discrete (0,5,10)

    def test_random(self, simple_space):
        df = simple_space.generate_candidates(method="random", n_max=100)
        assert len(df) == 100
        assert df["x"].min() >= 0
        assert df["x"].max() <= 1

    def test_lhs(self, simple_space):
        df = simple_space.generate_candidates(method="lhs", n_max=50)
        assert len(df) == 50

    def test_auto(self, simple_space):
        df = simple_space.generate_candidates(method="auto", n_per_dim=5)
        assert len(df) > 0

    def test_grid_downsample(self, simple_space):
        df = simple_space.generate_candidates(method="grid_downsample", n_per_dim=5, n_max=10)
        assert len(df) <= 10

    def test_random_lhs(self, simple_space):
        df = simple_space.generate_candidates(method="random_lhs", n_max=100)
        assert len(df) == 100

    def test_invalid_method(self, simple_space):
        with pytest.raises(ValueError, match="不明な"):
            simple_space.generate_candidates(method="invalid")

    def test_no_variables(self):
        with pytest.raises(ValueError, match="変数"):
            SearchSpace().generate_candidates()

    def test_estimate_grid_size(self, simple_space):
        size = simple_space.estimate_grid_size(n_per_dim=10)
        assert size == 10 * 3

    def test_auto_recommend(self, simple_space):
        method = simple_space.auto_recommend_method(n_per_dim=5)
        assert method in ("grid", "grid_downsample", "random_lhs")

    def test_from_dataframe(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10, 20, 30]})
        space = SearchSpace.from_dataframe(df, margin=0.1)
        assert space.dim == 2

    def test_add_variable(self):
        space = SearchSpace()
        space.add(Variable("z", VarType.CONTINUOUS, lo=0, hi=1))
        assert space.dim == 1

    def test_with_categorical(self):
        space = SearchSpace([
            Variable("x", VarType.CONTINUOUS, lo=0, hi=1),
            Variable("c", VarType.CATEGORICAL, categories=["a", "b"]),
        ])
        df = space.generate_candidates(method="grid", n_per_dim=5)
        assert "c" in df.columns
        assert set(df["c"].unique()) == {"a", "b"}
