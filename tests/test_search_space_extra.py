"""
tests/test_search_space_extra.py

search_space.py の低カバレッジ部分を補うテスト。
Variable, SearchSpace, 候補生成(grid/random/lhs/auto/hybrid),
from_dataframe, 推定/推奨ロジックを網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.optim.search_space import (
    VarType,
    Variable,
    SearchSpace,
    MAX_GRID_DIRECT,
)


# ============================================================
# Variable テスト
# ============================================================

class TestVariable:
    def test_continuous(self):
        v = Variable("x", VarType.CONTINUOUS, lo=0, hi=10)
        assert v.n_levels == 20  # default

    def test_discrete(self):
        v = Variable("x", VarType.DISCRETE, lo=0, hi=100, step=10)
        assert v.n_levels == 11  # 0, 10, 20, ..., 100

    def test_categorical(self):
        v = Variable("x", VarType.CATEGORICAL, categories=["a", "b", "c"])
        assert v.n_levels == 3

    def test_grid_continuous(self):
        v = Variable("x", VarType.CONTINUOUS, lo=0, hi=1)
        vals = v.grid_values(n_per_dim=5)
        assert len(vals) == 5
        assert vals[0] == 0
        assert vals[-1] == 1

    def test_grid_discrete(self):
        v = Variable("x", VarType.DISCRETE, lo=0, hi=30, step=10)
        vals = v.grid_values()
        np.testing.assert_array_equal(vals, [0, 10, 20, 30])

    def test_grid_categorical(self):
        v = Variable("x", VarType.CATEGORICAL, categories=["a", "b"])
        vals = v.grid_values()
        assert list(vals) == ["a", "b"]

    def test_validation_no_bounds(self):
        with pytest.raises(ValueError, match="lo/hi"):
            Variable("x", VarType.CONTINUOUS)

    def test_validation_lo_gt_hi(self):
        with pytest.raises(ValueError, match="lo.*hi"):
            Variable("x", VarType.CONTINUOUS, lo=10, hi=5)

    def test_validation_discrete_no_step(self):
        with pytest.raises(ValueError, match="step"):
            Variable("x", VarType.DISCRETE, lo=0, hi=10)

    def test_validation_categorical_empty(self):
        with pytest.raises(ValueError, match="categories"):
            Variable("x", VarType.CATEGORICAL, categories=[])

    def test_validation_categorical_none(self):
        with pytest.raises(ValueError, match="categories"):
            Variable("x", VarType.CATEGORICAL)


# ============================================================
# SearchSpace 基本テスト
# ============================================================

class TestSearchSpaceBasic:
    def test_add_variables(self):
        space = SearchSpace()
        space.add(Variable("x1", VarType.CONTINUOUS, 0, 10))
        space.add(Variable("x2", VarType.DISCRETE, 0, 100, step=10))
        assert space.dim == 2
        assert space.names == ["x1", "x2"]

    def test_init_with_list(self):
        vars_ = [
            Variable("x1", VarType.CONTINUOUS, 0, 10),
            Variable("x2", VarType.CONTINUOUS, 0, 5),
        ]
        space = SearchSpace(variables=vars_)
        assert space.dim == 2

    def test_estimate_grid_size(self):
        space = SearchSpace([
            Variable("x1", VarType.DISCRETE, 0, 2, step=1),
            Variable("x2", VarType.DISCRETE, 0, 3, step=1),
        ])
        # x1: 3 levels (0,1,2), x2: 4 levels (0,1,2,3)
        assert space.estimate_grid_size() == 3 * 4

    def test_estimate_empty(self):
        space = SearchSpace()
        assert space.estimate_grid_size() == 0

    def test_auto_recommend_grid(self):
        space = SearchSpace([
            Variable("x1", VarType.DISCRETE, 0, 5, step=1),
            Variable("x2", VarType.DISCRETE, 0, 5, step=1),
        ])
        assert space.auto_recommend_method() == "grid"

    def test_auto_recommend_random(self):
        """Many continuous dimensions → random_lhs."""
        vars_ = [Variable(f"x{i}", VarType.CONTINUOUS, 0, 10) for i in range(10)]
        space = SearchSpace(vars_)
        # 20^10 = 10^13 → random_lhs
        method = space.auto_recommend_method()
        assert method == "random_lhs"


# ============================================================
# 候補生成テスト
# ============================================================

class TestGenerateCandidates:
    def test_grid(self):
        space = SearchSpace([
            Variable("x1", VarType.DISCRETE, 0, 2, step=1),
            Variable("x2", VarType.CATEGORICAL, categories=["a", "b"]),
        ])
        df = space.generate_candidates(method="grid")
        # 3 * 2 = 6
        assert len(df) == 6
        assert set(df.columns) == {"x1", "x2"}

    def test_random(self):
        space = SearchSpace([
            Variable("x1", VarType.CONTINUOUS, 0, 10),
            Variable("x2", VarType.CATEGORICAL, categories=["a", "b", "c"]),
        ])
        df = space.generate_candidates(method="random", n_max=100)
        assert len(df) == 100

    def test_lhs(self):
        space = SearchSpace([
            Variable("x1", VarType.CONTINUOUS, 0, 10),
            Variable("x2", VarType.CONTINUOUS, -5, 5),
        ])
        df = space.generate_candidates(method="lhs", n_max=50)
        assert len(df) == 50
        assert df["x1"].min() >= 0
        assert df["x1"].max() <= 10

    def test_random_lhs(self):
        space = SearchSpace([
            Variable("x1", VarType.CONTINUOUS, 0, 10),
            Variable("x2", VarType.CONTINUOUS, 0, 5),
        ])
        df = space.generate_candidates(method="random_lhs", n_max=100)
        assert len(df) == 100

    def test_grid_downsample(self):
        space = SearchSpace([
            Variable("x1", VarType.CONTINUOUS, 0, 10),
            Variable("x2", VarType.CONTINUOUS, 0, 10),
        ])
        df = space.generate_candidates(method="grid_downsample", n_max=50, n_per_dim=20)
        assert len(df) <= 50

    def test_auto(self):
        space = SearchSpace([
            Variable("x1", VarType.DISCRETE, 0, 2, step=1),
        ])
        df = space.generate_candidates(method="auto")
        assert len(df) > 0

    def test_invalid_method(self):
        space = SearchSpace([
            Variable("x1", VarType.CONTINUOUS, 0, 10),
        ])
        with pytest.raises(ValueError, match="不明"):
            space.generate_candidates(method="invalid")

    def test_empty_variables(self):
        space = SearchSpace()
        with pytest.raises(ValueError, match="変数が定義"):
            space.generate_candidates()

    def test_lhs_with_discrete(self):
        """LHS with discrete variables snaps to grid."""
        space = SearchSpace([
            Variable("x1", VarType.DISCRETE, 0, 100, step=10),
            Variable("x2", VarType.CONTINUOUS, 0, 1),
        ])
        df = space.generate_candidates(method="lhs", n_max=50)
        assert len(df) == 50
        # Discrete values should be multiples of 10
        unique_x1 = sorted(df["x1"].unique())
        for val in unique_x1:
            assert val % 10 == 0 or abs(val % 10) < 1e-9

    def test_lhs_with_categorical(self):
        """LHS with categorical variables."""
        space = SearchSpace([
            Variable("x1", VarType.CONTINUOUS, 0, 10),
            Variable("cat", VarType.CATEGORICAL, categories=["a", "b", "c"]),
        ])
        df = space.generate_candidates(method="lhs", n_max=30)
        assert len(df) == 30
        assert set(df["cat"].unique()).issubset({"a", "b", "c"})


# ============================================================
# from_dataframe テスト
# ============================================================

class TestFromDataFrame:
    def test_numeric_columns(self):
        df = pd.DataFrame({
            "x1": np.random.randn(50),
            "x2": np.random.randn(50) * 10,
        })
        space = SearchSpace.from_dataframe(df)
        assert space.dim == 2
        assert all(v.var_type == VarType.CONTINUOUS for v in space.variables)

    def test_integer_columns(self):
        df = pd.DataFrame({
            "count": np.random.randint(0, 100, 50),
        })
        space = SearchSpace.from_dataframe(df)
        assert space.variables[0].var_type == VarType.DISCRETE

    def test_specific_columns(self):
        df = pd.DataFrame({
            "x1": np.random.randn(50),
            "x2": np.random.randn(50),
            "x3": np.random.randn(50),
        })
        space = SearchSpace.from_dataframe(df, columns=["x1", "x2"])
        assert space.dim == 2

    def test_margin(self):
        df = pd.DataFrame({"x": [0.0, 1.0]})
        space = SearchSpace.from_dataframe(df, margin=0.1)
        v = space.variables[0]
        assert v.lo < 0.0
        assert v.hi > 1.0

    def test_categorical_column(self):
        df = pd.DataFrame({
            "x": np.random.randn(50),
            "cat": pd.Categorical(["a", "b", "c"] * 16 + ["a", "b"]),
        })
        space = SearchSpace.from_dataframe(df, columns=["x", "cat"])
        cat_var = [v for v in space.variables if v.name == "cat"]
        assert len(cat_var) == 1
        assert cat_var[0].var_type == VarType.CATEGORICAL
