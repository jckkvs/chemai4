"""
tests/test_constraints_extra.py

constraints.py の低カバレッジ部分を補うテスト。
全制約クラス(Range/Sum/Inequality/AtLeastN/Custom)の
is_satisfied, mask, describe + apply_constraints を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.optim.constraints import (
    RangeConstraint,
    SumConstraint,
    InequalityConstraint,
    AtLeastNConstraint,
    AtLeastOneConstraint,
    CustomConstraint,
    apply_constraints,
)


# ============================================================
# テストデータ
# ============================================================

def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        "B": [5.0, 4.0, 3.0, 2.0, 1.0],
        "C": [0.0, 1.0, 0.0, 1.0, 0.0],
    })


# ============================================================
# RangeConstraint
# ============================================================

class TestRangeConstraint:
    def test_both_bounds(self):
        c = RangeConstraint(column="A", lo=2.0, hi=4.0)
        df = _sample_df()
        mask = c.mask(df)
        assert mask.tolist() == [False, True, True, True, False]

    def test_lo_only(self):
        c = RangeConstraint(column="A", lo=3.0)
        df = _sample_df()
        mask = c.mask(df)
        assert mask.tolist() == [False, False, True, True, True]

    def test_hi_only(self):
        c = RangeConstraint(column="A", hi=2.0)
        df = _sample_df()
        mask = c.mask(df)
        assert mask.tolist() == [True, True, False, False, False]

    def test_no_bounds(self):
        c = RangeConstraint(column="A")
        df = _sample_df()
        mask = c.mask(df)
        assert all(mask)

    def test_is_satisfied(self):
        c = RangeConstraint(column="A", lo=1.5, hi=3.5)
        row = _sample_df().iloc[0]  # A=1.0
        assert c.is_satisfied(row) is False
        row = _sample_df().iloc[1]  # A=2.0
        assert c.is_satisfied(row) is True

    def test_describe_both(self):
        c = RangeConstraint(column="A", lo=1.0, hi=5.0)
        desc = c.describe()
        assert "A" in desc
        assert "1.0" in desc
        assert "5.0" in desc

    def test_describe_lo_only(self):
        c = RangeConstraint(column="A", lo=2.0)
        desc = c.describe()
        assert "A" in desc
        assert "≥" in desc


# ============================================================
# SumConstraint
# ============================================================

class TestSumConstraint:
    def test_exact_sum(self):
        c = SumConstraint(columns=["A", "B"], target=6.0, tolerance=0.01)
        df = _sample_df()
        mask = c.mask(df)
        assert all(mask)  # A + B = 6 for all rows

    def test_with_tolerance(self):
        c = SumConstraint(columns=["A", "B"], target=6.0, tolerance=0.5)
        df = _sample_df()
        mask = c.mask(df)
        assert all(mask)

    def test_failing_sum(self):
        c = SumConstraint(columns=["A", "B"], target=10.0, tolerance=0.01)
        df = _sample_df()
        mask = c.mask(df)
        assert not any(mask)

    def test_is_satisfied(self):
        c = SumConstraint(columns=["A", "B"], target=6.0, tolerance=0.01)
        row = _sample_df().iloc[0]
        assert c.is_satisfied(row) == True

    def test_describe_tight(self):
        c = SumConstraint(columns=["A", "B"], target=100, tolerance=1e-8)
        desc = c.describe()
        assert "A + B = 100" == desc

    def test_describe_with_tolerance(self):
        c = SumConstraint(columns=["A", "B"], target=100, tolerance=5.0)
        desc = c.describe()
        assert "5.0" in desc


# ============================================================
# InequalityConstraint
# ============================================================

class TestInequalityConstraint:
    def test_le(self):
        c = InequalityConstraint(coefficients={"A": 1.0}, rhs=3.0, operator="le")
        df = _sample_df()
        mask = c.mask(df)
        assert mask.tolist() == [True, True, True, False, False]

    def test_ge(self):
        c = InequalityConstraint(coefficients={"A": 1.0}, rhs=3.0, operator="ge")
        df = _sample_df()
        mask = c.mask(df)
        assert mask.tolist() == [False, False, True, True, True]

    def test_lt(self):
        c = InequalityConstraint(coefficients={"A": 1.0}, rhs=3.0, operator="lt")
        df = _sample_df()
        mask = c.mask(df)
        assert mask.tolist() == [True, True, False, False, False]

    def test_gt(self):
        c = InequalityConstraint(coefficients={"A": 1.0}, rhs=3.0, operator="gt")
        df = _sample_df()
        mask = c.mask(df)
        assert mask.tolist() == [False, False, False, True, True]

    def test_multi_coeff(self):
        c = InequalityConstraint(
            coefficients={"A": 1.0, "B": -1.0}, rhs=0.0, operator="le"
        )
        df = _sample_df()
        mask = c.mask(df)
        # A - B: [-4, -2, 0, 2, 4] ≤ 0
        assert mask.tolist() == [True, True, True, False, False]

    def test_is_satisfied(self):
        c = InequalityConstraint(coefficients={"A": 1.0}, rhs=3.0, operator="le")
        row = _sample_df().iloc[0]  # A=1.0
        assert c.is_satisfied(row) == True
        row = _sample_df().iloc[4]  # A=5.0
        assert c.is_satisfied(row) == False

    def test_is_satisfied_all_operators(self):
        for op in ["le", "ge", "lt", "gt"]:
            c = InequalityConstraint(coefficients={"A": 1.0}, rhs=3.0, operator=op)
            row = _sample_df().iloc[2]  # A=3.0
            c.is_satisfied(row)  # Should not raise

    def test_invalid_operator(self):
        c = InequalityConstraint(coefficients={"A": 1.0}, rhs=3.0, operator="invalid")
        row = _sample_df().iloc[0]
        with pytest.raises(ValueError):
            c.is_satisfied(row)

    def test_invalid_operator_mask(self):
        c = InequalityConstraint(coefficients={"A": 1.0}, rhs=3.0, operator="invalid")
        df = _sample_df()
        with pytest.raises(ValueError):
            c.mask(df)

    def test_describe(self):
        c = InequalityConstraint(
            coefficients={"A": 1.0, "B": -1.0, "C": 2.5},
            rhs=10.0, operator="le",
        )
        desc = c.describe()
        assert "A" in desc
        assert "≤" in desc
        assert "10.0" in desc

    def test_describe_coeff_one(self):
        c = InequalityConstraint(coefficients={"A": 1.0}, rhs=5.0, operator="ge")
        desc = c.describe()
        assert "A" in desc  # Should not show "1.0*A"

    def test_describe_coeff_neg_one(self):
        c = InequalityConstraint(coefficients={"A": -1.0}, rhs=5.0, operator="le")
        desc = c.describe()
        assert "-A" in desc


# ============================================================
# AtLeastNConstraint
# ============================================================

class TestAtLeastNConstraint:
    def test_at_least_one(self):
        c = AtLeastNConstraint(columns=["A", "C"], min_count=1, threshold=0.0)
        df = _sample_df()
        mask = c.mask(df)
        assert all(mask)  # A > 0 for all rows

    def test_at_least_two(self):
        c = AtLeastNConstraint(columns=["A", "B", "C"], min_count=3, threshold=0.0)
        df = _sample_df()
        mask = c.mask(df)
        # C has zeros at rows 0, 2, 4
        assert mask.tolist() == [False, True, False, True, False]

    def test_threshold(self):
        c = AtLeastNConstraint(columns=["A", "B"], min_count=1, threshold=4.5)
        df = _sample_df()
        mask = c.mask(df)
        # A > 4.5: [F,F,F,F,T], B > 4.5: [T,F,F,F,F]
        assert mask.tolist() == [True, False, False, False, True]

    def test_is_satisfied(self):
        c = AtLeastNConstraint(columns=["A", "C"], min_count=2, threshold=0.0)
        row = _sample_df().iloc[1]  # A=2, C=1 → both > 0
        assert c.is_satisfied(row) is True
        row = _sample_df().iloc[0]  # A=1, C=0 → only A > 0
        assert c.is_satisfied(row) is False

    def test_describe(self):
        c = AtLeastNConstraint(columns=["A", "B"], min_count=1)
        desc = c.describe()
        assert "A" in desc
        assert "1" in desc

    def test_describe_with_label(self):
        c = AtLeastNConstraint(columns=["A", "B"], min_count=1, label="test_label")
        desc = c.describe()
        assert "test_label" in desc

    def test_alias(self):
        """AtLeastOneConstraintはAtLeastNConstraintのエイリアス"""
        assert AtLeastOneConstraint is AtLeastNConstraint


# ============================================================
# CustomConstraint
# ============================================================

class TestCustomConstraint:
    def test_simple_expression(self):
        c = CustomConstraint(expression="A > 2")
        df = _sample_df()
        mask = c.mask(df)
        assert mask.tolist() == [False, False, True, True, True]

    def test_compound_expression(self):
        c = CustomConstraint(expression="(A > 2) & (B < 4)")
        df = _sample_df()
        mask = c.mask(df)
        # A > 2: [F,F,T,T,T], B < 4: [F,F,T,T,T]
        assert mask.tolist() == [False, False, True, True, True]

    def test_is_satisfied(self):
        c = CustomConstraint(expression="A * B <= 10")
        row = _sample_df().iloc[0]  # A=1, B=5 → 5 <= 10
        assert c.is_satisfied(row) is True

    def test_invalid_expression(self):
        c = CustomConstraint(expression="INVALID((((")
        df = _sample_df()
        mask = c.mask(df)
        # Should not raise, fallback to row-by-row
        assert isinstance(mask, pd.Series)

    def test_invalid_is_satisfied(self):
        c = CustomConstraint(expression="INVALID((((")
        row = _sample_df().iloc[0]
        assert c.is_satisfied(row) is False

    def test_describe(self):
        c = CustomConstraint(expression="A + B <= 10")
        assert "A + B <= 10" in c.describe()


# ============================================================
# apply_constraints
# ============================================================

class TestApplyConstraints:
    def test_single_constraint(self):
        df = _sample_df()
        constraints = [RangeConstraint(column="A", lo=2.0, hi=4.0)]
        filtered, info = apply_constraints(df, constraints)
        assert info["before"] == 5
        assert info["after"] == 3
        assert info["removed"] == 2
        assert len(info["details"]) == 1

    def test_multiple_constraints(self):
        df = _sample_df()
        constraints = [
            RangeConstraint(column="A", lo=2.0),
            RangeConstraint(column="B", lo=2.0),
        ]
        filtered, info = apply_constraints(df, constraints)
        assert info["before"] == 5
        assert info["after"] <= 5

    def test_no_constraints(self):
        df = _sample_df()
        filtered, info = apply_constraints(df, [])
        assert info["before"] == info["after"]
        assert info["removed"] == 0

    def test_all_filtered(self):
        df = _sample_df()
        constraints = [RangeConstraint(column="A", lo=100.0)]
        filtered, info = apply_constraints(df, constraints)
        assert info["after"] == 0

    def test_mixed_constraints(self):
        df = _sample_df()
        constraints = [
            RangeConstraint(column="A", lo=2.0),
            SumConstraint(columns=["A", "B"], target=6.0, tolerance=0.1),
            InequalityConstraint(coefficients={"A": 1.0}, rhs=4.0, operator="le"),
        ]
        filtered, info = apply_constraints(df, constraints)
        assert len(info["details"]) == 3
        assert info["before"] == 5
