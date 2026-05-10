"""制約処理モジュール.

Implements: F-C01〜C06
    RangeConstraint: 個別変数の範囲制約
    SumConstraint: 合計制約（A + B + C = 100 等）
    InequalityConstraint: 線形不等式制約
    AtLeastOneConstraint: 少なくとも1つ > 0
    CustomConstraint: Python式制約（高度ユーザー向け）
    apply_constraints: 候補DFにフィルタを適用
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import pandas as pd


class Constraint(abc.ABC):
    """制約の基底クラス."""

    @abc.abstractmethod
    def is_satisfied(self, row: pd.Series) -> bool:
        """単一行が制約を満たすか."""

    @abc.abstractmethod
    def mask(self, df: pd.DataFrame) -> pd.Series:
        """DataFrameの各行が制約を満たすかのboolマスク."""

    @abc.abstractmethod
    def describe(self) -> str:
        """人間が読める制約の説明."""


@dataclass
class RangeConstraint(Constraint):
    """個別変数の範囲制約: lo ≤ x_i ≤ hi.

    Attributes:
        column: 対象列名
        lo: 下限（Noneは制約なし）
        hi: 上限（Noneは制約なし）
    """

    column: str
    lo: float | None = None
    hi: float | None = None

    def is_satisfied(self, row: pd.Series) -> bool:
        val = row[self.column]
        if self.lo is not None and val < self.lo:
            return False
        if self.hi is not None and val > self.hi:
            return False
        return True

    def mask(self, df: pd.DataFrame) -> pd.Series:
        m = pd.Series(True, index=df.index)
        if self.lo is not None:
            m &= df[self.column] >= self.lo
        if self.hi is not None:
            m &= df[self.column] <= self.hi
        return m

    def describe(self) -> str:
        parts = []
        if self.lo is not None:
            parts.append(f"{self.lo}")
        parts.append(f"≤ {self.column} ≤")
        if self.hi is not None:
            parts.append(f"{self.hi}")
        return " ".join(parts) if self.hi is not None else f"{self.column} ≥ {self.lo}"


@dataclass
class SumConstraint(Constraint):
    """合計制約: |Σ x_i - target| ≤ tolerance.

    Attributes:
        columns: 合計対象列リスト
        target: 目標合計値
        tolerance: 許容誤差（デフォルト1e-6）
    """

    columns: list[str]
    target: float
    tolerance: float = 1e-6

    def is_satisfied(self, row: pd.Series) -> bool:
        total = sum(row[c] for c in self.columns)
        return abs(total - self.target) <= self.tolerance

    def mask(self, df: pd.DataFrame) -> pd.Series:
        total = df[self.columns].sum(axis=1)
        return (total - self.target).abs() <= self.tolerance

    def describe(self) -> str:
        cols = " + ".join(self.columns)
        if self.tolerance < 1e-3:
            return f"{cols} = {self.target}"
        return f"|{cols} - {self.target}| ≤ {self.tolerance}"


@dataclass
class InequalityConstraint(Constraint):
    """線形不等式制約: Σ(coeff_i * x_i) ≤ rhs.

    Attributes:
        coefficients: {列名: 係数} の辞書
        rhs: 右辺値
        operator: "le" (≤) / "ge" (≥) / "lt" (<) / "gt" (>)
    """

    coefficients: dict[str, float]
    rhs: float
    operator: str = "le"

    def _compute(self, df_or_row: pd.DataFrame | pd.Series) -> Any:
        if isinstance(df_or_row, pd.Series):
            return sum(df_or_row[c] * coeff for c, coeff in self.coefficients.items())
        return sum(
            df_or_row[c] * coeff for c, coeff in self.coefficients.items()
        )

    def is_satisfied(self, row: pd.Series) -> bool:
        val = self._compute(row)
        if self.operator == "le":
            return val <= self.rhs
        elif self.operator == "ge":
            return val >= self.rhs
        elif self.operator == "lt":
            return val < self.rhs
        elif self.operator == "gt":
            return val > self.rhs
        raise ValueError(f"不明な演算子: {self.operator}")

    def mask(self, df: pd.DataFrame) -> pd.Series:
        val = sum(df[c] * coeff for c, coeff in self.coefficients.items())
        ops = {"le": val.__le__, "ge": val.__ge__, "lt": val.__lt__, "gt": val.__gt__}
        if self.operator not in ops:
            raise ValueError(f"不明な演算子: {self.operator}")
        return ops[self.operator](self.rhs)

    def describe(self) -> str:
        terms = []
        for c, coeff in self.coefficients.items():
            if coeff == 1.0:
                terms.append(c)
            elif coeff == -1.0:
                terms.append(f"-{c}")
            else:
                terms.append(f"{coeff}*{c}")
        lhs = " + ".join(terms)
        sym = {"le": "≤", "ge": "≥", "lt": "<", "gt": ">"}
        return f"{lhs} {sym.get(self.operator, self.operator)} {self.rhs}"


@dataclass
class AtLeastNConstraint(Constraint):
    """少なくともN個が閾値超: sum(x_{cols} > threshold) >= min_count.

    Attributes:
        columns: 対象列リスト
        min_count: 最低必要数（デフォルト1）
        threshold: 閾値（デフォルト0）
        label: UIでの表示名（オプション）
    """

    columns: list[str]
    min_count: int = 1
    threshold: float = 0.0
    label: str = ""

    def is_satisfied(self, row: pd.Series) -> bool:
        count = sum(1 for c in self.columns if row[c] > self.threshold)
        return count >= self.min_count

    def mask(self, df: pd.DataFrame) -> pd.Series:
        return (df[self.columns] > self.threshold).sum(axis=1) >= self.min_count

    def describe(self) -> str:
        cols = ", ".join(self.columns)
        prefix = f"[{self.label}] " if self.label else ""
        return f"{prefix}[{cols}] から少なくとも{self.min_count}つ > {self.threshold}"


# 後方互換エイリアス
AtLeastOneConstraint = AtLeastNConstraint


@dataclass
class CustomConstraint(Constraint):
    """Python式による任意制約（高度ユーザー向け）.

    Attributes:
        expression: Python式文字列。列名を変数として使用可能。
            例: "A * B <= 50", "(A > 0) | (B > 0)"
    """

    expression: str

    def is_satisfied(self, row: pd.Series) -> bool:
        local_vars = {c: row[c] for c in row.index}
        try:
            return bool(eval(self.expression, {"__builtins__": {}}, local_vars))  # noqa: S307
        except Exception:
            return False

    def mask(self, df: pd.DataFrame) -> pd.Series:
        try:
            return df.eval(self.expression).astype(bool)
        except Exception:
            # fallback: 行単位
            return df.apply(self.is_satisfied, axis=1)

    def describe(self) -> str:
        return f"カスタム: {self.expression}"


def apply_constraints(
    df: pd.DataFrame,
    constraints: list[Constraint],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """候補DFに制約リストを適用しフィルタリング.

    Args:
        df: 候補点DataFrame
        constraints: 制約リスト

    Returns:
        (フィルタ後のDF, {"before": 元の行数, "after": フィルタ後, "removed": 除去数,
                          "details": [{制約名: 除去数}]})
    """
    n_before = len(df)
    current = df.copy()
    details: list[dict[str, Any]] = []

    for c in constraints:
        m = c.mask(current)
        n_removed = int((~m).sum())
        details.append({"constraint": c.describe(), "removed": n_removed})
        current = current[m].reset_index(drop=True)

    return current, {
        "before": n_before,
        "after": len(current),
        "removed": n_before - len(current),
        "details": details,
    }
