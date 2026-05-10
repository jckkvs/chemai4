"""探索空間の定義と候補点生成.

Implements: F-B01〜B05
    Variable: 連続/離散/カテゴリ変数の定義
    SearchSpace: 変数コレクション + 候補点生成（grid/random/LHS/hybrid）
    候補数の推定と自動戦略推奨
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np
import pandas as pd


# ─── 変数型 ───────────────────────────────────────────────
class VarType(str, Enum):
    """変数の型."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


@dataclass
class Variable:
    """探索空間の個別変数定義.

    Attributes:
        name: 変数名（DataFrame列名と一致）
        var_type: continuous / discrete / categorical
        lo: 下限（continuous/discrete）
        hi: 上限（continuous/discrete）
        step: ステップ幅（discrete用、Noneはcontinuous扱い）
        categories: カテゴリ値リスト（categorical用）
    """

    name: str
    var_type: VarType = VarType.CONTINUOUS
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    categories: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.var_type in (VarType.CONTINUOUS, VarType.DISCRETE):
            if self.lo is None or self.hi is None:
                raise ValueError(f"Variable '{self.name}': lo/hi は必須。")
            if self.lo > self.hi:
                raise ValueError(
                    f"Variable '{self.name}': lo({self.lo}) > hi({self.hi})。"
                )
        if self.var_type == VarType.DISCRETE and self.step is None:
            raise ValueError(f"Variable '{self.name}': discrete型にはstepが必須。")
        if self.var_type == VarType.CATEGORICAL:
            if not self.categories or len(self.categories) == 0:
                raise ValueError(
                    f"Variable '{self.name}': categorical型にはcategoriesが必須。"
                )

    @property
    def n_levels(self) -> int:
        """この変数が取りうる値の数."""
        if self.var_type == VarType.CATEGORICAL:
            return len(self.categories)  # type: ignore[arg-type]
        if self.var_type == VarType.DISCRETE:
            return max(1, int(np.ceil((self.hi - self.lo) / self.step)) + 1)  # type: ignore[operator]
        # continuous → グリッドで20分割をデフォルト
        return 20

    def grid_values(self, n_per_dim: int = 20) -> np.ndarray:
        """グリッド値を返す."""
        if self.var_type == VarType.CATEGORICAL:
            return np.array(self.categories)
        if self.var_type == VarType.DISCRETE:
            vals = np.arange(self.lo, self.hi + self.step * 0.5, self.step)  # type: ignore[operator]
            return vals[vals <= self.hi]  # type: ignore[operator]
        # continuous
        return np.linspace(self.lo, self.hi, n_per_dim)  # type: ignore[arg-type]


# ─── 探索空間 ─────────────────────────────────────────────
# 候補数の上限
MAX_GRID_DIRECT = 100_000
MAX_GRID_DOWNSAMPLE = 1_000_000
DEFAULT_RANDOM_SIZE = 100_000


class SearchSpace:
    """探索空間: 変数定義＋候補点生成.

    Usage::
        space = SearchSpace()
        space.add(Variable("x1", VarType.CONTINUOUS, 0, 10))
        space.add(Variable("x2", VarType.DISCRETE, 0, 100, step=10))
        candidates = space.generate_candidates(method="auto", n_max=50000)
    """

    def __init__(self, variables: Sequence[Variable] | None = None) -> None:
        self.variables: list[Variable] = list(variables) if variables else []

    def add(self, v: Variable) -> None:
        """変数を追加."""
        self.variables.append(v)

    @property
    def dim(self) -> int:
        return len(self.variables)

    @property
    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    # ── 候補数推定 ──
    def estimate_grid_size(self, n_per_dim: int = 20) -> int:
        """全組み合わせのグリッド候補数を推定."""
        if not self.variables:
            return 0
        sizes = []
        for v in self.variables:
            if v.var_type == VarType.CATEGORICAL:
                sizes.append(len(v.categories))  # type: ignore[arg-type]
            elif v.var_type == VarType.DISCRETE:
                sizes.append(v.n_levels)
            else:
                sizes.append(n_per_dim)
        total = 1
        for s in sizes:
            total *= s
            if total > 10**12:
                return total  # 早期打ち切り
        return total

    def auto_recommend_method(self, n_per_dim: int = 20) -> str:
        """候補数に応じて推奨生成方法を返す.

        Returns:
            "grid" / "grid_downsample" / "random_lhs"
        """
        est = self.estimate_grid_size(n_per_dim)
        if est <= MAX_GRID_DIRECT:
            return "grid"
        if est <= MAX_GRID_DOWNSAMPLE:
            return "grid_downsample"
        return "random_lhs"

    # ── 候補生成 ──
    def generate_candidates(
        self,
        method: str = "auto",
        n_max: int = DEFAULT_RANDOM_SIZE,
        n_per_dim: int = 20,
        seed: int = 42,
    ) -> pd.DataFrame:
        """候補点DataFrameを生成.

        Args:
            method: "grid" / "random" / "lhs" / "grid_downsample" / "auto"
            n_max: 生成する最大候補数
            n_per_dim: グリッド時の連続変数分割数
            seed: 乱数シード

        Returns:
            候補点DataFrame（列名 = 変数名）
        """
        if not self.variables:
            raise ValueError("変数が定義されていません。")

        if method == "auto":
            method = self.auto_recommend_method(n_per_dim)

        rng = np.random.RandomState(seed)

        if method == "grid":
            return self._generate_grid(n_per_dim)
        elif method == "grid_downsample":
            full = self._generate_grid(n_per_dim)
            if len(full) > n_max:
                idx = rng.choice(len(full), size=n_max, replace=False)
                return full.iloc[idx].reset_index(drop=True)
            return full
        elif method == "random":
            return self._generate_random(n_max, rng)
        elif method == "lhs":
            return self._generate_lhs(n_max, rng)
        elif method == "random_lhs":
            # 半分LHS、半分ランダム
            n_lhs = n_max // 2
            n_rand = n_max - n_lhs
            df_lhs = self._generate_lhs(n_lhs, rng)
            df_rand = self._generate_random(n_rand, rng)
            return pd.concat([df_lhs, df_rand], ignore_index=True)
        else:
            raise ValueError(f"不明な生成方法: {method}")

    def _generate_grid(self, n_per_dim: int = 20) -> pd.DataFrame:
        """全組み合わせグリッド."""
        grids = [v.grid_values(n_per_dim) for v in self.variables]
        est = 1
        for g in grids:
            est *= len(g)
        if est > MAX_GRID_DOWNSAMPLE * 10:
            raise ValueError(
                f"グリッド候補数が{est:,}と大きすぎます。"
                "n_per_dim を減らすか、random/lhs を使用してください。"
            )
        mesh = list(itertools.product(*grids))
        return pd.DataFrame(mesh, columns=self.names)

    def _generate_random(self, n: int, rng: np.random.RandomState) -> pd.DataFrame:
        """一様ランダムサンプリング."""
        data: dict[str, np.ndarray] = {}
        for v in self.variables:
            if v.var_type == VarType.CATEGORICAL:
                data[v.name] = rng.choice(v.categories, size=n)  # type: ignore[arg-type]
            elif v.var_type == VarType.DISCRETE:
                levels = v.grid_values()
                data[v.name] = rng.choice(levels, size=n)
            else:
                data[v.name] = rng.uniform(v.lo, v.hi, size=n)  # type: ignore[arg-type]
        return pd.DataFrame(data)

    def _generate_lhs(self, n: int, rng: np.random.RandomState) -> pd.DataFrame:
        """Latin Hypercube Sampling (LHS).

        数値変数をLHS、カテゴリ変数はランダム。
        """
        numeric_vars = [
            v for v in self.variables
            if v.var_type != VarType.CATEGORICAL
        ]
        cat_vars = [
            v for v in self.variables
            if v.var_type == VarType.CATEGORICAL
        ]

        data: dict[str, np.ndarray] = {}

        # LHS for numeric
        if numeric_vars:
            d = len(numeric_vars)
            # 各次元を n 等分してシャッフル
            intervals = np.zeros((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                intervals[:, j] = (perm + rng.uniform(size=n)) / n

            for j, v in enumerate(numeric_vars):
                lo = float(v.lo)  # type: ignore[arg-type]
                hi = float(v.hi)  # type: ignore[arg-type]
                vals = lo + intervals[:, j] * (hi - lo)
                if v.var_type == VarType.DISCRETE:
                    # 最近接グリッド点にスナップ
                    grid = v.grid_values()
                    idx = np.abs(vals[:, None] - grid[None, :]).argmin(axis=1)
                    vals = grid[idx]
                data[v.name] = vals

        # カテゴリ変数はランダム
        for v in cat_vars:
            data[v.name] = rng.choice(v.categories, size=n)  # type: ignore[arg-type]

        return pd.DataFrame(data, columns=self.names)

    # ── ヘルパー ──
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        margin: float = 0.1,
    ) -> "SearchSpace":
        """DataFrameの統計量から探索空間を自動推定.

        Args:
            df: 既存データ
            columns: 対象列（Noneは全数値列）
            margin: 範囲のマージン（例: 0.1=10%拡張）

        Returns:
            SearchSpace
        """
        if columns is None:
            columns = list(df.select_dtypes(include="number").columns)

        space = cls()
        for col in columns:
            s = df[col]
            if s.dtype == object or str(s.dtype) == "category":
                space.add(Variable(
                    name=col,
                    var_type=VarType.CATEGORICAL,
                    categories=sorted(s.dropna().unique().tolist()),
                ))
            else:
                lo_val = float(s.min())
                hi_val = float(s.max())
                span = hi_val - lo_val if hi_val > lo_val else abs(lo_val) * 0.1 + 1e-6
                lo_ext = lo_val - span * margin
                hi_ext = hi_val + span * margin
                # 整数列はdiscrete
                if pd.api.types.is_integer_dtype(s):
                    step = 1.0
                    lo_ext = float(int(np.floor(lo_ext)))
                    hi_ext = float(int(np.ceil(hi_ext)))
                    space.add(Variable(
                        name=col, var_type=VarType.DISCRETE,
                        lo=lo_ext, hi=hi_ext, step=step,
                    ))
                else:
                    space.add(Variable(
                        name=col, var_type=VarType.CONTINUOUS,
                        lo=round(lo_ext, 6), hi=round(hi_ext, 6),
                    ))
        return space
