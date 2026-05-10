"""
backend/doe/factor.py

実験計画法の因子定義（連続値 / カテゴリ）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class FactorType(str, Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


@dataclass
class Factor:
    """1つの実験因子を表す。"""

    name: str
    type: FactorType
    # --- 連続値 ---
    low: float = 0.0
    high: float = 1.0
    n_levels: int = 5   # 候補点として使う水準数
    # --- カテゴリ ---
    categories: list[Any] = field(default_factory=list)

    # ─────────────────────────────────────────────────────
    # ファクトリ
    # ─────────────────────────────────────────────────────
    @classmethod
    def continuous(cls, name: str, low: float, high: float, n_levels: int = 5) -> "Factor":
        return cls(name=name, type=FactorType.CONTINUOUS, low=low, high=high, n_levels=n_levels)

    @classmethod
    def categorical(cls, name: str, categories: list[Any]) -> "Factor":
        return cls(name=name, type=FactorType.CATEGORICAL, categories=list(categories))

    # ─────────────────────────────────────────────────────
    # プロパティ
    # ─────────────────────────────────────────────────────
    @property
    def levels(self) -> list:
        """候補水準のリスト。"""
        if self.type == FactorType.CONTINUOUS:
            return [round(v, 10) for v in np.linspace(self.low, self.high, self.n_levels)]
        return list(self.categories)

    @property
    def n_cols(self) -> int:
        """設計行列中の列数（効果コーディング）。"""
        if self.type == FactorType.CONTINUOUS:
            return 1
        return max(1, len(self.categories) - 1)

    def col_names(self) -> list[str]:
        """設計行列の列名。"""
        if self.type == FactorType.CONTINUOUS:
            return [self.name]
        return [f"{self.name}[{c}]" for c in self.categories[:-1]]

    # ─────────────────────────────────────────────────────
    # コーディング
    # ─────────────────────────────────────────────────────
    def encode(self, values: np.ndarray) -> np.ndarray:
        """
        因子値を設計行列列にエンコードする。

        連続値 → [-1, 1] に正規化（1列）
        カテゴリ → 効果コーディング k-1 列
          基準カテゴリ(最後)は全 -1
        """
        if self.type == FactorType.CONTINUOUS:
            half = (self.high - self.low) / 2.0
            mid = (self.high + self.low) / 2.0
            if half == 0:
                return np.zeros((len(values), 1), dtype=float)
            return ((np.asarray(values, dtype=float) - mid) / half).reshape(-1, 1)

        cats = self.categories
        k = len(cats)
        if k == 0:
            return np.ones((len(values), 1), dtype=float)
        if k == 1:
            return np.ones((len(values), 1), dtype=float)

        cat_idx = {c: i for i, c in enumerate(cats)}
        n = len(values)
        X = np.zeros((n, k - 1), dtype=float)
        for row, v in enumerate(values):
            idx = cat_idx.get(v, k - 1)  # 未知は基準扱い
            if idx < k - 1:
                X[row, idx] = 1.0
            else:
                X[row, :] = -1.0
        return X

    def decode(self, encoded_row: np.ndarray) -> Any:
        """エンコード済み行から元の水準値を復元（デバッグ用）。"""
        if self.type == FactorType.CONTINUOUS:
            half = (self.high - self.low) / 2.0
            mid = (self.high + self.low) / 2.0
            return float(encoded_row[0]) * half + mid
        cats = self.categories
        col = np.round(encoded_row[:len(cats) - 1])
        for i, v in enumerate(col):
            if v == 1:
                return cats[i]
        return cats[-1]
