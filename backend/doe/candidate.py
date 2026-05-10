"""
backend/doe/candidate.py

候補点集合の生成。水準数が多い場合はランダムサンプリングで候補を制限する。
"""
from __future__ import annotations

import itertools
from functools import reduce
from operator import mul

import numpy as np
import pandas as pd

from .factor import Factor


# ─────────────────────────────────────────────────────────────────────────────
# 公開API
# ─────────────────────────────────────────────────────────────────────────────

def generate_candidate_set(
    factors: list[Factor],
    max_candidates: int = 5000,
    random_seed: int = 42,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    候補点集合を生成する。

    全水準の直積が max_candidates 以下の場合は全網羅（Full Factorial）、
    それを超える場合はランダムサンプリングで max_candidates 件に絞る。

    Returns:
        X_model: np.ndarray (n_cand, p) - モデル行列（切片を含む）
        df_candidates: pd.DataFrame - 元の因子値
    """
    levels_list = [f.levels for f in factors]
    n_cols = [len(lvls) for lvls in levels_list]
    total = reduce(mul, n_cols, 1)

    rng = np.random.default_rng(random_seed)

    if total <= max_candidates:
        # Full Factorial
        prod = list(itertools.product(*levels_list))
        rows = list(prod)
    else:
        # Random sampling: サンプルを重複なし整数インデックスから生成
        sample_size = min(max_candidates, total)
        # 大きな全体空間からランダムに整数インデックスをサンプリング
        # total が巨大でも整数演算で各因子の水準を逆引きできる
        if total < 2**53:  # numpy の整数精度内
            idx_arr = rng.choice(int(total), size=sample_size, replace=False)
        else:
            # 超巨大空間: 重複を気にせずサンプリング
            idx_arr = rng.integers(0, total, size=sample_size)

        rows = []
        for flat_idx in idx_arr:
            row = []
            remaining = int(flat_idx)
            for lvls in reversed(levels_list):
                nl = len(lvls)
                row.insert(0, lvls[remaining % nl])
                remaining //= nl
            rows.append(row)

    df = pd.DataFrame(rows, columns=[f.name for f in factors])
    X = build_model_matrix(factors, df)
    return X, df


def build_model_matrix(factors: list[Factor], df: pd.DataFrame) -> np.ndarray:
    """切片 + 主効果モデル行列を構築する。"""
    n = len(df)
    cols = [np.ones((n, 1), dtype=float)]  # 切片
    for f in factors:
        vals = df[f.name].values
        encoded = f.encode(vals)
        cols.append(encoded)
    return np.hstack(cols)


def col_names(factors: list[Factor]) -> list[str]:
    """モデル行列列名（切片 + 各因子）。"""
    names = ["Intercept"]
    for f in factors:
        names.extend(f.col_names())
    return names
