"""
backend/doe/design.py

D最適・E最適・I最適 の座標交換アルゴリズム実装。
既存実験点を固定したオーグメンテーション計画に対応。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .candidate import build_model_matrix, generate_candidate_set
from .factor import Factor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 結果データクラス
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DoEResult:
    """実験計画の最適化結果。"""
    design_df: pd.DataFrame       # 全設計点（既存 + 新規）
    is_new: list[bool]            # 各行が新規かどうか
    criterion_value: float        # 最適化基準値
    criterion_name: str           # "D" / "E" / "I"
    d_efficiency: float           # D効率（0〜1）
    model_matrix: np.ndarray      # 最終設計行列
    info: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# オプティマイザ
# ─────────────────────────────────────────────────────────────────────────────

class DoEOptimizer:
    """
    座標交換法による最適実験計画生成。

    Parameters
    ----------
    factors       : 因子リスト
    n_new         : 新規追加する実験数
    criterion     : "D" / "E" / "I" / "MAXIMIN" / "MINIMAX"
    max_candidates: 候補集合の最大サイズ（超過時はランダムサンプリング）
    random_seed   : 乱数シード
    n_starts      : マルチスタート数（局所最適を避ける）
    max_iter      : 1スタートあたりの最大反復数
    existing_df   : 既存実験データ（None の場合は全点を新規最適化）
    """

    def __init__(
        self,
        factors: list[Factor],
        n_new: int,
        criterion: str = "D",
        max_candidates: int = 5000,
        random_seed: int = 42,
        n_starts: int = 5,
        max_iter: int = 300,
        existing_df: pd.DataFrame | None = None,
    ):
        self.factors = factors
        self.n_new = n_new
        self.criterion = criterion.upper()
        self.max_candidates = max_candidates
        self.random_seed = random_seed
        self.n_starts = n_starts
        self.max_iter = max_iter
        self.existing_df = existing_df

        # 候補集合生成
        self.cand_X, self.cand_df = generate_candidate_set(
            factors, max_candidates, random_seed
        )
        logger.info(f"[DoE] 候補集合: {len(self.cand_X)}点, 基準: {self.criterion}")

        # 既存実験の設計行列
        self.existing_X: np.ndarray | None = None
        if existing_df is not None and len(existing_df) > 0:
            factor_names = [f.name for f in factors]
            ex = existing_df.copy()
            # 不足列を中央値 / 第1候補で埋める
            for f in factors:
                if f.name not in ex.columns:
                    ex[f.name] = f.levels[len(f.levels) // 2]
            avail = [c for c in factor_names if c in ex.columns]
            ex_sub = ex[avail].copy()
            for fn in factor_names:
                if fn not in ex_sub.columns:
                    f_ = next(f for f in factors if f.name == fn)
                    ex_sub[fn] = f_.levels[0]
            self.existing_X = build_model_matrix(factors, ex_sub[factor_names])

        # スケール依存性解消 (S3空間): 距離計算用に標準化器を用意
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        if len(self.cand_X) > 0:
            self.cand_X_scaled = self.scaler.fit_transform(self.cand_X)
        else:
            self.cand_X_scaled = self.cand_X

    # ─────────────────────────────────────────────────────
    # 公開メソッド
    # ─────────────────────────────────────────────────────

    def optimize(self) -> DoEResult:
        """マルチスタート座標交換法で最適設計を求める。"""
        rng = np.random.default_rng(self.random_seed)
        n_cand = len(self.cand_X)

        best_score = -np.inf
        best_idx: list[int] = []
        best_X_full: np.ndarray | None = None

        for start in range(self.n_starts):
            seed_s = int(rng.integers(0, 2**31))
            rng_s = np.random.default_rng(seed_s)

            # 初期化: n_new 点をランダム選択
            init_idx = rng_s.choice(n_cand, size=min(self.n_new, n_cand), replace=False)
            cur_idx = list(init_idx)
            cur_X_new = self.cand_X[init_idx].copy()

            for _ in range(self.max_iter):
                improved = False
                for i in range(self.n_new):
                    X_without_i = self._assemble(cur_X_new, exclude_row=i)
                    best_local_score = -np.inf
                    best_local_j = cur_idx[i]

                    for j in range(n_cand):
                        X_trial = np.vstack([X_without_i, self.cand_X[j:j+1]])
                        score = self._score(X_trial)
                        if score > best_local_score:
                            best_local_score = score
                            best_local_j = j

                    if best_local_j != cur_idx[i]:
                        cur_idx[i] = best_local_j
                        cur_X_new[i] = self.cand_X[best_local_j]
                        improved = True

                if not improved:
                    break

            X_final = self._assemble_full(cur_X_new)
            score_final = self._score(X_final)

            if score_final > best_score:
                best_score = score_final
                best_idx = cur_idx.copy()
                best_X_full = X_final

        # 結果DataFrame構築
        return self._build_result(best_idx, best_X_full)

    # ─────────────────────────────────────────────────────
    # 内部メソッド
    # ─────────────────────────────────────────────────────

    def _assemble(self, X_new: np.ndarray, exclude_row: int) -> np.ndarray:
        """既存 + X_new から exclude_row 行を除いた行列を返す。"""
        rows_new = [X_new[r] for r in range(len(X_new)) if r != exclude_row]
        parts = []
        if self.existing_X is not None:
            parts.append(self.existing_X)
        if rows_new:
            parts.append(np.array(rows_new))
        return np.vstack(parts) if parts else np.empty((0, X_new.shape[1]))

    def _assemble_full(self, X_new: np.ndarray) -> np.ndarray:
        parts = []
        if self.existing_X is not None:
            parts.append(self.existing_X)
        parts.append(X_new)
        return np.vstack(parts)

    def _score(self, X: np.ndarray) -> float:
        """基準スコア（最大化方向）。"""
        if len(X) == 0:
            return -np.inf

        # ── 空間充填基準 ──
        if self.criterion == "MAXIMIN":
            # 最小点間距離を最大化 (S3空間で計算)
            from scipy.spatial.distance import pdist
            if len(X) < 2:
                return -np.inf
            X_scaled = self.scaler.transform(X)
            dists = pdist(X_scaled, metric="euclidean")
            return float(np.min(dists)) if len(dists) > 0 else -np.inf

        elif self.criterion == "MINIMAX":
            # 全候補点から最近傍設計点への最大距離を最小化 (S3空間で計算)
            from scipy.spatial.distance import cdist
            if len(X) < 1:
                return -np.inf
            X_scaled = self.scaler.transform(X)
            d = cdist(self.cand_X_scaled, X_scaled, metric="euclidean")
            max_min_dist = float(np.max(np.min(d, axis=1)))
            return -max_min_dist  # 最小化 → 負値で最大化

        # ── 情報行列ベース基準 ──
        XtX = X.T @ X
        try:
            if self.criterion == "D":
                sign, logdet = np.linalg.slogdet(XtX)
                return float(logdet) if sign > 0 else -1e18

            elif self.criterion == "E":
                eigvals = np.linalg.eigvalsh(XtX)
                return float(np.min(eigvals))

            elif self.criterion == "I":
                XtX_inv = np.linalg.pinv(XtX)
                M = self.cand_X.T @ self.cand_X / len(self.cand_X)
                return -float(np.trace(XtX_inv @ M))  # 最小化の負

        except Exception:
            return -1e18
        return -1e18

    def _d_efficiency(self, X: np.ndarray) -> float:
        """D効率 ∈ [0,1]: (det(X'X)/n)^(1/p) を正規化。"""
        n, p = X.shape
        XtX = X.T @ X
        sign, logdet = np.linalg.slogdet(XtX)
        if sign <= 0 or p == 0:
            return 0.0
        return float(np.exp((logdet - p * np.log(n)) / p))

    def _criterion_human(self, X: np.ndarray) -> tuple[float, float]:
        """(criterion_value, d_efficiency) を返す。"""
        d_eff = self._d_efficiency(X)

        if self.criterion == "MAXIMIN":
            from scipy.spatial.distance import pdist
            if len(X) < 2:
                return (0.0, round(d_eff, 4))
            X_scaled = self.scaler.transform(X)
            dists = pdist(X_scaled, metric="euclidean")
            return (round(float(np.min(dists)), 4), round(d_eff, 4))

        elif self.criterion == "MINIMAX":
            from scipy.spatial.distance import cdist
            X_scaled = self.scaler.transform(X)
            d = cdist(self.cand_X_scaled, X_scaled, metric="euclidean")
            max_min_dist = float(np.max(np.min(d, axis=1)))
            return (round(max_min_dist, 4), round(d_eff, 4))

        XtX = X.T @ X
        try:
            if self.criterion == "D":
                sign, logdet = np.linalg.slogdet(XtX)
                return (round(float(logdet), 4) if sign > 0 else -1e18, round(d_eff, 4))
            elif self.criterion == "E":
                eigval = float(np.min(np.linalg.eigvalsh(XtX)))
                return (round(eigval, 4), round(d_eff, 4))
            elif self.criterion == "I":
                XtX_inv = np.linalg.pinv(XtX)
                M = self.cand_X.T @ self.cand_X / len(self.cand_X)
                i_val = float(np.trace(XtX_inv @ M))
                return (round(i_val, 4), round(d_eff, 4))
        except Exception:
            pass
        return (0.0, round(d_eff, 4))

    def _build_result(self, best_idx: list[int], X_full: np.ndarray) -> DoEResult:
        new_df = self.cand_df.iloc[best_idx].reset_index(drop=True)

        factor_names = [f.name for f in self.factors]

        if self.existing_df is not None and len(self.existing_df) > 0:
            ex_cols = [c for c in factor_names if c in self.existing_df.columns]
            existing_part = self.existing_df[ex_cols].copy().reset_index(drop=True)
            result_df = pd.concat([existing_part, new_df[factor_names]], ignore_index=True)
            is_new = [False] * len(existing_part) + [True] * len(new_df)
        else:
            result_df = new_df[factor_names].copy()
            is_new = [True] * len(new_df)

        crit_val, d_eff = self._criterion_human(X_full)

        return DoEResult(
            design_df=result_df,
            is_new=is_new,
            criterion_value=crit_val,
            criterion_name=self.criterion,
            d_efficiency=d_eff,
            model_matrix=X_full,
            info={
                "n_existing": 0 if self.existing_df is None else len(self.existing_df),
                "n_new": self.n_new,
                "n_candidates": len(self.cand_X),
                "n_starts": self.n_starts,
                "n_params": X_full.shape[1] if X_full is not None else 0,
            },
        )
