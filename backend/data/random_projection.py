# -*- coding: utf-8 -*-
"""
backend/data/random_projection.py

Johnson-Lindenstrauss (JL) 補題に基づく自動ランダム射影前処理。

## 理論的根拠 (JL Lemma)

JL補題（Johnson & Lindenstrauss, 1984）:
    n点をε-等長埋め込みで保持するには、次元d_jl = O(log(n) / ε²) で十分。

    より厳密には:
        d_jl = johnson_lindenstrauss_min_dim(n_samples, eps)

    元の特徴量次元 d が d_jl より大きい場合のみ、RP で d_jl 次元に削減することに
    意味がある。d <= d_jl の場合は既に「十分低次元」なのでRPは不益（むしろ損）。

## sklearn の実装
    SparseRandomProjection: メモリ効率が高い（高次元・スパース行列に最適）
    GaussianRandomProjection: 理論的保証が厳密（密行列・中規模に最適）

## デフォルト選択戦略
    d > 1000 → SparseRandomProjection （SMILES記述子など超高次元）
    d <= 1000 → GaussianRandomProjection （中規模）

References:
    - Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings
      into a Hilbert space. Contemp. Math., 26, 189-206.
    - sklearn.random_projection.johnson_lindenstrauss_min_dim
    - Achlioptas, D. (2003). Database-friendly random projections.
      JCSS, 66(4), 671-687. [SparseRP の理論的根拠]
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.random_projection import (
    SparseRandomProjection,
    GaussianRandomProjection,
    johnson_lindenstrauss_min_dim,
)
from sklearn.utils.validation import check_is_fitted

from backend.utils.config import RANDOM_STATE

logger = logging.getLogger(__name__)


class JLRandomProjection(BaseEstimator, TransformerMixin):
    """
    Johnson-Lindenstrauss 補題に基づく自動ランダム射影。

    fit() 時に以下を自動判定する:
        1. JL最小次元 d_jl = johnson_lindenstrauss_min_dim(n_samples, eps) を計算
        2. 入力次元 d_in を確認
        3. d_in > d_jl の場合: SparseRP または GaussianRP で d_jl 次元に削減
        4. d_in <= d_jl の場合: passthrough（射影不要）

    Implements: 要件定義書 §3.3 前処理パイプライン拡張（JL Random Projection）

    Args:
        eps: JL歪み許容誤差（0 < eps < 1）。小さいほど精度↑・次元↑。
            推奨値: 0.1（±10%の距離誤差を許容）
        method: "auto" | "sparse" | "gaussian"
            "auto" → d_in > 1000 なら sparse、それ以下なら gaussian
        density: SparseRPの非ゼロ率（"auto" で Achlioptas(2003)の推奨値）
        random_state: 再現性のための乱数シード

    Attributes:
        n_components_: 実際に使用した出力次元数
        jl_min_dim_: JL補題が要求する最小次元（d_jl）
        n_features_in_: 入力特徴量数
        projection_active_: True = 射影適用中, False = passthrough
        projector_: 実際のRandomProjectionオブジェクト（または None）

    Example:
        >>> rp = JLRandomProjection(eps=0.1)
        >>> X_transformed = rp.fit_transform(X_train)
        >>> print(f"削減: {X_train.shape[1]} → {X_transformed.shape[1]} 次元")
    """

    def __init__(
        self,
        eps: float = 0.1,
        method: str = "auto",
        density: str | float = "auto",
        random_state: int | None = RANDOM_STATE,
    ) -> None:
        self.eps = eps
        self.method = method
        self.density = density
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> "JLRandomProjection":
        """
        fit時にJL判定を行い、必要なら射影行列を準備する。

        Args:
            X: 特徴量行列 (n_samples, n_features)
            y: 未使用（sklearn互換のため）

        Returns:
            self
        """
        X_arr = self._to_array(X)
        n_samples, n_features = X_arr.shape

        # JL補題による最小次元を計算
        # johnson_lindenstrauss_min_dim(n_samples, eps) = 4 log(n) / (eps²/2 - eps³/3)
        jl_min = int(johnson_lindenstrauss_min_dim(n_samples, eps=self.eps))

        self.n_features_in_ = n_features
        self.jl_min_dim_ = jl_min

        if n_features <= jl_min:
            # 既に十分低次元 → RPは不要（passthrough）
            self.projection_active_ = False
            self.projector_ = None
            self.n_components_ = n_features
            logger.info(
                f"JLRandomProjection: 射影不要 "
                f"(n_features={n_features} ≤ jl_min_dim={jl_min}, "
                f"n_samples={n_samples}, eps={self.eps})"
            )
        else:
            # 高次元 → RP適用
            self.projection_active_ = True
            method = self._resolve_method(n_features)

            if method == "sparse":
                rp = SparseRandomProjection(
                    n_components=jl_min,
                    density=self.density,
                    eps=self.eps,
                    random_state=self.random_state,
                )
            else:
                rp = GaussianRandomProjection(
                    n_components=jl_min,
                    eps=self.eps,
                    random_state=self.random_state,
                )

            rp.fit(X_arr)
            self.projector_ = rp
            self.n_components_ = jl_min
            logger.info(
                f"JLRandomProjection: 射影適用 "
                f"{n_features} → {jl_min} 次元 "
                f"[method={method}, n_samples={n_samples}, eps={self.eps}]"
            )

        return self

    def transform(self, X: Any, y: Any = None) -> np.ndarray:
        """
        fit時の判定結果に基づいてtransformする。

        Args:
            X: 特徴量行列 (n_samples, n_features)

        Returns:
            変換後の行列 (n_samples, n_components_)
        """
        check_is_fitted(self, ["projection_active_", "n_components_"])
        X_arr = self._to_array(X)

        if not self.projection_active_:
            return X_arr

        projected = self.projector_.transform(X_arr)
        # sparse → dense
        if hasattr(projected, "toarray"):
            projected = projected.toarray()
        return projected

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """変換後の特徴量名を返す（"rp_0", "rp_1", ...）。"""
        check_is_fitted(self, ["n_components_", "projection_active_"])
        if not self.projection_active_ and input_features is not None:
            return np.asarray(input_features)
        return np.array([f"rp_{i}" for i in range(self.n_components_)])

    def _resolve_method(self, n_features: int) -> str:
        """methodパラメータを解決する。"""
        if self.method == "auto":
            # d > 1000 → sparse (Achlioptas2003の効率), それ以下 → gaussian
            return "sparse" if n_features > 1000 else "gaussian"
        return self.method

    @staticmethod
    def _to_array(X: Any) -> np.ndarray:
        """入力をnumpy配列に変換する。sparse matrixも対応。"""
        if hasattr(X, "toarray"):
            return X.toarray()
        if hasattr(X, "values"):  # pd.DataFrame
            return X.values.astype(float)
        return np.asarray(X, dtype=float)

    def summary(self) -> str:
        """設定と結果のサマリー文字列を返す。"""
        check_is_fitted(self, ["projection_active_"])
        if not self.projection_active_:
            return (
                f"JLRandomProjection: 不適用 "
                f"(n_features={self.n_features_in_} ≤ jl_min={self.jl_min_dim_})"
            )
        return (
            f"JLRandomProjection: {self.n_features_in_} → {self.n_components_} 次元 "
            f"[eps={self.eps}, method={self._resolve_method(self.n_features_in_)}]"
        )


def should_apply_random_projection(
    n_features: int,
    n_samples: int,
    eps: float = 0.1,
) -> tuple[bool, int]:
    """
    JL条件を事前チェックするユーティリティ関数。

    Args:
        n_features: 入力特徴量数
        n_samples: サンプル数
        eps: 歪み許容誤差

    Returns:
        (should_apply, jl_min_dim)
        - should_apply: True = RPが有益
        - jl_min_dim: JLが要求する最小次元
    """
    jl_min = int(johnson_lindenstrauss_min_dim(n_samples, eps=eps))
    return n_features > jl_min, jl_min
