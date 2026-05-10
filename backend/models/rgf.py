"""
backend/models/rgf.py

Regularized Greedy Forest (RGF) のフルスクラッチ実装。
sklearn 完全互換（clone, GridSearchCV, Pipeline 対応）。

Reference:
  Johnson & Zhang (2014). "Learning Nonlinear Functions Using Regularized Greedy Forest"
  https://arxiv.org/abs/1109.0887

アルゴリズム概要:
  通常の勾配ブースティング(GBM)が既存ツリーを変更しない「前向き段階的」なのに対し、
  RGFは**全ノード重みを毎ステップ全面更新（fully-corrective）** する。

  擬似コード:
    1. 木の集合 T = {} で初期化
    2. for m in 1..n_iter:
         a. 新しい弱学習器(決定木 t_m)を残差に対してグリーディに選択・追加
         b. 全木の葉ノード重みを L1/L2 正則化付き最小二乗で全面再最適化
         c. F(x) = Σ_k w_k * φ_k(x)  (φ_k は各葉の指示関数)
    3. 予測: F(x)

  実装上の近似:
    - 各ステップで最大 max_leaf_nodes 葉を持つ二分木を1本追加
    - 全ノード重みは Ridge 回帰相当の閉形式解で全面更新
    - 正則化: λ_l1 (L1) と λ_l2 (L2) をサポート
    - 分類: 二値=LogLoss、多クラス=one-vs-rest
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────────

def _to_numpy(X: Any) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.values.astype(float)
    return np.asarray(X, dtype=float)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────────────────────
# RGF コア (回帰用)
# ─────────────────────────────────────────────────────────────

class _RGFCore:
    """
    RGF の中核ロジック。
    葉インジケータ行列の管理と、全面更新（fully-corrective）の重み最適化を担う。
    """

    def _init_forest_state(self) -> None:
        """フォレスト状態の初期化。"""
        self.trees_: list[DecisionTreeRegressor] = []
        self.leaf_offsets_: list[int] = []  # 各木の葉インデックスのオフセット
        self.leaf_id_maps_: list[dict[int, int]] = []  # 各木の葉ID→ローカルIDマッピング
        self.leaf_counts_: list[int] = []  # 各木の葉数
        self.weights_: np.ndarray = np.array([])  # 全葉重み
        self._total_leaves: int = 0

    def _register_tree_leaves(self, tree: DecisionTreeRegressor, X_train: np.ndarray) -> int:
        """
        新しい木のfit直後に呼び出し、葉IDマッピングを記録する。
        Returns: この木の葉数
        """
        leaf_ids = tree.apply(X_train)
        unique = np.unique(leaf_ids)
        local_map = {int(lid): j for j, lid in enumerate(unique)}
        self.leaf_id_maps_.append(local_map)
        n_leaves = len(unique)
        self.leaf_counts_.append(n_leaves)
        return n_leaves

    def _get_leaf_indicators(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        全木の葉インジケータ行列 Φ を構築。
        Φ[i, k] = 1 iff サンプル i が葉 k に落ちる
        fit時に記録した葉IDマッピングを使用し、train/predict間の列数整合を保証。

        Returns:
            shape (n_samples, total_leaves)
        """
        if not self.trees_:
            return np.zeros((X.shape[0], 0))

        cols = []
        for tree, leaf_map, n_leaves in zip(self.trees_, self.leaf_id_maps_, self.leaf_counts_):
            leaf_ids = tree.apply(X)               # shape (n,)
            ind = np.zeros((X.shape[0], n_leaves), dtype=np.float32)
            for i, lid in enumerate(leaf_ids):
                local = leaf_map.get(int(lid))
                if local is not None:
                    ind[i, local] = 1.0
                # predict時にtrain時に見なかった葉に落ちた場合は0ベクトル（安全）
            cols.append(ind)

        return np.hstack(cols)     # (n, total_leaves)

    def _update_weights(
        self,
        Phi: np.ndarray,
        residuals: np.ndarray,
        lambda_l2: float,
    ) -> None:
        """
        全面更新: 正則化付き最小二乗 (Ridge) で全葉重みを再最適化。

        min_w  ||Φw - r||²  +  λ ||w||²
        解: w = (Φ'Φ + λI)⁻¹ Φ'r

        大規模時のため、葉数 > max_solve_dim の場合は疑似逆行列で近似。
        """
        n_leaves = Phi.shape[1]
        if n_leaves == 0:
            return

        A = Phi.T @ Phi + lambda_l2 * np.eye(n_leaves)
        b = Phi.T @ residuals
        try:
            self.weights_ = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.weights_ = np.linalg.lstsq(A, b, rcond=None)[0]

    def _predict_from_weights(self, X: np.ndarray) -> np.ndarray:
        """全葉重みから予測値を計算。"""
        if not self.trees_:
            return np.zeros(X.shape[0])
        Phi = self._get_leaf_indicators(X)
        if Phi.shape[1] == 0 or len(self.weights_) == 0:
            return np.zeros(X.shape[0])
        w = self.weights_
        # 重みサイズの整合
        min_len = min(Phi.shape[1], len(w))
        return Phi[:, :min_len] @ w[:min_len]


# ─────────────────────────────────────────────────────────────
# RGFRegressor
# ─────────────────────────────────────────────────────────────

class RGFRegressor(_RGFCore, BaseEstimator, RegressorMixin):
    """
    Regularized Greedy Forest Regressor。
    (Johnson & Zhang 2014, https://arxiv.org/abs/1109.0887)

    Args:
        n_estimators: 追加する木の本数
        max_leaf_nodes: 各木の最大葉数（木の複雑さを制御）
        learning_rate: 各木のスケーリング係数（≠GBMのlr; ここではΦwへの係数）
        lambda_l2: L2 正則化係数（全面更新の Ridge ペナルティ）
        lambda_l1: L1 正則化係数（近似：重みの閾値処理で実現）
        loss: "squared" のみ（将来拡張用）
        min_samples_leaf: 葉の最小サンプル数
        max_features: 各木の特徴量サンプリング
        subsample: 各ステップで使うサンプル割合
        random_state: 乱数シード
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_leaf_nodes: int = 32,
        learning_rate: float = 0.1,
        lambda_l2: float = 1.0,
        lambda_l1: float = 0.0,
        min_samples_leaf: int = 5,
        max_features: Any = "sqrt",
        subsample: float = 1.0,
        loss: str = "squared",
        random_state: int | None = None,
    ) -> None:
        self.n_estimators   = n_estimators
        self.max_leaf_nodes = max_leaf_nodes
        self.learning_rate  = learning_rate
        self.lambda_l2      = lambda_l2
        self.lambda_l1      = lambda_l1
        self.min_samples_leaf = min_samples_leaf
        self.max_features   = max_features
        self.subsample      = subsample
        self.loss           = loss
        self.random_state   = random_state

    def fit(self, X: Any, y: Any) -> "RGFRegressor":
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y, dtype=float).ravel()
        n, p  = X_arr.shape

        rng = np.random.default_rng(self.random_state)
        self._init_forest_state()
        self.init_value_ = float(np.mean(y_arr))

        for m in range(self.n_estimators):
            # 現在の予測値と残差
            F = self._predict_from_weights(X_arr) + self.init_value_
            residuals = y_arr - F

            # サブサンプリング
            if self.subsample < 1.0:
                n_sub = max(2, int(n * self.subsample))
                idx = rng.choice(n, size=n_sub, replace=False)
                X_sub, r_sub = X_arr[idx], residuals[idx]
            else:
                X_sub, r_sub = X_arr, residuals

            # --- Step a: 新しい木を残差にフィット ---
            seed = int(rng.integers(0, 2**31))
            tree = DecisionTreeRegressor(
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed,
            )
            tree.fit(X_sub, r_sub)
            self.trees_.append(tree)
            self.leaf_offsets_.append(self._total_leaves)

            # 葉IDマッピングを記録し、葉数を更新
            n_new_leaves = self._register_tree_leaves(tree, X_arr)
            self._total_leaves += n_new_leaves

            # --- Step b: 全面更新 ---
            Phi = self._get_leaf_indicators(X_arr)  # (n, total_leaves)
            full_residuals = y_arr - self.init_value_

            self._update_weights(Phi, full_residuals, self.lambda_l2)

            # L1 近似: ソフト閾値処理
            if self.lambda_l1 > 0:
                self.weights_ = np.sign(self.weights_) * np.maximum(
                    0.0, np.abs(self.weights_) - self.lambda_l1
                )

        self.n_features_in_ = X_arr.shape[1]
        logger.info(
            f"RGFRegressor: {self.n_estimators}本, 総葉数={self._total_leaves}, "
            f"λ_l2={self.lambda_l2}"
        )
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "trees_")
        X_arr = _to_numpy(X)
        return self._predict_from_weights(X_arr) + self.init_value_


# ─────────────────────────────────────────────────────────────
# RGFClassifier
# ─────────────────────────────────────────────────────────────

class RGFClassifier(_RGFCore, BaseEstimator, ClassifierMixin):
    """
    Regularized Greedy Forest Classifier。
    二値分類: LogLoss に対する RGF
    多クラス分類: One-vs-Rest で各クラスに RGFRegressor を適用

    Args: RGFRegressor と同様のパラメータ
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_leaf_nodes: int = 32,
        learning_rate: float = 0.1,
        lambda_l2: float = 1.0,
        lambda_l1: float = 0.0,
        min_samples_leaf: int = 5,
        max_features: Any = "sqrt",
        subsample: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators   = n_estimators
        self.max_leaf_nodes = max_leaf_nodes
        self.learning_rate  = learning_rate
        self.lambda_l2      = lambda_l2
        self.lambda_l1      = lambda_l1
        self.min_samples_leaf = min_samples_leaf
        self.max_features   = max_features
        self.subsample      = subsample
        self.random_state   = random_state

    def fit(self, X: Any, y: Any) -> "RGFClassifier":
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y).ravel()

        self.classes_   = np.unique(y_arr)
        self.n_classes_ = len(self.classes_)
        y_int = np.searchsorted(self.classes_, y_arr)
        K = self.n_classes_

        if K == 2:
            # 二値: LogLoss 最小化
            self._fit_binary(X_arr, y_int)
        else:
            # 多クラス: One-vs-Rest
            self.ovr_models_: list[RGFRegressor] = []
            for k in range(K):
                y_k = (y_int == k).astype(float)
                m = RGFRegressor(
                    n_estimators=self.n_estimators,
                    max_leaf_nodes=self.max_leaf_nodes,
                    learning_rate=self.learning_rate,
                    lambda_l2=self.lambda_l2,
                    lambda_l1=self.lambda_l1,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    subsample=self.subsample,
                    random_state=self.random_state,
                )
                m.fit(X_arr, y_k)
                self.ovr_models_.append(m)

        self.n_features_in_ = X_arr.shape[1]
        self.binary_ = (K == 2)
        return self

    def _fit_binary(self, X_arr: np.ndarray, y_int: np.ndarray) -> None:
        """二値分類 (LogLoss) の RGF フィット。"""
        n = X_arr.shape[0]
        rng = np.random.default_rng(self.random_state)
        self._init_forest_state()

        p_mean  = y_int.mean()
        p_mean  = np.clip(p_mean, 1e-7, 1 - 1e-7)
        self.F0_ = float(np.log(p_mean / (1 - p_mean)))

        for m in range(self.n_estimators):
            F = self._predict_from_weights(X_arr) + self.F0_
            p = _sigmoid(F)
            # LogLoss の疑似残差 (= y - p)
            residuals = y_int.astype(float) - p

            if self.subsample < 1.0:
                n_sub = max(2, int(n * self.subsample))
                idx   = rng.choice(n, size=n_sub, replace=False)
                X_sub, r_sub = X_arr[idx], residuals[idx]
            else:
                X_sub, r_sub = X_arr, residuals

            seed = int(rng.integers(0, 2**31))
            tree = DecisionTreeRegressor(
                max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed,
            )
            tree.fit(X_sub, r_sub)
            self.trees_.append(tree)
            self.leaf_offsets_.append(self._total_leaves)
            n_new = self._register_tree_leaves(tree, X_arr)
            self._total_leaves += n_new

            # 全面更新: Φw を y-F0 の疑似残差に近似
            # 注: y_intは0/1なのでlogit変換すると±infになるため、
            #     回帰と同様に (y - init_value) をターゲットにする
            Phi = self._get_leaf_indicators(X_arr)
            full_residuals = y_int.astype(float) - _sigmoid(np.full(len(y_int), self.F0_))
            self._update_weights(Phi, full_residuals, self.lambda_l2)
            if self.lambda_l1 > 0:
                self.weights_ = np.sign(self.weights_) * np.maximum(
                    0.0, np.abs(self.weights_) - self.lambda_l1
                )

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "binary_")
        X_arr = _to_numpy(X)
        if self.binary_:
            F = self._predict_from_weights(X_arr) + self.F0_
            p1 = _sigmoid(F)
            return np.stack([1 - p1, p1], axis=1)
        else:
            scores = np.stack(
                [m.predict(X_arr) for m in self.ovr_models_], axis=1
            )
            return _softmax(scores)

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
