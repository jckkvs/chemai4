"""
backend/models/linear_tree.py

Linear Tree / Linear Forest / Linear Boost のフルスクラッチ実装。
sklearn 完全互換（clone, GridSearchCV, Pipeline 対応）。

実装クラス:
  LinearTreeRegressor    - 葉に線形回帰モデルを持つ決定木（回帰）
  LinearTreeClassifier   - 葉に線形分類モデルを持つ決定木（分類）
  LinearForestRegressor  - LinearTreeRegressor のバギングアンサンブル
  LinearForestClassifier - LinearTreeClassifier のバギングアンサンブル
  LinearBoostRegressor   - LinearTree を基底とするブースティング（回帰）
  LinearBoostClassifier  - LinearTree を基底とするブースティング（分類）

Base Estimator:
  回帰: Ridge（RidgeTree）, Lasso, ElasticNet, BayesianRidge, HuberRegressor 等
  分類: LogisticRegression, RidgeClassifier 等
  any sklearn estimator が指定可能。

Algorithm (LinearTreeRegressor):
  fit(X, y):
    1. ルートノードで base_estimator をフィット
    2. 残差 r = y - base_estimator.predict(X) を計算
    3. 各特徴量・分割点で左右の base_estimator をフィットし
       残差二乗和（或いはGini）が最小になる分割を選択
    4. 分割後の子ノードで再帰的に繰り返す (深さ・サンプル数で停止)
  
  predict(X):
    各サンプルを葉まで辿り、葉の線形モデルで予測
"""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import (
    BaseEstimator, RegressorMixin, ClassifierMixin, clone,
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# ノードデータ構造
# ─────────────────────────────────────────────────────────────

@dataclass
class _Node:
    """決定木の1ノード。"""
    # リーフなら linear_model を持つ
    linear_model: Any = None

    # 内部ノードなら分割情報を持つ
    feature_idx: int = -1
    threshold: float = 0.0
    left: "_Node | None" = None
    right: "_Node | None" = None

    # 統計（デバッグ用）
    n_samples: int = 0
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


# ─────────────────────────────────────────────────────────────
# 内部ユーティリティ
# ─────────────────────────────────────────────────────────────

def _to_numpy(X: Any) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.values.astype(float)
    arr = np.asarray(X)
    if arr.dtype.kind not in ("f", "i", "u"):
        arr = arr.astype(float)
    return arr


def _fit_linear(base: Any, X: np.ndarray, y: np.ndarray) -> Any:
    """base_estimatorをcloneしてfitし返す。失敗するとNoneを返す。"""
    if X.shape[0] == 0:
        return None
    m = clone(base)
    try:
        m.fit(X, y)
        return m
    except Exception as e:
        logger.debug(f"_fit_linear失敗: {e}")
        return None


def _predict_linear(model: Any, X: np.ndarray) -> np.ndarray:
    """線形モデルで予測。失敗時はゼロ配列。"""
    try:
        return model.predict(X).ravel()
    except Exception:
        return np.zeros(X.shape[0])


def _predict_proba_linear(model: Any, X: np.ndarray, n_classes: int) -> np.ndarray:
    """分類モデルで確率予測。失敗時は均等分布。n_classesに合わせてパディング/スライスする。"""
    n_samples = X.shape[0]
    try:
        if hasattr(model, "predict_proba"):
            raw = model.predict_proba(X)
            # 出力クラス数が期待値と異なる場合は調整する
            if raw.shape[1] == n_classes:
                return raw
            # 少ない → ゼロパディング
            if raw.shape[1] < n_classes:
                padded = np.zeros((n_samples, n_classes))
                # モデルが学習したクラスのインデックスを取得
                model_classes = getattr(model, "classes_", None)
                if model_classes is not None:
                    for ci, cls_id in enumerate(model_classes):
                        if 0 <= cls_id < n_classes:
                            padded[:, int(cls_id)] = raw[:, ci]
                else:
                    padded[:, :raw.shape[1]] = raw
                return padded
            # 多い → スライス
            return raw[:, :n_classes]
        else:
            pred = model.predict(X)
            proba = np.zeros((X.shape[0], n_classes))
            for i, p in enumerate(pred):
                cls_idx = int(p)
                if 0 <= cls_idx < n_classes:
                    proba[i, cls_idx] = 1.0
            return proba
    except Exception:
        return np.full((n_samples, n_classes), 1.0 / n_classes)



def _mse(y: np.ndarray) -> float:
    """平均二乗誤差。"""
    if len(y) == 0:
        return 0.0
    return float(np.mean((y - np.mean(y)) ** 2))


def _gini(y: np.ndarray, n_classes: int) -> float:
    """ジニ不純度。"""
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y.astype(int), minlength=n_classes)
    probs = counts / len(y)
    return float(1.0 - np.sum(probs ** 2))


def _linear_residual_mse(base: Any, X: np.ndarray, y: np.ndarray) -> float:
    """線形モデルをフィットした後の残差MSE。"""
    if X.shape[0] < 2:
        return _mse(y)
    m = _fit_linear(base, X, y)
    if m is None:
        return _mse(y)
    pred = _predict_linear(m, X)
    return float(np.mean((y - pred) ** 2))


def _linear_gini(base: Any, X: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    """線形モデル予測を使ったGini（分類用）。"""
    if X.shape[0] < 2:
        return _gini(y, n_classes)
    m = _fit_linear(base, X, y)
    if m is None:
        return _gini(y, n_classes)
    try:
        pred = m.predict(X).astype(int)
        return _gini(pred, n_classes)
    except Exception:
        return _gini(y, n_classes)


# ─────────────────────────────────────────────────────────────
# 木構築コア
# ─────────────────────────────────────────────────────────────

class _LinearTreeCore:
    """
    LinearTree の木構築ロジックを共有するMixin。
    回帰・分類の両方に使用。
    """

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
    ) -> _Node:
        n_samples = X.shape[0]
        node = _Node(n_samples=n_samples, depth=depth)

        # 停止条件: 深さ上限 or サンプル数下限
        stop = (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or n_samples <= self.min_samples_leaf * 2
        )

        if stop:
            node.linear_model = _fit_linear(
                getattr(self, "_base_ref", self.base_estimator), X, y
            )
            return node

        # 最適分割を探索
        best_score = np.inf
        best_feat  = -1
        best_thr   = None
        best_left_mask: np.ndarray | None = None

        n_features = X.shape[1]
        node_id = id(node) % 10000  # ノード毎の疑似一意ID
        feat_indices = self._get_feature_indices(n_features, node_depth=depth, node_id=node_id)

        for feat_idx in feat_indices:
            col = X[:, feat_idx]
            unique_vals = np.unique(col)
            if len(unique_vals) < 2:
                continue

            # 候補しきい値: 隣接ユニーク値の中点 (最大 max_bins 個)
            thresholds = self._get_thresholds(unique_vals)

            for thr in thresholds:
                left_mask  = col <= thr
                right_mask = ~left_mask
                nl = left_mask.sum()
                nr = right_mask.sum()
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue

                score = self._split_score(X, y, left_mask, right_mask)

                if score < best_score:
                    best_score      = score
                    best_feat       = feat_idx
                    best_thr        = thr
                    best_left_mask  = left_mask

        if best_feat == -1 or best_left_mask is None:
            # 有効分割なし → 葉
            node.linear_model = _fit_linear(self.base_estimator, X, y)
            return node

        # ノードにも線形モデルを保持（predict フォールバック用）
        node.linear_model = _fit_linear(
            getattr(self, "_base_ref", self.base_estimator), X, y
        )
        node.feature_idx  = best_feat
        node.threshold    = best_thr  # type: ignore[assignment]

        right_mask = ~best_left_mask
        node.left  = self._build_tree(X[best_left_mask],  y[best_left_mask],  depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _get_feature_indices(self, n_features: int, node_depth: int = 0, node_id: int = 0) -> np.ndarray:
        """サンプリングする特徴量インデックスを返す。ノードごとに異なるシードを使用。"""
        if self.max_features is None:
            return np.arange(n_features)
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n_features))
        elif isinstance(self.max_features, int):
            k = min(n_features, self.max_features)
        else:
            k = n_features

        # ノードごとに異なるシードで特徴量をサンプリング（毎ノード同一特徴量のバイアスを防ぐ）
        base_seed = (self.random_state or 0) + node_depth * 1000 + node_id
        rng = np.random.default_rng(base_seed)
        return rng.choice(n_features, size=k, replace=False)

    def _get_thresholds(self, unique_vals: np.ndarray) -> np.ndarray:
        """候補しきい値リストを返す。"""
        midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2.0
        if len(midpoints) > self.max_bins:
            idx = np.linspace(0, len(midpoints) - 1, self.max_bins, dtype=int)
            midpoints = midpoints[idx]
        return midpoints

    def _split_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
    ) -> float:
        raise NotImplementedError

    def _route_to_leaf(self, node: _Node, x: np.ndarray) -> _Node:
        """1サンプル x をルートから葉まで辿る。"""
        if node.is_leaf:
            return node
        if x[node.feature_idx] <= node.threshold:
            return self._route_to_leaf(node.left, x)  # type: ignore[arg-type]
        else:
            return self._route_to_leaf(node.right, x)  # type: ignore[arg-type]

    def _count_leaves(self, node: _Node) -> int:
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────
# LinearTreeRegressor
# ─────────────────────────────────────────────────────────────

class LinearTreeRegressor(_LinearTreeCore, BaseEstimator, RegressorMixin):
    """
    Linear Tree Regressor。
    
    各葉ノードに線形回帰モデルを持つ決定木。
    `base_estimator` を変えることで以下が実現できる:
      - RidgeTree  : base_estimator=Ridge()
      - LassoTree  : base_estimator=Lasso()
      - HuberTree  : base_estimator=HuberRegressor()

    Args:
        base_estimator: 葉ノードに使う線形回帰モデル（sklearn互換）
        max_depth: 木の最大深さ
        min_samples_split: 分割に必要な最小サンプル数
        min_samples_leaf: 葉の最小サンプル数
        max_features: 分割候補特徴量の割合/数("sqrt","log2",int,float,None)
        max_bins: 候補しきい値の最大数
        random_state: 乱数シード
    """

    def __init__(
        self,
        base_estimator: Any = None,
        max_depth: int = 5,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_features: Any = None,
        max_bins: int = 64,
        random_state: int | None = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_bins = max_bins
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "LinearTreeRegressor":
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y, dtype=float).ravel()

        # fitのたびにbase_estimatorを直接書き換えないようclone経由で取得
        if self.base_estimator is None:
            base = Ridge()
        else:
            base = self.base_estimator
        # _build_tree内でclone()するのでここではself._base_refとして保持
        self._base_ref = base

        self.n_features_in_ = X_arr.shape[1]
        self.feature_names_in_ = (
            list(X.columns) if isinstance(X, pd.DataFrame) else None
        )
        self.root_ = self._build_tree(X_arr, y_arr, depth=0)
        self.n_leaves_ = self._count_leaves(self.root_)
        logger.info(
            f"LinearTreeRegressor.fit(): 葉数={self.n_leaves_}, "
            f"depth={self.max_depth}, n={X_arr.shape[0]}"
        )
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "root_")
        X_arr = _to_numpy(X)
        preds = np.zeros(X_arr.shape[0])
        for i, x in enumerate(X_arr):
            leaf = self._route_to_leaf(self.root_, x)
            if leaf.linear_model is not None:
                preds[i] = _predict_linear(leaf.linear_model, x.reshape(1, -1))[0]
            else:
                preds[i] = 0.0
        return preds

    def _split_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
    ) -> float:
        """加重平均の線形残差MSEを分割スコアとして使用。"""
        nl = left_mask.sum()
        nr = right_mask.sum()
        n  = nl + nr
        base = getattr(self, "_base_ref", self.base_estimator)
        score_l = _linear_residual_mse(base, X[left_mask],  y[left_mask])
        score_r = _linear_residual_mse(base, X[right_mask], y[right_mask])
        return (nl * score_l + nr * score_r) / n


# ─────────────────────────────────────────────────────────────
# LinearTreeClassifier
# ─────────────────────────────────────────────────────────────

class LinearTreeClassifier(_LinearTreeCore, BaseEstimator, ClassifierMixin):
    """
    Linear Tree Classifier。

    Args:
        base_estimator: 葉ノードに使う線形分類モデル（sklearn互換）
        その他パラメータは LinearTreeRegressor と同様
    """

    def __init__(
        self,
        base_estimator: Any = None,
        max_depth: int = 5,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_features: Any = None,
        max_bins: int = 64,
        random_state: int | None = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_bins = max_bins
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "LinearTreeClassifier":
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y).ravel()

        # fitのたびにbase_estimatorを直接書き換えないようclone経由で取得
        if self.base_estimator is None:
            base = LogisticRegression(max_iter=500, random_state=0)
        else:
            base = self.base_estimator
        self._base_ref = base

        self.classes_ = np.unique(y_arr)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_arr.shape[1]
        self.feature_names_in_ = (
            list(X.columns) if isinstance(X, pd.DataFrame) else None
        )
        # クラスラベルを 0-indexed int に変換
        y_int = np.searchsorted(self.classes_, y_arr)
        self.root_ = self._build_tree(X_arr, y_int, depth=0)
        self.n_leaves_ = self._count_leaves(self.root_)
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "root_")
        X_arr = _to_numpy(X)
        proba = self.predict_proba(X_arr)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "root_")
        X_arr = _to_numpy(X)
        proba = np.zeros((X_arr.shape[0], self.n_classes_))
        for i, x in enumerate(X_arr):
            leaf = self._route_to_leaf(self.root_, x)
            if leaf.linear_model is not None:
                proba[i] = _predict_proba_linear(
                    leaf.linear_model, x.reshape(1, -1), self.n_classes_
                )[0]
            else:
                proba[i] = 1.0 / self.n_classes_
        return proba

    def _split_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
    ) -> float:
        nl = left_mask.sum()
        nr = right_mask.sum()
        n  = nl + nr
        base = getattr(self, "_base_ref", self.base_estimator)
        g_l = _linear_gini(base, X[left_mask],  y[left_mask],  self.n_classes_)
        g_r = _linear_gini(base, X[right_mask], y[right_mask], self.n_classes_)
        return (nl * g_l + nr * g_r) / n


# ─────────────────────────────────────────────────────────────
# LinearForestRegressor / Classifier
# ─────────────────────────────────────────────────────────────

class LinearForestRegressor(BaseEstimator, RegressorMixin):
    """
    Linear Forest Regressor。
    LinearTreeRegressor のバギングアンサンブル。

    Args:
        base_estimator: 葉の線形モデル（RidgeTree等）
        n_estimators: 木の本数
        max_depth: 各木の最大深さ
        max_features: 各木の特徴量サンプリング比率（"sqrt","log2",float,None）
        bootstrap: Trueでブートストラップサンプリング
        max_samples: ブートストラップサンプル比率（0.0〜1.0）
        min_samples_split, min_samples_leaf: 各木の停止条件
        max_bins: 候補しきい値数
        n_jobs: 並列数（現在は逐次、将来拡張用）
        random_state: 乱数シード
    """

    def __init__(
        self,
        base_estimator: Any = None,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_features: Any = "sqrt",
        max_bins: int = 64,
        bootstrap: bool = True,
        max_samples: float = 1.0,
        n_jobs: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.base_estimator    = base_estimator
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self.max_bins          = max_bins
        self.bootstrap         = bootstrap
        self.max_samples       = max_samples
        self.n_jobs            = n_jobs
        self.random_state      = random_state

    def fit(self, X: Any, y: Any) -> "LinearForestRegressor":
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y, dtype=float).ravel()
        n = X_arr.shape[0]

        rng = np.random.default_rng(self.random_state)
        seeds = [int(rng.integers(0, 2**31)) for _ in range(self.n_estimators)]

        def _fit_one(seed: int) -> LinearTreeRegressor:
            tree = LinearTreeRegressor(
                base_estimator=deepcopy(self.base_estimator),
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                max_bins=self.max_bins,
                random_state=seed,
            )
            if self.bootstrap:
                _rng = np.random.default_rng(seed)
                n_sub = max(1, int(n * self.max_samples))
                idx = _rng.choice(n, size=n_sub, replace=True)
                tree.fit(X_arr[idx], y_arr[idx])
            else:
                tree.fit(X_arr, y_arr)
            return tree

        n_jobs = self.n_jobs if self.n_jobs != 0 else 1
        self.estimators_: list[LinearTreeRegressor] = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one)(seed) for seed in seeds
        )
        self.n_features_in_ = X_arr.shape[1]
        logger.info(f"LinearForestRegressor: {self.n_estimators} 本の木を学習 (n_jobs={n_jobs})")
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X_arr = _to_numpy(X)
        n_jobs = self.n_jobs if self.n_jobs != 0 else 1
        preds_list = Parallel(n_jobs=n_jobs)(
            delayed(t.predict)(X_arr) for t in self.estimators_
        )
        return np.stack(preds_list, axis=1).mean(axis=1)


class LinearForestClassifier(BaseEstimator, ClassifierMixin):
    """
    Linear Forest Classifier。
    LinearTreeClassifier のバギングアンサンブル。
    """

    def __init__(
        self,
        base_estimator: Any = None,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_features: Any = "sqrt",
        max_bins: int = 64,
        bootstrap: bool = True,
        max_samples: float = 1.0,
        n_jobs: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.base_estimator    = base_estimator
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self.max_bins          = max_bins
        self.bootstrap         = bootstrap
        self.max_samples       = max_samples
        self.n_jobs            = n_jobs
        self.random_state      = random_state

    def fit(self, X: Any, y: Any) -> "LinearForestClassifier":
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y).ravel()

        self.classes_ = np.unique(y_arr)
        self.n_classes_ = len(self.classes_)
        rng = np.random.default_rng(self.random_state)
        seeds = [int(rng.integers(0, 2**31)) for _ in range(self.n_estimators)]
        n = X_arr.shape[0]

        def _fit_one_cls(seed: int) -> LinearTreeClassifier:
            tree = LinearTreeClassifier(
                base_estimator=deepcopy(self.base_estimator),
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                max_bins=self.max_bins,
                random_state=seed,
            )
            if self.bootstrap:
                _rng = np.random.default_rng(seed)
                n_sub = max(1, int(n * self.max_samples))
                idx = _rng.choice(n, size=n_sub, replace=True)
                tree.fit(X_arr[idx], y_arr[idx])
            else:
                tree.fit(X_arr, y_arr)
            return tree

        n_jobs = self.n_jobs if self.n_jobs != 0 else 1
        self.estimators_: list[LinearTreeClassifier] = Parallel(n_jobs=n_jobs)(
            delayed(_fit_one_cls)(seed) for seed in seeds
        )
        self.n_features_in_ = X_arr.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X_arr = _to_numpy(X)
        n_jobs = self.n_jobs if self.n_jobs != 0 else 1
        probas_list = Parallel(n_jobs=n_jobs)(
            delayed(t.predict_proba)(X_arr) for t in self.estimators_
        )
        return np.stack(probas_list, axis=0).mean(axis=0)


# ─────────────────────────────────────────────────────────────
# LinearBoostRegressor / Classifier
# ─────────────────────────────────────────────────────────────

class LinearBoostRegressor(BaseEstimator, RegressorMixin):
    """
    Linear Boost Regressor。
    残差にステージごとに LinearTree をフィットするブースティング。

    Algorithm:
      F_0(x) = mean(y)
      for m in 1..n_estimators:
        r_m = y - F_{m-1}(x)   (疑似残差)
        h_m = LinearTree(X, r_m)
        F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)

    Args:
        base_estimator: 各ステージの葉の線形モデル
        n_estimators: ブースティングのラウンド数
        learning_rate: 縮小係数（0.0〜1.0）
        max_depth: 各ステージの木の最大深さ
        subsample: 各ステージで使うサンプル比率（0.0〜1.0）
        min_samples_split, min_samples_leaf: 各木の停止条件
        max_bins: 候補しきい値数
        random_state: 乱数シード
    """

    def __init__(
        self,
        base_estimator: Any = None,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_bins: int = 64,
        subsample: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.base_estimator    = base_estimator
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_bins          = max_bins
        self.subsample         = subsample
        self.random_state      = random_state

    def fit(self, X: Any, y: Any) -> "LinearBoostRegressor":
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y, dtype=float).ravel()

        rng = np.random.default_rng(self.random_state)
        n = X_arr.shape[0]
        self.init_value_ = float(np.mean(y_arr))
        F = np.full(n, self.init_value_)
        self.estimators_: list[LinearTreeRegressor] = []

        for m in range(self.n_estimators):
            r = y_arr - F
            # サブサンプリング
            if self.subsample < 1.0:
                n_sub = max(1, int(n * self.subsample))
                idx = rng.choice(n, size=n_sub, replace=False)
            else:
                idx = np.arange(n)

            seed = int(rng.integers(0, 2**31))
            tree = LinearTreeRegressor(
                base_estimator=deepcopy(self.base_estimator),
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_bins=self.max_bins,
                random_state=seed,
            )
            tree.fit(X_arr[idx], r[idx])
            update = tree.predict(X_arr) * self.learning_rate
            F += update
            self.estimators_.append(tree)

        self.n_features_in_ = X_arr.shape[1]
        logger.info(f"LinearBoostRegressor: {self.n_estimators} ラウンド学習完了")
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X_arr = _to_numpy(X)
        F = np.full(X_arr.shape[0], self.init_value_)
        for tree in self.estimators_:
            F += tree.predict(X_arr) * self.learning_rate
        return F


class LinearBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    Linear Boost Classifier。
    二値分類ではLogLoss、多クラスでは one-vs-rest ブースティング。

    Algorithm (二値分類 binary cross-entropy):
      F_0(x) = log(p/(1-p)), p = mean(y)
      for m in 1..n_estimators:
        p_m = sigmoid(F_{m-1}(x))
        r_m = y - p_m  (疑似残差)
        h_m = LinearTree(X, r_m)
        F_m = F_{m-1} + lr * h_m
      predict_proba: sigmoid(F)
    """

    def __init__(
        self,
        base_estimator: Any = None,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_bins: int = 64,
        subsample: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.base_estimator    = base_estimator
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_bins          = max_bins
        self.subsample         = subsample
        self.random_state      = random_state

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X: Any, y: Any) -> "LinearBoostClassifier":
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y).ravel()

        self.classes_ = np.unique(y_arr)
        self.n_classes_ = len(self.classes_)
        y_int = np.searchsorted(self.classes_, y_arr)

        rng = np.random.default_rng(self.random_state)
        n = X_arr.shape[0]
        K = self.n_classes_

        if K == 2:
            # 二値分類
            p_mean = y_int.mean()
            p_mean = np.clip(p_mean, 1e-6, 1 - 1e-6)
            self.F0_ = np.log(p_mean / (1 - p_mean))
            F = np.full(n, self.F0_)
            self.estimators_: list[list[LinearTreeRegressor]] = []

            for m in range(self.n_estimators):
                p = self._sigmoid(F)
                r = y_int - p
                if self.subsample < 1.0:
                    n_sub = max(1, int(n * self.subsample))
                    idx = rng.choice(n, size=n_sub, replace=False)
                else:
                    idx = np.arange(n)
                seed = int(rng.integers(0, 2**31))
                tree = LinearTreeRegressor(
                    base_estimator=deepcopy(self.base_estimator),
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_bins=self.max_bins,
                    random_state=seed,
                )
                tree.fit(X_arr[idx], r[idx])
                F += tree.predict(X_arr) * self.learning_rate
                self.estimators_.append([tree])
        else:
            # 多クラス: One-vs-Rest
            self.F0_ = np.zeros(K)
            F = np.zeros((n, K))
            for k in range(K):
                p_k = (y_int == k).mean()
                p_k = np.clip(p_k, 1e-6, 1 - 1e-6)
                self.F0_[k] = np.log(p_k / (1 - p_k))
                F[:, k] = self.F0_[k]

            self.estimators_ = []
            for m in range(self.n_estimators):
                proba = self._softmax(F)
                stage_trees = []
                for k in range(K):
                    r_k = (y_int == k).astype(float) - proba[:, k]
                    if self.subsample < 1.0:
                        n_sub = max(1, int(n * self.subsample))
                        idx = rng.choice(n, size=n_sub, replace=False)
                    else:
                        idx = np.arange(n)
                    seed = int(rng.integers(0, 2**31))
                    tree = LinearTreeRegressor(
                        base_estimator=deepcopy(self.base_estimator),
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        max_bins=self.max_bins,
                        random_state=seed,
                    )
                    tree.fit(X_arr[idx], r_k[idx])
                    F[:, k] += tree.predict(X_arr) * self.learning_rate
                    stage_trees.append(tree)
                self.estimators_.append(stage_trees)

        self.n_features_in_ = X_arr.shape[1]
        self.binary_ = (K == 2)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        X_arr = _to_numpy(X)
        n = X_arr.shape[0]

        if self.binary_:
            F = np.full(n, self.F0_)
            for stage in self.estimators_:
                F += stage[0].predict(X_arr) * self.learning_rate
            p1 = self._sigmoid(F)
            return np.stack([1 - p1, p1], axis=1)
        else:
            F = np.tile(self.F0_, (n, 1)).astype(float)
            for stage_trees in self.estimators_:
                for k, tree in enumerate(stage_trees):
                    F[:, k] += tree.predict(X_arr) * self.learning_rate
            return self._softmax(F)

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ─────────────────────────────────────────────────────────────
# ファクトリー関数（base_estimator のプリセット）
# ─────────────────────────────────────────────────────────────

def RidgeTreeRegressor(**kw: Any) -> LinearTreeRegressor:
    """Ridge を葉に使う LinearTreeRegressor のショートカット。"""
    alpha = kw.pop("alpha", 1.0)
    return LinearTreeRegressor(base_estimator=Ridge(alpha=alpha), **kw)


def RidgeTreeClassifier(**kw: Any) -> LinearTreeClassifier:
    """LogisticRegression(C=1) を葉に使う LinearTreeClassifier のショートカット。"""
    C = kw.pop("C", 1.0)
    return LinearTreeClassifier(
        base_estimator=LogisticRegression(C=C, max_iter=500, random_state=0), **kw
    )
