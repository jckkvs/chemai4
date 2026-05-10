"""
backend/models/monotonic_kernel.py

カーネル系モデル（SVR / KernelRidge / GaussianProcessRegressor / SVC等）に
ソフト単調性制約を付与するラッパー。

設計方針
────────
ネイティブに単調性制約を持たないカーネル系モデルに対して、
学習データの範囲（± sigma_factor * σ）でグリッドサンプリングし、
単調性違反箇所に仮想サンプルを追加して再フィッティングすることで
ソフト（近似）単調性制約を実現する。

アルゴリズム概要:
  1. 通常の fit(X, y) を実行
  2. 各制約特徴量 i について:
       grid_k = linspace(μ_i - sigma * σ_i, μ_i + sigma * σ_i, n_grid)
       （他特徴量は中央値で固定）
  3. グリッド点で予測し、単調性違反 Σmax(0, violation)^2 を計算
  4. 違反が閾値を超えた場合:
       違反箇所を仮想サンプル（penalty_weight 倍の重みを付与）として
       X_aug, y_aug, sample_weight に追加して再フィッティング
  5. 最大 max_iter 回繰り返す

XGBoost 等のネイティブ対応モデルは本ラッパーを使わず、
pipeline_builder.apply_monotonic_constraints() で直接処理する。
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────────

def _to_numpy(X: Any) -> np.ndarray:
    """DataFrame / array-like を numpy に変換。"""
    if isinstance(X, pd.DataFrame):
        return X.values.astype(float)
    return np.asarray(X, dtype=float)


def _compute_monotonic_violation(
    y_grid: np.ndarray,
    direction: int,
) -> float:
    """
    グリッド予測値の単調性違反量を計算する。

    Args:
        y_grid: グリッド点での予測値 (n_grid,)
        direction: +1=単調増加, -1=単調減少

    Returns:
        違反量の二乗和（0以上。0なら完全単調）
    """
    diff = np.diff(y_grid)          # shape (n_grid-1,)
    # 増加制約: diff > 0 が望ましい → 違反 = max(0, -diff)
    # 減少制約: diff < 0 が望ましい → 違反 = max(0, +diff)
    violation = np.maximum(0.0, -direction * diff)
    return float(np.sum(violation ** 2))


def _build_monotonic_augmented_data(
    X_orig: np.ndarray,
    y_orig: np.ndarray,
    estimator: Any,
    monotonic_constraints: tuple[int, ...],
    n_grid: int,
    sigma_factor: float,
    penalty_weight: float,
    feature_stats: dict[int, tuple[float, float]],  # idx -> (mean, std)
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    単調性違反箇所に仮想サンプルを追加して拡張データを生成する。

    Returns:
        (X_aug, y_aug, sample_weight_aug)
    """
    n_samples, n_features = X_orig.shape
    median_vals = np.median(X_orig, axis=0)

    X_penalty_list = []
    y_penalty_list = []

    for feat_idx, direction in enumerate(monotonic_constraints):
        if direction == 0:
            continue

        mu, sigma = feature_stats[feat_idx]
        if sigma < 1e-10:
            continue

        lo = mu - sigma_factor * sigma
        hi = mu + sigma_factor * sigma
        grid_vals = np.linspace(lo, hi, n_grid)

        # グリッド点での X（他変数は中央値固定）
        X_grid = np.tile(median_vals, (n_grid, 1))
        X_grid[:, feat_idx] = grid_vals

        y_grid = estimator.predict(X_grid)
        if y_grid.ndim > 1:
            y_grid = y_grid.ravel()

        # 違反箇所を特定
        diff = np.diff(y_grid)
        violation_mask = (-direction * diff) > 0   # True where violated

        for k in range(len(diff)):
            if not violation_mask[k]:
                continue
            # 違反した連続2点をペナルティサンプルとして追加
            # 正しい順序になるよう y を修正
            x0 = X_grid[k]
            x1 = X_grid[k + 1]
            y0_correct = y_grid[k]
            # 方向に応じた "あるべき" 予測値
            eps = abs(diff[k]) + 0.01
            if direction == 1:   # 増加: y[k+1] >= y[k]
                y1_correct = y0_correct + eps
            else:                # 減少: y[k+1] <= y[k]
                y1_correct = y0_correct - eps

            X_penalty_list.extend([x0, x1])
            y_penalty_list.extend([y0_correct, y1_correct])

    if not X_penalty_list:
        return X_orig, y_orig, None

    X_aug = np.vstack([X_orig, np.array(X_penalty_list)])
    y_aug = np.concatenate([y_orig, np.array(y_penalty_list)])

    # 元のサンプルは重み1、ペナルティサンプルは penalty_weight
    sw_orig = np.ones(len(X_orig))
    sw_pen  = np.full(len(X_penalty_list), penalty_weight)
    sw_aug  = np.concatenate([sw_orig, sw_pen])

    return X_aug, y_aug, sw_aug


# ─────────────────────────────────────────────────────────────
# 回帰ラッパー
# ─────────────────────────────────────────────────────────────

class MonotonicKernelWrapper(BaseEstimator, RegressorMixin):
    """
    カーネル系回帰モデル（SVR / KernelRidge / GPR等）に
    ソフト単調性制約を付与するラッパー。

    sklearn の clone() / GridSearchCV に完全対応。

    Args:
        base_estimator: ラップするsklearnモデル
        monotonic_constraints: 特徴量ごとの制約 (0, 1, -1) のタプル
            0=制約なし, +1=単調増加, -1=単調減少
        n_grid: 単調性チェック用グリッド点数
        sigma_factor: 学習データの平均±sigma_factor*σの範囲でチェック
        penalty_weight: 仮想ペナルティサンプルの重み
        max_iter: 反復フィッティングの最大回数
        violation_threshold: この値以下なら制約満足とみなして終了
    """

    def __init__(
        self,
        base_estimator: Any = None,
        monotonic_constraints: tuple[int, ...] = (),
        n_grid: int = 20,
        sigma_factor: float = 1.5,
        penalty_weight: float = 10.0,
        max_iter: int = 3,
        violation_threshold: float = 1e-4,
    ) -> None:
        self.base_estimator = base_estimator
        self.monotonic_constraints = monotonic_constraints
        self.n_grid = n_grid
        self.sigma_factor = sigma_factor
        self.penalty_weight = penalty_weight
        self.max_iter = max_iter
        self.violation_threshold = violation_threshold

    def fit(self, X: Any, y: Any, sample_weight: np.ndarray | None = None) -> "MonotonicKernelWrapper":
        """
        ソフト単調性制約付きフィッティング。

        1. base_estimator を cloneして fit
        2. 単調性違反を検出し、ペナルティサンプルを追加して再fit
        3. max_iter 回まで繰り返す
        """
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y, dtype=float).ravel()

        if self.base_estimator is None:
            from sklearn.svm import SVR
            base = SVR()
        else:
            base = self.base_estimator

        constraints = self.monotonic_constraints
        has_constraint = any(c != 0 for c in constraints) if constraints else False

        # 特徴量の統計量を記録
        n_features = X_arr.shape[1]
        feature_stats: dict[int, tuple[float, float]] = {}
        for i in range(n_features):
            mu  = float(np.mean(X_arr[:, i]))
            std = float(np.std(X_arr[:, i]))
            feature_stats[i] = (mu, std)

        # パッド: constraints が短い場合は 0 で埋める
        mc = list(constraints) + [0] * max(0, n_features - len(constraints))
        mc = mc[:n_features]

        # 初回フィット
        fitted = clone(base)
        _fit_with_weight(fitted, X_arr, y_arr, sample_weight)

        if not has_constraint:
            self.estimator_ = fitted
            self.feature_names_in_ = (
                list(X.columns) if isinstance(X, pd.DataFrame) else None
            )
            self.n_features_in_ = n_features
            logger.debug("MonotonicKernelWrapper: 制約なし → 通常 fit のみ")
            return self

        # 反復ペナルティフィッティング
        X_cur, y_cur, sw_cur = X_arr, y_arr, sample_weight
        total_violation = float("inf")

        for iteration in range(self.max_iter):
            total_violation = sum(
                _compute_monotonic_violation(
                    fitted.predict(
                        _build_grid_X(X_arr, i, feature_stats, self.sigma_factor, self.n_grid)
                    ),
                    direction=mc[i],
                )
                for i, direction in enumerate(mc)
                if direction != 0
            )
            logger.info(f"Iteration {iteration}: violation={total_violation:.6f}")

            if total_violation <= self.violation_threshold:
                logger.info(f"単調性制約達成 (iter={iteration})")
                break

            X_cur, y_cur, sw_cur = _build_monotonic_augmented_data(
                X_arr, y_arr,
                fitted,
                tuple(mc),
                self.n_grid,
                self.sigma_factor,
                self.penalty_weight,
                feature_stats,
            )

            # 重みを受け取れない推定器はサンプルウェイトなしで試みる
            refitted = clone(base)
            _fit_with_weight(refitted, X_cur, y_cur, sw_cur)
            fitted = refitted

        # 最終違反を記録・警告
        if total_violation > self.violation_threshold:
            warnings.warn(
                f"MonotonicKernelWrapper: {self.max_iter}回反復後も単調性制約を"
                f"完全には満足できませんでした (violation={total_violation:.4f})。"
                f"ソフト制約として近似的に適用されています。",
                UserWarning,
                stacklevel=2,
            )

        self.estimator_ = fitted
        self.feature_names_in_ = (
            list(X.columns) if isinstance(X, pd.DataFrame) else None
        )
        self.n_features_in_ = n_features
        self.monotonic_violation_ = total_violation
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(_to_numpy(X))

    def score(self, X: Any, y: Any) -> float:
        return self.estimator_.score(_to_numpy(X), np.asarray(y))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = {
            "base_estimator": self.base_estimator,
            "monotonic_constraints": self.monotonic_constraints,
            "n_grid": self.n_grid,
            "sigma_factor": self.sigma_factor,
            "penalty_weight": self.penalty_weight,
            "max_iter": self.max_iter,
            "violation_threshold": self.violation_threshold,
        }
        if deep and self.base_estimator is not None:
            for k, v in self.base_estimator.get_params(deep=True).items():
                params[f"base_estimator__{k}"] = v
        return params

    def set_params(self, **params: Any) -> "MonotonicKernelWrapper":
        base_params = {}
        own_params  = {}
        for k, v in params.items():
            if k.startswith("base_estimator__"):
                base_params[k[len("base_estimator__"):]] = v
            else:
                own_params[k] = v
        for k, v in own_params.items():
            setattr(self, k, v)
        if base_params and self.base_estimator is not None:
            self.base_estimator.set_params(**base_params)
        return self


# ─────────────────────────────────────────────────────────────
# 分類ラッパー（SVC 向け）
# ─────────────────────────────────────────────────────────────

class MonotonicKernelClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    カーネル系分類モデル（SVC 等）に
    ソフト単調性制約を付与するラッパー。

    分類では predict_proba の確率値に対して単調性チェックを行う。

    Args:
        base_estimator: ラップするsklearn分類モデル（probability=True推奨）
        monotonic_constraints: 特徴量ごとの制約 (0, 1, -1) のタプル
        その他パラメータは MonotonicKernelWrapper と同様
    """

    def __init__(
        self,
        base_estimator: Any = None,
        monotonic_constraints: tuple[int, ...] = (),
        n_grid: int = 20,
        sigma_factor: float = 1.5,
        penalty_weight: float = 10.0,
        max_iter: int = 3,
        violation_threshold: float = 1e-4,
    ) -> None:
        self.base_estimator = base_estimator
        self.monotonic_constraints = monotonic_constraints
        self.n_grid = n_grid
        self.sigma_factor = sigma_factor
        self.penalty_weight = penalty_weight
        self.max_iter = max_iter
        self.violation_threshold = violation_threshold

    def fit(self, X: Any, y: Any, sample_weight: np.ndarray | None = None) -> "MonotonicKernelClassifierWrapper":
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y)

        if self.base_estimator is None:
            from sklearn.svm import SVC
            base = SVC(probability=True)
        else:
            base = self.base_estimator

        n_features = X_arr.shape[1]
        feature_stats: dict[int, tuple[float, float]] = {}
        for i in range(n_features):
            feature_stats[i] = (float(np.mean(X_arr[:, i])), float(np.std(X_arr[:, i])))

        mc = list(self.monotonic_constraints) + [0] * max(0, n_features - len(self.monotonic_constraints))
        mc = mc[:n_features]
        has_constraint = any(c != 0 for c in mc)

        fitted = clone(base)
        _fit_with_weight(fitted, X_arr, y_arr, sample_weight)

        if not has_constraint:
            self.estimator_ = fitted
            self.classes_ = fitted.classes_
            self.n_features_in_ = n_features
            return self

        # 分類では predict_proba（クラス1の確率）に対してペナルティ
        for iteration in range(self.max_iter):
            violations = []
            for i, direction in enumerate(mc):
                if direction == 0:
                    continue
                X_grid = _build_grid_X(X_arr, i, feature_stats, self.sigma_factor, self.n_grid)
                try:
                    proba_grid = fitted.predict_proba(X_grid)[:, 1]
                except Exception:
                    proba_grid = fitted.predict(X_grid).astype(float)
                violations.append(_compute_monotonic_violation(proba_grid, direction))

            total_violation = sum(violations) if violations else 0.0
            if total_violation <= self.violation_threshold:
                break

            # ペナルティサンプル追加（ラベルを正しい方向に引っ張る）
            X_pen_list, y_pen_list = [], []
            for i, direction in enumerate(mc):
                if direction == 0:
                    continue
                mu, sigma = feature_stats[i]
                grid_vals = np.linspace(
                    mu - self.sigma_factor * sigma,
                    mu + self.sigma_factor * sigma,
                    self.n_grid
                )
                median_vals = np.median(X_arr, axis=0)
                X_grid = np.tile(median_vals, (self.n_grid, 1))
                X_grid[:, i] = grid_vals
                try:
                    proba_grid = fitted.predict_proba(X_grid)[:, 1]
                except Exception:
                    continue
                diff = np.diff(proba_grid)
                for k in range(len(diff)):
                    if (-direction * diff[k]) > 0:
                        # 違反: 正しいラベルを付けてサポート
                        if direction == 1:
                            # 増加すべき → 後の点をクラス1
                            X_pen_list.append(X_grid[k + 1])
                            y_pen_list.append(np.max(y_arr))   # positiveクラス
                        else:
                            # 減少すべき → 前の点をクラス1
                            X_pen_list.append(X_grid[k])
                            y_pen_list.append(np.max(y_arr))

            if not X_pen_list:
                break

            X_aug = np.vstack([X_arr, np.array(X_pen_list)])
            y_aug = np.concatenate([y_arr, np.array(y_pen_list)])
            sw_aug = np.concatenate([
                np.ones(len(X_arr)),
                np.full(len(X_pen_list), self.penalty_weight)
            ])
            refitted = clone(base)
            _fit_with_weight(refitted, X_aug, y_aug, sw_aug)
            fitted = refitted

        self.estimator_ = fitted
        self.classes_ = fitted.classes_
        self.n_features_in_ = n_features
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(_to_numpy(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(_to_numpy(X))

    def score(self, X: Any, y: Any) -> float:
        return self.estimator_.score(_to_numpy(X), np.asarray(y))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = {
            "base_estimator": self.base_estimator,
            "monotonic_constraints": self.monotonic_constraints,
            "n_grid": self.n_grid,
            "sigma_factor": self.sigma_factor,
            "penalty_weight": self.penalty_weight,
            "max_iter": self.max_iter,
            "violation_threshold": self.violation_threshold,
        }
        if deep and self.base_estimator is not None:
            for k, v in self.base_estimator.get_params(deep=True).items():
                params[f"base_estimator__{k}"] = v
        return params

    def set_params(self, **params: Any) -> "MonotonicKernelClassifierWrapper":
        base_params = {}
        own_params  = {}
        for k, v in params.items():
            if k.startswith("base_estimator__"):
                base_params[k[len("base_estimator__"):]] = v
            else:
                own_params[k] = v
        for k, v in own_params.items():
            setattr(self, k, v)
        if base_params and self.base_estimator is not None:
            self.base_estimator.set_params(**base_params)
        return self


# ─────────────────────────────────────────────────────────────
# 内部ヘルパー
# ─────────────────────────────────────────────────────────────

def _fit_with_weight(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
) -> None:
    """sample_weight を受け付けるモデルとそうでないモデルを統一的にフィット。"""
    if sample_weight is None:
        estimator.fit(X, y)
        return
    try:
        estimator.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        # sample_weight 非対応（GaussianProcessRegressor 等）
        logger.debug(
            f"{type(estimator).__name__} は sample_weight 非対応。"
            "重みなしで fit します。"
        )
        estimator.fit(X, y)


def _build_grid_X(
    X_ref: np.ndarray,
    feat_idx: int,
    feature_stats: dict[int, tuple[float, float]],
    sigma_factor: float,
    n_grid: int,
) -> np.ndarray:
    """指定特徴量をグリッド化し、他特徴量を中央値で固定したXを生成。"""
    median_vals = np.median(X_ref, axis=0)
    mu, sigma = feature_stats[feat_idx]
    if sigma < 1e-10:
        sigma = 1.0
    lo = mu - sigma_factor * sigma
    hi = mu + sigma_factor * sigma
    grid_vals = np.linspace(lo, hi, n_grid)
    X_grid = np.tile(median_vals, (n_grid, 1))
    X_grid[:, feat_idx] = grid_vals
    return X_grid


# ─────────────────────────────────────────────────────────────
# ラッパー適用ファクトリー
# ─────────────────────────────────────────────────────────────

#: ソフト単調性制約ラッパーを適用すべきクラス名（小文字部分マッチ）
_SOFT_MONOTONIC_KEYWORDS = [
    "svr", "svc", "kernelridge", "gaussianprocess", "krr",
    "nusvr", "linearsvr", "nusvc",
]


def is_soft_monotonic_candidate(estimator: Any) -> bool:
    """
    estimator がソフト単調性制約ラッパーの適用対象かを判定する。

    Args:
        estimator: sklearn 互換の推定器

    Returns:
        True ならラッパーを適用すべき
    """
    name = type(estimator).__name__.lower()
    return any(kw in name for kw in _SOFT_MONOTONIC_KEYWORDS)


def wrap_with_soft_monotonic(
    estimator: Any,
    monotonic_constraints: tuple[int, ...],
    *,
    n_grid: int = 20,
    sigma_factor: float = 1.5,
    penalty_weight: float = 10.0,
    max_iter: int = 3,
) -> Any:
    """
    estimator にソフト単調性制約ラッパーを適用する。

    回帰器 → MonotonicKernelWrapper
    分類器 → MonotonicKernelClassifierWrapper

    制約が全て 0 の場合は元の estimator をそのまま返す。
    ネイティブ対応モデル（XGBoost 等）は先に pipeline_builder で処理すべき。

    Args:
        estimator: ラップするモデル
        monotonic_constraints: (0, 1, -1, ...) の制約タプル
        n_grid: グリッド点数
        sigma_factor: 範囲係数
        penalty_weight: ペナルティサンプルの重み
        max_iter: 反復フィッティングの最大回数

    Returns:
        ラップ済みの estimator（制約なしの場合は元のまま）
    """
    if not any(c != 0 for c in monotonic_constraints):
        return estimator

    from sklearn.base import is_classifier as _is_clf
    if _is_clf(estimator):
        logger.info(
            f"MonotonicKernelClassifierWrapper でラップ: {type(estimator).__name__}, "
            f"制約={monotonic_constraints}"
        )
        return MonotonicKernelClassifierWrapper(
            base_estimator=estimator,
            monotonic_constraints=monotonic_constraints,
            n_grid=n_grid,
            sigma_factor=sigma_factor,
            penalty_weight=penalty_weight,
            max_iter=max_iter,
        )
    else:
        logger.info(
            f"MonotonicKernelWrapper でラップ: {type(estimator).__name__}, "
            f"制約={monotonic_constraints}"
        )
        return MonotonicKernelWrapper(
            base_estimator=estimator,
            monotonic_constraints=monotonic_constraints,
            n_grid=n_grid,
            sigma_factor=sigma_factor,
            penalty_weight=penalty_weight,
            max_iter=max_iter,
        )
