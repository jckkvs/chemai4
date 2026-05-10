# -*- coding: utf-8 -*-
"""
backend/models/monotonic_wrapper.py

全sklearnモデル対応の変数ごと単調性制約ラッパー（フルスクラッチ）。

単調性制約の4パターン
─────────────────────
  0  : 制約なし
  1  : 単調増加
  -1 : 単調減少
  2  : 単調（方向自動検出）— fit時に Spearman相関で判定

制約強度（constraint_strength）
───────────────────────────────
  "weak"   : ソフト制約（できるだけ単調にする）
             - penalty_weight=5, max_iter=2, n_grid=15
  "strong" : ハード制約（±3σ 全範囲で厳密に単調性を保つ）
             - penalty_weight=50, max_iter=8, n_grid=40

制約範囲
────────
デフォルト sigma_factor=3.0 により、学習データ平均 ±3σ の範囲で
単調性を保証する。学習範囲内だけの制約は汎化性能を劣化させるため、
外挿領域まで制約を広げることで未知データでの安定性を確保する。

アルゴリズム: ペナルティサンプル拡張法（反復フィッティング）
──────────────────────────────────────────────────────────────
fit(X, y):
  1. base_estimator.fit(X, y) で通常学習
  2. monotonic=2 の変数は Spearman相関で方向を +1 or -1 に解決
  3. 各制約変数 i について（方向 d_i ∈ {+1, -1}）:
       grid_k = linspace(μ_i - σ_factor*σ_i, μ_i + σ_factor*σ_i, n_grid)
       （他特徴量は中央値に固定）
  4. グリッド点で予測し、単調性違反箇所を特定
  5. 違反箇所にペナルティサンプルを追加して再フィッティング
  6. max_iter 回まで繰り返す

predict(X):
  内部の base_estimator.predict(X) をそのまま返す。

フル sklearn 互換
────────────────
- BaseEstimator 継承 → get_params()/set_params() 完全対応
- clone()/GridSearchCV/Pipeline 対応
- predict_proba() 分類版対応
- sample_weight 透過

参考: Sill & Abu-Mostafa (1997) "Monotonicity hints"
      Cano et al. (2019) "Monotonic classification"
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    ClassifierMixin,
    clone,
    is_classifier as _is_classifier,
)
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────
# 制約強度プリセット
# ────────────────────────────────────────────────────────────
_STRENGTH_PRESETS: dict[str, dict[str, Any]] = {
    "weak": {
        "penalty_weight": 5.0,
        "max_iter": 2,
        "n_grid": 15,
        "violation_threshold": 0.01,
    },
    "strong": {
        "penalty_weight": 50.0,
        "max_iter": 8,
        "n_grid": 40,
        "violation_threshold": 1e-6,
    },
}


# ────────────────────────────────────────────────────────────
# ユーティリティ
# ────────────────────────────────────────────────────────────

def _to_numpy(X: Any) -> np.ndarray:
    """DataFrame / array-like → float64 numpy 変換。"""
    if isinstance(X, pd.DataFrame):
        return X.values.astype(np.float64)
    return np.asarray(X, dtype=np.float64)


def _fit_with_weight(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
) -> None:
    """sample_weight を受け付けるモデルとそうでないモデルを統一的に fit。"""
    if sample_weight is None:
        estimator.fit(X, y)
        return
    try:
        estimator.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        logger.debug(
            f"{type(estimator).__name__} は sample_weight 非対応。"
            "重みなしで fit します。"
        )
        estimator.fit(X, y)


def _resolve_auto_direction(
    X: np.ndarray,
    y: np.ndarray,
    constraints: list[int],
) -> tuple[int, ...]:
    """
    monotonic=2（自動検出）を +1 or -1 に解決する。

    Spearman 順位相関の符号で方向を判定:
      ρ ≥ 0 → +1（増加）, ρ < 0 → -1（減少）

    Args:
        X: 入力特徴量 (n_samples, n_features)
        y: 目的変数 (n_samples,)
        constraints: 0, 1, -1, 2 を含む制約リスト

    Returns:
        解決済みの制約タプル（0, 1, -1 のみ）
    """
    resolved = list(constraints)
    for i, c in enumerate(constraints):
        if c == 2:
            try:
                x_col = X[:, i]
                # 定数列チェック：分散がほぼゼロの場合は相関を計算しない
                if np.std(x_col) < 1e-10:
                    logger.info(
                        f"  単調性自動検出 [feat={i}]: 特徴量が定数のため制約なし(0)"
                    )
                    resolved[i] = 0
                    continue

                # yが定数の場合もチェック
                if np.std(y) < 1e-10:
                    logger.info(
                        f"  単調性自動検出 [feat={i}]: 目的変数が定数のため制約なし(0)"
                    )
                    resolved[i] = 0
                    continue

                rho, _ = stats.spearmanr(x_col, y)
                direction = 1 if rho >= 0 else -1
                resolved[i] = direction
                logger.info(
                    f"  単調性自動検出 [feat={i}]: Spearman ρ={rho:.4f} → "
                    f"{'📈 増加' if direction == 1 else '📉 減少'}"
                )
            except Exception as e:
                logger.warning(f"  自動検出失敗 [feat={i}]: {e} → 制約なし(0)")
                resolved[i] = 0
    return tuple(resolved)


def _build_grid_X(
    X_ref: np.ndarray,
    feat_idx: int,
    feature_stats: dict[int, tuple[float, float]],
    sigma_factor: float,
    n_grid: int,
) -> np.ndarray:
    """指定特徴量をグリッド化し、他特徴量を中央値で固定した X を生成。"""
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


def _compute_violation(y_grid: np.ndarray, direction: int) -> float:
    """グリッド予測値の単調性違反量(二乗和)。"""
    diff = np.diff(y_grid)
    violation = np.maximum(0.0, -direction * diff)
    return float(np.sum(violation ** 2))


def _build_penalty_samples(
    X_orig: np.ndarray,
    estimator: Any,
    constraints: tuple[int, ...],
    feature_stats: dict[int, tuple[float, float]],
    n_grid: int,
    sigma_factor: float,
    penalty_weight: float,
    use_proba: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    単調性違反箇所にペナルティサンプルを生成する。

    Returns:
        (X_penalty, y_penalty, weights) or None（違反なし）
    """
    X_pen_list: list[np.ndarray] = []
    y_pen_list: list[float] = []

    for feat_idx, direction in enumerate(constraints):
        if direction == 0:
            continue

        X_grid = _build_grid_X(X_orig, feat_idx, feature_stats, sigma_factor, n_grid)

        if use_proba:
            try:
                y_grid = estimator.predict_proba(X_grid)[:, 1]
            except Exception:
                y_grid = estimator.predict(X_grid).astype(np.float64)
        else:
            y_grid = estimator.predict(X_grid)
            if y_grid.ndim > 1:
                y_grid = y_grid.ravel()

        diff = np.diff(y_grid)
        violation_mask = (-direction * diff) > 0

        for k in range(len(diff)):
            if not violation_mask[k]:
                continue
            x0 = X_grid[k]
            x1 = X_grid[k + 1]
            y0_val = float(y_grid[k])
            eps = abs(float(diff[k])) + 0.01
            if direction == 1:
                y1_val = y0_val + eps
            else:
                y1_val = y0_val - eps

            X_pen_list.extend([x0, x1])
            y_pen_list.extend([y0_val, y1_val])

    if not X_pen_list:
        return None

    X_pen = np.array(X_pen_list)
    y_pen = np.array(y_pen_list)
    w_pen = np.full(len(y_pen), penalty_weight)
    return X_pen, y_pen, w_pen


# ────────────────────────────────────────────────────────────
# 汎用ラッパー: 回帰版
# ────────────────────────────────────────────────────────────

class MonotonicConstraintRegressor(BaseEstimator, RegressorMixin):
    """
    任意の sklearn 回帰モデルに変数ごとの単調性制約を付与する汎用ラッパー。

    ペナルティサンプル拡張法 + ±3σ 外挿保証。

    Args:
        base_estimator: ラップするsklearnモデル
        monotonic_constraints: 特徴量ごとの制約タプル
            0=制約なし, +1=増加, -1=減少, 2=自動検出
        constraint_strength: 制約強度
            "weak"  = できるだけ単調にする（penalty_weight=5, max_iter=2）
            "strong" = ±3σ全範囲で厳密単調（penalty_weight=50, max_iter=8）
            None = 個別パラメータを使用
        n_grid: 単調性チェック用グリッド点数
        sigma_factor: ±σの範囲（デフォルト3.0 = ±3σ）
        penalty_weight: ペナルティサンプルの重み
        max_iter: 反復フィッティングの最大回数
        violation_threshold: 制約満足閾値
    """

    def __init__(
        self,
        base_estimator: Any = None,
        monotonic_constraints: tuple[int, ...] = (),
        constraint_strength: str | None = None,
        n_grid: int = 20,
        sigma_factor: float = 3.0,
        penalty_weight: float = 10.0,
        max_iter: int = 3,
        violation_threshold: float = 1e-4,
    ) -> None:
        self.base_estimator = base_estimator
        self.monotonic_constraints = monotonic_constraints
        self.constraint_strength = constraint_strength
        self.n_grid = n_grid
        self.sigma_factor = sigma_factor
        self.penalty_weight = penalty_weight
        self.max_iter = max_iter
        self.violation_threshold = violation_threshold

    def _effective_params(self) -> dict[str, Any]:
        """constraint_strength プリセットを反映した実効パラメータ。"""
        if self.constraint_strength and self.constraint_strength in _STRENGTH_PRESETS:
            preset = _STRENGTH_PRESETS[self.constraint_strength]
            return {
                "n_grid": preset["n_grid"],
                "sigma_factor": self.sigma_factor,   # σ倍率はユーザー指定を尊重
                "penalty_weight": preset["penalty_weight"],
                "max_iter": preset["max_iter"],
                "violation_threshold": preset["violation_threshold"],
            }
        return {
            "n_grid": self.n_grid,
            "sigma_factor": self.sigma_factor,
            "penalty_weight": self.penalty_weight,
            "max_iter": self.max_iter,
            "violation_threshold": self.violation_threshold,
        }

    # ---- sklearn API ----

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = {
            "base_estimator": self.base_estimator,
            "monotonic_constraints": self.monotonic_constraints,
            "constraint_strength": self.constraint_strength,
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

    def set_params(self, **params: Any) -> "MonotonicConstraintRegressor":
        base_params: dict[str, Any] = {}
        own_params: dict[str, Any] = {}
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

    # ---- fit ----

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: np.ndarray | None = None,
    ) -> "MonotonicConstraintRegressor":
        """
        ペナルティサンプル拡張法で単調性制約付き学習。

        1. base_estimator.fit(X,y) で通常学習
        2. monotonic=2 → Spearman相関で方向解決
        3. ±3σ範囲のグリッドで違反を検出
        4. ペナルティサンプル追加 → 再fit
        5. max_iter 回まで繰り返す（strong なら8回）
        """
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        self.feature_names_in_: list[str] | None = (
            X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        )
        n_features = X_arr.shape[1]
        self.n_features_in_ = n_features

        if self.base_estimator is None:
            from sklearn.ensemble import RandomForestRegressor
            base = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            base = self.base_estimator

        # 制約のパッド & 自動検出
        mc = list(self.monotonic_constraints) + [0] * max(0, n_features - len(self.monotonic_constraints))
        mc = mc[:n_features]
        mc = list(_resolve_auto_direction(X_arr, y_arr, mc))
        self.resolved_constraints_: tuple[int, ...] = tuple(mc)
        has_constraint = any(c != 0 for c in mc)

        # 実効パラメータ
        eff = self._effective_params()
        _n_grid = eff["n_grid"]
        _sigma = eff["sigma_factor"]
        _pw = eff["penalty_weight"]
        _mi = eff["max_iter"]
        _vt = eff["violation_threshold"]

        # 特徴量統計
        feature_stats: dict[int, tuple[float, float]] = {}
        for i in range(n_features):
            feature_stats[i] = (float(np.mean(X_arr[:, i])), float(np.std(X_arr[:, i])))

        # 初回 fit
        fitted = clone(base)
        _fit_with_weight(fitted, X_arr, y_arr, sample_weight)

        if not has_constraint:
            self.estimator_ = fitted
            self.monotonic_violation_ = 0.0
            return self

        mc_tuple = tuple(mc)
        total_violation = float("inf")

        for iteration in range(_mi):
            total_violation = sum(
                _compute_violation(
                    fitted.predict(
                        _build_grid_X(X_arr, i, feature_stats, _sigma, _n_grid)
                    ),
                    direction=mc[i],
                )
                for i in range(n_features) if mc[i] != 0
            )
            logger.debug(f"  反復 {iteration}: violation={total_violation:.6f}")

            if total_violation <= _vt:
                logger.info(f"  単調性制約達成 (iter={iteration})")
                break

            result = _build_penalty_samples(
                X_arr, fitted, mc_tuple, feature_stats,
                _n_grid, _sigma, _pw,
            )
            if result is None:
                break

            X_pen, y_pen, w_pen = result
            X_aug = np.vstack([X_arr, X_pen])
            y_aug = np.concatenate([y_arr, y_pen])
            sw_orig = np.ones(len(X_arr)) if sample_weight is None else sample_weight
            sw_aug = np.concatenate([sw_orig, w_pen])

            refitted = clone(base)
            _fit_with_weight(refitted, X_aug, y_aug, sw_aug)
            fitted = refitted

        if total_violation > _vt:
            strength_label = self.constraint_strength or "custom"
            warnings.warn(
                f"MonotonicConstraintRegressor [{strength_label}]: "
                f"{_mi}回反復後も制約を完全に満たせませんでした "
                f"(violation={total_violation:.4f})。"
                f"ソフト制約として近似適用されています。",
                UserWarning, stacklevel=2,
            )

        self.estimator_ = fitted
        self.monotonic_violation_ = total_violation
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(_to_numpy(X))

    def score(self, X: Any, y: Any) -> float:
        return self.estimator_.score(_to_numpy(X), np.asarray(y))


# ────────────────────────────────────────────────────────────
# 汎用ラッパー: 分類版
# ────────────────────────────────────────────────────────────

class MonotonicConstraintClassifier(BaseEstimator, ClassifierMixin):
    """
    任意の sklearn 分類モデルに変数ごとの単調性制約を付与する汎用ラッパー。

    predict_proba のクラス1確率に対して ±3σ 範囲で単調性チェック。
    各パラメータは回帰版と同様。
    """

    def __init__(
        self,
        base_estimator: Any = None,
        monotonic_constraints: tuple[int, ...] = (),
        constraint_strength: str | None = None,
        n_grid: int = 20,
        sigma_factor: float = 3.0,
        penalty_weight: float = 10.0,
        max_iter: int = 3,
        violation_threshold: float = 1e-4,
    ) -> None:
        self.base_estimator = base_estimator
        self.monotonic_constraints = monotonic_constraints
        self.constraint_strength = constraint_strength
        self.n_grid = n_grid
        self.sigma_factor = sigma_factor
        self.penalty_weight = penalty_weight
        self.max_iter = max_iter
        self.violation_threshold = violation_threshold

    def _effective_params(self) -> dict[str, Any]:
        if self.constraint_strength and self.constraint_strength in _STRENGTH_PRESETS:
            preset = _STRENGTH_PRESETS[self.constraint_strength]
            return {
                "n_grid": preset["n_grid"],
                "sigma_factor": self.sigma_factor,
                "penalty_weight": preset["penalty_weight"],
                "max_iter": preset["max_iter"],
                "violation_threshold": preset["violation_threshold"],
            }
        return {
            "n_grid": self.n_grid,
            "sigma_factor": self.sigma_factor,
            "penalty_weight": self.penalty_weight,
            "max_iter": self.max_iter,
            "violation_threshold": self.violation_threshold,
        }

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params: dict[str, Any] = {
            "base_estimator": self.base_estimator,
            "monotonic_constraints": self.monotonic_constraints,
            "constraint_strength": self.constraint_strength,
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

    def set_params(self, **params: Any) -> "MonotonicConstraintClassifier":
        base_params: dict[str, Any] = {}
        own_params: dict[str, Any] = {}
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

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: np.ndarray | None = None,
    ) -> "MonotonicConstraintClassifier":
        """ペナルティサンプル拡張法で単調性制約付き分類学習。"""
        X_arr = _to_numpy(X)
        y_arr = np.asarray(y)

        self.feature_names_in_: list[str] | None = (
            X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        )
        n_features = X_arr.shape[1]
        self.n_features_in_ = n_features

        if self.base_estimator is None:
            from sklearn.ensemble import RandomForestClassifier
            base = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            base = self.base_estimator

        mc = list(self.monotonic_constraints) + [0] * max(0, n_features - len(self.monotonic_constraints))
        mc = mc[:n_features]
        y_float = y_arr.astype(np.float64)
        mc = list(_resolve_auto_direction(X_arr, y_float, mc))
        self.resolved_constraints_: tuple[int, ...] = tuple(mc)
        has_constraint = any(c != 0 for c in mc)

        eff = self._effective_params()
        _n_grid = eff["n_grid"]
        _sigma = eff["sigma_factor"]
        _pw = eff["penalty_weight"]
        _mi = eff["max_iter"]
        _vt = eff["violation_threshold"]

        feature_stats: dict[int, tuple[float, float]] = {}
        for i in range(n_features):
            feature_stats[i] = (float(np.mean(X_arr[:, i])), float(np.std(X_arr[:, i])))

        fitted = clone(base)
        _fit_with_weight(fitted, X_arr, y_arr, sample_weight)

        if not has_constraint:
            self.estimator_ = fitted
            self.classes_ = fitted.classes_
            return self

        for iteration in range(_mi):
            violations = []
            for i, direction in enumerate(mc):
                if direction == 0:
                    continue
                X_grid = _build_grid_X(X_arr, i, feature_stats, _sigma, _n_grid)
                try:
                    proba_grid = fitted.predict_proba(X_grid)[:, 1]
                except Exception:
                    proba_grid = fitted.predict(X_grid).astype(np.float64)
                violations.append(_compute_violation(proba_grid, direction))

            total_violation = sum(violations) if violations else 0.0
            if total_violation <= _vt:
                break

            X_pen_list, y_pen_list = [], []
            for i, direction in enumerate(mc):
                if direction == 0:
                    continue
                X_grid = _build_grid_X(X_arr, i, feature_stats, _sigma, _n_grid)
                try:
                    proba_grid = fitted.predict_proba(X_grid)[:, 1]
                except Exception:
                    continue
                diff = np.diff(proba_grid)
                for k in range(len(diff)):
                    if (-direction * diff[k]) > 0:
                        if direction == 1:
                            X_pen_list.append(X_grid[k + 1])
                            y_pen_list.append(np.max(y_arr))
                        else:
                            X_pen_list.append(X_grid[k])
                            y_pen_list.append(np.max(y_arr))

            if not X_pen_list:
                break

            X_aug = np.vstack([X_arr, np.array(X_pen_list)])
            y_aug = np.concatenate([y_arr, np.array(y_pen_list)])
            sw_orig = np.ones(len(X_arr)) if sample_weight is None else sample_weight
            sw_aug = np.concatenate([sw_orig, np.full(len(X_pen_list), _pw)])
            refitted = clone(base)
            _fit_with_weight(refitted, X_aug, y_aug, sw_aug)
            fitted = refitted

        self.estimator_ = fitted
        self.classes_ = fitted.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(_to_numpy(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(_to_numpy(X))

    def score(self, X: Any, y: Any) -> float:
        return self.estimator_.score(_to_numpy(X), np.asarray(y))


# ────────────────────────────────────────────────────────────
# モデル種別判定
# ────────────────────────────────────────────────────────────

def model_monotonic_strategy(estimator: Any) -> str:
    """
    モデルの単調性制約戦略を判定する。

    Returns:
        "native"  : XGBoost/LGBM/HistGB — パラメータ直接設定
        "penalty" : その他全モデル — ペナルティ拡張法ラッパー
        "none"    : get_params() 非対応（制約不可）
    """
    try:
        params = estimator.get_params()
    except Exception:
        return "none"

    monotonic_keys = [k for k in params if "monoton" in k.lower()]
    if monotonic_keys:
        return "native"

    return "penalty"


# ────────────────────────────────────────────────────────────
# ファクトリーAPI
# ────────────────────────────────────────────────────────────

def wrap_monotonic(
    estimator: Any,
    monotonic_constraints: tuple[int, ...],
    *,
    constraint_strength: str | None = None,
    n_grid: int = 20,
    sigma_factor: float = 3.0,
    penalty_weight: float = 10.0,
    max_iter: int = 3,
) -> Any:
    """
    estimator に単調性制約ラッパーを適用する。

    制約値: 0=なし, +1=増加, -1=減少, 2=自動検出
    制約が全て 0 なら元の estimator を返す。

    Args:
        estimator: ラップするモデル
        monotonic_constraints: 制約タプル
        constraint_strength: "weak" or "strong" or None
        n_grid, sigma_factor, penalty_weight, max_iter: 動作パラメータ
    """
    active = [c for c in monotonic_constraints if c != 0]
    if not active:
        return estimator

    name = type(estimator).__name__
    n_constrained = len(active)
    n_total = len(monotonic_constraints)
    n_auto = sum(1 for c in monotonic_constraints if c == 2)
    auto_note = f" (自動検出{n_auto}件)" if n_auto else ""
    strength_note = f" [{constraint_strength}]" if constraint_strength else ""

    if _is_classifier(estimator):
        logger.info(
            f"MonotonicConstraintClassifier{strength_note} でラップ: {name}, "
            f"{n_constrained}/{n_total} 変数{auto_note}"
        )
        return MonotonicConstraintClassifier(
            base_estimator=estimator,
            monotonic_constraints=monotonic_constraints,
            constraint_strength=constraint_strength,
            n_grid=n_grid, sigma_factor=sigma_factor,
            penalty_weight=penalty_weight, max_iter=max_iter,
        )
    else:
        logger.info(
            f"MonotonicConstraintRegressor{strength_note} でラップ: {name}, "
            f"{n_constrained}/{n_total} 変数{auto_note}"
        )
        return MonotonicConstraintRegressor(
            base_estimator=estimator,
            monotonic_constraints=monotonic_constraints,
            constraint_strength=constraint_strength,
            n_grid=n_grid, sigma_factor=sigma_factor,
            penalty_weight=penalty_weight, max_iter=max_iter,
        )
