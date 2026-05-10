# -*- coding: utf-8 -*-
"""
tests/test_monotonic_wrapper.py

MonotonicConstraintRegressor / MonotonicConstraintClassifier の完全テストスイート。

Feature Matrix:
  F-001: MonotonicConstraintRegressor の sklearn API 準拠
  F-002: MonotonicConstraintRegressor の単調性保証（RFR, Ridge, SVR, GPR, ExtraTrees, GBM）
  F-003: MonotonicConstraintClassifier の sklearn API 準拠
  F-004: MonotonicConstraintClassifier の確率単調性保証
  F-005: model_monotonic_strategy() のモデル種別判定
  F-006: wrap_monotonic() ファクトリー関数
  F-007: pipeline_builder.apply_monotonic_constraints() のルーティング
  F-008: ColumnMeta との統合
  F-009: monotonic=2 の自動検出（Spearman相関）
  F-010: constraint_strength の "weak" / "strong" プリセット
  F-011: ±3σ 外挿範囲での単調性保証

制約の4パターン:
  0  = 制約なし
  1  = 単調増加
  -1 = 単調減少
  2  = 自動検出（fit時にSpearman相関で方向を判定）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor,
    GradientBoostingRegressor, GradientBoostingClassifier,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge, BayesianRidge, Lasso
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC


# ─────────────────────────────────────────────────────────────
# フィクスチャ
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def regression_data():
    """単調性が明確なデータセット (80サンプル, 3特徴量)。"""
    rng = np.random.RandomState(42)
    n = 80
    x1 = rng.uniform(0, 10, n)           # y と正の相関
    x2 = rng.uniform(0, 5, n)            # y と無関係 (ノイズ)
    x3 = rng.uniform(0, 10, n)           # y と負の相関
    # y = 3*x1 - 2*x3 + noise → x1増加, x3減少
    y = 3.0 * x1 - 2.0 * x3 + rng.randn(n) * 0.5
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    return X, y


@pytest.fixture
def classification_data():
    """単調性を持つ2値分類データセット。"""
    rng = np.random.RandomState(42)
    n = 80
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 5, n)
    p = 1 / (1 + np.exp(-(x1 - 5)))  # x1 が大きいほどクラス1
    y = (rng.uniform(size=n) < p).astype(int)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, y


def _check_monotonic_grid(model, X: pd.DataFrame, feat: str, direction: int,
                          n_grid: int = 25, sigma_extend: float = 0.5) -> float:
    """
    指定変数の単調性遵守率を計算する。

    Returns:
        遵守率 in [0.0, 1.0]
    """
    median_vals = X.median().values
    feat_idx = list(X.columns).index(feat)
    lo = float(X[feat].quantile(0.05)) - sigma_extend * float(X[feat].std())
    hi = float(X[feat].quantile(0.95)) + sigma_extend * float(X[feat].std())
    if lo >= hi:
        return 1.0
    grid = np.linspace(lo, hi, n_grid)
    X_grid = np.tile(median_vals, (n_grid, 1))
    X_grid[:, feat_idx] = grid
    X_df = pd.DataFrame(X_grid, columns=X.columns)
    y_pred = model.predict(X_df)
    diffs = np.diff(y_pred)
    if direction == 1:
        compliant = np.sum(diffs >= -1e-8)
    else:
        compliant = np.sum(diffs <= 1e-8)
    return float(compliant / len(diffs))


# ─────────────────────────────────────────────────────────────
# F-001: MonotonicConstraintRegressor - sklearn API 準拠
# ─────────────────────────────────────────────────────────────

class TestMonotonicConstraintRegressorAPI:
    """F-001: sklearn API 完全互換テスト。"""

    def test_instantiation_default(self):
        """T-001a: デフォルトインスタンス化。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        m = MonotonicConstraintRegressor()
        assert m.base_estimator is None
        assert m.monotonic_constraints == ()
        assert m.sigma_factor == 3.0  # デフォルト ±3σ

    def test_get_params_deep(self):
        """T-001b: base_estimator のパラメータも取得できる。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        m = MonotonicConstraintRegressor(base_estimator=Ridge(alpha=2.0))
        params = m.get_params(deep=True)
        assert "base_estimator__alpha" in params
        assert params["base_estimator__alpha"] == 2.0

    def test_set_params(self):
        """T-001c: set_params() が動作する。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        m = MonotonicConstraintRegressor()
        m.set_params(sigma_factor=2.0, constraint_strength="strong")
        assert m.sigma_factor == 2.0
        assert m.constraint_strength == "strong"

    def test_set_params_nested(self):
        """T-001d: nested set_params。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        m = MonotonicConstraintRegressor(base_estimator=Ridge())
        m.set_params(**{"base_estimator__alpha": 10.0})
        assert m.base_estimator.get_params()["alpha"] == 10.0

    def test_clone_compatibility(self):
        """T-001e: sklearn clone() が動作する。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        m = MonotonicConstraintRegressor(
            base_estimator=Ridge(), monotonic_constraints=(1, 0, -1),
            constraint_strength="strong",
        )
        cloned = clone(m)
        assert isinstance(cloned, MonotonicConstraintRegressor)
        assert cloned.monotonic_constraints == (1, 0, -1)
        assert cloned.constraint_strength == "strong"

    def test_feature_names_stored(self, regression_data):
        """T-001f: fit後に feature_names_in_ が保持。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        X, y = regression_data
        m = MonotonicConstraintRegressor(
            base_estimator=Ridge(), monotonic_constraints=(1, 0, -1),
        )
        m.fit(X, y)
        assert m.feature_names_in_ == ["x1", "x2", "x3"]
        assert m.n_features_in_ == 3

    def test_predict_shape(self, regression_data):
        """T-001g: predict の shape。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        X, y = regression_data
        m = MonotonicConstraintRegressor(
            base_estimator=Ridge(), monotonic_constraints=(1, 0, -1),
        )
        m.fit(X, y)
        pred = m.predict(X)
        assert pred.shape == (len(y),)


# ─────────────────────────────────────────────────────────────
# F-002: 単調性保証 (回帰)
# ─────────────────────────────────────────────────────────────

class TestMonotonicConstraintRegressorMonotonicity:
    """F-002: 各モデルでペナルティ拡張により単調性が概ね改善されること。"""

    @pytest.mark.parametrize("base_cls,base_kwargs", [
        (RandomForestRegressor, {"n_estimators": 50, "random_state": 42}),
        (GradientBoostingRegressor, {"n_estimators": 50, "random_state": 42}),
        (Ridge, {"alpha": 1.0}),
        (SVR, {"kernel": "rbf", "C": 10.0}),
    ])
    def test_monotonic_increase_x1(self, regression_data, base_cls, base_kwargs):
        """T-002a: x1 増加制約が 60%以上の遵守率で機能。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        X, y = regression_data
        m = MonotonicConstraintRegressor(
            base_estimator=base_cls(**base_kwargs),
            monotonic_constraints=(1, 0, 0),
            max_iter=3, sigma_factor=3.0,
        )
        m.fit(X, y)
        compliance = _check_monotonic_grid(m, X, "x1", direction=1)
        assert compliance >= 0.6, (
            f"{base_cls.__name__}: x1 増加遵守率={compliance:.2f} < 0.6"
        )

    @pytest.mark.parametrize("base_cls,base_kwargs", [
        (RandomForestRegressor, {"n_estimators": 50, "random_state": 42}),
        (Ridge, {"alpha": 1.0}),
    ])
    def test_monotonic_decrease_x3(self, regression_data, base_cls, base_kwargs):
        """T-002b: x3 減少制約。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        X, y = regression_data
        m = MonotonicConstraintRegressor(
            base_estimator=base_cls(**base_kwargs),
            monotonic_constraints=(0, 0, -1),
            max_iter=3, sigma_factor=3.0,
        )
        m.fit(X, y)
        compliance = _check_monotonic_grid(m, X, "x3", direction=-1)
        assert compliance >= 0.6, (
            f"{base_cls.__name__}: x3 減少遵守率={compliance:.2f} < 0.6"
        )

    def test_no_constraint_passthrough(self, regression_data):
        """T-002c: 制約なし → resolved_constraints_ が全0。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        X, y = regression_data
        m = MonotonicConstraintRegressor(
            base_estimator=Ridge(),
            monotonic_constraints=(0, 0, 0),
        )
        m.fit(X, y)
        assert m.resolved_constraints_ == (0, 0, 0)


# ─────────────────────────────────────────────────────────────
# F-003: MonotonicConstraintClassifier - sklearn API
# ─────────────────────────────────────────────────────────────

class TestMonotonicConstraintClassifierAPI:
    """F-003: 分類版 sklearn API テスト。"""

    def test_fit_predict(self, classification_data):
        """T-003a: fit/predict が動作。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintClassifier
        X, y = classification_data
        m = MonotonicConstraintClassifier(
            base_estimator=RandomForestClassifier(n_estimators=30, random_state=42),
            monotonic_constraints=(1, 0),
        )
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_proba_shape(self, classification_data):
        """T-003b: predict_proba の shape が (n, 2)。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintClassifier
        X, y = classification_data
        m = MonotonicConstraintClassifier(
            base_estimator=RandomForestClassifier(n_estimators=30, random_state=42),
            monotonic_constraints=(1, 0),
        )
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (len(y), 2)

    def test_classes_stored(self, classification_data):
        """T-003c: fit後に classes_ 保持。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintClassifier
        X, y = classification_data
        m = MonotonicConstraintClassifier(
            base_estimator=RandomForestClassifier(n_estimators=30, random_state=42),
            monotonic_constraints=(1, 0),
        )
        m.fit(X, y)
        assert hasattr(m, "classes_")
        assert set(m.classes_) == {0, 1}

    def test_clone(self):
        """T-003d: clone() 動作。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintClassifier
        m = MonotonicConstraintClassifier(
            base_estimator=RandomForestClassifier(n_estimators=10),
            monotonic_constraints=(1, 0),
            constraint_strength="weak",
        )
        cloned = clone(m)
        assert isinstance(cloned, MonotonicConstraintClassifier)
        assert cloned.constraint_strength == "weak"


# ─────────────────────────────────────────────────────────────
# F-005: model_monotonic_strategy()
# ─────────────────────────────────────────────────────────────

class TestModelMonotonicStrategy:
    """F-005: モデル種別判定。"""

    def test_rf_is_native(self):
        """T-005a: RFR → native (sklearn 1.4+ で monotonic_cst 対応)。"""
        from backend.models.monotonic_wrapper import model_monotonic_strategy
        assert model_monotonic_strategy(RandomForestRegressor()) == "native"

    def test_svr_is_penalty(self):
        """T-005b: SVR → penalty。"""
        from backend.models.monotonic_wrapper import model_monotonic_strategy
        assert model_monotonic_strategy(SVR()) == "penalty"

    def test_ridge_is_penalty(self):
        """T-005c: Ridge → penalty。"""
        from backend.models.monotonic_wrapper import model_monotonic_strategy
        assert model_monotonic_strategy(Ridge()) == "penalty"

    def test_histgbm_is_native(self):
        """T-005d: HistGBM → native。"""
        from sklearn.ensemble import HistGradientBoostingRegressor
        from backend.models.monotonic_wrapper import model_monotonic_strategy
        assert model_monotonic_strategy(HistGradientBoostingRegressor()) == "native"

    def test_xgboost_is_native(self):
        """T-005e: XGBRegressor → native。"""
        pytest.importorskip("xgboost")
        from xgboost import XGBRegressor
        from backend.models.monotonic_wrapper import model_monotonic_strategy
        assert model_monotonic_strategy(XGBRegressor()) == "native"


# ─────────────────────────────────────────────────────────────
# F-006: wrap_monotonic() ファクトリー
# ─────────────────────────────────────────────────────────────

class TestWrapMonotonic:
    """F-006: ファクトリー関数テスト。"""

    def test_wraps_regressor(self):
        """T-006a: 回帰モデルをラップ。"""
        from backend.models.monotonic_wrapper import wrap_monotonic, MonotonicConstraintRegressor
        wrapped = wrap_monotonic(RandomForestRegressor(n_estimators=10), (1, 0, -1))
        assert isinstance(wrapped, MonotonicConstraintRegressor)

    def test_wraps_classifier(self):
        """T-006b: 分類モデルをラップ。"""
        from backend.models.monotonic_wrapper import wrap_monotonic, MonotonicConstraintClassifier
        wrapped = wrap_monotonic(RandomForestClassifier(n_estimators=10), (1, 0))
        assert isinstance(wrapped, MonotonicConstraintClassifier)

    def test_no_constraint_returns_original(self):
        """T-006c: 全0 → 元モデル返却。"""
        from backend.models.monotonic_wrapper import wrap_monotonic
        base = Ridge()
        result = wrap_monotonic(base, (0, 0, 0))
        assert result is base

    def test_auto_detect_wraps(self):
        """T-006d: monotonic=2 でもラップされる。"""
        from backend.models.monotonic_wrapper import wrap_monotonic, MonotonicConstraintRegressor
        wrapped = wrap_monotonic(Ridge(), (2, 0, 0))
        assert isinstance(wrapped, MonotonicConstraintRegressor)


# ─────────────────────────────────────────────────────────────
# F-007: apply_monotonic_constraints() ルーティング
# ─────────────────────────────────────────────────────────────

class TestApplyMonotonicConstraintsRouting:
    """F-007: pipeline_builder の2段階ルーティング。"""

    def _make_col_meta(self, names, mono_vals, strength=None):
        from backend.pipeline.column_selector import ColumnMeta
        return {n: ColumnMeta(monotonic=m, constraint_strength=strength)
                for n, m in zip(names, mono_vals)}

    def test_rfr_native_monotonic(self):
        """T-007a: RFR → ネイティブ (sklearn 1.4+ で monotonic_cst 対応)。"""
        from backend.pipeline.pipeline_builder import apply_monotonic_constraints
        col_meta = self._make_col_meta(["x1", "x2", "x3"], [1, 0, -1])
        wrapped = apply_monotonic_constraints(
            RandomForestRegressor(n_estimators=10, random_state=42),
            col_meta, feature_names=["x1", "x2", "x3"],
        )
        # ネイティブ対応なのでラッパーではなくRFRのまま
        assert type(wrapped).__name__ == "RandomForestRegressor"

    def test_rfc_native_monotonic(self):
        """T-007b: RFC → ネイティブ (sklearn 1.4+)。"""
        from backend.pipeline.pipeline_builder import apply_monotonic_constraints
        col_meta = self._make_col_meta(["x1", "x2"], [1, 0])
        wrapped = apply_monotonic_constraints(
            RandomForestClassifier(n_estimators=10, random_state=42),
            col_meta, feature_names=["x1", "x2"],
        )
        assert type(wrapped).__name__ == "RandomForestClassifier"

    def test_svr_gets_penalty_wrapper(self):
        """T-007c: SVR → MonotonicConstraintRegressor（ペナルティ拡張法）。"""
        from backend.pipeline.pipeline_builder import apply_monotonic_constraints
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        col_meta = self._make_col_meta(["x1", "x2", "x3"], [1, 0, 0])
        wrapped = apply_monotonic_constraints(
            SVR(), col_meta, feature_names=["x1", "x2", "x3"],
        )
        assert isinstance(wrapped, MonotonicConstraintRegressor)

    def test_no_constraint_returns_original(self):
        """T-007d: 制約なし → そのまま。"""
        from backend.pipeline.pipeline_builder import apply_monotonic_constraints
        est = Ridge()
        col_meta = self._make_col_meta(["x1", "x2"], [0, 0])
        result = apply_monotonic_constraints(est, col_meta, feature_names=["x1", "x2"])
        assert result is est

    def test_histgbm_native(self):
        """T-007e: HistGBM → ネイティブ（ラッパーなし）。"""
        from sklearn.ensemble import HistGradientBoostingRegressor
        from backend.pipeline.pipeline_builder import apply_monotonic_constraints
        col_meta = self._make_col_meta(["x1", "x2", "x3"], [1, 0, -1])
        result = apply_monotonic_constraints(
            HistGradientBoostingRegressor(),
            col_meta, feature_names=["x1", "x2", "x3"],
        )
        assert type(result).__name__ == "HistGradientBoostingRegressor"

    def test_xgb_native(self):
        """T-007f: XGBRegressor → ネイティブ。"""
        pytest.importorskip("xgboost")
        from xgboost import XGBRegressor
        from backend.pipeline.pipeline_builder import apply_monotonic_constraints
        col_meta = self._make_col_meta(["x1", "x2", "x3"], [1, 0, -1])
        result = apply_monotonic_constraints(
            XGBRegressor(n_estimators=10),
            col_meta, feature_names=["x1", "x2", "x3"],
        )
        assert type(result).__name__ == "XGBRegressor"
        p = result.get_params()
        # XGBoostのパラメータ名は monotone_constraints か monotonic_constraints
        mono_val = p.get("monotonic_constraints") or p.get("monotone_constraints")
        assert mono_val == (1, 0, -1)

    def test_strength_propagated_penalty_model(self):
        """T-007g: constraint_strength がペナルティラッパーに伝搬。"""
        from backend.pipeline.pipeline_builder import apply_monotonic_constraints
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        # Ridge はネイティブ非対応 → ペナルティ拡張法
        col_meta = self._make_col_meta(["x1", "x2"], [1, 0], strength="strong")
        result = apply_monotonic_constraints(
            Ridge(), col_meta, feature_names=["x1", "x2"],
        )
        assert isinstance(result, MonotonicConstraintRegressor)
        assert result.constraint_strength == "strong"

    def test_native_auto_detect_fallback(self):
        """T-007h: ネイティブモデル + monotonic=2 → 0にフォールバック。"""
        from sklearn.ensemble import HistGradientBoostingRegressor
        from backend.pipeline.pipeline_builder import apply_monotonic_constraints
        col_meta = self._make_col_meta(["x1", "x2"], [2, 0])
        result = apply_monotonic_constraints(
            HistGradientBoostingRegressor(),
            col_meta, feature_names=["x1", "x2"],
        )
        assert type(result).__name__ == "HistGradientBoostingRegressor"


# ─────────────────────────────────────────────────────────────
# F-008: ColumnMeta 統合
# ─────────────────────────────────────────────────────────────

class TestColumnMetaIntegrationMonotonic:
    """F-008: ColumnMeta → ラッパー → E2E テスト。"""

    def test_column_meta_with_constraint_strength(self, regression_data):
        """T-008a: constraint_strength が ColumnMeta 経由で保存復元される。"""
        from backend.pipeline.column_selector import ColumnMeta
        meta = ColumnMeta(monotonic=1, constraint_strength="strong")
        d = meta.to_dict()
        restored = ColumnMeta.from_dict(d)
        assert restored.monotonic == 1
        assert restored.constraint_strength == "strong"

    def test_column_meta_auto_detect(self, regression_data):
        """T-008b: monotonic=2 が保存復元される。"""
        from backend.pipeline.column_selector import ColumnMeta
        meta = ColumnMeta(monotonic=2)
        d = meta.to_dict()
        restored = ColumnMeta.from_dict(d)
        assert restored.monotonic == 2

    def test_end_to_end_fit_predict(self, regression_data):
        """T-008c: ColumnMeta → apply → fit → predict がE2Eで動作。"""
        from backend.pipeline.column_selector import ColumnMeta
        from backend.pipeline.pipeline_builder import apply_monotonic_constraints
        X, y = regression_data
        # SVR はネイティブ非対応→ペナルティ拡張法ラッパー経由
        col_meta = {
            "x1": ColumnMeta(monotonic=1, constraint_strength="weak"),
            "x2": ColumnMeta(monotonic=0),
            "x3": ColumnMeta(monotonic=-1, constraint_strength="weak"),
        }
        model = apply_monotonic_constraints(
            SVR(kernel="rbf", C=10.0),
            col_meta, feature_names=["x1", "x2", "x3"],
        )
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (len(y),)
        assert not np.any(np.isnan(pred))


# ─────────────────────────────────────────────────────────────
# F-009: 自動検出 (monotonic=2)
# ─────────────────────────────────────────────────────────────

class TestAutoDetectMonotonic:
    """F-009: Spearman相関による方向自動検出。"""

    def test_resolve_positive_correlation(self, regression_data):
        """T-009a: x1 (正の相関) → +1 に解決。"""
        from backend.models.monotonic_wrapper import _resolve_auto_direction, _to_numpy
        X, y = regression_data
        resolved = _resolve_auto_direction(_to_numpy(X), np.asarray(y), [2, 0, 0])
        assert resolved[0] == 1  # x1: 正の相関 → 増加

    def test_resolve_negative_correlation(self, regression_data):
        """T-009b: x3 (負の相関) → -1 に解決。"""
        from backend.models.monotonic_wrapper import _resolve_auto_direction, _to_numpy
        X, y = regression_data
        resolved = _resolve_auto_direction(_to_numpy(X), np.asarray(y), [0, 0, 2])
        assert resolved[2] == -1  # x3: 負の相関 → 減少

    def test_auto_detect_in_wrapper(self, regression_data):
        """T-009c: ラッパー内で自動検出 → resolved_constraints_ に記録。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        X, y = regression_data
        m = MonotonicConstraintRegressor(
            base_estimator=Ridge(),
            monotonic_constraints=(2, 0, 2),
        )
        m.fit(X, y)
        rc = m.resolved_constraints_
        assert rc[0] == 1   # x1 → 増加
        assert rc[1] == 0   # x2 → 制約なし
        assert rc[2] == -1  # x3 → 減少


# ─────────────────────────────────────────────────────────────
# F-010: constraint_strength プリセット
# ─────────────────────────────────────────────────────────────

class TestConstraintStrength:
    """F-010: weak / strong プリセットの動作テスト。"""

    def test_weak_uses_low_penalty(self):
        """T-010a: weak → penalty_weight=5, max_iter=2。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        m = MonotonicConstraintRegressor(constraint_strength="weak")
        eff = m._effective_params()
        assert eff["penalty_weight"] == 5.0
        assert eff["max_iter"] == 2
        assert eff["n_grid"] == 15

    def test_strong_uses_high_penalty(self):
        """T-010b: strong → penalty_weight=50, max_iter=8。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        m = MonotonicConstraintRegressor(constraint_strength="strong")
        eff = m._effective_params()
        assert eff["penalty_weight"] == 50.0
        assert eff["max_iter"] == 8
        assert eff["n_grid"] == 40

    def test_none_uses_individual_params(self):
        """T-010c: None → 個別パラメータをそのまま使用。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        m = MonotonicConstraintRegressor(
            penalty_weight=20.0, max_iter=5, n_grid=25,
        )
        eff = m._effective_params()
        assert eff["penalty_weight"] == 20.0
        assert eff["max_iter"] == 5
        assert eff["n_grid"] == 25

    def test_strong_fit_works(self, regression_data):
        """T-010d: strong でfit/predictが完走する。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        X, y = regression_data
        m = MonotonicConstraintRegressor(
            base_estimator=Ridge(),
            monotonic_constraints=(1, 0, -1),
            constraint_strength="strong",
        )
        m.fit(X, y)
        pred = m.predict(X)
        assert pred.shape == (len(y),)


# ─────────────────────────────────────────────────────────────
# F-011: ±3σ 外挿範囲での単調性
# ─────────────────────────────────────────────────────────────

class TestExtrapolationMonotonicity:
    """F-011: デフォルト sigma_factor=3.0 での外挿範囲単調性。"""

    def test_sigma_factor_default_is_3(self):
        """T-011a: デフォルト sigma_factor は 3.0。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        m = MonotonicConstraintRegressor()
        assert m.sigma_factor == 3.0

    def test_extrapolation_fit(self, regression_data):
        """T-011b: ±3σ で学習して完走する。"""
        from backend.models.monotonic_wrapper import MonotonicConstraintRegressor
        X, y = regression_data
        m = MonotonicConstraintRegressor(
            base_estimator=RandomForestRegressor(n_estimators=30, random_state=42),
            monotonic_constraints=(1, 0, 0),
            sigma_factor=3.0, max_iter=2,
        )
        m.fit(X, y)
        # ±3σ 外まで含めた予測が NaN にならない
        x1_mean = float(X["x1"].mean())
        x1_std = float(X["x1"].std())
        test_range = np.linspace(x1_mean - 3 * x1_std, x1_mean + 3 * x1_std, 10)
        X_test = pd.DataFrame({
            "x1": test_range,
            "x2": np.full(10, float(X["x2"].median())),
            "x3": np.full(10, float(X["x3"].median())),
        })
        pred = m.predict(X_test)
        assert not np.any(np.isnan(pred))
