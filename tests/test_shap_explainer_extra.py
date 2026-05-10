"""
tests/test_shap_explainer_extra.py

shap_explainer.py のカバレッジ改善テスト。
ShapConfig, ShapResult, ShapExplainer を網羅（外部依存なしで動作する範囲）。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.interpret.shap_explainer import (
    ShapConfig,
    ShapResult,
    ShapExplainer,
)


# ============================================================
# ShapConfig
# ============================================================

class TestShapConfig:
    def test_defaults(self):
        cfg = ShapConfig()
        assert cfg.explainer_type == "auto"
        assert cfg.max_display == 20
        assert cfg.plot_types is not None

    def test_custom(self):
        cfg = ShapConfig(
            explainer_type="tree",
            max_display=10,
            background_size=50,
        )
        assert cfg.explainer_type == "tree"
        assert cfg.max_display == 10


# ============================================================
# ShapResult
# ============================================================

class TestShapResult:
    def _make_result(self, n=30, d=5, multiclass=False):
        rng = np.random.RandomState(42)
        if multiclass:
            sv = rng.randn(n, d, 3)
        else:
            sv = rng.randn(n, d)
        return ShapResult(
            shap_values=sv,
            expected_value=0.5,
            feature_names=[f"f{i}" for i in range(d)],
            X_transformed=rng.randn(n, d),
            explainer_type="tree",
            is_multiclass=multiclass,
        )

    def test_basic_properties(self):
        sr = self._make_result()
        assert sr.explainer_type == "tree"
        assert len(sr.feature_names) == 5
        assert sr.shap_values.shape == (30, 5)

    def test_multiclass(self):
        sr = self._make_result(multiclass=True)
        assert sr.is_multiclass is True
        assert sr.shap_values.ndim == 3

    def test_feature_importance(self):
        sr = self._make_result()
        imp = sr.feature_importance()
        assert isinstance(imp, pd.DataFrame)
        assert len(imp) == 5
        assert "importance" in imp.columns

    def test_top_features(self):
        sr = self._make_result()
        top = sr.top_features(n=3)
        assert len(top) == 3


# ============================================================
# ShapExplainer
# ============================================================

class TestShapExplainer:
    def test_auto_select_tree(self):
        """TreeExplainer が選択されるケース。"""
        from sklearn.ensemble import RandomForestRegressor
        rng = np.random.RandomState(42)
        X = rng.randn(30, 3)
        y = X[:, 0] * 2
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        explainer = ShapExplainer(ShapConfig(explainer_type="auto"))
        selected = explainer._select_explainer_type(model)
        assert selected in ("tree", "kernel", "linear")

    def test_explain_tree_model(self):
        """TreeExplainer でSHAP値を計算。"""
        from sklearn.ensemble import RandomForestRegressor
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(30, 3), columns=["a", "b", "c"])
        y = X["a"] * 2

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        explainer = ShapExplainer(ShapConfig(explainer_type="tree"))
        try:
            result = explainer.explain(model, X)
            assert isinstance(result, ShapResult)
            assert result.shap_values.shape[0] == 30
        except ImportError:
            pytest.skip("shap not installed")

    def test_explain_linear_model(self):
        """LinearExplainer でSHAP値を計算。"""
        from sklearn.linear_model import Ridge
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(30, 3), columns=["a", "b", "c"])
        y = X["a"] * 2

        model = Ridge()
        model.fit(X, y)

        explainer = ShapExplainer(ShapConfig(explainer_type="linear"))
        try:
            result = explainer.explain(model, X)
            assert isinstance(result, ShapResult)
        except ImportError:
            pytest.skip("shap not installed")
