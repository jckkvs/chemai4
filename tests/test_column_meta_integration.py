"""
tests/test_column_meta_integration.py

ColumnMeta の全パイプライン統合テスト。

テスト対象:
  - ColumnMeta の拡張フィールド (scale_hint, description, fixed)
  - to_dict / from_dict シリアライズ
  - extract_monotonic_from_column_meta
  - build_column_meta_dict
  - FeatureSelector の fixed 変数保護
  - AutoMLEngine の column_meta_dict 受け入れ
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ============================================================
# T-001: ColumnMeta の拡張フィールド
# ============================================================

class TestColumnMetaExtended:
    """F-001: ColumnMeta 拡張 - scale_hint, description, fixed の追加フィールド。"""

    def test_default_values(self):
        """T-001a: デフォルト値が正しい。"""
        from backend.pipeline.column_selector import ColumnMeta
        m = ColumnMeta()
        assert m.monotonic == 0
        assert m.linearity == "unknown"
        assert m.group is None
        assert m.scale_hint is None
        assert m.description == ""
        assert m.fixed is False

    def test_custom_values(self):
        """T-001b: カスタム値が正しく設定される。"""
        from backend.pipeline.column_selector import ColumnMeta
        m = ColumnMeta(
            monotonic=1,
            linearity="linear",
            group="env",
            scale_hint="standard",
            description="温度変数",
            fixed=True,
        )
        assert m.monotonic == 1
        assert m.linearity == "linear"
        assert m.group == "env"
        assert m.scale_hint == "standard"
        assert m.description == "温度変数"
        assert m.fixed is True

    def test_to_dict(self):
        """T-001c: to_dict() がすべてのフィールドを返す。"""
        from backend.pipeline.column_selector import ColumnMeta
        m = ColumnMeta(monotonic=-1, linearity="nonlinear", group="grp1",
                       scale_hint="robust", description="desc", fixed=True)
        d = m.to_dict()
        assert d == {
            "monotonic": -1,
            "constraint_strength": None,
            "linearity": "nonlinear",
            "group": "grp1",
            "scale_hint": "robust",
            "description": "desc",
            "fixed": True,
        }

    def test_from_dict_roundtrip(self):
        """T-001d: from_dict → to_dict がラウンドトリップする。"""
        from backend.pipeline.column_selector import ColumnMeta
        original = ColumnMeta(monotonic=1, linearity="linear", group="g1",
                              scale_hint="minmax", description="x", fixed=False)
        d = original.to_dict()
        restored = ColumnMeta.from_dict(d)
        assert restored.monotonic == original.monotonic
        assert restored.linearity == original.linearity
        assert restored.group == original.group
        assert restored.scale_hint == original.scale_hint
        assert restored.description == original.description
        assert restored.fixed == original.fixed

    def test_from_dict_defaults(self):
        """T-001e: 不完全な辞書でも安全にデシリアライズできる。"""
        from backend.pipeline.column_selector import ColumnMeta
        m = ColumnMeta.from_dict({})
        assert m.monotonic == 0
        assert m.linearity == "unknown"
        assert m.group is None
        assert m.scale_hint is None
        assert m.fixed is False

    def test_from_dict_empty_string_becomes_none(self):
        """T-001f: group/scale_hint の空文字列は None に変換される。"""
        from backend.pipeline.column_selector import ColumnMeta
        m = ColumnMeta.from_dict({"group": "", "scale_hint": ""})
        assert m.group is None
        assert m.scale_hint is None


# ============================================================
# T-002: column_meta_editor ユーティリティ
# ============================================================

class TestColumnMetaEditorUtils:
    """F-002: state ↔ ColumnMeta 辞書 変換ユーティリティ。"""

    def test_build_column_meta_dict_from_state(self):
        """T-002a: state["column_meta"] から ColumnMeta 辞書が構築できる。"""
        from frontend_nicegui.components.column_meta_editor import build_column_meta_dict
        from backend.pipeline.column_selector import ColumnMeta
        state = {
            "column_meta": {
                "temp": {"monotonic": 1, "linearity": "linear", "group": "env",
                         "scale_hint": "standard", "description": "温度", "fixed": False},
                "pressure": {"monotonic": -1, "linearity": "nonlinear", "group": "env",
                             "scale_hint": None, "description": "圧力", "fixed": True},
            }
        }
        col_meta = build_column_meta_dict(state)
        assert isinstance(col_meta["temp"], ColumnMeta)
        assert col_meta["temp"].monotonic == 1
        assert col_meta["pressure"].fixed is True

    def test_build_column_meta_dict_empty_state(self):
        """T-002b: column_meta が空の state でも空辞書を返す。"""
        from frontend_nicegui.components.column_meta_editor import build_column_meta_dict
        state = {}
        result = build_column_meta_dict(state)
        assert result == {}

    def test_extract_monotonic_from_column_meta(self):
        """T-002c: monotonic != 0 の列のみ抽出される。"""
        from frontend_nicegui.components.column_meta_editor import extract_monotonic_from_column_meta
        state = {
            "column_meta": {
                "a": {"monotonic": 1, "linearity": "unknown", "group": None,
                      "scale_hint": None, "description": "", "fixed": False},
                "b": {"monotonic": 0, "linearity": "unknown", "group": None,
                      "scale_hint": None, "description": "", "fixed": False},
                "c": {"monotonic": -1, "linearity": "unknown", "group": None,
                      "scale_hint": None, "description": "", "fixed": False},
            }
        }
        result = extract_monotonic_from_column_meta(state)
        assert result == {"a": 1, "c": -1}
        assert "b" not in result

    def test_extract_monotonic_merges_existing(self):
        """T-002d: state["monotonic_constraints"] とマージされ、直接設定が優先される。"""
        from frontend_nicegui.components.column_meta_editor import extract_monotonic_from_column_meta
        state = {
            "column_meta": {
                "a": {"monotonic": 1, "linearity": "unknown", "group": None,
                      "scale_hint": None, "description": "", "fixed": False},
            },
            "monotonic_constraints": {
                "a": -1,   # 直接設定が優先
                "b": 1,    # column_meta にないが追加される
            }
        }
        result = extract_monotonic_from_column_meta(state)
        assert result["a"] == -1   # 直接設定優先
        assert result["b"] == 1    # 直接設定から追加


# ============================================================
# T-003: FeatureSelector の fixed 変数保護
# ============================================================

class TestFeatureSelectorFixed:
    """F-003: fixed 変数保護 - 特徴量選択から除外されない。"""

    @pytest.fixture
    def sample_data(self):
        """回帰用サンプルデータ。"""
        np.random.seed(42)
        n = 50
        a = np.random.randn(n)
        b = np.random.randn(n)
        c_const = np.zeros(n)   # 定数列（Lassoは選ばない）
        y = a + 2 * b + np.random.randn(n) * 0.1
        X = pd.DataFrame({"a": a, "b": b, "c_fixed": c_const})
        return X, y

    def test_fixed_column_preserved_by_lasso(self, sample_data):
        """T-003a: Lasso が選ばない定数列でも fixed=True なら保持される。"""
        from backend.pipeline.column_selector import ColumnMeta
        from backend.pipeline.feature_selector import FeatureSelector, FeatureSelectorConfig
        X, y = sample_data
        col_meta = {
            "a": ColumnMeta(fixed=False),
            "b": ColumnMeta(fixed=False),
            "c_fixed": ColumnMeta(fixed=True),
        }
        cfg = FeatureSelectorConfig(method="lasso", task="regression")
        sel = FeatureSelector(config=cfg, column_meta=col_meta)
        sel.fit(X, y)
        result_arr = sel.transform(X)
        names = sel.get_feature_names_out()
        assert "c_fixed" in names.tolist(), f"fixed列が保持されていない: {names}"
        assert result_arr.shape[1] >= 1

    def test_fixed_column_preserved_in_names(self, sample_data):
        """T-003b: get_feature_names_out() に fixed 列が含まれる。"""
        from backend.pipeline.column_selector import ColumnMeta
        from backend.pipeline.feature_selector import FeatureSelector, FeatureSelectorConfig
        X, y = sample_data
        col_meta = {"c_fixed": ColumnMeta(fixed=True), "a": ColumnMeta(), "b": ColumnMeta()}
        cfg = FeatureSelectorConfig(method="rfr", task="regression")
        sel = FeatureSelector(config=cfg, column_meta=col_meta)
        sel.fit(X, y)
        names = sel.get_feature_names_out().tolist()
        assert "c_fixed" in names

    def test_none_method_returns_all_anyway(self, sample_data):
        """T-003c: method=none の場合は全列がそのまま返される。"""
        from backend.pipeline.column_selector import ColumnMeta
        from backend.pipeline.feature_selector import FeatureSelector, FeatureSelectorConfig
        X, y = sample_data
        col_meta = {"c_fixed": ColumnMeta(fixed=True), "a": ColumnMeta(), "b": ColumnMeta()}
        cfg = FeatureSelectorConfig(method="none", task="regression")
        sel = FeatureSelector(config=cfg, column_meta=col_meta)
        sel.fit(X, y)
        result = sel.transform(X)
        assert result.shape == X.shape

    def test_fixed_indices_initialized_before_fit(self, sample_data):
        """T-003d: fit前にも _fixed_indices が初期化されている。"""
        from backend.pipeline.column_selector import ColumnMeta
        from backend.pipeline.feature_selector import FeatureSelector, FeatureSelectorConfig
        X, y = sample_data
        cfg = FeatureSelectorConfig(method="none")
        sel = FeatureSelector(config=cfg)
        assert hasattr(sel, "_fixed_indices")
        assert isinstance(sel._fixed_indices, list)


# ============================================================
# T-004: AutoMLEngine の column_meta_dict 受け入れ
# ============================================================

class TestAutoMLEngineColumnMeta:
    """F-004: AutoMLEngine が column_meta_dict を受け入れて monotonic 適用する。"""

    def test_engine_accepts_column_meta_dict(self):
        """T-004a: column_meta_dict を渡してもエラーが起きない。"""
        from backend.models.automl import AutoMLEngine
        from backend.pipeline.column_selector import ColumnMeta
        col_meta = {"a": ColumnMeta(monotonic=1, fixed=False)}
        engine = AutoMLEngine(
            task="regression",
            cv_folds=2,
            model_keys=["ridge"],
            column_meta_dict=col_meta,
        )
        assert engine.column_meta_dict is not None
        assert "a" in engine.column_meta_dict

    def test_engine_accepts_dict_format_column_meta(self):
        """T-004b: dict 形式（state 保存形式）の column_meta_dict でも動く。"""
        from backend.models.automl import AutoMLEngine
        col_meta_raw = {
            "temperature": {"monotonic": 1, "linearity": "linear", "group": None,
                            "scale_hint": None, "description": "", "fixed": False},
        }
        engine = AutoMLEngine(
            task="regression",
            cv_folds=2,
            model_keys=["ridge"],
            column_meta_dict=col_meta_raw,
        )
        assert "temperature" in engine.column_meta_dict

    def test_engine_run_with_column_meta(self):
        """T-004c: column_meta_dict を渡した状態で engine.run() が完走する。"""
        from backend.models.automl import AutoMLEngine
        from backend.pipeline.column_selector import ColumnMeta
        np.random.seed(42)
        n = 40
        df = pd.DataFrame({
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "y": np.random.randn(n),
        })
        col_meta = {
            "x1": ColumnMeta(monotonic=1, linearity="linear"),
            "x2": ColumnMeta(monotonic=0, linearity="nonlinear"),
        }
        engine = AutoMLEngine(
            task="regression",
            cv_folds=2,
            model_keys=["ridge"],
            column_meta_dict=col_meta,
        )
        result = engine.run(df, target_col="y")
        assert result is not None
        assert result.best_model_key == "ridge"
