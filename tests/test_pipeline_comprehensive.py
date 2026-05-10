"""
tests/test_pipeline_comprehensive.py

ColumnSelectorWrapper / PipelineConfig / build_pipeline / 
apply_monotonic_constraints / extract_group_array の包括テスト。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from backend.pipeline.column_selector import (
    ColumnMeta,
    ColumnSelectorWrapper,
)
from backend.pipeline.pipeline_builder import (
    PipelineConfig,
    build_pipeline,
    apply_monotonic_constraints,
    extract_group_array,
)


# ============================================================
# ColumnMeta テスト
# ============================================================

class TestColumnMeta:
    def test_defaults(self):
        meta = ColumnMeta()
        assert meta.monotonic == 0
        assert meta.linearity == "unknown"
        assert meta.group is None

    def test_custom(self):
        meta = ColumnMeta(monotonic=1, linearity="linear", group="alkyl")
        assert meta.monotonic == 1
        assert meta.group == "alkyl"


# ============================================================
# ColumnSelectorWrapper テスト
# ============================================================

class TestColumnSelectorWrapper:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        })

    def test_mode_all(self, sample_df):
        sel = ColumnSelectorWrapper(mode="all")
        sel.fit(sample_df)
        result = sel.transform(sample_df)
        assert list(result.columns) == ["a", "b", "c"]

    def test_mode_include(self, sample_df):
        sel = ColumnSelectorWrapper(mode="include", columns=["a", "c"])
        sel.fit(sample_df)
        result = sel.transform(sample_df)
        assert list(result.columns) == ["a", "c"]

    def test_mode_exclude(self, sample_df):
        sel = ColumnSelectorWrapper(mode="exclude", columns=["b"])
        sel.fit(sample_df)
        result = sel.transform(sample_df)
        assert list(result.columns) == ["a", "c"]

    def test_invalid_mode_raises(self, sample_df):
        sel = ColumnSelectorWrapper(mode="invalid")
        with pytest.raises(ValueError, match="未知の mode"):
            sel.fit(sample_df)

    def test_empty_selection_raises(self, sample_df):
        sel = ColumnSelectorWrapper(mode="include", columns=["nonexistent"])
        with pytest.raises(ValueError, match="選択された列が0件"):
            sel.fit(sample_df)

    def test_non_dataframe_raises(self):
        sel = ColumnSelectorWrapper(mode="all")
        with pytest.raises(TypeError):
            sel.fit(np.array([[1, 2], [3, 4]]))

    def test_get_feature_names_out(self, sample_df):
        sel = ColumnSelectorWrapper(mode="all")
        sel.fit(sample_df)
        names = sel.get_feature_names_out()
        assert list(names) == ["a", "b", "c"]

    def test_selected_columns_property(self, sample_df):
        sel = ColumnSelectorWrapper(mode="include", columns=["a", "b"])
        sel.fit(sample_df)
        assert sel.selected_columns == ["a", "b"]

    def test_clone_compatibility(self):
        meta = {"x": ColumnMeta(monotonic=1)}
        sel = ColumnSelectorWrapper(mode="include", columns=["x"], column_meta=meta)
        cloned = clone(sel)
        assert cloned.mode == "include"

    def test_col_range(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        sel = ColumnSelectorWrapper(mode="include", col_range=(1, 3))
        sel.fit(df)
        result = sel.transform(df)
        assert list(result.columns) == ["b", "c"]

    def test_get_column_meta_default(self):
        sel = ColumnSelectorWrapper(mode="all")
        meta = sel.get_column_meta("unknown_col")
        assert meta.monotonic == 0

    def test_get_monotonic_constraints(self, sample_df):
        meta = {
            "a": ColumnMeta(monotonic=1),
            "b": ColumnMeta(monotonic=-1),
            "c": ColumnMeta(monotonic=0),
        }
        sel = ColumnSelectorWrapper(mode="all", column_meta=meta)
        sel.fit(sample_df)
        constraints = sel.get_monotonic_constraints()
        assert constraints == (1, -1, 0)

    def test_get_groups_array(self, sample_df):
        meta = {
            "a": ColumnMeta(group="G1"),
            "b": ColumnMeta(group="G1"),
            "c": ColumnMeta(group="G2"),
        }
        sel = ColumnSelectorWrapper(mode="all", column_meta=meta)
        sel.fit(sample_df)
        groups = sel.get_groups_array()
        assert groups == ["G1", "G1", "G2"]

    def test_missing_column_in_transform(self, sample_df):
        sel = ColumnSelectorWrapper(mode="all")
        sel.fit(sample_df)
        # transformに一部欠損するDFを渡す
        df_partial = sample_df[["a", "b"]]
        result = sel.transform(df_partial)
        assert "c" not in result.columns


# ============================================================
# extract_group_array テスト
# ============================================================

class TestExtractGroupArray:
    def test_basic(self):
        meta = {
            "a": ColumnMeta(group="G1"),
            "b": ColumnMeta(group="G2"),
            "c": ColumnMeta(group="G1"),
        }
        result = extract_group_array(meta, ["a", "b", "c"])
        assert list(result) == [0, 1, 0]

    def test_all_none(self):
        meta = {"a": ColumnMeta(), "b": ColumnMeta()}
        result = extract_group_array(meta, ["a", "b"])
        assert result is None

    def test_mixed(self):
        meta = {"a": ColumnMeta(group="G1"), "b": ColumnMeta()}
        result = extract_group_array(meta, ["a", "b"])
        assert result[0] == 0
        assert result[1] == -1


# ============================================================
# PipelineConfig テスト
# ============================================================

class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.task == "regression"
        assert cfg.estimator_key == "rf"
        assert cfg.apply_monotonic is True

    def test_custom(self):
        cfg = PipelineConfig(task="classification", estimator_key="xgb")
        assert cfg.task == "classification"
        assert cfg.estimator_key == "xgb"


# ============================================================
# build_pipeline テスト
# ============================================================

class TestBuildPipeline:
    def test_minimal_pipeline(self):
        cfg = PipelineConfig(task="regression", estimator_key="ridge")
        pipe = build_pipeline(cfg)
        assert len(pipe.steps) == 5
        step_names = [name for name, _ in pipe.steps]
        assert "col_select" in step_names
        assert "preprocess" in step_names
        assert "estimator" in step_names

    def test_pipeline_with_monotonic(self):
        meta = {"x": ColumnMeta(monotonic=1)}
        cfg = PipelineConfig(
            task="regression",
            estimator_key="rf",
            column_meta=meta,
            apply_monotonic=True,
        )
        pipe = build_pipeline(cfg)
        assert pipe is not None

    def test_classification_pipeline(self):
        cfg = PipelineConfig(task="classification", estimator_key="logistic")
        pipe = build_pipeline(cfg)
        assert pipe is not None


# ============================================================
# apply_monotonic_constraints テスト
# ============================================================

class TestApplyMonotonicConstraints:
    def test_no_constraints(self):
        """全制約が0ならスキップ"""
        from sklearn.ensemble import RandomForestRegressor
        est = RandomForestRegressor(n_estimators=10)
        meta = {"a": ColumnMeta(monotonic=0)}
        result = apply_monotonic_constraints(est, meta)
        assert result is est  # 変更なし

    def test_with_hist_gb(self):
        """HistGradientBoostingにネイティブ制約を適用"""
        from sklearn.ensemble import HistGradientBoostingRegressor
        est = HistGradientBoostingRegressor()
        meta = {
            "a": ColumnMeta(monotonic=1),
            "b": ColumnMeta(monotonic=-1),
        }
        result = apply_monotonic_constraints(est, meta, feature_names=["a", "b"])
        params = result.get_params()
        assert params.get("monotonic_cst") is not None
