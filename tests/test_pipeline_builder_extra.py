"""
tests/test_pipeline_builder_extra.py

pipeline_builder.py のカバレッジ改善テスト。
build_pipeline, apply_monotonic_constraints, extract_group_array,
PipelineConfig を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.pipeline.pipeline_builder import (
    PipelineConfig,
    build_pipeline,
    apply_monotonic_constraints,
    extract_group_array,
)
from backend.pipeline.column_selector import ColumnMeta


# ============================================================
# PipelineConfig
# ============================================================

class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.task == "regression"
        assert cfg.estimator_key == "rf"
        assert cfg.apply_monotonic is True
        assert cfg.col_select_mode == "all"

    def test_custom(self):
        cfg = PipelineConfig(
            task="classification",
            estimator_key="logistic",
            estimator_params={"C": 10.0},
        )
        assert cfg.task == "classification"


# ============================================================
# build_pipeline
# ============================================================

class TestBuildPipeline:
    def test_basic_regression_pipeline(self):
        cfg = PipelineConfig(task="regression", estimator_key="ridge")
        pipe = build_pipeline(cfg)
        assert len(pipe.steps) == 5
        step_names = [s[0] for s in pipe.steps]
        assert "col_select" in step_names
        assert "preprocess" in step_names
        assert "estimator" in step_names

    def test_classification_pipeline(self):
        cfg = PipelineConfig(task="classification", estimator_key="logistic")
        pipe = build_pipeline(cfg)
        assert len(pipe.steps) == 5

    def test_with_estimator_params(self):
        cfg = PipelineConfig(
            task="regression",
            estimator_key="ridge",
            estimator_params={"alpha": 10.0},
        )
        pipe = build_pipeline(cfg)
        est = pipe.named_steps["estimator"]
        assert est.alpha == 10.0


# ============================================================
# apply_monotonic_constraints
# ============================================================

class TestApplyMonotonicConstraints:
    def test_no_constraints(self):
        from sklearn.ensemble import GradientBoostingRegressor
        est = GradientBoostingRegressor(n_estimators=10)
        meta = {"a": ColumnMeta(monotonic=0), "b": ColumnMeta(monotonic=0)}
        result = apply_monotonic_constraints(est, meta)
        # No constraints → should return unmodified estimator
        assert result is est

    def test_with_constraints_on_non_monotonic_model(self):
        from sklearn.linear_model import Ridge
        est = Ridge()
        meta = {"a": ColumnMeta(monotonic=1), "b": ColumnMeta(monotonic=-1)}
        result = apply_monotonic_constraints(est, meta)
        # Ridge doesn't support monotonic → should return as-is or wrapped
        assert result is not None


# ============================================================
# extract_group_array
# ============================================================

class TestExtractGroupArray:
    def test_with_groups(self):
        meta = {
            "a": ColumnMeta(group="g1"),
            "b": ColumnMeta(group="g1"),
            "c": ColumnMeta(group="g2"),
        }
        arr = extract_group_array(meta, ["a", "b", "c"])
        assert arr is not None
        assert len(arr) == 3
        assert arr[0] == arr[1]  # Same group
        assert arr[0] != arr[2]  # Different group

    def test_no_groups(self):
        meta = {
            "a": ColumnMeta(),
            "b": ColumnMeta(),
        }
        arr = extract_group_array(meta, ["a", "b"])
        assert arr is None

    def test_partial_groups(self):
        meta = {
            "a": ColumnMeta(group="g1"),
            "b": ColumnMeta(),  # No group
        }
        arr = extract_group_array(meta, ["a", "b"])
        assert arr is not None
        assert arr[1] == -1  # No group → -1
