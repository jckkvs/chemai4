"""
tests/test_pipeline_grid_comprehensive.py

pipeline_grid.py の包括テスト。
PipelineGridConfig / generate_pipeline_grid / count_combinations /
_build_gen_combinations / _ensure_nonempty / _make_label を網羅。
"""
from __future__ import annotations

import pytest

from backend.pipeline.pipeline_grid import (
    PipelineGridConfig,
    PipelineCombination,
    generate_pipeline_grid,
    count_combinations,
    _build_gen_combinations,
    _ensure_nonempty,
    _make_label,
)


class TestPipelineGridConfig:
    def test_defaults(self):
        cfg = PipelineGridConfig()
        assert cfg.task == "regression"
        assert cfg.estimator_keys == ["rf"]

    def test_custom(self):
        cfg = PipelineGridConfig(
            task="classification",
            estimator_keys=["logistic", "rf_c"],
        )
        assert cfg.task == "classification"
        assert len(cfg.estimator_keys) == 2


class TestCountCombinations:
    def test_single(self):
        cfg = PipelineGridConfig()
        assert count_combinations(cfg) == 1

    def test_multiple(self):
        cfg = PipelineGridConfig(
            numeric_scalers=["standard", "robust"],
            estimator_keys=["ridge", "rf"],
        )
        assert count_combinations(cfg) == 4


class TestEnsureNonempty:
    def test_nonempty(self):
        assert _ensure_nonempty(["a", "b"], "x") == ["a", "b"]

    def test_empty(self):
        assert _ensure_nonempty([], "x") == ["x"]


class TestMakeLabel:
    def test_basic(self):
        label = _make_label("mean", "standard", "most_frequent", "onehot", "ordinal",
                            "most_frequent", "none", 2, "none", "ridge")
        assert "ridge" in label
        assert "mean" in label


class TestBuildGenCombinations:
    def test_none_only(self):
        cfg = PipelineGridConfig(feature_gen_methods=["none"])
        combos = _build_gen_combinations(cfg)
        assert len(combos) == 1
        assert combos[0] == ("none", 2)

    def test_polynomial(self):
        cfg = PipelineGridConfig(
            feature_gen_methods=["polynomial"],
            feature_gen_degrees=[2, 3],
        )
        combos = _build_gen_combinations(cfg)
        assert len(combos) == 2


class TestGeneratePipelineGrid:
    def test_single(self):
        cfg = PipelineGridConfig(estimator_keys=["ridge"])
        results = generate_pipeline_grid(cfg)
        assert len(results) == 1
        assert isinstance(results[0], PipelineCombination)
        assert results[0].pipeline is not None

    def test_multiple_estimators(self):
        cfg = PipelineGridConfig(estimator_keys=["ridge", "rf"])
        results = generate_pipeline_grid(cfg)
        assert len(results) == 2

    def test_max_combinations(self):
        cfg = PipelineGridConfig(
            numeric_scalers=["standard", "robust"],
            estimator_keys=["ridge", "rf", "knn"],
        )
        results = generate_pipeline_grid(cfg, max_combinations=3)
        assert len(results) <= 3

    def test_empty_estimators(self):
        cfg = PipelineGridConfig(estimator_keys=[])
        with pytest.raises(ValueError, match="estimator_keys"):
            generate_pipeline_grid(cfg)

    def test_with_feature_gen(self):
        cfg = PipelineGridConfig(
            feature_gen_methods=["polynomial"],
            feature_gen_degrees=[2],
            estimator_keys=["ridge"],
        )
        results = generate_pipeline_grid(cfg)
        assert len(results) == 1
