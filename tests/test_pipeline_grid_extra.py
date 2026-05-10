"""
tests/test_pipeline_grid_extra.py

pipeline_grid.py のカバレッジ改善テスト。
PipelineGridConfig, generate_pipeline_grid, count_combinations を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np

from backend.pipeline.pipeline_grid import (
    PipelineGridConfig,
    PipelineCombination,
    generate_pipeline_grid,
    count_combinations,
    _ensure_nonempty,
    _make_label,
    _build_gen_combinations,
)


# ============================================================
# PipelineGridConfig
# ============================================================

class TestPipelineGridConfig:
    def test_defaults(self):
        cfg = PipelineGridConfig()
        assert cfg.task == "regression"
        assert cfg.estimator_keys == ["rf"]
        assert cfg.numeric_imputers == ["mean"]

    def test_custom(self):
        cfg = PipelineGridConfig(
            task="classification",
            estimator_keys=["logistic", "rf"],
            numeric_scalers=["standard", "robust"],
        )
        assert cfg.task == "classification"
        assert len(cfg.estimator_keys) == 2


# ============================================================
# ヘルパー関数
# ============================================================

class TestHelpers:
    def test_ensure_nonempty_with_list(self):
        assert _ensure_nonempty(["a", "b"], "x") == ["a", "b"]

    def test_ensure_nonempty_empty(self):
        assert _ensure_nonempty([], "x") == ["x"]

    def test_make_label(self):
        label = _make_label("mean", "standard", "most_frequent",
                            "onehot", "ordinal", "most_frequent",
                            "none", 2, "none", "rf")
        assert "est=rf" in label
        assert "imp=mean" in label

    def test_build_gen_combinations_none(self):
        cfg = PipelineGridConfig(feature_gen_methods=["none"])
        combos = _build_gen_combinations(cfg)
        assert len(combos) == 1
        assert combos[0] == ("none", 2)

    def test_build_gen_combinations_poly(self):
        cfg = PipelineGridConfig(
            feature_gen_methods=["polynomial"],
            feature_gen_degrees=[2, 3],
        )
        combos = _build_gen_combinations(cfg)
        assert len(combos) == 2


# ============================================================
# count_combinations
# ============================================================

class TestCountCombinations:
    def test_default_count(self):
        cfg = PipelineGridConfig()
        count = count_combinations(cfg)
        assert count >= 1

    def test_multi_estimator_count(self):
        cfg = PipelineGridConfig(estimator_keys=["ridge", "rf"])
        count = count_combinations(cfg)
        assert count >= 2

    def test_multi_scaler_count(self):
        cfg = PipelineGridConfig(numeric_scalers=["standard", "robust"])
        count = count_combinations(cfg)
        # Should be at least 2x the base
        cfg2 = PipelineGridConfig(numeric_scalers=["standard"])
        count2 = count_combinations(cfg2)
        assert count == 2 * count2


# ============================================================
# generate_pipeline_grid
# ============================================================

class TestGeneratePipelineGrid:
    def test_basic(self):
        cfg = PipelineGridConfig(estimator_keys=["ridge"])
        combos = generate_pipeline_grid(cfg)
        assert len(combos) >= 1
        assert isinstance(combos[0], PipelineCombination)
        assert combos[0].pipeline is not None

    def test_multiple_estimators(self):
        cfg = PipelineGridConfig(estimator_keys=["ridge", "rf"])
        combos = generate_pipeline_grid(cfg)
        est_keys = {c.config.estimator_key for c in combos}
        assert "ridge" in est_keys
        assert "rf" in est_keys

    def test_max_combinations(self):
        cfg = PipelineGridConfig(
            estimator_keys=["ridge", "rf", "svr"],
            numeric_scalers=["standard", "robust"],
        )
        combos = generate_pipeline_grid(cfg, max_combinations=2)
        assert len(combos) <= 2

    def test_empty_estimators_error(self):
        cfg = PipelineGridConfig(estimator_keys=[])
        with pytest.raises(ValueError, match="estimator_keys"):
            generate_pipeline_grid(cfg)

    def test_with_estimator_params(self):
        cfg = PipelineGridConfig(
            estimator_keys=["ridge"],
            estimator_params_list=[{"alpha": 10.0}],
        )
        combos = generate_pipeline_grid(cfg)
        assert len(combos) >= 1
        est = combos[0].pipeline.named_steps["estimator"]
        assert est.alpha == 10.0
