"""
tests/test_config_extra.py

config.py のカバレッジ改善テスト。
AppConfig, グローバル定数値, Path設定 を網羅。
"""
from __future__ import annotations

import pytest
from pathlib import Path

from backend.utils.config import (
    PROJECT_ROOT,
    DATA_DIR,
    RANDOM_STATE,
    TYPE_DETECTOR_CARDINALITY_THRESHOLD,
    TYPE_DETECTOR_SKEWNESS_THRESHOLD,
    TYPE_DETECTOR_OUTLIER_IQR_FACTOR,
    AUTOML_CV_FOLDS,
    AUTOML_TIMEOUT_SECONDS,
    SHAP_MAX_DISPLAY,
    SHAP_KERNEL_NSAMPLES,
    AppConfig,
    default_config,
)


class TestConstants:
    def test_random_state(self):
        assert RANDOM_STATE == 42

    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()

    def test_automl_defaults(self):
        assert AUTOML_CV_FOLDS >= 2
        assert AUTOML_TIMEOUT_SECONDS > 0

    def test_shap_defaults(self):
        assert SHAP_MAX_DISPLAY > 0
        assert SHAP_KERNEL_NSAMPLES > 0

    def test_type_detector_defaults(self):
        assert TYPE_DETECTOR_CARDINALITY_THRESHOLD > 0
        assert TYPE_DETECTOR_SKEWNESS_THRESHOLD > 0
        assert TYPE_DETECTOR_OUTLIER_IQR_FACTOR > 0


class TestAppConfig:
    def test_defaults(self):
        cfg = AppConfig()
        assert cfg.random_state == 42
        assert cfg.automl_cv_folds == AUTOML_CV_FOLDS
        assert cfg.n_jobs != 0
        assert isinstance(cfg.extra, dict)

    def test_custom(self):
        cfg = AppConfig(random_state=0, automl_cv_folds=10)
        assert cfg.random_state == 0
        assert cfg.automl_cv_folds == 10

    def test_default_config_instance(self):
        assert isinstance(default_config, AppConfig)
        assert default_config.random_state == 42

    def test_extra_field(self):
        cfg = AppConfig(extra={"key": "value"})
        assert cfg.extra["key"] == "value"
