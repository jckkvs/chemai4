"""
tests/test_cv_bias_evaluator_extra.py

cv_bias_evaluator.py のカバレッジ改善テスト。
estimate_tibshirani_bias, estimate_bbc_cv_bias,
CVBiasResult, format_bias_report を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

from backend.models.cv_bias_evaluator import (
    CVBiasResult,
    estimate_tibshirani_bias,
    estimate_bbc_cv_bias,
    format_bias_report,
)


# ============================================================
# CVBiasResult
# ============================================================

class TestCVBiasResult:
    def test_to_dict(self):
        r = CVBiasResult(
            method="tibshirani",
            raw_score=0.85,
            bias_estimate=0.02,
            corrected_score=0.83,
        )
        d = r.to_dict()
        assert d["method"] == "tibshirani"
        assert d["raw_score"] == 0.85
        assert d["corrected_score"] == 0.83

    def test_to_dict_with_ci(self):
        r = CVBiasResult(
            method="bbc_cv",
            raw_score=0.85,
            bias_estimate=0.02,
            corrected_score=0.83,
            ci_lower=0.80,
            ci_upper=0.86,
            n_bootstrap=200,
        )
        d = r.to_dict()
        assert "ci_lower" in d
        assert "ci_upper" in d
        assert d["n_bootstrap"] == 200


# ============================================================
# Tibshirani-Tibshirani法
# ============================================================

class TestTibshiraniBias:
    def test_basic_higher_is_better(self):
        # 5 folds × 3 params
        rng = np.random.RandomState(42)
        curves = 0.8 + rng.randn(5, 3) * 0.05
        result = estimate_tibshirani_bias(curves, higher_is_better=True)
        assert result.method == "tibshirani"
        assert result.bias_estimate >= 0
        assert result.corrected_score <= result.raw_score

    def test_basic_lower_is_better(self):
        rng = np.random.RandomState(42)
        curves = 1.0 + rng.randn(5, 3) * 0.1
        result = estimate_tibshirani_bias(curves, higher_is_better=False)
        assert result.method == "tibshirani"

    def test_with_param_values(self):
        rng = np.random.RandomState(42)
        curves = rng.randn(5, 4)
        params = [0.01, 0.1, 1.0, 10.0]
        result = estimate_tibshirani_bias(curves, param_values=params)
        assert "best_param_value" in result.details

    def test_single_param(self):
        curves = np.array([[0.8], [0.82], [0.79], [0.81], [0.83]])
        result = estimate_tibshirani_bias(curves)
        assert result.bias_estimate == 0.0

    def test_invalid_shape(self):
        with pytest.raises(ValueError, match="2次元"):
            estimate_tibshirani_bias(np.array([1, 2, 3]))


# ============================================================
# BBC-CV
# ============================================================

class TestBBCCVBias:
    def test_basic(self):
        rng = np.random.RandomState(42)
        n = 50
        y_true = rng.randn(n)
        preds = {
            "model_A": y_true + rng.randn(n) * 0.1,
            "model_B": y_true + rng.randn(n) * 0.2,
        }
        result = estimate_bbc_cv_bias(
            preds, y_true,
            scoring_func=lambda yt, yp: -float(mean_squared_error(yt, yp)),
            n_bootstrap=50,
            higher_is_better=True,
        )
        assert result.method == "bbc_cv"
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.n_bootstrap == 50

    def test_single_config(self):
        rng = np.random.RandomState(42)
        n = 30
        y_true = rng.randn(n)
        preds = {"only_model": y_true + rng.randn(n) * 0.1}
        result = estimate_bbc_cv_bias(
            preds, y_true,
            scoring_func=lambda yt, yp: -float(mean_squared_error(yt, yp)),
        )
        assert result.bias_estimate == 0.0
        assert result.n_bootstrap == 0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="空"):
            estimate_bbc_cv_bias({}, np.array([1, 2, 3]),
                                 scoring_func=lambda y, p: 0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="不一致"):
            estimate_bbc_cv_bias(
                {"m": np.array([1, 2])},
                np.array([1, 2, 3]),
                scoring_func=lambda y, p: 0.0,
            )

    def test_lower_is_better(self):
        rng = np.random.RandomState(42)
        n = 50
        y_true = rng.randn(n)
        preds = {
            "A": y_true + rng.randn(n) * 0.1,
            "B": y_true + rng.randn(n) * 0.3,
        }
        result = estimate_bbc_cv_bias(
            preds, y_true,
            scoring_func=lambda yt, yp: float(mean_squared_error(yt, yp)),
            n_bootstrap=30,
            higher_is_better=False,
        )
        assert result.method == "bbc_cv"


# ============================================================
# format_bias_report
# ============================================================

class TestFormatBiasReport:
    def test_tibshirani(self):
        r = CVBiasResult(
            method="tibshirani",
            raw_score=0.85,
            bias_estimate=0.02,
            corrected_score=0.83,
        )
        report = format_bias_report(r)
        assert "Tibshirani" in report
        assert "0.85" in report
        assert "0.83" in report

    def test_bbc_cv_with_ci(self):
        r = CVBiasResult(
            method="bbc_cv",
            raw_score=0.9,
            bias_estimate=0.03,
            corrected_score=0.87,
            ci_lower=0.84,
            ci_upper=0.90,
        )
        report = format_bias_report(r)
        assert "BBC-CV" in report
        assert "0.84" in report
        assert "0.90" in report
