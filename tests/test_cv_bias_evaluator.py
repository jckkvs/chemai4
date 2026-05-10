# -*- coding: utf-8 -*-
"""
tests/test_cv_bias_evaluator.py

cv_bias_evaluator.py のユニットテスト。

カバー対象:
  - CVBiasResult: to_dict()
  - estimate_tibshirani_bias: 正常ケース、単一パラメータ、エッジケース
  - estimate_bbc_cv_bias: 正常ケース、単一構成、エッジケース
  - format_bias_report: 文字列フォーマット
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

from backend.models.cv_bias_evaluator import (
    CVBiasResult,
    estimate_tibshirani_bias,
    estimate_bbc_cv_bias,
    format_bias_report,
)


# ═══════════════════════════════════════════════════════════════════
# CVBiasResult テスト
# ═══════════════════════════════════════════════════════════════════

class TestCVBiasResult:

    def test_to_dict_basic(self):
        r = CVBiasResult(
            method="tibshirani",
            raw_score=0.85,
            bias_estimate=0.03,
            corrected_score=0.82,
        )
        d = r.to_dict()
        assert d["method"] == "tibshirani"
        assert d["raw_score"] == 0.85
        assert "ci_lower" not in d

    def test_to_dict_with_ci(self):
        r = CVBiasResult(
            method="bbc_cv",
            raw_score=0.85,
            bias_estimate=0.03,
            corrected_score=0.82,
            ci_lower=0.78,
            ci_upper=0.86,
            n_bootstrap=200,
        )
        d = r.to_dict()
        assert "ci_lower" in d
        assert d["n_bootstrap"] == 200


# ═══════════════════════════════════════════════════════════════════
# Tibshirani-Tibshirani法テスト
# ═══════════════════════════════════════════════════════════════════

class TestTibshiraniBias:

    def test_basic_higher_is_better(self):
        """複数パラメータでバイアスが正方向に推定されること"""
        np.random.seed(42)
        K, P = 5, 10
        # 各foldに異なるノイズを加えた誤差曲線
        base = np.linspace(0.6, 0.9, P)
        curves = np.tile(base, (K, 1)) + np.random.randn(K, P) * 0.05

        result = estimate_tibshirani_bias(
            curves, higher_is_better=True
        )
        assert result.method == "tibshirani"
        assert result.bias_estimate >= 0  # 楽観的バイアス = 正
        assert result.corrected_score <= result.raw_score

    def test_basic_lower_is_better(self):
        """RMSE等でバイアスが正方向に推定されること"""
        np.random.seed(0)
        K, P = 5, 8
        base = np.linspace(0.5, 0.1, P)  # 小さいほど良い
        curves = np.tile(base, (K, 1)) + np.random.randn(K, P) * 0.02

        result = estimate_tibshirani_bias(
            curves, higher_is_better=False
        )
        assert result.bias_estimate >= 0
        assert result.corrected_score >= result.raw_score  # 補正で悪化方向

    def test_single_param_zero_bias(self):
        """単一パラメータではバイアス=0"""
        curves = np.array([[0.8], [0.7], [0.9]])
        result = estimate_tibshirani_bias(curves)
        assert result.bias_estimate == 0.0
        assert result.corrected_score == result.raw_score

    def test_invalid_shape_raises(self):
        """1次元配列でValueError"""
        with pytest.raises(ValueError, match="2次元"):
            estimate_tibshirani_bias(np.array([0.5, 0.6, 0.7]))

    def test_param_values_in_details(self):
        """param_valuesが結果detailsに含まれること"""
        curves = np.random.randn(3, 4)
        result = estimate_tibshirani_bias(
            curves,
            param_values=[0.01, 0.1, 1.0, 10.0],
            higher_is_better=True,
        )
        assert "best_param_value" in result.details

    def test_bias_per_fold_length(self):
        """bias_per_foldがfold数と一致すること"""
        K, P = 4, 5
        curves = np.random.randn(K, P)
        result = estimate_tibshirani_bias(curves)
        assert len(result.details["bias_per_fold"]) == K


# ═══════════════════════════════════════════════════════════════════
# BBC-CV テスト
# ═══════════════════════════════════════════════════════════════════

class TestBBCCV:

    def _make_classification_data(self, n=200, n_configs=3, seed=42):
        """テスト用の分類OOS予測データを生成"""
        rng = np.random.RandomState(seed)
        y_true = rng.randint(0, 2, n)
        preds = {}
        for i in range(n_configs):
            # 各構成は異なる精度のモデルをシミュレート
            noise_rate = 0.1 + i * 0.05
            y_pred = y_true.copy()
            flip_idx = rng.choice(n, size=int(n * noise_rate), replace=False)
            y_pred[flip_idx] = 1 - y_pred[flip_idx]
            preds[f"config_{i}"] = y_pred
        return y_true, preds

    def test_basic_classification(self):
        """バイアスが正方向に推定されること"""
        y_true, preds = self._make_classification_data()
        result = estimate_bbc_cv_bias(
            preds, y_true,
            scoring_func=accuracy_score,
            n_bootstrap=100,
            higher_is_better=True,
        )
        assert result.method == "bbc_cv"
        # BBC-CVのバイアスは概ね正だが、Bootstrap変動で微小負値もありうる
        assert result.bias_estimate >= -0.05
        # 補正後スコアは概ね raw_score 以下（バイアス方向に補正）
        assert result.corrected_score <= result.raw_score + 0.05
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.n_bootstrap == 100

    def test_regression_mse(self):
        """回帰（MSE、lower is better）でも動作すること"""
        rng = np.random.RandomState(0)
        n = 100
        y_true = rng.randn(n)
        preds = {
            "good": y_true + rng.randn(n) * 0.1,
            "bad": y_true + rng.randn(n) * 0.5,
        }

        def neg_mse(y, yp):
            return -mean_squared_error(y, yp)

        result = estimate_bbc_cv_bias(
            preds, y_true,
            scoring_func=neg_mse,
            n_bootstrap=50,
            higher_is_better=True,
        )
        assert result.method == "bbc_cv"
        assert result.n_bootstrap == 50

    def test_single_config_zero_bias(self):
        """単一構成ではバイアス=0"""
        y = np.array([0, 1, 0, 1, 0])
        preds = {"only": np.array([0, 1, 0, 0, 0])}
        result = estimate_bbc_cv_bias(
            preds, y,
            scoring_func=accuracy_score,
            higher_is_better=True,
        )
        assert result.bias_estimate == 0.0
        assert result.n_bootstrap == 0

    def test_empty_predictions_raises(self):
        """空の予測辞書でValueError"""
        with pytest.raises(ValueError, match="空です"):
            estimate_bbc_cv_bias(
                {}, np.array([0, 1]),
                scoring_func=accuracy_score,
            )

    def test_mismatched_length_raises(self):
        """y_trueと予測長不一致でValueError"""
        with pytest.raises(ValueError, match="不一致"):
            estimate_bbc_cv_bias(
                {"a": np.array([0, 1, 0])},
                np.array([0, 1]),
                scoring_func=accuracy_score,
            )

    def test_details_contain_config_info(self):
        """detailsに構成情報が含まれること"""
        y_true, preds = self._make_classification_data(n=50, n_configs=2)
        result = estimate_bbc_cv_bias(
            preds, y_true,
            scoring_func=accuracy_score,
            n_bootstrap=30,
        )
        assert "best_config" in result.details
        assert "n_configs" in result.details
        assert "original_scores" in result.details

    def test_reproducibility_with_seed(self):
        """同じseedで再現性があること"""
        y_true, preds = self._make_classification_data()
        r1 = estimate_bbc_cv_bias(
            preds, y_true, scoring_func=accuracy_score,
            n_bootstrap=50, random_state=0,
        )
        r2 = estimate_bbc_cv_bias(
            preds, y_true, scoring_func=accuracy_score,
            n_bootstrap=50, random_state=0,
        )
        assert r1.bias_estimate == r2.bias_estimate
        assert r1.corrected_score == r2.corrected_score


# ═══════════════════════════════════════════════════════════════════
# format_bias_report テスト
# ═══════════════════════════════════════════════════════════════════

class TestFormatBiasReport:

    def test_tibshirani_format(self):
        r = CVBiasResult(
            method="tibshirani",
            raw_score=0.85,
            bias_estimate=0.03,
            corrected_score=0.82,
        )
        text = format_bias_report(r)
        assert "Tibshirani" in text
        assert "0.8500" in text
        assert "0.0300" in text

    def test_bbc_format_with_ci(self):
        r = CVBiasResult(
            method="bbc_cv",
            raw_score=0.90,
            bias_estimate=0.02,
            corrected_score=0.88,
            ci_lower=0.85,
            ci_upper=0.91,
        )
        text = format_bias_report(r)
        assert "BBC-CV" in text
        assert "信頼区間" in text
