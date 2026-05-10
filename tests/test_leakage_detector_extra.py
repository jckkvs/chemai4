"""
tests/test_leakage_detector_extra.py

leakage_detector.py のカバレッジ改善テスト。
compute_hat_matrix, compute_rbf_gram, compute_rf_proximity,
estimate_groups, detect_leakage, check_feature_leakage を網羅。
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backend.data.leakage_detector import (
    LeakagePair,
    LeakageReport,
    FeatureLeakageWarning,
    FeatureLeakageReport,
    compute_hat_matrix,
    compute_rbf_gram,
    compute_rf_proximity,
    estimate_groups,
    detect_leakage,
    check_feature_leakage,
    _find_suspicious_pairs,
    _compute_group_consistency_score,
)


# ============================================================
# テストデータ
# ============================================================

def _make_data(n: int = 30, d: int = 3):
    rng = np.random.RandomState(42)
    return rng.randn(n, d)


# ============================================================
# 類似度行列 計算
# ============================================================

class TestComputeHatMatrix:
    def test_basic(self):
        X = _make_data()
        H = compute_hat_matrix(X)
        assert H.shape == (30, 30)
        # Hat matrix: 対角要素は 0 ≤ h_ii ≤ 1
        diag = np.diag(H)
        assert np.all(diag >= -0.01)
        assert np.all(diag <= 1.01)

    def test_rank_deficient(self):
        """n <= p の場合はリッジ正則化が適用される"""
        X = np.random.randn(3, 10)  # n < p
        H = compute_hat_matrix(X)
        assert H.shape == (3, 3)


class TestComputeRBFGram:
    def test_basic(self):
        X = _make_data()
        K = compute_rbf_gram(X)
        assert K.shape == (30, 30)
        # RBFグラム行列: 対角要素は1
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-6)
        # 全要素 0 ≤ K_ij ≤ 1
        assert np.all(K >= -1e-6)
        assert np.all(K <= 1.0 + 1e-6)

    def test_custom_gamma(self):
        X = _make_data()
        K = compute_rbf_gram(X, gamma=0.1)
        assert K.shape == (30, 30)


class TestComputeRFProximity:
    def test_with_y(self):
        X = _make_data()
        y = X[:, 0] + np.random.randn(30) * 0.1
        P = compute_rf_proximity(X, y, n_estimators=10)
        assert P.shape == (30, 30)
        np.testing.assert_allclose(np.diag(P), 1.0)

    def test_without_y(self):
        """Unsupervised RF"""
        X = _make_data()
        P = compute_rf_proximity(X, None, n_estimators=10)
        assert P.shape == (30, 30)


# ============================================================
# ヘルパー関数
# ============================================================

class TestFindSuspiciousPairs:
    def test_basic(self):
        S = np.eye(5)
        S[0, 1] = S[1, 0] = 0.99
        pairs = _find_suspicious_pairs(S, threshold=0.95, method="test")
        assert len(pairs) == 1
        assert pairs[0].idx_a == 0
        assert pairs[0].idx_b == 1

    def test_no_pairs(self):
        S = np.eye(5) * 0.5
        pairs = _find_suspicious_pairs(S, threshold=0.95, method="test")
        assert len(pairs) == 0


class TestGroupConsistency:
    def test_basic(self):
        S = np.eye(10) + np.random.randn(10, 10) * 0.1
        S = (S + S.T) / 2
        score = _compute_group_consistency_score(S, top_k=3)
        assert 0 <= score <= 1

    def test_small_n(self):
        S = np.eye(3)
        score = _compute_group_consistency_score(S, top_k=5)
        assert 0 <= score <= 1


# ============================================================
# グループ推定
# ============================================================

class TestEstimateGroups:
    def test_basic(self):
        rng = np.random.RandomState(42)
        # 2グループの明確なデータ
        S = np.zeros((10, 10))
        S[:5, :5] = 0.9
        S[5:, 5:] = 0.9
        np.fill_diagonal(S, 1.0)
        labels, n_groups = estimate_groups(S)
        assert n_groups >= 2
        assert len(labels) == 10

    def test_tiny_data(self):
        S = np.eye(2)
        labels, n_groups = estimate_groups(S)
        assert len(labels) == 2


# ============================================================
# detect_leakage (メインAPI)
# ============================================================

class TestDetectLeakage:
    def test_hat_method(self):
        X = _make_data(n=20, d=3)
        report = detect_leakage(X, method="hat")
        assert isinstance(report, LeakageReport)
        assert report.risk_level in ("low", "medium", "high")

    def test_rbf_method(self):
        X = _make_data(n=20, d=3)
        report = detect_leakage(X, method="rbf")
        assert isinstance(report, LeakageReport)

    def test_rf_method(self):
        X = _make_data(n=30, d=3)
        y = X[:, 0] + np.random.randn(30) * 0.1
        report = detect_leakage(X, y=y, method="rf", rf_n_estimators=10)
        assert isinstance(report, LeakageReport)

    def test_auto_method(self):
        X = _make_data(n=20, d=3)
        report = detect_leakage(X, method="auto")
        assert isinstance(report, LeakageReport)

    def test_with_dataframe(self):
        df = pd.DataFrame(_make_data(n=20, d=3), columns=["a", "b", "c"])
        report = detect_leakage(df, method="hat")
        assert isinstance(report, LeakageReport)

    def test_with_nan(self):
        X = _make_data(n=20, d=3).astype(float)
        X[0, 0] = np.nan
        X[5, 1] = np.nan
        report = detect_leakage(X, method="rbf")
        assert isinstance(report, LeakageReport)

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="未知"):
            detect_leakage(_make_data(), method="unknown")


# ============================================================
# check_feature_leakage
# ============================================================

class TestCheckFeatureLeakage:
    def test_no_leakage(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "x1": rng.randn(50),
            "x2": rng.randn(50),
            "y": rng.randn(50),
        })
        report = check_feature_leakage(df, "y")
        assert isinstance(report, FeatureLeakageReport)

    def test_high_correlation_leakage(self):
        rng = np.random.RandomState(42)
        y = rng.randn(50)
        df = pd.DataFrame({
            "x_clean": rng.randn(50),
            "x_leaky": y + rng.randn(50) * 0.001,  # Almost perfect correlation
            "y": y,
        })
        report = check_feature_leakage(df, "y")
        assert report.has_risk
        leaky = [w for w in report.warnings if w.feature == "x_leaky"]
        assert len(leaky) > 0

    def test_name_similarity(self):
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "target_shifted": [2, 3, 4],
            "target": [1.1, 2.2, 3.3],
        })
        report = check_feature_leakage(df, "target")
        # 名前類似は実装に依存。leakage警告の有無のみ確認
        assert isinstance(report, FeatureLeakageReport)

    def test_missing_target(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        report = check_feature_leakage(df, "nonexistent_target")
        assert not report.has_risk

    def test_classification_separation(self):
        rng = np.random.RandomState(42)
        n = 50
        labels = np.array([0] * 25 + [1] * 25)
        # Perfect separator
        x_sep = np.concatenate([rng.randn(25) - 100, rng.randn(25) + 100])
        df = pd.DataFrame({
            "x_normal": rng.randn(n),
            "x_separator": x_sep,
            "label": labels,
        })
        report = check_feature_leakage(df, "label")
        assert report.has_risk
