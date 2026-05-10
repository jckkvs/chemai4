# -*- coding: utf-8 -*-
"""
tests/test_leakage_detector.py

リーケージ検出モジュールのユニットテスト。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.data.leakage_detector import (
    compute_hat_matrix,
    compute_rbf_gram,
    compute_rf_proximity,
    estimate_groups,
    detect_leakage,
    _compute_group_consistency_score,
    _find_suspicious_pairs,
    LeakageReport,
)


# ─── ヘルパー ──────────────────────────────────────────────────

def _make_grouped_data(n_per_group: int = 10, n_groups: int = 3, noise: float = 0.01):
    """意図的にグループ構造を持つデータを生成。"""
    rng = np.random.RandomState(42)
    X_list, y_list, groups = [], [], []
    for g in range(n_groups):
        center = rng.randn(5) * 10  # 各グループの中心
        X_group = center + rng.randn(n_per_group, 5) * noise
        y_group = center.sum() + rng.randn(n_per_group) * noise
        X_list.append(X_group)
        y_list.append(y_group)
        groups.extend([g] * n_per_group)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y, np.array(groups)


def _make_independent_data(n: int = 50, p: int = 5):
    """グループ構造のない独立データを生成。"""
    rng = np.random.RandomState(42)
    X = rng.randn(n, p)
    y = X @ rng.randn(p) + rng.randn(n) * 0.1
    return X, y


# ─── ハット行列テスト ─────────────────────────────────────────

class TestHatMatrix:
    """F-ld-001: ハット行列の数学的正確性"""

    def test_symmetry(self):
        """H は対称行列"""
        X, _ = _make_independent_data(20, 3)
        H = compute_hat_matrix(X)
        np.testing.assert_allclose(H, H.T, atol=1e-10)

    def test_idempotent(self):
        """H² = H（冪等性）"""
        X, _ = _make_independent_data(20, 3)
        H = compute_hat_matrix(X)
        H2 = H @ H
        np.testing.assert_allclose(H, H2, atol=1e-8)

    def test_trace_equals_p(self):
        """trace(H) = p（特徴量数）"""
        X, _ = _make_independent_data(30, 5)
        H = compute_hat_matrix(X)
        np.testing.assert_allclose(np.trace(H), 5.0, atol=1e-8)

    def test_diagonal_between_0_and_1(self):
        """対角要素は [0, 1] の範囲"""
        X, _ = _make_independent_data(30, 5)
        H = compute_hat_matrix(X)
        diag = np.diag(H)
        assert np.all(diag >= -1e-10)
        assert np.all(diag <= 1.0 + 1e-10)

    def test_singular_case(self):
        """p >= n の場合もエラーなく計算可能（リッジ正則化）"""
        X, _ = _make_independent_data(5, 10)
        H = compute_hat_matrix(X)
        assert H.shape == (5, 5)


# ─── RBFグラム行列テスト ─────────────────────────────────────

class TestRBFGram:
    """F-ld-002: RBFグラム行列の正確性"""

    def test_symmetry(self):
        """K は対称行列"""
        X, _ = _make_independent_data(20, 3)
        K = compute_rbf_gram(X)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_diagonal_is_one(self):
        """対角要素は 1.0"""
        X, _ = _make_independent_data(20, 3)
        K = compute_rbf_gram(X)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_values_between_0_and_1(self):
        """全要素が [0, 1] の範囲"""
        X, _ = _make_independent_data(20, 3)
        K = compute_rbf_gram(X)
        assert np.all(K >= -1e-10)
        assert np.all(K <= 1.0 + 1e-10)

    def test_custom_gamma(self):
        """カスタム gamma を指定"""
        X, _ = _make_independent_data(20, 3)
        K = compute_rbf_gram(X, gamma=0.1)
        assert K.shape == (20, 20)

    def test_identical_samples_have_high_similarity(self):
        """同一サンプルの複製は類似度 1.0"""
        X = np.array([[1.0, 2.0, 3.0]] * 5)
        K = compute_rbf_gram(X)
        np.testing.assert_allclose(K, 1.0, atol=1e-10)


# ─── RF Proximity テスト ──────────────────────────────────────

class TestRFProximity:
    """F-ld-003: RF Proximity の正確性"""

    def test_symmetry(self):
        """P は対称行列"""
        X, y = _make_independent_data(30, 3)
        P = compute_rf_proximity(X, y, n_estimators=50)
        np.testing.assert_allclose(P, P.T, atol=1e-6)

    def test_diagonal_is_one(self):
        """対角要素は 1.0"""
        X, y = _make_independent_data(30, 3)
        P = compute_rf_proximity(X, y, n_estimators=50)
        np.testing.assert_allclose(np.diag(P), 1.0, atol=1e-6)

    def test_values_between_0_and_1(self):
        """全要素が [0, 1] の範囲"""
        X, y = _make_independent_data(30, 3)
        P = compute_rf_proximity(X, y, n_estimators=50)
        assert np.all(P >= -1e-6)
        assert np.all(P <= 1.0 + 1e-6)

    def test_unsupervised_mode(self):
        """y=None で Unsupervised RF が動作"""
        X, _ = _make_independent_data(30, 3)
        P = compute_rf_proximity(X, y=None, n_estimators=50)
        assert P.shape == (30, 30)

    def test_grouped_data_high_within_group(self):
        """グループデータでは同グループ内の類似度が高い"""
        X, y, groups = _make_grouped_data(10, 3, noise=0.001)
        P = compute_rf_proximity(X, y, n_estimators=100)
        # グループ0内の平均類似度
        in_group_0 = P[:10, :10]
        # グループ0-1間の平均類似度
        between_0_1 = P[:10, 10:20]
        assert np.mean(in_group_0) > np.mean(between_0_1)


# ─── グループ一貫性スコアテスト ──────────────────────────────

class TestGroupConsistencyScore:
    """F-ld-004: グループ一貫性スコアの正確性"""

    def test_perfect_groups(self):
        """完全なグループ構造ではスコアが高い"""
        # 3グループ × 5サンプル、ブロック対角の類似度行列
        S = np.zeros((15, 15))
        for i in range(3):
            S[i*5:(i+1)*5, i*5:(i+1)*5] = 1.0
        score = _compute_group_consistency_score(S, top_k=4)
        assert score >= 0.8

    def test_no_groups(self):
        """ランダムデータではスコアが低い"""
        rng = np.random.RandomState(42)
        S = rng.rand(20, 20)
        S = (S + S.T) / 2
        np.fill_diagonal(S, 1.0)
        score = _compute_group_consistency_score(S, top_k=3)
        # ランダムでも一定の相互性があるが、完全グループよりは低い
        assert score < 0.8

    def test_score_range(self):
        """スコアは [0, 1] の範囲"""
        rng = np.random.RandomState(42)
        S = rng.rand(10, 10)
        S = (S + S.T) / 2
        np.fill_diagonal(S, 1.0)
        score = _compute_group_consistency_score(S)
        assert 0.0 <= score <= 1.0


# ─── グループ推定テスト ──────────────────────────────────────

class TestEstimateGroups:
    """F-ld-005: グループ推定のテスト"""

    def test_clear_groups_detected(self):
        """明確なグループ構造を正しく検出"""
        X, _, true_groups = _make_grouped_data(10, 3, noise=0.001)
        from sklearn.preprocessing import StandardScaler
        X_s = StandardScaler().fit_transform(X)
        S = compute_rbf_gram(X_s)
        labels, n_groups = estimate_groups(S)
        assert n_groups >= 2  # 少なくとも2グループは検出

    def test_returns_correct_shape(self):
        """返却ラベルのshapeがサンプル数と一致"""
        X, _, _ = _make_grouped_data(10, 3, noise=0.01)
        from sklearn.preprocessing import StandardScaler
        X_s = StandardScaler().fit_transform(X)
        S = compute_rbf_gram(X_s)
        labels, n_groups = estimate_groups(S)
        assert labels.shape[0] == X.shape[0]

    def test_too_few_samples(self):
        """サンプル数が少なすぎる場合"""
        S = np.ones((2, 2))
        labels, n_groups = estimate_groups(S)
        assert n_groups <= 1


# ─── detect_leakage メインAPIテスト ──────────────────────────

class TestDetectLeakage:
    """F-ld-006: detect_leakage メインAPIのテスト"""

    def test_returns_leakage_report(self):
        """LeakageReport を返す"""
        X, y = _make_independent_data(30, 5)
        report = detect_leakage(X, y, method="hat")
        assert isinstance(report, LeakageReport)

    def test_low_risk_for_independent_data(self):
        """独立データではリスクが低い"""
        X, y = _make_independent_data(50, 5)
        report = detect_leakage(X, y, method="rbf")
        assert report.risk_level in ("low", "medium")

    def test_high_risk_for_grouped_data(self):
        """グループデータではリスクが高い"""
        X, y, _ = _make_grouped_data(15, 3, noise=0.001)
        report = detect_leakage(X, y, method="rbf")
        assert report.risk_level in ("medium", "high")
        assert report.risk_score > 0.2

    def test_group_labels_assigned_for_high_risk(self):
        """高リスク時はグループラベルが推定される"""
        X, y, _ = _make_grouped_data(15, 3, noise=0.001)
        report = detect_leakage(X, y, method="rbf")
        if report.risk_level in ("medium", "high"):
            assert report.group_labels is not None
            assert len(report.group_labels) == X.shape[0]

    def test_cv_recommendation(self):
        """CV推奨が適切"""
        X, y = _make_independent_data(50, 5)
        report = detect_leakage(X, y, method="hat")
        assert report.recommended_cv in ("KFold", "GroupKFold", "LeaveOneGroupOut")
        assert len(report.cv_reason) > 0

    def test_accepts_dataframe(self):
        """DataFrameを受け付ける"""
        X, y = _make_independent_data(30, 5)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        report = detect_leakage(df, pd.Series(y))
        assert isinstance(report, LeakageReport)

    def test_auto_method_selection(self):
        """method='auto' でエラーなく動作"""
        X, y = _make_independent_data(30, 5)
        report = detect_leakage(X, y, method="auto")
        assert report.method_used in ("hat", "rbf", "rf")

    def test_rf_method(self):
        """method='rf' で動作"""
        X, y = _make_independent_data(30, 3)
        report = detect_leakage(X, y, method="rf", rf_n_estimators=50)
        assert report.method_used == "rf"

    def test_suspicious_pairs_sorted(self):
        """疑わしいペアが類似度降順でソート"""
        X, y, _ = _make_grouped_data(10, 3, noise=0.001)
        report = detect_leakage(X, y, method="rbf", similarity_threshold=0.5)
        if len(report.suspicious_pairs) >= 2:
            for i in range(len(report.suspicious_pairs) - 1):
                assert report.suspicious_pairs[i].similarity >= report.suspicious_pairs[i + 1].similarity

    def test_handles_nan(self):
        """NaN含みデータでエラーなく動作"""
        X, y = _make_independent_data(30, 5)
        X[0, 0] = np.nan
        X[5, 2] = np.nan
        report = detect_leakage(X, y, method="rbf")
        assert isinstance(report, LeakageReport)

    def test_details_contain_metadata(self):
        """detailsに診断情報が含まれる"""
        X, y = _make_independent_data(30, 5)
        report = detect_leakage(X, y, method="hat")
        assert "group_consistency_score" in report.details
        assert "n_samples" in report.details
        assert "n_features" in report.details
