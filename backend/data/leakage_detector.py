# -*- coding: utf-8 -*-
"""
backend/data/leakage_detector.py

リーケージ検出・グループ推定モジュール。

説明変数空間でのサンプル間類似度を評価し、train/test 間のリーケージ
危険性を定量化する。グループ構造が検出された場合は GroupKFold 用の
グループラベルを自動推定し、最適な CV 戦略を提案する。

数学的根拠:
  - ハット行列: H = X(X^T X)^{-1} X^T  (Chatterjee & Hadi, 1988)
  - RBFカーネル: K_{ij} = exp(-γ‖x_i - x_j‖²)
  - RF Proximity: Breiman, L. (2001). Random Forests. Machine Learning, 45, 5-32.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# データクラス
# ═══════════════════════════════════════════════════════════════

@dataclass
class LeakagePair:
    """リーケージが疑われるサンプルペア。"""
    idx_a: int
    idx_b: int
    similarity: float
    method: str


@dataclass
class LeakageReport:
    """リーケージ検出の結果レポート。"""
    risk_level: Literal["low", "medium", "high"]
    risk_score: float  # 0.0 (安全) ~ 1.0 (危険)
    n_suspicious_pairs: int
    suspicious_pairs: list[LeakagePair]
    group_labels: np.ndarray | None  # グループ推定結果
    n_groups: int
    recommended_cv: str  # "KFold", "GroupKFold", "LeaveOneGroupOut"
    cv_reason: str
    similarity_matrix: np.ndarray | None = None
    method_used: str = ""
    details: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# 類似度行列の計算
# ═══════════════════════════════════════════════════════════════

def compute_hat_matrix(X: np.ndarray) -> np.ndarray:
    """
    ハット行列（smoother行列）を計算する。

    H = X (X^T X)^{-1} X^T

    引用: Chatterjee & Hadi (1988), Sensitivity Analysis in Linear Regression.
    対角要素 h_ii = レバレッジ（自己影響度）。
    非対角要素 h_ij = サンプル j がサンプル i の予測にどれだけ寄与するか。

    Args:
        X: (n_samples, n_features) の特徴量行列（標準化推奨）

    Returns:
        H: (n_samples, n_samples) のハット行列
    """
    n, p = X.shape
    if n <= p:
        # 特異の場合はリッジ正則化
        reg = 1e-6 * np.eye(p)
        H = X @ np.linalg.solve(X.T @ X + reg, X.T)
    else:
        H = X @ np.linalg.solve(X.T @ X, X.T)
    return H


def compute_rbf_gram(
    X: np.ndarray,
    gamma: float | None = None,
) -> np.ndarray:
    """
    RBF（ガウス）カーネルのグラム行列を計算する。

    K_{ij} = exp(-γ ‖x_i - x_j‖²)

    γ のデフォルト: メディアンヒューリスティック
    γ = 1 / (2 × median(‖x_i - x_j‖²))

    Args:
        X: (n_samples, n_features) の特徴量行列
        gamma: RBFパラメータ（Noneで自動推定）

    Returns:
        K: (n_samples, n_samples) のグラム行列
    """
    from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

    if gamma is None:
        dists_sq = euclidean_distances(X) ** 2
        # メディアンヒューリスティック（対角除く）
        mask = np.triu(np.ones_like(dists_sq, dtype=bool), k=1)
        median_dist_sq = np.median(dists_sq[mask])
        gamma = 1.0 / (2.0 * max(median_dist_sq, 1e-10))

    K = rbf_kernel(X, gamma=gamma)
    return K


def compute_rf_proximity(
    X: np.ndarray,
    y: np.ndarray | None = None,
    n_estimators: int = 200,
    random_state: int = 42,
) -> np.ndarray:
    """
    ランダムフォレストの近接度行列（Proximity Matrix）を計算する。

    P_{ij} = (1/T) Σ_t 1[leaf_t(x_i) == leaf_t(x_j)]

    引用: Breiman, L. (2001). Random Forests. Machine Learning, 45, 5-32.
    sklearn の apply() メソッドで各サンプルの leaf node ID を取得し、
    leaf co-occurrence 率を計算する。

    Args:
        X: (n_samples, n_features) の特徴量行列
        y: 目的変数（Noneの場合はUnsupervised RF）
        n_estimators: 木の数
        random_state: 乱数シード

    Returns:
        P: (n_samples, n_samples) の近接度行列 (値域 [0, 1])
    """
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    n = X.shape[0]

    if y is not None:
        # 教師あり RF
        if pd.api.types.is_float_dtype(y):
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
                max_features="sqrt",
            )
        else:
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
                max_features="sqrt",
            )
        rf.fit(X, y)
    else:
        # Unsupervised RF: 元データ vs シャッフルデータの2クラス分類
        X_shuffled = X.copy()
        rng = np.random.RandomState(random_state)
        for col in range(X_shuffled.shape[1]):
            rng.shuffle(X_shuffled[:, col])
        X_combined = np.vstack([X, X_shuffled])
        y_combined = np.concatenate([np.ones(n), np.zeros(n)])
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_combined, y_combined)

    # leaf node IDを取得
    leaf_ids = rf.apply(X)  # (n_samples, n_estimators)

    # 近接度行列を計算（ベクトル化）
    P = np.zeros((n, n), dtype=np.float32)
    n_trees = leaf_ids.shape[1]

    for t in range(n_trees):
        leaves = leaf_ids[:, t]
        # 同じ leaf に属するサンプルペアをカウント
        for leaf_val in np.unique(leaves):
            members = np.where(leaves == leaf_val)[0]
            if len(members) > 1:
                for i_idx in range(len(members)):
                    for j_idx in range(i_idx + 1, len(members)):
                        P[members[i_idx], members[j_idx]] += 1
                        P[members[j_idx], members[i_idx]] += 1

    P /= n_trees
    np.fill_diagonal(P, 1.0)
    return P


# ═══════════════════════════════════════════════════════════════
# リーケージリスク評価
# ═══════════════════════════════════════════════════════════════

def _find_suspicious_pairs(
    S: np.ndarray,
    threshold: float,
    method: str,
) -> list[LeakagePair]:
    """類似度行列から閾値以上のペアを抽出。"""
    n = S.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] >= threshold:
                pairs.append(LeakagePair(
                    idx_a=i, idx_b=j,
                    similarity=float(S[i, j]),
                    method=method,
                ))
    # 類似度降順でソート
    pairs.sort(key=lambda p: p.similarity, reverse=True)
    return pairs


def _compute_group_consistency_score(
    S: np.ndarray,
    top_k: int = 5,
) -> float:
    """
    グループ一貫性スコアを計算する。

    各サンプルの寄与度 top-K 集合の相互性を評価:
    - サンプル A の top-K に B が含まれ、かつ B の top-K に A が含まれる
      → 相互的グループ（高リスク）
    - 連鎖的（A→B→C だが A→C ではない）
      → やや低リスク

    Returns:
        0.0 (グループ性なし) ~ 1.0 (完全グループ構造)
    """
    n = S.shape[0]
    if n <= top_k:
        top_k = max(1, n - 1)

    # 対角を除く
    S_no_diag = S.copy()
    np.fill_diagonal(S_no_diag, -np.inf)

    # 各サンプルの top-K インデックス
    top_k_sets = []
    for i in range(n):
        topk_idx = np.argsort(S_no_diag[i])[-top_k:]
        top_k_sets.append(set(topk_idx.tolist()))

    # 相互性スコア: A の top-K に B が含まれ、かつ B の top-K に A が含まれる
    mutual_count = 0
    total_count = 0
    for i in range(n):
        for j in top_k_sets[i]:
            total_count += 1
            if i in top_k_sets[j]:
                mutual_count += 1

    if total_count == 0:
        return 0.0

    return mutual_count / total_count


# ═══════════════════════════════════════════════════════════════
# グループ推定
# ═══════════════════════════════════════════════════════════════

def estimate_groups(
    S: np.ndarray,
    n_clusters_range: tuple[int, int] = (2, 10),
    method: str = "hierarchical",
) -> tuple[np.ndarray, int]:
    """
    類似度行列からグループ（クラスタ）を推定する。

    階層的クラスタリングを使用し、シルエットスコアで最適クラスタ数を選定。

    Args:
        S: (n, n) 類似度行列 (値域 [0, 1])
        n_clusters_range: 探索するクラスタ数の範囲
        method: "hierarchical" (default)

    Returns:
        (group_labels, n_groups)
    """
    n = S.shape[0]
    max_clusters = min(n_clusters_range[1], n - 1)
    min_clusters = max(n_clusters_range[0], 2)

    if max_clusters < min_clusters or n < 3:
        return np.zeros(n, dtype=int), 1

    # 距離行列に変換
    D = 1.0 - np.clip(S, 0, 1)
    np.fill_diagonal(D, 0.0)

    # 対称性を保証
    D = (D + D.T) / 2.0

    # 階層的クラスタリング
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")

    best_score = -1
    best_labels = np.zeros(n, dtype=int)
    best_k = 1

    for k in range(min_clusters, max_clusters + 1):
        labels = fcluster(Z, t=k, criterion="maxclust") - 1
        if len(np.unique(labels)) < 2:
            continue
        try:
            score = silhouette_score(D, labels, metric="precomputed")
            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k
        except Exception:
            continue

    return best_labels, best_k


# ═══════════════════════════════════════════════════════════════
# メインAPI: detect_leakage
# ═══════════════════════════════════════════════════════════════

def detect_leakage(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | None = None,
    method: Literal["hat", "rbf", "rf", "auto"] = "auto",
    similarity_threshold: float = 0.95,
    top_k: int = 5,
    rf_n_estimators: int = 200,
) -> LeakageReport:
    """
    リーケージ検出のメインAPI。

    説明変数 X のサンプル間類似度を評価し、リーケージリスクを報告する。
    グループ構造が検出された場合は GroupKFold 用のグループラベルも推定する。

    Args:
        X: 説明変数 (DataFrame or ndarray)
        y: 目的変数 (optional, RF proximity で使用)
        method: "hat" / "rbf" / "rf" / "auto"
            - auto: サンプル数に応じて自動選択
        similarity_threshold: この値以上の類似度を「疑わしい」と判定
        top_k: グループ一貫性スコア計算の top-K 数
        rf_n_estimators: RF proximity の木の数

    Returns:
        LeakageReport
    """
    # --- データ準備 ---
    if isinstance(X, pd.DataFrame):
        X_arr = X.select_dtypes(include=[np.number]).values.copy()
    else:
        X_arr = np.asarray(X, dtype=float).copy()

    if isinstance(y, pd.Series):
        y_arr = y.values
    elif y is not None:
        y_arr = np.asarray(y)
    else:
        y_arr = None

    n, p = X_arr.shape

    # NaN 処理
    col_means = np.nanmean(X_arr, axis=0)
    for j in range(p):
        mask = np.isnan(X_arr[:, j])
        X_arr[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.0

    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    # --- 手法選択 ---
    if method == "auto":
        if n > 5000:
            method = "rbf"  # 大規模データはRBFが高速
        elif n > 500:
            method = "rf"   # 中規模はRFが最もロバスト
        elif p >= n:
            method = "rbf"  # 高次元はハット行列が不安定
        else:
            method = "hat"  # 小規模はハット行列で十分

    # --- 類似度行列の計算 ---
    if method == "hat":
        S_raw = compute_hat_matrix(X_scaled)
        # ハット行列の値域を [0,1] に正規化
        S = np.clip(np.abs(S_raw), 0, 1)
    elif method == "rbf":
        S = compute_rbf_gram(X_scaled)
    elif method == "rf":
        S = compute_rf_proximity(X_scaled, y_arr, n_estimators=rf_n_estimators)
    else:
        raise ValueError(f"未知の method: {method}")

    # --- リーケージペアの検出 ---
    pairs = _find_suspicious_pairs(S, similarity_threshold, method)
    n_suspicious = len(pairs)

    # --- グループ一貫性スコアの計算 ---
    group_score = _compute_group_consistency_score(S, top_k=top_k)

    # --- リスク評価 ---
    # 疑わしいペアの比率
    total_pairs = n * (n - 1) / 2
    pair_ratio = n_suspicious / max(total_pairs, 1)

    # 総合スコア（ペア比率 + グループ一貫性の加重平均）
    risk_score = min(1.0, 0.4 * min(pair_ratio * 100, 1.0) + 0.6 * group_score)

    if risk_score >= 0.7:
        risk_level = "high"
    elif risk_score >= 0.3:
        risk_level = "medium"
    else:
        risk_level = "low"

    # --- グループ推定 ---
    group_labels = None
    n_groups = 0
    if risk_level in ("medium", "high") and n >= 4:
        group_labels, n_groups = estimate_groups(S)

    # --- CV推奨 ---
    if risk_level == "high" and group_labels is not None and n_groups >= 2:
        if n_groups <= 5:
            recommended_cv = "LeaveOneGroupOut"
            cv_reason = (
                f"グループ構造が強く検出されました（{n_groups}グループ）。"
                "LeaveOneGroupOut での評価を推奨します。"
            )
        else:
            recommended_cv = "GroupKFold"
            cv_reason = (
                f"グループ構造が検出されました（{n_groups}グループ）。"
                "GroupKFold で同一グループが train/test に分割されないようにしてください。"
            )
    elif risk_level == "medium" and group_labels is not None and n_groups >= 2:
        recommended_cv = "GroupKFold"
        cv_reason = (
            f"中程度のグループ構造が検出されました（{n_groups}グループ）。"
            "GroupKFold の使用を検討してください。"
        )
    else:
        recommended_cv = "KFold"
        cv_reason = "顕著なグループ構造は検出されませんでした。通常の KFold で問題ありません。"

    return LeakageReport(
        risk_level=risk_level,
        risk_score=risk_score,
        n_suspicious_pairs=n_suspicious,
        suspicious_pairs=pairs[:50],  # 上位50ペアのみ
        group_labels=group_labels,
        n_groups=n_groups,
        recommended_cv=recommended_cv,
        cv_reason=cv_reason,
        similarity_matrix=S if n <= 500 else None,  # 大規模データではメモリ節約
        method_used=method,
        details={
            "group_consistency_score": group_score,
            "pair_ratio": pair_ratio,
            "n_samples": n,
            "n_features": p,
            "similarity_threshold": similarity_threshold,
        },
    )


# ═══════════════════════════════════════════════════════════════
# 特徴量レベルのリーケージ検出（軽量・自動実行向け）
# ═══════════════════════════════════════════════════════════════

@dataclass
class FeatureLeakageWarning:
    """特徴量レベルのリーケージ警告。"""
    feature: str
    risk: Literal["high", "medium"]
    reason: str
    score: float  # 相関やAUC等の具体的数値


@dataclass
class FeatureLeakageReport:
    """特徴量レベルのリーケージ検出レポート。"""
    has_risk: bool
    warnings: list[FeatureLeakageWarning]
    summary: str


def check_feature_leakage(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: list[str] | None = None,
    corr_threshold_high: float = 0.98,
    corr_threshold_medium: float = 0.95,
    max_features_to_check: int = 500,
) -> FeatureLeakageReport:
    """
    特徴量と目的変数間のリーケージリスクを自動チェックする（軽量版）。

    以下の観点でチェック:
    1. **異常高相関**: |Pearson r| ≥ threshold → リーケージの可能性
    2. **名前類似**: 目的変数名と部分一致する列名 → 派生変数の疑い
    3. **定数列からの完全分離**: カテゴリ目的変数の場合、
       単一特徴量で完全分離できる列 → リーケージの可能性

    Args:
        df: データフレーム
        target_col: 目的変数列名
        exclude_cols: 除外する列名リスト
        corr_threshold_high: 高リスク判定閾値（|r| ≥ この値）
        corr_threshold_medium: 中リスク判定閾値
        max_features_to_check: チェックする特徴量数の上限

    Returns:
        FeatureLeakageReport
    """
    warnings_list: list[FeatureLeakageWarning] = []
    _exclude = set(exclude_cols or [])
    _exclude.add(target_col)

    if target_col not in df.columns:
        return FeatureLeakageReport(
            has_risk=False, warnings=[], summary="目的変数が見つかりません。"
        )

    target = df[target_col]
    feature_cols = [c for c in df.columns if c not in _exclude]

    # 特徴量数が多い場合は数値列のみチェック
    if len(feature_cols) > max_features_to_check:
        feature_cols = [
            c for c in feature_cols
            if pd.api.types.is_numeric_dtype(df[c])
        ][:max_features_to_check]

    # ── 1. 相関チェック ────────────────────────────────────────
    if pd.api.types.is_numeric_dtype(target):
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            try:
                r = df[col].corr(target)
                if pd.isna(r):
                    continue
                abs_r = abs(r)
                if abs_r >= corr_threshold_high:
                    warnings_list.append(FeatureLeakageWarning(
                        feature=col,
                        risk="high",
                        reason=f"目的変数との相関が極めて高い (|r|={abs_r:.4f})",
                        score=abs_r,
                    ))
                elif abs_r >= corr_threshold_medium:
                    warnings_list.append(FeatureLeakageWarning(
                        feature=col,
                        risk="medium",
                        reason=f"目的変数との相関が非常に高い (|r|={abs_r:.4f})",
                        score=abs_r,
                    ))
            except Exception:
                continue

    # ── 2. 名前類似チェック ──────────────────────────────────────
    target_lower = target_col.lower().replace(" ", "").replace("_", "")
    for col in feature_cols:
        col_lower = col.lower().replace(" ", "").replace("_", "")
        # 完全一致・部分一致のチェック（同名列は除外済み）
        if len(target_lower) >= 3 and len(col_lower) >= 3:
            if (target_lower in col_lower or col_lower in target_lower):
                # 既にhigh相関で検出済みか確認
                already = any(w.feature == col and w.risk == "high" for w in warnings_list)
                if not already:
                    warnings_list.append(FeatureLeakageWarning(
                        feature=col,
                        risk="medium",
                        reason=f"目的変数「{target_col}」と名前が類似",
                        score=0.0,
                    ))

    # ── 3. 分類タスク: 単一特徴量による完全分離チェック ────────────
    if not pd.api.types.is_float_dtype(target):
        n_classes = target.nunique()
        if 2 <= n_classes <= 20:  # カテゴリ数が妥当な範囲
            for col in feature_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                try:
                    # グループ内分散 / 全体分散 が極端に小さい → ほぼ完全分離
                    valid = df[[col, target_col]].dropna()
                    if len(valid) < 10:
                        continue
                    overall_var = valid[col].var()
                    if overall_var < 1e-10:
                        continue
                    within_var = valid.groupby(target_col)[col].var().mean()
                    ratio = within_var / overall_var
                    if ratio < 0.01:  # グループ内分散 < 全体分散の1%
                        already = any(w.feature == col for w in warnings_list)
                        if not already:
                            warnings_list.append(FeatureLeakageWarning(
                                feature=col,
                                risk="high",
                                reason=f"この特徴量だけで目的変数をほぼ完全に分離可能 (分散比={ratio:.4f})",
                                score=1.0 - ratio,
                            ))
                except Exception:
                    continue

    # ── サマリー生成 ──────────────────────────────────────────
    n_high = sum(1 for w in warnings_list if w.risk == "high")
    n_medium = sum(1 for w in warnings_list if w.risk == "medium")

    if n_high > 0:
        summary = f"⚠️ 高リスク {n_high}件のリーケージ疑いが検出されました。これらの特徴量を確認してください。"
    elif n_medium > 0:
        summary = f"⚡ 中リスク {n_medium}件の注意が必要な特徴量があります。"
    else:
        summary = "✅ リーケージリスクは検出されませんでした。"

    # スコア順ソート
    warnings_list.sort(key=lambda w: (0 if w.risk == "high" else 1, -w.score))

    return FeatureLeakageReport(
        has_risk=len(warnings_list) > 0,
        warnings=warnings_list,
        summary=summary,
    )

