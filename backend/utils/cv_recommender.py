"""
backend/utils/cv_recommender.py

データ特性を自動分析し、最適なCV戦略を推薦するモジュール。

Implements: F-CVR01〜CVR05
    CVRecommendation: 推薦結果データクラス
    recommend_cv_strategy: メインAPI — データ特性に基づくCV自動推薦
    _detect_timeseries: 時系列データ検出
    _detect_groups: 類似サンプルグループ検出
    _detect_imbalance: クラス不均衡検出
    _assess_sample_size: サンプルサイズ評価

設計思想:
    - データ特性（時系列・グループ・不均衡・サンプルサイズ）を独立に検出
    - 各検出結果の重みと優先度で最適CVを決定
    - 推薦理由をユーザーに説明できる形式で返す

参考文献:
    - Arlot & Celisse (2010) "A survey of cross-validation procedures
      for model selection"
      原文: "The choice of V [in V-fold CV] depends on the bias-variance
             trade-off ... when the sample size is small, large V can lead
             to high variance."
      訳: CVの分割数Vの選択はバイアス-バリアンストレードオフに依存する。
          サンプルサイズが小さい場合、大きなVは高い分散をもたらしうる。
    - Roberts et al. (2017) "Cross-validation strategies for data with
      temporal, spatial, hierarchical, or phylogenetic structure"
      原文: "Standard random cross-validation can yield overly optimistic
             assessments of predictive performance when observations are
             not independent."
      訳: 観測値が独立でない場合、標準的なランダムCVは
          予測性能の過度に楽観的な評価をもたらしうる。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# 結果データクラス
# ============================================================

@dataclass
class CVRecommendation:
    """CV推薦の結果。

    Attributes:
        recommended_cv: 推奨するCV手法キー (cv_manager.py のキー)
        confidence: 推薦の確信度 (0.0〜1.0)
        reason: ユーザー向けの推薦理由（日本語）
        alternative_cvs: 代替候補キーのリスト
        warnings: 注意事項リスト
        detected_features: 検出されたデータ特性
        recommended_params: 推奨パラメータ (n_splits等)
    """

    recommended_cv: str
    confidence: float
    reason: str
    alternative_cvs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    detected_features: dict[str, Any] = field(default_factory=dict)
    recommended_params: dict[str, Any] = field(default_factory=dict)


# ============================================================
# メインAPI
# ============================================================

def recommend_cv_strategy(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> CVRecommendation:
    """データ特性に基づき最適なCV戦略を推薦する。

    判定優先順位:
        1. 時系列データ → TimeSeriesSplit / WalkForward
        2. グループ構造 → GroupKFold / LOGO
        3. クラス不均衡 → StratifiedKFold
        4. 小サンプル → LOO / RepeatedKFold
        5. 通常 → KFold

    Args:
        X: 特徴量行列 (DataFrame推奨。列名から自動検出に使用)
        y: 目的変数
        metadata: 追加メタデータ
            - group_col: str — グループ列名
            - time_col: str — 時系列列名
            - task_type: str — "regression" | "classification"
            - groups: np.ndarray — グループラベル配列

    Returns:
        CVRecommendation

    Complexity: 6
    Description: 5段階のデータ特性検出（時系列/グループ/不均衡/サンプルサイズ/通常）
                 に基づくCV自動推薦エンジン。
    """
    meta = metadata or {}
    n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]
    task_type = meta.get("task_type", "regression")

    # DataFrameでない場合の対応
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X)
    else:
        X_df = X

    if isinstance(y, np.ndarray):
        y_s = pd.Series(y)
    else:
        y_s = y

    detected: dict[str, Any] = {
        "n_samples": n_samples,
        "n_features": X_df.shape[1],
        "task_type": task_type,
    }
    warnings: list[str] = []

    # ── 1. 時系列検出 ──
    ts_result = _detect_timeseries(X_df, meta)
    detected["timeseries"] = ts_result

    if ts_result["is_timeseries"]:
        logger.info(f"[CV推薦] 時系列データ検出: {ts_result['reason']}")
        n_splits = _recommend_ts_splits(n_samples)
        return CVRecommendation(
            recommended_cv="timeseries",
            confidence=ts_result["confidence"],
            reason=f"⏰ 時系列データを検出: {ts_result['reason']}。"
                   f"TimeSeriesSplit(n_splits={n_splits})を推奨します。",
            alternative_cvs=["walk_forward"],
            warnings=warnings,
            detected_features=detected,
            recommended_params={"n_splits": n_splits},
        )

    # ── 2. グループ検出 ──
    grp_result = _detect_groups(X_df, y_s, meta)
    detected["groups"] = grp_result

    if grp_result["has_groups"]:
        logger.info(f"[CV推薦] グループ構造検出: {grp_result['reason']}")
        n_groups = grp_result.get("n_groups", 5)
        n_splits = min(n_groups, 10)

        if n_groups <= 5:
            cv_key = "logo"
            reason = (
                f"👥 {n_groups}グループを検出。グループ数が少ないため "
                f"Leave-One-Group-Out (LOGO) を推奨します。"
            )
            alt = ["group_kfold"]
            params: dict[str, Any] = {}
        else:
            cv_key = "group_kfold"
            reason = (
                f"👥 {n_groups}グループを検出: {grp_result['reason']}。"
                f"GroupKFold(n_splits={n_splits})を推奨します。"
            )
            alt = ["logo", "group_shuffle_split"]
            params = {"n_splits": n_splits}

        return CVRecommendation(
            recommended_cv=cv_key,
            confidence=grp_result["confidence"],
            reason=reason,
            alternative_cvs=alt,
            warnings=warnings,
            detected_features=detected,
            recommended_params=params,
        )

    # ── 3. クラス不均衡検出 (分類タスク) ──
    if task_type == "classification":
        imb_result = _detect_imbalance(y_s)
        detected["imbalance"] = imb_result

        if imb_result["is_imbalanced"]:
            logger.info(f"[CV推薦] クラス不均衡検出: {imb_result['reason']}")
            n_splits = _recommend_n_splits(n_samples)
            return CVRecommendation(
                recommended_cv="stratified_kfold",
                confidence=imb_result["confidence"],
                reason=f"⚖️ クラス不均衡を検出: {imb_result['reason']}。"
                       f"StratifiedKFold(n_splits={n_splits})を推奨します。",
                alternative_cvs=["stratified_shuffle_split",
                                 "repeated_stratified_kfold"],
                warnings=warnings,
                detected_features=detected,
                recommended_params={"n_splits": n_splits, "shuffle": True},
            )

    # ── 4. サンプルサイズ評価 ──
    size_result = _assess_sample_size(n_samples, X_df.shape[1])
    detected["size_assessment"] = size_result

    if size_result["is_small"]:
        logger.info(f"[CV推薦] 小サンプルデータ: {size_result['reason']}")

        if n_samples <= 20:
            return CVRecommendation(
                recommended_cv="loo",
                confidence=0.85,
                reason=f"📏 サンプル数が{n_samples}件と非常に少ないため、"
                       f"Leave-One-Out (LOO) を推奨します。",
                alternative_cvs=["repeated_kfold"],
                warnings=[
                    "⚠️ LOOは計算コストが高いため、サンプル数が増えた場合は"
                    "KFoldへの切替を検討してください。"
                ],
                detected_features=detected,
                recommended_params={},
            )
        else:
            n_splits = min(3, n_samples // 2)
            return CVRecommendation(
                recommended_cv="repeated_kfold",
                confidence=0.75,
                reason=f"📏 サンプル数が{n_samples}件と少ないため、"
                       f"RepeatedKFold(n_splits={n_splits}, n_repeats=5)を推奨します。",
                alternative_cvs=["kfold", "loo"],
                warnings=[
                    f"⚠️ {n_samples}件のデータでは分割数を{n_splits}に抑え、"
                    f"繰り返し数を増やして分散を低減します。"
                ],
                detected_features=detected,
                recommended_params={
                    "n_splits": n_splits,
                    "n_repeats": 5,
                    "shuffle": True,
                },
            )

    # ── 5. デフォルト ──
    n_splits = _recommend_n_splits(n_samples)

    if task_type == "classification":
        return CVRecommendation(
            recommended_cv="stratified_kfold",
            confidence=0.70,
            reason=f"📊 標準的な分類タスク（{n_samples}件）。"
                   f"StratifiedKFold(n_splits={n_splits})を推奨します。",
            alternative_cvs=["kfold", "repeated_stratified_kfold"],
            detected_features=detected,
            recommended_params={"n_splits": n_splits, "shuffle": True},
        )
    else:
        return CVRecommendation(
            recommended_cv="kfold",
            confidence=0.70,
            reason=f"📊 標準的な回帰タスク（{n_samples}件）。"
                   f"KFold(n_splits={n_splits})を推奨します。",
            alternative_cvs=["repeated_kfold", "shuffle_split"],
            detected_features=detected,
            recommended_params={"n_splits": n_splits, "shuffle": True},
        )


# ============================================================
# 検出関数群
# ============================================================

def _detect_timeseries(
    X: pd.DataFrame,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """時系列データを検出する。

    検出方法:
        1. metadata に time_col が明示指定されている
        2. 列名パターンマッチ (date, time, year, month, timestamp 等)
        3. 単調増加する整数/日時列の存在

    Returns:
        {"is_timeseries": bool, "confidence": float, "reason": str,
         "detected_column": str|None}
    """
    result: dict[str, Any] = {
        "is_timeseries": False,
        "confidence": 0.0,
        "reason": "",
        "detected_column": None,
    }

    # 1. 明示指定
    time_col = meta.get("time_col")
    if time_col and time_col in X.columns:
        result.update({
            "is_timeseries": True,
            "confidence": 0.95,
            "reason": f"ユーザー指定の時系列列 '{time_col}' を検出",
            "detected_column": time_col,
        })
        return result

    # 2. 列名パターンマッチ
    ts_patterns = [
        r"(?i)^(date|time|timestamp|datetime|日付|日時|年月日)$",
        r"(?i)^(year|month|day|hour|minute|年|月|日)$",
        r"(?i)(date|time|period|epoch|created|updated)(_at|_on|_stamp)?$",
    ]
    for col in X.columns:
        for pat in ts_patterns:
            if re.search(pat, str(col)):
                result.update({
                    "is_timeseries": True,
                    "confidence": 0.80,
                    "reason": f"列名パターン '{col}' が時系列を示唆",
                    "detected_column": col,
                })
                return result

    # 3. 単調増加列の検出（数値列のみ）
    for col in X.select_dtypes(include=[np.number]).columns:
        vals = X[col].dropna().values
        if len(vals) < 10:
            continue
        # 90%以上の隣接差が正なら単調増加とみなす
        diffs = np.diff(vals)
        positive_ratio = np.mean(diffs > 0)
        if positive_ratio >= 0.90:
            # さらに: 等間隔に近いかチェック（CVの15%以内なら時系列的）
            if len(diffs) > 0:
                cv = np.std(diffs) / (np.abs(np.mean(diffs)) + 1e-15)
                if cv < 0.15:
                    result.update({
                        "is_timeseries": True,
                        "confidence": 0.70,
                        "reason": f"列 '{col}' が等間隔の単調増加列"
                                  f"（正差分率={positive_ratio:.0%}, CV={cv:.3f}）",
                        "detected_column": str(col),
                    })
                    return result

    return result


def _detect_groups(
    X: pd.DataFrame,
    y: pd.Series,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """グループ構造を検出する。

    検出方法:
        1. metadata に group_col/groups が明示指定されている
        2. リーケージチェックのグループラベルが提供されている
        3. RBFカーネル類似度による階層的クラスタリング（高類似度ペア検出）

    Returns:
        {"has_groups": bool, "confidence": float, "reason": str,
         "n_groups": int, "group_labels": ndarray|None}
    """
    result: dict[str, Any] = {
        "has_groups": False,
        "confidence": 0.0,
        "reason": "",
        "n_groups": 0,
        "group_labels": None,
    }

    # 1. 明示指定 (group_col)
    group_col = meta.get("group_col")
    if group_col and isinstance(X, pd.DataFrame) and group_col in X.columns:
        labels = X[group_col].values
        n_groups = len(np.unique(labels[~pd.isna(labels)]))
        result.update({
            "has_groups": True,
            "confidence": 0.95,
            "reason": f"ユーザー指定のグループ列 '{group_col}'（{n_groups}グループ）",
            "n_groups": n_groups,
            "group_labels": labels,
        })
        return result

    # 2. 事前計算済みグループラベル
    groups = meta.get("groups")
    if groups is not None:
        groups_arr = np.asarray(groups)
        n_groups = len(np.unique(groups_arr))
        if n_groups >= 2:
            result.update({
                "has_groups": True,
                "confidence": 0.90,
                "reason": f"事前計算済みグループラベル（{n_groups}グループ）",
                "n_groups": n_groups,
                "group_labels": groups_arr,
            })
            return result

    # 3. リーケージチェック結果
    leakage_labels = meta.get("leakage_group_labels")
    if leakage_labels is not None:
        leakage_arr = np.asarray(leakage_labels)
        n_groups = len(np.unique(leakage_arr))
        if n_groups >= 2:
            result.update({
                "has_groups": True,
                "confidence": 0.85,
                "reason": f"リーケージチェックで検出されたグループ（{n_groups}グループ）",
                "n_groups": n_groups,
                "group_labels": leakage_arr,
            })
            return result

    # 4. 自動検出: RBFカーネル類似度による階層的クラスタリング
    #    （大規模データではスキップ — O(n²)のため）
    n_samples = len(X)
    if n_samples <= 500:
        try:
            X_num = X.select_dtypes(include=[np.number]).dropna(axis=1)
            if X_num.shape[1] >= 2:
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics.pairwise import rbf_kernel
                from scipy.cluster.hierarchy import fcluster, linkage

                X_scaled = StandardScaler().fit_transform(X_num.values)
                # RBFカーネル(ガンマ=自動)で類似度行列
                gamma = 1.0 / X_num.shape[1]
                sim_matrix = rbf_kernel(X_scaled, gamma=gamma)

                # 距離行列に変換して凝集型クラスタリング
                dist_matrix = 1.0 - sim_matrix
                np.fill_diagonal(dist_matrix, 0.0)
                dist_matrix = np.maximum(dist_matrix, 0.0)

                # 縮約距離行列を取得
                from scipy.spatial.distance import squareform
                condensed = squareform(dist_matrix, checks=False)
                Z = linkage(condensed, method="average")

                # 高類似度ペアの割合で判定
                # 閾値: 類似度 > 0.95 のペアが 5% 以上存在
                high_sim_mask = sim_matrix > 0.95
                np.fill_diagonal(high_sim_mask, False)
                n_pairs = n_samples * (n_samples - 1) / 2
                high_sim_ratio = high_sim_mask.sum() / 2 / max(n_pairs, 1)

                if high_sim_ratio > 0.05:
                    # 距離閾値 0.3 でクラスタリング
                    labels = fcluster(Z, t=0.3, criterion="distance")
                    n_groups = len(np.unique(labels))
                    if 2 <= n_groups <= n_samples // 3:
                        result.update({
                            "has_groups": True,
                            "confidence": 0.65,
                            "reason": (
                                f"RBFカーネル類似度分析で{n_groups}グループを自動検出"
                                f"（高類似度ペア率={high_sim_ratio:.1%}）"
                            ),
                            "n_groups": n_groups,
                            "group_labels": labels,
                        })
                        return result
        except Exception as e:
            logger.debug(f"グループ自動検出中のエラー（無視）: {e}")

    return result


def _detect_imbalance(y: pd.Series) -> dict[str, Any]:
    """クラス不均衡を検出する。

    判定基準:
        - 最多クラス / 最少クラス の比率 > 3:1
        - 最少クラスのサンプル数 < 10

    Returns:
        {"is_imbalanced": bool, "confidence": float, "reason": str,
         "imbalance_ratio": float, "class_counts": dict}
    """
    result: dict[str, Any] = {
        "is_imbalanced": False,
        "confidence": 0.0,
        "reason": "",
        "imbalance_ratio": 1.0,
        "class_counts": {},
    }

    counts = y.value_counts()
    if len(counts) < 2:
        result["reason"] = "1クラスのみ"
        return result

    result["class_counts"] = counts.to_dict()
    max_count = counts.max()
    min_count = counts.min()
    ratio = max_count / max(min_count, 1)
    result["imbalance_ratio"] = float(ratio)

    if ratio > 10:
        result.update({
            "is_imbalanced": True,
            "confidence": 0.90,
            "reason": f"高度なクラス不均衡（比率={ratio:.1f}:1, "
                      f"最多={max_count}件, 最少={min_count}件）",
        })
    elif ratio > 3:
        result.update({
            "is_imbalanced": True,
            "confidence": 0.75,
            "reason": f"中程度のクラス不均衡（比率={ratio:.1f}:1, "
                      f"最多={max_count}件, 最少={min_count}件）",
        })
    elif min_count < 10:
        result.update({
            "is_imbalanced": True,
            "confidence": 0.70,
            "reason": f"最少クラスのサンプル数が少ない（{min_count}件）",
        })

    return result


def _assess_sample_size(
    n_samples: int,
    n_features: int,
) -> dict[str, Any]:
    """サンプルサイズを評価する。

    判定基準:
        - n < 30: 非常に小さい
        - 30 <= n < 100: 小さい
        - n/p < 2: 特徴量に対してサンプルが少ない (p = n_features)

    Returns:
        {"is_small": bool, "category": str, "reason": str,
         "n_per_feature": float}
    """
    n_per_feature = n_samples / max(n_features, 1)

    result: dict[str, Any] = {
        "is_small": False,
        "category": "normal",
        "reason": "",
        "n_per_feature": round(n_per_feature, 2),
    }

    if n_samples <= 20:
        result.update({
            "is_small": True,
            "category": "very_small",
            "reason": f"サンプル数が{n_samples}件と非常に少ない",
        })
    elif n_samples <= 50:
        result.update({
            "is_small": True,
            "category": "small",
            "reason": f"サンプル数が{n_samples}件と少ない",
        })
    elif n_per_feature < 2.0:
        result.update({
            "is_small": True,
            "category": "high_dim",
            "reason": f"特徴量数({n_features})に対してサンプル数({n_samples})が少ない"
                      f"（比率={n_per_feature:.1f}）",
        })

    return result


# ============================================================
# ヘルパー関数
# ============================================================

def _recommend_n_splits(n_samples: int) -> int:
    """サンプル数に応じた推奨分割数を返す。

    参考: Hastie, Tibshirani, Friedman (2009) "The Elements of
          Statistical Learning", Section 7.10
    """
    if n_samples >= 10000:
        return 10
    elif n_samples >= 1000:
        return 5
    elif n_samples >= 200:
        return 5
    elif n_samples >= 50:
        return 3
    else:
        return min(n_samples, 2)


def _recommend_ts_splits(n_samples: int) -> int:
    """時系列データ用の推奨分割数を返す。

    時系列ではテストサイズが十分確保できる分割数を推奨。
    """
    if n_samples >= 500:
        return 5
    elif n_samples >= 200:
        return 4
    elif n_samples >= 50:
        return 3
    else:
        return 2
