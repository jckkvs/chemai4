"""
backend/interpret/sri.py

SHAP SRI分解モジュール: Synergy / Redundancy / Independence

論文: Ittner et al. "Feature Synergy, Redundancy, and Independence in Global
      Model Explanations using SHAP Vector Decomposition"
      arXiv:2107.12436 (2021)

Reference (原文引用):
    "We decompose the SHAP vectors of feature pairs into three components:
     synergistic, redundant, and independent effects."
    訳: 特徴ペアのSHAPベクトルを Synergy (相乗), Redundancy (冗長),
        Independence (独立) の3成分に幾何学的に分解する。

    "The decomposition allows us to understand how features share predictive
     information and is directly applicable to global model explanation."
    訳: この分解により特徴量間で予測情報をどのように共有しているかを理解でき、
        グローバルモデル説明に直接適用できる。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backend.interpret.shap_explainer import ShapResult

logger = logging.getLogger(__name__)


@dataclass
class SRIResult:
    """
    SRI分解結果を保持するデータクラス。

    Attributes:
        synergy_matrix:    ペアごとのSynergy成分 (n_features x n_features)
        redundancy_matrix: ペアごとのRedundancy成分 (n_features x n_features)
        independence_vec:  各特徴量のIndependent成分 (n_features,)
        feature_names:     特徴量名リスト
        total_sri:         全特徴量の (synergy_total, redundancy_total, independence_total)
    """
    synergy_matrix: np.ndarray
    redundancy_matrix: np.ndarray
    independence_vec: np.ndarray
    feature_names: list[str]
    total_sri: tuple[float, float, float]

    def summary_df(self) -> pd.DataFrame:
        """
        特徴量ごとのSRI集計DataFrameを返す（GUI表示用）。

        Returns:
            列 [feature, synergy_sum, redundancy_sum, independence] のDataFrame
        """
        n = len(self.feature_names)
        rows = []
        for i in range(n):
            rows.append({
                "feature": self.feature_names[i],
                "synergy": float(self.synergy_matrix[i].sum()),
                "redundancy": float(self.redundancy_matrix[i].sum()),
                "independence": float(self.independence_vec[i]),
            })
        df = pd.DataFrame(rows)
        # 合計をsriスコアで正規化（可視化しやすくする）
        total = df[["synergy", "redundancy", "independence"]].abs().sum().sum()
        if total > 0:
            df["synergy_norm"] = df["synergy"] / total
            df["redundancy_norm"] = df["redundancy"] / total
            df["independence_norm"] = df["independence"] / total
        else:
            df["synergy_norm"] = df["redundancy_norm"] = df["independence_norm"] = 0.0
        return df.sort_values("independence", ascending=False).reset_index(drop=True)

    def pairwise_df(self) -> pd.DataFrame:
        """
        全特徴ペアの Synergy/Redundancy をDataFrame形式で返す。

        Returns:
            列 [feature_i, feature_j, synergy, redundancy] のDataFrame
        """
        n = len(self.feature_names)
        rows = []
        for i in range(n):
            for j in range(i + 1, n):
                rows.append({
                    "feature_i": self.feature_names[i],
                    "feature_j": self.feature_names[j],
                    "synergy": float(self.synergy_matrix[i, j]),
                    "redundancy": float(self.redundancy_matrix[i, j]),
                })
        return pd.DataFrame(rows).sort_values("synergy", ascending=False).reset_index(drop=True)


class SRIDecomposer:
    """
    SHAP ベクトルの SRI 分解器。

    アルゴリズム概要（Ittner et al. 2021 に基づく）:
    ──────────────────────────────────────────────────────────────────────────
    Given SHAP values Φ ∈ R^{n×d} (n samples, d features):

    1. 全特徴ペア (i, j) について:
       S_ij = CoVar(φ_i, φ_j)  ... co-variation（SHAPベクトルの内積）
       R_ij = (||φ_i|| ||φ_j|| cos²θ) / (||φ_i|| + ||φ_j||)  ... Redundancy
       Y_ij = S_ij - R_ij  ... Synergy

    2. 各特徴量の Independent 成分:
       I_i = ||φ_i||^2 - Σ_{j≠i} |S_ij|

    論文原文 (Section 3.1):
    "S_ij measures the co-variation between feature i and feature j in the
     SHAP decomposition. A positive value corresponds to synergistic behavior."
    訳: S_ij はSHAP分解における特徴量i,jの共変動を測定する。
        正の値は相乗的な振る舞いに対応する。
    ──────────────────────────────────────────────────────────────────────────

    Args:
        center: True の場合、SHAPベクトルを列ごとに中心化してから計算する
    """

    def __init__(self, center: bool = True) -> None:
        self.center = center

    def decompose(self, shap_result: ShapResult) -> SRIResult:
        """
        ShapResult からSRI分解を実行して SRIResult を返す。

        Args:
            shap_result: ShapExplainer.explain() の出力

        Returns:
            SRIResult インスタンス

        Raises:
            ValueError: SHAP値が2次元でない場合（マルチクラスは非対応）
        """
        Phi = shap_result.shap_values
        if Phi.ndim == 3:
            # マルチクラス: クラス0のSHAPを使用（平均も選択肢）
            logger.warning("マルチクラスSHAP検出: クラス0のSHAP値でSRI分解を実行します。")
            Phi = Phi[:, :, 0]

        if Phi.ndim != 2:
            raise ValueError(f"SHAP values は2次元配列が必要です。shape={Phi.shape}")

        n, d = Phi.shape
        feature_names = shap_result.feature_names

        if self.center:
            Phi = Phi - Phi.mean(axis=0, keepdims=True)

        # ── Synergy と Redundancy の計算 ──────────────────────────────────────
        # Eq. (3):  S_ij = (1/n) * φ_i^T φ_j  (内積の平均)
        # Eq. (4):  R_ij = (||φ_i||² ||φ_j||² cos²θ) / (||φ_i||² + ||φ_j||²)
        #                = (S_ij)² / (||φ_i||² + ||φ_j||²)   [論文の近似式]

        norms_sq = np.einsum("ni,ni->i", Phi, Phi)  # shape (d,)
        # 共分散行列 (d x d): S_ij
        cov_matrix = (Phi.T @ Phi) / n              # shape (d, d)

        synergy_matrix = np.zeros((d, d))
        redundancy_matrix = np.zeros((d, d))

        for i in range(d):
            for j in range(i + 1, j_end := d):
                s_ij = cov_matrix[i, j]
                denom = norms_sq[i] + norms_sq[j]
                r_ij = (s_ij ** 2 / denom) if denom > 1e-12 else 0.0
                y_ij = s_ij - r_ij

                synergy_matrix[i, j] = y_ij
                synergy_matrix[j, i] = y_ij
                redundancy_matrix[i, j] = r_ij
                redundancy_matrix[j, i] = r_ij

        # ── Independence 成分 ────────────────────────────────────────────────
        # I_i = ||φ_i||² - Σ_{j≠i} |S_ij|
        independence_vec = norms_sq / n - np.abs(cov_matrix).sum(axis=1) + np.diag(np.abs(cov_matrix))
        independence_vec = np.clip(independence_vec, 0, None)

        total_syn = float(np.abs(synergy_matrix).sum() / 2)     # 対称行列なので半分
        total_red = float(np.abs(redundancy_matrix).sum() / 2)
        total_ind = float(independence_vec.sum())

        logger.info(
            f"SRI分解完了: Synergy={total_syn:.4f}, "
            f"Redundancy={total_red:.4f}, Independence={total_ind:.4f}"
        )

        return SRIResult(
            synergy_matrix=synergy_matrix,
            redundancy_matrix=redundancy_matrix,
            independence_vec=independence_vec,
            feature_names=feature_names,
            total_sri=(total_syn, total_red, total_ind),
        )


def plot_sri_heatmap(
    sri_result: SRIResult,
    component: str = "synergy",
    top_n: int = 20,
    ax: Any = None,
    save_path: str | None = None,
) -> None:
    """
    SRI の Synergy/Redundancy ヒートマップを表示する。

    Args:
        sri_result: SRIResult インスタンス
        component: "synergy" | "redundancy"
        top_n: 表示する上位特徴量数（重要度順）
        ax: matplotlib Axes（省略時は新規作成）
        save_path: 保存パス
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if component == "synergy":
        matrix = sri_result.synergy_matrix
        title = "SHAP SRI: Synergy Matrix"
        cmap = "RdBu_r"
    else:
        matrix = sri_result.redundancy_matrix
        title = "SHAP SRI: Redundancy Matrix"
        cmap = "Blues"

    fnames = sri_result.feature_names

    # 重要度トップNに絞る
    importance = np.abs(matrix).sum(axis=0)
    if len(fnames) > top_n:
        top_idx = np.argsort(importance)[::-1][:top_n]
        matrix = matrix[np.ix_(top_idx, top_idx)]
        fnames = [fnames[i] for i in top_idx]

    df_heatmap = pd.DataFrame(matrix, index=fnames, columns=fnames)

    if ax is None:
        fig_size = max(8, len(fnames) // 2)
        _, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        df_heatmap,
        ax=ax,
        cmap=cmap,
        center=0,
        square=True,
        annot=len(fnames) <= 15,
        fmt=".2f",
        cbar_kws={"label": component.capitalize()},
    )
    ax.set_title(title, fontsize=14)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()


def select_features_by_independence(
    sri_result: SRIResult,
    top_n: int | None = None,
    threshold: float | None = None,
) -> list[str]:
    """
    Independence スコアを基準に特徴量を選択する。

    高いIndependenceスコア = 他の特徴量と重複しない独自の予測力を持つ。

    Args:
        sri_result: SRIResult インスタンス
        top_n: 上位N特徴量を返す（threshold と排他）
        threshold: このスコア以上の特徴量を返す

    Returns:
        選択された特徴量名のリスト（Independence降順）
    """
    idx_sorted = np.argsort(sri_result.independence_vec)[::-1]
    fnames = [sri_result.feature_names[i] for i in idx_sorted]
    scores = sri_result.independence_vec[idx_sorted]

    if top_n is not None:
        return fnames[:top_n]
    if threshold is not None:
        return [f for f, s in zip(fnames, scores, strict=False) if s >= threshold]
    return fnames
