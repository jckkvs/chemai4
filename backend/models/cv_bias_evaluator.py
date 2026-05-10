# -*- coding: utf-8 -*-
"""
backend/models/cv_bias_evaluator.py

CVバイアス評価モジュール。

CVの最小誤差が真のテスト誤差に対して楽観的バイアスを持つ問題に対処する。
以下の2手法を実装:

1. Tibshirani-Tibshirani法 (2009)
   — CV誤差曲線からバイアスを推定
   Ref: R.J. Tibshirani & R. Tibshirani,
        "A bias correction for the minimum error rate in cross-validation",
        Ann. Appl. Stat. 3(2), 822–829. (2009)
   式: Bias = (1/K) Σ_k [e_k(θ̂) - e_k(θ̂_k)]

2. BBC-CV (Bootstrap Bias Corrected CV)
   — Out-of-sample予測値のBootstrapリサンプリングでバイアスを推定
   Ref: I. Tsamardinos, E. Greasidou, G. Borboudakis,
        "Bootstrapping the out-of-sample predictions for efficient
         and accurate cross-validation",
        Machine Learning 107(12), 1895–1922. (2018)

両手法ともオプショナルな評価で、追加の学習は不要。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 結果データクラス
# ============================================================

@dataclass
class CVBiasResult:
    """CVバイアス評価の結果。

    Attributes:
        method: 評価手法名 ("tibshirani" or "bbc_cv")
        raw_score: 補正前の最良CVスコア
        bias_estimate: バイアスの推定値（正＝楽観的方向）
        corrected_score: 補正後のスコア (raw_score - bias_estimate)
        ci_lower: 補正後スコアの95%信頼区間下限（BBC-CVのみ）
        ci_upper: 補正後スコアの95%信頼区間上限（BBC-CVのみ）
        n_bootstrap: Bootstrap回数（BBC-CVのみ）
        details: 追加情報
    """
    method: str
    raw_score: float
    bias_estimate: float
    corrected_score: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    n_bootstrap: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """結果をdictに変換する（UI表示用）。"""
        d = {
            "method": self.method,
            "raw_score": round(self.raw_score, 6),
            "bias_estimate": round(self.bias_estimate, 6),
            "corrected_score": round(self.corrected_score, 6),
        }
        if self.ci_lower is not None:
            d["ci_lower"] = round(self.ci_lower, 6)
            d["ci_upper"] = round(self.ci_upper, 6)
        if self.n_bootstrap is not None:
            d["n_bootstrap"] = self.n_bootstrap
        return d


# ============================================================
# 手法1: Tibshirani-Tibshirani法 (2009)
# ============================================================

def estimate_tibshirani_bias(
    fold_error_curves: np.ndarray,
    param_values: np.ndarray | list[float] | None = None,
    higher_is_better: bool = True,
) -> CVBiasResult:
    """
    Tibshirani-Tibshirani法によるCVバイアス推定。

    CV誤差曲線（各fold × パラメータ候補）から楽観的バイアスを推定する。

    アルゴリズム (論文 §2 式(2)):
        θ̂ = argmin_θ CV(θ)  ← 全体最適パラメータ
        θ̂_k = argmin_θ e_k(θ)  ← fold k の最適パラメータ
        Bias = (1/K) Σ_k [e_k(θ̂) - e_k(θ̂_k)]

    higher_is_better=True のとき (accuracy等):
        θ̂ = argmax, Bias = (1/K) Σ_k [e_k(θ̂_k) - e_k(θ̂)]

    Args:
        fold_error_curves: shape=(K, P) のCV誤差曲線。
            K=fold数, P=パラメータ候補数。
            各 fold_error_curves[k, p] は fold k でパラメータ p の性能値。
        param_values: パラメータ候補のリスト（情報用のみ・省略可）
        higher_is_better: True=大きいほど良い（accuracy等）、
                         False=小さいほど良い（RMSE等）

    Returns:
        CVBiasResult

    Raises:
        ValueError: fold_error_curvesの形状が不正な場合

    Implements: F-CV-BIAS-001
    論文: Tibshirani & Tibshirani (2009), §2 式(2)
    """
    curves = np.asarray(fold_error_curves, dtype=float)
    if curves.ndim != 2:
        raise ValueError(
            f"fold_error_curves は 2次元配列 (K, P) である必要があります。"
            f"shape={curves.shape}"
        )
    K, P = curves.shape

    if P < 2:
        # パラメータ候補が1つだけ（チューニングなし）→バイアス=0
        raw_score = float(curves.mean(axis=0)[0])
        return CVBiasResult(
            method="tibshirani",
            raw_score=raw_score,
            bias_estimate=0.0,
            corrected_score=raw_score,
            details={"n_folds": K, "n_params": P, "note": "単一パラメータ"},
        )

    # CV(θ) = 全foldの平均スコア for each θ
    cv_mean = curves.mean(axis=0)  # shape=(P,)

    if higher_is_better:
        # θ̂ = argmax CV(θ)
        theta_hat_idx = int(np.argmax(cv_mean))
        raw_score = float(cv_mean[theta_hat_idx])

        # 各foldの最適: θ̂_k = argmax e_k(θ)
        theta_hat_k_idx = np.argmax(curves, axis=1)  # shape=(K,)

        # Bias = (1/K) Σ_k [e_k(θ̂_k) - e_k(θ̂)]
        bias_terms = np.array([
            curves[k, theta_hat_k_idx[k]] - curves[k, theta_hat_idx]
            for k in range(K)
        ])
    else:
        # θ̂ = argmin CV(θ)
        theta_hat_idx = int(np.argmin(cv_mean))
        raw_score = float(cv_mean[theta_hat_idx])

        # 各foldの最適: θ̂_k = argmin e_k(θ)
        theta_hat_k_idx = np.argmin(curves, axis=1)

        # Bias = (1/K) Σ_k [e_k(θ̂) - e_k(θ̂_k)]
        bias_terms = np.array([
            curves[k, theta_hat_idx] - curves[k, theta_hat_k_idx[k]]
            for k in range(K)
        ])

    bias = float(np.mean(bias_terms))
    corrected = raw_score - bias if higher_is_better else raw_score + bias

    details: dict[str, Any] = {
        "n_folds": K,
        "n_params": P,
        "best_param_idx": theta_hat_idx,
        "bias_per_fold": bias_terms.tolist(),
    }
    if param_values is not None:
        details["best_param_value"] = float(np.asarray(param_values)[theta_hat_idx])

    logger.info(
        f"TT法バイアス推定: raw={raw_score:.4f}, "
        f"bias={bias:.4f}, corrected={corrected:.4f}"
    )

    return CVBiasResult(
        method="tibshirani",
        raw_score=raw_score,
        bias_estimate=bias,
        corrected_score=corrected,
        details=details,
    )


# ============================================================
# 手法2: BBC-CV (Bootstrap Bias Corrected CV)
# ============================================================

def estimate_bbc_cv_bias(
    oos_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    scoring_func: callable,
    n_bootstrap: int = 200,
    random_state: int = 42,
    higher_is_better: bool = True,
) -> CVBiasResult:
    """
    BBC-CVによるCVバイアス推定。

    Out-of-sample予測値をBootstrapリサンプリングし、
    最良構成の選択プロセスをシミュレートすることで
    バイアスを推定する。

    アルゴリズム (Tsamardinos+ 2018, Algorithm 1):
        1. 各構成 c のOOS予測 ŷ_c を用意
        2. B=n_bootstrap 回のループ:
            a. y_true, ŷ_c をペアで同じインデックスでリサンプリング
            b. 各構成のBootstrapスコアを計算
            c. 最良構成 c* を選択しそのBootstrapスコアを記録
        3. バイアス = mean(Boot最良スコア) - 元の最良OOSスコア
        4. 補正後スコア = 元のOOSスコア - バイアス

    Args:
        oos_predictions: {config_name: oos_pred_array} の辞書。
            各配列は y_true と同じ長さのout-of-sample予測。
        y_true: 真のラベル/値
        scoring_func: callable(y_true, y_pred) → float のスコア関数
        n_bootstrap: Bootstrap回数（デフォルト200）
        random_state: Bootstrap乱数シード
        higher_is_better: スコアが大きいほど良い場合True

    Returns:
        CVBiasResult

    Implements: F-CV-BIAS-002
    論文: Tsamardinos et al. (2018), Algorithm 1
    """
    y = np.asarray(y_true)
    n = len(y)
    config_names = list(oos_predictions.keys())

    if len(config_names) == 0:
        raise ValueError("oos_predictions が空です。")

    # 各構成のOOS予測を配列に変換
    preds = {k: np.asarray(v) for k, v in oos_predictions.items()}
    for k, v in preds.items():
        if len(v) != n:
            raise ValueError(
                f"構成 '{k}' のOOS予測長({len(v)})が y_true 長({n})と不一致。"
            )

    # Step 1: 元のOOSスコアを計算
    original_scores = {}
    for name, pred in preds.items():
        try:
            original_scores[name] = scoring_func(y, pred)
        except Exception as e:
            logger.warning(f"BBC-CV: 構成'{name}'のスコア計算失敗: {e}")
            original_scores[name] = float("-inf") if higher_is_better else float("inf")

    # 最良構成の選択
    if higher_is_better:
        best_config = max(original_scores, key=original_scores.get)
    else:
        best_config = min(original_scores, key=original_scores.get)
    raw_score = original_scores[best_config]

    # 単一構成 → バイアス推定不要
    if len(config_names) == 1:
        return CVBiasResult(
            method="bbc_cv",
            raw_score=raw_score,
            bias_estimate=0.0,
            corrected_score=raw_score,
            n_bootstrap=0,
            details={"note": "単一構成のためバイアス推定不要"},
        )

    # Step 2: Bootstrap loop
    rng = np.random.RandomState(random_state)
    boot_best_scores = []

    for b in range(n_bootstrap):
        # リサンプリング（ペアで同じインデックス）
        idx = rng.choice(n, size=n, replace=True)
        y_boot = y[idx]

        # 各構成のBootstrapスコア
        boot_scores = {}
        for name, pred in preds.items():
            pred_boot = pred[idx]
            try:
                boot_scores[name] = scoring_func(y_boot, pred_boot)
            except Exception:
                boot_scores[name] = (
                    float("-inf") if higher_is_better else float("inf")
                )

        # Bootstrap内での最良構成を選択
        if higher_is_better:
            boot_best = max(boot_scores, key=boot_scores.get)
        else:
            boot_best = min(boot_scores, key=boot_scores.get)

        boot_best_scores.append(boot_scores[boot_best])

    boot_best_scores = np.array(boot_best_scores)

    # Step 3: バイアス推定
    # bias = mean(Bootstrap最良スコア) - 元の最良OOSスコア
    boot_mean = float(np.mean(boot_best_scores))
    bias = boot_mean - raw_score

    # Step 4: 補正
    corrected = raw_score - bias

    # 95%信頼区間（Bootstrap分布から）
    ci_lower = float(np.percentile(boot_best_scores, 2.5))
    ci_upper = float(np.percentile(boot_best_scores, 97.5))
    # 信頼区間もバイアス補正
    ci_lower_corrected = ci_lower - bias
    ci_upper_corrected = ci_upper - bias

    logger.info(
        f"BBC-CV バイアス推定: raw={raw_score:.4f}, "
        f"bias={bias:.4f}, corrected={corrected:.4f}, "
        f"95%CI=[{ci_lower_corrected:.4f}, {ci_upper_corrected:.4f}]"
    )

    return CVBiasResult(
        method="bbc_cv",
        raw_score=raw_score,
        bias_estimate=bias,
        corrected_score=corrected,
        ci_lower=ci_lower_corrected,
        ci_upper=ci_upper_corrected,
        n_bootstrap=n_bootstrap,
        details={
            "best_config": best_config,
            "n_configs": len(config_names),
            "boot_mean": boot_mean,
            "boot_std": float(np.std(boot_best_scores)),
            "original_scores": {k: round(v, 6) for k, v in original_scores.items()},
        },
    )


# ============================================================
# ユーティリティ: 結果フォーマット
# ============================================================

def format_bias_report(result: CVBiasResult) -> str:
    """CVBiasResultを人間可読な文字列に整形する。

    Args:
        result: CVBiasResult インスタンス

    Returns:
        フォーマット済み文字列
    """
    method_name = {
        "tibshirani": "Tibshirani-Tibshirani法 (2009)",
        "bbc_cv": "BBC-CV (Tsamardinos+ 2018)",
    }.get(result.method, result.method)

    lines = [
        f"📊 CVバイアス評価 — {method_name}",
        f"  補正前スコア: {result.raw_score:.4f}",
        f"  バイアス推定: {result.bias_estimate:+.4f}",
        f"  補正後スコア: {result.corrected_score:.4f}",
    ]
    if result.ci_lower is not None and result.ci_upper is not None:
        lines.append(
            f"  95%信頼区間:  [{result.ci_lower:.4f}, {result.ci_upper:.4f}]"
        )
    return "\n".join(lines)
