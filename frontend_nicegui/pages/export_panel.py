"""
frontend_nicegui/pages/export_panel.py

解析レポートのエクスポートUIパネル。
PDF / Word / Jupyter Notebook / ZIP の4形式に対応し、
バックエンドの backend.export モジュールを直接呼び出す。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from nicegui import ui, run

logger = logging.getLogger(__name__)


# ================================================================
# 定型レポート生成（LLMなし）
# ================================================================

def _generate_structured_report(state: dict) -> dict:
    """
    解析レポートの定型レポート用の構造化データを生成する。
    LLMを使わずに、統計的指標と診断結果をまとめる。

    Returns:
        レポート構造化データ（dict）
    """
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_diagnostics": {},
        "data_quality": {},
        "feature_insights": {},
        "outlier_analysis": {},
        "practical_evaluation": {},
        "recommendations": [],
    }

    ar = state.get("automl_result")
    if ar is None:
        return report

    # 1. モデル診断
    report["model_diagnostics"] = _diagnose_model(ar, state)

    # 2. データ品質レポート
    report["data_quality"] = _analyze_data_quality(state)

    # 3. 特徴量洞察
    report["feature_insights"] = _analyze_feature_insights(ar, state)

    # 4. 外れ値解析
    report["outlier_analysis"] = _analyze_outliers(ar, state)

    # 5. 実用性評価
    report["practical_evaluation"] = _evaluate_practicality(ar, state)

    # 6. 推奨事項（リスト）
    report["recommendations"] = _generate_recommendations(report)

    return report


def _diagnose_model(ar, state: dict) -> dict:
    """モデルの過学習・安定性・学習傾向を診断する。"""
    import numpy as np

    diag = {
        "best_model": ar.best_model_key,
        "cv_score": ar.best_score,
        "task": getattr(ar, "task", "regression"),
        "overfitting": {"detected": False, "gap": None, "severity": "none"},
        "stability": {"cv_std": None, "coeff_var": None, "rating": "unknown", "fold_scores": []},
        "learning_curve": {"train_score": None, "cv_score": None, "gap": None},
    }

    y_true = getattr(ar, "oof_true", None)
    y_pred = getattr(ar, "oof_pred", None)

    if y_true is not None and y_pred is not None:
        y_t = np.asarray(y_true).ravel()
        y_p = np.asarray(y_pred).ravel()

        if diag["task"] == "regression":
            from sklearn.metrics import r2_score

            # CVスコア（OOFベース）
            try:
                cv_r2 = r2_score(y_t, y_p)
                diag["cv_metrics"] = {"R2": round(float(cv_r2), 4)}
            except Exception:
                pass

            # Trainスコアとの比較（過学習診断）
            y_train = getattr(ar, "y_train", None)
            y_train_pred = getattr(ar, "train_pred", None)
            if y_train is not None and y_train_pred is not None:
                try:
                    y_tr_t = np.asarray(y_train).ravel()
                    y_tr_p = np.asarray(y_train_pred).ravel()
                    train_r2 = r2_score(y_tr_t, y_tr_p)
                    gap = train_r2 - cv_r2
                    severity = "none"
                    if gap > 0.15:
                        severity = "high"
                    elif gap > 0.05:
                        severity = "medium"

                    diag["overfitting"] = {
                        "detected": gap > 0.05,
                        "gap": round(float(gap), 4),
                        "severity": severity,
                        "train_r2": round(float(train_r2), 4),
                        "cv_r2": round(float(cv_r2), 4),
                    }
                except Exception:
                    pass

        # Foldごとの安定性
        model_details = getattr(ar, "model_details", {})
        best_detail = model_details.get(ar.best_model_key, {})
        fold_scores = best_detail.get("cv_scores", [])
        if fold_scores and len(fold_scores) > 1:
            std = float(np.std(fold_scores))
            mean = float(np.mean(fold_scores))
            cv = std / abs(mean) if abs(mean) > 0 else float("inf")
            rating = "excellent" if cv < 0.05 else ("good" if cv < 0.1 else ("fair" if cv < 0.2 else "poor"))
            diag["stability"] = {
                "cv_std": round(std, 4),
                "coeff_var": round(float(cv), 4),
                "rating": rating,
                "fold_scores": [round(float(s), 4) for s in fold_scores],
                "n_folds": len(fold_scores),
            }

    return diag


def _analyze_data_quality(state: dict) -> dict:
    """データの品質を多角的に評価する。"""
    import numpy as np
    import pandas as pd

    quality = {
        "n_samples": None,
        "n_features_raw": None,
        "n_features_processed": None,
        "missing_rate": None,
        "leakage_detected": [],
        "multicollinearity": {"detected": False, "max_vif": None, "high_corr_pairs": []},
        "class_balance": None,
    }

    df = state.get("df")
    if df is None:
        return quality

    quality["n_samples"] = len(df)
    quality["n_features_raw"] = df.shape[1]

    # 欠損率
    missing_rate = df.isna().mean().mean()
    quality["missing_rate"] = round(float(missing_rate), 4)

    # ターゲットの欠損
    target_col = state.get("target_col")
    if target_col and target_col in df.columns:
        target_missing = df[target_col].isna().mean()
        quality["target_missing_rate"] = round(float(target_missing), 4)

        # クラスバランス（分類問題）
        if df[target_col].dtype == "object" or df[target_col].nunique() < 20:
            counts = df[target_col].value_counts()
            quality["class_balance"] = {
                "n_classes": len(counts),
                "min_class_size": int(counts.min()),
                "max_class_size": int(counts.max()),
                "imbalance_ratio": round(float(counts.max() / counts.min()), 2) if counts.min() > 0 else None,
            }

    # 前処理データの品質
    ar = state.get("automl_result")
    if ar:
        proc_X = getattr(ar, "processed_X", None)
        if proc_X is not None and hasattr(proc_X, "shape"):
            quality["n_features_processed"] = proc_X.shape[1]

            # 多重共線性（VIF計算 via 相関係数）
            try:
                numeric_df = proc_X.select_dtypes(include=["float64", "int64"])
                if numeric_df.shape[1] >= 2 and numeric_df.shape[1] <= 100:
                    corr_matrix = numeric_df.corr().abs()
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    high_corr = []
                    for col in upper.columns:
                        high = upper[col][upper[col] > 0.8]
                        for idx, val in high.items():
                            high_corr.append((col, idx, round(float(val), 3)))

                    if high_corr:
                        quality["multicollinearity"]["detected"] = True
                        quality["multicollinearity"]["high_corr_pairs"] = high_corr[:10]
            except Exception:
                pass

    # データリーク検出
    leakage_check = state.get("_leakage_check")
    if leakage_check:
        quality["leakage_detected"] = leakage_check.get("leakage_features", [])

    return quality


def _analyze_feature_insights(ar, state: dict) -> dict:
    """特徴量の重要度と相関性を解析する。"""
    import numpy as np

    insights = {
        "top_features": [],
        "feature_count": None,
        "interpretability": {"has_feature_importance": False, "has_shap": False, "model_type": None},
        "chemical_relevance": {"top_features_chemical": [], "note": ""},
    }

    model = getattr(ar, "best_pipeline", None)
    proc_X = getattr(ar, "processed_X", None)

    if model is None or proc_X is None:
        return insights

    # 特徴量名の取得
    try:
        if hasattr(proc_X, "columns"):
            feat_names = list(proc_X.columns)
        else:
            feat_names = [f"f{i}" for i in range(proc_X.shape[1])]
    except Exception:
        feat_names = []

    insights["feature_count"] = len(feat_names)

    # Feature Importance
    try:
        estimator = model
        if hasattr(model, "steps"):
            estimator = model.steps[-1][1]
            if hasattr(estimator, "steps"):
                estimator = estimator.steps[-1][1]

        if hasattr(estimator, "feature_importances_"):
            imp = estimator.feature_importances_
            indices = np.argsort(imp)[::-1]
            top_n = min(10, len(indices))
            for i in range(top_n):
                idx = indices[i]
                if idx < len(feat_names):
                    insights["top_features"].append({
                        "rank": i + 1,
                        "name": feat_names[idx],
                        "importance": round(float(imp[idx]), 4),
                    })
            insights["interpretability"]["has_feature_importance"] = True
            insights["interpretability"]["model_type"] = type(estimator).__name__
        elif hasattr(estimator, "coef_"):
            coefs = estimator.coef_.ravel()
            indices = np.argsort(np.abs(coefs))[::-1]
            top_n = min(10, len(indices))
            for i in range(top_n):
                idx = indices[i]
                if idx < len(feat_names):
                    insights["top_features"].append({
                        "rank": i + 1,
                        "name": feat_names[idx],
                        "coefficient": round(float(coefs[idx]), 4),
                        "abs_coefficient": round(float(abs(coefs[idx])), 4),
                    })
            insights["interpretability"]["has_feature_importance"] = True
            insights["interpretability"]["model_type"] = type(estimator).__name__
    except Exception as e:
        logger.debug(f"Feature importance取得エラー: {e}")

    # 化学的関連性（SMILES特徴量の判定）
    smiles_col = state.get("smiles_col")
    if smiles_col:
        for feat in insights["top_features"]:
            feat_name = feat["name"]
            if any(kw in feat_name.lower() for kw in ["mw", "mol", "weight", "mass"]):
                insights["chemical_relevance"]["top_features_chemical"].append(f"{feat_name} (分子量関連)")
            elif any(kw in feat_name.lower() for kw in ["logp", "lipo", "clogp"]):
                insights["chemical_relevance"]["top_features_chemical"].append(f"{feat_name} (脂溶性関連)")
            elif any(kw in feat_name.lower() for kw in ["hbd", "hba", "donor", "acceptor"]):
                insights["chemical_relevance"]["top_features_chemical"].append(f"{feat_name} (水素結合関連)")
            elif any(kw in feat_name.lower() for kw in ["rot", "ring", "aromatic"]):
                insights["chemical_relevance"]["top_features_chemical"].append(f"{feat_name} (構造特性関連)")

    return insights


def _analyze_outliers(ar, state: dict) -> dict:
    """外れ値の検出と解析を行う。"""
    import numpy as np

    analysis = {
        "n_outliers_residual": None,
        "n_outliers_leverage": None,
        "high_residual_samples": [],
        "residual_stats": {},
    }

    y_true = getattr(ar, "oof_true", None)
    y_pred = getattr(ar, "oof_pred", None)

    if y_true is None or y_pred is None:
        return analysis

    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    residuals = y_t - y_p

    # 残差の統計
    analysis["residual_stats"] = {
        "mean": round(float(np.mean(residuals)), 4),
        "std": round(float(np.std(residuals)), 4),
        "max_positive": round(float(np.max(residuals)), 4),
        "max_negative": round(float(np.min(residuals)), 4),
    }

    # 3シグマ外れ値
    std = np.std(residuals)
    mean = np.mean(residuals)
    outlier_mask = np.abs(residuals - mean) > 3 * std
    n_outliers = int(outlier_mask.sum())
    analysis["n_outliers_residual"] = n_outliers

    # 上位5件の高残差サンプル
    abs_residuals = np.abs(residuals)
    top_indices = np.argsort(abs_residuals)[::-1][:5]
    for idx in top_indices:
        analysis["high_residual_samples"].append({
            "index": int(idx),
            "true": round(float(y_t[idx]), 4),
            "predicted": round(float(y_p[idx]), 4),
            "residual": round(float(residuals[idx]), 4),
        })

    return analysis


def _evaluate_practicality(ar, state: dict) -> dict:
    """モデルの実用性を評価する。"""
    import numpy as np

    eval_result = {
        "prediction_speed": {"estimated_per_sample_ms": None, "rating": "unknown"},
        "uncertainty": {"has_interval": False, "note": "予測区間は未実装です"},
        "business_metrics": {},
        "deployment_readiness": {"ready": False, "missing": []},
    }

    # 予測速度（学習時スコアから概算）
    model_details = getattr(ar, "model_details", {})
    best_detail = model_details.get(ar.best_model_key, {})
    fit_time = best_detail.get("fit_time", 0)
    if fit_time and fit_time > 0:
        proc_X = getattr(ar, "processed_X", None)
        n_samples = proc_X.shape[0] if proc_X is not None else 1000
        ms_per_sample = (fit_time / n_samples) * 1000
        rating = "fast" if ms_per_sample < 1 else ("moderate" if ms_per_sample < 10 else "slow")
        eval_result["prediction_speed"] = {
            "estimated_per_sample_ms": round(float(ms_per_sample), 4),
            "rating": rating,
        }

    # ビジネス指標（R2から実用性を判定）
    if ar.task == "regression":
        r2 = ar.best_score
        if r2 >= 0.9:
            practical_rating = "excellent"
            note = "実用性は非常に高いです。重要な予測に使用可能。"
        elif r2 >= 0.7:
            practical_rating = "good"
            note = "実用性は良好です。多くの用途で使用可能。"
        elif r2 >= 0.5:
            practical_rating = "fair"
            note = "実用性は普通です。追加的な診断と改善が推奨されます。"
        else:
            practical_rating = "poor"
            note = "実用性は低いです。モデルの見直しが必要です。"

        eval_result["business_metrics"] = {
            "r2_practical_rating": practical_rating,
            "note": note,
        }

    # デプロイメント準備状況
    missing = []
    if not hasattr(ar, "best_pipeline"):
        missing.append("学習済みモデルがない")
    if not getattr(ar, "processed_X", None) is not None:
        missing.append("前処理済みデータの情報がない")

    eval_result["deployment_readiness"] = {
        "ready": len(missing) == 0,
        "missing": missing,
    }

    return eval_result


def _generate_recommendations(report: dict) -> list:
    """レポート内容から推奨事項を生成する（リスト）。"""
    recs = []

    # 過学習に関する推奨
    overfitting = report.get("model_diagnostics", {}).get("overfitting", {})
    if overfitting.get("detected"):
        severity = overfitting.get("severity", "none")
        if severity == "high":
            recs.append({
                "category": "過学習",
                "priority": "高",
                "message": "深刻な過学習が見つかりました（Train-CV差が0.15以上）。"
                          "正則化の強化、特徴量削減、またはよりシンプルなモデルの検討をお勧めします。",
            })
        elif severity == "medium":
            recs.append({
                "category": "過学習",
                "priority": "中",
                "message": "軽度の過学習が見つかりました。パラメータチューニングや"
                          "データ拡張を検討してください。",
            })

    # 安定性に関する推奨
    stability = report.get("model_diagnostics", {}).get("stability", {})
    if stability.get("rating") == "poor":
        recs.append({
            "category": "安定性",
            "priority": "高",
            "message": "Foldごとのスコアの変動が大きいです。データの偏りの影響、または"
                      "CV方法（Stratified K-Fold等）の見直しをお勧めします。",
        })

    # データ品質に関する推奨
    quality = report.get("data_quality", {})
    missing_rate = quality.get("missing_rate", 0)
    if missing_rate > 0.1:
        recs.append({
            "category": "データ品質",
            "priority": "中",
            "message": f"欠損率が{missing_rate:.1%}と高いです。欠損補完方法の見直しや"
                      "欠損の多い特徴量の除去をお勧めします。",
        })

    # 多重共線性に関する推奨
    multicol = quality.get("multicollinearity", {})
    if multicol.get("detected"):
        recs.append({
            "category": "多重共線性",
            "priority": "中",
            "message": "高い相関を持つ特徴量が見つかりました。"
                      "VIF算出や特徴量選択をお勧めします。",
        })

    # 実用性に関する推奨
    practical = report.get("practical_evaluation", {})
    business = practical.get("business_metrics", {})
    if business.get("r2_practical_rating") == "poor":
        recs.append({
            "category": "実用性",
            "priority": "高",
            "message": "モデルの実用性が低いです。特徴量やアルゴリズムの"
                      "見直し、またはより大量のデータ収集をお勧めします。",
        })

    # 外れ値に関する推奨
    outliers = report.get("outlier_analysis", {})
    n_outliers = outliers.get("n_outliers_residual") or 0
    if n_outliers > 10:
        recs.append({
            "category": "外れ値",
            "priority": "中",
            "message": f"残差ベースの外れ値が{n_outliers}件見つかりました。"
                      "これらのサンプルを詳細に確認し、データ品質や特異性の確認をお勧めします。",
        })

    return recs


def render_export_panel(state: dict[str, Any]) -> None:
    """エクスポートパネルを描画する。"""

    with ui.column().classes("full-width q-pa-md q-gutter-md"):

        # ── ヘッダー ──
        with ui.card().classes("full-width q-pa-md").style(
            "background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(123,47,247,0.08));"
            "border: 1px solid rgba(0,212,255,0.2); border-radius: 12px;"
        ):
            with ui.row().classes("items-center q-gutter-sm"):
                ui.html('<span style="font-size:28px;">📤</span>')
                ui.label("解析レポート エクスポート").style(
                    "font-size: 20px; font-weight: 800; "
                    "background: linear-gradient(90deg, #00d4ff, #7b2ff7); "
                    "-webkit-background-clip: text; -webkit-text-fill-color: transparent;"
                )
            ui.label(
                "解析が完了した結果をPDF、Word、Jupyter Notebook、またはチャートZIPとしてダウンロードします。"
            ).classes("text-caption text-grey-5 q-mt-xs")

        # ── 解析結果なしのガード ──
        ar = state.get("automl_result")
        if ar is None:
            with ui.card().classes("full-width q-pa-lg text-center").style(
                "border: 1px dashed rgba(255,255,255,0.15); border-radius: 10px;"
            ):
                ui.html('<span style="font-size:48px; opacity:0.3;">📊</span>')
                ui.label("解析結果がまだありません").classes("text-h6 text-grey-5 q-mt-sm")
                ui.label(
                    "「📂 解析設定」タブでデータを読み込み、解析開始ボタンを押してください。"
                ).classes("text-caption text-grey-6")
            return

        # ── 解析サマリー表示 ──
        with ui.card().classes("full-width q-pa-md glass-card"):
            with ui.row().classes("q-gutter-md items-center"):
                ui.html('<span style="font-size:20px;">🏆</span>')
                ui.label(f"最良モデル: {ar.best_model_key}").classes("text-subtitle1 text-bold")
                ui.badge(f"スコア: {ar.best_score:.4f}", color="cyan").props("dense")

            metrics: dict = {}
            if ar.oof_true is not None and ar.oof_predictions is not None:
                import numpy as np
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                y_t = np.asarray(ar.oof_true).ravel()
                y_p = np.asarray(ar.oof_predictions).ravel()
                if ar.task == "regression":
                    metrics = {
                        "R²":   round(float(r2_score(y_t, y_p)), 4),
                        "RMSE": round(float(mean_squared_error(y_t, y_p, squared=False)), 4),
                        "MAE":  round(float(mean_absolute_error(y_t, y_p)), 4),
                    }

            if metrics:
                with ui.row().classes("q-gutter-sm q-mt-xs"):
                    for k, v in metrics.items():
                        with ui.card().classes("q-pa-xs").style(
                            "background: rgba(0,212,255,0.06); border-radius:6px; min-width:80px;"
                        ):
                            ui.label(str(v)).classes("text-body2 text-bold text-cyan")
                            ui.label(k).classes("text-caption text-grey-5")

        # ── フォーマット選択 ──
        with ui.card().classes("full-width q-pa-md glass-card"):
            ui.label("📋 出力形式を選択").classes("text-subtitle2 q-mb-sm")

            format_val = {"v": "pdf"}

            with ui.row().classes("q-gutter-sm"):
                for fmt, icon, label, desc in [
                    ("pdf",   "📄", "PDF",              "ReportLab製の高品質PDFレポート"),
                    ("docx",  "📝", "Word (.docx)",     "編集可能なWordドキュメント"),
                    ("ipynb", "📓", "Jupyter Notebook", "実行可能な解析ノートブック"),
                    ("zip",   "🗜️", "チャートZIP",       "全チャート画像を一括ダウンロード"),
                    ("py",    "✅", "再現スクリプト",       "モデルを利用したスタンドアロン予測スクリプト"),
                ]:
                    is_sel = format_val["v"] == fmt

                    def _select(f=fmt):
                        format_val["v"] = f
                        _rebuild()

                    ui.button(
                        f"{icon} {label}",
                        on_click=_select,
                    ).style(
                        f"border: {'2px' if is_sel else '1px'} solid "
                        f"{'#00d4ff' if is_sel else 'rgba(255,255,255,0.15)'};"
                        f"background: {'rgba(0,212,255,0.1)' if is_sel else 'transparent'};"
                        f"color: {'#00d4ff' if is_sel else '#9ca3af'};"
                        "border-radius: 8px; padding: 8px 16px; font-size:13px; cursor:pointer;"
                    ).props("flat no-caps").tooltip(desc)

            ui.label(f"選択中: {format_val['v'].upper()}").classes("text-caption text-cyan q-mt-xs")

        # ── 詳細設定 ──
        with ui.expansion("⚙️ 詳細設定", icon="settings").classes("full-width glass-card"):
            include_importance = ui.checkbox("特徴量重要度を含める", value=True)
            include_charts     = ui.checkbox("解析チャート画像を含める", value=True)
            include_data_head  = ui.checkbox("データサンプル（先頭5行）を含める", value=False)
            exp_name_input     = ui.input(
                label="ファイル名（拡張子不要）",
                value=f"chemai_report_{datetime.now().strftime('%Y%m%d_%H%M')}",
                placeholder="report_filename",
            ).props("outlined dense").classes("full-width q-mt-sm")
            output_dir_input   = ui.input(
                label="保存先フォルダ",
                value="exports",
                placeholder="exports",
            ).props("outlined dense").classes("full-width q-mt-sm")

        # ── エクスポートボタン ──
        status_container = ui.column().classes("full-width")

        async def _do_export():
            fmt = format_val["v"]
            filename = exp_name_input.value.strip() or "chemai_report"
            output_dir = output_dir_input.value.strip() or "exports"

            status_container.clear()
            with status_container:
                prog = ui.linear_progress(value=0, show_value=False).props("color=cyan rounded")
                lbl  = ui.label(f"⏳ {fmt.upper()} を生成中...").classes("text-grey-5 text-caption")

            # 結果辞書の組み立て
            importances: dict = {}
            try:
                if include_importance.value:
                    estimator = ar.best_pipeline
                    if hasattr(ar.best_pipeline, "steps"):
                        estimator = ar.best_pipeline.steps[-1][1]
                        if hasattr(estimator, "steps"):
                            estimator = estimator.steps[-1][1]
                    if hasattr(estimator, "feature_importances_"):
                        import numpy as np
                        proc_X = getattr(ar, "processed_X", None)
                        names = (
                            list(proc_X.columns) if proc_X is not None and hasattr(proc_X, "columns")
                            else [f"f{i}" for i in range(len(estimator.feature_importances_))]
                        )
                        importances = dict(zip(
                            names[:len(estimator.feature_importances_)],
                            estimator.feature_importances_.tolist(),
                        ))
            except Exception:
                pass

            result_dict: dict[str, Any] = {
                "best_model_name": ar.best_model_key,
                "metrics": metrics,
                "feature_importances": importances if include_importance.value else {},
                "chart_paths": state.get("_chart_paths", []) if include_charts.value else [],
                "ai_commentary": state.get("_ai_commentary", ""),
                # Notebook 用追加情報
                "target_col":   state.get("target_col", "target"),
                "feature_cols": (
                    list(getattr(ar, "processed_X", None).columns)
                    if getattr(ar, "processed_X", None) is not None
                    and hasattr(getattr(ar, "processed_X", None), "columns")
                    else []
                ),
                "best_params":  (
                    ar.model_details.get(ar.best_model_key, {}).get("params", {})
                    if hasattr(ar, "model_details") else {}
                ),
                "cv_folds": getattr(ar, "cv_folds", 5),
            }

            prog.value = 0.3
            lbl.text = f"⏳ {fmt.upper()} を書き込み中..."
            def _run_export():
                from backend.export import PDFExporter, WordExporter, NotebookExporter, ChartBundleExporter
                if fmt == "pdf":
                    return PDFExporter(output_dir).export(result_dict, filename)
                elif fmt == "docx":
                    return WordExporter(output_dir).export(result_dict, filename)
                elif fmt == "ipynb":
                    return NotebookExporter(output_dir).export(result_dict, filename)
                elif fmt == "zip":
                    return ChartBundleExporter(output_dir).export(result_dict, filename)
                elif fmt == "py":
                    import os
                    from backend.export.reproducibility import generate_reproduction_script
                    import joblib
                    
                    out_dir = Path(output_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    
                    base_name = filename
                    if base_name.endswith(".py"):
                        base_name = base_name[:-3]
                        
                    py_path = out_dir / f"{base_name}.py"
                    model_path = out_dir / f"{base_name}_model.pkl"
                    
                    # 1. Dump best model pipeline
                    joblib.dump(ar.best_pipeline, model_path)
                    
                    # 2. Get columns
                    t_col = state.get("target_col", "target")
                    s_col = state.get("smiles_cols", [])
                    s_col_str = s_col[0] if isinstance(s_col, list) and len(s_col) > 0 else (s_col if isinstance(s_col, str) else None)
                    if s_col_str is None:
                        s_col_str = state.get("smiles_col")
                        if isinstance(s_col_str, list) and len(s_col_str) > 0:
                            s_col_str = s_col_str[0].get("smiles_col")

                    # 3. Generate reproduction script
                    script_content = generate_reproduction_script(
                        model_path=model_path.name,
                        data_path="your_data.csv",
                        target_col=t_col,
                        task=ar.task,
                        smiles_col=s_col_str,
                    )
                    
                    with open(py_path, "w", encoding="utf-8") as f:
                        f.write(script_content)
                    
                    return py_path
                else:
                    raise ValueError(f"未対応のフォーマット: {fmt}")

            try:
                out_path: Path = await run.io_bound(_run_export)
                prog.value = 1.0
                lbl.text = f"✅ {out_path.name} を生成しました"
                ui.notify(f"✅ エクスポート完了: {out_path.name}", type="positive", timeout=5000)

                # ダウンロードボタン
                status_container.clear()
                with status_container:
                    with ui.row().classes("items-center q-gutter-sm"):
                        ui.icon("check_circle", color="green")
                        ui.label(f"✅ {out_path.name}").classes("text-green text-bold")
                    ui.label(f"保存先: {out_path}").classes("text-caption text-grey-5")
                    ui.button(
                        f"📥 {out_path.name} をダウンロード",
                        on_click=lambda: ui.download(str(out_path)),
                    ).props("outline color=cyan no-caps").classes("q-mt-sm")

            except Exception as ex:
                prog.value = 0
                lbl.text = f"❌ エクスポートエラー: {ex}"
                ui.notify(f"❌ エクスポートエラー: {str(ex)[:200]}", type="negative", timeout=8000)
                logger.exception("エクスポートエラー")

        ui.button(
            "🚀 エクスポート実行",
            on_click=_do_export,
        ).style(
            "background: linear-gradient(135deg, #00d4ff, #7b2ff7);"
            "color: white; border-radius: 10px; font-weight: 800;"
            "font-size: 15px; padding: 12px 32px; width: 100%;"
            "box-shadow: 0 4px 20px rgba(0,212,255,0.3);"
        ).props("no-caps")

        status_container


    def _rebuild():
        """フォーマット切替時に全体を再描画する（現状はnotify のみ）。"""
        pass
