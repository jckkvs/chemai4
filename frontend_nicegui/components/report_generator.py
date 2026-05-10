"""
frontend_nicegui/components/report_generator.py

ワンクリックレポート生成：解析結果のMarkdown/HTMLレポートをダウンロード。

Implements: 会話 82f7fa3b — ワンクリックレポート生成

設計思想:
  - 解析完了後にボタン1つでレポートを生成
  - Markdown → HTML 変換も対応
  - データ概要、モデル比較、Feature Importance、OOFメトリクスを含む
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui

logger = logging.getLogger(__name__)


def render_report_tab(state: dict[str, Any]) -> None:
    """レポート生成タブを描画する。"""

    ar = state.get("automl_result")
    if ar is None:
        with ui.card().classes("glass-card q-pa-xl full-width"):
            ui.icon("description", color="grey-7", size="xl").classes("q-mb-md")
            ui.label("レポート生成にはモデル学習が必要です").classes("text-h6 text-grey-5")
            ui.label(
                "「📂 解析設定」→「🚀 解析開始」でモデルを学習してください。"
            ).classes("text-grey-6 q-mt-sm")
        return

    # ── ヘッダー ──
    with ui.row().classes("items-center q-gutter-sm full-width"):
        ui.icon("summarize").classes("text-teal text-h5")
        ui.label("解析レポート生成").classes("text-h6")

    ui.label(
        "解析結果をMarkdown / HTMLレポートとしてダウンロードできます。"
    ).classes("text-caption text-grey q-mb-md")

    # ── レポートプレビューコンテナ ──
    preview_container = ui.column().classes("full-width q-mt-md")

    # ── 生成＆ダウンロードボタン ──
    with ui.row().classes("q-gutter-md"):
        def _gen_md():
            md = _generate_markdown_report(ar, state)
            _show_preview(md, preview_container)
            ui.download(
                md.encode("utf-8"),
                f"chemai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            )
            ui.notify("📝 Markdownレポートをダウンロード", type="positive")

        def _gen_html():
            md = _generate_markdown_report(ar, state)
            html = _markdown_to_html(md)
            _show_preview(md, preview_container)
            ui.download(
                html.encode("utf-8"),
                f"chemai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            )
            ui.notify("🌐 HTMLレポートをダウンロード", type="positive")

        ui.button("📝 Markdownレポート", on_click=_gen_md).props(
            "color=teal unelevated no-caps icon=description"
        ).classes("btn-primary")
        ui.button("🌐 HTMLレポート", on_click=_gen_html).props(
            "outline color=teal no-caps icon=html"
        )

    # ── レポート内容のカスタマイズ ──
    with ui.expansion("⚙️ レポート設定", icon="settings").classes("full-width q-mt-sm"):
        ui.label("レポートに含める項目を選択できます（現在は全項目が含まれます）。").classes(
            "text-caption text-grey"
        )
        with ui.column().classes("q-gutter-xs"):
            for label in [
                "📊 データ概要", "🏆 モデルスコア比較", "📈 Feature Importance Top 20",
                "📈 OOFメトリクス", "⚙️ パイプライン設定", "📅 生成日時"
            ]:
                ui.checkbox(label, value=True).props("dense color=teal")


def _generate_markdown_report(ar, state: dict) -> str:
    """解析結果からMarkdownレポートを生成する。"""
    lines: list[str] = []

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = state.get("filename", "不明")
    target_col = state.get("target_col", "不明")
    task_type = ar.task if hasattr(ar, "task") else state.get("task_type", "不明")

    # ── タイトル ──
    lines.append("# ChemAI ML Studio — 解析レポート\n")
    lines.append(f"**生成日時**: {now}  ")
    lines.append(f"**データソース**: {filename}  ")
    lines.append(f"**目的変数**: {target_col}  ")
    lines.append(f"**タスク**: {task_type}  ")
    lines.append(f"**スコアリング**: {ar.scoring}  \n")

    # ── データ概要 ──
    lines.append("## 📊 データ概要\n")
    df = state.get("df")
    if df is not None:
        lines.append(f"- **行数**: {df.shape[0]:,}")
        lines.append(f"- **列数**: {df.shape[1]}")
        lines.append(f"- **欠損率**: {df.isna().mean().mean():.1%}")
        n_num = df.select_dtypes(include="number").shape[1]
        lines.append(f"- **数値列**: {n_num}")
        smiles_col = state.get("smiles_col", "")
        if smiles_col:
            lines.append(f"- **SMILES列**: {smiles_col}")
        precalc = state.get("precalc_df")
        if precalc is not None:
            lines.append(f"- **SMILES記述子数**: {precalc.shape[1]}")
    lines.append("")

    # ── 最良モデル ──
    lines.append("## 🏆 最良モデル\n")
    lines.append(f"**{ar.best_model_key}** — スコア: **{ar.best_score:.4f}**  ")
    lines.append(f"所要時間: {ar.elapsed_seconds:.1f}秒\n")

    # ── モデルスコア比較テーブル ──
    lines.append("## 📈 モデルスコア比較\n")
    lines.append("| モデル | 平均スコア | 標準偏差 | 学習時間(秒) |")
    lines.append("|--------|-----------|---------|-------------|")
    for key, score in sorted(ar.model_scores.items(), key=lambda x: -x[1]):
        detail = ar.model_details.get(key, {})
        std = detail.get("std", 0)
        fit_time = detail.get("fit_time", 0)
        best_mark = " 🏆" if key == ar.best_model_key else ""
        lines.append(f"| {key}{best_mark} | {score:.4f} | ±{std:.4f} | {fit_time:.2f} |")
    lines.append("")

    # ── Fold別スコア ──
    lines.append("## 📊 Fold別スコア\n")
    for key, detail in ar.model_details.items():
        fold_scores = detail.get("fold_scores", [])
        if fold_scores:
            fold_text = ", ".join(f"{s:.4f}" for s in fold_scores)
            best_mark = " 🏆" if key == ar.best_model_key else ""
            lines.append(f"- **{key}{best_mark}**: [{fold_text}]")
    lines.append("")

    # ── OOFメトリクス ──
    if ar.oof_predictions is not None and ar.oof_true is not None:
        lines.append("## 📈 Out-of-Fold メトリクス\n")
        try:
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            if task_type == "regression":
                r2 = r2_score(ar.oof_true, ar.oof_predictions)
                rmse = np.sqrt(mean_squared_error(ar.oof_true, ar.oof_predictions))
                mae = mean_absolute_error(ar.oof_true, ar.oof_predictions)
                lines.append(f"- **R² (OOF)**: {r2:.4f}")
                lines.append(f"- **RMSE (OOF)**: {rmse:.4f}")
                lines.append(f"- **MAE (OOF)**: {mae:.4f}")
            else:
                from sklearn.metrics import accuracy_score, f1_score
                acc = accuracy_score(ar.oof_true, ar.oof_predictions)
                f1 = f1_score(ar.oof_true, ar.oof_predictions, average="weighted", zero_division=0)
                lines.append(f"- **Accuracy (OOF)**: {acc:.4f}")
                lines.append(f"- **F1-weighted (OOF)**: {f1:.4f}")
        except Exception as ex:
            lines.append(f"*OOFメトリクス計算エラー: {ex}*")
        lines.append("")

    # ── Feature Importance ──
    try:
        estimator = ar.best_pipeline
        if hasattr(estimator, "steps"):
            estimator = estimator.steps[-1][1]
            if hasattr(estimator, "steps"):
                estimator = estimator.steps[-1][1]

        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
            try:
                feat_names = ar.best_pipeline[:-1].get_feature_names_out().tolist()
            except Exception:
                X = getattr(ar, "X_train", None)
                feat_names = list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(len(importances))]

            if len(feat_names) != len(importances):
                feat_names = [f"f{i}" for i in range(len(importances))]

            indices = np.argsort(importances)[::-1]
            top_n = min(20, len(indices))

            lines.append("## 📊 Feature Importance (Top 20)\n")
            lines.append("| 順位 | 特徴量 | 重要度 |")
            lines.append("|------|--------|--------|")
            for i in range(top_n):
                idx = indices[i]
                name = feat_names[idx] if idx < len(feat_names) else f"f{idx}"
                lines.append(f"| {i+1} | {name} | {importances[idx]:.4f} |")
            lines.append("")
    except Exception:
        pass

    # ── 警告 ──
    if ar.warnings:
        lines.append("## ⚠️ 警告\n")
        for w in ar.warnings:
            lines.append(f"- {w}")
        lines.append("")

    # ── フッター ──
    lines.append("---\n")
    lines.append(f"*Generated by ChemAI ML Studio v2.0 — {now}*\n")

    return "\n".join(lines)


def _markdown_to_html(md_text: str) -> str:
    """MarkdownテキストをHTMLに変換する。"""
    # markdownライブラリがあれば使用、なければ最低限のHTML wrapping
    try:
        import markdown
        body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    except ImportError:
        # 最低限のHTML化
        body = md_text.replace("\n", "<br>\n")

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChemAI ML Studio — 解析レポート</title>
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            background: #0d0d1a;
            color: #e0e0f0;
            line-height: 1.6;
        }}
        h1 {{
            background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff6b9d);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        h2 {{ color: #00d4ff; border-bottom: 1px solid rgba(255,255,255,0.12); padding-bottom: 8px; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 12px 0;
        }}
        th, td {{
            border: 1px solid rgba(255,255,255,0.12);
            padding: 8px 12px;
            text-align: left;
        }}
        th {{ background: rgba(255,255,255,0.05); color: #00d4ff; }}
        code {{ background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 4px; }}
        a {{ color: #00d4ff; }}
        hr {{ border-color: rgba(255,255,255,0.12); }}
    </style>
</head>
<body>
{body}
</body>
</html>"""


def _show_preview(md_text: str, container) -> None:
    """レポートプレビューを表示する。"""
    container.clear()
    with container:
        with ui.card().classes("glass-card q-pa-md full-width"):
            ui.label("📋 レポートプレビュー").classes("text-subtitle2 q-mb-sm")
            # 先頭80行だけ表示
            preview_lines = md_text.split("\n")[:80]
            preview = "\n".join(preview_lines)
            if len(md_text.split("\n")) > 80:
                preview += "\n\n... (以下省略)"
            ui.markdown(preview).classes("full-width")
