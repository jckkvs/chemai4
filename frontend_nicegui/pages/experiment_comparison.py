"""
frontend_nicegui/pages/experiment_comparison.py

実験比較ダッシュボード。
backend.session.version_manager の SQLite DBから
過去の実験を取得し、Plotly でスコア・指標を比較する。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from nicegui import ui

logger = logging.getLogger(__name__)


def _get_manager():
    """VersionManager のシングルトンを返す。"""
    from backend.session.version_manager import VersionManager
    return VersionManager()


def render_experiment_comparison(state: dict[str, Any]) -> None:
    """実験比較ダッシュボードを描画する。"""

    container = ui.column().classes("full-width q-pa-md q-gutter-md")

    def _rebuild():
        container.clear()
        with container:
            _render_dashboard(state)

    _rebuild()
    state["_refresh_experiment_comparison"] = _rebuild


def _render_dashboard(state: dict[str, Any]) -> None:

    # ── ヘッダー ──
    with ui.card().classes("full-width q-pa-md").style(
        "background: linear-gradient(135deg, rgba(123,47,247,0.08), rgba(0,212,255,0.05));"
        "border: 1px solid rgba(123,47,247,0.25); border-radius: 12px;"
    ):
        with ui.row().classes("items-center q-gutter-sm"):
            ui.html('<span style="font-size:28px;">🔬</span>')
            ui.label("実験比較ダッシュボード").style(
                "font-size: 20px; font-weight: 800; "
                "background: linear-gradient(90deg, #7b2ff7, #00d4ff); "
                "-webkit-background-clip: text; -webkit-text-fill-color: transparent;"
            )

        ui.label("過去の実験結果を選択してスコア・指標・ハイパーパラメータを比較します。").classes(
            "text-caption text-grey-5"
        )

    # ── 実験一覧の取得 ──
    try:
        manager = _get_manager()
        all_exps = manager.list_experiments(limit=100)
    except Exception as ex:
        ui.label(f"❌ DB接続エラー: {ex}").classes("text-red")
        return

    if not all_exps:
        with ui.card().classes("full-width q-pa-xl text-center").style(
            "border: 1px dashed rgba(255,255,255,0.15); border-radius: 10px;"
        ):
            ui.html('<span style="font-size:48px; opacity:0.3;">📋</span>')
            ui.label("実験履歴がありません").classes("text-h6 text-grey-5 q-mt-sm")
            ui.label(
                "解析を実行すると自動的に実験が記録されます。"
            ).classes("text-caption text-grey-6")
        return

    # ── 実験選択チェックボックス ──
    selected_hashes: set[str] = set()

    with ui.card().classes("full-width q-pa-md glass-card"):
        with ui.row().classes("items-center justify-between q-mb-sm"):
            ui.label(f"📋 実験履歴 ({len(all_exps)}件)").classes("text-subtitle1 text-bold")
            with ui.row().classes("q-gutter-xs"):
                def _select_all():
                    for exp in all_exps:
                        selected_hashes.add(exp["exp_hash"])
                    _redraw_comparison()

                def _deselect_all():
                    selected_hashes.clear()
                    _redraw_comparison()

                ui.button("全選択", on_click=_select_all).props("flat dense size=sm no-caps")
                ui.button("全解除", on_click=_deselect_all).props("flat dense size=sm no-caps")

        columns = [
            {"name": "sel",        "label": "選択",       "field": "sel",        "align": "center", "sortable": False},
            {"name": "created_at", "label": "日時",       "field": "created_at", "align": "left",   "sortable": True},
            {"name": "exp_name",   "label": "実験名",     "field": "exp_name",   "align": "left",   "sortable": True},
            {"name": "task_type",  "label": "タスク",     "field": "task_type",  "align": "center", "sortable": True},
            {"name": "best_model", "label": "最良モデル", "field": "best_model", "align": "left",   "sortable": True},
            {"name": "best_score", "label": "スコア",     "field": "best_score", "align": "center", "sortable": True},
            {"name": "metrics",    "label": "R²/Acc",    "field": "metrics",    "align": "center", "sortable": False},
            {"name": "n_samples",  "label": "N",          "field": "n_samples",  "align": "center", "sortable": True},
            {"name": "elapsed",    "label": "時間(秒)",   "field": "elapsed",    "align": "center", "sortable": True},
            {"name": "action",     "label": "操作",       "field": "action",     "align": "center", "sortable": False},
        ]

        rows = []
        for exp in all_exps:
            metrics_dict = json.loads(exp.get("metrics_json") or "{}")
            r2_or_acc = metrics_dict.get("R2", metrics_dict.get("Accuracy", "—"))
            if isinstance(r2_or_acc, float):
                r2_or_acc = f"{r2_or_acc:.4f}"
            rows.append({
                "exp_hash":  exp["exp_hash"],
                "sel":       "☑" if exp["exp_hash"] in selected_hashes else "☐",
                "created_at": exp["created_at"][:16].replace("T", " "),
                "exp_name":  exp["exp_name"],
                "task_type": exp["task_type"],
                "best_model": exp["best_model_key"],
                "best_score": f"{exp['best_score']:.4f}",
                "metrics":   str(r2_or_acc),
                "n_samples": exp.get("n_samples", "—"),
                "elapsed":   f"{exp.get('elapsed_seconds', 0):.1f}",
                "action":    "🗑️",
            })

        table = ui.table(columns=columns, rows=rows).classes("full-width").props(
            "dense flat bordered"
        )

        def _on_row_click(e):
            row = e.args.get("row", {})
            exp_hash = row.get("exp_hash", "")
            if exp_hash in selected_hashes:
                selected_hashes.discard(exp_hash)
            else:
                selected_hashes.add(exp_hash)
            _redraw_comparison()

        table.on("rowClick", _on_row_click)

    # ── 比較チャートエリア ──
    comparison_area = ui.column().classes("full-width")

    def _redraw_comparison():
        comparison_area.clear()
        sel = [e for e in all_exps if e["exp_hash"] in selected_hashes]
        if len(sel) < 1:
            with comparison_area:
                ui.label("ℹ️ 実験を1件以上選択してください").classes("text-grey-5 text-caption")
            return
        with comparison_area:
            _render_comparison_charts(sel)

    _redraw_comparison()


def _render_comparison_charts(exps: list[dict]) -> None:
    """選択された実験のスコア比較チャートを描画する。"""
    import plotly.graph_objects as go

    names = [e["exp_name"][:20] for e in exps]
    scores = [e["best_score"] for e in exps]
    models = [e["best_model_key"] for e in exps]
    elapsed = [e.get("elapsed_seconds", 0) for e in exps]

    # カラーパレット
    palette = ["#00d4ff", "#7b2ff7", "#4ade80", "#fb923c", "#f472b6", "#facc15"]
    colors = [palette[i % len(palette)] for i in range(len(exps))]

    with ui.card().classes("full-width q-pa-md glass-card"):
        ui.label(f"📊 スコア比較 ({len(exps)}件)").classes("text-subtitle1 text-bold q-mb-sm")
        with ui.row().classes("full-width q-gutter-md"):

            # ── スコア棒グラフ ──
            fig_bar = go.Figure(go.Bar(
                x=names,
                y=scores,
                marker_color=colors,
                text=[f"{s:.4f}" for s in scores],
                textposition="outside",
            ))
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                margin=dict(l=10, r=10, t=30, b=60),
                yaxis_title="スコア",
                xaxis_tickangle=-30,
                title="最良スコア比較",
            )
            with ui.card().style("flex:1; min-width:300px; background:rgba(0,0,0,0.2); border-radius:10px;"):
                ui.plotly(fig_bar).classes("full-width")

            # ── 所要時間棒グラフ ──
            fig_time = go.Figure(go.Bar(
                x=names,
                y=elapsed,
                marker_color=[c + "99" for c in colors],
                text=[f"{t:.1f}s" for t in elapsed],
                textposition="outside",
            ))
            fig_time.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                margin=dict(l=10, r=10, t=30, b=60),
                yaxis_title="秒",
                xaxis_tickangle=-30,
                title="解析所要時間",
            )
            with ui.card().style("flex:1; min-width:300px; background:rgba(0,0,0,0.2); border-radius:10px;"):
                ui.plotly(fig_time).classes("full-width")

    # ── 詳細指標テーブル ──
    with ui.card().classes("full-width q-pa-md glass-card"):
        ui.label("📋 詳細指標比較").classes("text-subtitle1 text-bold q-mb-sm")

        # 全実験のメトリクスキーを収集
        all_metric_keys: list[str] = []
        for exp in exps:
            m = json.loads(exp.get("metrics_json") or "{}")
            for k in m:
                if k not in all_metric_keys:
                    all_metric_keys.append(k)

        cols = [
            {"name": "exp_name",    "label": "実験名",     "field": "exp_name",   "align": "left"},
            {"name": "best_model",  "label": "最良モデル", "field": "best_model", "align": "left"},
            {"name": "best_score",  "label": "CVスコア",   "field": "best_score", "align": "center"},
        ] + [
            {"name": k, "label": k, "field": k, "align": "center"}
            for k in all_metric_keys
        ] + [
            {"name": "cv_folds",    "label": "CV Fold",   "field": "cv_folds",   "align": "center"},
            {"name": "n_samples",   "label": "N",         "field": "n_samples",  "align": "center"},
            {"name": "elapsed",     "label": "時間(秒)",  "field": "elapsed",    "align": "center"},
        ]

        detail_rows = []
        for exp in exps:
            m = json.loads(exp.get("metrics_json") or "{}")
            row = {
                "exp_name":   exp["exp_name"][:30],
                "best_model": exp["best_model_key"],
                "best_score": f"{exp['best_score']:.4f}",
                "cv_folds":   exp.get("cv_folds", "—"),
                "n_samples":  exp.get("n_samples", "—"),
                "elapsed":    f"{exp.get('elapsed_seconds', 0):.1f}",
            }
            for k in all_metric_keys:
                v = m.get(k, "—")
                row[k] = f"{v:.4f}" if isinstance(v, float) else str(v)
            detail_rows.append(row)

        ui.table(columns=cols, rows=detail_rows).classes("full-width").props(
            "dense flat bordered"
        )

    # ── ハイパーパラメータ比較 ──
    with ui.expansion("⚙️ ハイパーパラメータ比較", icon="tune").classes("full-width glass-card"):
        all_param_keys: list[str] = []
        for exp in exps:
            p = json.loads(exp.get("hyperparams_json") or "{}")
            for k in p:
                if k not in all_param_keys:
                    all_param_keys.append(k)

        if not all_param_keys:
            ui.label("ハイパーパラメータ情報がありません").classes("text-grey text-caption")
        else:
            param_cols = [
                {"name": "exp_name", "label": "実験名", "field": "exp_name", "align": "left"},
            ] + [
                {"name": k, "label": k, "field": k, "align": "center"}
                for k in all_param_keys[:15]
            ]
            param_rows = []
            for exp in exps:
                p = json.loads(exp.get("hyperparams_json") or "{}")
                row = {"exp_name": exp["exp_name"][:20]}
                for k in all_param_keys[:15]:
                    row[k] = str(p.get(k, "—"))
                param_rows.append(row)
            ui.table(columns=param_cols, rows=param_rows).classes("full-width").props(
                "dense flat bordered"
            )

    # ── 前処理設定比較 ──
    with ui.expansion("🔄 前処理設定比較", icon="settings").classes("full-width glass-card"):
        prep_cols = [
            {"name": "exp_name",   "label": "実験名",    "field": "exp_name",   "align": "left"},
            {"name": "scaler",     "label": "スケーラー", "field": "scaler",     "align": "center"},
            {"name": "imputer",    "label": "欠損補完",  "field": "imputer",    "align": "center"},
            {"name": "encoder",    "label": "エンコーダ", "field": "encoder",    "align": "center"},
            {"name": "selector",   "label": "特徴選択",  "field": "selector",   "align": "center"},
            {"name": "cv_strategy","label": "CV戦略",    "field": "cv_strategy","align": "center"},
        ]
        prep_rows = []
        for exp in exps:
            prep = json.loads(exp.get("preprocess_json") or "{}")
            prep_rows.append({
                "exp_name":    exp["exp_name"][:20],
                "scaler":      prep.get("num_scaler", "—"),
                "imputer":     prep.get("num_imputer", "—"),
                "encoder":     prep.get("cat_encoder", "—"),
                "selector":    prep.get("feature_selector", "—"),
                "cv_strategy": exp.get("cv_strategy", "—"),
            })
        ui.table(columns=prep_cols, rows=prep_rows).classes("full-width").props(
            "dense flat bordered"
        )
