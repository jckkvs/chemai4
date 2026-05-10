"""
frontend_nicegui/components/post_analysis_config.py

解析後の自動処理設定 — 順解析完了後に実行するタスクを事前に設定する。

Implements: F-3-1 | 順解析時に逆解析設定も同時に可能にする
設計:
  - パイプライン設定タブの末尾に配置
  - 逆解析の目標・手法・パラメータを事前設定
  - 順解析完了後に自動で逆解析を実行（事前設定済みの場合のみ）
  - レポート自動生成・SHAP自動実行も設定可能
"""
from __future__ import annotations

import logging
from typing import Any

from nicegui import ui

logger = logging.getLogger(__name__)


def render_post_analysis_config(state: dict[str, Any]) -> None:
    """解析後の自動処理設定UIを描画する。"""

    # ── 設定の初期化 ──
    if "_post_analysis" not in state:
        state["_post_analysis"] = {
            "auto_inverse": False,
            "auto_report": False,
            "auto_shap": False,
            # 逆解析プリセット
            "inv_target_mode": "maximize",
            "inv_target_min": None,
            "inv_target_max": None,
            "inv_method": "random",
            "inv_method_params": {},
            # 逆解析制約（自動範囲）
            "inv_auto_constraints": True,
            "inv_constraint_expand": 0.2,  # ±20%
        }
    pa = state["_post_analysis"]

    with ui.expansion(
        "🔮 解析後の自動処理",
        icon="auto_awesome",
    ).classes("full-width q-mt-sm").style(
        "border: 1px solid rgba(244, 114, 182, 0.3); border-radius: 10px;"
        "background: rgba(40, 10, 30, 0.2);"
    ):
        ui.label(
            "順解析完了後に自動で実行するタスクを事前に設定できます。"
        ).classes("text-caption text-grey q-mb-sm")

        # ═══════════════════════════════════════════════
        # セクション1: 逆解析の自動実行
        # ═══════════════════════════════════════════════
        with ui.card().classes("full-width q-pa-md q-mb-sm").style(
            "border: 1px solid rgba(123, 47, 247, 0.25); border-radius: 8px;"
            "background: rgba(20, 10, 40, 0.3);"
        ):
            with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
                auto_inv_cb = ui.checkbox(
                    "🔮 順解析完了後に逆解析を自動実行",
                    value=pa.get("auto_inverse", False),
                    on_change=lambda e: pa.update({"auto_inverse": e.value}),
                ).props("dense color=purple")

                if pa.get("auto_inverse"):
                    ui.badge("有効", color="purple").props("outline dense")

            # 逆解析設定（有効時のみ表示）
            if pa.get("auto_inverse"):
                target_col = state.get("target_col", "目的変数")

                # ── 目標モード ──
                with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
                    ui.label("目標:").classes("text-body2 text-bold")
                    ui.toggle(
                        {
                            "maximize": "📈 最大化",
                            "minimize": "📉 最小化",
                            "range": "📏 範囲指定",
                        },
                        value=pa.get("inv_target_mode", "maximize"),
                        on_change=lambda e: pa.update({"inv_target_mode": e.value}),
                    ).props("dense no-caps color=purple size=sm")

                # 範囲指定の場合のみ入力欄
                if pa.get("inv_target_mode") == "range":
                    with ui.row().classes("q-gutter-md q-mb-sm"):
                        ui.number(
                            f"{target_col} 最小値",
                            value=pa.get("inv_target_min"),
                            on_change=lambda e: pa.update({"inv_target_min": e.value}),
                        ).props("dense outlined").classes("w-40")
                        ui.number(
                            f"{target_col} 最大値",
                            value=pa.get("inv_target_max"),
                            on_change=lambda e: pa.update({"inv_target_max": e.value}),
                        ).props("dense outlined").classes("w-40")

                    # データ範囲のヒント
                    df = state.get("df")
                    if df is not None and target_col in df.columns:
                        import pandas as pd
                        if pd.api.types.is_numeric_dtype(df[target_col]):
                            col_data = df[target_col].dropna()
                            ui.label(
                                f"データ範囲: {col_data.min():.4g} ～ {col_data.max():.4g}"
                            ).classes("text-caption text-grey")

                # ── 最適化手法 ──
                with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
                    ui.label("手法:").classes("text-body2 text-bold")
                    method_options = {
                        "random": "🎲 ランダム",
                        "bayesian": "🧠 ベイズ",
                        "ga": "🧬 GA",
                        "dirichlet": "🎯 ディリクレ（組成系）",
                    }
                    ui.select(
                        method_options,
                        value=pa.get("inv_method", "random"),
                        on_change=lambda e: pa.update({"inv_method": e.value}),
                    ).props("dense outlined").style("min-width: 200px;")

                # ── 手法パラメータ（簡易） ──
                method = pa.get("inv_method", "random")
                with ui.row().classes("q-gutter-sm q-mb-sm"):
                    if method == "random":
                        ui.number(
                            "サンプル数",
                            value=pa.get("inv_method_params", {}).get("n_samples", 1000),
                            min=100, max=100000,
                            on_change=lambda e: pa["inv_method_params"].update({"n_samples": int(e.value)}),
                        ).props("dense outlined").style("width: 130px;")
                    elif method == "bayesian":
                        ui.number(
                            "試行回数",
                            value=pa.get("inv_method_params", {}).get("n_trials", 100),
                            min=10, max=10000,
                            on_change=lambda e: pa["inv_method_params"].update({"n_trials": int(e.value)}),
                        ).props("dense outlined").style("width: 130px;")
                        ui.select(
                            {"EI": "Expected Improvement", "PI": "Prob. Improvement", "UCB": "UCB"},
                            value=pa.get("inv_method_params", {}).get("acq_func", "EI"),
                            label="獲得関数",
                            on_change=lambda e: pa["inv_method_params"].update({"acq_func": e.value}),
                        ).props("dense outlined").style("width: 180px;")
                    elif method == "ga":
                        ui.number(
                            "個体数",
                            value=pa.get("inv_method_params", {}).get("pop_size", 50),
                            min=10, max=500,
                            on_change=lambda e: pa["inv_method_params"].update({"pop_size": int(e.value)}),
                        ).props("dense outlined").style("width: 120px;")
                        ui.number(
                            "世代数",
                            value=pa.get("inv_method_params", {}).get("n_generations", 100),
                            min=10, max=1000,
                            on_change=lambda e: pa["inv_method_params"].update({"n_generations": int(e.value)}),
                        ).props("dense outlined").style("width: 120px;")
                    elif method == "dirichlet":
                        ui.number(
                            "ラウンド数",
                            value=pa.get("inv_method_params", {}).get("n_rounds", 20),
                            min=3, max=100,
                            on_change=lambda e: pa["inv_method_params"].update({"n_rounds": int(e.value)}),
                        ).props("dense outlined").style("width: 120px;")
                        ui.number(
                            "合計値",
                            value=pa.get("inv_method_params", {}).get("total_sum", 1.0),
                            min=0.01, max=1000.0, step=0.01,
                            on_change=lambda e: pa["inv_method_params"].update({"total_sum": float(e.value)}),
                        ).props("dense outlined").style("width: 120px;")

                # ── 制約の自動設定 ──
                with ui.row().classes("items-center q-gutter-sm"):
                    ui.checkbox(
                        "制約をデータ範囲から自動設定",
                        value=pa.get("inv_auto_constraints", True),
                        on_change=lambda e: pa.update({"inv_auto_constraints": e.value}),
                    ).props("dense")

                    if pa.get("inv_auto_constraints"):
                        ui.number(
                            "拡張幅 (±%)",
                            value=pa.get("inv_constraint_expand", 0.2) * 100,
                            min=0, max=100, step=5,
                            on_change=lambda e: pa.update({"inv_constraint_expand": e.value / 100}),
                        ).props("dense outlined").style("width: 100px;")

                ui.label(
                    "💡 逆解析の詳細な制約は「🔮 逆解析」タブで設定できます。"
                    "ここでは順解析後の自動実行用に簡易設定ができます。"
                ).classes("text-caption text-grey q-mt-xs")

        # ═══════════════════════════════════════════════
        # セクション2: その他の自動処理
        # ═══════════════════════════════════════════════
        with ui.row().classes("q-gutter-md q-mb-sm"):
            ui.checkbox(
                "📝 レポート自動生成",
                value=pa.get("auto_report", False),
                on_change=lambda e: pa.update({"auto_report": e.value}),
            ).props("dense").tooltip(
                "順解析完了後にレポートを自動生成します"
            )

            ui.checkbox(
                "🔬 SHAP解析を自動実行",
                value=pa.get("auto_shap", False),
                on_change=lambda e: pa.update({"auto_shap": e.value}),
            ).props("dense").tooltip(
                "順解析完了後にSHAP解析を自動実行します（時間がかかる場合があります）"
            )

        # ── サマリー ──
        active_tasks = []
        if pa.get("auto_inverse"):
            mode_label = {"maximize": "最大化", "minimize": "最小化", "range": "範囲指定"}.get(
                pa.get("inv_target_mode", "maximize"), "最大化"
            )
            method_label = {
                "random": "ランダム", "bayesian": "ベイズ",
                "ga": "GA", "dirichlet": "ディリクレ",
            }.get(pa.get("inv_method", "random"), "ランダム")
            active_tasks.append(f"🔮 逆解析（{mode_label} / {method_label}）")
        if pa.get("auto_report"):
            active_tasks.append("📝 レポート生成")
        if pa.get("auto_shap"):
            active_tasks.append("🔬 SHAP解析")

        if active_tasks:
            with ui.row().classes("items-center q-gutter-xs"):
                ui.icon("check_circle", color="green").classes("text-body2")
                ui.label(
                    f"解析完了後に自動実行: {' / '.join(active_tasks)}"
                ).classes("text-caption text-green")
        else:
            with ui.row().classes("items-center q-gutter-xs"):
                ui.icon("info_outline", color="grey").classes("text-caption")
                ui.label(
                    "解析後の自動処理は設定されていません"
                ).classes("text-caption text-grey")
