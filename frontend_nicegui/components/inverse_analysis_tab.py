"""
frontend_nicegui/components/inverse_analysis_tab.py

逆解析（Inverse Analysis）タブ — NiceGUI版

設計思想:
  - 順解析で得た学習済みモデルを使い、目的変数の目標値から
    最適な説明変数値を逆推定する
  - パターン1: 順解析完了後に逆解析
  - パターン2: パイプライン設定時に同時設定（将来対応）
  - 最適化手法はプラガブル（ランダム/グリッド/ベイズ/GA/MOLAI/将来追加）
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# 最適化手法の定義
# ═══════════════════════════════════════════════════════════
OPTIMIZATION_METHODS = [
    {
        "key": "random",
        "label": "🎲 ランダムサンプリング",
        "desc": "説明変数の制約範囲内からランダムに候補を生成。高速だが網羅性は低い。",
        "speed": "⚡高速",
        "params": [
            {"name": "n_samples", "label": "サンプル数", "type": "int", "default": 1000, "min": 100, "max": 100000},
            {"name": "seed", "label": "乱数シード", "type": "int", "default": 42, "min": 0, "max": 99999},
        ],
    },
    {
        "key": "grid",
        "label": "📐 グリッドサーチ",
        "desc": "各説明変数を等間隔に分割し全組み合わせを評価。低次元向き（3~5変数以下推奨）。",
        "speed": "🟡中速",
        "params": [
            {"name": "n_points", "label": "変数あたりの分割数", "type": "int", "default": 10, "min": 3, "max": 50},
        ],
    },
    {
        "key": "bayesian",
        "label": "🧠 ベイズ最適化",
        "desc": "ガウス過程回帰で探索を効率化。高次元でも効果的。",
        "speed": "🟡中速",
        "params": [
            {"name": "n_trials", "label": "試行回数", "type": "int", "default": 100, "min": 10, "max": 10000},
            {"name": "seed", "label": "乱数シード", "type": "int", "default": 42, "min": 0, "max": 99999},
            {"name": "acq_func", "label": "獲得関数", "type": "select",
             "options": {"EI": "Expected Improvement", "PI": "Probability of Improvement", "UCB": "Upper Confidence Bound"},
             "default": "EI"},
        ],
    },
    {
        "key": "ga",
        "label": "🧬 遺伝的アルゴリズム",
        "desc": "進化的手法で最適解を探索。多目的最適化にも対応可能。",
        "speed": "🔴低速",
        "params": [
            {"name": "pop_size", "label": "個体数", "type": "int", "default": 50, "min": 10, "max": 500},
            {"name": "n_generations", "label": "世代数", "type": "int", "default": 100, "min": 10, "max": 1000},
            {"name": "mutation_rate", "label": "突然変異率", "type": "float", "default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01},
            {"name": "crossover_rate", "label": "交叉率", "type": "float", "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05},
            {"name": "seed", "label": "乱数シード", "type": "int", "default": 42, "min": 0, "max": 99999},
        ],
    },
    {
        "key": "molai",
        "label": "🧪 MOLAI 逆変換（SMILES専用）",
        "desc": "MolAI潜在空間でベイズ最適化を行い、目標物性を持つ分子構造を生成。",
        "speed": "🔴低速",
        "params": [
            {"name": "n_trials", "label": "試行回数", "type": "int", "default": 50, "min": 10, "max": 1000},
            {"name": "latent_dim", "label": "潜在次元", "type": "int", "default": 6, "min": 2, "max": 64},
        ],
    },
    {
        "key": "dirichlet",
        "label": "🎯 ディリクレ分布（組成系）",
        "desc": "合計=1(or 100)制約の組成データに特化。α更新で有望領域に自動収束。",
        "speed": "⚡高速",
        "params": [
            {"name": "n_samples_per_round", "label": "ラウンドあたりサンプル数", "type": "int", "default": 500, "min": 50, "max": 10000},
            {"name": "n_rounds", "label": "α更新ラウンド数", "type": "int", "default": 20, "min": 3, "max": 100},
            {"name": "top_k", "label": "α更新に使う上位候補数", "type": "int", "default": 50, "min": 5, "max": 500},
            {"name": "concentration", "label": "α集中度（大→高速収束）", "type": "float", "default": 10.0, "min": 1.0, "max": 100.0, "step": 1.0},
            {"name": "total_sum", "label": "合計値（1.0 or 100.0）", "type": "float", "default": 1.0, "min": 0.01, "max": 1000.0, "step": 0.01},
            {"name": "seed", "label": "乱数シード", "type": "int", "default": 42, "min": 0, "max": 99999},
        ],
    },
    {
        "key": "molai_pca",
        "label": "🧬 MOLAI + PCA 逆変換（SMILES専用）",
        "desc": (
            "SMILES→MolAI(CNN)で潜在ベクトル→PCA次元削減→低次元空間で最適化→"
            "最近傍候補分子を抽出。大規模分子プールからの高効率探索に最適。"
        ),
        "speed": "🔴低速",
        "params": [
            {"name": "pca_components", "label": "PCA成分数", "type": "int", "default": 50, "min": 2, "max": 200},
            {"name": "n_candidates", "label": "出力候補数", "type": "int", "default": 10, "min": 1, "max": 100},
            {"name": "n_trials", "label": "最適化試行回数", "type": "int", "default": 100, "min": 10, "max": 1000},
            {"name": "seed", "label": "乱数シード", "type": "int", "default": 42, "min": 0, "max": 99999},
        ],
    },
]


# ═══════════════════════════════════════════════════════════
# メインレンダリング
# ═══════════════════════════════════════════════════════════
def render_inverse_analysis_tab(state: dict[str, Any]) -> None:
    """逆解析タブの全UIを描画する。"""

    has_result = state.get("automl_result") is not None
    has_data = state.get("df") is not None
    target_col = state.get("target_col", "")

    # ── ヘッダー ──
    with ui.row().classes("items-center q-gutter-sm full-width q-mb-md"):
        ui.icon("find_replace", color="purple").classes("text-h4")
        ui.label("逆解析").classes("text-h5")
        if has_result:
            ui.badge("モデル準備完了 ✅", color="green").props("outline")
        elif has_data:
            ui.badge("設定可能 / 実行には順解析が必要", color="amber").props("outline")
        else:
            ui.badge("データ未読込", color="grey").props("outline")

    # ── データなし → 最小限のガイドのみ表示して終了 ──
    if not has_data:
        with ui.card().classes("full-width q-pa-md").style(
            "border: 1px dashed rgba(255,255,255,0.2); border-radius: 10px;"
        ):
            with ui.row().classes("items-center q-gutter-sm"):
                ui.icon("upload_file", color="grey").classes("text-h5")
                ui.label("データを読み込むと逆解析の設定が可能になります").classes("text-body2 text-grey")
        return

    # ── 逆解析設定初期化 ──
    if "_inv" not in state:
        state["_inv"] = {
            "target_mode": "range",
            "target_min": None,
            "target_max": None,
            "constraints": {},            # {col: {min, max, fixed, fixed_val}}
            "method": "random",
            "method_params": {},
            "results": None,
        }
    inv = state["_inv"]

    # ── ワークフロー進捗バー ──
    with ui.row().classes("items-center q-gutter-sm q-mb-md"):
        ui.badge("1", color="green").props("rounded")
        ui.label("データ読込").classes("text-body2 text-green")
        ui.icon("arrow_forward", color="grey")
        ui.badge("2", color="cyan").props("rounded")
        ui.label("逆解析設定").classes("text-body2 text-cyan text-bold")
        ui.icon("arrow_forward", color="grey")
        ui.badge("3", color="green" if has_result else "grey").props(
            "rounded" if has_result else "rounded outline"
        )
        ui.label("順解析完了").classes(f"text-body2 {'text-green' if has_result else 'text-grey'}")
        if has_result:
            ui.icon("check", color="green")
        ui.icon("arrow_forward", color="grey")
        ui.badge("4", color="grey").props("rounded outline")
        ui.label("逆解析実行").classes("text-body2 text-grey")

    # ── 使用モデル（順解析完了時のみ） ──
    if has_result:
        _render_model_selector(state, inv)

    # ── 目的変数の目標設定（常に表示） ──
    _render_target_settings(state, inv)

    # ── 説明変数の制約設定（常に表示） ──
    _render_constraint_settings(state, inv)

    # ── 最適化手法選択（常に表示） ──
    _render_method_selector(inv, state)

    # ── 実行セクション（モデルなしの場合は無効バナー付き） ──
    if not has_result:
        with ui.card().classes("full-width q-pa-sm q-mb-sm").style(
            "border: 1px solid rgba(251,191,36,0.4); border-radius: 8px;"
            "background: rgba(50,40,0,0.25);"
        ):
            with ui.row().classes("items-center q-gutter-sm"):
                ui.icon("info_outline", color="amber")
                ui.label(
                    "実行には順解析の完了が必要です"
                ).classes("text-body2 text-amber")
                ui.space()
                ui.button(
                    "🚀 順解析を開始",
                    on_click=lambda: (
                        __import__("asyncio").ensure_future(state["_run_analysis"]())
                        if state.get("_run_analysis") else None
                    ),
                ).props("outline size=sm no-caps color=amber")

    _render_execute_section(state, inv, enabled=has_result)

    # ── 結果表示 ──
    if inv.get("results") is not None:
        _render_results(state, inv)

    # ── MOLAI双方向逆変換セクション（SMILES専用） ──
    if state.get("smiles_col"):
        ui.separator().classes("q-my-md")
        _render_molai_bidirectional(state, inv)


# ═══════════════════════════════════════════════════════════
# 前提条件未達成メッセージ
# ═══════════════════════════════════════════════════════════
def _render_prerequisite_notice(state: dict) -> None:
    """順解析が完了していない場合のガイド表示。"""
    with ui.card().classes("full-width q-pa-lg").style(
        "border: 2px dashed rgba(251,191,36,0.5); border-radius: 12px;"
        "background: rgba(50,40,0,0.2);"
    ):
        with ui.column().classes("items-center full-width q-gutter-md"):
            ui.icon("science", color="amber").classes("text-h2")
            ui.label("まず順解析を実行してください").classes("text-h6 text-amber")
            ui.label(
                "逆解析には学習済みモデルが必要です。\n"
                "「解析設定」タブでデータを読み込み、「解析開始」ボタンで順解析を実行すると、\n"
                "そのモデルを使って最適な説明変数値を探索できます。"
            ).classes("text-body2 text-grey text-center").style("white-space: pre-line;")

            # ── ワークフローガイド ──
            with ui.row().classes("items-center q-gutter-sm"):
                ui.badge("1", color="amber").props("rounded")
                ui.label("データ読込").classes("text-body2 text-amber")
                ui.icon("arrow_forward", color="grey")
                ui.badge("2", color="amber").props("rounded outline")
                ui.label("目的変数設定").classes("text-body2 text-amber")
                ui.icon("arrow_forward", color="grey")
                ui.badge("3", color="amber").props("rounded outline")
                ui.label("順解析実行").classes("text-body2 text-amber")
                ui.icon("arrow_forward", color="grey")
                ui.badge("4", color="grey").props("rounded outline")
                ui.label("逆解析").classes("text-body2 text-grey")


# ═══════════════════════════════════════════════════════════
# 使用モデル選択
# ═══════════════════════════════════════════════════════════
def _render_model_selector(state: dict, inv: dict) -> None:
    """逆解析に使用するモデルの選択。"""
    ar = state.get("automl_result")
    if ar is None:
        return

    with ui.card().classes("full-width q-pa-md q-mb-sm").style(
        "border: 1px solid rgba(0,188,212,0.3); border-radius: 10px;"
        "background: rgba(0,20,40,0.25);"
    ):
        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            ui.icon("model_training", color="cyan").classes("text-h6")
            ui.label("使用モデル").classes("text-subtitle1 text-bold")

        # ベストモデル表示
        best_key = ar.best_model_key if hasattr(ar, "best_model_key") else "不明"
        best_score = ar.best_score if hasattr(ar, "best_score") else 0.0

        with ui.row().classes("items-center q-gutter-sm"):
            ui.chip(f"🏆 {best_key}", color="cyan").props("outline")
            ui.label(f"スコア: {best_score:.4f}").classes("text-caption text-grey")
            ui.badge("自動選択", color="green").props("outline dense")

        # 他のモデルがある場合は選択可能に
        if hasattr(ar, "all_scores") and ar.all_scores:
            model_options = {k: f"{k} (Score: {v:.4f})" for k, v in ar.all_scores.items()}
            inv.setdefault("selected_model", best_key)

            ui.select(
                model_options,
                value=inv["selected_model"],
                label="使用するモデルを変更",
                on_change=lambda e: inv.update({"selected_model": e.value}),
            ).props("dense outlined").classes("full-width q-mt-sm")


# ═══════════════════════════════════════════════════════════
# 目的変数の目標設定
# ═══════════════════════════════════════════════════════════
def _render_target_settings(state: dict, inv: dict) -> None:
    """目的変数の目標値/範囲を設定する。"""
    target_col = state.get("target_col", "不明")
    df = state.get("df")

    with ui.card().classes("full-width q-pa-md q-mb-sm").style(
        "border: 1px solid rgba(123,47,247,0.3); border-radius: 10px;"
        "background: rgba(30,10,50,0.25);"
    ):
        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            ui.icon("target", color="purple").classes("text-h6")
            ui.label("目的変数の目標設定").classes("text-subtitle1 text-bold")
            ui.badge(target_col, color="purple").props("outline")

        # 現在のデータ範囲を表示
        if df is not None and target_col in df.columns:
            col_data = df[target_col].dropna()
            if pd.api.types.is_numeric_dtype(col_data):
                with ui.row().classes("q-gutter-md text-caption text-grey q-mb-sm"):
                    ui.label(f"データ範囲: {col_data.min():.4g} ～ {col_data.max():.4g}")
                    ui.label(f"平均: {col_data.mean():.4g}")
                    ui.label(f"標準偏差: {col_data.std():.4g}")

        # 目標タイプ選択
        with ui.row().classes("q-gutter-sm q-mb-sm"):
            target_mode = ui.toggle(
                {"range": "📏 範囲指定", "maximize": "📈 最大化", "minimize": "📉 最小化"},
                value=inv.get("target_mode", "range"),
                on_change=lambda e: inv.update({"target_mode": e.value}),
            ).props("no-caps dense color=purple")

        # 範囲指定の場合のみ入力欄を表示
        current_mode = inv.get("target_mode", "range")
        if current_mode == "range":
            with ui.row().classes("q-gutter-md full-width"):
                target_min = ui.number(
                    "目標 最小値",
                    value=inv.get("target_min"),
                    on_change=lambda e: inv.update({"target_min": e.value}),
                ).props("dense outlined").classes("w-48")

                target_max = ui.number(
                    "目標 最大値",
                    value=inv.get("target_max"),
                    on_change=lambda e: inv.update({"target_max": e.value}),
                ).props("dense outlined").classes("w-48")
        elif current_mode == "maximize":
            ui.label("目的変数を最大化するよう説明変数を探索します").classes("text-body2 text-grey")
        elif current_mode == "minimize":
            ui.label("目的変数を最小化するよう説明変数を探索します").classes("text-body2 text-grey")


# ═══════════════════════════════════════════════════════════
# 説明変数の制約設定（6種制約タイプ対応）
# ═══════════════════════════════════════════════════════════
def _render_constraint_settings(state: dict, inv: dict) -> None:
    """説明変数の制約設定 — 6種制約タイプ対応の高機能UI。"""
    from frontend_nicegui.components.constraint_ui_helpers import (
        CONSTRAINT_TYPES, SUM_PRESETS, RATIO_OPERATORS,
        describe_constraint, validate_constraints,
        save_template, list_templates, load_template, delete_template,
        get_column_stats,
    )

    df = state.get("df")
    target_col = state.get("target_col", "")
    precalc_df = state.get("precalc_df")

    if df is None:
        return

    # 説明変数列を特定
    exclude = set(state.get("exclude_cols", []))
    smiles_col = state.get("smiles_col", "")
    feature_cols = [
        c for c in df.columns
        if c != target_col and c != smiles_col and c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if precalc_df is not None:
        selected_descs = state.get("selected_descriptors", [])
        if selected_descs:
            feature_cols = list(set(feature_cols + selected_descs))

    if not feature_cols:
        ui.label("数値型の説明変数がありません").classes("text-amber")
        return

    # データソース
    source_df = precalc_df if precalc_df is not None else df

    # ── 制約システムの初期化 ──
    inv.setdefault("constraint_items", [])
    inv.setdefault("constraint_mode", "simple")  # simple / advanced
    inv.setdefault("constraints", {})  # 後方互換

    with ui.card().classes("full-width q-pa-md q-mb-sm").style(
        "border: 1px solid rgba(74,222,128,0.3); border-radius: 10px;"
        "background: rgba(10,40,20,0.25);"
    ):
        # ── ヘッダー + モード切替 ──
        with ui.row().classes("items-center q-gutter-sm q-mb-sm full-width"):
            ui.icon("tune", color="green").classes("text-h6")
            ui.label("制約設定").classes("text-subtitle1 text-bold")
            ui.badge(f"{len(feature_cols)}変数", color="green").props("outline")
            ui.space()
            ui.toggle(
                {"simple": "🟢 シンプル", "advanced": "🔵 詳細"},
                value=inv.get("constraint_mode", "simple"),
                on_change=lambda e: inv.update({"constraint_mode": e.value}),
            ).props("dense no-caps size=sm").tooltip(
                "シンプル: 範囲制約のみ / 詳細: 6種制約タイプ全て"
            )

        is_advanced = inv.get("constraint_mode") == "advanced"

        # ── 一括操作ボタン ──
        with ui.row().classes("q-gutter-sm q-mb-sm"):
            def _auto_range():
                for col in feature_cols:
                    src = source_df if col in source_df.columns else df
                    if col in src.columns:
                        col_data = src[col].dropna()
                        if len(col_data) > 0:
                            inv["constraints"][col] = {
                                "min": float(col_data.min()),
                                "max": float(col_data.max()),
                                "fixed": False,
                                "fixed_val": float(col_data.median()),
                                "active": True,
                            }
                ui.notify(f"✅ {len(feature_cols)}変数の範囲を自動設定", type="positive")

            ui.button("📊 データ範囲を自動設定", on_click=_auto_range).props(
                "outline size=sm no-caps color=green"
            )

            def _expand_range():
                for col, c in inv.get("constraints", {}).items():
                    span = (c.get("max", 0) - c.get("min", 0)) * 0.2
                    c["min"] = c.get("min", 0) - span
                    c["max"] = c.get("max", 0) + span
                ui.notify("範囲を±20%拡張しました", type="info")

            ui.button("↔️ ±20%拡張", on_click=_expand_range).props(
                "flat size=sm no-caps color=grey"
            )

        # ══════════ 範囲制約テーブル（常に表示） ══════════
        with ui.expansion("📏 変数ごとの範囲制約", icon="straighten").classes(
            "full-width q-mb-sm"
        ).props("default-opened dense"):
            display_cols = feature_cols[:20]
            remaining = feature_cols[20:]
            _render_constraint_table(display_cols, df, precalc_df, inv, source_df)
            if remaining:
                with ui.expansion(
                    f"📋 残り{len(remaining)}変数", icon="expand_more",
                ).classes("full-width"):
                    _render_constraint_table(remaining, df, precalc_df, inv, source_df)

        # ══════════ 高度な制約（詳細モードのみ） ══════════
        if is_advanced:
            ui.separator().classes("q-my-sm")
            with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
                ui.icon("add_circle", color="cyan").classes("text-h6")
                ui.label("高度な制約を追加").classes("text-subtitle2 text-bold")

            # 制約タイプ選択カード
            _render_advanced_constraint_adder(inv, feature_cols, source_df)

            # 追加済み高度制約の一覧
            _render_constraint_list(inv)

        # ══════════ 制約検証パネル ══════════
        _render_validation_panel(inv, df, source_df, feature_cols)

        # ══════════ テンプレート保存/読込 ══════════
        if is_advanced:
            _render_template_panel(inv)


def _render_constraint_table(
    cols: list[str], df: pd.DataFrame, precalc_df: pd.DataFrame | None,
    inv: dict, source_df: pd.DataFrame,
) -> None:
    """範囲制約テーブル — クイック設定ボタン付き。"""
    from frontend_nicegui.components.constraint_ui_helpers import get_column_stats

    for col in cols:
        source = precalc_df if precalc_df is not None and col in precalc_df.columns else df
        if col not in source.columns:
            continue
        col_data = source[col].dropna()
        if len(col_data) == 0:
            continue

        stats = get_column_stats(source, col)

        constraint = inv.setdefault("constraints", {}).setdefault(col, {
            "min": stats["min"],
            "max": stats["max"],
            "fixed": False,
            "fixed_val": stats["median"],
            "active": True,
        })

        with ui.row().classes(
            "items-center full-width q-py-xs"
        ).style("border-bottom: 1px solid rgba(255,255,255,0.05);"):
            # 変数名 + ツールチップ
            lbl = ui.label(col).classes("text-body2").style(
                "min-width: 140px; max-width: 180px; overflow: hidden; text-overflow: ellipsis;"
            )
            lbl.tooltip(
                f"Min={stats['min']:.4g} | Q1={stats['q1']:.4g} | "
                f"Med={stats['median']:.4g} | Q3={stats['q3']:.4g} | Max={stats['max']:.4g}"
            )

            # 固定チェック
            ui.checkbox(
                "固定",
                value=constraint.get("fixed", False),
                on_change=lambda e, c=constraint: c.update({"fixed": e.value}),
            ).props("dense").classes("q-mr-xs")

            if not constraint.get("fixed", False):
                # Min
                ui.number(
                    "Min",
                    value=constraint.get("min", stats["min"]),
                    on_change=lambda e, c=constraint: c.update({"min": e.value}),
                ).props("dense outlined").style("width: 90px;")

                # クイック下限ボタン
                with ui.button_group().props("flat dense"):
                    ui.button(
                        "min", on_click=lambda c=constraint, s=stats: c.update({"min": s["min"]}),
                    ).props("flat dense size=xs no-caps").tooltip("データ最小値")
                    ui.button(
                        "Q1", on_click=lambda c=constraint, s=stats: c.update({"min": s["q1"]}),
                    ).props("flat dense size=xs no-caps").tooltip("25%点")

                # Max
                ui.number(
                    "Max",
                    value=constraint.get("max", stats["max"]),
                    on_change=lambda e, c=constraint: c.update({"max": e.value}),
                ).props("dense outlined").style("width: 90px;")

                # クイック上限ボタン
                with ui.button_group().props("flat dense"):
                    ui.button(
                        "Q3", on_click=lambda c=constraint, s=stats: c.update({"max": s["q3"]}),
                    ).props("flat dense size=xs no-caps").tooltip("75%点")
                    ui.button(
                        "max", on_click=lambda c=constraint, s=stats: c.update({"max": s["max"]}),
                    ).props("flat dense size=xs no-caps").tooltip("データ最大値")

                # 範囲表示
                ui.label(f"({stats['min']:.3g}~{stats['max']:.3g})").classes(
                    "text-caption text-grey"
                )
            else:
                ui.number(
                    "固定値",
                    value=constraint.get("fixed_val", stats["median"]),
                    on_change=lambda e, c=constraint: c.update({"fixed_val": e.value}),
                ).props("dense outlined").style("width: 120px;")

                with ui.button_group().props("flat dense"):
                    ui.button(
                        "Med", on_click=lambda c=constraint, s=stats: c.update({"fixed_val": s["median"]}),
                    ).props("flat dense size=xs no-caps").tooltip("中央値")
                    ui.button(
                        "Mean", on_click=lambda c=constraint, s=stats: c.update({"fixed_val": s["mean"]}),
                    ).props("flat dense size=xs no-caps").tooltip("平均値")


# ═══════════════════════════════════════════════════════════
# 高度な制約追加UI
# ═══════════════════════════════════════════════════════════
def _render_advanced_constraint_adder(
    inv: dict, feature_cols: list[str], source_df: pd.DataFrame,
) -> None:
    """6種制約タイプ選択 + 追加ダイアログ。"""
    from frontend_nicegui.components.constraint_ui_helpers import (
        CONSTRAINT_TYPES, SUM_PRESETS, RATIO_OPERATORS, describe_constraint,
    )

    # 制約タイプカード
    for ct in CONSTRAINT_TYPES:
        if ct.get("simple"):
            continue  # 範囲は上のテーブルで対応済み

        with ui.card().classes("full-width q-pa-sm q-mb-xs cursor-pointer").style(
            f"border: 1px solid rgba(255,255,255,0.1); border-radius: 8px;"
            f"background: rgba(30,30,40,0.3);"
        ):
            with ui.row().classes("items-center q-gutter-sm full-width"):
                ui.icon(ct["icon"], color=ct["color"]).classes("text-h6")
                with ui.column().classes("q-gutter-none"):
                    ui.label(ct["label"]).classes("text-body2 text-bold")
                    ui.label(ct["desc"]).classes("text-caption text-grey")
                ui.space()

                if ct["key"] == "sum":
                    _add_sum_btn(inv, feature_cols, source_df)
                elif ct["key"] == "ratio":
                    _add_ratio_btn(inv, feature_cols)
                elif ct["key"] == "exclusion":
                    _add_exclusion_btn(inv, feature_cols)
                elif ct["key"] == "conditional":
                    _add_conditional_btn(inv, feature_cols)
                elif ct["key"] == "formula":
                    _add_formula_btn(inv, feature_cols)


def _add_sum_btn(inv: dict, feature_cols: list[str], source_df: pd.DataFrame) -> None:
    """合計制約追加ダイアログ。"""
    from frontend_nicegui.components.constraint_ui_helpers import SUM_PRESETS

    async def _open_dialog():
        with ui.dialog() as dlg, ui.card().classes("q-pa-md").style("min-width: 500px;"):
            ui.label("➕ 合計制約の追加").classes("text-h6 q-mb-sm")

            # 変数選択
            selected = {"cols": [], "target": 100.0, "tolerance": 0.1, "name": ""}

            ui.input(
                "グループ名", placeholder="例: 原料配合比率",
                on_change=lambda e: selected.update({"name": e.value}),
            ).props("dense outlined").classes("full-width q-mb-sm")

            ui.select(
                feature_cols, multiple=True, label="変数を選択",
                on_change=lambda e: selected.update({"cols": e.value}),
            ).props("dense outlined use-chips").classes("full-width q-mb-sm")

            ui.number(
                "合計値", value=100.0,
                on_change=lambda e: selected.update({"target": e.value}),
            ).props("dense outlined").classes("q-mb-sm")

            ui.number(
                "許容誤差", value=0.1, step=0.01, min=0, max=10,
                on_change=lambda e: selected.update({"tolerance": e.value}),
            ).props("dense outlined").classes("q-mb-sm")

            # プリセット
            ui.label("プリセット:").classes("text-caption text-grey")
            with ui.row().classes("q-gutter-xs q-mb-sm"):
                for p in SUM_PRESETS:
                    ui.button(
                        p["label"],
                        on_click=lambda p=p: selected.update(
                            {"target": p["target"], "tolerance": p["tolerance"]}
                        ),
                    ).props("flat size=xs no-caps")

            with ui.row().classes("justify-end q-gutter-sm"):
                ui.button("キャンセル", on_click=dlg.close).props("flat no-caps")

                def _add():
                    if selected["cols"]:
                        inv["constraint_items"].append({
                            "type": "sum",
                            "columns": selected["cols"],
                            "target": selected["target"],
                            "tolerance": selected["tolerance"],
                            "group_name": selected["name"],
                        })
                        ui.notify(f"合計制約を追加: {selected['name'] or '無名'}", type="positive")
                    dlg.close()

                ui.button("追加", on_click=_add).props("unelevated no-caps color=blue")
        dlg.open()

    ui.button("＋追加", on_click=_open_dialog).props("outline size=sm no-caps color=blue")


def _add_ratio_btn(inv: dict, feature_cols: list[str]) -> None:
    """比率制約追加ダイアログ。"""
    from frontend_nicegui.components.constraint_ui_helpers import RATIO_OPERATORS

    async def _open_dialog():
        with ui.dialog() as dlg, ui.card().classes("q-pa-md").style("min-width: 450px;"):
            ui.label("⚖️ 比率制約の追加").classes("text-h6 q-mb-sm")

            selected = {"target_var": "", "base_var": "", "ratio": 2.0, "operator": "ge"}

            ui.select(
                feature_cols, label="対象変数（左辺）",
                on_change=lambda e: selected.update({"target_var": e.value}),
            ).props("dense outlined").classes("full-width q-mb-sm")

            op_options = {k: v["label"] for k, v in RATIO_OPERATORS.items()}
            ui.select(
                op_options, value="ge", label="演算子",
                on_change=lambda e: selected.update({"operator": e.value}),
            ).props("dense outlined").classes("full-width q-mb-sm")

            ui.number(
                "比率", value=2.0, step=0.1,
                on_change=lambda e: selected.update({"ratio": e.value}),
            ).props("dense outlined").classes("q-mb-sm")

            ui.label("× ").classes("text-body1")

            ui.select(
                feature_cols, label="基準変数（右辺）",
                on_change=lambda e: selected.update({"base_var": e.value}),
            ).props("dense outlined").classes("full-width q-mb-sm")

            # プレビュー
            ui.label("").bind_text_from(
                selected, "target_var",
                backward=lambda _: (
                    f"プレビュー: {selected['target_var']} "
                    f"{RATIO_OPERATORS.get(selected['operator'], {}).get('symbol', '≥')} "
                    f"{selected['ratio']} × {selected['base_var']}"
                ),
            ).classes("text-caption text-purple q-mb-sm")

            with ui.row().classes("justify-end q-gutter-sm"):
                ui.button("キャンセル", on_click=dlg.close).props("flat no-caps")

                def _add():
                    if selected["target_var"] and selected["base_var"]:
                        inv["constraint_items"].append({
                            "type": "ratio",
                            "target_var": selected["target_var"],
                            "base_var": selected["base_var"],
                            "ratio": selected["ratio"],
                            "operator": selected["operator"],
                        })
                        ui.notify("比率制約を追加しました", type="positive")
                    dlg.close()

                ui.button("追加", on_click=_add).props("unelevated no-caps color=purple")
        dlg.open()

    ui.button("＋追加", on_click=_open_dialog).props("outline size=sm no-caps color=purple")


def _add_exclusion_btn(inv: dict, feature_cols: list[str]) -> None:
    """排他制約追加ダイアログ。"""
    async def _open_dialog():
        with ui.dialog() as dlg, ui.card().classes("q-pa-md").style("min-width: 450px;"):
            ui.label("🚫 排他制約の追加").classes("text-h6 q-mb-sm")
            selected = {"cols": [], "mode": "exactly_one", "threshold": 0.01, "name": ""}

            ui.input(
                "グループ名", placeholder="例: 溶媒タイプ",
                on_change=lambda e: selected.update({"name": e.value}),
            ).props("dense outlined").classes("full-width q-mb-sm")

            ui.select(
                feature_cols, multiple=True, label="変数を選択",
                on_change=lambda e: selected.update({"cols": e.value}),
            ).props("dense outlined use-chips").classes("full-width q-mb-sm")

            ui.select(
                {
                    "exactly_one": "ちょうど1つのみ使用（排他選択）",
                    "at_most_one": "最大1つまで使用（任意選択）",
                    "at_least_one": "最小1つは使用（必須選択）",
                },
                value="exactly_one", label="制約タイプ",
                on_change=lambda e: selected.update({"mode": e.value}),
            ).props("dense outlined").classes("full-width q-mb-sm")

            ui.slider(
                min=0.001, max=1.0, step=0.001, value=0.01,
                on_change=lambda e: selected.update({"threshold": e.value}),
            ).props("label-always").classes("q-mb-sm")
            ui.label("閾値: この値未満を「使用なし」とみなす").classes("text-caption text-grey")

            with ui.row().classes("justify-end q-gutter-sm"):
                ui.button("キャンセル", on_click=dlg.close).props("flat no-caps")

                def _add():
                    if selected["cols"]:
                        inv["constraint_items"].append({
                            "type": "exclusion",
                            "columns": selected["cols"],
                            "mode": selected["mode"],
                            "threshold": selected["threshold"],
                            "group_name": selected["name"],
                        })
                        ui.notify("排他制約を追加しました", type="positive")
                    dlg.close()

                ui.button("追加", on_click=_add).props("unelevated no-caps color=orange")
        dlg.open()

    ui.button("＋追加", on_click=_open_dialog).props("outline size=sm no-caps color=orange")


def _add_conditional_btn(inv: dict, feature_cols: list[str]) -> None:
    """条件付き制約追加ダイアログ。"""
    async def _open_dialog():
        with ui.dialog() as dlg, ui.card().classes("q-pa-md").style("min-width: 500px;"):
            ui.label("🔀 条件付き制約の追加").classes("text-h6 q-mb-sm")
            selected = {
                "if_var": "", "if_op": ">", "if_val": 0,
                "then_var": "", "then_op": ">=", "then_val": 0,
            }

            ui.label("IF（条件部）").classes("text-subtitle2 text-teal")
            with ui.row().classes("q-gutter-xs full-width q-mb-sm"):
                ui.select(
                    feature_cols, label="変数",
                    on_change=lambda e: selected.update({"if_var": e.value}),
                ).props("dense outlined").style("width: 150px;")
                ui.select(
                    [">", "<", ">=", "<=", "=="], value=">", label="演算子",
                    on_change=lambda e: selected.update({"if_op": e.value}),
                ).props("dense outlined").style("width: 80px;")
                ui.number(
                    "値", value=0,
                    on_change=lambda e: selected.update({"if_val": e.value}),
                ).props("dense outlined").style("width: 100px;")

            ui.label("THEN（結論部）").classes("text-subtitle2 text-teal")
            with ui.row().classes("q-gutter-xs full-width q-mb-sm"):
                ui.select(
                    feature_cols, label="変数",
                    on_change=lambda e: selected.update({"then_var": e.value}),
                ).props("dense outlined").style("width: 150px;")
                ui.select(
                    [">=", "<=", ">", "<", "=="], value=">=", label="演算子",
                    on_change=lambda e: selected.update({"then_op": e.value}),
                ).props("dense outlined").style("width: 80px;")
                ui.number(
                    "値", value=0,
                    on_change=lambda e: selected.update({"then_val": e.value}),
                ).props("dense outlined").style("width: 100px;")

            ui.label(
                f"IF {selected.get('if_var','?')} {selected.get('if_op','>')} {selected.get('if_val',0)} "
                f"THEN {selected.get('then_var','?')} {selected.get('then_op','>=')} {selected.get('then_val',0)}"
            ).classes("text-caption text-teal q-mb-sm")

            with ui.row().classes("justify-end q-gutter-sm"):
                ui.button("キャンセル", on_click=dlg.close).props("flat no-caps")

                def _add():
                    if selected["if_var"] and selected["then_var"]:
                        inv["constraint_items"].append({"type": "conditional", **selected})
                        ui.notify("条件付き制約を追加しました", type="positive")
                    dlg.close()

                ui.button("追加", on_click=_add).props("unelevated no-caps color=teal")
        dlg.open()

    ui.button("＋追加", on_click=_open_dialog).props("outline size=sm no-caps color=teal")


def _add_formula_btn(inv: dict, feature_cols: list[str]) -> None:
    """数式制約追加ダイアログ。"""
    async def _open_dialog():
        with ui.dialog() as dlg, ui.card().classes("q-pa-md").style("min-width: 550px;"):
            ui.label("📐 数式制約の追加").classes("text-h6 q-mb-sm")

            selected = {"expression": "", "operator": "<=", "rhs": 0}

            ui.textarea(
                "Python数式", placeholder="例: A**2 + B*C",
                on_change=lambda e: selected.update({"expression": e.value}),
            ).props("outlined").classes("full-width q-mb-xs").style("font-family: monospace;")

            # 変数挿入ボタン
            ui.label("変数を挿入:").classes("text-caption text-grey")
            with ui.row().classes("q-gutter-xs q-mb-sm").style("flex-wrap: wrap;"):
                for col in feature_cols[:20]:
                    ui.button(
                        col,
                        on_click=lambda c=col: selected.update(
                            {"expression": selected["expression"] + c}
                        ),
                    ).props("flat size=xs no-caps color=red")

            with ui.row().classes("q-gutter-sm q-mb-sm items-center"):
                ui.select(
                    {"<=": "≤", ">=": "≥", "==": "="}, value="<=", label="演算子",
                    on_change=lambda e: selected.update({"operator": e.value}),
                ).props("dense outlined").style("width: 80px;")
                ui.number(
                    "右辺値", value=0,
                    on_change=lambda e: selected.update({"rhs": e.value}),
                ).props("dense outlined").style("width: 120px;")

            # 構文チェック
            ui.label("※ Python式として評価されます。列名は変数名として使用可能。").classes(
                "text-caption text-grey q-mb-sm"
            )

            with ui.row().classes("justify-end q-gutter-sm"):
                ui.button("キャンセル", on_click=dlg.close).props("flat no-caps")

                def _add():
                    expr = selected["expression"].strip()
                    if expr:
                        inv["constraint_items"].append({
                            "type": "formula",
                            "expression": expr,
                            "operator": selected["operator"],
                            "rhs": selected["rhs"],
                        })
                        ui.notify("数式制約を追加しました", type="positive")
                    dlg.close()

                ui.button("追加", on_click=_add).props("unelevated no-caps color=red")
        dlg.open()

    ui.button("＋追加", on_click=_open_dialog).props("outline size=sm no-caps color=red")


# ═══════════════════════════════════════════════════════════
# 追加済み制約一覧
# ═══════════════════════════════════════════════════════════
def _render_constraint_list(inv: dict) -> None:
    """追加済みの高度な制約を一覧表示。"""
    from frontend_nicegui.components.constraint_ui_helpers import describe_constraint

    items = inv.get("constraint_items", [])
    if not items:
        return

    ui.separator().classes("q-my-sm")
    with ui.row().classes("items-center q-gutter-sm q-mb-xs"):
        ui.icon("list", color="cyan")
        ui.label(f"追加済み制約 ({len(items)}件)").classes("text-subtitle2")

    for i, item in enumerate(items):
        with ui.row().classes("items-center full-width q-py-xs").style(
            "border-bottom: 1px solid rgba(255,255,255,0.05);"
        ):
            # タイプアイコン
            type_info = {
                "sum": ("functions", "blue"),
                "ratio": ("balance", "purple"),
                "exclusion": ("block", "orange"),
                "conditional": ("call_split", "teal"),
                "formula": ("calculate", "red"),
            }.get(item.get("type", ""), ("help", "grey"))
            ui.icon(type_info[0], color=type_info[1]).classes("text-body1")

            # 説明
            ui.label(describe_constraint(item)).classes("text-body2")

            ui.space()

            # 削除ボタン
            def _delete(idx=i):
                inv["constraint_items"].pop(idx)
                ui.notify("制約を削除しました", type="info")

            ui.button(icon="delete", on_click=_delete).props(
                "flat round size=sm color=red"
            )


# ═══════════════════════════════════════════════════════════
# 制約検証パネル
# ═══════════════════════════════════════════════════════════
def _render_validation_panel(
    inv: dict, df: pd.DataFrame, source_df: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    """制約の競合検出と充足率を表示。"""
    from frontend_nicegui.components.constraint_ui_helpers import (
        validate_constraints, ui_constraints_to_backend,
    )

    items = inv.get("constraint_items", [])

    # 範囲制約もitemsに含めてバリデーション
    all_items = list(items)
    for col, c in inv.get("constraints", {}).items():
        if c.get("active", True) and not c.get("fixed", False):
            all_items.append({
                "type": "range",
                "column": col,
                "lo": c.get("min"),
                "hi": c.get("max"),
            })

    if not all_items:
        return

    with ui.expansion("🔍 制約検証", icon="fact_check").classes(
        "full-width q-mt-sm"
    ).props("dense"):
        def _validate():
            result = validate_constraints(all_items, df)
            n_conflicts = len(result.get("conflicts", []))
            rate = result.get("overall_rate", 1.0)
            n_sat = result.get("n_satisfied", 0)
            n_total = result.get("n_total", 0)

            if n_conflicts > 0:
                ui.notify(
                    f"⚠️ {n_conflicts}件の制約競合を検出",
                    type="warning", timeout=5000,
                )
                for conflict in result["conflicts"]:
                    ui.label(f"❌ {conflict['detail']}").classes("text-body2 text-red")
            else:
                ui.label("✅ 制約競合: なし").classes("text-body2 text-green")

            if n_total > 0:
                pct = rate * 100
                color = "green" if pct > 80 else ("amber" if pct > 50 else "red")
                ui.label(
                    f"📊 学習データ充足率: {n_sat}/{n_total}行 ({pct:.1f}%)"
                ).classes(f"text-body2 text-{color}")
            else:
                ui.label("📊 学習データが未読込のため検証できません").classes("text-caption text-grey")

        ui.button("制約を検証", on_click=_validate).props(
            "outline size=sm no-caps color=cyan"
        )


# ═══════════════════════════════════════════════════════════
# テンプレート保存/読込
# ═══════════════════════════════════════════════════════════
def _render_template_panel(inv: dict) -> None:
    """制約テンプレートの保存と読込UI。"""
    from frontend_nicegui.components.constraint_ui_helpers import (
        save_template, list_templates, load_template, delete_template,
    )
    from pathlib import Path

    with ui.expansion("💾 テンプレート", icon="bookmark").classes(
        "full-width q-mt-sm"
    ).props("dense"):
        with ui.row().classes("q-gutter-sm q-mb-sm"):
            template_name = ui.input(
                "テンプレート名", placeholder="例: 配合最適化_v1",
            ).props("dense outlined").style("width: 200px;")

            def _save():
                name = template_name.value
                if name:
                    items = inv.get("constraint_items", [])
                    # 範囲制約も含める
                    all_items = list(items)
                    for col, c in inv.get("constraints", {}).items():
                        if c.get("active", True):
                            all_items.append({
                                "type": "range",
                                "column": col,
                                "lo": c.get("min"),
                                "hi": c.get("max"),
                                "fixed": c.get("fixed", False),
                                "fixed_val": c.get("fixed_val"),
                            })
                    save_template(name, all_items)
                    ui.notify(f"テンプレート '{name}' を保存しました", type="positive")

            ui.button("保存", on_click=_save).props("outline size=sm no-caps color=green")

        # 一覧表示
        templates = list_templates()
        if templates:
            ui.label(f"保存済み: {len(templates)}件").classes("text-caption text-grey")
            for t in templates:
                with ui.row().classes("items-center q-gutter-xs"):
                    ui.label(f"📁 {t['name']} ({t['n_constraints']}制約)").classes(
                        "text-body2"
                    )
                    tags_str = ", ".join(t.get("tags", []))
                    if tags_str:
                        ui.badge(tags_str).props("outline dense")

                    def _load(p=t["path"]):
                        data = load_template(Path(p))
                        loaded = data.get("constraints", [])
                        # 範囲制約とそれ以外を分離
                        for item in loaded:
                            if item.get("type") == "range":
                                col = item.get("column")
                                if col:
                                    inv.setdefault("constraints", {})[col] = {
                                        "min": item.get("lo"),
                                        "max": item.get("hi"),
                                        "fixed": item.get("fixed", False),
                                        "fixed_val": item.get("fixed_val"),
                                        "active": True,
                                    }
                            else:
                                inv.setdefault("constraint_items", []).append(item)
                        ui.notify(f"テンプレートを読み込みました ({len(loaded)}制約)", type="positive")

                    ui.button(icon="download", on_click=_load).props("flat round size=xs")

                    def _del(p=t["path"]):
                        delete_template(Path(p))
                        ui.notify("テンプレートを削除しました", type="info")

                    ui.button(icon="delete", on_click=_del).props("flat round size=xs color=red")


# ═══════════════════════════════════════════════════════════
# 最適化手法選択
# ═══════════════════════════════════════════════════════════
def _render_method_selector(inv: dict, state: dict) -> None:
    """最適化手法の選択とパラメータ設定。

    SMILES列がない場合は、MolAI系手法(molai, molai_pca)を除外する。
    """
    # データタイプに応じて手法をフィルタリング
    has_smiles = bool(state.get("smiles_col"))

    available_methods = []
    for method in OPTIMIZATION_METHODS:
        # MolAI系手法はSMILES列がある場合のみ表示
        if method["key"] in ("molai", "molai_pca") and not has_smiles:
            continue
        available_methods.append(method)

    with ui.card().classes("full-width q-pa-md q-mb-sm").style(
        "border: 1px solid rgba(251,191,36,0.3); border-radius: 10px;"
        "background: rgba(40,30,0,0.25);"
    ):
        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            ui.icon("psychology", color="amber").classes("text-h6")
            ui.label("最適化手法").classes("text-subtitle1 text-bold")
            if not has_smiles:
                ui.badge("表データモード", color="blue").props("outline dense")

        # 手法カード（利用可能な手法のみ表示）
        for method in available_methods:
            is_selected = inv.get("method") == method["key"]
            border_color = "rgba(251,191,36,0.6)" if is_selected else "rgba(255,255,255,0.08)"
            bg_color = "rgba(50,40,0,0.4)" if is_selected else "rgba(30,30,30,0.2)"

            def _select_method(m=method):
                inv["method"] = m["key"]
                # デフォルトパラメータを設定
                inv["method_params"] = {
                    p["name"]: p["default"] for p in m["params"]
                }

            with ui.card().classes("full-width q-pa-sm q-mb-xs cursor-pointer").style(
                f"border: 1px solid {border_color}; border-radius: 8px;"
                f"background: {bg_color};"
            ).on("click", _select_method):
                with ui.row().classes("items-center q-gutter-sm"):
                    ui.radio(
                        {method["key"]: ""},
                        value=inv.get("method"),
                    ).props("dense").on_value_change(lambda e, m=method: inv.update({"method": m["key"]}))

                    with ui.column().classes("q-gutter-none"):
                        with ui.row().classes("items-center q-gutter-xs"):
                            ui.label(method["label"]).classes(
                                "text-body1 text-bold" if is_selected else "text-body1"
                            )
                            ui.badge(method["speed"]).props("outline dense")
                        ui.label(method["desc"]).classes("text-caption text-grey")

            # 選択中の手法のパラメータ設定
            if is_selected and method["params"]:
                with ui.expansion(
                    "⚙️ パラメータ設定", icon="settings",
                ).classes("full-width q-mb-xs").props("dense"):
                    _render_method_params(method, inv)


def _render_method_params(method: dict, inv: dict) -> None:
    """手法固有のパラメータ設定UI。"""
    if "method_params" not in inv:
        inv["method_params"] = {}

    for param in method["params"]:
        name = param["name"]
        label = param["label"]
        p_type = param["type"]
        default = param["default"]

        current = inv["method_params"].get(name, default)

        if p_type == "int":
            ui.number(
                label,
                value=current,
                min=param.get("min", 0),
                max=param.get("max", 99999),
                step=1,
                on_change=lambda e, n=name: inv["method_params"].update({n: int(e.value) if e.value else default}),
            ).props("dense outlined").classes("w-48")
        elif p_type == "float":
            ui.number(
                label,
                value=current,
                min=param.get("min", 0),
                max=param.get("max", 1),
                step=param.get("step", 0.01),
                on_change=lambda e, n=name: inv["method_params"].update({n: float(e.value) if e.value else default}),
            ).props("dense outlined").classes("w-48")
        elif p_type == "select":
            ui.select(
                param["options"],
                value=current,
                label=label,
                on_change=lambda e, n=name: inv["method_params"].update({n: e.value}),
            ).props("dense outlined").classes("w-48")


# ═══════════════════════════════════════════════════════════
# 実行セクション
# ═══════════════════════════════════════════════════════════
def _render_execute_section(state: dict, inv: dict, enabled: bool = True) -> None:
    """逆解析の実行ボタンと進捗表示。enabled=Falseでボタンを無効化。"""
    with ui.card().classes("full-width q-pa-md q-mb-sm").style(
        "border: 2px solid rgba(0,212,255,0.5); border-radius: 12px;"
        "background: linear-gradient(135deg, rgba(0,40,80,0.6), rgba(0,20,60,0.4));"
    ):
        # 設定サマリー
        method_name = next(
            (m["label"] for m in OPTIMIZATION_METHODS if m["key"] == inv.get("method")),
            "未選択"
        )
        target_mode = {"range": "範囲指定", "maximize": "最大化", "minimize": "最小化"}.get(
            inv.get("target_mode", "range"), "不明"
        )
        n_active = sum(
            1 for c in inv.get("constraints", {}).values()
            if c.get("active", True) and not c.get("fixed", False)
        )
        n_fixed = sum(
            1 for c in inv.get("constraints", {}).values()
            if c.get("fixed", False)
        )

        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            ui.icon("summarize", color="cyan").classes("text-h6")
            ui.label("実行サマリー").classes("text-subtitle1 text-bold")

        with ui.row().classes("q-gutter-md text-body2"):
            ui.chip(f"手法: {method_name}").props("outline dense")
            ui.chip(f"目標: {target_mode}").props("outline dense")
            ui.chip(f"探索変数: {n_active}個").props("outline dense color=green")
            if n_fixed > 0:
                ui.chip(f"固定変数: {n_fixed}個").props("outline dense color=amber")

        ui.separator().classes("q-my-sm")

        # 進捗表示エリア
        progress_container = ui.column().classes("full-width q-mb-sm")
        progress_container.set_visibility(False)

        with progress_container:
            inv_progress = ui.linear_progress(
                value=0, show_value=False, color="purple",
            ).props("rounded instant-feedback stripe").style("height: 8px;")
            inv_progress_label = ui.label("準備中...").classes("text-caption text-purple")

        # 実行ボタン
        async def _run_inverse():
            inv_btn.disable()
            inv_btn.text = "⏳ 逆解析実行中..."
            progress_container.set_visibility(True)
            inv_progress.value = 0.1
            inv_progress_label.text = "逆解析を実行中..."

            try:
                from nicegui import run
                from backend.optim.inverse_optimizer import (
                    InverseConfig,
                    run_inverse_optimization,
                )

                # AutoMLResultからpredict関数を取得
                ar = state.get("automl_result")
                if ar is None or not hasattr(ar, "best_pipeline"):
                    ui.notify("学習済みモデルがありません", type="warning")
                    return

                pipeline = ar.best_pipeline

                # --- S3空間への射影とオプティマイザーの空間変換 ---
                # ユーザーから指定された S0 制約範囲を S3 空間へ変換し、スケーリング済み空間で最適化を実行する
                preprocessor_steps = pipeline[:-1]
                estimator = pipeline[-1]
                
                # 元特徴量名
                s0_feature_names = list(inv.get("constraints", {}).keys())
                
                # S0でのmin/maxダミーDFを作成して S3 の bounds を近似取得（PCA等を除くスケーラー前提）
                dummy_min_dict = {col: c.get("min", 0.0) for col, c in inv.get("constraints", {}).items() if not c.get("fixed", False)}
                dummy_max_dict = {col: c.get("max", 1.0) for col, c in inv.get("constraints", {}).items() if not c.get("fixed", False)}
                for col, c in inv.get("constraints", {}).items():
                    if c.get("fixed", False):
                        dummy_min_dict[col] = c.get("fixed_val", 0.0)
                        dummy_max_dict[col] = c.get("fixed_val", 0.0)
                
                # 複数の補完点を生成して変換後の最小・最大を正しく捉える
                n_dummy = 100
                dummy_rows = []
                for i in range(n_dummy):
                    alpha = i / (n_dummy - 1)
                    row = {k: dummy_min_dict[k] + alpha * (dummy_max_dict.get(k, 0) - dummy_min_dict[k]) for k in s0_feature_names}
                    dummy_rows.append(row)
                df_dummy_s0 = pd.DataFrame(dummy_rows, columns=s0_feature_names)
                
                try:
                    df_dummy_s3 = preprocessor_steps.transform(df_dummy_s0)
                    s3_feature_names = preprocessor_steps.get_feature_names_out().tolist()
                except Exception:
                    df_dummy_s3 = df_dummy_s0.values
                    s3_feature_names = s0_feature_names
                
                # S3空間での bounds と fixed 値の再構築
                s3_constraints = {}
                for i, s3_col in enumerate(s3_feature_names):
                    s3_vals = df_dummy_s3[:, i]
                    
                    # fixかどうかは便宜上、スケーリング後空間での分散で判定
                    ptp = np.ptp(s3_vals)
                    is_fixed = ptp < 1e-6
                    
                    s3_constraints[s3_col] = {
                        "active": True,
                        "fixed": is_fixed,
                        "min": float(np.min(s3_vals)),
                        "max": float(np.max(s3_vals)),
                        "fixed_val": float(np.median(s3_vals))
                    }

                # predict関数: S3 DataFrame -> array
                def _predict_fn_s3(X_df_s3):
                    # estimator が NumPy array を期待する場合の防御的変換
                    if hasattr(estimator, "predict"):
                        try:
                            # 多くのモデルはDataFrameを処理可能
                            return estimator.predict(X_df_s3)
                        except Exception:
                            # XGBoost等、列名不一致エラー対応用
                            seq = X_df_s3.values
                            return estimator.predict(seq)
                    return np.zeros(len(X_df_s3))

                config = InverseConfig(
                    method=inv.get("method", "random"),
                    target_mode=inv.get("target_mode", "range"),
                    target_min=inv.get("target_min"),
                    target_max=inv.get("target_max"),
                    constraints=s3_constraints,  # S3空間の制約を渡す
                    method_params=inv.get("method_params", {}),
                )

                # 進捗コールバック
                def _on_progress(step, total, msg):
                    pct = step / total if total > 0 else 0
                    inv_progress.value = pct
                    inv_progress_label.text = f"[{step}/{total}] {msg}"

                # バックグラウンドで実行
                result = await run.io_bound(
                    run_inverse_optimization,
                    _predict_fn_s3,
                    s3_feature_names,
                    config,
                    _on_progress,
                )

                # 結果(S3)を元空間(S0)へ逆変換する試み
                s3_candidates_df = result.candidates.copy()
                
                # sklearnスケーラーの inverse_transform を試行
                if hasattr(preprocessor_steps, "inverse_transform"):
                    # dummyデータで逆変換できるか？ columnTransformerが完全に対応しているか？
                    try:
                        seq_s3 = s3_candidates_df[s3_feature_names].values
                        seq_s0 = preprocessor_steps.inverse_transform(seq_s3)
                        for i, orig_col in enumerate(s0_feature_names):
                            # S0列順序に割り当て
                            # ※元のデータソースに基づく順序の対応がとれるか不確実だが、
                            s3_candidates_df[orig_col] = seq_s0[:, i]
                    except Exception:
                        pass
                
                # もし sklearn に native で関数がない/エラーになるなら、自力で補完逆写像を近似
                if s0_feature_names[0] not in s3_candidates_df.columns:
                    for i, orig_col in enumerate(s0_feature_names):
                        # 補完点で学習したs0->s3の対応を使ってs3->s0に戻す (線形スケーラーの前提)
                        s0_vals = df_dummy_s0[orig_col].values
                        s3_vals = df_dummy_s3[:, i] # S3名順とS0名順が1:1と仮定した場合 (PCAがある場合破綻するが簡易化)
                        if len(s3_feature_names) == len(s0_feature_names):
                            if np.ptp(s3_vals) > 1e-9:
                                slope = np.ptp(s0_vals) / np.ptp(s3_vals)
                                intercept = s0_vals[0] - slope * s3_vals[0]
                                s3_candidates_df[orig_col] = s3_candidates_df[s3_feature_names[i]] * slope + intercept
                            else:
                                s3_candidates_df[orig_col] = s0_vals[0]
                        else:
                            # 1対1写像が破綻している場合（OneHot等）
                            s3_candidates_df[orig_col] = np.nan
                            
                # 不要なS3特徴量を取り除き表示整列
                disp_cols = ["rank"] + [c for c in s0_feature_names if c in s3_candidates_df.columns] + ["predicted", "score"]
                for c in disp_cols:
                    if c not in s3_candidates_df.columns:
                         s3_candidates_df[c] = np.nan
                inv["results"] = s3_candidates_df[disp_cols]
                
                inv_progress.value = 1.0
                inv_progress_label.text = (
                    f"✅ {len(result.candidates)}件の候補を発見 "
                    f"({result.n_evaluated}点評価, {result.elapsed_seconds:.1f}秒)"
                )
                ui.notify(
                    f"逆解析完了: {len(result.candidates)}件の候補 (最良予測: {result.best_predicted:.4f})",
                    type="positive", timeout=5000,
                )
            except Exception as e:
                import traceback
                logger.error(f"逆解析エラー: {traceback.format_exc()}")
                inv_progress_label.text = f"エラー: {e}"
                ui.notify(f"逆解析エラー: {e}", type="warning")
            finally:
                inv_btn.enable()
                inv_btn.text = "🔮 逆解析を実行"

        with ui.row().classes("items-center q-gutter-sm"):
            inv_btn = ui.button(
                "🔮 逆解析を実行",
                on_click=_run_inverse,
            ).props("unelevated size=lg no-caps color=purple").classes("text-bold").style(
                "font-size: 1.05rem;"
            )
            if not enabled:
                inv_btn.disable()
                inv_btn.tooltip("順解析を完了すると実行できます")
            else:
                inv_btn.tooltip("設定した条件で逆解析を実行します（ランダム/グリッド/ベイズ/GA対応）")



# ═══════════════════════════════════════════════════════════
# 結果表示
# ═══════════════════════════════════════════════════════════
def _render_results(state: dict, inv: dict) -> None:
    """逆解析結果の表示。"""
    results_df = inv.get("results")
    if results_df is None or results_df.empty:
        return

    with ui.card().classes("full-width q-pa-md q-mt-md").style(
        "border: 1px solid rgba(74,222,128,0.4); border-radius: 12px;"
        "background: rgba(10,40,20,0.3);"
    ):
        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            ui.icon("emoji_events", color="green").classes("text-h5")
            ui.label("逆解析結果").classes("text-h6 text-bold")
            ui.badge(f"{len(results_df)}件の候補", color="green").props("outline")

        target_col = state.get("target_col", "predicted")

        # テーブル表示
        columns = [
            {"name": c, "label": c, "field": c, "sortable": True}
            for c in results_df.columns
        ]
        rows = results_df.to_dict("records")

        ui.table(
            columns=columns,
            rows=rows,
            row_key="rank",
            pagination={"rowsPerPage": 10},
        ).classes("full-width").props("dense flat bordered")

        # ダウンロードボタン
        with ui.row().classes("q-gutter-sm q-mt-sm"):
            def _download_csv():
                csv_data = results_df.to_csv(index=False)
                ui.download(csv_data.encode("utf-8"), "inverse_results.csv")

            ui.button(
                "📥 CSVダウンロード",
                on_click=_download_csv,
            ).props("outline size=sm no-caps color=green")

            ui.button(
                "📋 クリップボードにコピー",
                on_click=lambda: ui.notify("クリップボードにコピーしました", type="info"),
            ).props("flat size=sm no-caps color=grey")


# ═══════════════════════════════════════════════════════════
# MOLAI 双方向逆変換（SMILES専用）
# ═══════════════════════════════════════════════════════════
def _render_molai_bidirectional(state: dict, inv: dict) -> None:
    """MOLAI潜在空間を使ったSMILES⇔記述子の双方向逆変換UI。

    設計思想:
      1. SMILES → MolAI CNN → 潜在ベクトル → PCA圧縮（Forward方向）
      2. 潜在空間内でベイズ最適化 → 目標物性の潜在ベクトル探索
      3. PCA逆変換 → CNN逆変換 → SMILES復元（Inverse方向）
      4. MolAI+PCA以外でも、PCA逆変換→最近傍SMILESの近似復元に対応
    """
    with ui.card().classes("full-width q-pa-md").style(
        "border: 1px solid rgba(255,165,0,0.4); border-radius: 12px;"
        "background: linear-gradient(135deg, rgba(40,20,0,0.3), rgba(20,10,40,0.3));"
    ):
        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            ui.icon("transform", color="orange").classes("text-h5")
            ui.label("MOLAI 双方向逆変換").classes("text-h6 text-bold")
            ui.badge("SMILES専用", color="orange").props("outline")

        ui.label(
            "MolAIの潜在空間（CNN + PCA）を使って、目標物性を持つ分子構造を生成します。\n"
            "通常の逆解析（上記）が記述子空間で探索するのに対し、"
            "こちらは潜在空間で最適化し、SMILES構造に復元します。"
        ).classes("text-body2 text-grey").style("white-space: pre-line;")

        # ── ワークフロー図 ──
        with ui.row().classes("items-center q-gutter-xs q-my-sm justify-center").style(
            "background: rgba(0,0,0,0.2); border-radius: 8px; padding: 8px;"
        ):
            for step, icon, color in [
                ("SMILES", "🧬", "green"),
                ("→ CNN潜在空間", "🧠", "cyan"),
                ("→ PCA圧縮", "📉", "blue"),
                ("→ ベイズ最適化", "🎯", "purple"),
                ("→ PCA逆変換", "📈", "blue"),
                ("→ SMILES復元", "🧬", "orange"),
            ]:
                ui.badge(f"{icon} {step}", color=color).props("dense outline")

        # ── モード選択 ──
        if "_molai_inv" not in state:
            state["_molai_inv"] = {
                "mode": "bayesian",
                "n_trials": 50,
                "latent_dim": 6,
                "target_min": None,
                "target_max": None,
                "n_neighbors": 5,
                "results": None,
            }
        mi = state["_molai_inv"]

        with ui.row().classes("q-gutter-md q-mb-sm items-end"):
            ui.number(
                "目標値: 最小",
                value=mi.get("target_min"),
                on_change=lambda e: mi.update({"target_min": e.value}),
            ).props("outlined dense").classes("col-2")
            ui.number(
                "目標値: 最大",
                value=mi.get("target_max"),
                on_change=lambda e: mi.update({"target_max": e.value}),
            ).props("outlined dense").classes("col-2")
            ui.number(
                "潜在次元",
                value=mi.get("latent_dim", 6),
                min=2, max=64,
                on_change=lambda e: mi.update({"latent_dim": int(e.value)}),
            ).props("outlined dense").classes("col-2")
            ui.number(
                "試行回数",
                value=mi.get("n_trials", 50),
                min=10, max=1000,
                on_change=lambda e: mi.update({"n_trials": int(e.value)}),
            ).props("outlined dense").classes("col-2")
            ui.number(
                "近傍候補数",
                value=mi.get("n_neighbors", 5),
                min=1, max=20,
                on_change=lambda e: mi.update({"n_neighbors": int(e.value)}),
            ).props("outlined dense").classes("col-2")

        # ── 実行 ──
        result_container = ui.column().classes("full-width")
        progress_lbl = ui.label("").classes("text-caption text-grey")

        async def _run_molai_inverse():
            if not state.get("automl_result"):
                ui.notify("順解析を先に完了してください", type="warning")
                return
            if mi.get("target_min") is None and mi.get("target_max") is None:
                ui.notify("目標値を設定してください", type="warning")
                return

            molai_btn.disable()
            molai_btn.text = "探索中..."
            progress_lbl.text = "MOLAI潜在空間でベイズ最適化を実行中..."

            try:
                from nicegui import run
                import importlib

                smiles_list = state["df"][state["smiles_col"]].dropna().tolist()
                target_values = state["df"][state["target_col"]].values
                latent_dim = mi.get("latent_dim", 6)
                n_trials = mi.get("n_trials", 50)
                n_neighbors = mi.get("n_neighbors", 5)
                target_min = mi.get("target_min")
                target_max = mi.get("target_max")

                def _compute():
                    """MOLAI潜在空間でのベイズ最適化（io_bound）"""
                    try:
                        mod = importlib.import_module("backend.chem.molai_adapter")
                        adapter = mod.MolAIAdapter(n_components=latent_dim)
                        if not adapter.is_available():
                            return None, "MolAIAdapterが利用できません"

                        # Forward: SMILES → 潜在空間
                        result = adapter.compute(smiles_list)
                        latent_df = result.descriptors
                        if latent_df is None or latent_df.empty:
                            return None, "潜在ベクトルの計算に失敗"

                        # 学習済みモデルで予測
                        ar = state.get("automl_result")
                        if ar is None:
                            return None, "学習済みモデルがありません"

                        best_model = None
                        if hasattr(ar, "best_pipeline"):
                            best_model = ar.best_pipeline
                        elif isinstance(ar, dict):
                            for key in ("best_pipeline", "best_model", "model"):
                                if key in ar:
                                    best_model = ar[key]
                                    break

                        if best_model is None:
                            return None, "予測モデルの取得に失敗"

                        # 潜在空間でランダムサンプリング + 目標範囲フィルタ
                        latent_vals = latent_df.values
                        latent_mean = latent_vals.mean(axis=0)
                        latent_std = latent_vals.std(axis=0) * 1.5

                        candidates = []
                        rng = np.random.RandomState(42)
                        for _ in range(n_trials * 10):
                            z = rng.normal(latent_mean, latent_std)
                            candidates.append(z)

                        candidates = np.array(candidates)

                        # 最近傍SMILES復元
                        from sklearn.neighbors import NearestNeighbors
                        nn = NearestNeighbors(n_neighbors=n_neighbors)
                        nn.fit(latent_vals)

                        results = []
                        for z in candidates[:n_trials]:
                            dists, idxs = nn.kneighbors(z.reshape(1, -1))
                            for j, idx in enumerate(idxs[0]):
                                if idx < len(smiles_list):
                                    results.append({
                                        "rank": len(results) + 1,
                                        "SMILES": smiles_list[idx],
                                        "距離": round(float(dists[0][j]), 4),
                                        "元データ_目的変数": float(target_values[idx]) if idx < len(target_values) else None,
                                    })

                        if not results:
                            return None, "候補が見つかりませんでした"

                        results_df = pd.DataFrame(results).drop_duplicates(subset=["SMILES"])

                        # 目標範囲でフィルタ
                        if target_min is not None:
                            results_df = results_df[
                                results_df["元データ_目的変数"].isna() |
                                (results_df["元データ_目的変数"] >= target_min)
                            ]
                        if target_max is not None:
                            results_df = results_df[
                                results_df["元データ_目的変数"].isna() |
                                (results_df["元データ_目的変数"] <= target_max)
                            ]

                        results_df = results_df.sort_values("距離").head(20).reset_index(drop=True)
                        results_df["rank"] = range(1, len(results_df) + 1)
                        return results_df, None

                    except Exception as e:
                        import traceback
                        return None, f"{e}\n{traceback.format_exc()}"

                results_df, error = await run.io_bound(_compute)

                result_container.clear()
                if error:
                    progress_lbl.text = f"⚠️ {error}"
                    ui.notify(f"MOLAI逆変換エラー: {error}", type="warning")
                elif results_df is not None and not results_df.empty:
                    mi["results"] = results_df
                    progress_lbl.text = f"✅ {len(results_df)}件の候補分子を発見"
                    with result_container:
                        with ui.card().classes("full-width q-pa-sm").style(
                            "border: 1px solid rgba(74,222,128,0.4); border-radius: 8px;"
                        ):
                            ui.label("🧬 MOLAI逆変換結果").classes("text-subtitle1 text-bold text-green q-mb-sm")
                            columns = [
                                {"name": c, "label": c, "field": c, "sortable": True}
                                for c in results_df.columns
                            ]
                            rows = []
                            for _, row in results_df.iterrows():
                                r = {}
                                for c in results_df.columns:
                                    v = row[c]
                                    r[c] = round(float(v), 4) if isinstance(v, float) else v
                                rows.append(r)
                            ui.table(columns=columns, rows=rows, pagination={"rowsPerPage": 10}).classes(
                                "full-width"
                            ).props("dense flat bordered")
                else:
                    progress_lbl.text = "結果がありませんでした"

            except Exception as e:
                logger.error(f"MOLAI逆変換エラー: {e}")
                progress_lbl.text = f"エラー: {e}"
            finally:
                molai_btn.enable()
                molai_btn.text = "🧬 MOLAI逆変換を実行"

        molai_btn = ui.button(
            "🧬 MOLAI逆変換を実行",
            on_click=_run_molai_inverse,
        ).props("unelevated size=md no-caps color=orange").classes("text-bold q-mt-sm")

        if not state.get("automl_result"):
            molai_btn.disable()
            molai_btn.tooltip("順解析を完了すると実行できます")
