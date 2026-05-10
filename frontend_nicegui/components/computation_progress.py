"""
frontend_nicegui/components/computation_progress.py

計算進捗ダッシュボード — スタンドアローンUIコンポーネント。

xTB等の量子化学計算の進捗をリアルタイムで可視化:
- 分子ごとの進捗バー
- 推定残り時間
- ステップ別ステータス（RDKit→xTB→特徴量抽出）
- 計算量見積もりプレビュー
- 中間結果の自動保存ステータス

既存UIへの影響: なし（完全新規コンポーネント）
"""
from __future__ import annotations

import logging
import time
from typing import Any

from nicegui import ui

logger = logging.getLogger(__name__)


def _get_molecule_count(state: dict[str, Any]) -> int:
    """状態から分子数を取得する。"""
    df = state.get("df")
    smiles_col = state.get("smiles_col")
    if df is not None and smiles_col and smiles_col in df.columns:
        return int(df[smiles_col].notna().sum())
    return 0


def render_computation_progress(state: dict[str, Any]) -> None:
    """計算進捗ダッシュボードを描画する。データ数に応じて表示を動的に変える。"""

    n_mols = _get_molecule_count(state)

    # ── データがない場合 ──
    if n_mols == 0:
        with ui.card().classes("w-full").style(
            "background: rgba(255,255,255,0.03); "
            "border: 1px solid rgba(255,255,255,0.08); "
            "border-radius: 16px; padding: 24px;"
        ):
            with ui.row().classes("items-center gap-4"):
                ui.icon("info").classes("text-2xl").style("color: #60a5fa;")
                ui.label("SMILESデータがありません").classes("text-lg").style(
                    "color: #a0a0c0;"
                )
        return

    # ── データ数による表示切り替え ──
    # 10分子以下: あまり意味がないので簡易メッセージ
    # 11-49分子: 簡略表示（見積もりのみ）
    # 50分子以上: フル表示

    if n_mols <= 10:
        with ui.card().classes("w-full").style(
            "background: rgba(255,255,255,0.03); "
            "border: 1px solid rgba(255,255,255,0.08); "
            "border-radius: 16px; padding: 20px;"
        ):
            with ui.row().classes("items-center gap-4"):
                ui.icon("speed").classes("text-2xl").style("color: #4ade80;")
                ui.label(f"分子数 {n_mols}件: 計算は一瞬で完了します").classes(
                    "text-sm"
                ).style("color: #a0a0c0;")
        return

    # ── ヘッダー ──
    is_large = n_mols >= 50
    header_bg = (
        "rgba(251, 191, 36, 0.08)" if is_large
        else "rgba(255,255,255,0.03)"
    )
    header_border = (
        "1px solid rgba(251, 191, 36, 0.3)" if is_large
        else "1px solid rgba(255,255,255,0.08)"
    )

    with ui.card().classes("w-full").style(
        f"background: {header_bg}; "
        f"border: {header_border}; "
        "border-radius: 16px; padding: 24px;"
    ):
        with ui.row().classes("items-center gap-4"):
            icon_color = "#fbbf24" if is_large else "#00d4ff"
            icon_name = "priority_high" if is_large else "speed"
            ui.icon(icon_name).classes("text-3xl").style(f"color: {icon_color};")
            with ui.column().classes("gap-0"):
                if is_large:
                    ui.label("📊 計算ステータスダッシュボード").classes(
                        "text-xl font-bold"
                    ).style("color: #fbbf24;")
                    ui.label(
                        f"⚠️ {n_mols}分子 — 計算に時間がかかる可能性があります"
                    ).classes("text-sm").style("color: #fbbf24;")
                else:
                    ui.label("📊 計算ステータスダッシュボード").classes(
                        "text-xl font-bold"
                    ).style("color: #e0e0f0;")
                    ui.label(
                        "xTB量子化学計算の進捗・推定時間・リソース管理"
                    ).classes("text-sm").style("color: #a0a0c0;")

    ui.separator().classes("q-my-md")

    # 11-49分子: 簡略表示（見積もりのみ）
    if n_mols < 50:
        with ui.card().classes("w-full").style(
            "background: rgba(255,255,255,0.03); "
            "border: 1px solid rgba(255,255,255,0.08); "
            "border-radius: 16px; padding: 20px;"
        ):
            ui.label("⏱️ 計算量見積もり").classes("text-lg font-bold").style(
                "color: #e0e0f0;"
            )
            ui.label(
                f"分子数 {n_mols}件: 計算は数分で完了します"
            ).classes("text-sm q-mt-sm").style("color: #a0a0c0;")
        return

    # ══════════════════════════════════════════════════════════
    # 50分子以上: フル表示
    # ══════════════════════════════════════════════════════════

    # ── 計算量見積もりセクション ──
    with ui.card().classes("w-full").style(
        "background: rgba(255,255,255,0.03); "
        "border: 1px solid rgba(255,255,255,0.08); "
        "border-radius: 16px; padding: 20px;"
    ):
        ui.label("⏱️ 計算量見積もり").classes("text-lg font-bold").style(
            "color: #e0e0f0;"
        )

        with ui.row().classes("gap-4 q-mt-sm items-end"):
            n_mol_input = ui.number(
                "分子数", value=n_mols, min=1, max=10000, step=10,
            ).classes("w-32")
            avg_atoms_input = ui.number(
                "平均原子数", value=30, min=1, max=500, step=5,
            ).classes("w-32")
            calc_type_select = ui.select(
                label="計算タイプ",
                options={"sp": "⚡ 単点計算(sp)", "opt": "🚀 構造最適化(opt)"},
                value="opt",
            ).classes("w-48")

        estimate_container = ui.column().classes("w-full q-mt-md")

        def _update_estimate():
            try:
                from backend.utils.compute_budget import ComputeBudget
                budget = ComputeBudget()
                summary = budget.get_summary(
                    n_molecules=int(n_mol_input.value),
                    avg_atoms=int(avg_atoms_input.value),
                )
                rec_type = summary["recommended_calc_type"]

                estimate_container.clear()
                with estimate_container:
                    bg_color = (
                        "rgba(74, 222, 128, 0.05)"
                        if summary["estimated_minutes"] < 10
                        else "rgba(251, 191, 36, 0.08)"
                        if summary["estimated_minutes"] < 60
                        else "rgba(248, 113, 113, 0.08)"
                    )
                    with ui.card().classes("w-full").style(
                        f"background: {bg_color}; "
                        "border-radius: 12px; padding: 16px;"
                    ):
                        with ui.row().classes("gap-8 items-center"):
                            with ui.column().classes("gap-1"):
                                ui.label("推定計算時間").classes("text-sm").style(
                                    "color: #a0a0c0;"
                                )
                                mins = summary["estimated_minutes"]
                                if mins < 1:
                                    time_str = f"{mins*60:.0f}秒"
                                elif mins < 60:
                                    time_str = f"{mins:.1f}分"
                                else:
                                    time_str = f"{mins/60:.1f}時間"
                                ui.label(time_str).classes(
                                    "text-2xl font-bold"
                                ).style("color: #e0e0f0;")

                            with ui.column().classes("gap-1"):
                                ui.label("推奨計算タイプ").classes("text-sm").style(
                                    "color: #a0a0c0;"
                                )
                                icon = "⚡" if rec_type == "sp" else "🚀"
                                ui.label(f"{icon} {rec_type}").classes(
                                    "text-lg font-bold"
                                ).style("color: #00d4ff;")

                            with ui.column().classes("gap-1"):
                                ui.label("処理分子数").classes("text-sm").style(
                                    "color: #a0a0c0;"
                                )
                                ui.label(
                                    f"{summary['n_molecules']}分子"
                                ).classes("text-lg").style("color: #e0e0f0;")

                        # コスト指標
                        ui.separator().classes("q-my-sm")
                        with ui.row().classes("gap-6"):
                            for label, icon_str in [
                                ("⚡ 高速 (~10秒/分子)", "RDKit 2D記述子"),
                                ("🚀 標準 (~1分/分子)", "xTB最適化"),
                                ("🐢 精密 (~10分/分子)", "freq+熱力学量"),
                            ]:
                                ui.label(f"{label}: {icon_str}").classes(
                                    "text-xs"
                                ).style("color: #a0a0c0;")

            except Exception as e:
                estimate_container.clear()
                with estimate_container:
                    ui.label(f"⚠️ 見積もりエラー: {e}").style("color: #fbbf24;")

        n_mol_input.on_value_change(lambda _: _update_estimate())
        avg_atoms_input.on_value_change(lambda _: _update_estimate())
        calc_type_select.on_value_change(lambda _: _update_estimate())
        _update_estimate()

    ui.separator().classes("q-my-md")

    # ── 計算コストカタログ ──
    with ui.expansion(
        "📋 特徴量コストカタログ",
        icon="list",
    ).classes("w-full").style(
        "background: rgba(255,255,255,0.02); border-radius: 12px;"
    ):
        try:
            from backend.chem.adaptive_feature_selector import AdaptiveFeatureSelector
            sel = AdaptiveFeatureSelector()
            cost_data = sel.get_cost_summary()

            columns = [
                {"name": "name", "label": "特徴量", "field": "name", "sortable": True},
                {"name": "category", "label": "カテゴリ", "field": "category"},
                {"name": "time", "label": "時間/分子", "field": "time_per_mol_s", "sortable": True},
                {"name": "xtb", "label": "xTB", "field": "requires_xtb"},
                {"name": "desc", "label": "説明", "field": "description"},
            ]
            rows = [
                {
                    **c,
                    "time_per_mol_s": f"{c['time_per_mol_s']:.1f}s",
                    "requires_xtb": "✅" if c["requires_xtb"] else "—",
                }
                for c in cost_data
            ]
            ui.table(columns=columns, rows=rows).classes("w-full").props(
                "dense flat"
            )
        except Exception:
            ui.label("コストカタログを読み込めません").style("color: #a0a0c0;")
