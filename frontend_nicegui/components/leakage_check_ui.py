# -*- coding: utf-8 -*-
"""
frontend_nicegui/components/leakage_check_ui.py

リーケージ検出の事前チェックUIコンポーネント（NiceGUI版）。
解析実行前にデータのリーケージリスクを評価し、
適切な CV 戦略を提案する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from nicegui import ui


def render_leakage_check_panel(state: dict) -> None:
    """リーケージ事前チェックパネルを描画する。"""
    df = state.get("df")
    target_col = state.get("target_col")
    if df is None or not target_col:
        ui.label("データまたは目的変数が未設定です").classes("text-caption text-grey")
        return

    with ui.expansion("🔍 リーケージ事前チェック（推奨）").classes("full-width q-mb-sm").props("dense"):
        ui.label(
            "説明変数間のサンプル類似度を分析し、train/testリーケージのリスクを評価します。"
        ).classes("text-caption text-grey q-mb-sm")

        # 手法選択
        with ui.row().classes("items-center q-gutter-sm full-width"):
            method_sel = ui.select(
                ["auto（自動選択）", "hat（ハット行列）", "rbf（RBFカーネル）", "rf（ランダムフォレスト）"],
                value="auto（自動選択）",
                label="類似度推定手法",
            ).props("outlined dense").classes("col-6")

            threshold_slider = ui.slider(
                min=0.80, max=0.999, step=0.01, value=0.95,
            ).props("label-always").classes("col-4")
            ui.label("類似度閾値").classes("text-caption")

        # 結果表示コンテナ
        result_container = ui.column().classes("full-width q-mt-sm")

        async def _run_check():
            method_key = method_sel.value.split("（")[0]
            threshold = threshold_slider.value

            # 説明変数のみ抽出
            exclude_cols = [target_col]
            smiles_col = state.get("smiles_col")
            if smiles_col:
                exclude_cols.append(smiles_col)
            exclude_cols.extend(state.get("exclude_cols", []))

            feature_cols = [c for c in df.columns if c not in exclude_cols]
            X_check = df[feature_cols].select_dtypes(include=[np.number])
            y_check = df[target_col]

            if X_check.shape[1] < 2:
                ui.notify("⚠️ 数値の説明変数が2列未満のため、チェックできません。", type="warning")
                return

            ui.notify("🔍 類似度を分析中...", type="info", timeout=3000)

            try:
                from nicegui import run
                from backend.data.leakage_detector import detect_leakage
                report = await run.io_bound(
                    detect_leakage, X_check, y_check,
                    method=method_key,
                    similarity_threshold=threshold,
                )
                state["leakage_report"] = report

                # 結果表示
                result_container.clear()
                with result_container:
                    _render_result(report, X_check.shape[0], state)

            except ImportError:
                ui.notify("⚠️ backend.data.leakage_detector が見つかりません", type="warning")
            except Exception as e:
                ui.notify(f"⚠️ リーケージチェックエラー: {e}", type="warning")

        ui.button("🔍 リーケージチェック実行", on_click=_run_check).props(
            "color=cyan no-caps"
        ).classes("q-mt-sm")


def _render_result(report, n_samples: int, state: dict) -> None:
    """リーケージ検出結果を表示。"""
    risk_config = {
        "low": ("🟢", "低リスク", "positive"),
        "medium": ("🟡", "中リスク", "warning"),
        "high": ("🔴", "高リスク", "negative"),
    }
    icon, label, color_type = risk_config.get(
        report.risk_level, ("⚪", "不明", "info")
    )

    # リスクレベル
    ui.label(f"{icon} リーケージリスク: {label}（スコア: {report.risk_score:.2f})").classes(
        "text-subtitle1 text-bold"
    )

    # メトリクス
    with ui.row().classes("q-gutter-sm"):
        with ui.card().classes("q-pa-xs").style("min-width:80px;text-align:center;"):
            ui.label("手法").classes("text-caption text-grey")
            ui.label(report.method_used.upper()).classes("text-bold")
        with ui.card().classes("q-pa-xs").style("min-width:80px;text-align:center;"):
            ui.label("疑わしいペア").classes("text-caption text-grey")
            ui.label(f"{report.n_suspicious_pairs:,}").classes("text-bold")
        with ui.card().classes("q-pa-xs").style("min-width:80px;text-align:center;"):
            ui.label("グループ数").classes("text-caption text-grey")
            ui.label(str(report.n_groups) if report.n_groups > 0 else "—").classes("text-bold")

    # CV推奨
    if report.risk_level == "low":
        ui.label(f"✅ 推奨CV: {report.recommended_cv} — {report.cv_reason}").classes("text-positive q-mt-sm")
    elif report.risk_level == "medium":
        ui.label(f"⚠️ 推奨CV: {report.recommended_cv} — {report.cv_reason}").classes("text-warning q-mt-sm")
    else:
        ui.label(f"🚨 推奨CV: {report.recommended_cv} — {report.cv_reason}").classes("text-negative q-mt-sm")

    # グループラベル適用ボタン
    if report.group_labels is not None and report.n_groups >= 2:
        def _apply_groups():
            state["leakage_group_labels"] = report.group_labels
            ui.notify("グループラベルをパイプラインに適用しました。", type="positive")

        ui.button("✅ グループラベルを適用", on_click=_apply_groups).props(
            "color=teal no-caps size=sm"
        ).classes("q-mt-sm")
