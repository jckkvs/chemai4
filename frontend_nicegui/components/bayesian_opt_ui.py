# -*- coding: utf-8 -*-
"""
frontend_nicegui/components/bayesian_opt_ui.py

ベイズ最適化UI コンポーネント（NiceGUI版）。
5ステップウィザード形式:
  Step1: 目的設定（最小化/最大化/目標範囲、カーネル、獲得関数）
  Step2: 探索空間設定（変数ごとの範囲）
  Step3: 制約設定（合計/AtLeastN/カスタム式）
  Step4: 候補生成（Grid/Random/LHS + バッチ戦略）
  Step5: 結果表示（テーブル/PCA/CSV出力）
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui


def render_bayesian_opt_ui(state: dict) -> None:
    """ベイズ最適化UIをレンダリング。"""
    df = state.get("df")
    target_col = state.get("target_col")

    if df is None or not target_col:
        ui.label("データまたは目的変数が未設定です。").classes("text-caption text-grey")
        return

    with ui.card().classes("full-width q-pa-md").style(
        "border:1px solid rgba(0,188,212,0.3);border-radius:12px;"
        "background:rgba(0,20,40,0.25);"
    ):
        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            ui.icon("science", color="amber").classes("text-h5")
            ui.label("🎯 ベイズ最適化 — 実験計画").classes("text-h6")

        ui.label(
            "学習済みモデルの知見に基づき、次の実験候補をベイズ最適化で提案します。"
        ).classes("text-caption text-grey q-mb-sm")

        # BO内部state
        bo = state.setdefault("_bo", {
            "obj_type": "最小化",
            "multi_obj": False,
            "kernel": "default",
            "acq_fn": "ei",
            "xi": 0.01,
            "kappa": 2.0,
            "gen_method": "auto",
            "n_max": 50000,
            "n_suggest": 5,
            "strategy": "kriging_believer",
            "suggestions": None,
        })

        numeric_cols = list(df.select_dtypes(include="number").columns)
        feature_cols = [c for c in numeric_cols if c != target_col]

        # ── 5ステップタブ ──
        with ui.tabs().classes("full-width").props(
            "dense no-caps active-color=amber indicator-color=amber"
        ) as bo_tabs:
            ui.tab("bo_obj", label="1⃣ 目的設定")
            ui.tab("bo_space", label="2⃣ 探索空間")
            ui.tab("bo_constr", label="3⃣ 制約")
            ui.tab("bo_gen", label="4⃣ 候補生成")
            ui.tab("bo_result", label="5⃣ 結果")

        with ui.tab_panels(bo_tabs, value="bo_obj").classes("full-width"):

            # ═══ Step 1: 目的設定 ═══
            with ui.tab_panel("bo_obj"):
                ui.label("目的設定").classes("text-subtitle1 text-bold")

                ui.select(
                    ["最小化", "最大化", "目標範囲"],
                    value=bo.get("obj_type", "最小化"),
                    label="最適化の方向",
                    on_change=lambda e: bo.update({"obj_type": e.value}),
                ).props("outlined dense").classes("col-4 q-mb-sm")

                # カーネル設定
                ui.label("カーネル設定").classes("text-body2 text-bold q-mt-sm")
                ui.select(
                    ["default", "matern", "dotproduct"],
                    value=bo.get("kernel", "default"),
                    label="カーネルタイプ",
                    on_change=lambda e: bo.update({"kernel": e.value}),
                ).props("outlined dense").classes("col-4 q-mb-sm")

                # 獲得関数
                ui.label("獲得関数").classes("text-body2 text-bold q-mt-sm")
                ui.select(
                    ["ei", "pi", "ucb", "ptr"],
                    value=bo.get("acq_fn", "ei"),
                    label="獲得関数",
                    on_change=lambda e: bo.update({"acq_fn": e.value}),
                ).props("outlined dense").classes("col-4 q-mb-sm")

                ui.label("✅ Step 1 設定完了").classes("text-positive text-caption q-mt-sm")

            # ═══ Step 2: 探索空間 ═══
            with ui.tab_panel("bo_space"):
                ui.label("探索空間設定").classes("text-subtitle1 text-bold")
                ui.label("各特徴量の探索範囲を設定します。").classes("text-caption text-grey q-mb-sm")

                if not feature_cols:
                    ui.label("探索対象の特徴量がありません。").classes("text-caption text-grey")
                else:
                    # 範囲テーブル
                    range_data = []
                    for col in feature_cols:
                        col_data = df[col].dropna()
                        range_data.append({
                            "変数": col,
                            "下限": round(float(col_data.min() * 0.9), 4),
                            "上限": round(float(col_data.max() * 1.1), 4),
                            "データ最小": round(float(col_data.min()), 4),
                            "データ最大": round(float(col_data.max()), 4),
                        })
                    range_df = pd.DataFrame(range_data)
                    columns = [
                        {"name": c, "label": c, "field": c, "sortable": True}
                        for c in range_df.columns
                    ]
                    ui.table(
                        columns=columns,
                        rows=range_df.to_dict("records"),
                    ).classes("full-width").props("dense flat separator=cell")

                    # 候補生成方法
                    ui.select(
                        ["auto", "grid", "random", "lhs"],
                        value=bo.get("gen_method", "auto"),
                        label="候補生成方法",
                        on_change=lambda e: bo.update({"gen_method": e.value}),
                    ).props("outlined dense").classes("col-4 q-mt-sm")

                    ui.number(
                        "最大候補数", value=bo.get("n_max", 50000),
                        min=1000, max=200000, step=1000,
                        on_change=lambda e: bo.update({"n_max": int(e.value)}),
                    ).props("outlined dense").classes("col-4 q-mt-xs")

                    ui.label("✅ Step 2 設定完了").classes("text-positive text-caption q-mt-sm")

            # ═══ Step 3: 制約 ═══
            with ui.tab_panel("bo_constr"):
                ui.label("制約設定").classes("text-subtitle1 text-bold")
                from frontend_nicegui.components.constraint_ui_helpers import (
                    render_constraint_builder,
                )
                constraint_state = state.setdefault("_bo_constraints_state", {})
                render_constraint_builder(
                    feature_cols=feature_cols,
                    constraint_state=constraint_state,
                )
                ui.label("✅ Step 3 設定完了").classes("text-positive text-caption q-mt-sm")

            # ═══ Step 4: 候補生成 ═══
            with ui.tab_panel("bo_gen"):
                ui.label("候補生成").classes("text-subtitle1 text-bold")

                ui.number(
                    "提案候補数", value=bo.get("n_suggest", 5),
                    min=1, max=50, step=1,
                    on_change=lambda e: bo.update({"n_suggest": int(e.value)}),
                ).props("outlined dense").classes("col-4 q-mb-sm")

                ui.select(
                    ["kriging_believer", "single", "doe_then_bo", "bo_then_doe"],
                    value=bo.get("strategy", "kriging_believer"),
                    label="バッチ戦略",
                    on_change=lambda e: bo.update({"strategy": e.value}),
                ).props("outlined dense").classes("col-4 q-mb-sm")

                gen_result_container = ui.column().classes("full-width q-mt-sm")

                async def _generate():
                    ui.notify("🔄 候補を生成中...", type="info", timeout=3000)
                    try:
                        from nicegui import run
                        from backend.optim.search_space import SearchSpace
                        from backend.optim.bayesian_optimizer import BOConfig, BayesianOptimizer

                        space = SearchSpace.from_dataframe(df, columns=feature_cols, margin=0.1)
                        candidates_df = space.generate_candidates(
                            method=bo["gen_method"], n_max=bo["n_max"],
                        )

                        config = BOConfig(
                            objective={"最小化": "minimize", "最大化": "maximize", "目標範囲": "minimize"}.get(bo["obj_type"], "minimize"),
                            acquisition=bo["acq_fn"],
                            xi=bo.get("xi", 0.01),
                            kappa=bo.get("kappa", 2.0),
                            kernel_type=bo.get("kernel", "default"),
                            batch_strategy=bo["strategy"],
                            n_candidates=bo["n_suggest"],
                        )

                        X_train = df[feature_cols].values
                        y_train = df[target_col].values

                        optimizer = BayesianOptimizer(config)
                        optimizer.fit(X_train, y_train)
                        suggestions = optimizer.suggest(candidates_df[feature_cols], n=bo["n_suggest"])

                        bo["suggestions"] = suggestions
                        state["_bo_optimizer"] = optimizer

                        ui.notify(f"✅ {len(suggestions)}件の候補を生成しました！", type="positive")

                        gen_result_container.clear()
                        with gen_result_container:
                            ui.label(f"生成完了: {len(suggestions)}件").classes("text-positive")

                    except ImportError as ie:
                        ui.notify(f"⚠️ モジュール未インストール: {ie}", type="warning")
                    except Exception as e:
                        ui.notify(f"⚠️ エラー: {e}", type="warning")

                ui.button("🚀 候補を生成する", on_click=_generate).props(
                    "color=amber no-caps"
                ).classes("q-mt-sm")

            # ═══ Step 5: 結果 ═══
            with ui.tab_panel("bo_result"):
                ui.label("結果").classes("text-subtitle1 text-bold")

                suggestions = bo.get("suggestions")
                if suggestions is None:
                    ui.label("まだ候補が生成されていません。Step 4で生成ボタンを押してください。").classes(
                        "text-caption text-grey"
                    )
                else:
                    # テーブル表示
                    if isinstance(suggestions, pd.DataFrame):
                        columns = [
                            {"name": c, "label": c, "field": c, "sortable": True}
                            for c in suggestions.columns
                        ]
                        ui.table(
                            columns=columns,
                            rows=suggestions.to_dict("records"),
                        ).classes("full-width").props("dense flat separator=cell")

                        # CSVダウンロード
                        csv_buf = io.StringIO()
                        suggestions.to_csv(csv_buf, index=False, encoding="utf-8-sig")
                        ui.button(
                            "📥 CSVダウンロード",
                            on_click=lambda: ui.download(
                                csv_buf.getvalue().encode("utf-8-sig"),
                                "bo_suggestions.csv",
                            ),
                        ).props("color=cyan no-caps size=sm").classes("q-mt-sm")
