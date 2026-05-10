"""
frontend_nicegui/components/batch_predict_tab.py

バッチ予測タブ：学習済みモデルに新データを適用して予測結果をCSVダウンロード。

Implements: 会話 82f7fa3b — バッチ予測タブ

設計思想:
  - 解析完了後にのみ表示
  - 新CSVをアップロード → 学習済みパイプラインで予測 → 結果ダウンロード
  - 元データ + 予測値を結合してCSV出力
"""
from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui

logger = logging.getLogger(__name__)


def render_batch_predict_tab(state: dict[str, Any]) -> None:
    """バッチ予測タブを描画する。"""

    ar = state.get("automl_result")
    if ar is None:
        with ui.card().classes("glass-card q-pa-xl full-width"):
            ui.icon("pending", color="grey-7", size="xl").classes("q-mb-md")
            ui.label("モデルがまだ学習されていません").classes("text-h6 text-grey-5")
            ui.label(
                "「📂 解析設定」→「🚀 解析開始」でモデルを学習してから、"
                "バッチ予測を利用してください。"
            ).classes("text-grey-6 q-mt-sm")
        return

    pipeline = getattr(ar, "best_pipeline", None)
    if pipeline is None:
        ui.label("⚠️ 学習済みパイプラインが取得できませんでした").classes("text-amber")
        return

    # ── ヘッダー ──
    with ui.row().classes("items-center q-gutter-sm full-width"):
        ui.icon("batch_prediction").classes("text-purple text-h5")
        ui.label("バッチ予測").classes("text-h6")
        ui.badge(f"最良モデル: {ar.best_model_key}", color="purple").props("outline")

    ui.label(
        "新しいCSVデータをアップロードすると、学習済みモデルで予測を実行します。"
        "SMILESが含まれる場合は記述子も自動計算されます。"
    ).classes("text-caption text-grey q-mb-md")

    # ── 予測結果表示用コンテナ ──
    pred_container = ui.column().classes("full-width q-mt-md")

    # ── チェック対象の列情報 ──
    target_col = state.get("target_col", "target")
    smiles_col = state.get("smiles_col", "")

    with ui.card().classes("glass-card q-pa-md full-width"):
        ui.label("📋 学習時の設定").classes("text-subtitle2")
        with ui.row().classes("q-gutter-md"):
            ui.chip(f"目的変数: {target_col}", icon="label", color="cyan").props("outline dense")
            ui.chip(f"タスク: {ar.task}", icon="category", color="teal").props("outline dense")
            ui.chip(f"スコア: {ar.best_score:.4f}", icon="star", color="amber").props("outline dense")
            if smiles_col:
                ui.chip(f"SMILES列: {smiles_col}", icon="science", color="green").props("outline dense")

    ui.separator().classes("q-my-md")

    # ── ファイルアップロード ──
    async def _on_upload(e):
        content = e.content.read()
        name = e.name
        try:
            if name.endswith(".csv"):
                new_df = pd.read_csv(io.BytesIO(content))
            elif name.endswith((".xlsx", ".xls")):
                new_df = pd.read_excel(io.BytesIO(content))
            else:
                ui.notify("❌ CSV/Excelファイルのみ対応", type="warning")
                return

            ui.notify(f"📂 {name} 読込完了 ({len(new_df)}行 × {new_df.shape[1]}列)", type="info")

            # SMILES記述子計算（必要な場合）
            desc_df = None
            if smiles_col and smiles_col in new_df.columns:
                try:
                    from backend.chem.descriptors import compute_all_descriptors
                    smiles_list = new_df[smiles_col].dropna().tolist()
                    if smiles_list:
                        ui.notify("⚗️ SMILES記述子を計算中...", type="info", timeout=3000)
                        from nicegui import run
                        desc_df = await run.io_bound(compute_all_descriptors, smiles_list)
                        ui.notify(f"✅ 記述子{desc_df.shape[1]}個計算完了", type="positive")
                except Exception as ex:
                    logger.warning(f"記述子計算エラー: {ex}")
                    ui.notify(f"⚠️ 記述子計算エラー: {ex}", type="warning")

            # 特徴量の準備
            X_new = new_df.copy()

            # 目的変数列は除外
            if target_col in X_new.columns:
                X_new = X_new.drop(columns=[target_col])

            # SMILES列は除外（記述子で代替）
            if smiles_col and smiles_col in X_new.columns:
                X_new = X_new.drop(columns=[smiles_col])

            # 除外列を除外
            exclude_cols = state.get("exclude_cols", [])
            drop_cols = [c for c in exclude_cols if c in X_new.columns]
            if drop_cols:
                X_new = X_new.drop(columns=drop_cols)

            # 記述子を結合
            if desc_df is not None:
                desc_df = desc_df.iloc[:len(X_new)].reset_index(drop=True)
                X_new = X_new.reset_index(drop=True)
                X_new = pd.concat([X_new, desc_df], axis=1)

            # 予測実行
            try:
                predictions = pipeline.predict(X_new)
                new_df["_predicted_" + target_col] = np.nan
                new_df.loc[:len(predictions) - 1, "_predicted_" + target_col] = predictions

                # 分類タスクの確信度
                if ar.task == "classification" and hasattr(pipeline, "predict_proba"):
                    try:
                        proba = pipeline.predict_proba(X_new)
                        if proba.ndim == 2:
                            for i in range(proba.shape[1]):
                                new_df[f"_proba_class_{i}"] = np.nan
                                new_df.loc[:len(proba) - 1, f"_proba_class_{i}"] = proba[:, i]
                    except Exception:
                        pass

                _show_prediction_results(new_df, target_col, name, pred_container)
                ui.notify(f"✅ {len(predictions)}件の予測完了", type="positive")

            except Exception as ex:
                ui.notify(f"❌ 予測エラー: {ex}", type="negative")
                logger.error(f"バッチ予測エラー: {ex}", exc_info=True)

        except Exception as ex:
            ui.notify(f"❌ ファイル読込エラー: {ex}", type="negative")

    ui.upload(
        on_upload=_on_upload,
        label="新データCSV/Excelをドラッグ＆ドロップ",
        auto_upload=True,
    ).props('accept=".csv,.xlsx,.xls" color="purple"').classes("full-width")

    # ── サンプル予測（既存データで動作確認） ──
    with ui.expansion("🧪 既存データでテスト予測", icon="science").classes("full-width q-mt-sm"):
        ui.label(
            "学習に使ったデータの先頭5行でテスト予測を実行します（動作確認用）。"
        ).classes("text-caption text-grey q-mb-sm")

        async def _test_predict():
            df = state.get("df")
            if df is None:
                ui.notify("データがありません", type="warning")
                return

            test_df = df.head(5).copy()

            # 特徴量の準備
            X_test = test_df.copy()
            if target_col in X_test.columns:
                X_test = X_test.drop(columns=[target_col])
            if smiles_col and smiles_col in X_test.columns:
                X_test = X_test.drop(columns=[smiles_col])
            exclude_cols = state.get("exclude_cols", [])
            drop_cols = [c for c in exclude_cols if c in X_test.columns]
            if drop_cols:
                X_test = X_test.drop(columns=drop_cols)

            # 記述子結合
            precalc = state.get("precalc_df")
            if precalc is not None:
                desc_slice = precalc.head(5).reset_index(drop=True)
                X_test = X_test.reset_index(drop=True)
                X_test = pd.concat([X_test, desc_slice], axis=1)

            try:
                preds = pipeline.predict(X_test)
                test_df["_predicted_" + target_col] = preds
                _show_prediction_results(test_df, target_col, "test_sample", pred_container)
                ui.notify("✅ テスト予測完了", type="positive")
            except Exception as ex:
                ui.notify(f"❌ テスト予測エラー: {ex}", type="negative")

        ui.button("🧪 テスト予測を実行", on_click=_test_predict).props(
            "outline color=purple size=sm no-caps"
        )


def _show_prediction_results(
    df: pd.DataFrame, target_col: str, source_name: str, container,
) -> None:
    """予測結果をテーブル表示＋CSVダウンロード。"""
    container.clear()
    pred_col = "_predicted_" + target_col

    with container:
        ui.label(f"📊 予測結果: {len(df)}行").classes("text-subtitle1 q-mt-md")

        # メトリクスカード
        with ui.row().classes("q-gutter-md"):
            with ui.card().classes("glass-card q-pa-sm"):
                ui.label(f"{len(df):,}").classes("text-h5 text-bold hero-gradient")
                ui.label("予測件数").classes("text-caption text-grey-5")
            if pred_col in df.columns:
                preds = df[pred_col].dropna()
                if len(preds) > 0:
                    with ui.card().classes("glass-card q-pa-sm"):
                        ui.label(f"{preds.mean():.4g}").classes("text-h5 text-bold hero-gradient")
                        ui.label("予測平均").classes("text-caption text-grey-5")
                    with ui.card().classes("glass-card q-pa-sm"):
                        ui.label(f"{preds.std():.4g}").classes("text-h5 text-bold hero-gradient")
                        ui.label("予測標準偏差").classes("text-caption text-grey-5")

        # テーブルプレビュー（先頭30行）
        ui.separator()
        preview = df.head(30)
        show_cols = [c for c in preview.columns if c.startswith("_predicted_") or c.startswith("_proba_")]
        # 元データの先頭3列 + 予測列
        orig_cols = [c for c in preview.columns if not c.startswith("_")][:5]
        display_cols = orig_cols + show_cols

        columns = [
            {"name": col, "label": col, "field": col, "align": "left", "sortable": True}
            for col in display_cols
        ]
        rows = []
        for _, row in preview.iterrows():
            row_dict = {}
            for col in display_cols:
                v = row.get(col)
                if pd.isna(v):
                    row_dict[col] = "—"
                elif isinstance(v, float):
                    row_dict[col] = f"{v:.4g}"
                else:
                    row_dict[col] = str(v)
            rows.append(row_dict)

        ui.table(columns=columns, rows=rows).classes("full-width").props("dense flat bordered")

        if len(df) > 30:
            ui.label(f"... 他 {len(df) - 30} 行").classes("text-caption text-grey-6")

        # CSVダウンロード
        csv_data = df.to_csv(index=False)
        ui.button(
            "📥 予測結果をCSVダウンロード",
            on_click=lambda: ui.download(
                csv_data.encode("utf-8"), f"predictions_{source_name}.csv"
            ),
        ).props("color=purple unelevated no-caps").classes("q-mt-md btn-primary")
