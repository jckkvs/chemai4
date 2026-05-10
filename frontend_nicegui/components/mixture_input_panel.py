"""
frontend_nicegui/components/mixture_input_panel.py

混合物入力パネル — スタンドアロンUIコンポーネント。

- 動的行追加 (2〜任意数)
- 比率タイプ切り替え (重量比/mol比/その他)
- CSVテンプレートダウンロード
- CSVアップロード（wide形式・long形式自動判別）
- 混合物特徴量計算のトリガー
- 計算結果をメインDataFrame (state["df"]) に統合
- 特徴量ごとの加重方法カスタマイズUI
- 計算結果のCSVエクスポート

既存UIへの影響: なし（完全新規コンポーネント）
main.pyのタブパネル内でrender_mixture_panel(state)で呼び出して使用。
"""
from __future__ import annotations

import io
import logging
from typing import Any

from nicegui import ui

logger = logging.getLogger(__name__)


def render_mixture_panel(state: dict[str, Any]) -> None:
    """混合物入力パネルを描画する。"""

    # ── 内部状態 ──
    mixture_state: dict[str, Any] = {
        "components": [],       # [{smiles, name, ratio, row_ref}, ...]
        "ratio_type": "weight",
        "other_unit": "",
        "result": None,
    }

    # ═══════════════════════════════════════════════════════════
    # ヘッダー
    # ═══════════════════════════════════════════════════════════
    with ui.card().classes("w-full").style(
        "background: rgba(255,255,255,0.03); "
        "border: 1px solid rgba(255,255,255,0.08); "
        "border-radius: 16px; padding: 24px;"
    ):
        with ui.row().classes("items-center gap-4 w-full"):
            ui.icon("science").classes("text-3xl").style("color: #a78bfa;")
            with ui.column().classes("gap-0"):
                ui.label("🧪 混合物特徴量計算").classes("text-xl font-bold").style(
                    "color: #e0e0f0;"
                )
                ui.label(
                    "複数化合物の混合比を指定し、加重平均記述子を計算"
                ).classes("text-sm").style("color: #a0a0c0;")

    ui.separator().classes("q-my-md")

    # ═══════════════════════════════════════════════════════════
    # CSVテンプレートダウンロード
    # ═══════════════════════════════════════════════════════════
    with ui.expansion(
        "📥 CSVテンプレート",
        icon="download",
    ).classes("w-full").style(
        "background: rgba(255,255,255,0.02); border-radius: 12px;"
    ):
        ui.markdown(
            "- 同一行が1混合物として処理されます\n"
            "- `Compound_N_SMILES`: 成分NのSMILES\n"
            "- `Compound_N_WT%`: 成分Nの重量パーセンテージ\n"
            "- `Target_Property`: 目的変数値（ML学習用・任意）\n"
            "- 複数混合物を一括処理できます"
        ).classes("text-sm").style("color: #a0a0c0;")

        with ui.row().classes("gap-4 q-mt-sm"):
            def _download_template():
                try:
                    from backend.chem.mixture_csv_template import generate_template_csv
                    csv_bytes = generate_template_csv()
                    ui.download(
                        csv_bytes,
                        "chemai2_mixture_template_v2.0.csv",
                    )
                    ui.notify("📥 テンプレートをダウンロードしました", type="positive")
                except Exception as e:
                    ui.notify(f"❌ ダウンロード失敗: {e}", type="negative")

            ui.button(
                "⬇️ CSVテンプレートをダウンロード",
                on_click=_download_template,
                icon="download",
            ).props("outline color=cyan")

        # CSVアップロード
        ui.separator().classes("q-my-sm")
        ui.label("📤 CSVファイルをアップロード").classes("text-sm font-bold").style(
            "color: #e0e0f0;"
        )

        async def _on_csv_upload(e):
            try:
                from backend.chem.mixture_csv_template import parse_mixture_csv
                content = e.content.read()
                mixtures = parse_mixture_csv(content)
                if not mixtures:
                    ui.notify("❌ 混合物データが見つかりません", type="negative")
                    return

                state["_mixture_csv_parsed"] = mixtures
                ui.notify(
                    f"✅ {len(mixtures)}件の混合物を読み込みました",
                    type="positive",
                )
                for m in mixtures:
                    for w in m.warnings:
                        ui.notify(f"⚠️ {w}", type="warning")

                # 一括して特徴量計算 + DataFrame統合
                await _batch_process_mixtures(mixtures)
            except Exception as ex:
                ui.notify(f"❌ CSV解析エラー: {ex}", type="negative")
                logger.error("CSV upload error: %s", ex, exc_info=True)

        ui.upload(
            on_upload=_on_csv_upload,
            label="CSVファイルを選択",
            auto_upload=True,
        ).props("accept=.csv").classes("w-full")

    ui.separator().classes("q-my-md")

    # ═══════════════════════════════════════════════════════════
    # 手動入力セクション
    # ═══════════════════════════════════════════════════════════
    with ui.card().classes("w-full").style(
        "background: rgba(255,255,255,0.03); "
        "border: 1px solid rgba(255,255,255,0.08); "
        "border-radius: 16px; padding: 20px;"
    ):
        ui.label("✏️ 手動入力").classes("text-lg font-bold").style(
            "color: #e0e0f0;"
        )

        # ターゲット物性入力
        target_input = ui.number(
            "Target_Property（任意）",
            placeholder="目的変数値を入力",
        ).classes("w-full q-mt-sm")
        ui.label("※ ML学習で目的変数として使用されます").classes("text-xs").style(
            "color: #a0a0c0;"
        )

        # 比率タイプ選択
        with ui.row().classes("items-center gap-4 q-mt-sm"):
            ui.label("比率タイプ:").style("color: #a0a0c0;")
            ratio_radio = ui.radio(
                options={
                    "weight": "⚖️ 重量比",
                    "mole": "🔢 mol比",
                    "other": "⚙️ その他",
                },
                value="weight",
            ).props("inline")

            other_input = ui.input(
                "単位名",
                placeholder="例: volume_fraction",
            ).classes("w-48")
            other_input.set_visibility(False)

            def _on_ratio_change(e):
                mixture_state["ratio_type"] = e.value
                other_input.set_visibility(e.value == "other")

            ratio_radio.on_value_change(_on_ratio_change)

        # 成分入力テーブル
        ui.separator().classes("q-my-sm")
        components_container = ui.column().classes("w-full gap-2")

        def _add_component(order: int | None = None, smiles: str = "", name: str = "", ratio: float = 1.0):
            if order is None:
                order = len(mixture_state["components"]) + 1

            comp_data: dict[str, Any] = {"order": order}

            with components_container:
                with ui.row().classes("w-full items-end gap-2").style(
                    "background: rgba(255,255,255,0.02); "
                    "border-radius: 8px; padding: 8px;"
                ) as row_ref:
                    ui.label(f"#{order}").classes("w-8 text-center font-bold").style(
                        "color: #00d4ff;"
                    )
                    comp_data["smiles_input"] = ui.input(
                        "SMILES", placeholder="例: CCO", value=smiles,
                    ).classes("flex-grow")
                    comp_data["name_input"] = ui.input(
                        "名称（任意）", placeholder="例: ethanol", value=name,
                    ).classes("w-32")
                    comp_data["ratio_input"] = ui.number(
                        "比率", min=0.001, step=0.1, value=ratio,
                    ).classes("w-24")

                    def _remove(ref=row_ref, data=comp_data):
                        ref.delete()
                        if data in mixture_state["components"]:
                            mixture_state["components"].remove(data)

                    ui.button(
                        icon="delete", on_click=_remove,
                    ).props("flat color=red size=sm round")

                    comp_data["row_ref"] = row_ref

            mixture_state["components"].append(comp_data)

        # 初期行（2行）
        _add_component(1)
        _add_component(2)

        with ui.row().classes("q-mt-sm gap-4"):
            ui.button(
                "➕ 成分を追加",
                on_click=lambda: _add_component(),
                icon="add",
            ).props("outline color=cyan")

    ui.separator().classes("q-my-md")

    # ═══════════════════════════════════════════════════════════
    # 実行ボタン + 結果表示
    # ═══════════════════════════════════════════════════════════
    result_container = ui.column().classes("w-full")

    # ── 一括処理関数 ──
    async def _batch_process_mixtures(mixtures):
        """CSVアップロードされた複数混合物を一括処理してDataFrameに統合。"""
        from backend.chem.mixture_feature_extractor import MixtureFeatureExtractor

        user_overrides = state.get("mixture_weighting_overrides", {})
        extractor = MixtureFeatureExtractor(user_overrides=user_overrides)

        all_rows = []
        for m in mixtures:
            try:
                result = extractor.extract(m.components)
                row = result.to_dataframe(strip_prefix=True)

                # 変換情報をカラムとして追加
                info = result.conversion_info
                for i, (wt, mole) in enumerate(zip(info['weight_fractions'], info['mole_fractions'])):
                    row[f"Component_{i+1}_SMILES"] = m.components[i]["smiles"] if i < len(m.components) else ""
                    row[f"Component_{i+1}_weight_frac"] = wt
                    row[f"Component_{i+1}_mole_frac"] = mole

                # Sample_ID
                row["Sample_ID"] = m.session_id

                # Target_Property
                if m.target_property is not None:
                    row["Target_Property"] = m.target_property

                all_rows.append(row)

            except Exception as ex:
                ui.notify(f"❌ 混合物 {m.session_id} の計算エラー: {ex}", type="warning")
                logger.warning("Mixture %s calc error: %s", m.session_id, ex)

        if not all_rows:
            ui.notify("❌ 計算できた混合物がありません", type="negative")
            return

        # DataFrameに統合
        new_df = pd.concat(all_rows, ignore_index=True)

        if state.get("df") is None:
            state["df"] = new_df
        else:
            state["df"] = pd.concat([state["df"], new_df], ignore_index=True)

        # ターゲット設定
        if "Target_Property" in state["df"].columns:
            state["target_col"] = "Target_Property"

        ui.notify(
            f"✅ {len(new_df)}件の混合物特徴量をデータセットに追加しました（計{len(state['df'])}行）",
            type="positive",
        )

        # 結果表示
        _display_results_batch(all_rows, mixtures)

        # UI更新
        _refresh_ui_after_calc()

    # ── バッチ処理結果の表示 ──
    def _display_results_batch(all_rows, mixtures):
        """バッチ処理結果を表示する。"""
        result_container.clear()
        with result_container:
            with ui.card().classes("w-full q-mt-md").style(
                "background: rgba(74, 222, 128, 0.05); "
                "border: 1px solid rgba(74, 222, 128, 0.2); "
                "border-radius: 12px; padding: 16px;"
            ):
                ui.label("✅ 一括計算完了").classes("text-lg font-bold").style(
                    "color: #4ade80;"
                )
                ui.label(f"🧪 {len(all_rows)}件の混合物特徴量を計算").classes(
                    "q-mt-sm text-sm"
                ).style("color: #a0a0c0;")

            # 警告があれば表示
            for m in mixtures:
                for w in m.warnings:
                    ui.label(f"⚠️ {w}").classes("text-sm").style("color: #fbbf24;")

            # CSVエクスポートボタン
            def _export_batch():
                try:
                    export_df = pd.concat(all_rows, ignore_index=True)
                    buf = io.StringIO()
                    export_df.to_csv(buf, index=False, encoding='utf-8-sig')
                    csv_bytes = buf.getvalue().encode('utf-8-sig')
                    ui.download(csv_bytes, f"mixture_batch_results.csv")
                    ui.notify("📥 CSVをダウンロードしました", type="positive")
                except Exception as e:
                    ui.notify(f"❌ エクスポート失敗: {e}", type="negative")

            ui.button(
                "📤 一括結果をCSV出力",
                on_click=_export_batch,
                icon="download",
            ).props("outline color=teal size=sm")

    # ── UI更新関数 ──
    def _refresh_ui_after_calc():
        """計算後にUIを更新する。"""
        # ターゲット物性の表示更新
        if "Target_Property" in state.get("df", pd.DataFrame()).columns:
            if not state.get("target_col"):
                state["target_col"] = "Target_Property"

        # データタブの更新をトリガー
        for key in ["_refresh_eda_main", "_refresh_tabs"]:
            refresh_fn = state.get(key)
            if refresh_fn:
                try:
                    refresh_fn()
                except Exception:
                    pass

    # ── 手動計算実行 ──
    async def _run_mixture_calc():
        components = []
        for comp in mixture_state["components"]:
            smiles = comp["smiles_input"].value.strip()
            if not smiles:
                ui.notify("❌ SMILESが空の成分があります", type="negative")
                return
            components.append({
                "smiles": smiles,
                "compound_name": comp["name_input"].value.strip() or None,
                "ratio_value": float(comp["ratio_input"].value),
                "ratio_unit": mixture_state["ratio_type"],
            })

        if len(components) < 2:
            ui.notify("❌ 成分は2つ以上必要です", type="negative")
            return

        ui.notify("⏳ 混合物特徴量を計算中...", type="info")

        try:
            from backend.chem.mixture_feature_extractor import MixtureFeatureExtractor
            import pandas as pd

            user_overrides = state.get("mixture_weighting_overrides", {})
            extractor = MixtureFeatureExtractor(user_overrides=user_overrides)
            result = extractor.extract(components)
            mixture_state["result"] = result

            # 結果をDataFrameに変換
            mixture_df = result.to_dataframe(strip_prefix=True)

            # 変換情報を追加
            info = result.conversion_info
            for i, (wt, mole) in enumerate(zip(info['weight_fractions'], info['mole_fractions'])):
                mixture_df[f"Component_{i+1}_SMILES"] = components[i]["smiles"] if i < len(components) else ""
                mixture_df[f"Component_{i+1}_weight_frac"] = wt
                mixture_df[f"Component_{i+1}_mole_frac"] = mole

            # Sample_ID
            sample_id = f"MIX_{len(state.get('_mixture_samples', [])) + 1}"
            mixture_df["Sample_ID"] = sample_id

            # Target_Property
            target_value = None
            if hasattr(target_input, 'value') and target_input.value is not None:
                target_value = target_input.value
            if target_value is not None:
                mixture_df["Target_Property"] = target_value

            # state["df"] に統合（重要！）
            if state.get("df") is None:
                state["df"] = mixture_df
            else:
                state["df"] = pd.concat([state["df"], mixture_df], ignore_index=True)

            # ターゲット設定
            if "Target_Property" in state["df"].columns:
                state["target_col"] = "Target_Property"

            # サンプル記録
            if "_mixture_samples" not in state:
                state["_mixture_samples"] = []
            state["_mixture_samples"].append({
                "sample_id": sample_id,
                "features": result.mixture_features,
                "target": target_value,
            })

            # 結果表示
            result_container.clear()
            with result_container:
                # 変換情報
                with ui.card().classes("w-full q-mt-md").style(
                    "background: rgba(74, 222, 128, 0.05); "
                    "border: 1px solid rgba(74, 222, 128, 0.2); "
                    "border-radius: 12px; padding: 16px;"
                ):
                    ui.label("✅ 計算完了").classes("text-lg font-bold").style(
                        "color: #4ade80;"
                    )

                    # 変換テーブル
                    headers = ["#", "SMILES", "分子量", "重量分率", "モル分率"]
                    rows = []
                    for i in range(len(components)):
                        rows.append({
                            "#": i + 1,
                            "SMILES": components[i]["smiles"],
                            "分子量": f"{info['molecular_weights'][i]:.2f}",
                            "重量分率": f"{info['weight_fractions'][i]*100:.1f}%",
                            "モル分率": f"{info['mole_fractions'][i]*100:.1f}%",
                        })

                    with ui.table(
                        columns=[{"name": h, "label": h, "field": h} for h in headers],
                        rows=rows,
                    ).classes("w-full").props("dense flat"):
                        pass

                    ui.label(
                        f"🧪 混合物特徴量: {len(result.mixture_features)}列"
                    ).classes("q-mt-sm text-sm").style("color: #a0a0c0;")

                # 警告があれば表示
                for w in result.warnings:
                    ui.label(f"⚠️ {w}").classes("text-sm").style("color: #fbbf24;")

                # データセット統合の確認表示
                ui.label(
                    f"✅ データセットに追加済み（計{len(state['df'])}行）"
                ).classes("q-mt-sm text-sm text-cyan")

                # ── 加重方法カスタマイズUI（Phase 3）──
                if result.weighting_log:
                    with ui.expansion(
                        "⚙️ 特徴量ごとの加重方法",
                        icon="tune",
                    ).classes("w-full q-mt-md").style(
                        "background: rgba(255,255,255,0.02); border-radius: 12px;"
                    ):
                        ui.label("自動分類結果（選択で上書き）").classes("text-sm").style(
                            "color: #a0a0c0;"
                        )

                        # 状態保存
                        if "mixture_weighting_overrides" not in state:
                            state["mixture_weighting_overrides"] = {}

                        weighting_container = ui.column().classes("w-full gap-1")

                        with weighting_container:
                            for feat_name, wtype_str in result.weighting_log.items():
                                base_type = wtype_str.split("(")[0].strip()

                                with ui.row().classes("items-center w-full gap-2"):
                                    ui.label(feat_name).classes("w-48 text-sm font-mono")
                                    ui.label(f"→ {wtype_str}").classes("text-xs text-grey w-32")

                                    select = ui.select(
                                        options={"weight": "⚖️ 重量比", "mole": "🔢 mol比", "context": "📝 文脈依存"},
                                        value=base_type,
                                        label="",
                                    ).classes("w-32")

                                    def _make_handler(feat=feat_name, sel=select):
                                        def _on_change(e):
                                            state["mixture_weighting_overrides"][feat] = e.value
                                            ui.notify(f"🔄 {feat}: {e.value}に変更", type="info")
                                        return _on_change
                                    select.on_value_change(_make_handler())

                    # 再計算ボタン
                    def _recalc():
                        ui.notify("🔄 再計算中...", type="info")
                        # オーバーライドを反映して再計算
                        _run_mixture_calc()

                    ui.button(
                        "🔄 再計算",
                        on_click=lambda: _run_mixture_calc(),
                        icon="refresh",
                    ).props("outline size=sm color=amber")

                # ── CSVエクスポート（Phase 5）──
                def _export_results():
                    try:
                        import pandas as pd
                        export_df = result.to_dataframe()
                        info = result.conversion_info
                        for i, (wt, mole) in enumerate(zip(info['weight_fractions'], info['mole_fractions'])):
                            export_df[f"Component_{i+1}_SMILES"] = components[i]["smiles"] if i < len(components) else ""
                            export_df[f"Component_{i+1}_weight_frac"] = wt
                            export_df[f"Component_{i+1}_mole_frac"] = mole
                        if target_value is not None:
                            export_df["Target_Property"] = target_value
                        export_df["Sample_ID"] = sample_id

                        buf = io.StringIO()
                        export_df.to_csv(buf, index=False, encoding='utf-8-sig')
                        csv_bytes = buf.getvalue().encode('utf-8-sig')
                        ui.download(csv_bytes, f"mixture_{sample_id}.csv")
                        ui.notify("📥 CSVをダウンロードしました", type="positive")
                    except Exception as e:
                        ui.notify(f"❌ エクスポート失敗: {e}", type="negative")

                ui.button(
                    "📤 計算結果をCSV出力",
                    on_click=_export_results,
                    icon="download",
                ).props("outline color=teal size=sm")

            # UI更新
            _refresh_ui_after_calc()

            ui.notify(
                f"✅ {len(result.mixture_features)}列の混合物特徴量を計算・統合しました",
                type="positive",
            )

        except Exception as e:
            ui.notify(f"❌ 計算エラー: {e}", type="negative")
            logger.error("混合物特徴量計算エラー: %s", e, exc_info=True)

    ui.button(
        "🚀 混合物特徴量を計算",
        on_click=_run_mixture_calc,
        icon="play_arrow",
    ).props("color=primary size=lg").classes("w-full").style(
        "background: linear-gradient(135deg, #7b2ff7, #00d4ff) !important; "
        "border-radius: 12px; font-size: 16px; font-weight: 600;"
    )
