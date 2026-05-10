from nicegui import ui

@ui.refreshable
def render_workflow_navigator(state: dict, main_tabs):
    """現在の解析フェーズに応じたガイダンス・操作権限を制御"""
    
    # We deduce step from state variables
    df_raw = state.get("df")
    target_col = state.get("target_col")
    
    # Deduce if pipeline/results exist
    analyzed = state.get("automl_result") is not None
    is_running = state.get("_analysis_running", False)

    # Determine step
    if df_raw is None:
        step = "data_upload"
    elif target_col is None:
        step = "target_missing"
    elif is_running:
        step = "analyzing"
    elif analyzed:
        step = "results"
    else:
        step = "settings"

    banner_map = {
        "data_upload": "📂 Step 1: データタブからCSVを読み込んでください",
        "target_missing": "🎯 Step 2: 目的変数（予測対象）を選択してください",
        "settings": "⚙️ Step 3: 設定を確認後、「設定」タブ最下部の【▶️ 解析実行】をクリックしてください",
        "analyzing": "⏳ 解析実行中... 計算完了までお待ちください",
        "results": "✅ Step 4: 解析完了！「結果確認」タブで多角的検証が可能です"
    }

    with ui.row().classes("w-full items-center p-3 text-slate-100 rounded mb-4 shadow").style("background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(148, 163, 184, 0.2);"):
        ui.icon("psychology", size="md", color="cyan-400")
        ui.label(banner_map.get(step, "")).classes("text-sm font-medium ml-2")
        ui.space()

        if step == "results":
            ui.button("🔄 設定変更", on_click=lambda: main_tabs.set_value("pipeline")).props("flat dense color=white no-caps")
        elif step == "analyzing":
            ui.spinner(color="cyan", size="sm")
            ui.label("計算中").classes("text-xs ml-2 text-cyan-200")
            
    # Also we want to limit tab access optionally, but since users might want to peek around,
    # we can just gray out tabs or rely on the banner for guidance.
