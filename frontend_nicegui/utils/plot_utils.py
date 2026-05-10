import uuid
import json
from nicegui import ui, app
import plotly.graph_objects as go

def render_plot_with_expand(fig: go.Figure, title: str = "", height: str = "400px"):
    """プロット＋別ウィンドウボタンを安全に生成"""
    plot_id = f"plot_{uuid.uuid4().hex[:8]}"
    
    with ui.card().classes("w-full p-2 shadow-sm").style("border: 1px solid rgba(0,188,212,0.3); border-radius: 8px; background: rgba(0,20,40,0.25);"):
        with ui.row().classes("w-full items-center justify-between q-mb-xs"):
            ui.label(title).classes("text-subtitle2 text-bold q-ml-sm")
            ui.button(
                icon="open_in_new", text="別ウィンドウ",
                on_click=lambda: _store_and_open(plot_id, fig),
                color="grey"
            ).props("dense flat size=sm no-caps color=cyan")
        
        ui.plotly(fig).classes("w-full").style(f"height: {height};")

def _store_and_open(plot_id: str, fig: go.Figure):
    """セッションストレージへ保存し新タブ遷移（TTL管理不要な設計）"""
    app.storage.user[f"ext_plot_{plot_id}"] = fig.to_plotly_json()
    ui.open(f"/ext_plot/{plot_id}", new_tab=True)

@ui.page('/ext_plot/{plot_id}')
def external_plot_viewer(plot_id: str):
    """外部ビュー専用ページ（取得と同時に削除してメモリリークを防止）"""
    fig_json = app.storage.user.pop(f"ext_plot_{plot_id}", None)
    if fig_json is None:
        ui.label("❌ 有効期限切れまたは既に開かれています").classes("text-red-500 text-xl p-4")
        ui.button("戻る", on_click=lambda: ui.run_javascript("window.close()"))
        return

    fig = go.Figure(fig_json)
    fig.update_layout(height=900, margin=dict(l=40, r=40, t=40, b=40))
    ui.plotly(fig).classes("w-full").style("height: 90vh;")
    ui.button("✕ 閉じる", on_click=lambda: ui.run_javascript("window.close()"))\
        .classes("absolute-bottom-right q-ma-md").props("color=cyan")
