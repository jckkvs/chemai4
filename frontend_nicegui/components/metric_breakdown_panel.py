"""
階層別メトリック表示パネル（事前計算済みデータを使用）

特徴:
- 計算待ちゼロ: データは既に全レベル存在するため描画のみ
- タブ切替・列選択が即時応答
- R² 値に応じた条件付き書式で視覚的洞察を支援
"""

from nicegui import ui
import pandas as pd
from typing import Dict, Any

def _r2_color_class(r2: float) -> str:
    """R² 値に応じた色クラスを返す"""
    if pd.isna(r2):
        return "text-grey-500"
    elif r2 > 0.7:
        return "text-positive font-bold"
    elif r2 > 0.3:
        return "text-warning"
    else:
        return "text-negative"


def _r2_icon(r2: float) -> str:
    """R² 値に応じた判定アイコンを返す"""
    if pd.isna(r2):
        return "⚪"
    elif r2 > 0.7:
        return "🟢"
    elif r2 > 0.3:
        return "🟡"
    else:
        return "🔴"


def _format_metric(value: float, is_r2: bool = False) -> str:
    """メトリック値をフォーマット"""
    if pd.isna(value):
        return "N/A"
    return f"{value:.4f}" if is_r2 else f"{value:.3f}"


@ui.refreshable
def render_metric_breakdown_panel(metrics_dict: Dict[str, Any]):
    """
    事前計算済みメトリックを即時表示。
    
    Args:
        metrics_dict: StratifiedMetrics.to_dict() の出力辞書
    """
    if not metrics_dict:
        ui.label("⚠️ 評価指標が利用できません").classes("text-warning bg-warning/10 p-2 rounded")
        return
    
    # ── タブ切替（計算不要：データは既に全レベル存在） ──
    with ui.tabs().classes("w-full") as level_tabs:
        ui.tab("🌐 全体")
        if metrics_dict.get("by_category"):
            ui.tab("📁 カテゴリ別")
        if metrics_dict.get("by_cluster"):
            ui.tab("🔷 クラスタ別")
    
    with ui.tab_panels(level_tabs, value="🌐 全体").classes("w-full"):
        
        # L1: Global
        with ui.tab_panel("🌐 全体"):
            _render_metric_card(metrics_dict["global_metrics"], title="全体性能")
        
        # L2: Category
        if metrics_dict.get("by_category"):
            with ui.tab_panel("📁 カテゴリ別"):
                _render_category_selector(metrics_dict)
        
        # L3: Cluster
        if metrics_dict.get("by_cluster"):
            with ui.tab_panel("🔷 クラスタ別"):
                _render_cluster_table(metrics_dict["by_cluster"], metrics_dict.get("min_group_size", 10))


def _render_metric_card(record: Dict[str, Any], title: str):
    """メトリックをカード形式で表示"""
    with ui.card().classes("w-full p-4 shadow-sm"):
        ui.label(title).classes("text-subtitle2 font-bold mb-3")
        
        with ui.row().classes("w-full gap-4"):
            # R²
            with ui.column().classes("flex-1 items-center"):
                ui.label("R²").classes("text-caption text-grey-6").tooltip("モデルの当てはまりの良さ (1.0が最高)")
                r2_val = record.get("r2", float('nan'))
                icon = _r2_icon(r2_val)
                ui.label(f"{icon} {_format_metric(r2_val, is_r2=True)}").classes(
                    f"text-h5 {_r2_color_class(r2_val)}"
                )
            
            # MAE
            with ui.column().classes("flex-1 items-center"):
                ui.label("MAE").classes("text-caption text-grey-6").tooltip("平均絶対誤差 (予測と実測のズレの平均)")
                ui.label(_format_metric(record.get("mae"))).classes("text-h5 font-bold")
            
            # RMSE
            with ui.column().classes("flex-1 items-center"):
                ui.label("RMSE").classes("text-caption text-grey-6").tooltip("二乗平均平方根誤差 (大きな誤差を重く評価)")
                ui.label(_format_metric(record.get("rmse"))).classes("text-h5 font-bold")
            
            # N
            with ui.column().classes("flex-1 items-center"):
                ui.label("N").classes("text-caption text-grey-6").tooltip("評価対象サンプル数")
                ui.label(str(record.get("n", 0))).classes("text-h5 font-bold")


def _render_category_selector(metrics_dict: Dict[str, Any]):
    """カテゴリ列選択＋テーブル表示"""
    categories = metrics_dict.get("available_categories", [])
    by_category = metrics_dict.get("by_category", {})
    
    if not categories:
        ui.label("ℹ️ 利用可能なカテゴリ列がありません").classes("text-grey-6")
        return
    
    # 選択状態を管理
    selected_col = {"value": categories[0]}  # mutable container for closure
    
    def on_col_change(e):
        selected_col["value"] = e.value
        _render_category_table.refresh(by_category, e.value, metrics_dict.get("min_group_size", 10))
    
    with ui.row().classes("w-full items-end mb-2"):
        ui.select(
            options=categories,
            value=categories[0],
            label="カテゴリ列",
            on_change=on_col_change
        ).props("dense")
    
    # 初期表示
    _render_category_table(by_category, categories[0], metrics_dict.get("min_group_size", 10))


@ui.refreshable
def _render_category_table(
    by_category: Dict[str, Dict[str, Dict[str, Any]]], 
    column: str,
    min_group_size: int
):
    """カテゴリ別メトリックをテーブル表示"""
    groups = by_category.get(column, {})
    if not groups:
        ui.label("ℹ️ 該当するグループが見つかりません").classes("text-grey-6")
        return
    
    # DataFrame 化してソート（絶対値で）
    rows = []
    for name, rec in groups.items():
        r2 = rec.get("r2", float('nan'))
        rows.append({
            "group": name,
            "r2": r2,
            "r2_display": f"{_r2_icon(r2)} {_format_metric(r2, is_r2=True)}",
            "r2_class": _r2_color_class(r2),
            "mae": rec.get("mae", float('nan')),
            "mae_display": _format_metric(rec.get("mae")),
            "rmse": rec.get("rmse", float('nan')),
            "rmse_display": _format_metric(rec.get("rmse")),
            "n": rec.get("n", 0),
            "warning": "⚠️" if rec.get("n", 0) < min_group_size else ""
        })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("r2", key=lambda x: x.abs(), ascending=False, na_position='last')
    
    # NiceGUI table 用に変換
    table_rows = df.to_dict(orient="records")
    
    columns = [
        {"name": "group", "label": "Group", "field": "group", "sortable": True},
        {"name": "r2_display", "label": "R²", "field": "r2_display", "sortable": True, "classes": "font-mono"},
        {"name": "mae_display", "label": "MAE", "field": "mae_display", "sortable": True},
        {"name": "rmse_display", "label": "RMSE", "field": "rmse_display", "sortable": True},
        {"name": "n", "label": "N", "field": "n", "sortable": True},
        {"name": "warning", "label": "", "field": "warning"},
    ]
    
    ui.table(columns=columns, rows=table_rows, row_key="group").props("dense flat bordered")
    
    # 補足情報
    if any(r["warning"] for r in table_rows):
        ui.label(f"⚠️ N<{min_group_size} のグループは統計的信頼性が低いため参考値として扱ってください").classes(
            "text-caption text-orange-6 mt-2"
        )


def _render_cluster_table(
    by_cluster: Dict[str, Dict[str, Any]], 
    min_group_size: int
):
    """クラスタ別メトリックをテーブル表示"""
    rows = []
    for name, rec in by_cluster.items():
        r2 = rec.get("r2", float('nan'))
        rows.append({
            "group": name,
            "r2": r2,
            "r2_display": f"{_r2_icon(r2)} {_format_metric(r2, is_r2=True)}",
            "r2_class": _r2_color_class(r2),
            "n": rec.get("n", 0),
        })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("r2", key=lambda x: x.abs(), ascending=False, na_position='last')
    
    table_rows = df.to_dict(orient="records")
    
    columns = [
        {"name": "group", "label": "Cluster", "field": "group"},
        {"name": "r2_display", "label": "R²", "field": "r2_display", "classes": "font-mono"},
        {"name": "n", "label": "N", "field": "n"},
    ]
    
    ui.table(columns=columns, rows=table_rows, row_key="group").props("dense flat")
    
    # 補足
    ui.label("💡 クラスタは化学的類似度に基づき自動生成されています").classes(
        "text-caption text-grey-6 mt-2"
    )
