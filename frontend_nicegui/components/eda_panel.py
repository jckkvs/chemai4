# -*- coding: utf-8 -*-
"""
frontend_nicegui/components/eda_panel.py

探索的データ分析（EDA）統合パネル — NiceGUI版

機械学習の教科書レベルの包括的EDA:
  1.  統計サマリー（歪度・尖度含む）
  2.  分布（ヒストグラム + KDE + ラグ）
  3.  相関行列（Pearson / Spearman / Kendall）
  4.  目的変数 vs 説明変数（散布図 + 回帰線）
  5.  Pairplot（散布図マトリックス）
  6.  ボックス/バイオリンプロット
  7.  QQプロット（正規性検定）
  8.  VIF（多重共線性）
  9.  外れ値検出（IQR / Z-score）
  10. 欠損解析
  11. 次元削減（PCA / t-SNE + 寄与率・ローディング）
  12. 特徴量重要度（相互情報量）
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui
# from frontend_nicegui.components.eda_dim_reduction import render_dim_reduction_panel

logger = logging.getLogger(__name__)

# Plotlyのダークテーマ共通レイアウト
_LAYOUT_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e0e0f0", size=11),
)


def _dark_fig(fig, height: int = 400, **kwargs):
    """Plotly figureにダークテーマを適用。"""
    layout = {**_LAYOUT_DARK, "height": height, "margin": dict(l=50, r=20, t=40, b=40)}
    layout.update(kwargs)
    fig.update_layout(**layout)
    return fig


def _plotly_and_save(fig, name: str, session_id: str | None = None, **plotly_kwargs):
    """
    ui.plotly() でフロントエンドに表示しつつ、バックグラウンドで保存も実行するヘルパー。

    Args:
        fig:          plotly.graph_objects.Figure
        name:         保存ファイル名のベース（例: "eda_pairplot"）
        session_id:   保存先サブディレクトリ名
        **plotly_kwargs: ui.plotly() に渡す追加引数
    """
    try:
        from backend.utils.plot_exporter import save_plot_versions
        save_plot_versions(fig, name=name, session_id=session_id, run_async=True)
    except Exception as _e:
        logger.debug(f"[EDA PlotSave] {name} 保存失敗: {_e}")
    return ui.plotly(fig, **plotly_kwargs)


# ═══════════════════════════════════════════════════════════
# メインエントリーポイント（統合版）
# ═══════════════════════════════════════════════════════════
def render_eda_panel(state: dict) -> None:
    """EDAパネルをレンダリング。

    descriptor_sets がある場合はセットごとにタブを動的生成。
    表データのみの場合は元データで直接EDA表示。
    data_tab内のEDAからも呼び出される（統合版）。
    """
    df = state.get("df")
    if df is None:
        with ui.card().classes("full-width q-pa-lg").style(
            "border: 1px dashed rgba(255,255,255,0.2); border-radius: 10px;"
        ):
            ui.icon("upload_file", color="grey").classes("text-h4")
            ui.label("データを読み込むとEDAが表示されます").classes("text-body2 text-grey")
        return

    target_col = state.get("target_col", "")
    precalc_df = state.get("precalc_df")
    descriptor_sets_raw = state.get("descriptor_sets", {})

    # EDA対象データセットを収集
    eda_datasets: list[tuple[str, pd.DataFrame]] = []

    if descriptor_sets_raw and precalc_df is not None:
        # descriptor_sets は {name: {"active": bool, "descriptors": [...], "cols": [...]}} の形式
        # または {name: [列名リスト]} の旧形式の両方に対応する
        for set_name, set_info in descriptor_sets_raw.items():
            if isinstance(set_info, dict):
                # 新形式: cols（計算済み列名）または descriptors（記述子ID）から列を取得
                cols = set_info.get("cols") or set_info.get("descriptors") or []
            else:
                # 旧形式: 直接リスト
                cols = list(set_info) if set_info else []

            # precalc_df に実在する列のみ使う
            valid_cols = [c for c in cols if c in precalc_df.columns]

            # cols が空 or 有効列なしの場合は set_name をプレフィックスとして列をマッチ
            if not valid_cols:
                prefix = set_name.lower().replace(" ", "_")
                valid_cols = [
                    c for c in precalc_df.columns
                    if c.lower().startswith(prefix)
                ]

            if valid_cols:
                set_df = precalc_df[valid_cols].copy()
                if target_col and target_col in df.columns:
                    set_df[target_col] = df[target_col].iloc[:len(set_df)].values
                eda_datasets.append((f"📦 {set_name}", set_df))

        # 全記述子タブ（常に追加）
        if not precalc_df.empty:
            all_df = precalc_df.copy()
            if target_col and target_col in df.columns:
                all_df[target_col] = df[target_col].iloc[:len(all_df)].values
            eda_datasets.append(("📊 全記述子", all_df))

    elif precalc_df is not None and not precalc_df.empty:
        # セット未定義だが記述子あり
        all_df = precalc_df.copy()
        if target_col and target_col in df.columns:
            all_df[target_col] = df[target_col].iloc[:len(all_df)].values
        eda_datasets.append(("📊 全記述子", all_df))

    # 元データ（説明変数）を常に追加
    num_df = df.select_dtypes(include="number").copy()
    if not num_df.empty:
        # 目的変数が数値列以外の場合もEDAで扱えるよう追加
        if target_col and target_col in df.columns and target_col not in num_df.columns:
            num_df[target_col] = df[target_col]
        eda_datasets.append(("📋 元データ", num_df))

    if not eda_datasets:
        ui.label("数値データがありません").classes("text-caption text-grey")
        return

    # ── タブ生成 ──
    with ui.card().classes("full-width q-pa-md").style(
        "border:1px solid rgba(0,188,212,0.3);border-radius:12px;"
        "background:rgba(0,20,40,0.25);"
    ):
        with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
            ui.icon("query_stats", color="cyan").classes("text-h5")
            ui.label("探索的データ分析（EDA）").classes("text-h6")
            ui.badge(f"{len(eda_datasets)}データセット", color="teal").props("dense")

        if len(eda_datasets) == 1:
            _render_full_eda(eda_datasets[0][1], target_col, state, uid="single")
        else:
            tab_keys = []
            with ui.tabs().classes("full-width").props(
                "dense no-caps active-color=cyan indicator-color=cyan scrollable"
            ) as ds_tabs:
                for i, (label, _) in enumerate(eda_datasets):
                    key = f"eda_ds_{i}"
                    tab_keys.append(key)
                    ui.tab(key, label=label)

            with ui.tab_panels(ds_tabs, value=tab_keys[0]).classes("full-width bg-transparent"):
                for i, (label, ds_df) in enumerate(eda_datasets):
                    with ui.tab_panel(tab_keys[i]):
                        _render_full_eda(ds_df, target_col, state, uid=f"ds{i}")


def _render_full_eda(df: pd.DataFrame, target_col: str, state: dict, uid: str = "0") -> None:
    """単一データセットの包括的EDA — より少ないタブとカードベースのUI

    Args:
        uid: ユニークID。ネストされたタブ名の衝突を防ぐため、
             同一ページ内で複数回呼ばれる場合はそれぞれ異なる値を渡す。
    """
    # uid をタブ名に付加し、複数データセットでの名前衝突を回避
    t_data_key = f"t_data_{uid}"
    t_adv_key = f"t_advanced_{uid}"

    with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
        ui.badge(f"{df.shape[0]}行 × {df.shape[1]}列", color="grey-7").props("dense")

    with ui.tabs().classes("full-width").props(
        "dense no-caps active-color=cyan indicator-color=cyan scrollable"
    ) as eda_tabs:
        ui.tab(t_data_key, label="📊 データ品質・分布・関係性")
        ui.tab(t_adv_key, label="🌀 次元の削減と重要度")

    with ui.tab_panels(eda_tabs, value=t_data_key).classes("full-width bg-transparent"):
        with ui.tab_panel(t_data_key):
            # ① Pairplot（最初に表示）
            ui.label("🔵 Pairplot（散布図マトリックス）").classes("text-subtitle2 text-bold")
            _render_pairplot(df, target_col)

            ui.separator().classes("q-my-md")
            # ② 相関行列
            ui.label("🔥 相関行列 (VIFフィルタ済)").classes("text-subtitle2 text-bold text-amber")
            _render_correlation(df, target_col)

            ui.separator().classes("q-my-md")
            # ③ 統計サマリー
            ui.label("📋 統計サマリー").classes("text-subtitle2 text-bold")
            _render_stats(df, target_col)

        with ui.tab_panel(t_adv_key):
            with ui.row().classes("full-width q-mb-md q-col-gutter-lg"):
                with ui.column().classes("col-12 col-md-6"):
                    ui.label("🎯 目的変数 vs 特徴量 (相関上位)").classes("text-subtitle2 text-bold")
                    _render_target_vs_features(df, target_col)
                with ui.column().classes("col-12 col-md-6"):
                    ui.label("📊 特徴量の分布 (相関上位)").classes("text-subtitle2 text-bold")
                    _render_distribution(df, target_col)

            ui.separator().classes("q-my-md")
            with ui.row().classes("full-width q-mb-md q-col-gutter-lg"):
                 with ui.column().classes("col-12"):
                     ui.label("🌀 次元の削減と特徴量重要度").classes("text-subtitle2 text-bold")
                     from frontend_nicegui.components.eda_dim_panel import dim_reduction_settings, dim_reduction_panel
                     
                     if "_dim_config" not in state:
                         state["_dim_config"] = {"scale": True}
                     
                     def on_apply_settings(scale: bool):
                         state["_dim_config"]["scale"] = scale
                         dim_reduction_panel.refresh()
                     
                     dim_reduction_settings(
                         on_apply=on_apply_settings,
                         default_scale=state["_dim_config"]["scale"]
                     )
                     
                     dim_reduction_panel(
                         df=df,
                         target_col=target_col,
                         scale=state["_dim_config"]["scale"]
                     )

            ui.separator().classes("q-my-md")
            with ui.row().classes("full-width q-col-gutter-lg"):
                 with ui.column().classes("col-12 col-md-6"):
                     ui.label("🎯 外れ値").classes("text-subtitle2 text-bold")
                     _render_outliers(df, target_col)
                 with ui.column().classes("col-12 col-md-5"):
                     ui.label("🧩 欠損").classes("text-subtitle2 text-bold")
                     _render_missing(df)

            # ── ⑤ ボックスプロット & QQプロット ──
            ui.separator().classes("q-my-md")
            with ui.row().classes("full-width q-col-gutter-lg"):
                with ui.column().classes("col-12 col-md-6"):
                    ui.label("📦 ボックスプロット（特徴量分布）").classes("text-subtitle2 text-bold")
                    _render_box_plots(df, target_col)
                with ui.column().classes("col-12 col-md-6"):
                    ui.label("📈 QQプロット（正規性）").classes("text-subtitle2 text-bold")
                    _render_qq_plots(df, target_col)
# ═══════════════════════════════════════════════════════════
# 1. 統計サマリー（歪度・尖度含む）
# ═══════════════════════════════════════════════════════════
def _render_stats(df: pd.DataFrame, target_col: str) -> None:
    num_df = df.select_dtypes(include="number")
    if num_df.empty:
        ui.label("数値列がありません").classes("text-caption text-grey")
        return

    stats = num_df.describe().T
    stats["欠損率(%)"] = (num_df.isna().mean() * 100).round(1)
    stats["歪度"] = num_df.skew().round(3)
    stats["尖度"] = num_df.kurtosis().round(3)
    stats = stats.reset_index().rename(columns={"index": "列名"})

    display_cols = ["列名", "count", "mean", "std", "min", "25%", "50%", "75%", "max", "欠損率(%)", "歪度", "尖度"]
    existing = [c for c in display_cols if c in stats.columns]
    for c in existing:
        if c not in ("列名", "count"):
            stats[c] = stats[c].apply(lambda x: round(x, 4) if isinstance(x, float) else x)

    rows_data = stats[existing].to_dict("records")
    columns = [{"name": c, "label": c, "field": c, "sortable": True} for c in existing]
    ui.table(columns=columns, rows=rows_data, pagination={"rowsPerPage": 30}).classes(
        "full-width"
    ).props("dense flat separator=cell").style("font-size:0.8rem;")


# ═══════════════════════════════════════════════════════════
# 2. 分布（ヒストグラム + KDE）
# ═══════════════════════════════════════════════════════════
def _render_distribution(df: pd.DataFrame, target_col: str) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        ui.label("数値列がありません").classes("text-caption text-grey")
        return

    # 目的変数との相関が高い上位6つの特徴量を選択（または全体から適当に）
    if target_col in num_cols:
        num_cols.remove(target_col)
        
    try:
        corr = df[num_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
        top_cols = corr.head(6).index.tolist()
    except Exception:
        top_cols = num_cols[:6]

    if not top_cols: return
    
    rows, cols = (len(top_cols)+1)//2, 2 if len(top_cols)>=2 else 1
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=top_cols)
    
    for i, c in enumerate(top_cols):
        r = i//cols + 1
        c_idx = i%cols + 1
        fig.add_trace(
            go.Histogram(x=df[c].dropna(), name=c, showlegend=False, marker_color="#00d4ff", nbinsx=30),
            row=r, col=c_idx
        )
        
    _dark_fig(fig, height=max(200, 180*rows))
    fig.update_layout(margin=dict(l=30, r=20, t=30, b=20))
    _plotly_and_save(fig, "eda_distribution").classes("full-width")


# ═══════════════════════════════════════════════════════════
# 3. 相関行列（3種）
# ═══════════════════════════════════════════════════════════
def _render_correlation(df: pd.DataFrame, target_col: str) -> None:
    import plotly.express as px

    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        ui.label("数値列が2列未満です").classes("text-caption text-grey")
        return

    method_sel = ui.toggle(
        {"pearson": "Pearson", "spearman": "Spearman", "kendall": "Kendall"},
        value="pearson",
    ).props("no-caps dense color=cyan").classes("q-mb-sm")

    chart_container = ui.column().classes("full-width")

    def _draw():
        chart_container.clear()
        method = method_sel.value
        work_df = num_df
        max_cols = 30
        if work_df.shape[1] > max_cols and target_col in work_df.columns:
            corr_abs = work_df.corr(method=method)[target_col].abs().sort_values(ascending=False)
            candidates = corr_abs.index.tolist()
            
            selected = []
            # VIF代用の相関フィルタリング
            # 互いの相関が非常に強い特徴量(>0.85)は排除し、独立した情報を優先
            corr_matrix = work_df.corr(method=method).abs()
            for c in candidates:
                if c == target_col: continue
                if not selected:
                    selected.append(c)
                else:
                    max_corr = corr_matrix.loc[c, selected].max()
                    if max_corr < 0.85: # 共線性が低い場合のみ追加
                        selected.append(c)
                if len(selected) >= max_cols:
                    break
                    
            top = selected + [target_col]
            work_df = work_df[top]

        corr = work_df.corr(method=method)
        with chart_container:
            if work_df.shape[1] < num_df.shape[1]:
                ui.label(f"上位 {max_cols} 列のみ表示").classes("text-caption text-amber q-mb-xs")
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, title=f"相関行列（{method}）")
            _dark_fig(fig, max(400, min(800, work_df.shape[1] * 25)), margin=dict(l=60, r=20, t=40, b=60))
            _plotly_and_save(fig, f"eda_correlation_{method}").classes("full-width")

            # 目的変数との相関TOP表示
            if target_col in corr.columns:
                target_corr = corr[target_col].drop(target_col, errors="ignore").abs().sort_values(ascending=False)
                top10 = target_corr.head(10)
                ui.label(f"🎯 {target_col} との相関 TOP10").classes("text-subtitle2 q-mt-md")
                import plotly.graph_objects as go
                fig2 = go.Figure(go.Bar(
                    x=top10.values, y=top10.index, orientation="h",
                    marker_color=["#00d4ff" if v > 0.3 else "#fbbf24" if v > 0.1 else "#555" for v in top10.values]
                ))
                fig2.update_layout(xaxis_title=f"|r| ({method})", yaxis=dict(autorange="reversed"))
                _dark_fig(fig2, max(250, len(top10) * 25))
                _plotly_and_save(fig2, f"eda_target_corr_top10_{method}").classes("full-width")

    method_sel.on_value_change(lambda: _draw())
    _draw()


# ═══════════════════════════════════════════════════════════
# 4. 目的変数 vs 説明変数（散布図 + 回帰線）
# ═══════════════════════════════════════════════════════════
def _render_target_vs_features(df: pd.DataFrame, target_col: str) -> None:
    """目的変数 vs 説明変数の散布図（相関上位）— 修正版"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    num_cols = [c for c in df.select_dtypes(include="number").columns]
    
    # 目的変数のチェックを緩和
    if not target_col or target_col not in df.columns:
        ui.label("⚠️ 目的変数が設定されていないか、数値列ではありません").classes("text-caption text-grey")
        return
    
    # 説明変数（目的変数を除く）
    feature_cols = [c for c in num_cols if c != target_col]
    
    if not feature_cols:
        ui.label("⚠️ 説明変数（特徴量）がありません").classes("text-caption text-grey")
        return
    
    try:
        # 相関計算（絶対値）
        corr = df[feature_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
        top_cols = corr.head(6).index.tolist()
    except Exception as e:
        logger.warning(f"相関計算エラー: {e}")
        # フォールバック: 先頭6列
        top_cols = feature_cols[:6]
    
    if not top_cols:
        ui.label("⚠️ 表示する特徴量がありません").classes("text-caption text-grey")
        return
    
    # サブプロットの行数・列数を計算
    rows, cols = (len(top_cols) + 1) // 2, 2 if len(top_cols) >= 2 else 1
    
    # 各サブプロットを正方形にするためにwidth/heightを均等に計算
    _CELL = 320  # 1セルあたりのピクセル（正方形）
    _GAP = 30    # セル間マージン
    total_w = cols * _CELL + (cols - 1) * _GAP + 60   # 左右余白込み
    total_h = rows * _CELL + (rows - 1) * _GAP + 60   # 上下余白込み
    
    # 相関係数をタイトルに含む
    titles = []
    for c in top_cols:
        try:
            r_val = df[c].corr(df[target_col])
            titles.append(f"{c} (r={r_val:.2f})")
        except:
            titles.append(c)
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=titles,
        horizontal_spacing=_GAP/total_w,
        vertical_spacing=_GAP/total_h,
    )
    
    # 各特徴量に対して散布図を作成
    for i, c in enumerate(top_cols):
        r = i // cols + 1
        c_idx = i % cols + 1
        
        plot_df = df[[c, target_col]].dropna()
        
        if not plot_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=plot_df[c], y=plot_df[target_col],
                    mode="markers", name=c, showlegend=False,
                    marker=dict(color="#00d4ff", size=4, opacity=0.6),
                ),
                row=r, col=c_idx,
            )
    
    # レイアウト設定
    fig.update_layout(
        **_LAYOUT_DARK,
        width=total_w,
        height=total_h,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    
    # 全軸を正方形にロック
    for i in range(1, rows * cols + 1):
        xkey = "xaxis" if i == 1 else f"xaxis{i}"
        ykey = "yaxis" if i == 1 else f"yaxis{i}"
        # Plotly scaleanchor は 'y', 'y2' などのIDを指定する
        anchor_id = "y" if i == 1 else f"y{i}"
        fig.update_layout(**{
            xkey: dict(scaleanchor=anchor_id, scaleratio=1,
                      gridcolor="rgba(255,255,255,0.08)"),
            ykey: dict(gridcolor="rgba(255,255,255,0.08)"),
        })
    _plotly_and_save(fig, "eda_target_vs_features").classes("full-width")


# ═══════════════════════════════════════════════════════════
# 5. Pairplot
# ═══════════════════════════════════════════════════════════
def _render_pairplot(df: pd.DataFrame, target_col: str) -> None:
    import plotly.express as px

    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        ui.label("数値列が2列未満です").classes("text-caption text-grey")
        return

    cols_list = num_df.columns.tolist()
    if target_col in cols_list:
        cols_list.remove(target_col)
        defaults = [target_col] + cols_list[:4]
    else:
        defaults = cols_list[:5]

    selected = ui.select(num_df.columns.tolist(), value=defaults, label="特徴量（5-6個推奨）",
                         multiple=True).props("outlined dense use-chips").classes("full-width q-mb-sm")
    chart_container = ui.column().classes("full-width")

    def _draw():
        chart_container.clear()
        cols = selected.value
        if not cols or len(cols) < 2:
            with chart_container:
                ui.label("2つ以上選択してください").classes("text-caption text-amber")
            return
        plot_df = num_df[cols].dropna()
        if len(plot_df) > 1000:
            plot_df = plot_df.sample(1000, random_state=42)
        with chart_container:
            color_col = target_col if target_col in cols and df[target_col].nunique() < 20 else None
            fig = px.scatter_matrix(plot_df, dimensions=cols,
                                    color=color_col if color_col in plot_df.columns else None,
                                    template="plotly_dark", color_continuous_scale="Viridis")
            fig.update_traces(diagonal_visible=False)
            _dark_fig(fig, max(450, len(cols) * 120))
            _plotly_and_save(fig, "eda_pairplot").classes("full-width")

    ui.button("描画更新", on_click=_draw).props("outline color=cyan size=sm no-caps")
    _draw()


# ═══════════════════════════════════════════════════════════
# 6. Box / Violin プロット
# ═══════════════════════════════════════════════════════════
def _render_box_violin(df: pd.DataFrame, target_col: str) -> None:
    import plotly.express as px

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        ui.label("数値列がありません").classes("text-caption text-grey")
        return

    with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
        col_select = ui.select(num_cols, value=num_cols[0], label="列を選択").props("outlined dense")
        plot_type = ui.toggle({"box": "📦 Box", "violin": "🎻 Violin"}, value="box").props("dense no-caps")

    chart_container = ui.column().classes("full-width")

    def _draw():
        chart_container.clear()
        col = col_select.value
        if not col:
            return
        with chart_container:
            if plot_type.value == "box":
                fig = px.box(df, y=col, points="outliers", title=f"{col} ボックスプロット",
                             template="plotly_dark")
            else:
                fig = px.violin(df, y=col, box=True, points="all", title=f"{col} バイオリンプロット",
                                template="plotly_dark")
            _dark_fig(fig, 400)
            _plotly_and_save(fig, f"eda_box_violin_{col_select.value}").classes("full-width")

    col_select.on("update:model-value", lambda: _draw())
    plot_type.on_value_change(lambda: _draw())
    _draw()


# ═══════════════════════════════════════════════════════════
# 7. QQプロット（正規性検定）
# ═══════════════════════════════════════════════════════════
def _render_qqplot(df: pd.DataFrame, target_col: str) -> None:
    import plotly.graph_objects as go
    from scipy import stats as sp_stats

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        ui.label("数値列がありません").classes("text-caption text-grey")
        return

    default = target_col if target_col in num_cols else num_cols[0]
    col_select = ui.select(num_cols, value=default, label="列を選択").props("outlined dense")
    chart_container = ui.column().classes("full-width")

    def _draw():
        chart_container.clear()
        col = col_select.value
        if not col:
            return
        series = df[col].dropna().values
        if len(series) < 5:
            return
        with chart_container:
            result = sp_stats.probplot(series, dist="norm")
            osm, osr = result[0]  # 理論分位数, サンプル分位数
            slope, intercept, r_val = result[1]  # 回帰パラメータ
            fig = go.Figure()
            fig.add_scatter(x=osm, y=osr, mode="markers",
                            marker=dict(color="#00d4ff", size=4), name="データ")
            x_range = [osm.min(), osm.max()]
            fig.add_scatter(x=x_range, y=[intercept + slope * x for x in x_range],
                            mode="lines", line=dict(color="#ff6b6b", dash="dash"), name="理論線")
            fig.update_layout(title=f"{col} QQプロット（正規性チェック）",
                              xaxis_title="理論分位数", yaxis_title="サンプル分位数")
            _dark_fig(fig, 400)
            _plotly_and_save(fig, f"eda_qqplot_{col_select.value}").classes("full-width")
            # Shapiro-Wilk検定
            if len(series) <= 5000:
                stat, p = sp_stats.shapiro(series[:5000])
                verdict = "✅ 正規分布に近い" if p > 0.05 else "⚠️ 正規分布から逸脱"
                ui.label(f"Shapiro-Wilk: W={stat:.4f}, p={p:.4g} → {verdict}").classes("text-caption text-cyan q-mt-xs")

    col_select.on("update:model-value", lambda: _draw())
    _draw()


# ═══════════════════════════════════════════════════════════
# 8. VIF（多重共線性）
# ═══════════════════════════════════════════════════════════
def _render_vif(df: pd.DataFrame, target_col: str) -> None:
    num_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]
    if len(num_cols) < 2:
        ui.label("説明変数が2列未満です").classes("text-caption text-grey")
        return

    chart_container = ui.column().classes("full-width")

    def _calc():
        chart_container.clear()
        with chart_container:
            ui.label("⏳ VIF計算中...").classes("text-caption text-grey")
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            work_df = df[num_cols[:50]].dropna()
            if work_df.shape[0] < 5:
                with chart_container:
                    ui.label("データが少なすぎます").classes("text-caption text-amber")
                return
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(work_df)
            vif_data = []
            for i, col in enumerate(work_df.columns):
                try:
                    vif_val = variance_inflation_factor(X, i)
                except Exception:
                    vif_val = float("nan")
                level = "🔴 高" if vif_val > 10 else ("🟡 中" if vif_val > 5 else "🟢 低")
                vif_data.append({"列名": col, "VIF": round(vif_val, 2), "共線性": level})
            vif_data.sort(key=lambda x: x["VIF"] if not np.isnan(x["VIF"]) else 999, reverse=True)
            chart_container.clear()
            with chart_container:
                ui.label("VIF > 10: 高い多重共線性 → 特徴量の削除を検討").classes("text-caption text-amber q-mb-sm")
                cols_def = [{"name": c, "label": c, "field": c, "sortable": True} for c in ["列名", "VIF", "共線性"]]
                ui.table(columns=cols_def, rows=vif_data, pagination={"rowsPerPage": 30}).classes(
                    "full-width"
                ).props("dense flat separator=cell")
        except ImportError:
            chart_container.clear()
            with chart_container:
                ui.label("statsmodelsが未インストールです (pip install statsmodels)").classes("text-caption text-red")
        except Exception as e:
            chart_container.clear()
            with chart_container:
                ui.label(f"VIF計算エラー: {e}").classes("text-caption text-red")

    ui.button("📏 VIFを計算", on_click=_calc).props("color=cyan no-caps size=sm").classes("q-mb-sm")


# ═══════════════════════════════════════════════════════════
# 9. 外れ値検出
# ═══════════════════════════════════════════════════════════
def _render_outliers(df: pd.DataFrame, target_col: str) -> None:
    import plotly.express as px

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        ui.label("数値列がありません").classes("text-caption text-grey")
        return

    default_col = target_col if target_col in num_cols else num_cols[0]
    with ui.row().classes("items-center q-gutter-sm q-mb-sm"):
        col_select = ui.select(num_cols, value=default_col, label="列を選択").props("outlined dense")
        method_sel = ui.toggle({"iqr": "IQR", "zscore": "Z-score"}, value="iqr").props("dense no-caps")

    chart_container = ui.column().classes("full-width")

    def _draw():
        chart_container.clear()
        col = col_select.value
        if not col:
            return
        series = df[col].dropna()
        if series.empty:
            return
        method = method_sel.value
        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = series[(series < lower) | (series > upper)]
        else:
            z = np.abs((series - series.mean()) / series.std())
            outliers = series[z > 3]
            lower, upper = series.mean() - 3 * series.std(), series.mean() + 3 * series.std()

        with chart_container:
            with ui.row().classes("q-gutter-md q-mb-sm"):
                ui.label(f"外れ値: {len(outliers)}件").classes("text-body2 text-bold text-amber")
                ui.label(f"全体: {len(series)}件").classes("text-caption text-grey")
                ui.label(f"下限: {lower:.4g} | 上限: {upper:.4g}").classes("text-caption text-cyan")
            fig = px.box(df, y=col, points="outliers", title=f"{col} ({method.upper()})", template="plotly_dark")
            _dark_fig(fig, 350)
            _plotly_and_save(fig, f"eda_outlier_{col}_{method}").classes("full-width")

    col_select.on("update:model-value", lambda: _draw())
    method_sel.on_value_change(lambda: _draw())
    _draw()


# ═══════════════════════════════════════════════════════════
# 10. 欠損解析
# ═══════════════════════════════════════════════════════════
def _render_missing(df: pd.DataFrame) -> None:
    import plotly.express as px

    total_missing = df.isna().sum().sum()
    if total_missing == 0:
        ui.label("✅ データセットに欠損値はありません").classes("text-body1 text-green text-bold")
        return

    with ui.row().classes("q-gutter-md q-mb-md"):
        ui.label(f"総欠損セル数: {total_missing:,}").classes("text-body2 text-bold text-amber")
        ui.label(f"欠損行数: {df.isna().any(axis=1).sum():,}").classes("text-body2 text-grey")

    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=True)

    fig = px.bar(x=missing_counts.values, y=missing_counts.index, orientation="h",
                 title="列ごとの欠損値数", labels={"x": "欠損数", "y": "列名"},
                 template="plotly_dark", color_discrete_sequence=["#fbbf24"], text_auto=True)
    _dark_fig(fig, max(300, len(missing_counts) * 25))
    _plotly_and_save(fig, "eda_missing_values").classes("full-width")


# ═══════════════════════════════════════════════════════════
# 11. 次元削減（PCA / t-SNE + 寄与率・ローディング）
# ═══════════════════════════════════════════════════════════
def _render_dim_reduction(df: pd.DataFrame, target_col: str, state: dict) -> None:
    """PCA + t-SNE を左右並列表示（プルダウン廃止・正方形・大きめフォント）。"""
    import plotly.express as px
    import plotly.graph_objects as go

    num_df = df.select_dtypes(include="number")
    # 欠損フィルタを緩和（50%以上の有効データがあれば保持）
    thresh = max(1, int(len(num_df) * 0.5))  # ← ✅ 0.2 → 0.5 に緩和
    num_df = num_df.dropna(axis=1, thresh=thresh)
    
    # 残った欠損を中央値で補完（行削除は最小限に）
    if not num_df.empty:
        num_df = num_df.fillna(num_df.median(numeric_only=True))
        # 完全に欠損した行のみ削除
        num_df = num_df.dropna(how='all')
    
    # 最低限のデータがなければフォールバック表示
    feature_cols = [c for c in num_df.columns if c != target_col]
    if len(feature_cols) < 2 or num_df.shape[0] < 3:
        logger.warning(f"⚠️ 次元削減に必要なデータ不足: features={len(feature_cols)}, rows={num_df.shape[0]}")
        with ui.expansion("⚠️ 表示に必要なデータが不足しています", icon="warning"):
            ui.label(f"• 有効な特徴量: {len(feature_cols)} 列（2列以上必要）").classes("text-caption")
            ui.label(f"• 有効な行数: {num_df.shape[0]} 行（3行以上必要）").classes("text-caption")
            ui.label("• 原因: 欠損値が多い、または数値型でない記述子が含まれています").classes("text-caption text-grey")
        return


    # 並列表示コンテナ
    with ui.row().classes("full-width q-gutter-sm q-mb-sm"):
        pca_col = ui.column().classes("col")
        tsne_col = ui.column().classes("col")
    chart_container = ui.column().classes("full-width")  # ローディング / エラー表示用

    _BASE = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0f0", size=11),
        width=440, height=440,
        margin=dict(l=55, r=15, t=50, b=55),
        hovermode="closest",
    )

    async def _run():
        pca_col.clear()
        tsne_col.clear()
        X = num_df[feature_cols].values
        y = num_df[target_col].values if target_col in num_df.columns else None
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X)

        # ── PCA ──────────────────────────────────────────────
        try:
            from sklearn.decomposition import PCA
            n_comp = min(len(feature_cols), 10)
            pca = PCA(n_components=n_comp)
            X_pca = pca.fit_transform(X_scaled)
            ev = pca.explained_variance_ratio_

            plot_df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1]})
            if y is not None:
                plot_df[target_col] = y[:len(X_pca)]

            fig = px.scatter(
                plot_df, x="PC1", y="PC2",
                color=target_col if y is not None else None,
                color_continuous_scale="Viridis", template="plotly_dark",
                title=f"PCA  PC1: {ev[0]:.1%} / PC2: {ev[1]:.1%}",
            )
            fig.update_layout(
                **_BASE,
                title=dict(font=dict(size=14)),
                xaxis=dict(
                    title=dict(text=f"PC1 ({ev[0]:.1%})", font=dict(size=13)),
                    tickfont=dict(size=11),
                    gridcolor="rgba(255,255,255,0.10)",
                    scaleanchor="y", scaleratio=1,  # 正方形
                ),
                yaxis=dict(
                    title=dict(text=f"PC2 ({ev[1]:.1%})", font=dict(size=13)),
                    tickfont=dict(size=11),
                    gridcolor="rgba(255,255,255,0.10)",
                ),
            )
            with pca_col:
                ui.label("📐 PCA（主成分分析）").classes("text-subtitle2 text-bold q-mb-xs")
                _plotly_and_save(fig, "eda_pca_scatter").classes("full-width")

                # 寄与率（コンパクト）
                fig2 = go.Figure()
                fig2.add_bar(
                    x=[f"PC{i+1}" for i in range(n_comp)],
                    y=ev, name="寄与率", marker_color="#00d4ff",
                )
                fig2.add_scatter(
                    x=[f"PC{i+1}" for i in range(n_comp)],
                    y=np.cumsum(ev), name="累積",
                    line=dict(color="#fbbf24", width=2),
                )
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0f0", size=10),
                    height=180, margin=dict(l=45, r=10, t=28, b=35),
                    title=dict(text="寄与率", font=dict(size=13)),
                    yaxis_title="寄与率",
                )
                _plotly_and_save(fig2, "eda_pca_variance").classes("full-width")

        except Exception as e:
            with pca_col:
                ui.label(f"PCA エラー: {e}").classes("text-caption text-red")

        # ── t-SNE ─────────────────────────────────────────────
        try:
            from sklearn.manifold import TSNE
            n_samples = min(X_scaled.shape[0], 3000)
            perp = min(30, n_samples - 1)
            X_2d = TSNE(
                n_components=2, perplexity=perp, random_state=42,
            ).fit_transform(X_scaled[:n_samples])
            y_sub = y[:n_samples] if y is not None else None

            plot_df2 = pd.DataFrame({"t-SNE 1": X_2d[:, 0], "t-SNE 2": X_2d[:, 1]})
            if y_sub is not None:
                plot_df2[target_col] = y_sub[:len(X_2d)]

            fig_t = px.scatter(
                plot_df2, x="t-SNE 1", y="t-SNE 2",
                color=target_col if y_sub is not None else None,
                color_continuous_scale="Viridis", template="plotly_dark",
                title="t-SNE 2D",
            )
            fig_t.update_layout(
                **_BASE,
                title=dict(font=dict(size=14)),
                xaxis=dict(
                    title=dict(text="t-SNE 1", font=dict(size=13)),
                    tickfont=dict(size=11),
                    gridcolor="rgba(255,255,255,0.10)",
                    scaleanchor="y", scaleratio=1,  # 正方形
                ),
                yaxis=dict(
                    title=dict(text="t-SNE 2", font=dict(size=13)),
                    tickfont=dict(size=11),
                    gridcolor="rgba(255,255,255,0.10)",
                ),
            )
            with tsne_col:
                ui.label("🎯 t-SNE").classes("text-subtitle2 text-bold q-mb-xs")
                _plotly_and_save(fig_t, "eda_tsne_scatter").classes("full-width")
                if n_samples < num_df.shape[0]:
                    ui.label(
                        f"※ 速度のため {n_samples} サンプルを使用"
                    ).classes("text-caption text-grey q-mt-xs")

        except Exception as e:
            with tsne_col:
                ui.label(f"t-SNE エラー: {e}").classes("text-caption text-red")

    ui.button("🌀 PCA + t-SNE を再実行", on_click=_run).props(
        "color=cyan no-caps size=sm outline"
    ).classes("q-mb-sm")

    # ── 初回表示時に自動実行 ──
    ui.timer(0.3, _run, once=True)


# ═══════════════════════════════════════════════════════════
# 12. 特徴量重要度（相互情報量）
# ═══════════════════════════════════════════════════════════
def _render_feature_importance(df: pd.DataFrame, target_col: str) -> None:
    import plotly.graph_objects as go

    num_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]
    if not num_cols or target_col not in df.columns:
        ui.label("目的変数または説明変数が不足しています").classes("text-caption text-grey")
        return

    chart_container = ui.column().classes("full-width")

    def _calc():
        chart_container.clear()
        with chart_container:
            ui.label("⏳ 特徴量重要度計算中...").classes("text-caption text-grey")
        try:
            work_df = df[num_cols + [target_col]].dropna()
            X = work_df[num_cols].values
            y = work_df[target_col].values

            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            is_class = df[target_col].nunique() < 20
            if is_class:
                mi = mutual_info_classif(X, y, random_state=42)
            else:
                mi = mutual_info_regression(X, y, random_state=42)

            mi_series = pd.Series(mi, index=num_cols).sort_values(ascending=True)
            top = mi_series.tail(min(20, len(mi_series)))

            chart_container.clear()
            with chart_container:
                fig = go.Figure(go.Bar(
                    x=top.values, y=top.index, orientation="h",
                    marker_color=["#00d4ff" if v > top.quantile(0.75) else "#fbbf24" if v > top.quantile(0.25) else "#555"
                                  for v in top.values]
                ))
                fig.update_layout(title="特徴量重要度（相互情報量）TOP20",
                                  xaxis_title="Mutual Information", yaxis=dict(autorange="reversed"))
                _dark_fig(fig, max(300, len(top) * 25))
                _plotly_and_save(fig, "eda_mutual_information").classes("full-width")
        except Exception as e:
            chart_container.clear()
            with chart_container:
                ui.label(f"エラー: {e}").classes("text-caption text-red")

    ui.button("⭐ 重要度を再計算", on_click=_calc).props("color=cyan no-caps size=sm outline").classes("q-mb-sm")

    # ── 初回表示時に自動実行 ──
    ui.timer(0.5, _calc, once=True)


# ═══════════════════════════════════════════════════════════
# 13. ボックスプロット（特徴量分布の可視化）
# ═══════════════════════════════════════════════════════════
def _render_box_plots(df: pd.DataFrame, target_col: str) -> None:
    """数値特徴量のボックスプロット（目的変数との相関上位5つ）。"""
    import plotly.graph_objects as go

    num_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]
    if not num_cols:
        ui.label("数値特徴量がありません").classes("text-caption text-grey")
        return

    # 目的変数との相関上位5列を選択
    if target_col and target_col in df.columns:
        corr = df[num_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
        top_cols = corr.head(min(5, len(corr))).index.tolist()
    else:
        top_cols = num_cols[:5]

    fig = go.Figure()
    colors = ["#00d4ff", "#fbbf24", "#4ade80", "#a78bfa", "#f472b6"]
    for i, col in enumerate(top_cols):
        fig.add_trace(go.Box(
            y=df[col].dropna(), name=col,
            marker_color=colors[i % len(colors)],
            boxmean="sd",
        ))

    fig.update_layout(
        title="特徴量のボックスプロット（相関上位5）",
        yaxis_title="値",
        showlegend=False,
    )
    _dark_fig(fig, 380)
    _plotly_and_save(fig, "eda_boxplot").classes("full-width")


# ═══════════════════════════════════════════════════════════
# 14. QQプロット（正規性検定）
# ═══════════════════════════════════════════════════════════
def _render_qq_plots(df: pd.DataFrame, target_col: str) -> None:
    """目的変数と上位特徴量のQQプロット。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # QQ対象列: 目的変数 + 相関上位3特徴量
    num_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]
    qq_cols: list[str] = []
    if target_col and target_col in df.columns:
        qq_cols.append(target_col)
        if num_cols:
            corr = df[num_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
            qq_cols.extend(corr.head(3).index.tolist())
    else:
        qq_cols = num_cols[:4]

    if not qq_cols:
        ui.label("QQプロット対象列がありません").classes("text-caption text-grey")
        return

    try:
        from scipy import stats
    except ImportError:
        ui.label("QQプロットにはscipy が必要です").classes("text-caption text-red")
        return

    n_plots = len(qq_cols)
    fig = make_subplots(
        rows=1, cols=n_plots,
        subplot_titles=[c[:20] for c in qq_cols],
        horizontal_spacing=0.08,
    )

    colors = ["#00d4ff", "#fbbf24", "#4ade80", "#a78bfa"]
    for i, col in enumerate(qq_cols):
        data = df[col].dropna().values
        if len(data) < 5:
            continue
        osm, osr = stats.probplot(data, dist="norm", fit=False)

        fig.add_trace(go.Scatter(
            x=osm, y=osr, mode="markers",
            marker=dict(size=4, color=colors[i % len(colors)], opacity=0.6),
            name=col,
        ), row=1, col=i + 1)

        # 参照線 (y=x scaled)
        mn, mx = float(osm.min()), float(osm.max())
        slope, intercept = np.polyfit(osm, osr, 1)
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[slope * mn + intercept, slope * mx + intercept],
            mode="lines", line=dict(color="rgba(255,255,255,0.35)", dash="dash"),
            showlegend=False,
        ), row=1, col=i + 1)

    fig.update_layout(title="QQプロット（正規性検定）", showlegend=False)
    _dark_fig(fig, 350)
    _plotly_and_save(fig, "eda_qqplot").classes("full-width")
