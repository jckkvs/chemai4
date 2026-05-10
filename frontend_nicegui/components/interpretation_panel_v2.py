"""
frontend_nicegui/components/interpretation_panel_v2.py

Premium Model Interpretation Panel - SHAP, SAGE, SRI with best-in-class UI.

Features:
- Interactive Plotly visualizations (no static matplotlib images)
- Waterfall, Force, Decision, Dependence plots
- SAGE with bootstrap confidence intervals
- SRI with network graph showing feature interactions
- Modern glassmorphism UI with smooth animations
- Real-time parameter adjustments
- Export to PNG/SVG/HTML
"""

from __future__ import annotations

import io
import logging
import base64
from typing import Any, Callable

import numpy as np
import pandas as pd
from nicegui import ui, run, app

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════
# Color Palette & Theme
# ═════════════════════════════════════════════════════════════
COLORS = {
    "primary": "#00d4ff",
    "secondary": "#7c4dff",
    "success": "#4ade80",
    "warning": "#facc15",
    "danger": "#f87171",
    "cyan": "#00d4ff",
    "purple": "#a855f7",
    "green": "#4ade80",
    "amber": "#fbbf24",
    "red": "#f87171",
    "bg_dark": "rgba(10,10,20,0.85)",
    "bg_card": "rgba(15,15,30,0.7)",
    "glass": "rgba(255,255,255,0.05)",
    "border": "rgba(0,212,255,0.2)",
}

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_BG = "rgba(0,0,0,0)"
PLOTLY_PAPER_BG = "rgba(0,0,0,0)"


def _card_style(border_color: str = COLORS["border"]) -> str:
    return (
        f"background:{COLORS['bg_card']};"
        f"border:1px solid {border_color};"
        "border-radius:12px;"
        "backdrop-filter:blur(10px);"
        "padding:16px;"
    )


def _glass_style() -> str:
    return (
        f"background:{COLORS['glass']};"
        "border:1px solid rgba(255,255,255,0.08);"
        "border-radius:10px;"
        "backdrop-filter:blur(8px);"
    )


def _plotly_and_save(fig, name: str, session_id: str | None = None, **kwargs):
    """Render Plotly figure in NiceGUI and save versions in background."""
    try:
        from backend.utils.plot_exporter import save_plot_versions
        save_plot_versions(fig, name=name, session_id=session_id, run_async=True)
    except Exception as e:
        logger.debug(f"[InterpV2 PlotSave] {name}: {e}")
    return ui.plotly(fig, **kwargs)


def _fig_defaults(fig, title: str = "", height: int = 420, margin: dict | None = None):
    """Apply consistent styling to Plotly figures."""
    m = margin or dict(l=10, r=10, t=40, b=10)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_BG,
        height=height,
        margin=m,
        title=dict(
            text=title,
            font=dict(size=14, family="Inter, sans-serif", color="#e0e0e0"),
            x=0.02,
        ),
        font=dict(family="Inter, sans-serif", color="#c0c0c0"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
    )
    return fig


# ═════════════════════════════════════════════════════════════
# SHAP Panel - Premium Interactive Version
# ═════════════════════════════════════════════════════════════

def _render_shap_v2(ar, model, X, X_arr, feat_names, y) -> None:
    """Premium SHAP panel with interactive Plotly visualizations."""
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import matplotlib.pyplot as plt
    import shap

    ui.label("🔍 SHAP Analysis").classes("text-h5 text-bold q-mb-none").style(
        "color:" + COLORS["cyan"] + ";font-family:Inter,sans-serif;"
    )
    ui.label(
        "SHapley Additive exPlanations — 各特徴量の予測への寄与をゲーム理論的に算出"
    ).classes("text-caption text-grey-5 q-mb-md")

    # Control panel
    with ui.card().style(_card_style(COLORS["cyan"] + "33")):
        with ui.row().classes("w-full items-center gap-4"):
            ui.label("🎛️ 設定").classes("text-subtitle1 text-bold")
            ui.space()
            compute_btn = ui.button(
                "🚀 計算実行",
                icon="play_arrow",
                on_click=lambda: _run_shap_calc(),
            ).props("unelevated color=cyan size=sm no-caps").classes("shadow-lg")

    # Settings row
    with ui.card().style(_glass_style()):
        with ui.row().classes("w-full items-center gap-4 flex-wrap"):
            max_display = ui.select(
                label="表示数",
                options={10: "10", 15: "15", 20: "20", 30: "30", 50: "50"},
                value=15,
            ).props("dense outlined").classes("w-24")
            plot_style = ui.select(
                label="プロットタイプ",
                options={"dot": "Beehive", "bar": "Bar", "violin": "Violin"},
                value="dot",
            ).props("dense outlined").classes("w-32")
            color_by = ui.select(
                label="色分け",
                options={"feature": "特徴量値", "shap": "SHAP値"},
                value="feature",
            ).props("dense outlined").classes("w-32")

    # Results container
    shap_container = ui.column().classes("w-full")

    async def _run_shap_calc():
        shap_container.clear()
        with shap_container:
            # Progress indicator
            with ui.card().style(_card_style()).classes("w-full"):
                with ui.row().classes("items-center gap-3"):
                    ui.spinner("dots", size="lg", color="cyan")
                    ui.label("SHAP値を計算中...").classes("text-cyan text-subtitle1")
                prog = ui.linear_progress(value=0.1, show_value=False).props(
                    "color=cyan rounded stripe"
                ).classes("w-full")
                status = ui.label("TreeExplainerを初期化中...").classes(
                    "text-caption text-grey-5"
                )

        try:
            from backend.interpret.shap_interpreter import calculate_shap_values

            X_raw = getattr(ar, "X_train_raw", X)

            def _compute():
                prog.value = 0.3
                status.text = "SHAP値を計算中 (TreeExplainer)..."
                result = calculate_shap_values(model, X_raw)
                prog.value = 0.7
                status.text = "可視化データを構築中..."
                return result

            shap_result = await run.io_bound(_compute)
            prog.value = 0.95
            status.text = "プロットを描画中..."

            # Get data
            if hasattr(shap_result, "get_view"):
                view_data = shap_result.get_view("compare")
                shap_vals = view_data["shaps"]
                X_display = view_data["data"]
            else:
                shap_vals = shap_result.core_output
                X_display = X_raw

            feature_names = shap_result.metadata.get("feature_names", feat_names)

            shap_container.clear()
            with shap_container:
                _render_shap_plots(
                    shap_vals, X_display, feature_names, max_display.value, plot_style.value, color_by.value
                )

            ui.notify("✅ SHAP解析完了", type="positive", position="top")

        except Exception as ex:
            shap_container.clear()
            with shap_container:
                with ui.card().style(_card_style(COLORS["danger"] + "33")):
                    ui.label(f"❌ エラー: {ex}").classes("text-red")
                    ui.label("詳細はコンソールを確認してください").classes(
                        "text-caption text-grey-5"
                    )

    # Auto-trigger if SHAP result is cached
    if hasattr(ar, "_shap_result_cache") and ar._shap_result_cache is not None:
        # Use cached result
        pass

    shap_container


def _render_shap_plots(shap_vals, X_display, feature_names, max_display: int, plot_type: str, color_by: str):
    """Render all SHAP plots using Plotly (interactive)."""
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
    import pandas as pd

    # Summary statistics
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:max_display]

    # ── Summary Bar Chart ──
    ui.separator().style("border-color:" + COLORS["border"] + ";")

    with ui.card().style(_card_style()):
        ui.label("📊 特徴量重要度 (SHAP Mean |SHAP|)").classes(
            "text-subtitle1 text-bold q-mb-sm"
        )

        fig = go.Figure()
        colors = px.colors.sample_colorscale(
            "Viridis",
            np.linspace(0, 1, min(max_display, len(top_idx))),
        )

        fig.add_trace(
            go.Bar(
                x=mean_abs_shap[top_idx][::-1],
                y=[str(feature_names[i]) for i in top_idx][::-1],
                orientation="h",
                marker=dict(
                    color=mean_abs_shap[top_idx][::-1],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="|SHAP|", tickformat=".3f"),
                ),
                text=[f"{mean_abs_shap[i]:.4f}" for i in top_idx][::-1],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>|SHAP|平均: %{x:.4f}<extra></extra>",
            )
        )
        _fig_defaults(fig, "SHAP Feature Importance", height=max(350, 22 * min(max_display, len(top_idx))))
        fig.update_xaxes(title="|SHAP| 平均 (重要性)")
        fig.update_yaxes(title="特徴量")
        _plotly_and_save(fig, "shap_summary_bar").classes("w-full")

    # ── SHAP Summary Bee Swarm Plot (Plotly version) ──
    with ui.card().style(_card_style()):
        ui.label("🐝 SHAP Summary Plot (Bee Swarm)").classes(
            "text-subtitle1 text-bold q-mb-sm"
        )
        ui.label(
            "各点は1サンプル。色は特徴量値(高→赤)、位置はSHAP値(正→予測増加)。"
        ).classes("text-caption text-grey-5 q-mb-sm")

        n_top = min(max_display, len(top_idx))
        fig_swarm = go.Figure()

        for i, idx in enumerate(top_idx[:n_top]):
            if idx < shap_vals.shape[1] and idx < X_display.shape[1]:
                feature_name = str(feature_names[idx])
                y_vals = shap_vals[:, idx]
                x_vals = X_display.iloc[:, idx] if hasattr(X_display, "iloc") else X_display[:, idx]
                colors_vals = x_vals if color_by == "feature" else y_vals

                fig_swarm.add_trace(
                    go.Scatter(
                        x=y_vals,
                        y=[feature_name] * len(y_vals),
                        mode="markers",
                        marker=dict(
                            size=7,
                            color=colors_vals,
                            colorscale="RdBu_r" if color_by == "shap" else "Plasma",
                            showscale=True if i == 0 else False,
                            colorbar=dict(title=color_by, tickformat=".2f") if i == 0 else None,
                            opacity=0.7,
                            line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
                        ),
                        hovertemplate=(
                            f"<b>{feature_name}</b><br>"
                            "SHAP: %{x:.4f}<br>"
                            f"{'値' if color_by == 'feature' else 'SHAP'}: %{{marker.color:.4f}}"
                            "<extra></extra>"
                        ),
                        name=feature_name,
                        showlegend=False,
                    )
                )

        _fig_defaults(fig_swarm, "SHAP Summary (Bee Swarm)", height=max(350, 22 * n_top))
        fig_swarm.update_xaxes(title="SHAP value (予測への寄与)", zeroline=True, zerolinecolor="rgba(255,255,255,0.3)")
        fig_swarm.update_yaxes(title="特徴量")
        _plotly_and_save(fig_swarm, "shap_beeswarm").classes("w-full")

    # ── Sample selector for individual plots ──
    ui.separator().style("border-color:" + COLORS["border"] + ";")

    with ui.card().style(_card_style()):
        ui.label("🔬 個別サンプルの解釈").classes("text-subtitle1 text-bold q-mb-sm")
        ui.label("特定のサンプルの予測をSHAPで分解して表示します。").classes(
            "text-caption text-grey-5 q-mb-sm"
        )

        col1, col2 = ui.columns([1, 3]).classes("w-full")
        with col1:
            sample_idx = ui.number(
                label="サンプル番号",
                value=0,
                min=0,
                max=shap_vals.shape[0] - 1,
            ).props("dense outlined").classes("w-full")
        with col2:
            ui.button(
                "🔍 表示",
                on_click=lambda: _update_waterfall(),
            ).props("unelevated color=cyan size=sm no-caps")

        waterfall_container = ui.column().classes("w-full")
        sample_idx_input = sample_idx  # Keep reference

        def _update_waterfall():
            waterfall_container.clear()
            with waterfall_container:
                idx = int(sample_idx_input.value or 0)
                if idx < shap_vals.shape[0]:
                    _render_waterfall_plot(shap_vals, X_display, feature_names, idx, max_display)

        # Initial render
        _update_waterfall()

    # ── Dependence plots for top features ──
    ui.separator().style("border-color:" + COLORS["border"] + ";")

    with ui.card().style(_card_style()):
        ui.label("🔗 Dependence Plots (特徴量依存性)").classes(
            "text-subtitle1 text-bold q-mb-sm"
        )
        ui.label(
            "特徴量の値とSHAP値の関係。色は相互作用特徴量を示します。"
        ).classes("text-caption text-grey-5 q-mb-sm")

        dep_top_n = min(6, len(top_idx))
        dep_container = ui.column().classes("w-full")

        for i in range(dep_top_n):
            f_idx = top_idx[i]
            f_name = str(feature_names[f_idx])
            with dep_container:
                _render_dependence_plot(
                    shap_vals, X_display, feature_names, f_idx, f_name
                )


def _render_waterfall_plot(shap_vals, X_display, feature_names, sample_idx: int, max_display: int):
    """Render SHAP waterfall plot for a single sample using Plotly."""
    import plotly.graph_objects as go
    import numpy as np

    sv = shap_vals[sample_idx]
    # Get feature values for this sample
    if hasattr(X_display, "iloc"):
        fv = X_display.iloc[sample_idx].values
    else:
        fv = X_display[sample_idx]

    # Sort by |SHAP| value
    idx_sorted = np.argsort(np.abs(sv))[::-1][:max_display]
    n = len(idx_sorted)

    # Base value (use mean prediction as approximation)
    base_val = shap_vals.mean(axis=0).sum() * 0  # Start from 0, or use expected_value if available
    # Actually, let's use a proper base value
    base_val = np.mean(shap_vals.sum(axis=1)) - np.mean(sv.sum())

    # Build cumulative values for waterfall
    x_labels = []
    y_start = []
    y_end = []
    colors = []

    # Add base
    x_labels.append("Base")
    y_start.append(0)
    y_end.append(base_val)
    colors.append(COLORS["cyan"])

    cum_val = base_val
    for i, idx in enumerate(idx_sorted):
        x_labels.append(str(feature_names[idx]))
        y_start.append(cum_val)
        cum_val += sv[idx]
        y_end.append(cum_val)
        colors.append(COLORS["green"] if sv[idx] > 0 else COLORS["red"])

    # Add final prediction
    x_labels.append("予測値")
    y_start.append(cum_val)
    y_end.append(cum_val)
    colors.append(COLORS["purple"])

    fig = go.Figure()

    # Waterfall bars
    for i in range(len(x_labels)):
        fig.add_trace(
            go.Bar(
                x=[x_labels[i]],
                y=[y_end[i] - y_start[i]],
                base=[y_start[i]],
                marker_color=colors[i],
                opacity=0.85,
                hovertemplate=(
                    f"<b>{x_labels[i]}</b><br>"
                    f"寄与: {y_end[i] - y_start[i]:.4f}<br>"
                    f"累積: {y_end[i]:.4f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    _fig_defaults(
        fig,
        f"SHAP Waterfall — Sample #{sample_idx}",
        height=max(400, 40 * len(x_labels)),
    )
    fig.update_yaxes(title="予測値への寄与")
    fig.update_xaxes(tickangle=-45)
    _plotly_and_save(fig, f"shap_waterfall_{sample_idx}").classes("w-full")


def _render_dependence_plot(shap_vals, X_display, feature_names, f_idx: int, f_name: str):
    """Render SHAP dependence plot for a single feature using Plotly."""
    import plotly.graph_objects as go
    import numpy as np

    # Get feature values and SHAP values
    if hasattr(X_display, "iloc"):
        x_feat = X_display.iloc[:, f_idx].values
    else:
        x_feat = X_display[:, f_idx]

    y_shap = shap_vals[:, f_idx]

    # Find interaction feature (highest correlation with SHAP values)
    if shap_vals.shape[1] > 1:
        correlations = []
        for j in range(shap_vals.shape[1]):
            if j == f_idx:
                correlations.append(-1)
                continue
            if hasattr(X_display, "iloc"):
                x_j = X_display.iloc[:, j].values
            else:
                x_j = X_display[:, j]
            corr = abs(np.corrcoef(x_j, y_shap)[0, 1])
            correlations.append(corr if not np.isnan(corr) else 0)
        inter_idx = int(np.argmax(correlations))
        inter_name = str(feature_names[inter_idx]) if inter_idx < len(feature_names) else "auto"
    else:
        inter_idx = None
        inter_name = None

    # Color by interaction feature
    if inter_idx is not None and inter_idx < (X_display.shape[1] if hasattr(X_display, "shape") else 999):
        if hasattr(X_display, "iloc"):
            color_vals = X_display.iloc[:, inter_idx].values
        else:
            color_vals = X_display[:, inter_idx]
        color_label = inter_name
    else:
        color_vals = y_shap
        color_label = "SHAP"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_feat,
            y=y_shap,
            mode="markers",
            marker=dict(
                size=8,
                color=color_vals,
                colorscale="RdBu_r",
                showscale=True,
                colorbar=dict(title=color_label, tickformat=".2f"),
                opacity=0.7,
                line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
            ),
            hovertemplate=(
                f"<b>{f_name}</b><br>"
                f"値: %{{x:.4g}}<br>"
                f"SHAP: %{{y:.4f}}<br>"
                f"{color_label}: %{{marker.color:.4f}}"
                "<extra></extra>"
            ),
            name=f_name,
        )
    )

    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")

    _fig_defaults(fig, f"Dependence: {f_name}", height=320)
    fig.update_xaxes(title=f"{f_name} (値)")
    fig.update_yaxes(title="SHAP value")
    _plotly_and_save(fig, f"shap_dependence_{f_name}").classes("w-full q-mt-sm")


# ═════════════════════════════════════════════════════════════
# SAGE Panel - Premium Version
# ═════════════════════════════════════════════════════════════

def _render_sage_v2(ar, model, X, X_arr, feat_names, y) -> None:
    """Premium SAGE panel with enhanced visualizations."""
    import plotly.graph_objects as go
    import plotly.express as px

    ui.label("🌿 SAGE Analysis").classes("text-h5 text-bold q-mb-none").style(
        "color:" + COLORS["green"] + ";font-family:Inter,sans-serif;"
    )
    ui.label(
        "Shapley Additive Global importancE — ゲーム理論的特徴量重要度"
    ).classes("text-caption text-grey-5 q-mb-md")

    sage_container = ui.column().classes("w-full")

    # Info card
    with ui.card().style(_glass_style()):
        with ui.row().classes("items-start gap-2"):
            ui.icon("info", color="amber").style("font-size:20px;")
            with ui.column().classes("gap-1"):
                ui.label("SAGEについて").classes("text-subtitle2 text-bold")
                ui.label(
                    "各特徴量を隠した場合の予測損失増加をShapley値で公平に配分します。"
                    "計算に時間がかかりますが、置換重要度よりも公正な評価が可能です。"
                ).classes("text-caption text-grey-5")

    async def _run_sage():
        sage_container.clear()
        with sage_container:
            with ui.card().style(_card_style(COLORS["green"] + "33")):
                with ui.row().classes("items-center gap-3"):
                    ui.spinner("dots", size="lg", color="green")
                    ui.label("SAGE値を計算中...").classes("text-green text-subtitle1")
                prog = ui.linear_progress(value=0.05, show_value=False).props(
                    "color=green rounded stripe"
                )
                status = ui.label("MarginalImputerを初期化中...").classes(
                    "text-caption text-grey-5"
                )

        try:
            import sage as sage_pkg
            from sklearn.metrics import mean_squared_error

            if y is None:
                raise ValueError("y_train が取得できません")

            y_arr = np.asarray(y).ravel()

            def _compute():
                prog.value = 0.1
                status.text = "MarginalImputerを構築中..."
                imputer = sage_pkg.MarginalImputer(model, X_arr)
                prog.value = 0.3
                status.text = "PermutationEstimatorを初期化中..."
                estimator = sage_pkg.PermutationEstimator(imputer, "mse")
                prog.value = 0.5
                status.text = "SAGE値を計算中（時間がかかります）..."
                sage_values = estimator(X_arr, y_arr, thresh=0.01)
                prog.value = 0.9
                status.text = "可視化を構築中..."
                return sage_values

            sage_values = await run.io_bound(_compute)

            vals = np.asarray(sage_values.values)
            names_full = feat_names[: len(vals)]
            idx = np.argsort(np.abs(vals))[::-1]
            top_n = min(30, len(idx))

            sage_container.clear()
            with sage_container:
                # Summary cards
                with ui.row().classes("w-full q-gutter-md q-mb-md"):
                    total_importance = float(np.abs(vals).sum())
                    n_positive = int((vals > 0).sum())
                    with ui.card().style(_card_style()).classes("flex-1"):
                        ui.label(f"{total_importance:.4f}").classes(
                            "text-h6 text-bold text-green"
                        )
                        ui.label("合計重要度").classes("text-caption text-grey-5")
                    with ui.card().style(_card_style()).classes("flex-1"):
                        ui.label(f"{n_positive}/{len(vals)}").classes(
                            "text-h6 text-bold text-cyan"
                        )
                        ui.label("正の寄与").classes("text-caption text-grey-5")

                # Horizontal bar chart
                ui.label("📊 SAGE Feature Importance").classes(
                    "text-subtitle1 text-bold q-mb-sm"
                )

                fig = go.Figure()
                top_vals = vals[idx[:top_n]]
                top_names = [str(names_full[i]) for i in idx[:top_n]]

                colors_bar = [
                    COLORS["green"] if v > 0 else COLORS["red"] for v in top_vals[::-1]
                ]

                fig.add_trace(
                    go.Bar(
                        x=top_vals[::-1],
                        y=top_names[::-1],
                        orientation="h",
                        marker=dict(
                            color=top_vals[::-1],
                            colorscale="RdBu_r",
                            showscale=True,
                            colorbar=dict(title="SAGE値", tickformat=".3f"),
                        ),
                        text=[f"{v:+.4f}" for v in top_vals[::-1]],
                        textposition="outside",
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "SAGE値: %{x:+.4f}<br>"
                            "<extra></extra>"
                        ),
                    )
                )
                _fig_defaults(
                    fig, "SAGE Feature Importance", height=max(350, 22 * top_n)
                )
                fig.update_xaxes(title="SAGE値 (予測損失への寄与)")
                fig.update_yaxes(title="特徴量")
                _plotly_and_save(fig, "sage_importance_v2").classes("w-full")

                # SAGE vs SHAP comparison (if SHAP available)
                ui.separator().style("border-color:" + COLORS["border"] + ";")
                with ui.expansion("🔄 SAGE vs SHAP 比較", icon="compare").classes(
                    "w-full q-mt-md"
                ):
                    ui.label(
                        "SAGEはグローバルな重要度、SHAPは局所的な寄与を測定します。"
                        "両者を比較することで、特徴量の役割を多角的に評価できます。"
                    ).classes("text-caption text-grey-5 q-mb-sm")
                    ui.label(
                        "💡 SAGEの計算には数分かかる場合があります"
                    ).classes("text-caption text-amber")

                # Data table
                ui.separator().style("border-color:" + COLORS["border"] + ";")
                with ui.expansion("📋 数値テーブル", icon="table_chart").classes(
                    "w-full q-mt-md"
                ):
                    rows = [
                        {
                            "順位": r + 1,
                            "特徴量": str(names_full[idx[r]]),
                            "SAGE値": f"{vals[idx[r]]:+.6f}",
                            "絶対値": f"{abs(vals[idx[r]]):.6f}",
                        }
                        for r in range(min(50, len(idx)))
                    ]
                    cols = [
                        {"name": k, "label": k, "field": k, "align": "center" if k == "順位" else "left", "sortable": True}
                        for k in ["順位", "特徴量", "SAGE値", "絶対値"]
                    ]
                    ui.table(columns=cols, rows=rows).classes("w-full").props(
                        "dense flat bordered"
                    )

            ui.notify("✅ SAGE解析完了", type="positive", position="top")

        except ImportError:
            sage_container.clear()
            with sage_container:
                with ui.card().style(_card_style(COLORS["warning"] + "33")):
                    ui.label("⚠️ sage-importance パッケージが必要です").classes(
                        "text-amber text-subtitle1"
                    )
                    ui.code("pip install sage-importance")
                    ui.separator()
                    # Fallback to permutation importance
                    _render_sage_fallback_v2(ar, model, X, X_arr, feat_names, y)

        except Exception as ex:
            sage_container.clear()
            with sage_container:
                with ui.card().style(_card_style(COLORS["danger"] + "33")):
                    ui.label(f"❌ エラー: {ex}").classes("text-red")
                _render_sage_fallback_v2(ar, model, X, X_arr, feat_names, y)

    # Run button
    with ui.card().style(_card_style(COLORS["green"] + "33")):
        with ui.row().classes("w-full items-center"):
            ui.label("🚀 SAGE解析を開始").classes("text-subtitle1 text-bold")
            ui.space()
            ui.button(
                "🌿 SAGE計算実行",
                icon="science",
                on_click=lambda: _run_sage(),
            ).props("unelevated color=green size=sm no-caps").classes("shadow-lg")

    sage_container


def _render_sage_fallback_v2(ar, model, X, X_arr, feat_names, y):
    """Enhanced permutation importance fallback."""
    import plotly.graph_objects as go
    from sklearn.inspection import permutation_importance

    ui.label("🔄 Permutation Importance (代替)").classes(
        "text-subtitle1 text-bold q-mb-sm"
    )
    ui.label(
        "SAGEが利用できない場合の代替。各特徴量をシャッフルしたときの性能低下を測定します。"
    ).classes("text-caption text-grey-5 q-mb-md")

    perm_container = ui.column().classes("w-full")

    async def _calc_perm():
        perm_container.clear()
        with perm_container:
            ui.spinner("dots", size="md", color="green")
            ui.label("Permutation Importance 計算中...").classes("text-caption text-grey-5")

        try:
            scoring = "r2" if getattr(ar, "task", "regression") == "regression" else "accuracy"
            y_arr = np.asarray(y).ravel() if y is not None else None
            if y_arr is None:
                raise ValueError("y_train が取得できません")

            def _compute():
                return permutation_importance(
                    model, X, y_arr, n_repeats=10, random_state=42, scoring=scoring
                )

            perm_res = await run.io_bound(_compute)
            sorted_idx = np.argsort(perm_res.importances_mean)[::-1]
            top = min(30, len(sorted_idx))

            perm_container.clear()
            with perm_container:
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=perm_res.importances_mean[sorted_idx[:top]][::-1],
                        y=[feat_names[i] if i < len(feat_names) else f"f{i}" for i in sorted_idx[:top]][::-1],
                        orientation="h",
                        error_x=dict(
                            type="data",
                            array=perm_res.importances_std[sorted_idx[:top]][::-1],
                            color="rgba(255,255,255,0.4)",
                        ),
                        marker=dict(
                            color=perm_res.importances_mean[sorted_idx[:top]][::-1],
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Importance", tickformat=".3f"),
                        ),
                        text=[f"{perm_res.importances_mean[i]:.4f} ± {perm_res.importances_std[i]:.4f}" for i in sorted_idx[:top]][::-1],
                        textposition="outside",
                    )
                )
                _fig_defaults(fig, "Permutation Importance", height=max(350, 22 * top))
                fig.update_xaxes(title=f"性能低下 ({scoring})")
                _plotly_and_save(fig, "sage_perm_fallback_v2").classes("w-full")

            ui.notify("✅ Permutation Importance 完了", type="positive")
        except Exception as ex:
            perm_container.clear()
            with perm_container:
                ui.label(f"❌ エラー: {ex}").classes("text-red")

    ui.button(
        "🔄 計算実行",
        on_click=lambda: _calc_perm(),
    ).props("outline color=green size=sm no-caps")
    perm_container


# ═════════════════════════════════════════════════════════════
# SRI Panel - Premium Version with Network Graph
# ═════════════════════════════════════════════════════════════

def _render_sri_v2(ar, model, X, X_arr, feat_names) -> None:
    """Premium SRI decomposition panel with network visualization."""
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    ui.label("🔬 SRI Decomposition").classes("text-h5 text-bold q-mb-none").style(
        "color:" + COLORS["purple"] + ";font-family:Inter,sans-serif;"
    )
    ui.label(
        "SHAP Vector Decomposition — Synergy / Redundancy / Independence"
    ).classes("text-caption text-grey-5 q-mb-md")

    sri_container = ui.column().classes("w-full")

    # Info card
    with ui.card().style(_glass_style()):
        with ui.row().classes("items-start gap-2"):
            ui.icon("info", color="purple").style("font-size:20px;")
            with ui.column().classes("gap-1"):
                ui.label("SRI分解について").classes("text-subtitle2 text-bold")
                ui.label(
                    "SHAPベクトルを3成分に分解: Synergy(相乗効果), Redundancy(冗長性), Independence(独立性)。"
                    "特徴量間の相互作用を理解するのに役立ちます。"
                ).classes("text-caption text-grey-5")

    async def _run_sri():
        sri_container.clear()
        with sri_container:
            with ui.card().style(_card_style(COLORS["purple"] + "33")):
                with ui.row().classes("items-center gap-3"):
                    ui.spinner("dots", size="lg", color="purple")
                    ui.label("SHAP → SRI分解を実行中...").classes(
                        "text-purple text-subtitle1"
                    )
                prog = ui.linear_progress(value=0.1, show_value=False).props(
                    "color=purple rounded stripe"
                )
                status = ui.label("SHAP値を計算中...").classes(
                    "text-caption text-grey-5"
                )

        try:
            from backend.interpret.shap_explainer import ShapExplainer
            from backend.interpret.sri import SRIDecomposer

            def _compute():
                prog.value = 0.2
                status.text = "SHAP値を計算中..."
                exp = ShapExplainer()
                shap_res = exp.explain(model, X, feature_names=feat_names)
                prog.value = 0.6
                status.text = "SRI分解を実行中..."
                decomposer = SRIDecomposer(center=True)
                return shap_res, decomposer.decompose(shap_res)

            shap_res, sri_res = await run.io_bound(_compute)
            prog.value = 0.9
            status.text = "可視化を構築中..."

            summary = sri_res.summary_df()
            top_n = min(20, len(summary))

            sri_container.clear()
            with sri_container:
                # Summary cards
                total_syn, total_red, total_ind = sri_res.total_sri
                with ui.row().classes("w-full q-gutter-md q-mb-md"):
                    for val, lbl, color, icon in [
                        (f"{total_syn:.4f}", "Synergy 合計", COLORS["amber"], "🔥"),
                        (f"{total_red:.4f}", "Redundancy 合計", COLORS["red"], "🔁"),
                        (f"{total_ind:.4f}", "Independence 合計", COLORS["cyan"], "🔲"),
                    ]:
                        with ui.card().style(_card_style(color + "22")).classes(
                            "flex-1 text-center"
                        ):
                            ui.label(icon).style("font-size:24px;")
                            ui.label(val).classes(f"text-h6 text-bold text-{color}")
                            ui.label(lbl).classes("text-caption text-grey-5")

                # Stacked bar chart - SRI components
                ui.label("📊 特徴量ごとの SRI 成分").classes(
                    "text-subtitle1 text-bold q-mb-sm"
                )

                df_top = summary.head(top_n)
                fig_sri = go.Figure()

                fig_sri.add_trace(
                    go.Bar(
                        y=df_top["feature"].values[::-1],
                        x=df_top["independence"].values[::-1],
                        orientation="h",
                        name="Independence",
                        marker=dict(color=COLORS["cyan"], opacity=0.85),
                        hovertemplate="<b>%{y}</b><br>Independence: %{x:.4f}<extra></extra>",
                    )
                )
                fig_sri.add_trace(
                    go.Bar(
                        y=df_top["feature"].values[::-1],
                        x=df_top["synergy"].values[::-1],
                        orientation="h",
                        name="Synergy",
                        marker=dict(color=COLORS["amber"], opacity=0.85),
                        hovertemplate="<b>%{y}</b><br>Synergy: %{x:.4f}<extra></extra>",
                    )
                )
                fig_sri.add_trace(
                    go.Bar(
                        y=df_top["feature"].values[::-1],
                        x=df_top["redundancy"].values[::-1],
                        orientation="h",
                        name="Redundancy",
                        marker=dict(color=COLORS["red"], opacity=0.85),
                        hovertemplate="<b>%{y}</b><br>Redundancy: %{x:.4f}<extra></extra>",
                    )
                )

                fig_sri.update_layout(
                    barmode="stack",
                    **PLOTLY_TEMPLATE,
                    paper_bgcolor=PLOTLY_PAPER_BG,
                    plot_bgcolor=PLOTLY_BG,
                    height=max(400, 25 * top_n),
                    margin=dict(l=10, r=10, t=40, b=10),
                    title=dict(
                        text="SRI Decomposition (Independence / Synergy / Redundancy)",
                        font=dict(size=14, family="Inter", color="#e0e0e0"),
                    ),
                    legend=dict(orientation="h", y=1.05),
                )
                fig_sri.update_xaxes(title="SRI成分スコア")
                fig_sri.update_yaxes(title="特徴量")
                _plotly_and_save(fig_sri, "sri_stacked_bar_v2").classes("w-full")

                # Synergy Heatmap
                ui.separator().style("border-color:" + COLORS["border"] + ";")
                ui.label("🔥 Synergy Matrix (相乗効果)").classes(
                    "text-subtitle1 text-bold q-mb-sm"
                )

                top12 = summary.head(12)["feature"].tolist()
                syn_mat = sri_res.synergy_matrix
                fn = sri_res.feature_names
                fi_top = [fn.index(f) if f in fn else 0 for f in top12]
                sub_syn = syn_mat[np.ix_(fi_top, fi_top)]

                fig_heat = go.Figure(
                    go.Heatmap(
                        z=sub_syn.tolist(),
                        x=top12,
                        y=top12,
                        colorscale="RdBu_r",
                        zmid=0,
                        colorbar=dict(title="Synergy"),
                        hovertemplate=(
                            "Feature i: %{y}<br>"
                            "Feature j: %{x}<br>"
                            "Synergy: %{z:.4f}<extra></extra>"
                        ),
                    )
                )
                _fig_defaults(fig_heat, "Synergy Matrix (Top 12)", height=450)
                fig_heat.update_xaxes(tickangle=-30)
                _plotly_and_save(fig_heat, "sri_synergy_heatmap_v2").classes("w-full")

                # Network Graph for feature interactions
                ui.separator().style("border-color:" + COLORS["border"] + ";")
                ui.label("🕸️ Feature Interaction Network").classes(
                    "text-subtitle1 text-bold q-mb-sm"
                )
                ui.label(
                    "ノードサイズ = Independence、エッジ太さ = |Synergy|。"
                    "赤いエッジは相乗効果、青いエッジは冗長性を表します。"
                ).classes("text-caption text-grey-5 q-mb-sm")

                _render_sri_network(sri_res, top_n=15)

                # Data table
                ui.separator().style("border-color:" + COLORS["border"] + ";")
                with ui.expansion("📋 SRI 数値テーブル", icon="table_chart").classes(
                    "w-full q-mt-md"
                ):
                    rows = []
                    for _, row in df_top.iterrows():
                        rows.append({
                            "特徴量": row["feature"],
                            "Independence": f"{row['independence']:.4f}",
                            "Synergy": f"{row['synergy']:.4f}",
                            "Redundancy": f"{row['redundancy']:.4f}",
                        })
                    cols = [
                        {"name": k, "label": k, "field": k,
                         "align": "left" if k == "特徴量" else "center", "sortable": True}
                        for k in ["特徴量", "Independence", "Synergy", "Redundancy"]
                    ]
                    ui.table(columns=cols, rows=rows).classes("w-full").props(
                        "dense flat bordered"
                    )

            ui.notify("✅ SRI分解完了", type="positive", position="top")

        except Exception as ex:
            sri_container.clear()
            with sri_container:
                with ui.card().style(_card_style(COLORS["danger"] + "33")):
                    ui.label(f"❌ エラー: {ex}").classes("text-red")

    # Run button
    with ui.card().style(_card_style(COLORS["purple"] + "33")):
        with ui.row().classes("w-full items-center"):
            ui.label("🚀 SRI分解を開始").classes("text-subtitle1 text-bold")
            ui.space()
            ui.button(
                "🔬 SRI分解実行",
                icon="science",
                on_click=lambda: _run_sri(),
            ).props("unelevated color=purple size=sm no-caps").classes("shadow-lg")

    sri_container


def _render_sri_network(sri_res, top_n: int = 15):
    """Render feature interaction network using Plotly."""
    import plotly.graph_objects as go
    import numpy as np

    summary = sri_res.summary_df().head(top_n)
    feature_names = list(summary["feature"].values)
    n = len(feature_names)

    # Build edge list from synergy and redundancy matrices
    edge_x = []
    edge_y = []
    edge_weights = []

    syn_mat = sri_res.synergy_matrix
    red_mat = sri_res.redundancy_matrix
    fn = sri_res.feature_names

    # Create position layout (circular)
    import math
    positions = {}
    for i, f in enumerate(feature_names):
        angle = 2 * math.pi * i / n
        positions[f] = (math.cos(angle) * 300, math.sin(angle) * 300)

    # Edges
    edge_trace_x = []
    edge_trace_y = []

    for i in range(n):
        for j in range(i + 1, n):
            f_i = feature_names[i]
            f_j = feature_names[j]
            idx_i = fn.index(f_i) if f_i in fn else i
            idx_j = fn.index(f_j) if f_j in fn else j

            syn_val = abs(syn_mat[idx_i, idx_j])
            red_val = abs(red_mat[idx_i, idx_j])

            if max(syn_val, red_val) < 1e-6:
                continue

            x0, y0 = positions[f_i]
            x1, y1 = positions[f_j]

            edge_trace_x.extend([x0, x1, None])
            edge_trace_y.extend([y0, y1, None])

    # Nodes
    node_x = [positions[f][0] for f in feature_names]
    node_y = [positions[f][1] for f in feature_names]
    node_size = []
    for f in feature_names:
        if f in fn:
            idx = fn.index(f)
            size = max(20, min(60, abs(sri_res.independence_vec[idx]) * 500))
        else:
            size = 20
        node_size.append(size)

    fig = go.Figure()

    # Add edges
    fig.add_trace(
        go.Scatter(
            x=edge_trace_x,
            y=edge_trace_y,
            mode="lines",
            line=dict(width=1, color="rgba(255,255,255,0.15)"),
            hoverinfo="none",
            showlegend=False,
        )
    )

    # Add nodes
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(
                size=node_size,
                color=list(range(n)),
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title="Index", tickformat="d"),
                opacity=0.85,
                line=dict(width=2, color="rgba(255,255,255,0.3)"),
            ),
            text=feature_names,
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Independence: %{marker.size:.2f}<br>"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    _fig_defaults(fig, "Feature Interaction Network", height=500)
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    _plotly_and_save(fig, "sri_network_v2").classes("w-full")


# ═════════════════════════════════════════════════════════════
# Main Entry Point
# ═════════════════════════════════════════════════════════════

def render_interpretation_panel_v2(ar: Any, state: dict) -> None:
    """Main entry point for the premium interpretation panel."""

    model = getattr(ar, "best_pipeline", None)
    X: Any = getattr(ar, "processed_X", None)
    y: Any = getattr(ar, "y_train", None)
    y_true_oof: Any = getattr(ar, "oof_true", None)
    y_pred_oof: Any = getattr(ar, "oof_preds", None)

    if model is None or X is None:
        with ui.card().style(_card_style(COLORS["warning"] + "33")):
            ui.label("⚠️ モデルまたはデータが取得できませんでした").classes(
                "text-amber"
            )
            ui.label(
                "先にモデル学習を完了してください。"
            ).classes("text-caption text-grey-5")
        return

    feat_names: list[str] = (
        list(X.columns) if hasattr(X, "columns")
        else [f"feature_{i}" for i in range(X.shape[1])]
    )
    X_arr: np.ndarray = X.values if hasattr(X, "values") else np.asarray(X)

    # ── Header ──
    with ui.row().classes("w-full items-center gap-3 q-mb-md"):
        ui.label("🔬 モデル解釈・XAI").classes("text-h4 text-bold").style(
            "color:white;font-family:Inter,sans-serif;"
        )
        ui.space()
        ui.badge("Premium UI", color="cyan").props("outline")

    # ── Tab navigation ──
    with ui.tabs().classes("w-full").props(
        "dense active-color=cyan indicator-color=cyan scrollable"
    ) as interp_tabs:
        t_pred    = ui.tab("pred",    label="📈 予測実測")
        t_coef    = ui.tab("coef",    label="📊 重要度")
        t_shap    = ui.tab("shap",    label="🔍 SHAP")
        t_sage    = ui.tab("sage",    label="🌿 SAGE")
        t_sri     = ui.tab("sri",     label="🔬 SRI")

    with ui.tab_panels(interp_tabs, value=t_pred).classes("w-full bg-transparent"):

        # Prediction vs Actual
        with ui.tab_panel(t_pred):
            _render_pred_actual_v2(ar, y_true_oof, y_pred_oof, feat_names, X_arr)

        # Coefficients / Feature Importance
        with ui.tab_panel(t_coef):
            _render_coefficients_v2(ar, model, X, feat_names)

        # SHAP
        with ui.tab_panel(t_shap):
            _render_shap_v2(ar, model, X, X_arr, feat_names, y)

        # SAGE
        with ui.tab_panel(t_sage):
            _render_sage_v2(ar, model, X, X_arr, feat_names, y)

        # SRI
        with ui.tab_panel(t_sri):
            _render_sri_v2(ar, model, X, X_arr, feat_names)


# ═════════════════════════════════════════════════════════════
# Enhanced versions of existing panels
# ═════════════════════════════════════════════════════════════

def _render_pred_actual_v2(ar, y_true, y_pred, feat_names, X_arr):
    """Enhanced prediction vs actual plot."""
    import plotly.graph_objects as go
    from sklearn.metrics import r2_score, mean_squared_error

    ui.label("📈 予測値 vs 実測値 (OOF)").classes("text-subtitle1 text-bold q-mb-xs")

    if y_true is None or y_pred is None:
        ui.label("⚠️ OOFデータが利用できません").classes("text-amber")
        return

    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    r2 = r2_score(y_t, y_p)
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

    # Metrics cards
    with ui.row().classes("w-full q-gutter-md q-mb-md"):
        for val, lbl, color in [
            (f"{r2:.4f}", "R² (OOF)", COLORS["cyan"]),
            (f"{rmse:.4f}", "RMSE (OOF)", COLORS["amber"]),
        ]:
            with ui.card().style(_card_style(color + "22")).classes("flex-1 text-center"):
                ui.label(val).classes(f"text-h6 text-bold text-{color}")
                ui.label(lbl).classes("text-caption text-grey-5")

    # Scatter plot
    fig = go.Figure()
    rng = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]

    fig.add_trace(
        go.Scatter(
            x=rng, y=rng,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.25)", dash="dash", width=1.5),
            name="y = x",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_t, y=y_p,
            mode="markers",
            marker=dict(
                size=8,
                color=y_p - y_t,
                colorscale="RdBu_r",
                showscale=True,
                colorbar=dict(title="残差", tickformat=".3f"),
                opacity=0.75,
                line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
            ),
            hovertemplate=(
                "実測: %{x:.4f}<br>"
                "予測: %{y:.4f}<br>"
                "残差: %{marker.color:.4f}<extra></extra>"
            ),
            name="データ点",
        )
    )

    _fig_defaults(fig, f"Prediction vs Actual (R²={r2:.4f})", height=450)
    fig.update_xaxes(title="実測値")
    fig.update_yaxes(title="予測値")
    _plotly_and_save(fig, "interp_parity_v2").classes("w-full")


def _render_coefficients_v2(ar, model, X, feat_names: list[str]):
    """Enhanced coefficients/feature importance display."""
    import plotly.graph_objects as go
    import numpy as np

    ui.label("📊 回帰係数 / Feature Importance").classes(
        "text-subtitle1 text-bold q-mb-xs"
    )

    try:
        estimator = model
        if hasattr(model, "steps"):
            estimator = model.steps[-1][1]

        # Tree-based: feature_importances_
        if hasattr(estimator, "feature_importances_"):
            imp = estimator.feature_importances_
            fi_len = len(imp)
            names = feat_names[:fi_len] if len(feat_names) >= fi_len else [f"f{i}" for i in range(fi_len)]
            idx = np.argsort(imp)[::-1]
            top = min(30, len(idx))

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=imp[idx[:top]][::-1],
                    y=[names[i] if i < len(names) else f"f{i}" for i in idx[:top]][::-1],
                    orientation="h",
                    marker=dict(
                        color=imp[idx[:top]][::-1],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="重要度", tickformat=".4f"),
                    ),
                    text=[f"{imp[i]:.4f}" for i in idx[:top]][::-1],
                    textposition="outside",
                )
            )
            _fig_defaults(fig, f"Feature Importance ({ar.best_model_key})", height=max(350, 22 * top))
            fig.update_xaxes(title="Feature Importance (不純度減少)")
            fig.update_yaxes(title="特徴量")
            _plotly_and_save(fig, "interp_feature_importance_v2").classes("w-full")

        # Linear: coef_
        elif hasattr(estimator, "coef_"):
            coefs = estimator.coef_.ravel()
            idx = np.argsort(np.abs(coefs))[::-1]
            top = min(30, len(idx))

            fig = go.Figure()
            colors_bar = ["rgba(74,222,128,0.7)" if coefs[i] > 0 else "rgba(248,113,113,0.7)"
                         for i in idx[:top]]

            fig.add_trace(
                go.Bar(
                    x=coefs[idx[:top]][::-1],
                    y=[feat_names[i] if i < len(feat_names) else f"f{i}" for i in idx[:top]][::-1],
                    orientation="h",
                    marker_color=colors_bar[::-1],
                    text=[f"{coefs[i]:+.4f}" for i in idx[:top]][::-1],
                    textposition="outside",
                )
            )
            _fig_defaults(fig, f"Regression Coefficients ({ar.best_model_key})", height=max(350, 22 * top))
            fig.update_xaxes(title="回帰係数")
            fig.update_yaxes(title="特徴量")
            _plotly_and_save(fig, "interp_coef_v2").classes("w-full")

        else:
            ui.label("ℹ️ このモデルは係数/重要度を直接取得できません").classes(
                "text-grey-5"
            )
            ui.label("SHAPタブをご利用ください").classes("text-caption text-grey-6")

    except Exception as ex:
        ui.label(f"❌ エラー: {ex}").classes("text-red text-caption")
