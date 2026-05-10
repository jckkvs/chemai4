# -*- coding: utf-8 -*-
"""
frontend_nicegui/components/eda_dim_reduction.py

次元削減（PCA / t-SNE）＆ 特徴量重要度パネル — NiceGUI版
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from nicegui import ui, run

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


def render_dim_reduction_panel(state: dict) -> None:
    """
    次元削減＆重要度タブの描画関数。

    ボタン押下で非同期計算を開始し、完了後にコンテナ内に結果を表示する。
    @ui.refreshable を使わず、コンテナの clear/再描画パターンで安定動作させる。
    """
    df = state.get("df")
    if df is None:
        with ui.card().classes("full-width q-pa-md").style(
            "border: 1px dashed rgba(255,255,255,0.2); border-radius: 10px;"
        ):
            ui.icon("info", color="grey").classes("text-h5")
            ui.label("⚠️ データが読み込まれていません。先にCSV/Excelをアップロードしてください。").classes(
                "text-caption text-grey"
            )
        return

    # 結果表示コンテナ
    result_container = ui.column().classes("full-width")

    # すでに計算済みの場合は即表示
    cached = state.get("dim_red_results")
    if cached is not None:
        _render_results(result_container, cached)
        return

    # 自動計算: ボタンなしで初回アクセス時に非同期計算を開始
    _trigger_computation(state, df, result_container)


def _trigger_computation(state: dict, df: pd.DataFrame, container) -> None:
    """非同期で次元削減を計算し、完了後にコンテナに描画する。"""
    # スピナー表示
    with container:
        with ui.column().classes("full-width items-center q-pa-lg q-gutter-sm"):
            ui.spinner("dots", size="lg", color="cyan")
            ui.label("次元削減を計算中...（データ規模により数十秒かかる場合があります）").classes(
                "text-caption text-grey"
            )

    async def _compute():
        try:
            from backend.data.dim_reduction import compute_dim_reduction_and_importance
            res = await run.io_bound(compute_dim_reduction_and_importance, df)
            state["dim_red_results"] = res
            _render_results(container, res)
        except Exception as e:
            logger.error(f"次元削減計算失敗: {e}", exc_info=True)
            container.clear()
            with container:
                ui.label(f"❌ 計算エラー: {e}").classes("text-negative q-pa-sm")

    ui.timer(0.1, _compute, once=True)


def _render_results(container, results: dict) -> None:
    """計算結果をコンテナ内に描画する。"""
    import plotly.express as px
    import plotly.graph_objects as go

    container.clear()

    if results.get("status") == "skip":
        with container:
            with ui.card().classes("full-width q-pa-md").style(
                "border: 1px solid rgba(255,193,7,0.3); border-radius: 8px;"
            ):
                ui.label(f"ℹ️ {results.get('message', 'スキップ')}").classes("text-amber")
        return

    if results.get("status") == "error":
        with container:
            with ui.card().classes("full-width q-pa-md").style(
                "border: 1px solid rgba(244,67,54,0.3); border-radius: 8px;"
            ):
                ui.label(f"❌ {results.get('message', 'エラー')}").classes("text-red")
        return

    # ── 成功時: 並列レイアウトでPCA + t-SNE ──
    with container:
        n_feat = results.get("n_features", "?")
        n_samp = results.get("n_samples", "?")
        ui.badge(f"特徴量: {n_feat} / サンプル: {n_samp}", color="teal").props("dense").classes("q-mb-sm")

        # ── PCA / t-SNE 散布図 (横並び) ──
        with ui.row().classes("full-width q-gutter-md q-mb-md"):
            # PCA 散布図
            with ui.column().classes("col"):
                ev = results["explained_var"]
                pca_df = results["pca_coords"]
                fig_pca = px.scatter(
                    pca_df, x="PC1", y="PC2",
                    color_discrete_sequence=["#00d4ff"],
                    title=f"PCA  (PC1: {ev[0]:.1%}, PC2: {ev[1]:.1%})",
                )
                fig_pca.update_traces(marker=dict(size=5, opacity=0.7))
                _dark_fig(fig_pca, 380,
                          xaxis=dict(
                              title=f"PC1 ({ev[0]:.1%})",
                              gridcolor="rgba(255,255,255,0.08)",
                              scaleanchor="y", scaleratio=1,
                          ),
                          yaxis=dict(
                              title=f"PC2 ({ev[1]:.1%})",
                              gridcolor="rgba(255,255,255,0.08)",
                          ))
                ui.plotly(fig_pca).classes("full-width")

            # t-SNE 散布図
            with ui.column().classes("col"):
                tsne_df = results["tsne_coords"]
                fig_tsne = px.scatter(
                    tsne_df, x="t-SNE1", y="t-SNE2",
                    color_discrete_sequence=["#a78bfa"],
                    title="t-SNE 非線形埋め込み",
                )
                fig_tsne.update_traces(marker=dict(size=5, opacity=0.7))
                _dark_fig(fig_tsne, 380,
                          xaxis=dict(
                              title="t-SNE1",
                              gridcolor="rgba(255,255,255,0.08)",
                              scaleanchor="y", scaleratio=1,
                          ),
                          yaxis=dict(
                              title="t-SNE2",
                              gridcolor="rgba(255,255,255,0.08)",
                          ))
                ui.plotly(fig_tsne).classes("full-width")

        # ── 特徴量重要度 (横並び) ──
        with ui.row().classes("full-width q-gutter-md"):
            # PCA ローディング（PC1 寄与度 TOP15）
            with ui.column().classes("col"):
                ui.label("📐 PC1 への寄与度 TOP15").classes("text-subtitle2 text-bold q-mb-xs")
                pca_imp = results["pca_importance"]
                top_pca = pca_imp.sort_values("PC1", ascending=False).head(15)
                fig_imp_pca = go.Figure(go.Bar(
                    x=top_pca["PC1"].values,
                    y=top_pca.index.tolist(),
                    orientation="h",
                    marker_color="#00d4ff",
                ))
                fig_imp_pca.update_layout(yaxis=dict(autorange="reversed"))
                _dark_fig(fig_imp_pca, max(280, len(top_pca) * 22),
                          xaxis_title="寄与度 (|loading|)")
                ui.plotly(fig_imp_pca).classes("full-width")

            # t-SNE 重要度（Spearman相関 TOP15）
            with ui.column().classes("col"):
                ui.label("🌀 t-SNE1 との相関 TOP15").classes("text-subtitle2 text-bold q-mb-xs")
                tsne_imp = results["tsne_importance"]
                top_tsne = tsne_imp.sort_values("t-SNE1", ascending=False).head(15)
                fig_imp_tsne = go.Figure(go.Bar(
                    x=top_tsne["t-SNE1"].values,
                    y=top_tsne.index.tolist(),
                    orientation="h",
                    marker_color="#a78bfa",
                ))
                fig_imp_tsne.update_layout(yaxis=dict(autorange="reversed"))
                _dark_fig(fig_imp_tsne, max(280, len(top_tsne) * 22),
                          xaxis_title="|Spearman ρ|")
                ui.plotly(fig_imp_tsne).classes("full-width")
