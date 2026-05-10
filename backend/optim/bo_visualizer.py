"""ベイズ最適化の可視化モジュール.

Implements: F-D01〜D05
    PCA 2D/3D散布図 + biplot + 累積寄与率
    パレートフロント可視化
    獲得関数サーフェス
    収束曲線
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA


def plot_pca_2d(
    X_existing: np.ndarray | pd.DataFrame,
    X_candidates: np.ndarray | pd.DataFrame | None = None,
    feature_names: list[str] | None = None,
    y_existing: np.ndarray | None = None,
    top_n_arrows: int = 6,
) -> tuple[go.Figure, dict[str, Any]]:
    """PCA 2D散布図 + biplot矢印 + 累積寄与率.

    Args:
        X_existing: 既存実験点 (n, d)
        X_candidates: BO候補点 (m, d)、Noneなら非表示
        feature_names: 変数名
        y_existing: 既存点の目的変数値（色分け用）
        top_n_arrows: biplotに表示する上位矢印数

    Returns:
        (plotlyフィギュア, { "explained_variance": [...], "cumulative": [...], "components": ndarray })
    """
    X_ex = np.asarray(X_existing, dtype=np.float64)
    if feature_names is None:
        if isinstance(X_existing, pd.DataFrame):
            feature_names = list(X_existing.columns)
        else:
            feature_names = [f"x{i}" for i in range(X_ex.shape[1])]

    # PCA
    pca = PCA(n_components=min(X_ex.shape[1], 2))
    X_all = X_ex.copy()
    if X_candidates is not None:
        X_cand = np.asarray(X_candidates, dtype=np.float64)
        X_all = np.vstack([X_all, X_cand])

    pca.fit(X_all)
    X_ex_pca = pca.transform(X_ex)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # Figure: 左にPCA散布図、右に累積寄与率
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.75, 0.25],
        subplot_titles=[
            f"PCA 2D (累積寄与率: {cumulative[-1]:.1%})",
            "累積寄与率",
        ],
    )

    # 既存点
    if y_existing is not None:
        fig.add_trace(
            go.Scatter(
                x=X_ex_pca[:, 0], y=X_ex_pca[:, 1] if X_ex_pca.shape[1] > 1 else np.zeros(len(X_ex_pca)),
                mode="markers",
                marker=dict(
                    size=8, color=y_existing, colorscale="Viridis",
                    showscale=True, colorbar=dict(title="y", x=0.68),
                ),
                name="既存データ",
                text=[f"y={v:.3f}" for v in y_existing],
            ),
            row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=X_ex_pca[:, 0], y=X_ex_pca[:, 1] if X_ex_pca.shape[1] > 1 else np.zeros(len(X_ex_pca)),
                mode="markers",
                marker=dict(size=8, color="rgba(0,180,255,0.7)"),
                name="既存データ",
            ),
            row=1, col=1,
        )

    # 候補点
    if X_candidates is not None:
        X_cand_pca = pca.transform(X_cand)
        fig.add_trace(
            go.Scatter(
                x=X_cand_pca[:, 0], y=X_cand_pca[:, 1] if X_cand_pca.shape[1] > 1 else np.zeros(len(X_cand_pca)),
                mode="markers",
                marker=dict(
                    size=12, color="rgba(255,60,60,0.8)",
                    symbol="star", line=dict(width=1, color="white"),
                ),
                name="BO候補",
            ),
            row=1, col=1,
        )

    # Biplot矢印
    components = pca.components_
    n_arrows = min(top_n_arrows, len(feature_names))
    # 矢印の長さでソート
    arrow_lengths = np.sqrt((components**2).sum(axis=0))
    top_features = np.argsort(arrow_lengths)[::-1][:n_arrows]

    scale = np.max(np.abs(X_ex_pca)) * 0.8
    for idx in top_features:
        dx = components[0, idx] * scale
        dy = components[1, idx] * scale if components.shape[0] > 1 else 0
        fig.add_annotation(
            x=dx, y=dy, ax=0, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1.5, arrowwidth=2,
            arrowcolor="rgba(255,200,0,0.8)",
            text=feature_names[idx],
            font=dict(size=10, color="rgba(255,200,0,0.9)"),
            row=1, col=1,
        )

    # 累積寄与率バー
    n_components_full = min(X_ex.shape[1], 10)
    pca_full = PCA(n_components=n_components_full)
    pca_full.fit(X_all)
    ev = pca_full.explained_variance_ratio_
    cum = np.cumsum(ev)

    fig.add_trace(
        go.Bar(
            x=[f"PC{i+1}" for i in range(len(ev))],
            y=ev,
            marker_color="rgba(0,180,255,0.6)",
            name="寄与率",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[f"PC{i+1}" for i in range(len(cum))],
            y=cum,
            mode="lines+markers",
            line=dict(color="rgba(255,60,60,0.8)", width=2),
            marker=dict(size=6),
            name="累積",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        showlegend=True,
        legend=dict(orientation="h", y=-0.15),
    )
    fig.update_xaxes(title_text="PC1", row=1, col=1)
    fig.update_yaxes(title_text="PC2", row=1, col=1)
    fig.update_yaxes(title_text="寄与率", row=1, col=2)

    info = {
        "explained_variance": explained.tolist(),
        "cumulative": cumulative.tolist(),
        "components": components,
    }
    return fig, info


def plot_pca_3d(
    X_existing: np.ndarray | pd.DataFrame,
    X_candidates: np.ndarray | pd.DataFrame | None = None,
    feature_names: list[str] | None = None,
    y_existing: np.ndarray | None = None,
    top_n_arrows: int = 6,
) -> tuple[go.Figure, dict[str, Any]]:
    """PCA 3Dインタラクティブ散布図 + biplot.

    Returns:
        (plotlyフィギュア, PCA情報dict)
    """
    X_ex = np.asarray(X_existing, dtype=np.float64)
    if feature_names is None:
        if isinstance(X_existing, pd.DataFrame):
            feature_names = list(X_existing.columns)
        else:
            feature_names = [f"x{i}" for i in range(X_ex.shape[1])]

    n_comp = min(X_ex.shape[1], 3)
    pca = PCA(n_components=n_comp)
    X_all = X_ex.copy()
    if X_candidates is not None:
        X_cand = np.asarray(X_candidates, dtype=np.float64)
        X_all = np.vstack([X_all, X_cand])

    pca.fit(X_all)
    X_ex_pca = pca.transform(X_ex)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    fig = go.Figure()

    # 既存点
    scatter_kwargs: dict[str, Any] = dict(
        x=X_ex_pca[:, 0],
        y=X_ex_pca[:, 1] if n_comp > 1 else np.zeros(len(X_ex_pca)),
        z=X_ex_pca[:, 2] if n_comp > 2 else np.zeros(len(X_ex_pca)),
        mode="markers",
        name="既存データ",
    )
    if y_existing is not None:
        scatter_kwargs["marker"] = dict(
            size=5, color=y_existing, colorscale="Viridis",
            showscale=True, colorbar=dict(title="y"),
        )
        scatter_kwargs["text"] = [f"y={v:.3f}" for v in y_existing]
    else:
        scatter_kwargs["marker"] = dict(size=5, color="rgba(0,180,255,0.7)")

    fig.add_trace(go.Scatter3d(**scatter_kwargs))

    # 候補点
    if X_candidates is not None:
        X_cand_pca = pca.transform(X_cand)
        fig.add_trace(go.Scatter3d(
            x=X_cand_pca[:, 0],
            y=X_cand_pca[:, 1] if n_comp > 1 else np.zeros(len(X_cand_pca)),
            z=X_cand_pca[:, 2] if n_comp > 2 else np.zeros(len(X_cand_pca)),
            mode="markers",
            marker=dict(
                size=8, color="rgba(255,60,60,0.8)",
                symbol="diamond", line=dict(width=1, color="white"),
            ),
            name="BO候補",
        ))

    # Biplot矢印（3Dでは線とテキストで表現）
    components = pca.components_
    n_arrows = min(top_n_arrows, len(feature_names))
    arrow_lengths = np.sqrt((components**2).sum(axis=0))
    top_features = np.argsort(arrow_lengths)[::-1][:n_arrows]

    scale = np.max(np.abs(X_ex_pca)) * 0.5
    for idx in top_features:
        dx = components[0, idx] * scale
        dy = components[1, idx] * scale if n_comp > 1 else 0
        dz = components[2, idx] * scale if n_comp > 2 else 0
        fig.add_trace(go.Scatter3d(
            x=[0, dx], y=[0, dy], z=[0, dz],
            mode="lines+text",
            line=dict(color="rgba(255,200,0,0.8)", width=3),
            text=["", feature_names[idx]],
            textposition="top center",
            textfont=dict(size=10, color="rgba(255,200,0,0.9)"),
            showlegend=False,
        ))

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis_title=f"PC1 ({explained[0]:.1%})",
            yaxis_title=f"PC2 ({explained[1]:.1%})" if n_comp > 1 else "PC2",
            zaxis_title=f"PC3 ({explained[2]:.1%})" if n_comp > 2 else "PC3",
        ),
        height=600,
        title=f"PCA 3D (累積寄与率: {cumulative[min(2, len(cumulative)-1)]:.1%})",
    )

    info = {
        "explained_variance": explained.tolist(),
        "cumulative": cumulative.tolist(),
        "components": components,
    }
    return fig, info


def plot_pareto_front(
    Y_existing: np.ndarray | pd.DataFrame,
    Y_candidates: np.ndarray | pd.DataFrame | None = None,
    objective_names: list[str] | None = None,
    directions: list[str] | None = None,
) -> go.Figure:
    """2目的のパレートフロント可視化.

    Args:
        Y_existing: 既存データの目的変数値 (n, 2)
        Y_candidates: BO候補の予測目的変数値 (m, 2)
        objective_names: 目的変数名
        directions: ["min", "max"] etc
    """
    Y_ex = np.asarray(Y_existing, dtype=np.float64)
    n_obj = Y_ex.shape[1]
    if objective_names is None:
        objective_names = [f"Objective {i+1}" for i in range(n_obj)]
    if directions is None:
        directions = ["min"] * n_obj

    fig = go.Figure()

    # 既存点
    fig.add_trace(go.Scatter(
        x=Y_ex[:, 0], y=Y_ex[:, 1],
        mode="markers",
        marker=dict(size=8, color="rgba(0,180,255,0.7)"),
        name="既存データ",
    ))

    # パレートフロントの計算
    pareto_mask = _is_pareto_efficient(Y_ex, directions)
    Y_pareto = Y_ex[pareto_mask]
    # ソートしてラインで結ぶ
    sort_idx = np.argsort(Y_pareto[:, 0])
    Y_pareto_sorted = Y_pareto[sort_idx]

    fig.add_trace(go.Scatter(
        x=Y_pareto_sorted[:, 0], y=Y_pareto_sorted[:, 1],
        mode="lines+markers",
        line=dict(color="rgba(0,255,150,0.8)", width=2),
        marker=dict(size=10, color="rgba(0,255,150,0.8)"),
        name="パレートフロント",
    ))

    # 候補点
    if Y_candidates is not None:
        Y_cand = np.asarray(Y_candidates, dtype=np.float64)
        fig.add_trace(go.Scatter(
            x=Y_cand[:, 0], y=Y_cand[:, 1],
            mode="markers",
            marker=dict(
                size=12, color="rgba(255,60,60,0.8)",
                symbol="star", line=dict(width=1, color="white"),
            ),
            name="BO候補",
        ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title=objective_names[0],
        yaxis_title=objective_names[1],
        title="パレートフロント",
        height=500,
    )
    return fig


def _is_pareto_efficient(Y: np.ndarray, directions: list[str]) -> np.ndarray:
    """パレート効率的な点のマスクを返す."""
    n = len(Y)
    Y_adj = Y.copy()
    for j, d in enumerate(directions):
        if d == "max":
            Y_adj[:, j] = -Y_adj[:, j]

    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        # i を支配する点があるか
        for j in range(n):
            if i == j or not is_efficient[j]:
                continue
            if np.all(Y_adj[j] <= Y_adj[i]) and np.any(Y_adj[j] < Y_adj[i]):
                is_efficient[i] = False
                break
    return is_efficient


def plot_convergence(
    y_history: list[float],
    objective: str = "minimize",
) -> go.Figure:
    """反復ごとの最良値推移を可視化.

    Args:
        y_history: 各反復での最良目的変数値
        objective: "minimize" / "maximize"
    """
    fig = go.Figure()

    n = len(y_history)
    if objective == "minimize":
        best_so_far = np.minimum.accumulate(y_history)
    else:
        best_so_far = np.maximum.accumulate(y_history)

    fig.add_trace(go.Scatter(
        x=list(range(1, n + 1)),
        y=y_history,
        mode="markers",
        marker=dict(size=6, color="rgba(0,180,255,0.5)"),
        name="観測値",
    ))
    fig.add_trace(go.Scatter(
        x=list(range(1, n + 1)),
        y=best_so_far.tolist(),
        mode="lines",
        line=dict(color="rgba(255,60,60,0.8)", width=2),
        name="最良値",
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="実験回数",
        yaxis_title="目的変数値",
        title="収束曲線",
        height=400,
    )
    return fig
