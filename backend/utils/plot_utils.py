"""
backend/utils/plot_utils.py

プロット用ユーティリティ：軸スケール自動調整・外れ値対策・パリティプロット
- 既存コードに1行追加するだけで利用可能（非破壊）
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    import plotly

import numpy as np


# ─────────────────────────────────────────────
# 軸スケール・外れ値
# ─────────────────────────────────────────────

def auto_clip_scale(
    values: Union[np.ndarray, list],
    percentiles: Tuple[float, float] = (1.0, 99.0),
    fallback_range: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """
    値のパーセンタイルから適切なスケール範囲を算出。

    Args:
        values:          数値の配列/リスト
        percentiles:     (下限%, 上限%) — デフォルト 1〜99%
        fallback_range:  全値が NaN 等の場合の代替範囲

    Returns:
        (vmin, vmax): 推奨スケール範囲
    """
    arr = np.asarray(values, dtype=float).ravel()
    valid = arr[np.isfinite(arr)]

    if len(valid) == 0:
        return fallback_range or (0.0, 1.0)

    vmin = float(np.percentile(valid, percentiles[0]))
    vmax = float(np.percentile(valid, percentiles[1]))

    # 範囲が極端に狭い場合は緩和
    if vmax - vmin < 1e-9:
        margin = abs(vmin) * 0.1 + 0.01
        vmin, vmax = vmin - margin, vmax + margin

    return vmin, vmax


def sanitize_heatmap_data(
    z: np.ndarray,
    clip_percentiles: Tuple[float, float] = (1.0, 99.0),
) -> dict:
    """
    ヒートマップデータの自動整形（外れ値対策＋スケール情報付与）

    Returns::

        {
            "z_clipped":     表示用データ（外れ値をクリッピング済み）,
            "z_original":    元データ（ツールチップ用）,
            "vmin", "vmax":  推奨スケール,
            "outlier_count": 除外された外れ値数,
        }
    """
    z = np.asarray(z, dtype=float)
    valid = z[np.isfinite(z)]

    if len(valid) == 0:
        return {
            "z_clipped": z.tolist(),
            "z_original": z.tolist(),
            "vmin": 0.0,
            "vmax": 1.0,
            "outlier_count": 0,
        }

    vmin, vmax = auto_clip_scale(valid, clip_percentiles)
    z_clipped = np.clip(z, vmin, vmax)
    outlier_count = int(np.sum((z < vmin) | (z > vmax) & np.isfinite(z)))

    return {
        "z_clipped": z_clipped.tolist(),
        "z_original": z.tolist(),
        "vmin": vmin,
        "vmax": vmax,
        "outlier_count": outlier_count,
    }


# ─────────────────────────────────────────────
# パリティプロット（予測 vs 実測）— 正方形固定
# ─────────────────────────────────────────────

def create_parity_plot(
    y_true,
    y_pred,
    title: str = "予測 vs 実測",
    size: int = 520,
    point_color: Optional[str] = None,
) -> "plotly.graph_objects.Figure":  # type: ignore[name-defined]
    """
    OOF 予測実測プロット（正方形・大きめフォント）。

    コメント3 対応:
    - scaleanchor="x" で正方形アスペクト比を強制
    - タイトル 17px / 軸ラベル 14px / 目盛り 12px

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    residuals = y_t - y_p

    rng_min = float(min(np.nanmin(y_t), np.nanmin(y_p)))
    rng_max = float(max(np.nanmax(y_t), np.nanmax(y_p)))
    margin = (rng_max - rng_min) * 0.05
    axis_range = [rng_min - margin, rng_max + margin]

    marker_opts = dict(
        size=7,
        opacity=0.72,
        line=dict(width=0.5, color="white"),
    )
    if point_color is None:
        marker_opts.update(
            color=residuals,
            colorscale=[
                [0.0, "#4575B4"],
                [0.5, "#F7F7F7"],
                [1.0, "#D73027"],
            ],
            cmid=0.0,
            showscale=True,
            colorbar=dict(title="残差", thickness=14, len=0.5, tickfont=dict(size=10)),
        )
    else:
        marker_opts["color"] = point_color

    fig = go.Figure()

    # 対角線（y = x）
    fig.add_trace(go.Scatter(
        x=axis_range, y=axis_range,
        mode="lines",
        name="y = x",
        line=dict(color="rgba(255,255,255,0.25)", dash="dash", width=1.5),
        showlegend=True,
    ))

    # データ点
    fig.add_trace(go.Scatter(
        x=y_t, y=y_p,
        mode="markers",
        name="データ点",
        marker=marker_opts,
        customdata=residuals,
        hovertemplate=(
            "実測: %{x:.4g}<br>予測: %{y:.4g}"
            "<br>残差: %{customdata:.4g}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text=f"{title}（n={len(y_t)}）", font=dict(size=17)),
        xaxis=dict(
            title=dict(text="実測値", font=dict(size=14)),
            tickfont=dict(size=12),
            range=axis_range,
            gridcolor="rgba(255,255,255,0.10)",
            zerolinecolor="rgba(255,255,255,0.10)",
        ),
        yaxis=dict(
            title=dict(text="予測値", font=dict(size=14)),
            tickfont=dict(size=12),
            range=axis_range,
            scaleanchor="x",   # ★ 正方形アスペクト比
            scaleratio=1,
            gridcolor="rgba(255,255,255,0.10)",
            zerolinecolor="rgba(255,255,255,0.10)",
        ),
        width=size,
        height=size,          # ★ 正方形
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.12)",
        legend=dict(
            font=dict(size=11),
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            bgcolor="rgba(0,0,0,0.3)",
        ),
        margin=dict(l=65, r=35, t=65, b=65),
        hovermode="closest",
    )

    return fig
