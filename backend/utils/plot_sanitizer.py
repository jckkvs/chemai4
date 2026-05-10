"""
backend/utils/plot_sanitizer.py

可視化の軸・色・アスペクト比・ツールチップを自動整形する Drop-in ユーティリティ。
既存の plot 生成関数の末尾に PlotSanitizer().apply(fig, chart_type="...") を
1行追加するだけで動作する非破壊設計。

既存の ChartTheme / ChartSanitizer との関係:
  - ChartTheme       : カラースケール定義（定数集）
  - chart_sanitizer  : NiceGUI ui.plotly 呼び出し直前のラッパー
  - PlotSanitizer    : backend 側の Figure オブジェクトに対する統合整形器
                       (WebGL切替・参照線・精密ツールチップを追加)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go


# ─────────────────────────────────────────────
# 設定クラス
# ─────────────────────────────────────────────

@dataclass
class SanitizeConfig:
    """PlotSanitizer の動作設定"""

    # 軸スケール: 外れ値クリッピングパーセンタイル
    clip_percentiles: Tuple[float, float] = (1.0, 99.0)

    # 正方形強制対象のチャートタイプ
    force_square_types: Tuple[str, ...] = ("parity", "pca", "tsne", "scatter_square")

    # カラースケール（Coolwarmベース・彩度控えめ）
    palette_sequential: list = field(default_factory=lambda: [
        [0.0,  "#E8F0F8"],
        [0.5,  "#94B8D8"],
        [1.0,  "#4A7AB5"],
    ])
    palette_diverging: list = field(default_factory=lambda: [
        [0.0,  "#5B8DBE"],
        [0.25, "#91BFD8"],
        [0.5,  "#F5F5F5"],
        [0.75, "#D89888"],
        [1.0,  "#C88070"],
    ])

    # WebGL 自動切替閾値（点数がこれを超えると scattergl に変換）
    gl_threshold: int = 1_000

    # フォント設定
    font_family: str = "Inter, Meiryo, sans-serif"
    title_size: int = 16
    axis_label_size: int = 13
    tick_size: int = 11

    # マージン
    margin: dict = field(default_factory=lambda: dict(l=60, r=40, t=55, b=55))


# ─────────────────────────────────────────────
# メインクラス
# ─────────────────────────────────────────────

class PlotSanitizer:
    """
    Plotly Figure の可視化品質を自動整形するユーティリティ。

    使い方::

        from backend.utils.plot_sanitizer import PlotSanitizer

        fig = go.Figure(...)
        fig = PlotSanitizer().apply(fig, chart_type="parity", metric_name="R²")

    chart_type 一覧:
        "parity"         予測 vs 実測（正方形 + y=x 参照線）
        "pca" / "tsne"   次元削減（正方形）
        "heatmap"        ヒートマップ（発散カラー + 外れ値クリップ）
        "correlation"    相関行列（発散カラー + zmid=0）
        "residual"       残差プロット（y=0 参照線）
        "bar"            棒グラフ
        "generic"        汎用（軸整形 + レイアウト統一のみ）
    """

    def __init__(self, config: Optional[SanitizeConfig] = None):
        self.cfg = config or SanitizeConfig()

    # ─────────────────────────────────────────────
    # 公開エントリーポイント
    # ─────────────────────────────────────────────

    def apply(
        self,
        fig: go.Figure,
        chart_type: str = "generic",
        metric_name: str = "",
        higher_better: bool = True,
    ) -> go.Figure:
        """
        全整形ステップを一括適用する。

        Args:
            fig:           plotly.graph_objects.Figure
            chart_type:    上記 chart_type 一覧参照
            metric_name:   カラーバーのタイトルに表示する指標名
            higher_better: True=高いほど良い（色方向）、False=逆
        """
        fig = self._robust_scaling(fig, chart_type)
        fig = self._enforce_square(fig, chart_type)
        fig = self._safe_colors(fig, chart_type, metric_name, higher_better)
        fig = self._reference_lines(fig, chart_type)
        fig = self._smart_tooltips(fig)
        fig = self._performance_optimize(fig)
        fig = self._layout_polish(fig)
        return fig

    # ─────────────────────────────────────────────
    # 1. 外れ値に強い軸スケーリング
    # ─────────────────────────────────────────────

    def _robust_scaling(self, fig: go.Figure, chart_type: str) -> go.Figure:
        """1~99%パーセンタイルでクリッピングして軸範囲を設定"""
        low, high = self.cfg.clip_percentiles

        # heatmap は z 軸なので別処理
        if chart_type in ("heatmap", "correlation"):
            for t in fig.data:
                if t.type == "heatmap" and t.z is not None:
                    z = np.asarray(t.z, dtype=float).ravel()
                    valid = z[np.isfinite(z)]
                    if len(valid) >= 5:
                        vmin, vmax = np.percentile(valid, [low, high])
                        if t.zmin is None:
                            t.zmin = float(vmin)
                        if t.zmax is None:
                            t.zmax = float(vmax)
            return fig

        for axis_key in ("xaxis", "yaxis"):
            ax = getattr(fig.layout, axis_key, None)
            if ax is None:
                continue
            if ax.range is not None:  # 既に明示指定済みなら触らない
                continue

            vals = []
            for t in fig.data:
                arr = getattr(t, "x" if axis_key == "xaxis" else "y", None)
                if arr is not None and len(arr) > 0:
                    vals.append(np.asarray(arr, dtype=float).ravel())
            if not vals:
                continue

            flat = np.concatenate(vals)
            valid = flat[np.isfinite(flat)]
            if len(valid) < 5:
                continue

            vmin, vmax = np.percentile(valid, [low, high])
            span = vmax - vmin
            if span < 1e-9:
                span = abs(vmin) * 0.1 + 0.01
            margin = span * 0.05
            getattr(fig.layout, axis_key).update(
                range=[float(vmin - margin), float(vmax + margin)]
            )

        return fig

    # ─────────────────────────────────────────────
    # 2. 正方形アスペクト比の強制
    # ─────────────────────────────────────────────

    def _enforce_square(self, fig: go.Figure, chart_type: str) -> go.Figure:
        if chart_type not in self.cfg.force_square_types:
            return fig

        w = fig.layout.width or 520
        h = fig.layout.height or 520
        size = min(w, h)

        fig.update_layout(
            width=size,
            height=size,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            margin=self.cfg.margin,
        )
        return fig

    # ─────────────────────────────────────────────
    # 3. 安全なカラーパレット
    # ─────────────────────────────────────────────

    def _safe_colors(
        self,
        fig: go.Figure,
        chart_type: str,
        metric: str,
        higher_better: bool,
    ) -> go.Figure:
        diverging = chart_type in ("heatmap", "correlation", "residual")
        scale = self.cfg.palette_diverging if diverging else self.cfg.palette_sequential

        for t in fig.data:
            if t.type == "heatmap":
                t.update(
                    colorscale=scale,
                    zmid=0 if diverging else None,
                    colorbar=dict(
                        title=dict(
                            text=metric or "",
                            font=dict(size=self.cfg.axis_label_size),
                        ),
                        thickness=18,
                        len=0.5,
                        tickfont=dict(size=self.cfg.tick_size),
                    ),
                )
        return fig

    # ─────────────────────────────────────────────
    # 4. 参照線の自動追加
    # ─────────────────────────────────────────────

    def _reference_lines(self, fig: go.Figure, chart_type: str) -> go.Figure:
        if chart_type == "parity":
            # y = x 完全一致線
            x_range = list(getattr(fig.layout.xaxis, "range", None) or [0, 1])
            fig.add_trace(go.Scatter(
                x=x_range, y=x_range,
                mode="lines",
                name="y = x（完全一致）",
                line=dict(color="rgba(239,68,68,0.6)", width=1.5, dash="dot"),
                hoverinfo="skip",
                showlegend=True,
            ))

        elif chart_type == "residual":
            # y = 0 基準線
            fig.add_hline(
                y=0,
                line_dash="dot",
                line_color="rgba(239,68,68,0.55)",
                line_width=1.5,
                opacity=0.8,
            )

        return fig

    # ─────────────────────────────────────────────
    # 5. ツールチップ精密化
    # ─────────────────────────────────────────────

    def _smart_tooltips(self, fig: go.Figure) -> go.Figure:
        for t in fig.data:
            if t.hovertemplate:  # 既存テンプレを尊重
                continue
            if t.type == "heatmap":
                t.hovertemplate = "値: %{z:.4f}<extra></extra>"
            elif t.type in ("scatter", "scattergl"):
                t.hovertemplate = "X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>"
            elif t.type == "bar":
                t.hovertemplate = "%{x}: %{y:.4f}<extra></extra>"
        return fig

    # ─────────────────────────────────────────────
    # 6. 大規模データ対応（WebGL 自動切替）
    # ─────────────────────────────────────────────

    def _performance_optimize(self, fig: go.Figure) -> go.Figure:
        new_data = []
        changed = False
        for t in fig.data:
            if t.type == "scatter":
                n = len(t.x or [])
                if n > self.cfg.gl_threshold:
                    new_data.append(go.Scattergl(
                        x=t.x, y=t.y, mode=t.mode,
                        marker=t.marker,
                        name=t.name,
                        hovertemplate=t.hovertemplate,
                        customdata=t.customdata,
                        showlegend=t.showlegend,
                    ))
                    changed = True
                    continue
            new_data.append(t)

        if changed:
            fig.data = []
            for td in new_data:
                fig.add_trace(td)

        return fig

    # ─────────────────────────────────────────────
    # 7. フォント・余白・ダークモード統一
    # ─────────────────────────────────────────────

    def _layout_polish(self, fig: go.Figure) -> go.Figure:
        fig.update_layout(
            font=dict(
                family=self.cfg.font_family,
                size=12,
                color="#E5E7EB",
            ),
            title=dict(font=dict(size=self.cfg.title_size, family=self.cfg.font_family)),
            xaxis=dict(
                title=dict(font=dict(size=self.cfg.axis_label_size)),
                tickfont=dict(size=self.cfg.tick_size),
                gridcolor="rgba(255,255,255,0.08)",
                zerolinecolor="rgba(255,255,255,0.15)",
            ),
            yaxis=dict(
                title=dict(font=dict(size=self.cfg.axis_label_size)),
                tickfont=dict(size=self.cfg.tick_size),
                gridcolor="rgba(255,255,255,0.08)",
                zerolinecolor="rgba(255,255,255,0.15)",
            ),
            legend=dict(
                font=dict(size=11),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(255,255,255,0.1)",
                borderwidth=1,
            ),
            hovermode="closest",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=self.cfg.margin,
        )
        return fig


# ─────────────────────────────────────────────
# モジュールレベルのデフォルトインスタンス
# ─────────────────────────────────────────────

_default_sanitizer = PlotSanitizer()


def sanitize(
    fig: go.Figure,
    chart_type: str = "generic",
    metric_name: str = "",
    higher_better: bool = True,
    config: Optional[SanitizeConfig] = None,
) -> go.Figure:
    """
    PlotSanitizer のモジュールレベル便利関数。

    使い方（最短形式）::

        from backend.utils.plot_sanitizer import sanitize
        fig = sanitize(fig, chart_type="parity")
    """
    sanitizer = PlotSanitizer(config) if config else _default_sanitizer
    return sanitizer.apply(fig, chart_type=chart_type, metric_name=metric_name, higher_better=higher_better)
