"""
backend/utils/chart_theme.py

chemai2 専用カラーテーマ（Coolwarmベース、彩度抑えめ）
- ColorBrewer 準拠・色覚多様性配慮
- 既存コードに apply_to_plotly(fig) 1行追加で利用可能
"""
from __future__ import annotations

from typing import Optional


class ChartTheme:
    """
    chemai2 専用カラーテーマ
    - Coolwarmベースだが彩度を抑えた落ち着いた配色
    - 色覚多様性に配慮（ColorBrewer 準拠）
    - 長時間の分析作業でも目が疲れない明度設計
    """

    # ── ヒートマップ用（発散型：正負の値） ────────────────
    HEATMAP_DIVERGING = [
        [0.0,  "#4575B4"],   # 落ち着いた青
        [0.25, "#91BFD8"],   # ミディアムブルー
        [0.5,  "#F7F7F7"],   # ほぼ白（中央値）
        [0.75, "#FC8D62"],   # ソフトオレンジ
        [1.0,  "#D73027"],   # 落ち着いた赤
    ]

    # ── ヒートマップ用（順次型：正の値のみ） ────────────
    HEATMAP_SEQUENTIAL = [
        [0.0,  "#F7FBFF"],
        [0.25, "#C6DBEF"],
        [0.5,  "#9ECAE1"],
        [0.75, "#6BAED6"],
        [1.0,  "#3182BD"],
    ]

    # ── 汎用カラーサイクル（最大8系列） ─────────────────
    COLOR_CYCLE = [
        "#3182BD",  # 青
        "#D73027",  # 赤
        "#74C476",  # 緑
        "#FD8D3C",  # オレンジ
        "#9E6AB0",  # 紫
        "#E7298A",  # ピンク
        "#666666",  # グレー
        "#B2B2B2",  # ライトグレー
    ]

    # ── レイアウト色 ─────────────────────────────────
    BG_PAPER    = "rgba(0,0,0,0)"   # 透過（ダークテーマに溶け込む）
    BG_PLOT     = "rgba(0,0,0,0)"
    TEXT_COLOR  = "#e0e0f0"
    GRID_COLOR  = "rgba(255,255,255,0.10)"
    FONT_FAMILY = "Inter, Meiryo, sans-serif"
    FONT_SIZE   = 12

    # ── プロット内フォントサイズ（コメント3対応） ──────
    TITLE_FONT_SIZE  = 16
    AXIS_FONT_SIZE   = 13
    TICK_FONT_SIZE   = 11
    LEGEND_FONT_SIZE = 11

    @classmethod
    def apply_to_plotly(cls, fig, chart_type: str = "generic", diverging: bool = False):
        """
        Plotly Figure にテーマを適用（非破壊・1行で利用可能）

        Args:
            fig:        plotly.graph_objects.Figure
            chart_type: "heatmap" | "scatter" | "bar" | "generic"
            diverging:  True → 発散型カラースケール、False → 順次型
        """
        fig.update_layout(
            paper_bgcolor=cls.BG_PAPER,
            plot_bgcolor=cls.BG_PLOT,
            colorway=cls.COLOR_CYCLE,
            font=dict(
                color=cls.TEXT_COLOR,
                family=cls.FONT_FAMILY,
                size=cls.FONT_SIZE,
            ),
            title=dict(font=dict(size=cls.TITLE_FONT_SIZE, family=cls.FONT_FAMILY)),
            xaxis=dict(
                gridcolor=cls.GRID_COLOR,
                zerolinecolor=cls.GRID_COLOR,
                linecolor=cls.GRID_COLOR,
                title_font=dict(size=cls.AXIS_FONT_SIZE),
                tickfont=dict(size=cls.TICK_FONT_SIZE),
            ),
            yaxis=dict(
                gridcolor=cls.GRID_COLOR,
                zerolinecolor=cls.GRID_COLOR,
                linecolor=cls.GRID_COLOR,
                title_font=dict(size=cls.AXIS_FONT_SIZE),
                tickfont=dict(size=cls.TICK_FONT_SIZE),
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0.25)",
                bordercolor=cls.GRID_COLOR,
                borderwidth=1,
                font=dict(size=cls.LEGEND_FONT_SIZE),
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
            ),
            coloraxis=dict(
                colorbar=dict(
                    thickness=18,
                    len=0.5,
                    ticks="outside",
                    tickfont=dict(size=cls.TICK_FONT_SIZE),
                    title=dict(font=dict(size=cls.AXIS_FONT_SIZE)),
                )
            ),
            margin=dict(l=55, r=35, t=55, b=55),
        )

        # トレース別設定
        for trace in fig.data:
            if trace.type == "heatmap":
                colorscale = cls.HEATMAP_DIVERGING if diverging else cls.HEATMAP_SEQUENTIAL
                trace.update(
                    colorscale=colorscale,
                    zmid=0 if diverging else None,
                    colorbar=dict(
                        title=dict(text=""),
                        thickness=18,
                        len=0.5,
                        tickfont=dict(size=cls.TICK_FONT_SIZE),
                    ),
                )
            elif trace.type in ("scatter", "scattergl"):
                if hasattr(trace, "marker") and trace.marker:
                    current_op = getattr(trace.marker, "opacity", None)
                    if current_op is None or (isinstance(current_op, (int, float)) and current_op > 0.8):
                        trace.update(marker=dict(opacity=0.75, line=dict(width=0.3, color="white")))
            elif trace.type == "bar":
                trace.update(opacity=0.85)

        return fig

    @classmethod
    def apply_square(cls, fig, size: int = 500):
        """
        散布図・PCA・パリティプロットを正方形にする（scaleanchor 使用）
        コメント3の aspecut 比問題対応
        """
        fig.update_layout(
            width=size,
            height=size,
            yaxis=dict(scaleanchor="x", scaleratio=1),
        )
        return fig

    @classmethod
    def get_heatmap_config(cls, diverging: bool = True, vmin: Optional[float] = None, vmax: Optional[float] = None) -> dict:
        """ヒートマップ作成時のキーワード引数辞書を返す"""
        return {
            "colorscale": cls.HEATMAP_DIVERGING if diverging else cls.HEATMAP_SEQUENTIAL,
            "zmid": 0 if diverging else None,
            "zmin": vmin,
            "zmax": vmax,
            "colorbar": dict(
                thickness=18,
                len=0.5,
                tickfont=dict(size=cls.TICK_FONT_SIZE),
            ),
        }
