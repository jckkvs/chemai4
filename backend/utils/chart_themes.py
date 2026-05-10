class BalancedChartTheme:
    """Coolwarmベースの明確だが目に優しいテーマ"""
    
    THEMES = {
        "coolwarm_balanced": {
            # 発散型カラー（正負の値を明確に）
            "diverging": [
                [0.0, "#3B75AF"],      # 落ち着いた青
                [0.25, "#7FA8D0"],     # ミディアムブルー
                [0.5, "#F5F5F5"],      # ほぼ白（中央値）
                [0.75, "#D89088"],     # ソフトレッド
                [1.0, "#B84840"]       # 落ち着いた赤
            ],
            # 順次カラー（一方向の値）
            "sequential": [
                [0.0, "#E8F0F8"],
                [0.25, "#B8D0E8"],
                [0.5, "#88B0D8"],
                [0.75, "#5890C8"],
                [1.0, "#3B75AF"]
            ],
            "colorway": [
                "#3B75AF",  # ブルー
                "#B84840",  # レッド
                "#5B9D6B",  # グリーン
                "#D89040",  # オレンジ
                "#8B6BC8",  # パープル
                "#C86B9E",  # ピンク
                "#4DB8B8",  # ティール
                "#B89E6B",  # ゴールド
            ],
            "background": "#FFFFFF",
            "paper_bg": "#FAFBFC",
            "font_color": "#2D3748",
            "grid_color": "#E2E8F0",
        },
        "spectral_soft": {
            # Spectralをソフトにしたバージョン
            "diverging": [
                [0.0, "#5E81C8"],      # ソフトブルー
                [0.2, "#8FA8D8"],
                [0.4, "#D8D8E8"],
                [0.5, "#F5F5F5"],
                [0.6, "#E8D8D8"],
                [0.8, "#D8A098"],
                [1.0, "#C86860"]       # ソフトレッド
            ],
            "sequential": [
                [0.0, "#E8F0F8"],
                [0.5, "#88B0D8"],
                [1.0, "#3B75AF"]
            ],
            "colorway": [
                "#5E81C8", "#C86860", "#6BA878", "#D89850",
                "#9878C8", "#C878A0", "#50B8B8", "#B8A060"
            ],
            "background": "#FFFFFF",
            "paper_bg": "#FAFBFC",
            "font_color": "#2D3748",
            "grid_color": "#E2E8F0",
        },
        "bluered_professional": {
            # 学術論文でよく使われる青赤
            "diverging": [
                [0.0, "#4575B4"],      # プロフェッショナルブルー
                [0.3, "#91BFDB"],
                [0.5, "#FEE090"],      # 中央は薄いイエロー
                [0.7, "#FC8D62"],
                [1.0, "#D73027"]       # プロフェッショナルレッド
            ],
            "sequential": [
                [0.0, "#F7FBFF"],
                [0.5, "#9ECAE1"],
                [1.0, "#2171B5"]
            ],
            "colorway": [
                "#4575B4", "#D73027", "#91BFDB", "#FC8D62",
                "#74ADD1", "#F46D43", "#313695", "#A50026"
            ],
            "background": "#FFFFFF",
            "paper_bg": "#FAFBFC",
            "font_color": "#2D3748",
            "grid_color": "#E2E8F0",
        }
    }
    
    @classmethod
    def apply(cls, fig, theme_name: str = "coolwarm_balanced", chart_type: str = "generic"):
        """チャートにバランスの取れたテーマを適用"""
        theme = cls.THEMES.get(theme_name, cls.THEMES["coolwarm_balanced"])
        
        fig.update_layout(
            colorway=theme["colorway"],
            paper_bgcolor=theme["paper_bg"],
            plot_bgcolor=theme["background"],
            font=dict(color=theme["font_color"], family="Inter, Meiryo, sans-serif", size=11),
            xaxis=dict(
                gridcolor=theme["grid_color"], 
                zerolinecolor=theme["grid_color"],
                linecolor=theme["grid_color"]
            ),
            yaxis=dict(
                gridcolor=theme["grid_color"], 
                zerolinecolor=theme["grid_color"],
                linecolor=theme["grid_color"]
            ),
            legend=dict(
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=theme["grid_color"],
                borderwidth=1,
                font=dict(size=10)
            ),
            margin=dict(l=50, r=40, t=50, b=50),
            coloraxis=dict(
                colorbar=dict(
                    thickness=25,
                    len=0.6,
                    ticks="outside",
                    tickfont=dict(size=9),
                    title=dict(font=dict(size=10))
                )
            )
        )
        
        # 各トレースに適用
        for trace in fig.data:
            if trace.type == "heatmap":
                colorscale = theme["diverging"] if chart_type == "diverging" else theme["sequential"]
                trace.update(
                    colorscale=colorscale,
                    zmid=0 if chart_type == "diverging" else None,
                    colorbar=dict(
                        thickness=20,
                        len=0.5,
                        ticks="outside",
                        tickfont=dict(size=9)
                    ),
                    showscale=True
                )
            elif trace.type in ["scatter", "scattergl"]:
                # マーカーの透明度を調整（重なりを見やすく）
                if hasattr(trace, 'marker'):
                    current_opacity = trace.marker.get('opacity', 1.0)
                    if current_opacity is None:
                        current_opacity = 1.0
                    if isinstance(current_opacity, (int, float)):
                        trace.update(
                            marker=dict(
                                opacity=min(current_opacity, 0.7),
                                line=dict(width=0.5, color='white')
                            )
                        )
            elif trace.type == "bar":
                trace.update(opacity=0.8)
        
        return fig
    
    @classmethod
    def get_heatmap_config(cls, theme_name: str = "coolwarm_balanced", 
                          vmin=None, vmax=None, zmid=0):
        """ヒートマップ用の設定を取得"""
        theme = cls.THEMES.get(theme_name, cls.THEMES["coolwarm_balanced"])
        return {
            "colorscale": theme["diverging"],
            "zmid": zmid,
            "zmin": vmin,
            "zmax": vmax,
            "colorbar": dict(
                title=dict(font=dict(size=10)),
                thickness=20,
                len=0.5
            )
        }
