import numpy as np
from backend.utils.chart_themes import BalancedChartTheme

def sanitize_axis(fig, auto_clip_percentiles=(1, 99), log_toggle=False):
    """既存Figureの軸を自動整頓（非破壊）"""
    for axis in ["xaxis", "yaxis"]:
        if not hasattr(fig.layout, axis): continue
        ax = getattr(fig.layout, axis)
        # 1. 自動クリッピング
        if auto_clip_percentiles:
            vals = [t.z if hasattr(t, "z") else t.y for t in fig.data if hasattr(t, "z") or hasattr(t, "y")]
            flat = []
            for v in vals:
                if v is not None:
                    try:
                        # numpy array or list
                        flat.extend(np.ravel(np.array(v, dtype=float)))
                    except Exception:
                        pass
            if len(flat) > 0:
                # remove nans
                flat = [v for v in flat if not np.isnan(v)]
                if len(flat) > 0:
                    vmin, vmax = np.percentile(flat, auto_clip_percentiles)
                    fig.update_layout({f"{axis}": {"range": [vmin, vmax], "autorange": False}})

        # 2. 対数軸トグル（UIフラグ連動）
        if log_toggle:
            fig.update_layout({f"{axis}": {"type": "log", "exponentformat": "power"}})
            
    return fig

class ChartSanitizer:
    """可視化の軸・色・メタ情報を一括制御（既存関数をラップ不要）"""
    def __init__(self, config: dict = None):
        self.cfg = config or {
            "clip_percentiles": (1, 99),
            "colorscale_matrix": "RdBu_r",
            "colorscale_sequential": "Viridis",
            "auto_log_threshold": 2.0,  # skew > この値ならlog推奨
            "max_shap_features": 30,
            "theme": "light"
        }

    def apply(self, fig, chart_type: str = "generic"):
        if chart_type == "matrix" or chart_type == "diverging":
            chart_theme_type = "diverging"
        elif chart_type in ["sequential", "distribution"]:
            chart_theme_type = "sequential"
        else:
            chart_theme_type = "generic"
            
        if chart_type == "distribution":
            skew = self._estimate_skew(fig)
            if abs(skew) > self.cfg["auto_log_threshold"]:
                fig.add_annotation(text="⚠️ 裾が長いため対数軸を推奨", xref="paper", yref="paper", x=1, y=1)
        elif chart_type == "parity":
             # 予測値 vs 実測値
             pass
        elif chart_type == "shap":
             # truncate features
             pass
        
        if chart_type in ["parity", "distribution"]:
            fig = sanitize_axis(fig, self.cfg["clip_percentiles"])
            
        fig = BalancedChartTheme.apply(fig, theme_name="coolwarm_balanced", chart_type=chart_theme_type)
        return fig

    def _estimate_skew(self, fig):
        # 簡易: 最初のtraceのyまたはzから算出
        if len(fig.data) == 0: return 0
        vals = getattr(fig.data[0], "y", None) or getattr(fig.data[0], "z", None)
        if vals is None: return 0
        from scipy.stats import skew
        try:
            return skew(np.ravel(np.array(vals, dtype=float)), nan_policy="omit")
        except Exception:
            return 0

_sanitizer = ChartSanitizer()

def sanitize_chart(fig, chart_type="generic"):
    return _sanitizer.apply(fig, chart_type=chart_type)
