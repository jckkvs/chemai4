"""
backend/utils/plot_exporter.py

プロット自動エクスポートユーティリティ。

すべてのプロットについて、バックグラウンドで以下の2形式を自動保存する：
  1. plotly版  - Plotly の write_image() を使った高解像度 PNG （kaleido必須）
  2. matplotlib版 - 資料作成・学術論文向けの高品質 PDF + PNG

使い方::

    from backend.utils.plot_exporter import save_plot_versions

    fig = go.Figure(...)  # Plotly figure
    save_plot_versions(fig, name="parity_plot", session_id="session_001")

設定::

    - 保存ベースディレクトリ: CHEMAI_PLOT_DIR 環境変数 (なければ ./exports/plots)
    - サブディレクトリ自動生成: <base_dir>/<session_id>/<datetime>/
"""
from __future__ import annotations

import logging
import os
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── サードパーティーの非推奨警告を抑制 ──────────────────────────
# plotly/kaleido の engine 引数非推奨警告（plotly 6.x + kaleido 1.2）
warnings.filterwarnings(
    "ignore",
    message="Support for the 'engine' argument is deprecated",
    category=DeprecationWarning,
)
# plotly scattermapbox → scattermap 非推奨警告
warnings.filterwarnings(
    "ignore",
    message=".*scattermapbox.*deprecated",
    category=DeprecationWarning,
)
# matplotlib boxplot labels → tick_labels 非推奨警告（古いコードパス）
warnings.filterwarnings(
    "ignore",
    message="The 'labels' parameter of boxplot.*renamed 'tick_labels'",
    category=DeprecationWarning,
)
# Glyph 欠落警告を抑制（MS Gothic 等で ￾ が無い場合）
warnings.filterwarnings(
    "ignore",
    message="Glyph.*missing from font",
    category=UserWarning,
)

# matplotlib backend を headless に固定（GUI 不要）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import matplotlib.font_manager as fm

# CJK（中国語・日本語・韓国語）フォントの設定
def _setup_cjk_font():
    """CJK文字を表示できるフォントを設定する。"""
    cjk_fonts = [
        "MS Gothic",        # Windows Japanese
        "MS Mincho",        # Windows Japanese
        "Yu Gothic",        # Windows Japanese
        "Meiryo",          # Windows Japanese
        "SimHei",          # Windows Chinese Simplified
        "SimSun",          # Windows Chinese
        "Noto Sans CJK JP", # Google Noto (if installed)
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "WenQuanYi Micro Hei",  # Linux
        "NanumGothic",     # Linux Korean
    ]
    # 利用可能なフォントを検索
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    found = None
    for font_name in cjk_fonts:
        if font_name in available_fonts:
            found = font_name
            break
    if found:
        # フォントリストを設定（フォールバック付き）
        plt.rcParams["font.family"] = ["sans-serif", found]
        plt.rcParams["font.sans-serif"] = [found, "DejaVu Sans", "Arial"]
        plt.rcParams["axes.unicode_minus"] = False
        logger.info(f"CJKフォントを設定: {found}")
    else:
        # 見つからない場合は警告を出力
        logger.warning("CJK対応フォントが見つかりません。日本語等が正しく表示されない可能性があります。")

_setup_cjk_font()

# ── デフォルト保存先 ──────────────────────────────────────────────────────────
_DEFAULT_EXPORT_DIR = Path(os.environ.get("CHEMAI_PLOT_DIR", "exports/plots"))

# 論文向けスタイル設定
_MPL_STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.8,
}


def _get_session_dir(session_id: str | None) -> Path:
    """セッション別の保存ディレクトリを返す（なければ作成）。"""
    ts = datetime.now().strftime("%Y%m%d")
    base = _DEFAULT_EXPORT_DIR
    if session_id:
        out = base / session_id / ts
    else:
        out = base / "default" / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# メインAPI
# ─────────────────────────────────────────────────────────────────────────────

def save_plot_versions(
    plotly_fig: Any,
    name: str,
    session_id: str | None = None,
    *,
    run_async: bool = True,
    make_matplotlib: bool = True,
    make_plotly_png: bool = True,
) -> None:
    """
    Plotly figure を受け取り、バックグラウンドで複数形式に保存する。

    Args:
        plotly_fig:      plotly.graph_objects.Figure インスタンス
        name:            ファイル名ベース（例: "parity_plot"）
        session_id:      セッション識別子（保存先サブディレクトリ名）
        run_async:       True → 別スレッドで非同期保存（UIをブロックしない）
        make_matplotlib: matplotlib 版を生成してPNG+PDFで保存
        make_plotly_png: Plotly の kaleido で PNG 保存
    """
    def _save():
        out_dir = _get_session_dir(session_id)
        stem = _sanitize_name(name)

        if make_plotly_png:
            _save_plotly_png(plotly_fig, out_dir / f"{stem}_plotly.png")

        if make_matplotlib:
            _save_matplotlib_version(plotly_fig, out_dir / f"{stem}_matplotlib", stem)

    if run_async:
        t = threading.Thread(target=_save, daemon=True, name=f"plot_export_{name}")
        t.start()
    else:
        _save()


# ─────────────────────────────────────────────────────────────────────────────
# Plotly → PNG (kaleido)
# ─────────────────────────────────────────────────────────────────────────────

def _save_plotly_png(fig: Any, out_path: Path) -> None:
    """Plotly figure を kaleido 経由で高解像度 PNG に保存する。"""
    try:
        fig.write_image(str(out_path), width=1400, height=900, scale=2)
        logger.debug(f"[PlotExporter] Plotly PNG 保存: {out_path}")
    except Exception as e:
        logger.warning(f"[PlotExporter] Plotly PNG 保存失敗 ({out_path.name}): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotly → Matplotlib 変換 + 保存
# ─────────────────────────────────────────────────────────────────────────────

def _save_matplotlib_version(plotly_fig: Any, out_stem: Path, title: str) -> None:
    """
    Plotly figure のトレース情報from Plotly を解析して、matplotlib 図を再構成して保存。

    すべてのトレースタイプに対応する汎用コンバーター。
    複雑なレイアウト（3D等）はフォールバックとして基本プロット生成。
    """
    try:
        traces = plotly_fig.data

        n_traces = len(traces)
        if n_traces == 0:
            logger.debug(f"[PlotExporter] トレースなし: {out_stem.name}")
            return

        with mplstyle.context(_MPL_STYLE):
            # サブプロット検出
            has_subplots = _has_subplots(plotly_fig)

            if has_subplots:
                _render_subplots_fig(plotly_fig, out_stem, title)
            else:
                _render_single_fig(plotly_fig, out_stem, title)

    except Exception as e:
        logger.warning(f"[PlotExporter] matplotlib 変換エラー ({out_stem.name}): {e}")
        _save_fallback_fig(out_stem, str(e))


def _has_subplots(plotly_fig: Any) -> bool:
    """サブプロットを含む figure かどうかを判定する。"""
    layout = plotly_fig.layout
    try:
        layout_dict = layout.to_plotly_json()
        for key in layout_dict:
            if key.startswith("xaxis") and key != "xaxis":
                return True
    except Exception:
        pass
    # グリッド情報チェック
    if hasattr(layout, "grid") and layout.grid is not None:
        return True
    return False


def _render_single_fig(plotly_fig: Any, out_stem: Path, title: str) -> None:
    """単一軸の figure を matplotlib で再描画。"""
    with mplstyle.context(_MPL_STYLE):
        fig_mpl, ax = plt.subplots(figsize=(8, 6))

        layout = plotly_fig.layout
        fig_title = _get_layout_title(layout) or title.replace("_", " ").title()
        ax.set_title(fig_title, fontsize=13, fontweight="bold", pad=12)

        x_title = _get_axis_title(layout, "xaxis")
        y_title = _get_axis_title(layout, "yaxis")
        if x_title:
            ax.set_xlabel(x_title)
        if y_title:
            ax.set_ylabel(y_title)

        _render_traces_to_ax(ax, plotly_fig.data, plotly_fig.layout)

        ax.legend(fontsize=9, framealpha=0.8) if ax.get_legend_handles_labels()[0] else None
        fig_mpl.tight_layout()
        _save_fig(fig_mpl, out_stem)
        plt.close(fig_mpl)


def _render_subplots_fig(plotly_fig: Any, out_stem: Path, title: str) -> None:
    """サブプロット figure を matplotlib で再描画（2列グリッドを想定）。"""
    with mplstyle.context(_MPL_STYLE):
        # Plotlyのトレースをグループ化
        traces_by_subplot = _group_traces_by_subplot(plotly_fig)
        n_subplots = len(traces_by_subplot)

        if n_subplots == 0:
            return

        cols = min(2, n_subplots)
        rows = (n_subplots + cols - 1) // cols
        fig_mpl, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows))

        if n_subplots == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        layout = plotly_fig.layout
        layout_dict = {}
        try:
            layout_dict = layout.to_plotly_json()
        except Exception:
            pass

        for idx, (subplot_key, traces) in enumerate(traces_by_subplot.items()):
            r, c = divmod(idx, cols)
            ax = axes[r, c]

            # サブプロットタイトル
            subplot_titles = _get_subplot_titles(layout)
            if idx < len(subplot_titles):
                ax.set_title(subplot_titles[idx], fontsize=11, fontweight="bold")

            # 各軸のラベル
            x_key = f"xaxis{'' if subplot_key == '1' else subplot_key}"
            y_key = f"yaxis{'' if subplot_key == '1' else subplot_key}"
            x_title = layout_dict.get(x_key, {}).get("title", {})
            y_title_d = layout_dict.get(y_key, {}).get("title", {})
            if isinstance(x_title, dict):
                ax.set_xlabel(x_title.get("text", ""))
            elif isinstance(x_title, str):
                ax.set_xlabel(x_title)
            if isinstance(y_title_d, dict):
                ax.set_ylabel(y_title_d.get("text", ""))
            elif isinstance(y_title_d, str):
                ax.set_ylabel(y_title_d)

            _render_traces_to_ax(ax, traces, layout)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8, framealpha=0.8)

        # 余分な軸を非表示に
        for idx in range(n_subplots, rows * cols):
            r, c = divmod(idx, cols)
            axes[r, c].set_visible(False)

        fig_title = _get_layout_title(layout) or title.replace("_", " ").title()
        fig_mpl.suptitle(fig_title, fontsize=14, fontweight="bold", y=1.01)
        fig_mpl.tight_layout()
        _save_fig(fig_mpl, out_stem)
        plt.close(fig_mpl)


# ─────────────────────────────────────────────────────────────────────────────
# トレース→Axes 変換
# ─────────────────────────────────────────────────────────────────────────────

def _render_traces_to_ax(ax, traces, layout=None) -> None:
    """トレースリストを matplotlib Axes に描画する。"""

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    _color_idx = [0]

    def _next_color():
        c = color_cycle[_color_idx[0] % len(color_cycle)]
        _color_idx[0] += 1
        return c

    for trace in traces:
        trace_type = type(trace).__name__.lower()

        try:
            if "scatter" in trace_type and "3d" not in trace_type:
                _render_scatter(ax, trace, _next_color())
            elif "bar" in trace_type:
                _render_bar(ax, trace, _next_color())
            elif "heatmap" in trace_type:
                _render_heatmap(ax, trace)
            elif "histogram" in trace_type:
                _render_histogram(ax, trace, _next_color())
            elif "box" in trace_type:
                _render_box(ax, trace, _next_color())
            elif "violin" in trace_type:
                _render_violin(ax, trace, _next_color())
            elif "waterfall" in trace_type:
                _render_waterfall(ax, trace)
            elif "pie" in trace_type:
                _render_pie(ax, trace)
            else:
                logger.debug(f"[PlotExporter] 未対応トレース型: {trace_type}")
        except Exception as e:
            logger.debug(f"[PlotExporter] トレース描画エラー ({trace_type}): {e}")


def _render_scatter(ax, trace, color: str) -> None:
    """Scatter トレースを描画。"""
    x = _to_array(trace.x)
    y = _to_array(trace.y)
    if x is None or y is None or len(x) == 0:
        return

    name = trace.name or ""
    mode = getattr(trace, "mode", "markers") or "markers"

    marker_kwargs: dict = {"s": 20, "alpha": 0.7, "label": name}
    line_kwargs: dict = {"alpha": 0.8, "label": name}

    # マーカー色の取得
    marker = getattr(trace, "marker", None)
    if marker is not None:
        mc = getattr(marker, "color", None)
        if isinstance(mc, str) and mc.startswith("rgba"):
            color = _rgba_to_mpl(mc) or color
        elif isinstance(mc, str) and mc.startswith("#"):
            color = mc
        marker_kwargs["color"] = color
        line_kwargs["color"] = color

        sz = getattr(marker, "size", None)
        if isinstance(sz, (int, float)):
            marker_kwargs["s"] = sz * 2

    # line style
    line = getattr(trace, "line", None)
    if line is not None:
        dash = getattr(line, "dash", None)
        if dash == "dash":
            line_kwargs["linestyle"] = "--"
        elif dash == "dot":
            line_kwargs["linestyle"] = ":"

    if "markers" in mode and "lines" in mode:
        ax.plot(x, y, "o-", markersize=4, **line_kwargs)
    elif "markers" in mode:
        ax.scatter(x, y, **marker_kwargs)
    elif "lines" in mode:
        ax.plot(x, y, **line_kwargs)
    elif "text" in mode:
        ax.scatter(x, y, s=10, alpha=0.5, color=color)


def _render_bar(ax, trace, color: str) -> None:
    """Bar トレースを描画。"""
    x = _to_array(trace.x)
    y = _to_array(trace.y)
    if x is None or y is None:
        return

    orientation = getattr(trace, "orientation", "v") or "v"
    name = trace.name or ""

    marker = getattr(trace, "marker", None)
    mc = None
    if marker is not None:
        mc = getattr(marker, "color", None)
    bar_color = (_rgba_to_mpl(mc) if isinstance(mc, str) else color)

    if orientation == "h":
        ax.barh(x, y, color=bar_color, alpha=0.8, label=name)
    else:
        ax.bar(x, y, color=bar_color, alpha=0.8, label=name)


def _render_heatmap(ax, trace) -> None:
    """Heatmap トレースを描画。"""
    z = _to_array(trace.z)
    if z is None:
        return
    z_arr = np.array(z, dtype=float)
    if z_arr.ndim != 2:
        return

    x_labels = list(trace.x) if trace.x is not None else None
    y_labels = list(trace.y) if trace.y is not None else None

    colorscale = getattr(trace, "colorscale", "RdBu_r") or "RdBu_r"
    cmap = _plotly_colorscale_to_mpl(colorscale)
    vmin = getattr(trace, "zmin", None)
    vmax = getattr(trace, "zmax", None)

    im = ax.imshow(z_arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if x_labels is not None and len(x_labels) <= 40:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels([str(label) for label in x_labels], rotation=45, ha="right", fontsize=8)
    if y_labels is not None and len(y_labels) <= 40:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels([str(label) for label in y_labels], fontsize=8)

    # テキスト注釈（小さいヒートマップのみ）
    text_arr = getattr(trace, "text", None)
    if text_arr is not None and z_arr.size <= 400:
        for i in range(z_arr.shape[0]):
            for j in range(z_arr.shape[1]):
                val = z_arr[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color="black" if abs(val) < 0.5 else "white")


def _render_histogram(ax, trace, color: str) -> None:
    """Histogram トレースを描画。"""
    x = _to_array(trace.x)
    if x is None or len(x) == 0:
        return
    nbinsx = getattr(trace, "nbinsx", 30) or 30
    name = trace.name or ""
    ax.hist(x, bins=int(nbinsx), color=color, alpha=0.7, label=name, edgecolor="white", lw=0.5)


def _render_box(ax, trace, color: str) -> None:
    """Box トレースを描画。"""
    y = _to_array(trace.y)
    if y is None or len(y) == 0:
        return
    name = trace.name or ""
    ax.boxplot(y, tick_labels=[name] if name else None, patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.7))


def _render_violin(ax, trace, color: str) -> None:
    """Violin トレースを描画（近似: boxplot で代用）。"""
    y = _to_array(trace.y)
    if y is None or len(y) == 0:
        return
    name = trace.name or ""
    try:
        parts = ax.violinplot([y], positions=[0], showmedians=True, showextrema=True)
        for pc in parts.get("bodies", []):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax.set_xticks([0])
        ax.set_xticklabels([name] if name else [""])
    except Exception:
        ax.boxplot(y, tick_labels=[name] if name else None)


def _render_waterfall(ax, trace) -> None:
    """Waterfall トレースを描画（棒グラフで近似）。"""
    x = _to_array(trace.x)
    y = _to_array(trace.y)
    if x is None or y is None:
        return
    y_arr = np.asarray(y, dtype=float)
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in y_arr]
    name = trace.name or ""
    ax.bar(range(len(y_arr)), y_arr, color=colors, alpha=0.8, label=name)
    ax.set_xticks(range(len(y_arr)))
    ax.set_xticklabels([str(xx) for xx in x], rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="gray", lw=0.8)


def _render_pie(ax, trace) -> None:
    """Pie トレースを描画（最初のサブプロットのみ適用）。"""
    values = _to_array(trace.values)
    labels = list(trace.labels) if getattr(trace, "labels", None) is not None else None
    if values is None or len(values) == 0:
        return
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")


# ─────────────────────────────────────────────────────────────────────────────
# ヘルパー
# ─────────────────────────────────────────────────────────────────────────────

def _to_array(v) -> Any | None:
    """Plotlyのデータソース値を numpy-like に変換。Noneなら None を返す。"""
    if v is None:
        return None
    try:
        return np.asarray(v)
    except Exception:
        return list(v) if hasattr(v, "__iter__") else None


def _rgba_to_mpl(rgba_str: str | None) -> str | None:
    """'rgba(R,G,B,A)' 形式の文字列を matplotlib 用 (R,G,B,A) タプルに変換。"""
    if not rgba_str:
        return None
    try:
        if rgba_str.startswith("rgba"):
            inner = rgba_str[5:-1]
            parts = [float(p.strip()) for p in inner.split(",")]
            if len(parts) == 4:
                r, g, b, a = parts
                return (r / 255, g / 255, b / 255, a)
        elif rgba_str.startswith("rgb"):
            inner = rgba_str[4:-1]
            parts = [float(p.strip()) for p in inner.split(",")]
            if len(parts) == 3:
                r, g, b = parts
                return (r / 255, g / 255, b / 255, 1.0)
    except Exception:
        pass
    return None


def _plotly_colorscale_to_mpl(cs) -> str:
    """Plotly colorscale 名称を matplotlib colormap 名に変換する。"""
    mapping = {
        "rdbu_r": "RdBu_r",
        "rdbu": "RdBu",
        "viridis": "viridis",
        "plasma": "plasma",
        "blues": "Blues",
        "reds": "Reds",
        "teal": "GnBu",
        "greens": "Greens",
        "oranges": "Oranges",
        "greys": "Greys",
    }
    if isinstance(cs, str):
        return mapping.get(cs.lower(), "viridis")
    return "viridis"


def _get_layout_title(layout) -> str | None:
    """layout.title.text を安全に取得する。"""
    try:
        t = layout.title
        if hasattr(t, "text"):
            return t.text or None
        return None
    except Exception:
        return None


def _get_axis_title(layout, axis_key: str) -> str | None:
    """layout.xaxis.title.text などを安全に取得する。"""
    try:
        axis = getattr(layout, axis_key, None)
        if axis is None:
            return None
        t = axis.title
        if hasattr(t, "text"):
            return t.text or None
        return None
    except Exception:
        return None


def _get_subplot_titles(layout) -> list[str]:
    """layout.annotations からサブプロットタイトルを取得する。"""
    try:
        titles = []
        for ann in layout.annotations or []:
            text = getattr(ann, "text", None)
            if text:
                titles.append(text)
        return titles
    except Exception:
        return []


def _group_traces_by_subplot(plotly_fig) -> dict[str, list]:
    """トレースをサブプロット（xaxis番号）別にグループ化する。"""
    groups: dict[str, list] = {}
    for trace in plotly_fig.data:
        # xaxis の番号を取得
        xaxis = getattr(trace, "xaxis", None) or "x"
        key = xaxis.lstrip("x") or "1"
        key = key if key else "1"
        groups.setdefault(key, []).append(trace)
    # キーをソート
    return dict(sorted(groups.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 99))


def _sanitize_name(name: str) -> str:
    """ファイル名に使えない文字を除去する。"""
    import re
    name = re.sub(r"[^\w\-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:80]


def _save_fig(fig_mpl, out_stem: Path) -> None:
    """matplotlib figure を PNG + PDF で保存する。"""
    png_path = out_stem.with_suffix(".png")
    pdf_path = out_stem.with_suffix(".pdf")
    try:
        fig_mpl.savefig(str(png_path), dpi=300, bbox_inches="tight", facecolor="white")
        logger.debug(f"[PlotExporter] matplotlib PNG 保存: {png_path}")
    except Exception as e:
        logger.warning(f"[PlotExporter] matplotlib PNG 保存失敗: {e}")
    try:
        fig_mpl.savefig(str(pdf_path), bbox_inches="tight", facecolor="white")
        logger.debug(f"[PlotExporter] matplotlib PDF 保存: {pdf_path}")
    except Exception as e:
        logger.debug(f"[PlotExporter] matplotlib PDF 保存失敗 (PDFバックエンド不可): {e}")


def _save_fallback_fig(out_stem: Path, error_msg: str) -> None:
    """変換失敗時のフォールバック（エラーメッセージ図）を保存する。"""
    try:
        with mplstyle.context(_MPL_STYLE):
            fig_mpl, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, f"Export error:\n{error_msg[:200]}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="red", wrap=True)
            ax.axis("off")
            _save_fig(fig_mpl, out_stem)
            plt.close(fig_mpl)
    except Exception:
        pass
