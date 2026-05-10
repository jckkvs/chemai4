"""
backend/utils/plot_export_hooks.py

解析結果から自動的にすべてのプロット（Plotly + matplotlib）をバックグラウンドで保存するフック。

使い方（analysis_runner.py の解析完了後に呼び出す）::

    from backend.utils.plot_export_hooks import export_all_plots_from_result
    export_all_plots_from_result(ar, state, session_id="my_session")

保存ディレクトリ: exports/plots/<session_id>/<date>/
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def export_all_plots_from_result(
    ar: Any,
    state: dict,
    session_id: str | None = None,
    *,
    run_async: bool = True,
) -> None:
    """
    AutoML 結果オブジェクト（ar）から全プロットをバックグラウンドで出力する。

    生成されるプロット:
        - parity_plot         : 予測 vs 実測（OOFスキャッタープロット）
        - residual_plot       : 残差プロット
        - feature_importance  : 特徴量重要度
        - cv_scores           : 各モデルのFold別CVスコア箱ひげ図
        - model_comparison    : 全モデルCVスコア棒グラフ
        - learning_curve      : 学習曲線（データが存在する場合）

    Args:
        ar:           AutoMLEngine の実行結果オブジェクト
        state:        セッション状態辞書
        session_id:   保存先サブディレクトリ名（Noneなら "default"）
        run_async:    True → 別スレッドで非同期実行
    """
    def _run():
        try:
            _export_impl(ar, state, session_id)
        except Exception as e:
            logger.warning(f"[PlotExportHooks] 全体エラー: {e}")

    if run_async:
        t = threading.Thread(target=_run, daemon=True, name="plot_export_all")
        t.start()
    else:
        _run()


def _export_impl(ar: Any, state: dict, session_id: str | None) -> None:
    """実際のエクスポート処理（スレッド内で実行）。"""
    from backend.utils.plot_exporter import save_plot_versions

    logger.info(f"[PlotExportHooks] プロットエクスポート開始: session={session_id}")

    y_true = getattr(ar, "oof_true", None)
    y_pred = getattr(ar, "oof_predictions", None)
    model = getattr(ar, "best_pipeline", None)
    proc_X = getattr(ar, "processed_X", None)
    task = getattr(ar, "task", "regression")
    best_model_key = getattr(ar, "best_model_key", "best_model")
    model_scores = getattr(ar, "model_scores", {})
    model_details = getattr(ar, "model_details", {})

    feat_names: list[str] = []
    if proc_X is not None and hasattr(proc_X, "columns"):
        feat_names = list(proc_X.columns)

    # ── 1. パリティプロット（予測 vs 実測） ──────────────────────────────
    if y_true is not None and y_pred is not None and task == "regression":
        try:
            fig = _make_parity_plotly(y_true, y_pred, title=f"予測 vs 実測 ({best_model_key})")
            save_plot_versions(fig, name="parity_plot", session_id=session_id, run_async=False)
            logger.debug("[PlotExportHooks] parity_plot 保存完了")
        except Exception as e:
            logger.warning(f"[PlotExportHooks] parity_plot エラー: {e}")

    # ── 2. 残差プロット ──────────────────────────────────────────────────
    if y_true is not None and y_pred is not None and task == "regression":
        try:
            fig = _make_residual_plotly(y_true, y_pred)
            save_plot_versions(fig, name="residual_plot", session_id=session_id, run_async=False)
            logger.debug("[PlotExportHooks] residual_plot 保存完了")
        except Exception as e:
            logger.warning(f"[PlotExportHooks] residual_plot エラー: {e}")

    # ── 3. Feature Importance ──────────────────────────────────────────
    if model is not None and feat_names:
        try:
            fig = _make_feature_importance_plotly(model, feat_names, title=f"Feature Importance ({best_model_key})")
            if fig is not None:
                save_plot_versions(fig, name="feature_importance", session_id=session_id, run_async=False)
                logger.debug("[PlotExportHooks] feature_importance 保存完了")
        except Exception as e:
            logger.warning(f"[PlotExportHooks] feature_importance エラー: {e}")

    # ── 4. モデル比較（棒グラフ） ────────────────────────────────────────
    if model_scores:
        try:
            fig = _make_model_comparison_plotly(model_scores, getattr(ar, "scoring", "score"))
            save_plot_versions(fig, name="model_comparison", session_id=session_id, run_async=False)
            logger.debug("[PlotExportHooks] model_comparison 保存完了")
        except Exception as e:
            logger.warning(f"[PlotExportHooks] model_comparison エラー: {e}")

    # ── 5. CVスコア箱ひげ図 ─────────────────────────────────────────────
    if model_details:
        try:
            fig = _make_cv_boxplot_plotly(model_details, getattr(ar, "scoring", "score"))
            if fig is not None:
                save_plot_versions(fig, name="cv_scores_boxplot", session_id=session_id, run_async=False)
                logger.debug("[PlotExportHooks] cv_scores_boxplot 保存完了")
        except Exception as e:
            logger.warning(f"[PlotExportHooks] cv_scores_boxplot エラー: {e}")

    # ── 6. 混同行列（分類タスクのみ） ───────────────────────────────────
    if task == "classification" and y_true is not None and y_pred is not None:
        try:
            fig = _make_confusion_matrix_plotly(y_true, y_pred)
            save_plot_versions(fig, name="confusion_matrix", session_id=session_id, run_async=False)
            logger.debug("[PlotExportHooks] confusion_matrix 保存完了")
        except Exception as e:
            logger.warning(f"[PlotExportHooks] confusion_matrix エラー: {e}")

    logger.info(f"[PlotExportHooks] プロットエクスポート完了: session={session_id}")


# ─────────────────────────────────────────────────────────────────────────────
# 個別プロット生成関数（Plotly）
# ─────────────────────────────────────────────────────────────────────────────

def _make_parity_plotly(y_true, y_pred, title: str = "予測 vs 実測"):
    """OOF 予測実測散布図（Plotly）を作成する。"""
    import plotly.graph_objects as go

    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    residuals = y_t - y_p

    rng_min = float(min(np.nanmin(y_t), np.nanmin(y_p)))
    rng_max = float(max(np.nanmax(y_t), np.nanmax(y_p)))
    margin = (rng_max - rng_min) * 0.05
    axis_range = [rng_min - margin, rng_max + margin]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=axis_range, y=axis_range, mode="lines",
        line=dict(color="rgba(255,255,255,0.25)", dash="dash", width=1.5),
        name="y = x",
    ))
    fig.add_trace(go.Scatter(
        x=y_t, y=y_p, mode="markers",
        marker=dict(
            size=7, color=residuals, colorscale="RdBu_r",
            showscale=True, colorbar=dict(title="残差"), opacity=0.75,
        ),
        name="データ点",
        customdata=residuals,
        hovertemplate="実測: %{x:.4g}<br>予測: %{y:.4g}<br>残差: %{customdata:.4g}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=f"{title}（n={len(y_t)}）", font=dict(size=17)),
        xaxis=dict(title="実測値", range=axis_range, gridcolor="rgba(255,255,255,0.10)"),
        yaxis=dict(title="予測値", range=axis_range, scaleanchor="x", scaleratio=1,
                   gridcolor="rgba(255,255,255,0.10)"),
        width=700, height=700,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.12)",
        margin=dict(l=65, r=35, t=65, b=65),
    )
    return fig


def _make_residual_plotly(y_true, y_pred):
    """残差プロット（Plotly）を作成する。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    residuals = y_t - y_p

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["残差 vs 予測値", "残差の分布"],
        column_widths=[0.6, 0.4],
    )

    # 残差 vs 予測値
    fig.add_trace(go.Scatter(
        x=y_p, y=residuals, mode="markers",
        marker=dict(size=5, color="rgba(0,180,255,0.6)", opacity=0.70),
        name="残差",
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", row=1, col=1)

    # 残差の分布
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=30, name="残差分布",
        marker_color="rgba(0,180,255,0.6)",
    ), row=1, col=2)

    fig.update_layout(
        template="plotly_dark", title="残差分析",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.12)",
        height=450, showlegend=False,
    )
    fig.update_xaxes(title_text="予測値", row=1, col=1)
    fig.update_yaxes(title_text="残差", row=1, col=1)
    fig.update_xaxes(title_text="残差", row=1, col=2)
    fig.update_yaxes(title_text="度数", row=1, col=2)
    return fig


def _make_feature_importance_plotly(model, feat_names: list[str], title: str = "Feature Importance"):
    """Feature Importance / 回帰係数の棒グラフ（Plotly）を作成する。"""
    import plotly.graph_objects as go

    # 推定器の取得
    estimator = model
    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]
        if hasattr(estimator, "steps"):
            estimator = estimator.steps[-1][1]

    if hasattr(estimator, "feature_importances_"):
        imp = estimator.feature_importances_
        names = feat_names[:len(imp)] if len(feat_names) >= len(imp) else [f"f{i}" for i in range(len(imp))]
        idx = np.argsort(imp)[::-1]
        top = min(30, len(idx))

        fig = go.Figure(go.Bar(
            x=imp[idx[:top]][::-1],
            y=[names[i] for i in idx[:top]][::-1],
            orientation="h",
            marker=dict(color=imp[idx[:top]][::-1], colorscale="Viridis", showscale=True),
        ))
        fig.update_layout(
            template="plotly_dark", title=title,
            xaxis_title="重要度",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.12)",
            height=max(400, 22 * top),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    elif hasattr(estimator, "coef_"):
        coefs = estimator.coef_.ravel()
        names = feat_names[:len(coefs)] if len(feat_names) >= len(coefs) else [f"f{i}" for i in range(len(coefs))]
        idx = np.argsort(np.abs(coefs))[::-1]
        top = min(30, len(idx))
        colors = ["rgba(74,222,128,0.75)" if coefs[i] >= 0 else "rgba(248,113,113,0.75)" for i in idx[:top]]

        fig = go.Figure(go.Bar(
            x=coefs[idx[:top]][::-1],
            y=[names[i] for i in idx[:top]][::-1],
            orientation="h", marker_color=colors[::-1],
        ))
        fig.update_layout(
            template="plotly_dark", title=title,
            xaxis_title="回帰係数",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.12)",
            height=max(400, 22 * top),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    return None


def _make_model_comparison_plotly(model_scores: dict, scoring: str = "score"):
    """全モデルのCVスコアを棒グラフで比較する（Plotly）。"""
    import plotly.graph_objects as go

    sorted_items = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, _ in sorted_items]
    scores = [v for _, v in sorted_items]
    best_score = max(scores) if scores else 0

    colors = [
        "rgba(255,215,0,0.8)" if s == best_score else "rgba(0,180,255,0.6)"
        for s in scores
    ]

    fig = go.Figure(go.Bar(
        x=names, y=scores,
        marker=dict(color=colors),
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark",
        title=f"モデル比較 ({scoring})",
        xaxis_tickangle=-30,
        yaxis_title=scoring,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.12)",
        height=400,
        margin=dict(l=10, r=10, t=50, b=80),
    )
    return fig


def _make_cv_boxplot_plotly(model_details: dict, scoring: str = "score"):
    """各モデルのFold別CVスコアを箱ひげ図で比較する（Plotly）。"""
    import plotly.graph_objects as go

    fold_data = [
        (mk, det.get("cv_scores", []))
        for mk, det in model_details.items()
        if det.get("cv_scores")
    ]
    if not fold_data:
        return None

    fig = go.Figure()
    for mk, cv_scores in sorted(fold_data, key=lambda x: float(np.mean(x[1])), reverse=True):
        fig.add_trace(go.Box(
            y=cv_scores, name=mk[:25], boxmean=True,
            marker_color="rgba(0,180,255,0.7)",
        ))
    fig.update_layout(
        template="plotly_dark",
        title=f"Fold別CVスコア分布 ({scoring})",
        yaxis_title=scoring,
        xaxis_tickangle=-30,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.12)",
        height=450,
        margin=dict(l=10, r=10, t=50, b=80),
    )
    return fig


def _make_confusion_matrix_plotly(y_true, y_pred):
    """混同行列のヒートマップ（Plotly）を作成する。"""
    import plotly.figure_factory as ff
    from sklearn.metrics import confusion_matrix

    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    classes = np.unique(np.concatenate([y_t, y_p]))
    cm = confusion_matrix(y_t, y_p, labels=classes)
    class_labels = [str(c) for c in classes]

    fig = ff.create_annotated_heatmap(
        z=cm.tolist(),
        x=class_labels, y=class_labels,
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        template="plotly_dark",
        title="混同行列",
        xaxis_title="予測クラス",
        yaxis_title="実クラス",
        paper_bgcolor="rgba(0,0,0,0)",
        height=max(400, 60 * len(classes)),
        margin=dict(l=10, r=10, t=50, b=60),
    )
    return fig
