"""
frontend_nicegui/components/results_tab_extras.py

results_tab.py で使用する追加ヘルパー関数群:
- _render_model_overview:   全モデル概要（棒グラフ・テーブル・CV箱ひげ図・レーダー）
- _render_per_model_tabs:   モデル別詳細サブタブ
- _render_single_model_detail: 単一モデルの詳細
- _render_pred_actual_inline:  OOF予測実測プロット（インライン版）
- _render_sample_table_inline: データ点表（インライン版）
- _render_extra_visualizations: PDP/特徴量相関/Permutation Importance等
"""
from __future__ import annotations

import numpy as np
from nicegui import run as nicegui_run, ui


# ================================================================
# 全モデル概要
# ================================================================
def _render_model_overview(ar) -> None:
    import plotly.graph_objects as go

    scores = ar.model_scores if hasattr(ar, "model_scores") else {}
    model_details = getattr(ar, "model_details", {})
    if not scores:
        ui.label("モデルスコアがありません").classes("text-grey")
        return

    sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    model_keys = [m[0] for m in sorted_models]
    model_vals = [m[1] for m in sorted_models]
    scoring = getattr(ar, "scoring", "score")

    ui.label("🏆 全モデル比較").classes("text-subtitle1 text-bold q-mb-xs")

    # スコア棒グラフ
    palette = [
        "rgba(0,212,255,0.85)" if i == 0 else "rgba(123,47,247,0.75)" if i < 3 else "rgba(74,85,104,0.65)"
        for i in range(len(model_keys))
    ]
    fig_bar = go.Figure(go.Bar(
        x=model_vals[::-1],
        y=model_keys[::-1],
        orientation="h",
        marker_color=palette[::-1],
        text=[f"{v:.4f}" for v in model_vals[::-1]],
        textposition="outside",
    ))
    fig_bar.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
        height=max(300, 28 * len(model_keys)),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=scoring,
        title=f"モデル比較スコア ({scoring})",
    )
    ui.plotly(fig_bar).classes("full-width")

    # 詳細テーブル
    ui.separator()
    ui.label("📋 詳細比較テーブル").classes("text-subtitle2 q-mt-md q-mb-xs")
    rows = []
    for rank, (mk, ms) in enumerate(sorted_models, 1):
        detail = model_details.get(mk, {})
        fold_scores = detail.get("cv_scores", [])
        std = float(np.std(fold_scores)) if fold_scores else 0.0
        rows.append({
            "順位": rank,
            "モデル": mk,
            "CVスコア": f"{ms:.4f}",
            "±std": f"{std:.4f}" if std else "—",
            "Fold数": len(fold_scores) if fold_scores else "—",
            "最良": "🏆" if rank == 1 else "",
        })
    cols = [
        {"name": k, "label": k, "field": k,
         "align": "left" if k == "モデル" else "center", "sortable": True}
        for k in ["順位", "モデル", "CVスコア", "±std", "Fold数", "最良"]
    ]
    ui.table(columns=cols, rows=rows).classes("full-width").props("dense flat bordered")

    # Fold別ボックスプロット
    fold_data = [
        (mk, det.get("cv_scores", []))
        for mk, det in model_details.items()
        if det.get("cv_scores")
    ]
    if fold_data:
        ui.separator()
        ui.label("📦 Fold別スコア分布（箱ひげ図）").classes("text-subtitle2 q-mt-md q-mb-xs")
        fig_box = go.Figure()
        for mk, cv_scores in sorted(fold_data, key=lambda x: np.mean(x[1]), reverse=True):
            fig_box.add_trace(go.Box(
                y=cv_scores, name=mk[:22],
                marker_color="#00d4ff", boxmean=True,
            ))
        fig_box.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.1)", height=380,
            margin=dict(l=10, r=10, t=30, b=60),
            yaxis_title=scoring, title="Fold別CVスコア分布",
            xaxis_tickangle=-30,
        )
        ui.plotly(fig_box).classes("full-width")

    # レーダーチャート（上位5モデル）
    if len(sorted_models) >= 3:
        try:
            n_top = min(5, len(sorted_models))
            top_models = sorted_models[:n_top]
            categories = [scoring, "安定性(1-std)", "速度スコア"]
            fig_rad = go.Figure()
            for mk, ms in top_models:
                detail = model_details.get(mk, {})
                cv_s = detail.get("cv_scores", [ms])
                std = float(np.std(cv_s)) if len(cv_s) > 1 else 0.0
                stability = max(0.0, 1.0 - std * 10.0)
                fit_time = detail.get("fit_time", 1.0) or 1.0
                speed = 1.0 - min(1.0, fit_time / 30.0)
                values = [ms, stability, speed]
                values_c = values + [values[0]]
                cats_c = categories + [categories[0]]
                fig_rad.add_trace(go.Scatterpolar(
                    r=values_c, theta=cats_c,
                    fill="toself", name=mk[:18], opacity=0.6,
                ))
            fig_rad.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                height=380, margin=dict(l=40, r=40, t=60, b=40),
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1]),
                    bgcolor="rgba(0,0,0,0.1)",
                ),
                title=f"モデル比較レーダー (上位{n_top})",
                legend=dict(orientation="h", y=-0.1),
            )
            ui.separator()
            ui.label("🕸️ レーダーチャート").classes("text-subtitle2 q-mt-md q-mb-xs")
            ui.plotly(fig_rad).classes("full-width")
        except Exception:
            pass


# ================================================================
# モデル別詳細タブ
# ================================================================
def _render_per_model_tabs(ar) -> None:
    model_details = getattr(ar, "model_details", {})
    scores = ar.model_scores if hasattr(ar, "model_scores") else {}
    all_keys = list(model_details.keys()) or list(scores.keys())
    if not all_keys:
        ui.label("モデル詳細情報がありません").classes("text-grey")
        return

    sorted_keys = sorted(all_keys, key=lambda k: scores.get(k, 0), reverse=True)

    with ui.tabs().classes("full-width").props(
        "dense active-color=purple indicator-color=purple scrollable"
    ) as model_tabs:
        for mk in sorted_keys:
            score_str = f" ({scores[mk]:.4f})" if mk in scores else ""
            label = f"🏆 {mk}{score_str}" if mk == ar.best_model_key else f"{mk}{score_str}"
            ui.tab(f"m_{mk}", label=label[:28])

    first_key = f"m_{sorted_keys[0]}"
    with ui.tab_panels(model_tabs, value=first_key).classes("full-width bg-transparent"):
        for mk in sorted_keys:
            with ui.tab_panel(f"m_{mk}"):
                _render_single_model_detail(ar, mk, model_details.get(mk, {}), scores)


def _render_single_model_detail(ar, model_key: str, detail: dict, scores: dict) -> None:
    import plotly.graph_objects as go

    ms = scores.get(model_key, 0.0)
    cv_scores = detail.get("cv_scores", [])
    params = detail.get("params", {})
    is_best = (model_key == getattr(ar, "best_model_key", ""))

    with ui.row().classes("items-center q-gutter-sm q-mb-md"):
        if is_best:
            ui.icon("emoji_events", color="amber")
        ui.label(model_key).classes("text-subtitle1 text-bold")
        ui.badge(f"CVスコア: {ms:.4f}", color="cyan" if is_best else "grey").props("dense")
        if is_best:
            ui.badge("BEST", color="amber").props("dense")

    if cv_scores:
        mean_s = float(np.mean(cv_scores))
        std_s  = float(np.std(cv_scores))

        with ui.row().classes("q-gutter-sm q-mb-sm"):
            for i, s in enumerate(cv_scores):
                with ui.card().classes("q-pa-xs").style(
                    "min-width:60px; background:rgba(0,212,255,0.07); border-radius:6px;"
                ):
                    ui.label(f"{s:.4f}").classes("text-caption text-bold text-cyan")
                    ui.label(f"Fold {i+1}").classes("text-caption text-grey-6")

        ui.label(
            f"平均: {mean_s:.4f} ± {std_s:.4f}  最小: {min(cv_scores):.4f}  最大: {max(cv_scores):.4f}"
        ).classes("text-caption text-grey-4 q-mb-sm")

        fig = go.Figure(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(cv_scores))],
            y=cv_scores,
            marker_color=[
                "rgba(0,212,255,0.8)" if s >= mean_s else "rgba(250,204,21,0.7)"
                for s in cv_scores
            ],
            text=[f"{s:.4f}" for s in cv_scores],
            textposition="outside",
        ))
        fig.add_hline(y=mean_s, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                      annotation_text=f"平均 {mean_s:.4f}")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.1)", height=280,
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis_title="CVスコア",
            title=f"{model_key} — Fold別スコア",
        )
        ui.plotly(fig).classes("full-width")

    if params:
        ui.separator()
        with ui.expansion("⚙️ ハイパーパラメータ", icon="tune").classes("full-width q-mt-sm"):
            param_rows = [{"パラメータ": str(k), "値": str(v)} for k, v in params.items()]
            ui.table(
                columns=[
                    {"name": "パラメータ", "label": "パラメータ", "field": "パラメータ", "align": "left"},
                    {"name": "値",         "label": "値",         "field": "値",         "align": "left"},
                ],
                rows=param_rows,
            ).classes("full-width").props("dense flat bordered")


# ================================================================
# 予測実測プロット（インライン版）
# ================================================================
def _render_pred_actual_inline(ar) -> None:
    from frontend_nicegui.components.interpretation_panel import _render_pred_actual

    y_true = getattr(ar, "oof_true", None)
    y_pred = getattr(ar, "oof_predictions", None)
    proc_X = getattr(ar, "processed_X", None)
    feat_names = list(proc_X.columns) if hasattr(proc_X, "columns") else []
    X_arr = proc_X.values if hasattr(proc_X, "values") else np.array([])
    _render_pred_actual(ar, y_true, y_pred, feat_names, X_arr)


# ================================================================
# データ点表（インライン版）
# ================================================================
def _render_sample_table_inline(ar) -> None:
    from frontend_nicegui.components.interpretation_panel import _render_sample_table

    proc_X = getattr(ar, "processed_X", None)
    feat_names = list(proc_X.columns) if hasattr(proc_X, "columns") else []
    y_true = getattr(ar, "oof_true", None)
    y_pred = getattr(ar, "oof_predictions", None)
    _render_sample_table(ar, proc_X, feat_names, y_true, y_pred)


# ================================================================
# 追加可視化
# ================================================================
def _render_extra_visualizations(ar, state: dict) -> None:
    import plotly.graph_objects as go

    ui.label("🎨 追加可視化").classes("text-subtitle1 text-bold q-mb-md")

    model = getattr(ar, "best_pipeline", None)
    X = getattr(ar, "processed_X", None)
    y = getattr(ar, "y_train", None)
    feat_names = list(X.columns) if hasattr(X, "columns") else []

    # ─── 1. PDP ───
    with ui.expansion("📐 PDP (Partial Dependence Plot)", icon="show_chart").classes("full-width q-mb-sm"):
        ui.label("特徴量を変化させたときの平均予測値の変化を可視化します。").classes("text-caption text-grey-5 q-mb-sm")
        if feat_names and model is not None:
            feat_sel = ui.select(feat_names[:50], value=feat_names[0], label="特徴量").props("outlined dense")
            pdp_container = ui.column().classes("full-width")

            async def _run_pdp():
                pdp_container.clear()
                fi = feat_names.index(feat_sel.value) if feat_sel.value in feat_names else 0
                try:
                    from sklearn.inspection import partial_dependence

                    def _calc():
                        # kind="both" を指定して PDP（平均）と ICE（個別）の両方を計算
                        return partial_dependence(model, X, features=[fi], grid_resolution=50, kind="both")

                    pd_result = await nicegui_run.io_bound(_calc)
                    grid = pd_result["grid_values"][0]
                    avg  = pd_result["average"][0]
                    individuals = pd_result.get("individual", np.array([]))
                    if len(individuals.shape) == 3:
                        individuals = individuals[0]  # num_classes x n_samples x grid_points -> n_samples x grid_points

                    # 生スケールへの逆変換 (X_train_raw が存在する場合)
                    display_grid = grid
                    X_raw = getattr(ar, "X_train_raw", None)
                    if X_raw is not None and hasattr(X_raw, "columns") and feat_sel.value in X_raw.columns:
                        raw_col = X_raw[feat_sel.value]
                        proc_col = X[feat_sel.value] if hasattr(X, "columns") else X[:, fi]
                        
                        p_min, p_max = float(proc_col.min()), float(proc_col.max())
                        r_min, r_max = float(raw_col.min()), float(raw_col.max())
                        
                        if abs(p_max - p_min) > 1e-6:
                            slope = (r_max - r_min) / (p_max - p_min)
                            intercept = r_min - slope * p_min
                            display_grid = grid * slope + intercept
                            logger.debug(f"PDP X-axis inverse transformed for {feat_sel.value}")

                    with pdp_container:
                        fig_pdp = go.Figure()
                        
                        # ICE線の描画（最大100本程度に間引く）
                        if len(individuals) > 0:
                            n_samples = len(individuals)
                            sample_idx = np.random.choice(n_samples, size=min(n_samples, 100), replace=False)
                            for i in sample_idx:
                                fig_pdp.add_trace(go.Scatter(
                                    x=display_grid, y=individuals[i], mode="lines",
                                    line=dict(color="rgba(100, 150, 200, 0.1)", width=1),
                                    showlegend=False,
                                    hoverinfo="skip"
                                ))

                        # PDP線（平均）の描画
                        fig_pdp.add_trace(go.Scatter(
                            x=display_grid, y=avg, mode="lines+markers",
                            line=dict(color="#00d4ff", width=3),
                            marker=dict(size=6),
                            name="Average (PDP)"
                        ))
                        
                        xaxis_type = "生データスケール" if X_raw is not None else "標準化スケール"
                        fig_pdp.update_layout(
                            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0.1)", height=380,
                            margin=dict(l=10, r=10, t=40, b=10),
                            xaxis_title=f"{feat_sel.value} ({xaxis_type})", yaxis_title="予測値",
                            title=f"PDP & ICE プロット: {feat_sel.value}",
                            showlegend=True,
                            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                        )
                        ui.plotly(fig_pdp).classes("full-width")
                except Exception as ex:
                    with pdp_container:
                        ui.label(f"PDP計算エラー: {ex}").classes("text-red text-caption")

            ui.button("📐 PDP & ICE を計算", on_click=_run_pdp).props("outline color=cyan size=sm no-caps")
            pdp_container

    # ─── 2. 特徴量相関ヒートマップ ───
    with ui.expansion("🔥 特徴量相関ヒートマップ", icon="grid_on").classes("full-width q-mb-sm"):
        ui.label("前処理後の特徴量間のSpearman相関係数を可視化します。").classes("text-caption text-grey-5 q-mb-sm")
        if X is not None and hasattr(X, "corr"):
            corr_container = ui.column().classes("full-width")

            def _show_corr():
                corr_container.clear()
                try:
                    import plotly.express as px
                    top_n = min(30, X.shape[1])
                    if X.shape[1] > top_n and y is not None:
                        y_arr = np.asarray(y).ravel()
                        abs_corr = X.apply(
                            lambda c: abs(float(np.corrcoef(c.fillna(0), y_arr)[0, 1]))
                            if len(c.dropna()) > 2 else 0.0
                        )
                        top_feats = abs_corr.nlargest(top_n).index.tolist()
                        work_X = X[top_feats]
                    else:
                        work_X = X.iloc[:, :top_n]
                    corr = work_X.corr(method="spearman")
                    fig_corr = px.imshow(
                        corr, text_auto=".2f",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title=f"Spearman相関行列（上位{top_n}特徴量）",
                    )
                    fig_corr.update_layout(
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                        height=max(420, top_n * 18),
                        margin=dict(l=10, r=10, t=40, b=80),
                    )
                    with corr_container:
                        ui.plotly(fig_corr).classes("full-width")
                except Exception as ex:
                    with corr_container:
                        ui.label(f"相関行列エラー: {ex}").classes("text-red text-caption")

            ui.button("🔥 相関行列を表示", on_click=_show_corr).props("outline color=red size=sm no-caps")
            corr_container

    # ─── 3. 分類専用指標 ───
    if getattr(ar, "task", "") in ("classification", "multiclass"):
        with ui.expansion("🔢 分類指標（PR曲線・混同行列）", icon="fact_check").classes("full-width q-mb-sm"):
            from frontend_nicegui.components.results_tab import _render_classification_metrics  # noqa: F401 (circuler safe)
            try:
                _render_classification_metrics(ar)
            except Exception as ex:
                ui.label(f"分類指標エラー: {ex}").classes("text-red text-caption")

    # ─── 4. Permutation Importance ───
    with ui.expansion("🔀 Permutation Importance", icon="shuffle").classes("full-width q-mb-sm"):
        ui.label("各特徴量をシャッフルしたとき性能がどれだけ低下するかを測定します。").classes("text-caption text-grey-5 q-mb-sm")
        perm_container = ui.column().classes("full-width")

        async def _run_perm():
            perm_container.clear()
            with perm_container:
                ui.label("⏳ 計算中...").classes("text-grey-5")
            try:
                from sklearn.inspection import permutation_importance
                scoring_key = "r2" if getattr(ar, "task", "regression") == "regression" else "accuracy"
                y_arr = np.asarray(y).ravel()

                def _calc():
                    return permutation_importance(
                        model, X, y_arr, n_repeats=5, random_state=42, scoring=scoring_key
                    )

                result = await nicegui_run.io_bound(_calc)
                sorted_idx = np.argsort(result.importances_mean)[::-1]
                top = min(20, len(sorted_idx))
                perm_container.clear()
                with perm_container:
                    fig_pi = go.Figure(go.Bar(
                        x=result.importances_mean[sorted_idx[:top]][::-1],
                        y=[feat_names[i] if i < len(feat_names) else f"f{i}"
                           for i in sorted_idx[:top]][::-1],
                        orientation="h",
                        error_x=dict(
                            type="data",
                            array=result.importances_std[sorted_idx[:top]][::-1],
                            visible=True, color="rgba(255,255,255,0.5)",
                        ),
                        marker_color="rgba(74,222,128,0.75)",
                        text=[f"{result.importances_mean[i]:.4f}" for i in sorted_idx[:top]][::-1],
                        textposition="outside",
                    ))
                    fig_pi.update_layout(
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0.1)",
                        height=max(300, 22 * top),
                        margin=dict(l=10, r=10, t=30, b=10),
                        xaxis_title=f"性能低下 ({scoring_key})",
                        title="Permutation Importance (±std, n_repeats=5)",
                    )
                    ui.plotly(fig_pi).classes("full-width")
                ui.notify("✅ Permutation Importance 完了", type="positive")
            except Exception as ex:
                perm_container.clear()
                with perm_container:
                    ui.label(f"❌ 計算エラー: {ex}").classes("text-red text-caption")

        ui.button("🔀 Permutation Importanceを計算", on_click=_run_perm).props(
            "unelevated color=green size=sm no-caps"
        )
        perm_container
