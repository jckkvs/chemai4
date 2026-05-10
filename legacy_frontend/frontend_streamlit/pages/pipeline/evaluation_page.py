"""frontend_streamlit/pages/evaluation_page.py - モデル評価ページ（強化版）"""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st


def _get_feature_df(df: "pd.DataFrame", target_col: str) -> "pd.DataFrame":
    """除外列・weight列・group列を考慮した説明変数DataFrameを返す。"""
    drop_cols = [target_col]
    # 除外列 + Info列
    drop_cols.extend(st.session_state.get("col_role_exclude", []))
    drop_cols.extend(st.session_state.get("col_role_info", []))
    # Weight列・Group列（説明変数ではない）
    _w = st.session_state.get("col_role_weight")
    if _w:
        drop_cols.append(_w)
    _g = st.session_state.get("col_role_group")
    if _g:
        drop_cols.append(_g)
    # 存在する列のみdrop
    drop_cols = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=drop_cols)


def render() -> None:
    st.markdown("## 📊 モデル評価")
    result = st.session_state.get("automl_result")
    if result is None:
        st.warning("⚠️ まずAutoMLを実行してください。")
        if st.button("🤖 AutoMLへ"):
            st.session_state["page"] = "automl"
            st.rerun()
        return

    df = st.session_state.get("df")
    target_col = st.session_state.get("target_col")

    # ─── サマリーカード ─────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    metric_data = [
        (col1, result.best_model_key, "🏆 最良モデル"),
        (col2, result.task, "📋 タスク"),
        (col3, f"{result.best_score:.4f}", f"📈 {result.scoring}"),
        (col4, f"{result.elapsed_seconds:.1f}s", "⏱️ 学習時間"),
    ]
    for col, val, label in metric_data:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value" style="font-size:1.3rem">'
                f'{val}</div><div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # P2-5: メトリクス用語ガイド
    with st.expander("📖 評価指標の読み方", expanded=False):
        if result.task == "regression":
            st.markdown(
                "- **R²** (決定係数): 1に近いほど高精度。0.9以上が理想的\n"
                "- **RMSE** (二乗平均平方根誤差): 小さいほど良い。目的変数のスケールに依存\n"
                "- **MAE** (平均絶対誤差): 外れ値に頑健な誤差指標\n"
                "- **CV OOF**: 各foldで未学習データに対する予測 → **最もリアルな汎化精度**"
            )
        else:
            st.markdown(
                "- **Accuracy**: 全体の正解率\n"
                "- **F1** (weighted): 精度とリコールの調和平均。不均衡データに有用\n"
                "- **Precision**: 陽性予測のうち実際に陽性の割合\n"
                "- **Recall**: 実際の陽性のうち正しく予測された割合"
            )

    # ─── タブ ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 予測精度", "📊 全モデル比較 & Fold分布", "📉 学習曲線", "🔍 残差/混同行列"]
    )

    # ─── Tab1: 予測精度 ───────────────────────────────────────────
    with tab1:
        if df is not None and target_col:
            X = _get_feature_df(df, target_col)
            y_true = df[target_col].values

            # --- SMILES列がある場合は事前展開してから予測する ---
            smiles_col = st.session_state.get("smiles_col")
            if smiles_col and smiles_col in X.columns:
                try:
                    from backend.chem.smiles_transformer import SmilesDescriptorTransformer
                    selected_descs = st.session_state.get("_adv", {}).get("selected_descriptors")
                    _t = SmilesDescriptorTransformer(
                        smiles_col=smiles_col,
                        selected_descriptors=selected_descs if selected_descs else None
                    )
                    X = _t.fit_transform(X)
                except Exception as _ex:
                    st.caption(f"ℹ️ SMILES展開スキップ: {_ex}")

            # --- モデル切替UI ---
            all_model_keys = list((result.model_scores or {}).keys()) if hasattr(result, "model_scores") and result.model_scores else [result.best_model_key]
            if len(all_model_keys) > 1:
                _sel_model = st.pills(
                    "📊 表示モデルを選択",
                    options=all_model_keys,
                    default=result.best_model_key,
                    key="eval_model_select",
                )
                if _sel_model is None:
                    _sel_model = result.best_model_key
            else:
                _sel_model = result.best_model_key

            # --- 選択モデルのpipelineを取得 ---
            _selected_pipeline = None
            if _sel_model == result.best_model_key:
                _selected_pipeline = result.best_pipeline
            elif hasattr(result, "all_pipelines") and result.all_pipelines:
                _selected_pipeline = result.all_pipelines.get(_sel_model)

            # --- 全データ予測 ---
            y_pred_full = None
            if _selected_pipeline is not None:
                try:
                    y_pred_full = _selected_pipeline.predict(X)
                except Exception as e:
                    st.warning(f"⚠️ 全データ予測エラー ({_sel_model}): {e}")

            # --- OOF予測 ---
            y_oof = getattr(result, "oof_predictions", None)
            y_oof_true = getattr(result, "oof_true", None)

            # --- Holdout予測 ---
            y_holdout = getattr(result, "holdout_predictions", None)
            y_holdout_true = getattr(result, "holdout_true", None)

            if result.task == "regression":
                _show_regression_separated(
                    model_name=_sel_model,
                    y_true=y_true,
                    y_pred_full=y_pred_full,
                    y_oof=y_oof,
                    y_oof_true=y_oof_true,
                    y_holdout=y_holdout,
                    y_holdout_true=y_holdout_true,
                    df=df,
                    target_col=target_col,
                )
            else:
                _show_classification_metrics(
                    y_true,
                    y_pred_full if y_pred_full is not None else np.zeros_like(y_true),
                    _selected_pipeline, X,
                )
                if y_oof is not None:
                    st.markdown("#### CV (OOF) での分類結果")
                    _show_classification_metrics(y_oof_true, y_oof, None, None)
        else:
            st.info("データまたは目的変数が設定されていません。")


    # ─── Tab2: 全モデル比較 ──────────────────────────────────────
    with tab2:
        scores = result.model_scores if hasattr(result, "model_scores") else {}
        if scores:
            import plotly.express as px
            rows = []
            details = result.model_details if hasattr(result, "model_details") else {}
            score_col_name = f"CV平均スコア ({result.scoring})"
            
            for key, mean_v in scores.items():
                d = details.get(key, {})
                rows.append({
                    "モデル": key,
                    score_col_name: mean_v or 0,
                    "CV標準偏差": d.get("std", 0),
                    "学習時間(s)": round(d.get("fit_time", 0), 2),
                    "最良": "⭐" if key == result.best_model_key else "",
                })
            cmp_df = pd.DataFrame(rows).sort_values(score_col_name, ascending=False)
            st.dataframe(
                cmp_df.style.background_gradient(
                    subset=[score_col_name], cmap="RdYlGn"
                ),
                use_container_width=True,
            )

            # バーチャート
            fig = px.bar(
                cmp_df, x="モデル", y=score_col_name,
                error_y="CV標準偏差",
                color=score_col_name,
                color_continuous_scale="RdYlGn",
                template="plotly_dark",
                title=f"全モデルCV比較 ({result.scoring})",
                text=cmp_df[score_col_name].map("{:.3f}".format),
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0f0"),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig, use_container_width=True)

            # CVバイアス推定はautoml_page.pyのリーダーボード側で表示
            # （fold_scoresが必要なため、ここでは省略）
        else:
            st.info("全モデルのCVスコアが取得できませんでした。")
            # 最良モデルのみ表示
            st.markdown(f"**最良モデル**: `{result.best_model_key}` — スコア: `{result.best_score:.4f}`")

        # ── Fold別CVスコアの箱ひげ図 / バイオリンプロット ───────────
        st.markdown("---")
        st.markdown("### 🎲 Fold別CVスコアの分布")
        st.caption(
            "各モデルのCV Foldごとのスコア分布。"
            "平均スコアだけでなく、**分布の幅（安定性）**もモデル選択の重要な判断材料です。"
        )
        details = result.model_details if hasattr(result, "model_details") else {}
        fold_data_rows = []
        for mkey, d in details.items():
            fs = d.get("fold_scores", [])
            if fs:
                for fi, sv in enumerate(fs):
                    fold_data_rows.append({
                        "モデル": mkey,
                        "Fold": f"Fold {fi+1}",
                        "スコア": sv,
                    })
        if fold_data_rows:
            import plotly.express as px
            fold_df = pd.DataFrame(fold_data_rows)
            # 平均スコア順でソート
            model_order = fold_df.groupby("モデル")["スコア"].mean().sort_values(ascending=False).index.tolist()
            fig_fold = px.box(
                fold_df, x="モデル", y="スコア",
                color="モデル",
                category_orders={"モデル": model_order},
                title=f"Fold別CVスコア分布 ({result.scoring})",
                template="plotly_dark",
                points="all",
            )
            fig_fold.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0f0"),
                showlegend=False,
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_fold, use_container_width=True)

            # CV安定性の自動判定 + アドバイス
            best_fold = details.get(result.best_model_key, {})
            best_fs = best_fold.get("fold_scores", [])
            if len(best_fs) >= 2:
                _cv_std = float(np.std(best_fs))
                _cv_mean = float(np.mean(best_fs))
                _cv_cv = _cv_std / abs(_cv_mean) if abs(_cv_mean) > 1e-10 else 0
                if _cv_cv < 0.05:
                    st.success(
                        f"✅ **{result.best_model_key}** のCVスコアは非常に安定しています"
                        f"（変動係数 = {_cv_cv:.1%}）。汎化性能の信頼度が高いです。"
                    )
                elif _cv_cv < 0.15:
                    st.info(
                        f"💡 **{result.best_model_key}** のCVスコアの変動係数は {_cv_cv:.1%} です。"
                        f"安定性は許容範囲内ですが、"
                        f"データの増量や特徴量エンジニアリングで改善余地があります。"
                    )
                else:
                    st.warning(
                        f"⚠️ **{result.best_model_key}** のCVスコアが不安定です"
                        f"（変動係数 = {_cv_cv:.1%}）。\n\n"
                        f"考えられる原因: データ不足・Fold間のデータ偏り・リーケージ・モデルの過学習\n"
                        f"対策: CV分割数を増やす・データを追加・正則化を強化・モデルを変更"
                    )
        else:
            st.info("Fold別スコアデータがありません。")

    # ─── Tab3: 学習曲線 ──────────────────────────────────────────
    with tab3:
        if df is not None and target_col:
            st.markdown("### 📉 学習曲線")
            st.caption("サンプル数を増やしたときの訓練・検証スコアの推移")
            cv_folds = st.slider("CV分割数", 2, 10, 5, key="lc_cv")
            scoring_lc = result.scoring if result.task == "regression" else "accuracy"

            if st.button("▶️ 学習曲線を計算", key="btn_lc"):
                with st.spinner("計算中..."):
                    try:
                        from backend.data.benchmark import compute_learning_curve
                        # Pipeline内にSMILES Transformerが有るので元データ（SMILES列含む）をそのまま渡す
                        X = _get_feature_df(df, target_col)
                        y = df[target_col]
                        lc = compute_learning_curve(
                            result.best_pipeline, X, y,
                            scoring=scoring_lc, cv=cv_folds, n_points=8,
                        )
                        st.session_state["_lc_data"] = lc
                    except Exception as e:
                        st.error(f"❌ 学習曲線の計算エラー: {e}")

            lc = st.session_state.get("_lc_data")
            if lc is not None:
                import plotly.graph_objects as go
                ts = lc["train_sizes"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ts, y=lc["train_scores_mean"],
                    mode="lines+markers", name="訓練スコア",
                    line=dict(color="#00d4ff"),
                    error_y=dict(array=lc["train_scores_std"],
                                 color="rgba(0,212,255,0.3)", thickness=1),
                ))
                fig.add_trace(go.Scatter(
                    x=ts, y=lc["val_scores_mean"],
                    mode="lines+markers", name="検証スコア",
                    line=dict(color="#ff6b9d"),
                    error_y=dict(array=lc["val_scores_std"],
                                 color="rgba(255,107,157,0.3)", thickness=1),
                ))
                fig.update_layout(
                    title=f"学習曲線 ({scoring_lc})",
                    xaxis_title="サンプル数", yaxis_title=scoring_lc,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e0e0f0"),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("データが読み込まれていません。")

    # ─── Tab4: 残差/混同行列 ─────────────────────────────────────
    with tab4:
        if df is not None and target_col:
            X = _get_feature_df(df, target_col)
            y_true = df[target_col].values

            # OOF予測の残差分析（推奨）
            y_oof = getattr(result, "oof_predictions", None)
            y_oof_true = getattr(result, "oof_true", None)
            if y_oof is not None and y_oof_true is not None and result.task == "regression":
                st.markdown(
                    '<div class="section-header">🟢 CV Out-of-Fold 残差分析（推奨）</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "OOF予測は各Foldで未学習のデータに対する予測です。"
                    "過学習の影響を受けない、**最も信頼できる残差分析**です。"
                )
                _show_residuals(y_oof_true, y_oof, label="OOF")

                # 残差の統計サマリー
                _res_oof = y_oof - y_oof_true
                _res_cols = st.columns(4)
                with _res_cols[0]:
                    st.metric("平均残差", f"{np.mean(_res_oof):.4f}")
                with _res_cols[1]:
                    st.metric("残差の標準偏差", f"{np.std(_res_oof):.4f}")
                with _res_cols[2]:
                    st.metric("最大過大評価", f"+{np.max(_res_oof):.4f}")
                with _res_cols[3]:
                    st.metric("最大過小評価", f"{np.min(_res_oof):.4f}")

                # バイアス判定
                _mean_res = np.mean(_res_oof)
                if abs(_mean_res) < np.std(_res_oof) * 0.1:
                    st.success("✅ 予測に系統的なバイアスは見られません。")
                else:
                    _dir = "過大評価" if _mean_res > 0 else "過小評価"
                    st.warning(
                        f"⚠️ 平均残差が {_mean_res:.4f} で、{_dir}の傾向があります。\n"
                        f"モデルの選択や特徴量エンジニアリングの見直しを検討してください。"
                    )
                st.markdown("---")

            # 全データ残差（参考）
            try:
                y_pred = result.best_pipeline.predict(X)
                if y_oof is not None and result.task == "regression":
                    st.markdown(
                        '<div class="section-header">🟡 全データ学習の残差（参考）</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption("ℹ️ 訓練データへの予測のため過楽観です。OOF残差を優先してください。")
                if result.task == "regression":
                    _show_residuals(y_true, y_pred, label="全データ")
                else:
                    _show_confusion_matrix(y_true, y_pred)
            except Exception as e:
                err_msg = str(e)
                if "columns are missing" in err_msg:
                    st.warning(
                        "⚠️ コラム不一致エラー。「🔄 AutoML再実行」で再学習してください。"
                    )
                else:
                    st.error(f"❌ エラー: {e}")
        else:
            st.info("データが読み込まれていません。")


# ============================================================
# 内部ヘルパー
# ============================================================

def _show_regression_separated(
    model_name: str = "",
    y_true: np.ndarray = None,
    y_pred_full: "np.ndarray | None" = None,
    y_oof: "np.ndarray | None" = None,
    y_oof_true: "np.ndarray | None" = None,
    y_holdout: "np.ndarray | None" = None,
    y_holdout_true: "np.ndarray | None" = None,
    df: "pd.DataFrame | None" = None,
    target_col: "str | None" = None,
) -> None:
    """全データ / CV-OOF / Holdout を別々のサブタブで表示する。"""
    import streamlit as st
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import plotly.graph_objects as go

    def _make_hover_ids(n: int, df: "pd.DataFrame | None" = None) -> list[str]:
        if df is not None:
            id_candidates = [c for c in df.columns
                             if any(k in c.lower() for k in ["smiles","id","name","cas","compound","mol"])]
            if id_candidates:
                col = id_candidates[0]
                return [f"{col}: {v}" for v in df[col].values[:n]]
        base_idx = list(df.index[:n]) if df is not None else list(range(n))
        return [f"行 #{i}" for i in base_idx]

    def _metrics(yt, yp):
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2   = float(r2_score(yt, yp))
        mae  = float(mean_absolute_error(yt, yp))
        return rmse, r2, mae

    def _pred_vs_actual_chart(yt, yp, color, title, df_ref=None):
        """散布図 + 完全一致ラインを1つのグラフとして返す"""
        n = len(yp)
        hover_ids = _make_hover_ids(n, df_ref)
        customdata = list(zip(
            hover_ids,
            [f"{v:.4f}" for v in yt],
            [f"{v:.4f}" for v in yp],
        ))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yt, y=yp, mode="markers",
            name="データ", marker=dict(color=color, size=6),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "実測値: %{customdata[1]}<br>"
                "予測値: %{customdata[2]}<br>"
                "<extra></extra>"
            ),
        ))
        rng = [min(float(yt.min()), float(yp.min())), max(float(yt.max()), float(yp.max()))]
        fig.add_trace(go.Scatter(
            x=rng, y=rng, mode="lines",
            line=dict(color="#fbbf24", dash="dash"), name="完全一致",
        ))
        fig.update_layout(
            title=title,
            xaxis_title="実測値", yaxis_title="予測値",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0f0"), template="plotly_dark",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        return fig

    # 利用可能なモード一覧
    modes = []
    if y_oof is not None and y_oof_true is not None:
        modes.append(("🟢 CV Out-of-Fold（推奨）", y_oof, y_oof_true, "rgba(80,220,120,0.6)"))
    if y_pred_full is not None:
        modes.append(("🔵 全データ学習（参考）", y_pred_full, y_true, "rgba(0,180,255,0.5)"))
    if y_holdout is not None and y_holdout_true is not None:
        modes.append(("🟠 Holdout テスト", y_holdout, y_holdout_true, "rgba(255,160,60,0.6)"))

    if not modes:
        st.warning("予測値が取得できませんでした。")
        return

    _title_prefix = f"【{model_name}】" if model_name else ""

    # メトリクス総覧
    metric_cols = st.columns(len(modes))
    for mc, (lbl, yp, yt, _) in zip(metric_cols, modes):
        rmse, r2, mae = _metrics(yt, yp)
        with mc:
            st.markdown(
                f'<div class="metric-card">'
                f'<div style="font-size:0.75rem;color:#8888aa;margin-bottom:3px;">{lbl}</div>'
                f'<div class="metric-value" style="font-size:1.05rem;">R²&nbsp;{r2:.4f}</div>'
                f'<div class="metric-label">RMSE {rmse:.4f} | MAE {mae:.4f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # 各モードを別サブタブで表示
    sub_labels = [lbl for lbl, _, _, _ in modes]
    sub_tabs = st.tabs(sub_labels)
    for stab, (lbl, yp, yt, col) in zip(sub_tabs, modes):
        with stab:
            fig = _pred_vs_actual_chart(yt, yp, col, f"{_title_prefix}{lbl} 実測 vs 予測", df)
            st.plotly_chart(fig, use_container_width=True)

            if "全データ" in lbl:
                st.caption("ℹ️ 訓練データへの予測のため過楽観な数値です。CV OOFを優先してください。")
            elif "OOF" in lbl:
                st.caption("✅ 各Foldで未学習のデータに対する予測。最もリアルな汎化精度です。")


# 後方互換性ラッパー
def _show_regression_multi(**kwargs):
    _show_regression_separated(**kwargs)


def _show_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pipeline: Any = None,
    X: Any = None,
) -> None:
    from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    cols = st.columns(4)
    for col, val, label in [
        (cols[0], f"{acc:.4f}", "Accuracy"),
        (cols[1], f"{f1:.4f}", "F1 (weighted)"),
        (cols[2], f"{prec:.4f}", "Precision"),
        (cols[3], f"{rec:.4f}", "Recall"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True
            )
    st.markdown("---")
    st.text(classification_report(y_true, y_pred, zero_division=0))


def _show_residuals(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> None:
    """残差プロット。"""
    import plotly.express as px
    residuals = y_pred - y_true
    res_df = pd.DataFrame({"予測値": y_pred, "残差": residuals, "実測値": y_true})
    _title_prefix = f"[{label}] " if label else ""

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.scatter(
            res_df, x="予測値", y="残差",
            title=f"{_title_prefix}残差プロット (予測値 vs 残差)",
            template="plotly_dark",
            color_discrete_sequence=["#00d4ff"],
            opacity=0.6,
        )
        fig1.add_hline(y=0, line_dash="dash", line_color="#fbbf24")
        # ±2σバンドを追加（異常値検出の目安）
        _std_res = float(np.std(residuals))
        fig1.add_hline(y=2*_std_res, line_dash="dot", line_color="rgba(255,107,157,0.5)",
                       annotation_text="+2σ")
        fig1.add_hline(y=-2*_std_res, line_dash="dot", line_color="rgba(255,107,157,0.5)",
                       annotation_text="-2σ")
        fig1.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0f0"),
        )
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.histogram(
            res_df, x="残差", nbins=30,
            title=f"{_title_prefix}残差の分布",
            template="plotly_dark",
            color_discrete_sequence=["#7b2ff7"],
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0f0"),
        )
        st.plotly_chart(fig2, use_container_width=True)


def _show_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """混同行列の可視化。"""
    from sklearn.metrics import confusion_matrix
    import plotly.figure_factory as ff

    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    labels_str = [str(l) for l in labels]

    fig = ff.create_annotated_heatmap(
        z=cm[::-1],
        x=labels_str,
        y=labels_str[::-1],
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        title="混同行列",
        xaxis_title="予測クラス",
        yaxis_title="実測クラス",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0f0"),
    )
    st.plotly_chart(fig, use_container_width=True)


# type alias for type checking only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any
