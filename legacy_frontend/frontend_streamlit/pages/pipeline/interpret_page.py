"""frontend_streamlit/pages/interpret_page.py - SHAP/SRI解釈ページ"""
from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd


def render() -> None:
    st.markdown("## 💡 モデル解釈・SHAP / SRI分解")

    result = st.session_state.get("automl_result")
    df = st.session_state.get("df")
    target_col = st.session_state.get("target_col")

    if result is None or df is None:
        st.warning("⚠️ まずAutoMLを実行してください。")
        if st.button("🤖 AutoMLへ"):
            st.session_state["page"] = "automl"
            st.rerun()
        return

    from backend.utils.optional_import import is_available
    shap_ok = is_available("shap")

    st.markdown(f"""
<div class="card">
<b>現在のモデル:</b> {result.best_model_key} &nbsp;|&nbsp;
<b>SHAP:</b> {'✅ 利用可能' if shap_ok else '⚠️ pip install shap でインストール'}
</div>""", unsafe_allow_html=True)

    if not shap_ok:
        st.warning("SHAPがインストールされていません。`pip install shap` を実行してください。")

        # SHAPなしでも特徴量重要度は表示
        _show_builtin_feature_importance(result, df, target_col)
        return

    # 除外列・weight列・info列を考慮
    _drop_interp = [target_col]
    _drop_interp.extend(st.session_state.get("col_role_exclude", []))
    _drop_interp.extend(st.session_state.get("col_role_info", []))
    _w_interp = st.session_state.get("col_role_weight")
    if _w_interp: _drop_interp.append(_w_interp)
    _drop_interp = [c for c in _drop_interp if c in df.columns]
    X = df.drop(columns=_drop_interp)
    tab1, tab2, tab3 = st.tabs(["📊 SHAP Summary", "💧 Waterfall", "🔺 SRI分解"])

    with tab1:
        if st.button("🔍 SHAP Summary プロットを計算", key="shap_summary_btn"):
            with st.spinner("SHAP計算中（数秒かかる場合があります）..."):
                try:
                    import shap
                    from backend.interpret.shap_explainer import ShapExplainer
                    explainer = ShapExplainer()
                    X_arr = result.best_pipeline[:-1].transform(X) if hasattr(result.best_pipeline, '__len__') else X.values
                    model = result.best_pipeline[-1]
                    shap_result = explainer.explain(model, X_arr[:min(200, len(X_arr))])
                    st.session_state["shap_result"] = shap_result

                    imp_df = explainer.get_feature_importance_df(shap_result)
                    _plot_importance(imp_df)
                except Exception as e:
                    st.error(f"SHAPエラー: {e}")
        elif "shap_result" in st.session_state:
            from backend.interpret.shap_explainer import ShapExplainer
            explainer = ShapExplainer()
            imp_df = explainer.get_feature_importance_df(st.session_state["shap_result"])
            _plot_importance(imp_df)
        else:
            st.info("「SHAP Summary プロットを計算」ボタンを押してください。")

    with tab2:
        st.info("SHAP Summary 計算後にWaterfallプロットが利用可能になります。")
        if "shap_result" in st.session_state:
            idx = st.slider("表示サンプルインデックス", 0, min(99, len(X) - 1), 0)
            shap_r = st.session_state["shap_result"]
            import plotly.graph_objects as go
            sv = shap_r.shap_values[idx]
            fnames = shap_r.feature_names[:len(sv)]
            sorted_idx = np.argsort(np.abs(sv))[::-1][:15]
            fig = go.Figure(go.Bar(
                x=sv[sorted_idx], y=[fnames[i] for i in sorted_idx],
                orientation="h",
                marker_color=["#00d4ff" if v >= 0 else "#ff6b9d" for v in sv[sorted_idx]],
            ))
            fig.update_layout(
                title=f"サンプル {idx} のSHAP値",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0f0"), height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if "shap_result" in st.session_state:
            if st.button("🔺 SRI分解を実行", key="sri_btn"):
                with st.spinner("SRI分解計算中..."):
                    try:
                        from backend.interpret.sri import SRIDecomposer, plot_sri_heatmap
                        decomposer = SRIDecomposer()
                        sri_result = decomposer.decompose(st.session_state["shap_result"])
                        st.session_state["sri_result"] = sri_result

                        syn, red, ind = sri_result.total_sri
                        c1, c2, c3 = st.columns(3)
                        for col, val, label in [(c1, f"{syn:.3f}", "Synergy合計"),
                                               (c2, f"{red:.3f}", "Redundancy合計"),
                                               (c3, f"{ind:.3f}", "Independence合計")]:
                            with col:
                                st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

                        summary = sri_result.summary_df()
                        st.dataframe(summary[["feature", "synergy", "redundancy", "independence"]].head(20),
                                     use_container_width=True)
                    except Exception as e:
                        st.error(f"SRI分解エラー: {e}")
        else:
            st.info("先に SHAP Summary を計算してください。")


def _show_builtin_feature_importance(result, df, target_col):
    """SHAPなしでモデル組み込みの特徴量重要度を表示する。"""
    model = result.best_pipeline[-1]
    if hasattr(model, "feature_importances_"):
        _drop_bi = [target_col]
        _drop_bi.extend(st.session_state.get("col_role_exclude", []))
        _drop_bi.extend(st.session_state.get("col_role_info", []))
        _w_bi = st.session_state.get("col_role_weight")
        if _w_bi: _drop_bi.append(_w_bi)
        _drop_bi = [c for c in _drop_bi if c in df.columns]
        X = df.drop(columns=_drop_bi)
        try:
            fnames = result.best_pipeline[:-1].get_feature_names_out()
        except Exception:
            fnames = [f"f{i}" for i in range(len(model.feature_importances_))]
        imp = model.feature_importances_
        _plot_importance(pd.DataFrame({"feature": list(fnames)[:len(imp)], "importance": imp}))
    elif hasattr(model, "coef_"):
        coef = model.coef_.ravel() if hasattr(model.coef_, "ravel") else model.coef_
        try:
            fnames = result.best_pipeline[:-1].get_feature_names_out()
        except Exception:
            fnames = [f"f{i}" for i in range(len(coef))]
        _plot_importance(pd.DataFrame({"feature": list(fnames)[:len(coef)], "importance": np.abs(coef)}))
    else:
        st.info("このモデルは組み込みの特徴量重要度を持ちません。SHAPをインストールしてください。")


def _plot_importance(imp_df: pd.DataFrame) -> None:
    import plotly.graph_objects as go
    top = imp_df.head(20).sort_values("importance")
    fig = go.Figure(go.Bar(
        x=top["importance"], y=top["feature"],
        orientation="h", marker_color="#7b2ff7",
        text=top["importance"].round(4), textposition="outside",
    ))
    fig.update_layout(
        title="特徴量重要度 (Top 20)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0f0"), height=450,
        margin=dict(l=150),
    )
    st.plotly_chart(fig, use_container_width=True)
