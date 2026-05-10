"""
interpretability_ui.py
=======================
モデル解釈性UIコンポーネント。
7タブ構成:
  1. 📊 重要度概観     - Feature Importance / 回帰係数 / Permutation Importance
  2. 🐝 SHAP Summary  - Beeswarm / Bar プロット
  3. 💧 SHAP 個別予測  - Waterfall / Force Plot
  4. 🔗 SHAP Dependence - 特徴量 vs SHAP値
  5. 🗺️ SHAP Heatmap  - 全サンプル × SHAP値
  6. 🔀 Shapley Interactions (shapiq) - 1次・2次 SI分解
  7. 🌍 SAGE           - グローバル重要度 (on-demand)
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────
# SHAP Explainer の自動選択
# ─────────────────────────────────────────────
def _get_shap_explainer(model, X_bg):
    """TreeExplainer → LinearExplainer → KernelExplainer の順で試す。"""
    import shap
    # パイプラインから最終推定器を取り出す
    est = model
    if hasattr(model, "steps"):
        est = model.steps[-1][1]

    try:
        exp = shap.TreeExplainer(est, data=X_bg, feature_perturbation="interventional")
        _ = exp.shap_values(X_bg.iloc[:1] if hasattr(X_bg, "iloc") else X_bg[:1])
        return exp, "tree"
    except Exception:
        pass

    try:
        exp = shap.LinearExplainer(est, X_bg)
        _ = exp.shap_values(X_bg.iloc[:1] if hasattr(X_bg, "iloc") else X_bg[:1])
        return exp, "linear"
    except Exception:
        pass

    bg = shap.sample(X_bg, min(50, len(X_bg)))
    exp = shap.KernelExplainer(model.predict, bg)
    return exp, "kernel"


def _compute_shap_values(model, X_input, max_samples: int = 300):
    """SHAP値を計算する（キャッシュをsession_stateで手動管理）。"""
    import shap
    # session_state キャッシュ
    cache_key = f"_shap_cache_{id(model)}_{len(X_input)}_{max_samples}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    X = X_input.iloc[:max_samples] if hasattr(X_input, "iloc") else X_input[:max_samples]
    explainer, kind = _get_shap_explainer(model, X)
    sv = explainer.shap_values(X)
    # 分類で3次元 → 最後クラスを使う
    if isinstance(sv, list):
        sv = sv[-1]
    if hasattr(sv, "values"):   # shap.Explanation
        sv = sv.values
    # expected_value を保存
    ev = getattr(explainer, "expected_value", None)
    if isinstance(ev, (list, np.ndarray)):
        ev = ev[-1] if len(ev) > 1 else ev[0]
    result = (sv, X, kind, ev)
    st.session_state[cache_key] = result
    return result


# ─────────────────────────────────────────────
# タブ1: 重要度概観
# ─────────────────────────────────────────────
def _tab_importance(model, X, y, feature_names):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    est = model.steps[-1][1] if hasattr(model, "steps") else model
    has_fi  = hasattr(est, "feature_importances_")
    has_coef = hasattr(est, "coef_")

    col_fi, col_perm = st.columns(2)

    # ── Feature Importances (tree系) ──
    with col_fi:
        if has_fi:
            st.markdown("#### 🌲 Feature Importances")
            fi = est.feature_importances_
            df_fi = pd.DataFrame({"Feature": feature_names, "Importance": fi})
            df_fi = df_fi.sort_values("Importance", ascending=True).tail(30)
            fig = go.Figure(go.Bar(
                x=df_fi["Importance"], y=df_fi["Feature"],
                orientation="h", marker_color="#4c9be8"
            ))
            fig.update_layout(height=max(300, len(df_fi)*22), margin=dict(l=0,r=10,t=20,b=20),
                              yaxis=dict(tickfont=dict(size=11)))
            st.plotly_chart(fig, use_container_width=True)
        elif has_coef:
            st.markdown("#### 📏 回帰係数 (coef)")
            coef = est.coef_.flatten()[:len(feature_names)]
            df_c = pd.DataFrame({"Feature": feature_names[:len(coef)], "Coefficient": coef})
            df_c = df_c.reindex(df_c["Coefficient"].abs().sort_values().index).tail(30)
            colors = ["#f87171" if v < 0 else "#4ade80" for v in df_c["Coefficient"]]
            fig = go.Figure(go.Bar(
                x=df_c["Coefficient"], y=df_c["Feature"],
                orientation="h", marker_color=colors
            ))
            fig.update_layout(height=max(300, len(df_c)*22), margin=dict(l=0,r=10,t=20,b=20))
            fig.add_vline(x=0, line_color="#888", line_width=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("💡 このモデルは `feature_importances_` も `coef_` も持っていません。SHAP タブをご利用ください。")

    # ── Permutation Importance ──
    with col_perm:
        st.markdown("#### 🔀 Permutation Importance")
        n_pi = st.slider("繰り返し数", 3, 20, 5, key="pi_repeats")
        if st.button("▶ 計算", key="calc_pi"):
            from sklearn.inspection import permutation_importance
            with st.spinner("計算中..."):
                r = permutation_importance(model, X, y, n_repeats=n_pi, random_state=42, n_jobs=-1)
            df_pi = pd.DataFrame({
                "Feature": feature_names,
                "Mean": r.importances_mean,
                "Std": r.importances_std
            }).sort_values("Mean", ascending=True).tail(30)
            fig2 = go.Figure(go.Bar(
                x=df_pi["Mean"], y=df_pi["Feature"],
                orientation="h", marker_color="#a78bfa",
                error_x=dict(type="data", array=df_pi["Std"], visible=True)
            ))
            fig2.update_layout(height=max(300, len(df_pi)*22),
                               margin=dict(l=0,r=10,t=20,b=20))
            st.plotly_chart(fig2, use_container_width=True)
            st.session_state["_pi_result"] = df_pi
        elif st.session_state.get("_pi_result") is not None:
            df_pi = st.session_state["_pi_result"]
            fig2 = go.Figure(go.Bar(
                x=df_pi["Mean"], y=df_pi["Feature"],
                orientation="h", marker_color="#a78bfa",
                error_x=dict(type="data", array=df_pi["Std"], visible=True)
            ))
            fig2.update_layout(height=max(300, len(df_pi)*22),
                                margin=dict(l=0,r=10,t=20,b=20))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("「▶ 計算」ボタンで Permutation Importance を計算します。")

    # ── PDP ──
    st.markdown("---")
    st.markdown("#### 📊 Partial Dependence Plot (PDP / ICE)")
    pdp_feat = st.selectbox("特徴量を選択", feature_names, key="pdp_feat_sel")
    pdp_kind = st.radio("種別", ["average (PDP)", "individual (ICE)", "both"], horizontal=True, key="pdp_kind")
    kind_map = {"average (PDP)": "average", "individual (ICE)": "individual", "both": "both"}
    if st.button("▶ PDP を描画", key="calc_pdp"):
        from sklearn.inspection import PartialDependenceDisplay
        import matplotlib.pyplot as plt
        feat_idx = list(feature_names).index(pdp_feat)
        with st.spinner("計算中..."):
            fig_pdp, ax = plt.subplots(figsize=(8, 4))
            try:
                disp = PartialDependenceDisplay.from_estimator(
                    model, X, features=[feat_idx],
                    kind=kind_map[pdp_kind], ax=ax,
                    subsample=min(200, len(X)), random_state=42
                )
                fig_pdp.tight_layout()
                st.pyplot(fig_pdp, use_container_width=True)
            except Exception as e:
                st.error(f"PDP計算エラー: {e}")
            finally:
                plt.close(fig_pdp)


# ─────────────────────────────────────────────
# タブ2: SHAP Summary
# ─────────────────────────────────────────────
def _tab_shap_summary(model, X, feature_names):
    import shap
    import matplotlib.pyplot as plt

    max_s = st.slider("使用サンプル数", 30, min(500, len(X)), min(200, len(X)), step=10, key="shap_sum_n")
    plot_type = st.radio("プロット種別", ["🐝 Beeswarm", "📊 Bar (mean |SHAP|)"], horizontal=True, key="shap_sum_type")

    try:
        with st.spinner("SHAP値を計算中..."):
            sv, X_sub, kind, ev = _compute_shap_values(model, X, max_samples=max_s)

        # feature_names と sv の列数を揃える
        n_feat = sv.shape[1] if sv.ndim > 1 else 1
        if len(feature_names) != n_feat:
            feature_names = list(feature_names[:n_feat]) if len(feature_names) > n_feat else \
                list(feature_names) + [f"feat_{i}" for i in range(len(feature_names), n_feat)]

        st.caption(f"Explainer: `{kind}` | サンプル数: {len(X_sub)} | 特徴量数: {n_feat}")

        plt.rcParams["figure.facecolor"] = "#0e1117"
        plt.rcParams["axes.facecolor"] = "#0e1117"
        plt.rcParams["text.color"] = "white"
        plt.rcParams["axes.labelcolor"] = "white"
        plt.rcParams["xtick.color"] = "white"
        plt.rcParams["ytick.color"] = "white"

        X_data = X_sub.values if hasattr(X_sub, "values") else X_sub
        # データ形状をSHAP値に合わせる
        if X_data.shape[1] != n_feat:
            X_data = X_data[:, :n_feat]

        exp_obj = shap.Explanation(
            values=sv,
            data=X_data,
            feature_names=list(feature_names)
        )

        if "Beeswarm" in plot_type:
            shap.plots.beeswarm(exp_obj, max_display=30, show=False)
        else:
            shap.plots.bar(exp_obj, max_display=30, show=False)

        st.pyplot(plt.gcf(), use_container_width=True)
        plt.close("all")
    except Exception as e:
        st.error(f"SHAP Summary エラー: {e}")
        import traceback
        with st.expander("🛠️ エラー詳細"):
            st.code(traceback.format_exc())


# ─────────────────────────────────────────────
# タブ3: SHAP 個別予測
# ─────────────────────────────────────────────
def _tab_shap_individual(model, X, feature_names):
    import shap
    import matplotlib.pyplot as plt

    try:
        max_s = min(300, len(X))
        sv, X_sub, kind, ev = _compute_shap_values(model, X, max_samples=max_s)

        # feature_names と sv の列数を揃える
        n_feat = sv.shape[1] if sv.ndim > 1 else len(sv[0]) if sv.ndim == 2 else 1
        if len(feature_names) != n_feat:
            feature_names = list(feature_names[:n_feat]) if len(feature_names) > n_feat else \
                list(feature_names) + [f"feat_{i}" for i in range(len(feature_names), n_feat)]

        sample_idx = st.slider("サンプルを選択", 0, len(X_sub) - 1, 0, key="shap_ind_idx")
        plot_type = st.radio("プロット種別", ["💧 Waterfall", "⚡ Force Plot"], horizontal=True, key="shap_ind_type")

        # base_values を安全に取得（キャッシュから取得したevを使用）
        if ev is not None:
            base_val = float(ev) if np.isscalar(ev) or (hasattr(ev, 'ndim') and ev.ndim == 0) else float(ev)
        else:
            base_val = float(sv.mean())

        sample_data = X_sub.iloc[sample_idx].values if hasattr(X_sub, "iloc") else X_sub[sample_idx]
        if len(sample_data) != n_feat:
            sample_data = sample_data[:n_feat]

        exp_obj = shap.Explanation(
            values=sv[sample_idx][:n_feat] if len(sv[sample_idx]) > n_feat else sv[sample_idx],
            base_values=base_val,
            data=sample_data,
            feature_names=list(feature_names)
        )

        # 予測値を表示
        try:
            pred_val = model.predict(X_sub.iloc[[sample_idx]] if hasattr(X_sub, "iloc") else X_sub[[sample_idx]])[0]
            st.metric("予測値", f"{pred_val:.4f}")
        except Exception:
            pass

        plt.rcParams["figure.facecolor"] = "#0e1117"
        plt.rcParams["axes.facecolor"] = "#0e1117"
        plt.rcParams["text.color"] = "white"

        if "Waterfall" in plot_type:
            shap.plots.waterfall(exp_obj, max_display=20, show=False)
        else:
            shap.plots.force(exp_obj, matplotlib=True, show=False)

        st.pyplot(plt.gcf(), use_container_width=True)
        plt.close("all")
    except Exception as e:
        st.error(f"SHAP 個別予測エラー: {e}")
        import traceback
        with st.expander("🛠️ エラー詳細"):
            st.code(traceback.format_exc())


# ─────────────────────────────────────────────
# タブ4: SHAP Dependence
# ─────────────────────────────────────────────
def _tab_shap_dependence(model, X, feature_names):
    import shap
    import plotly.graph_objects as go

    try:
        max_s = min(300, len(X))
        sv, X_sub, kind, ev = _compute_shap_values(model, X, max_samples=max_s)

        feat1 = st.selectbox("主特徴量", feature_names, key="dep_feat1")
        feat2 = st.selectbox("着色特徴量（交互作用）", ["自動"] + list(feature_names), key="dep_feat2")

        f1_idx = list(feature_names).index(feat1)
        X_arr = X_sub.values if hasattr(X_sub, "values") else X_sub
        shap_vals = sv[:, f1_idx]
        feat_vals = X_arr[:, f1_idx]

        if feat2 == "自動":
            # 相互作用が最も大きい特徴量を自動選択
            if sv.ndim > 1 and sv.shape[1] > 1:
                corrs = [abs(np.corrcoef(sv[:, j], feat_vals)[0, 1]) for j in range(sv.shape[1])]
                corrs[f1_idx] = -1
                color_idx = int(np.argmax(corrs))
                color_label = feature_names[color_idx]
                color_vals = X_arr[:, color_idx]
            else:
                color_vals = feat_vals
                color_label = feat1
        else:
            color_idx = list(feature_names).index(feat2)
            color_label = feat2
            color_vals = X_arr[:, color_idx]

        fig = go.Figure(go.Scatter(
            x=feat_vals, y=shap_vals,
            mode="markers",
            marker=dict(color=color_vals, colorscale="Viridis", size=6,
                        colorbar=dict(title=color_label), showscale=True),
            hovertemplate=f"{feat1}: %{{x:.3f}}<br>SHAP: %{{y:.3f}}<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title=feat1, yaxis_title=f"SHAP value of {feat1}",
            height=420, margin=dict(l=20, r=20, t=30, b=30),
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="white")
        )
        fig.add_hline(y=0, line_color="#555", line_width=1)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"SHAP Dependence エラー: {e}")
        import traceback
        with st.expander("🛠️ エラー詳細"):
            st.code(traceback.format_exc())


# ─────────────────────────────────────────────
# タブ5: SHAP Heatmap
# ─────────────────────────────────────────────
def _tab_shap_heatmap(model, X, feature_names):
    import shap
    import plotly.graph_objects as go

    try:
        max_s = min(200, len(X))
        sv, X_sub, kind, ev = _compute_shap_values(model, X, max_samples=max_s)

        top_n = st.slider("表示する特徴量数（重要度上位）", 5, min(50, sv.shape[1] if sv.ndim > 1 else 10), 20,
                           key="hm_topn")
        mean_abs = np.abs(sv).mean(axis=0) if sv.ndim > 1 else np.abs(sv)
        top_idx = np.argsort(mean_abs)[-top_n:][::-1]
        sv_top = sv[:, top_idx] if sv.ndim > 1 else sv.reshape(-1, 1)
        feat_top = [feature_names[i] for i in top_idx]

        # サンプルを mean|SHAP| でソート
        row_order = np.argsort(np.abs(sv_top).mean(axis=1))

        fig = go.Figure(go.Heatmap(
            z=sv_top[row_order].T,
            x=[f"S{i}" for i in range(len(row_order))],
            y=feat_top,
            colorscale="RdBu_r", zmid=0,
            colorbar=dict(title="SHAP値"),
            hovertemplate="Sample: %{x}<br>Feature: %{y}<br>SHAP: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            height=max(400, top_n * 20),
            margin=dict(l=10, r=10, t=30, b=30),
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="white"),
            xaxis=dict(showticklabels=False),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"SHAP Heatmap エラー: {e}")
        import traceback
        with st.expander("🛠️ エラー詳細"):
            st.code(traceback.format_exc())


# ─────────────────────────────────────────────
# タブ6: Shapley Interactions (shapiq)
# ─────────────────────────────────────────────
def _tab_shapiq(model, X, feature_names):
    try:
        import shapiq
    except ImportError:
        st.warning("⚠️ `shapiq` がインストールされていません。`pip install shapiq` を実行してください。")
        st.code("pip install shapiq")
        return

    import plotly.graph_objects as go

    st.markdown("""
**Shapley Interactions (SI)** は SHAP 値を高次の交互作用成分に分解します。
- **1次 (φ)**: 各特徴量の単独効果（通常のSHAP値に相当）
- **2次 (φᵢⱼ)**: 特徴量ペアの交互作用効果（基本効果に帰せない余剰効果）
    """)

    max_s = st.slider("サンプル数", 20, min(100, len(X)), 30, key="shapiq_n")
    order = st.radio("交互作用次数", ["1次のみ", "1次＋2次"], horizontal=True, key="shapiq_order")
    max_feat = st.slider("上位特徴量数", 5, min(20, len(feature_names)), 10, key="shapiq_topf")

    if not st.button("▶ Shapley Interactions を計算", key="calc_shapiq"):
        st.caption("計算には時間がかかる場合があります（サンプル数×特徴量数に比例）。")
        return

    X_sub = X.iloc[:max_s] if hasattr(X, "iloc") else X[:max_s]
    X_arr = X_sub.values if hasattr(X_sub, "values") else X_sub

    with st.spinner("Shapley Interactions 計算中... しばらくお待ちください。"):
        try:
            def pred_fn(x):
                return model.predict(pd.DataFrame(x, columns=feature_names))

            n_feat = X_arr.shape[1]
            imputer = shapiq.TabularExplainer(
                model=pred_fn,
                data=X_arr,
                index="k-SII",
                max_order=1 if "1次のみ" in order else 2,
                sample_size=max_s,
            )
            # 全サンプルの平均SI
            all_sv = []
            for i in range(min(20, len(X_arr))):
                try:
                    sv_i = imputer.explain(X_arr[[i]])
                    all_sv.append(sv_i)
                except Exception:
                    continue

            if not all_sv:
                st.error("Shapley Interactions の計算に失敗しました。")
                return

            # 1次効果（対角成分相当）を取り出す
            import shapiq as shapiq_mod
            # shapiq は InteractionValues オブジェクトを返す
            first_sv = all_sv[0]
            st.success(f"✅ 計算完了（{len(all_sv)} サンプル）")
            st.write(first_sv)

        except Exception as e:
            st.error(f"計算エラー: {e}")
            with st.expander("🛠️ エラー詳細（開発者向け）"):
                import traceback
                st.code(traceback.format_exc(), language="python")

    # ─ 1次効果を棒グラフで表示（SHAPキャッシュから代用）
    st.markdown("---")
    st.markdown("#### 1次効果（単独寄与）")
    sv_base, X_s, _, _ = _compute_shap_values(model, X, max_samples=min(200, len(X)))
    mean_abs = np.abs(sv_base).mean(axis=0) if sv_base.ndim > 1 else np.abs(sv_base)
    top_idx = np.argsort(mean_abs)[-max_feat:][::-1]
    df_si1 = pd.DataFrame({
        "Feature": [feature_names[i] for i in top_idx],
        "Mean |SHAP|": mean_abs[top_idx]
    })
    fig = go.Figure(go.Bar(x=df_si1["Mean |SHAP|"], y=df_si1["Feature"],
                            orientation="h", marker_color="#818cf8"))
    fig.update_layout(height=max(250, max_feat * 22), margin=dict(l=0, r=10, t=20, b=20),
                       paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True)

    # ─ 2次交互作用（近似：SHAP * SHAP の相関で代用）
    if "2次" in order and sv_base.ndim > 1:
        st.markdown("#### 2次交互作用（ペア寄与、近似）")
        n_show = min(max_feat, sv_base.shape[1])
        top5 = list(np.argsort(mean_abs)[-n_show:][::-1])
        interaction_matrix = np.zeros((n_show, n_show))
        for a in range(n_show):
            for b in range(n_show):
                if a != b:
                    ia, ib = top5[a], top5[b]
                    interaction_matrix[a, b] = abs(np.corrcoef(sv_base[:, ia], sv_base[:, ib])[0, 1])

        feat_labels = [feature_names[i] for i in top5]
        fig2 = go.Figure(go.Heatmap(
            z=interaction_matrix, x=feat_labels, y=feat_labels,
            colorscale="Purples", zmin=0, zmax=1,
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
            colorbar=dict(title="相関(近似)")
        ))
        fig2.update_layout(height=max(350, n_show * 30),
                            margin=dict(l=10, r=10, t=30, b=30),
                            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                            font=dict(color="white"))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("※ 2次交互作用は SHAP値間の相関で近似しています。完全なSI計算にはshapiqの計算ボタンをご利用ください。")


# ─────────────────────────────────────────────
# タブ7: SAGE
# ─────────────────────────────────────────────
def _tab_sage(model, X, y, feature_names):
    try:
        import sage
    except ImportError:
        st.warning("⚠️ `sage-importance` がインストールされていません。`pip install sage-importance` を実行してください。")
        st.code("pip install sage-importance")
        return

    import plotly.graph_objects as go

    st.markdown("""
**SAGE (Shapley Additive Global importancE)** はデータセット全体を使った大局的な特徴量重要度を算出します。
SHAP が個別予測の説明なのに対し、SAGE は「この特徴量がなければモデル全体の性能がどれだけ落ちるか」を測ります。
    """)
    st.warning("⚠️ SAGE は計算コストが高いです。サンプル数を少なめに設定して実行してください。")

    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider("サンプル数", 50, min(500, len(X)), min(200, len(X)), step=50, key="sage_n")
    with col2:
        n_permutations = st.slider("パーミュテーション数", 256, 1024, 512, step=64, key="sage_perm")

    if not st.button("▶ SAGE を計算", key="calc_sage"):
        st.caption(f"計算時間の目安: {n_samples}サンプル × {n_permutations}パーミュテーションで数十秒〜数分。")
        return

    X_sub = X.iloc[:n_samples] if hasattr(X, "iloc") else X[:n_samples]
    y_sub = y.iloc[:n_samples] if hasattr(y, "iloc") else y[:n_samples]
    X_arr = X_sub.values if hasattr(X_sub, "values") else X_sub
    y_arr = y_sub.values if hasattr(y_sub, "values") else y_sub

    with st.spinner(f"SAGE 計算中... ({n_samples}サンプル)"):
        try:
            # タスク判定
            is_regression = len(np.unique(y_arr)) > 10
            if is_regression:
                loss_fn = "mse"
                pred_fn = lambda x: model.predict(pd.DataFrame(x, columns=feature_names))
            else:
                loss_fn = "cross entropy"
                pred_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_names))

            imputer = sage.MarginalImputer(pred_fn, X_arr)
            estimator = sage.PermutationEstimator(imputer, loss_fn)
            sage_values = estimator(X_arr, y_arr, n_permutations=n_permutations)

            df_sage = pd.DataFrame({
                "Feature": feature_names,
                "SAGE": sage_values.values,
                "StdErr": sage_values.std
            }).sort_values("SAGE", ascending=True).tail(30)

            fig = go.Figure(go.Bar(
                x=df_sage["SAGE"],
                y=df_sage["Feature"],
                orientation="h",
                marker_color="#f472b6",
                error_x=dict(type="data", array=df_sage["StdErr"], visible=True)
            ))
            fig.update_layout(
                xaxis_title="SAGE 値（モデル性能への寄与）",
                height=max(300, len(df_sage) * 22),
                margin=dict(l=0, r=10, t=20, b=20),
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="white")
            )
            fig.add_vline(x=0, line_color="#888", line_width=1)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                df_sage[["Feature", "SAGE", "StdErr"]].sort_values("SAGE", ascending=False)
                    .rename(columns={"SAGE": "SAGE値", "StdErr": "標準誤差"}),
                use_container_width=True, hide_index=True
            )
        except Exception as e:
            st.error(f"SAGE 計算エラー: {e}")
            with st.expander("🛠️ エラー詳細（開発者向け）"):
                import traceback
                st.code(traceback.format_exc(), language="python")


# ─────────────────────────────────────────────
# タブ8: imodels ルール表示
# ─────────────────────────────────────────────
def _tab_rules(model, feature_names):
    """
    imodelsルールモデル（RuleFit / SkopeRules / FIGS）の解釈を表示する。
    QSARで重要な「どの条件でどの予測値になるか」を直感的に見せる。
    """
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    cls_name = type(est).__name__

    st.markdown(f"**モデル種別**: `{cls_name}`")

    # ── RuleFit ルール ──────────────────────────────────────
    if hasattr(est, "rules_") and hasattr(getattr(est, "rules_", None), "to_dataframe"):
        try:
            df_rules = est.rules_.to_dataframe()
            st.markdown("#### 📜 RuleFit ルール一覧")
            st.caption(f"全{len(df_rules)}件のルール読み込み済み")
            st.dataframe(df_rules.sort_values("importance", ascending=False).head(50),
                         use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"RuleFitルール表示エラー: {e}")

    # ── SkopeRules ──────────────────────────────────────────
    elif hasattr(est, "rules_") and isinstance(getattr(est, "rules_", None), list):
        st.markdown("#### 📜 SkopeRules ルール一覧")
        rules = est.rules_
        if rules:
            rows = []
            for r in rules[:50]:
                if hasattr(r, "args"):
                    rows.append({
                        "rule": r.args[0],
                        "precision": getattr(r, "score", ""),
                        "support": getattr(r, "support", "")
                    })
                else:
                    rows.append({"rule": str(r)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("ルールが見つかりませんでした。")

    # ── FIGS tree ──────────────────────────────────────────
    elif hasattr(est, "trees_"):
        st.markdown("#### 🌳 FIGS Trees")
        try:
            for i, tree in enumerate(est.trees_):
                with st.expander(f"Tree {i+1}"):
                    st.text(str(tree))
        except Exception as e:
            st.warning(f"FIGS表示エラー: {e}")

    # ── 線形モデルの係数 ────────────────────────────────────
    elif hasattr(est, "coef_"):
        st.markdown("#### 📏 線形モデル係数")
        coef = est.coef_.ravel()
        df_c = pd.DataFrame({"Feature": feature_names[:len(coef)], "Coefficient": coef})
        df_c = df_c.sort_values("Coefficient", key=abs, ascending=False)
        st.dataframe(df_c, use_container_width=True, hide_index=True)

    else:
        st.info(
            "ℹ️ このモデルはルール/係数表示に対応していません。\n"
            "RuleFit・SkopeRules・FIGSモデルで選択するとそのルールが表示されます。"
        )


# ─────────────────────────────────────────────
# メインエントリーポイント
# ─────────────────────────────────────────────
def render_interpretability_ui(
    model,
    X: pd.DataFrame,
    y,
    feature_names: list[str],
    task: str = "regression",
):
    """
    モデル解釈性UIをレンダリングする。

    Parameters
    ----------
    model       : 学習済みモデル（sklearn Pipeline 可）
    X           : 特徴量 DataFrame
    y           : ターゲット Series/array
    feature_names: 特徴量名リスト
    task        : 'regression' or 'classification'
    """

    if model is None or X is None or len(X) == 0:
        st.info("💡 解析結果が利用できません。")
        return

    # feature_names を X の列名に揃える
    if hasattr(X, "columns"):
        feature_names = list(X.columns)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 重要度概観",
        "🐝 SHAP Summary",
        "💧 SHAP 個別予測",
        "🔗 SHAP Dependence",
        "🗺️ SHAP Heatmap",
        "🔀 SI分解 (shapiq)",
        "🌍 SAGE",
        "📜 ルール表示 (imodels)",
    ])

    # 適用可能条件の表示
    try:
        import shap
        shap_ok = True
    except ImportError:
        shap_ok = False

    if not shap_ok:
        for t in [tab2, tab3, tab4, tab5]:
            with t:
                st.error("`shap` がインストールされていません。`pip install shap` を実行してください。")
        with tab1:
            _tab_importance(model, X, y, feature_names)
        with tab6:
            _tab_shapiq(model, X, feature_names)
        with tab7:
            _tab_sage(model, X, y, feature_names)
        return

    with tab1:
        _tab_importance(model, X, y, feature_names)

    with tab2:
        _tab_shap_summary(model, X, feature_names)

    with tab3:
        _tab_shap_individual(model, X, feature_names)

    with tab4:
        _tab_shap_dependence(model, X, feature_names)

    with tab5:
        _tab_shap_heatmap(model, X, feature_names)

    with tab6:
        _tab_shapiq(model, X, feature_names)

    with tab7:
        _tab_sage(model, X, y, feature_names)

    with tab8:
        _tab_rules(model, feature_names)
