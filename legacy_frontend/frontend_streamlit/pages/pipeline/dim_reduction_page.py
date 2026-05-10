"""
frontend_streamlit/pages/dim_reduction_page.py
次元削減（PCA / t-SNE / UMAP）可視化ページ。
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
import numpy as np


def _get_smiles_for_index(idx, df_raw) -> list[str]:
    """インデックスに対応するSMILES列を取得する。SMILES列が無ければ空のリストを返す。"""
    smiles_col = st.session_state.get("smiles_col")
    if smiles_col and df_raw is not None and smiles_col in df_raw.columns:
        return [str(df_raw.loc[i, smiles_col]) if i in df_raw.index else "" for i in idx]
    return [""] * len(idx)


def _apply_smiles_hover(fig, smiles_list: list[str]) -> None:
    """Plotly図にSMILESホバー（2D構造ポップアップ）を付与する。"""
    try:
        from frontend_streamlit.components.smiles_hover import add_smiles_hover_to_plotly
        if any(s for s in smiles_list):
            add_smiles_hover_to_plotly(fig, smiles_list)
    except Exception:
        pass  # ホバー機能不可でも散布図は表示する


def render() -> None:
    st.markdown("## 📐 次元削減")

    df_raw = st.session_state.get("df")
    if df_raw is None:
        st.warning("⚠️ まずデータを読み込んでください。")
        if st.button("📂 データ読み込みへ"):
            st.session_state["page"] = "data_load"
            st.rerun()
        return

    # SMILES記述子が事前計算されている場合は結合する
    df = df_raw.copy()
    precalc_df = st.session_state.get("precalc_smiles_df")
    if precalc_df is not None and not precalc_df.empty:
        # 重複列名（元のdfにある列名）は除外して結合
        cols_to_add = precalc_df.columns.difference(df.columns)
        df = pd.concat([df, precalc_df[cols_to_add]], axis=1)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        st.error("次元削減には2列以上の数値列が必要です。")
        return

    target_col = st.session_state.get("target_col")

    # ─── サイドパネル設定 ─────────────────────────────────────────────
    with st.expander("⚙️ 設定", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox("手法", ["PCA", "t-SNE", "UMAP"], key="dr_method")
            use_scale = st.checkbox("事前スケーリング (StandardScaler)", value=True)
            color_by = st.selectbox(
                "色付け列",
                ["なし"] + df.columns.tolist(),
                index=0 if not target_col else (df.columns.tolist().index(target_col) + 1 if target_col in df.columns else 0),
                key="dr_color",
            )
        with col2:
            n_components = st.slider("次元数", 2, 3, 2, key="dr_n_comp")
            if method == "t-SNE":
                perplexity = st.slider("パープレキシティ", 5.0, 50.0, 30.0, 1.0)
                n_iter = st.number_input("最大反復数", 250, 5000, 1000, 250)
            elif method == "UMAP":
                n_neighbors = st.slider("近傍数 (n_neighbors)", 2, 100, 15)
                min_dist = st.slider("最小距離 (min_dist)", 0.0, 1.0, 0.1, 0.05)

        exclude_cols = st.multiselect(
            "除外する列", numeric_cols,
            default=[target_col] if target_col and target_col in numeric_cols else []
        )

    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    if len(feature_cols) < 2:
        st.warning("特徴量列が2列以上必要です。除外列を減らしてください。")
        return

    st.info(f"✅ 使用する特徴量: **{len(feature_cols)}列** / サンプル数: **{len(df):,}件**")

    # ─── 実行ボタン ─────────────────────────────────────────────────
    if st.button(f"▶️ {method} を実行", type="primary"):
        with st.spinner(f"{method} を計算中..."):
            try:
                from backend.data.dim_reduction import DimReductionConfig, DimReducer

                sub_df = df[feature_cols].dropna()
                idx = sub_df.index

                if method == "PCA":
                    # PCAは全次元計算を試みる (Hardening 要件: 全次元保持)
                    cfg = DimReductionConfig(method="pca", n_components=len(feature_cols), scale=use_scale)
                elif method == "t-SNE":
                    safe_perp = min(perplexity, (len(sub_df) - 1) / 3)
                    cfg = DimReductionConfig(
                        method="tsne", n_components=n_components, scale=use_scale,
                        perplexity=safe_perp, tsne_max_iter=int(n_iter),
                    )
                else:  # UMAP
                    cfg = DimReductionConfig(
                        method="umap", n_components=n_components, scale=use_scale,
                        n_neighbors=n_neighbors, min_dist=min_dist,
                    )

                reducer = DimReducer(cfg)
                embedding = reducer.fit_transform(sub_df)

                # 表示用ラベル (n_components に基づく)
                dim_labels = (
                    [f"PC{i+1}" for i in range(len(feature_cols))] if method == "PCA"
                    else [f"{method}{i+1}" for i in range(n_components)]
                )
                emb_df = pd.DataFrame(embedding, columns=dim_labels, index=idx)

                # color列をマージ
                if color_by != "なし" and color_by in df.columns:
                    emb_df["_color"] = df.loc[idx, color_by].values
                    color_col_name = "_color"
                else:
                    color_col_name = None

                st.session_state["_dr_result"] = emb_df
                st.session_state["_dr_dim_labels"] = dim_labels
                st.session_state["_dr_method"] = method
                st.session_state["_dr_color"] = color_col_name
                st.session_state["_dr_n_comp_req"] = n_components # ユーザーが要求した表示次元数

                # PCA 特有の情報
                if method == "PCA":
                    if reducer.explained_variance_ratio_ is not None:
                        st.session_state["_dr_evr"] = reducer.explained_variance_ratio_
                    if reducer.loadings_ is not None:
                        st.session_state["_dr_loadings"] = reducer.loadings_
                    if reducer.reconstruction_error_ is not None:
                        st.session_state["_dr_recon_err"] = reducer.reconstruction_error_
                    st.session_state["_dr_features"] = reducer.feature_names_in_
                else:
                    for k in ["_dr_evr", "_dr_loadings", "_dr_features", "_dr_recon_err"]:
                        if k in st.session_state:
                            del st.session_state[k]

            except ImportError as e:
                st.error(f"❌ {e}")
                return
            except Exception as e:
                st.error(f"❌ 計算エラー: {e}")
                return

    # ─── 結果可視化 ──────────────────────────────────────────────────
    emb_df = st.session_state.get("_dr_result")
    if emb_df is None:
        return

    dim_labels = st.session_state.get("_dr_dim_labels", [])
    dr_method = st.session_state.get("_dr_method", "")
    color_col_name = st.session_state.get("_dr_color")
    evr = st.session_state.get("_dr_evr")

    import plotly.express as px

    # PCA寄与率バーチャート
    if dr_method == "PCA" and evr is not None:
        st.markdown("### 📊 PCA 寄与率")
        col1, col2 = st.columns([2, 1])
        with col1:
            evr_df = pd.DataFrame({
                "主成分": [f"PC{i+1}" for i in range(len(evr))],
                "寄与率": evr * 100,
                "累積寄与率": evr.cumsum() * 100,
            })
            fig_evr = px.bar(evr_df, x="主成分", y="寄与率",
                             text=evr_df["寄与率"].map("{:.1f}%".format),
                             color="寄与率", color_continuous_scale="Blues",
                             template="plotly_dark")
            fig_evr.add_scatter(x=evr_df["主成分"], y=evr_df["累積寄与率"],
                                mode="lines+markers", name="累積",
                                line=dict(color="#ff6b9d", width=2))
            fig_evr.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0f0"), coloraxis_showscale=False,
            )
            st.plotly_chart(fig_evr, use_container_width=True)
        with col2:
            st.dataframe(evr_df.round(2), use_container_width=True)

    # 2D / 3D 散布図
    st.markdown(f"### 🔵 {dr_method} 埋め込み可視化")
    
    color_data = emb_df[color_col_name] if color_col_name and color_col_name in emb_df.columns else None

    if dr_method == "PCA":
        tab_2d, tab_3d, tab_loadings, tab_anomaly = st.tabs([
            "🖼️ 2D 可視化", "🧊 3D インタラクティブ可視化", "🔗 Loading Matrix", "⚠️ 異常検知 (Reconstruction Error)"
        ])
        
        with tab_2d:
            c1, c2 = st.columns([2, 1])
            with c2:
                show_biplot = st.checkbox("🏹 Biplotを表示 (PC1 vs PC2)", value=True)
                if show_biplot:
                    top_n = st.slider("表示する特徴量数 (寄与上位)", 2, 30, 10, key="biplot_top_n")
            
            fig_2d = px.scatter(
                emb_df, x="PC1", y="PC2",
                color=color_data,
                opacity=0.75,
                color_continuous_scale="Viridis" if color_data is not None and pd.api.types.is_numeric_dtype(color_data) else None,
                template="plotly_dark",
                title=f"{dr_method} 2D埋め込み (PC1 vs PC2)",
                hover_data={c: True for c in emb_df.columns if not c.startswith("_")},
            )
            
            if show_biplot and "_dr_loadings" in st.session_state:
                loadings = st.session_state["_dr_loadings"]
                features = st.session_state["_dr_features"]
                scale_x = (emb_df["PC1"].max() - emb_df["PC1"].min())
                scale_y = (emb_df["PC2"].max() - emb_df["PC2"].min())
                scaling = min(scale_x, scale_y) * 0.8
                importance = np.sqrt(loadings[0]**2 + loadings[1]**2)
                indices = np.argsort(importance)[::-1][:top_n]
                
                for i in indices:
                    vx, vy = loadings[0, i] * scaling, loadings[1, i] * scaling
                    fig_2d.add_annotation(
                        x=vx, y=vy, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                        text=features[i], showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                        arrowcolor="#ff6b9d", font=dict(color="#ff6b9d", size=10),
                    )
            
            fig_2d.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e0f0"), height=600)
            _apply_smiles_hover(fig_2d, _get_smiles_for_index(emb_df.index, df_raw))
            st.plotly_chart(fig_2d, use_container_width=True)

        with tab_3d:
            if "PC3" in emb_df.columns:
                fig_3d = px.scatter_3d(
                    emb_df, x="PC1", y="PC2", z="PC3",
                    color=color_data,
                    opacity=0.7,
                    color_continuous_scale="Viridis" if color_data is not None and pd.api.types.is_numeric_dtype(color_data) else None,
                    template="plotly_dark",
                    title=f"{dr_method} 3D インタラクティブ (PC1, PC2, PC3)",
                    hover_data={c: True for c in emb_df.columns if not c.startswith("_")},
                )
                fig_3d.update_traces(marker=dict(size=4))
                fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40), height=750)
                _apply_smiles_hover(fig_3d, _get_smiles_for_index(emb_df.index, df_raw))
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.info("3Dプロットを表示するには、元の特徴量が3つ以上必要です。")

        with tab_loadings:
            st.markdown("### 🔗 Loading Matrix (成分重み)")
            if "_dr_loadings" in st.session_state:
                loadings = st.session_state["_dr_loadings"]
                features = st.session_state["_dr_features"]
                n_pc_show = min(10, loadings.shape[0])
                loadings_df = pd.DataFrame(
                    loadings[:n_pc_show].T,
                    index=features,
                    columns=[f"PC{i+1}" for i in range(n_pc_show)]
                )
                import plotly.graph_objects as go
                fig_hm = go.Figure(data=go.Heatmap(
                    z=loadings_df.values, x=loadings_df.columns.tolist(), y=loadings_df.index.tolist(),
                    colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                    text=loadings_df.values.round(3), texttemplate="%{text}",
                ))
                fig_hm.update_layout(template="plotly_dark", height=max(400, len(features) * 25),
                                     plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_hm, use_container_width=True)

        with tab_anomaly:
            st.markdown("### ⚠️ 再構成誤差による異常検知")
            st.write("PCAで圧縮した情報を元に戻せなかった誤差（Reconstruction Error）を表示します。誤差が大きいサンプルは、データセット内で「異質な構造」を持つ可能性があります。")
            if "_dr_recon_err" in st.session_state:
                recon_err = st.session_state["_dr_recon_err"]
                df_err = emb_df.copy()
                df_err["Reconstruction Error"] = recon_err
                
                fig_err = px.scatter(
                    df_err, x="PC1", y="PC2",
                    color="Reconstruction Error",
                    size="Reconstruction Error",
                    color_continuous_scale="Reds",
                    template="plotly_dark",
                    title="再構成誤差の可視化 (赤いほど異質)",
                    hover_data={c: True for c in df_err.columns if not c.startswith("_")},
                )
                fig_err.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=600)
                _apply_smiles_hover(fig_err, _get_smiles_for_index(emb_df.index, df_raw))
                st.plotly_chart(fig_err, use_container_width=True)
                
                st.markdown("#### 🚩 誤差の大きいサンプル (Top 10)")
                st.dataframe(df_err.sort_values("Reconstruction Error", ascending=False).head(10))
    
    else:
        # t-SNE / UMAP
        if len(dim_labels) >= 3:
            view_mode = st.radio("表示モード", ["2D", "3D"], horizontal=True)
        else:
            view_mode = "2D"

        if view_mode == "2D":
            fig = px.scatter(
                emb_df, x=dim_labels[0], y=dim_labels[1],
                color=color_data, opacity=0.75,
                color_continuous_scale="Viridis" if color_data is not None and pd.api.types.is_numeric_dtype(color_data) else None,
                template="plotly_dark",
                title=f"{dr_method} 2D埋め込み",
            )
        else:
            fig = px.scatter_3d(
                emb_df, x=dim_labels[0], y=dim_labels[1], z=dim_labels[2],
                color=color_data, opacity=0.7,
                color_continuous_scale="Viridis" if color_data is not None and pd.api.types.is_numeric_dtype(color_data) else None,
                template="plotly_dark",
                title=f"{dr_method} 3D埋め込み",
            )
            fig.update_traces(marker=dict(size=3))

        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e0f0"), height=650)
        _apply_smiles_hover(fig, _get_smiles_for_index(emb_df.index, df_raw))
        st.plotly_chart(fig, use_container_width=True)

    # 埋め込み結果ダウンロード
    st.download_button(
        "💾 埋め込み結果をCSVでダウンロード",
        emb_df.drop(columns=["_color"], errors="ignore").to_csv(index=True),
        file_name=f"{dr_method.lower()}_embedding.csv",
        mime="text/csv",
    )
