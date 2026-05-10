"""frontend_streamlit/pages/preprocess_page.py - 前処理・特徴量エンジニアリング設定ページ"""
from __future__ import annotations
import streamlit as st
from backend.data.preprocessor import PreprocessConfig


def render() -> None:
    st.markdown("## ⚙️ 前処理設定")
    df = st.session_state.get("df")
    if df is None:
        st.warning("⚠️ まずデータを読み込んでください。")
        return

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    tab1, tab2, tab3 = st.tabs(["🔧 前処理", "🛠️ 特徴量エンジニアリング", "📋 プレビュー"])

    # ─── Tab1: 前処理設定 ─────────────────────────────────────────
    with tab1:
        st.markdown("### 数値スケーラー")
        scaler = st.selectbox("スケーラー選択",
            ["auto（自動）", "standard（StandardScaler）", "robust（外れ値対応）",
             "minmax（0-1正規化）", "power_yj（YeoJohnson）", "quantile_normal（分位数）", "none（変換なし）"])

        st.markdown("### カテゴリエンコーダー")
        col1, col2 = st.columns(2)
        with col1:
            enc_low = st.selectbox("低cardinality (< 20ユニーク)", ["onehot", "ordinal", "target"])
        with col2:
            enc_high = st.selectbox("高cardinality (≥ 20ユニーク)", ["ordinal", "target", "hashing", "binary"])

        st.markdown("### 欠損値補完")
        col3, col4 = st.columns(2)
        with col3:
            imputer_num = st.selectbox("数値欠損", ["mean", "median", "knn", "iterative"])
        with col4:
            imputer_cat = st.selectbox("カテゴリ欠損", ["most_frequent", "constant"])

        st.markdown("### 除外設定")
        col5, col6, col7 = st.columns(3)
        with col5:
            excl_smiles = st.checkbox("SMILES列を除外", value=True)
        with col6:
            excl_dt = st.checkbox("DateTime列を除外", value=True)
        with col7:
            excl_const = st.checkbox("定数列を除外", value=True)

        if st.button("✅ 前処理設定を保存", use_container_width=True):
            cfg = PreprocessConfig(
                numeric_scaler=scaler.split("（")[0],
                cat_low_encoder=enc_low,
                cat_high_encoder=enc_high,
                numeric_imputer=imputer_num,
                categorical_imputer=imputer_cat,
                exclude_smiles=excl_smiles,
                exclude_datetime=excl_dt,
                exclude_constant=excl_const,
            )
            st.session_state["preprocess_config"] = cfg
            st.success("✅ 前処理設定を保存しました。AutoML実行時に反映されます。")

    # ─── Tab2: 特徴量エンジニアリング ─────────────────────────────
    with tab2:
        st.markdown("### 🔗 交互作用・多項式特徴量")
        add_interactions = st.checkbox("交互作用項を追加 (InteractionTransformer)", value=False)
        if add_interactions:
            col_a, col_b = st.columns(2)
            with col_a:
                interact_degree = st.slider("最大次数", 2, 3, 2)
            with col_b:
                interact_only = st.checkbox("交互作用のみ（自乗項なし）", value=True)

        st.divider()
        st.markdown("### ⏱️ 日時特徴量")
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        string_cols = [c for c in df.columns if df[c].dtype == object]
        dt_candidates = datetime_cols + string_cols

        if dt_candidates:
            dt_col_selection = st.multiselect("日時列として扱う列", dt_candidates)
            add_cyclic = st.checkbox("sin/cos循環特徴量を追加", value=True)
        else:
            st.info("日時型またはstring型の列がありません。")
            dt_col_selection = []
            add_cyclic = True

        st.divider()
        st.markdown("### 📈 ラグ・ローリング特徴量")
        add_lag = st.checkbox("ラグ・ローリング特徴量を追加 (LagRollingTransformer)", value=False)
        if add_lag:
            lag_cols = st.multiselect("対象列", numeric_cols)
            col_c, col_d = st.columns(2)
            with col_c:
                lag_sizes = st.text_input("ラグ数（カンマ区切り）", "1,2,3")
            with col_d:
                window_sizes = st.text_input("ウィンドウサイズ（カンマ区切り）", "3,7")

        # ── プレビュー実行 ──────────────────────────────────────
        if st.button("🔍 特徴量エンジニアリングをプレビュー", use_container_width=True):
            try:
                from backend.data.feature_engineer import (
                    InteractionTransformer,
                    DatetimeFeatureExtractor,
                    LagRollingTransformer,
                )
                import pandas as pd
                import numpy as np

                preview_parts: list[pd.DataFrame] = []

                # 交互作用項
                if add_interactions and numeric_cols:
                    it = InteractionTransformer(
                        degree=interact_degree if add_interactions else 2,
                        interaction_only=interact_only if add_interactions else True,
                    )
                    X_num = df[numeric_cols].fillna(0)
                    it.fit(X_num)
                    out = it.transform(X_num)
                    names = it.get_feature_names_out()
                    preview_parts.append(pd.DataFrame(out[:5], columns=names))
                    st.success(f"✅ 交互作用項: +{out.shape[1]}列")

                # 日時特徴量
                if dt_col_selection:
                    for dc in dt_col_selection:
                        dte = DatetimeFeatureExtractor(add_cyclic=add_cyclic)
                        sub = df[[dc]].head(5)
                        dte.fit(sub)
                        out = dte.transform(sub)
                        names = dte.get_feature_names_out()
                        preview_parts.append(pd.DataFrame(out, columns=names))
                        st.success(f"✅ 日時特徴量 '{dc}': +{len(names)}列")

                # ラグ・ローリング
                if add_lag and lag_cols:
                    lags = [int(x.strip()) for x in lag_sizes.split(",") if x.strip().isdigit()]
                    windows = [int(x.strip()) for x in window_sizes.split(",") if x.strip().isdigit()]
                    lr = LagRollingTransformer(lags=lags, windows=windows)
                    X_lag = df[lag_cols].fillna(0)
                    lr.fit(X_lag)
                    out = lr.transform(X_lag)
                    names = lr.get_feature_names_out()
                    preview_parts.append(pd.DataFrame(out[:5], columns=names))
                    st.success(f"✅ ラグ・ローリング: +{out.shape[1]}列")

                if preview_parts:
                    total_cols = sum(p.shape[1] for p in preview_parts)
                    st.info(f"📊 追加される特徴量: 合計 **{total_cols}列**")

                # 設定保存
                fe_config = {
                    "add_interactions": add_interactions,
                    "interact_degree": interact_degree if add_interactions else 2,  # type:ignore
                    "interact_only": interact_only if add_interactions else True,  # type:ignore
                    "dt_cols": dt_col_selection,
                    "add_cyclic": add_cyclic,
                    "add_lag": add_lag,
                    "lag_cols": lag_cols if add_lag else [],  # type:ignore
                    "lags": [int(x.strip()) for x in lag_sizes.split(",")  # type:ignore
                             if x.strip().isdigit()] if add_lag else [1, 2, 3],
                    "windows": [int(x.strip()) for x in window_sizes.split(",")  # type:ignore
                                if x.strip().isdigit()] if add_lag else [3, 7],
                }
                st.session_state["fe_config"] = fe_config

            except Exception as e:
                st.error(f"❌ エラー: {e}")

    # ─── Tab3: データプレビュー ────────────────────────────────────
    with tab3:
        st.markdown("### 📋 現在のデータプレビュー")
        st.markdown(f"**{df.shape[0]:,}行 × {df.shape[1]}列**")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("### 📊 列ごとの欠損値")
        null_df = df.isnull().sum().reset_index()
        null_df.columns = ["列名", "欠損数"]
        null_df["欠損率"] = (null_df["欠損数"] / len(df)).map("{:.1%}".format)
        null_df = null_df[null_df["欠損数"] > 0]
        if null_df.empty:
            st.success("✅ 欠損値なし")
        else:
            st.dataframe(null_df, use_container_width=True)
