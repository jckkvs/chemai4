# -*- coding: utf-8 -*-
"""
frontend_streamlit/components/leakage_check_ui.py

リーケージ検出の事前チェックUIコンポーネント。
解析実行前にデータのリーケージリスクを評価し、
適切な CV 戦略を提案する。
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np


def render_leakage_check_panel(df: pd.DataFrame, target_col: str) -> None:
    """
    リーケージ事前チェックパネルを描画する。

    Args:
        df: 読み込み済みデータ
        target_col: 目的変数の列名
    """
    with st.expander("🔍 リーケージ事前チェック（推奨）", expanded=False):
        st.caption(
            "説明変数間のサンプル類似度を分析し、"
            "train/test リーケージのリスクを評価します。"
            "グループ構造が検出された場合は GroupKFold を推奨します。"
        )

        # 手法選択
        col1, col2 = st.columns([2, 1])
        with col1:
            method = st.selectbox(
                "類似度推定手法",
                ["auto（自動選択）", "hat（ハット行列）", "rbf（RBFカーネル）", "rf（ランダムフォレスト）"],
                index=0,
                key="leakage_method",
                help=(
                    "auto: サンプル数に応じて自動選択\n"
                    "hat: 線形モデルベース（高速・小規模向け）\n"
                    "rbf: 非線形類似度（中速・汎用）\n"
                    "rf: RFの葉ノード共有率（低速・最も頑健）"
                ),
            )
        with col2:
            threshold = st.slider(
                "類似度閾値",
                min_value=0.80,
                max_value=0.999,
                value=0.95,
                step=0.01,
                key="leakage_threshold",
                help="この値以上の類似度を「疑わしい」と判定します。",
            )

        if st.button("🔍 リーケージチェック実行", key="run_leakage_check"):
            method_key = method.split("（")[0]  # "auto", "hat", "rbf", "rf"

            # 説明変数のみ抽出
            exclude_cols = [target_col]
            smiles_col = st.session_state.get("smiles_col")
            if smiles_col:
                exclude_cols.append(smiles_col)
            excluded = st.session_state.get("col_role_exclude", [])
            exclude_cols.extend(excluded)

            feature_cols = [c for c in df.columns if c not in exclude_cols]
            X_check = df[feature_cols].select_dtypes(include=[np.number])
            y_check = df[target_col]

            if X_check.shape[1] < 2:
                st.warning("⚠️ 数値の説明変数が2列未満のため、チェックできません。")
                return

            n_samples = X_check.shape[0]

            with st.spinner(f"🔍 {n_samples} サンプルの類似度を分析中..."):
                from backend.data.leakage_detector import detect_leakage
                report = detect_leakage(
                    X_check, y_check,
                    method=method_key,
                    similarity_threshold=threshold,
                )

            st.session_state["leakage_report"] = report

            # --- 結果表示 ---
            _render_leakage_result(report, n_samples)


def _render_leakage_result(report, n_samples: int) -> None:
    """リーケージ検出結果を表示する。"""
    # リスクレベル表示
    risk_colors = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    risk_labels = {"low": "低リスク", "medium": "中リスク", "high": "高リスク"}
    icon = risk_colors.get(report.risk_level, "⚪")
    label = risk_labels.get(report.risk_level, "不明")

    st.markdown(f"### {icon} リーケージリスク: **{label}** （スコア: {report.risk_score:.2f}）")

    # メトリクス
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("手法", report.method_used.upper())
    c2.metric("疑わしいペア", f"{report.n_suspicious_pairs:,}")
    c3.metric("グループ数", report.n_groups if report.n_groups > 0 else "—")
    c4.metric("一貫性スコア", f"{report.details.get('group_consistency_score', 0):.2f}")

    # CV推奨
    if report.risk_level == "low":
        st.success(f"✅ **推奨CV**: {report.recommended_cv} — {report.cv_reason}")
    elif report.risk_level == "medium":
        st.warning(f"⚠️ **推奨CV**: {report.recommended_cv} — {report.cv_reason}")
    else:
        st.error(f"🚨 **推奨CV**: {report.recommended_cv} — {report.cv_reason}")

    # グループラベル情報
    if report.group_labels is not None and report.n_groups >= 2:
        with st.expander(f"📊 推定グループ詳細（{report.n_groups}グループ）"):
            group_counts = pd.Series(report.group_labels).value_counts().sort_index()
            df_groups = pd.DataFrame({
                "グループ": group_counts.index,
                "サンプル数": group_counts.values,
                "割合 (%)": (group_counts.values / n_samples * 100).round(1),
            })
            st.dataframe(df_groups, hide_index=True, use_container_width=True)

            st.caption(
                "💡 このグループラベルを GroupKFold の `groups` 引数に使用できます。"
                "「適用」ボタンでパイプライン設定に反映されます。"
            )
            if st.button("✅ グループラベルを適用", key="apply_group_labels"):
                st.session_state["leakage_group_labels"] = report.group_labels
                st.success("グループラベルをパイプラインに適用しました。")

    # 疑わしいペア上位
    if report.suspicious_pairs:
        with st.expander(f"🔎 類似度の高いペア（上位{min(10, len(report.suspicious_pairs))}件）"):
            pairs_data = []
            for p in report.suspicious_pairs[:10]:
                pairs_data.append({
                    "サンプルA": p.idx_a,
                    "サンプルB": p.idx_b,
                    "類似度": f"{p.similarity:.4f}",
                })
            st.dataframe(pd.DataFrame(pairs_data), hide_index=True, use_container_width=True)
