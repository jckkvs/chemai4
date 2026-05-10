"""ベイズ最適化UI コンポーネント.

Implements: F-E01〜E08
    5ステップウィザード形式のインタラクティブUI
    Step1: 目的設定
    Step2: 探索空間設定
    Step3: 制約設定（基本GUI + 高度Python式）
    Step4: 候補生成
    Step5: 結果表示 + PCA可視化
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from backend.optim.bayesian_optimizer import BOConfig, BayesianOptimizer
from backend.optim.search_space import SearchSpace, Variable, VarType
from backend.optim.constraints import (
    RangeConstraint, SumConstraint, AtLeastNConstraint,
    CustomConstraint, apply_constraints, Constraint,
)
from backend.optim.bo_visualizer import (
    plot_pca_2d, plot_pca_3d, plot_pareto_front, plot_convergence,
)


def render_bayesian_opt_ui() -> None:
    """ベイズ最適化UIをレンダリング（結果タブ内のサブタブとして呼び出される）."""

    df = st.session_state.get("df")
    if df is None:
        st.warning("データが読み込まれていません。先にデータを読み込んでください。")
        return

    target_col = st.session_state.get("target_col")
    if not target_col:
        st.warning("目的変数が設定されていません。")
        return

    st.markdown(
        '<div class="section-header">'
        '\U0001f3af ベイズ最適化 — 実験計画'
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "学習済みモデルの知見に基づき、次の実験候補をベイズ最適化で提案します。"
        "オフライン実験（バッチ）に最適化されています。"
    )

    # ── ステップナビゲーション ──
    bo_steps = st.tabs([
        "1\u20e3 目的設定",
        "2\u20e3 探索空間",
        "3\u20e3 制約",
        "4\u20e3 候補生成",
        "5\u20e3 結果",
    ])

    # ════════════════════════════════════════════
    # Step 1: 目的設定
    # ════════════════════════════════════════════
    with bo_steps[0]:
        st.markdown("### 目的設定")

        # 単目的 or 多目的
        multi_obj = st.checkbox(
            "多目的最適化（ParEGO）",
            key="bo_multi_obj",
            help="複数の目的変数を同時に最適化します",
        )

        numeric_cols = list(df.select_dtypes(include="number").columns)

        if multi_obj:
            obj_cols = st.multiselect(
                "目的変数（複数選択）",
                options=numeric_cols,
                default=[target_col] if target_col in numeric_cols else [],
                key="bo_obj_cols",
            )
            if obj_cols:
                obj_dirs = []
                for col in obj_cols:
                    d = st.radio(
                        f"「{col}」の方向",
                        ["最小化", "最大化"],
                        key=f"bo_dir_{col}",
                        horizontal=True,
                    )
                    obj_dirs.append("min" if d == "最小化" else "max")
                st.session_state["_bo_obj_dirs"] = obj_dirs
                st.session_state["_bo_obj_cols"] = obj_cols
        else:
            st.info(f"目的変数: **{target_col}**")
            obj_type = st.radio(
                "最適化の方向",
                ["最小化", "最大化", "目標範囲"],
                key="bo_obj_type",
                horizontal=True,
            )
            if obj_type == "目標範囲":
                t_lo, t_hi = st.columns(2)
                with t_lo:
                    target_lo = st.number_input(
                        "目標下限", value=float(df[target_col].quantile(0.25)),
                        key="bo_target_lo",
                    )
                with t_hi:
                    target_hi = st.number_input(
                        "目標上限", value=float(df[target_col].quantile(0.75)),
                        key="bo_target_hi",
                    )
            st.session_state["_bo_obj_type"] = obj_type

        # カーネル設定
        st.markdown("#### カーネル設定")
        kernel_type = st.selectbox(
            "カーネルタイプ",
            ["DotProduct + Matern + White (推奨)", "Matern + White", "DotProduct + White"],
            key="bo_kernel",
        )
        matern_nu = st.select_slider(
            "Matern平滑度 (nu)",
            options=[0.5, 1.5, 2.5],
            value=2.5,
            key="bo_matern_nu",
            help="0.5=粗い, 1.5=中程度, 2.5=滑らか",
        )

        # 獲得関数
        st.markdown("#### 獲得関数")
        acq_map = {
            "Expected Improvement (EI) — 推奨": "ei",
            "Probability of Improvement (PI)": "pi",
            "Upper Confidence Bound (UCB)": "ucb",
            "Probability of Target Range (PTR)": "ptr",
        }
        acq_label = st.selectbox("獲得関数", list(acq_map.keys()), key="bo_acq")
        acq_fn = acq_map[acq_label]

        if acq_fn in ("ei", "pi"):
            xi = st.slider("探索パラメータ (xi)", 0.0, 1.0, 0.01, 0.01, key="bo_xi",
                           help="大きいほど探索的、小さいほど活用的")
        elif acq_fn == "ucb":
            kappa = st.slider("UCBパラメータ (kappa)", 0.0, 5.0, 2.0, 0.1, key="bo_kappa")

        st.success("Step 1 設定完了。「探索空間」タブに進んでください。")

    # ════════════════════════════════════════════
    # Step 2: 探索空間
    # ════════════════════════════════════════════
    with bo_steps[1]:
        st.markdown("### 探索空間設定")
        st.caption("各特徴量の探索範囲を設定します。データの統計量がデフォルトとして表示されます。")

        feature_cols = [c for c in numeric_cols if c != target_col]
        if multi_obj:
            obj_cols_set = set(st.session_state.get("_bo_obj_cols", [target_col]))
            feature_cols = [c for c in numeric_cols if c not in obj_cols_set]

        if not feature_cols:
            st.warning("探索対象の特徴量がありません。")
        else:
            space = SearchSpace.from_dataframe(df, columns=feature_cols, margin=0.1)

            # 範囲テーブルで一括編集
            range_data = []
            for v in space.variables:
                range_data.append({
                    "変数": v.name,
                    "下限": round(v.lo, 4) if v.lo is not None else 0,
                    "上限": round(v.hi, 4) if v.hi is not None else 0,
                    "ステップ": v.step if v.step else 0,
                    "データ最小": round(float(df[v.name].min()), 4),
                    "データ最大": round(float(df[v.name].max()), 4),
                })
            range_df = pd.DataFrame(range_data)
            edited_range = st.data_editor(
                range_df, key="bo_range_editor",
                use_container_width=True,
                disabled=["変数", "データ最小", "データ最大"],
            )

            # 候補生成方法
            gen_method = st.selectbox(
                "候補生成方法",
                ["自動推奨", "グリッド", "ランダム", "Latin Hypercube (LHS)", "グリッド+ランダム混合"],
                key="bo_gen_method",
            )
            method_map = {
                "自動推奨": "auto", "グリッド": "grid", "ランダム": "random",
                "Latin Hypercube (LHS)": "lhs", "グリッド+ランダム混合": "random_lhs",
            }

            n_max = st.slider(
                "最大候補数", 1000, 200000, 50000, 1000,
                key="bo_n_max",
                help="生成する最大候補点数",
            )

            # 候補数推定
            est = space.estimate_grid_size()
            recommended = space.auto_recommend_method()
            st.info(
                f"グリッド候補数（推定）: **{est:,}**\n\n"
                f"推奨方法: **{recommended}**"
            )

            # 編集された範囲をsession_stateに保存
            st.session_state["_bo_space"] = space
            st.session_state["_bo_edited_range"] = edited_range
            st.session_state["_bo_gen_method"] = method_map[gen_method]
            st.session_state["_bo_n_max"] = n_max

            st.success("Step 2 設定完了。「制約」タブに進んでください。")

    # ════════════════════════════════════════════
    # Step 3: 制約
    # ════════════════════════════════════════════
    with bo_steps[2]:
        st.markdown("### 制約設定")
        st.caption("複数のグループ制約を設定できます。「＋ グループ追加」で制約を増やせます。")

        constraints_list: list[Constraint] = []
        feature_cols_for_c = [c for c in numeric_cols if c != target_col]

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 合計制約（複数グループ対応）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("#### 📐 合計制約")
        st.caption("例: A+B+C = 100, D+E+F = 100 のように複数グループを設定可能")

        # グループ数管理
        if "_bo_sum_groups" not in st.session_state:
            st.session_state["_bo_sum_groups"] = 0

        n_sum_groups = st.session_state["_bo_sum_groups"]

        for g in range(n_sum_groups):
            with st.container():
                st.markdown(f"**合計グループ {g+1}**")
                gc1, gc2, gc3 = st.columns([3, 1, 1])
                with gc1:
                    sum_cols = st.multiselect(
                        f"列を選択", options=feature_cols_for_c,
                        key=f"bo_sum_cols_{g}",
                        label_visibility="collapsed",
                        placeholder="合計する列を選択...",
                    )
                with gc2:
                    sum_target = st.number_input(
                        "合計値", value=100.0, key=f"bo_sum_target_{g}",
                    )
                with gc3:
                    sum_tol = st.number_input(
                        "許容誤差", value=0.01, min_value=0.0,
                        key=f"bo_sum_tol_{g}",
                    )
                if sum_cols:
                    constraints_list.append(
                        SumConstraint(sum_cols, sum_target, sum_tol)
                    )
                st.divider()

        sc_add, sc_del = st.columns(2)
        with sc_add:
            if st.button("＋ 合計グループ追加", key="bo_add_sum"):
                st.session_state["_bo_sum_groups"] += 1
                st.rerun()
        with sc_del:
            if n_sum_groups > 0 and st.button("－ 最後のグループ削除", key="bo_del_sum"):
                st.session_state["_bo_sum_groups"] = max(0, n_sum_groups - 1)
                st.rerun()

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 少なくともN個使用（複数グループ対応）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("#### 🎲 少なくともN個使用")
        st.caption(
            "例: {A,B,C}から少なくとも1つ、{D,E,F}から少なくとも1つ、"
            "{H,I,G}から少なくとも2つ"
        )

        if "_bo_atleast_groups" not in st.session_state:
            st.session_state["_bo_atleast_groups"] = 0

        n_al_groups = st.session_state["_bo_atleast_groups"]

        for g in range(n_al_groups):
            with st.container():
                st.markdown(f"**選択グループ {g+1}**")
                ac1, ac2, ac3 = st.columns([3, 1, 1])
                with ac1:
                    al_cols = st.multiselect(
                        f"対象列", options=feature_cols_for_c,
                        key=f"bo_al_cols_{g}",
                        label_visibility="collapsed",
                        placeholder="対象列を選択...",
                    )
                with ac2:
                    min_n = st.number_input(
                        "最低N個", value=1, min_value=1,
                        key=f"bo_al_min_{g}",
                    )
                with ac3:
                    al_thresh = st.number_input(
                        "閾値", value=0.0, key=f"bo_al_thresh_{g}",
                    )
                if al_cols:
                    constraints_list.append(
                        AtLeastNConstraint(
                            columns=al_cols,
                            min_count=int(min_n),
                            threshold=al_thresh,
                            label=f"グループ{g+1}",
                        )
                    )
                st.divider()

        al_add, al_del = st.columns(2)
        with al_add:
            if st.button("＋ 選択グループ追加", key="bo_add_al"):
                st.session_state["_bo_atleast_groups"] += 1
                st.rerun()
        with al_del:
            if n_al_groups > 0 and st.button("－ 最後のグループ削除", key="bo_del_al"):
                st.session_state["_bo_atleast_groups"] = max(0, n_al_groups - 1)
                st.rerun()

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 高度な制約: Python式 ＋ 変数ボタン
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        st.markdown("#### ✏️ カスタム制約（Python式）")
        st.caption("下の変数ボタンをクリックして列名を挿入できます。")

        # 変数名ボタン → クリックでクリップボード/テキストに追加
        st.markdown("**変数名:**")
        btn_cols = st.columns(min(len(feature_cols_for_c), 8) if feature_cols_for_c else 1)
        for i, col_name in enumerate(feature_cols_for_c):
            col_idx = i % len(btn_cols)
            with btn_cols[col_idx]:
                if st.button(
                    f"`{col_name}`", key=f"bo_var_btn_{col_name}",
                    use_container_width=True,
                ):
                    cur = st.session_state.get("bo_custom_expr", "")
                    st.session_state["bo_custom_expr"] = cur + col_name + " "
                    st.rerun()

        # 演算子ボタン
        st.markdown("**演算子:**")
        op_cols = st.columns(8)
        ops_list = ["+", "-", "*", "/", "<=", ">=", "==", "|"]
        for i, op in enumerate(ops_list):
            with op_cols[i]:
                if st.button(op, key=f"bo_op_btn_{i}", use_container_width=True):
                    cur = st.session_state.get("bo_custom_expr", "")
                    st.session_state["bo_custom_expr"] = cur + f" {op} "
                    st.rerun()

        custom_expr = st.text_area(
            "制約式（1行に1つ）",
            key="bo_custom_expr",
            height=100,
            placeholder="temperature + pressure <= 100\ntime_h >= 5",
        )
        if custom_expr and custom_expr.strip():
            for line in custom_expr.strip().split("\n"):
                line = line.strip()
                if line:
                    constraints_list.append(CustomConstraint(line))

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 制約プレビュー
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if constraints_list:
            st.markdown("---")
            st.markdown("#### 📋 設定中の制約一覧")
            for i, c in enumerate(constraints_list):
                st.markdown(f"**{i+1}.** {c.describe()}")
        else:
            st.info("制約なし（全候補が探索対象）")

        st.session_state["_bo_constraints"] = constraints_list
        st.success(f"Step 3 設定完了（{len(constraints_list)}件の制約）。「候補生成」タブに進んでください。")

    # ════════════════════════════════════════════
    # Step 4: 候補生成
    # ════════════════════════════════════════════
    with bo_steps[3]:
        st.markdown("### 候補生成")

        # バッチ設定
        n_suggest = st.slider(
            "提案候補数", 1, 50, 5, 1, key="bo_n_suggest",
            help="1回のバッチで提案する実験候補数",
        )
        strategy_map = {
            "Kriging Believer (推奨: バッチ多様化)": "kriging_believer",
            "トップN（単純）": "single",
            "DoE → BO（多様性優先でBO選択）": "doe_then_bo",
            "BO → DoE（BO候補から多様性選択）": "bo_then_doe",
        }
        strategy_label = st.selectbox(
            "バッチ戦略", list(strategy_map.keys()), key="bo_strategy",
        )

        # 実行
        if st.button("候補を生成する", type="primary", use_container_width=True, key="bo_run"):
            space = st.session_state.get("_bo_space")
            constraints = st.session_state.get("_bo_constraints", [])
            edited_range = st.session_state.get("_bo_edited_range")

            if space is None:
                st.error("探索空間が設定されていません。Step 2を完了してください。")
            else:
                with st.spinner("候補を生成中..."):
                    # 編集された範囲を適用
                    if edited_range is not None:
                        for _, row in edited_range.iterrows():
                            for v in space.variables:
                                if v.name == row["変数"]:
                                    v.lo = float(row["下限"])
                                    v.hi = float(row["上限"])
                                    if row["ステップ"] > 0:
                                        v.step = float(row["ステップ"])
                                    break

                    # 候補生成
                    gen_method = st.session_state.get("_bo_gen_method", "auto")
                    n_max = st.session_state.get("_bo_n_max", 50000)
                    candidates_df = space.generate_candidates(
                        method=gen_method, n_max=n_max,
                    )

                    # 制約適用
                    if constraints:
                        candidates_df, c_report = apply_constraints(
                            candidates_df, constraints,
                        )
                        st.info(
                            f"制約適用: {c_report['before']:,} → {c_report['after']:,} 候補 "
                            f"({c_report['removed']:,} 除去)"
                        )

                    if len(candidates_df) == 0:
                        st.error("制約を満たす候補がありません。制約を緩和してください。")
                    else:
                        # BOConfig構築
                        obj_type = st.session_state.get("_bo_obj_type", "最小化")
                        multi_obj = st.session_state.get("bo_multi_obj", False)
                        acq_fn = acq_map.get(
                            st.session_state.get("bo_acq", "Expected Improvement (EI) — 推奨"), "ei"
                        )

                        config = BOConfig(
                            objective={
                                "最小化": "minimize", "最大化": "maximize",
                                "目標範囲": "minimize",
                            }.get(obj_type, "minimize"),
                            acquisition="ptr" if obj_type == "目標範囲" else acq_fn,
                            xi=st.session_state.get("bo_xi", 0.01),
                            kappa=st.session_state.get("bo_kappa", 2.0),
                            target_lo=st.session_state.get("bo_target_lo"),
                            target_hi=st.session_state.get("bo_target_hi"),
                            kernel_type={
                                "DotProduct + Matern + White (推奨)": "default",
                                "Matern + White": "matern",
                                "DotProduct + White": "dotproduct",
                            }.get(st.session_state.get("bo_kernel", ""), "default"),
                            matern_nu=st.session_state.get("bo_matern_nu", 2.5),
                            batch_strategy=strategy_map[strategy_label],
                            n_candidates=n_suggest,
                            multi_objective=multi_obj,
                            objective_columns=st.session_state.get("_bo_obj_cols", [target_col]),
                            objective_directions=st.session_state.get("_bo_obj_dirs", []),
                            parego_rho=0.05,
                        )

                        # GPフィット
                        feature_cols = [v.name for v in space.variables]
                        X_train = df[feature_cols].values
                        if multi_obj:
                            obj_cols = config.objective_columns
                            y_train = df[obj_cols].values
                        else:
                            y_train = df[target_col].values

                        optimizer = BayesianOptimizer(config)
                        optimizer.fit(X_train, y_train)

                        # 候補提案
                        suggestions = optimizer.suggest(
                            candidates_df[feature_cols], n=n_suggest,
                        )

                        st.session_state["_bo_suggestions"] = suggestions
                        st.session_state["_bo_optimizer"] = optimizer
                        st.session_state["_bo_feature_cols"] = feature_cols
                        st.session_state["_bo_candidates_df"] = candidates_df

                        st.success(
                            f"{len(suggestions)}件の候補を生成しました！"
                            "「結果」タブで確認できます。"
                        )

    # ════════════════════════════════════════════
    # Step 5: 結果
    # ════════════════════════════════════════════
    with bo_steps[4]:
        st.markdown("### 結果")

        suggestions = st.session_state.get("_bo_suggestions")
        optimizer = st.session_state.get("_bo_optimizer")
        feature_cols = st.session_state.get("_bo_feature_cols", [])

        if suggestions is None:
            st.info("まだ候補が生成されていません。Step 4 で「候補を生成する」を押してください。")
        else:
            # 候補テーブル
            st.markdown("#### 提案候補")
            st.dataframe(suggestions, use_container_width=True)

            # CSVダウンロード
            csv_buf = io.StringIO()
            suggestions.to_csv(csv_buf, index=False, encoding="utf-8-sig")
            st.download_button(
                "CSVダウンロード",
                csv_buf.getvalue(),
                "bo_suggestions.csv",
                "text/csv",
                key="bo_csv_dl",
            )

            # GP情報
            if optimizer:
                gp_info = optimizer.get_gp_info()
                with st.expander("GPモデル情報"):
                    st.json(gp_info)

            # ── PCA可視化 ──
            st.markdown("#### 空間可視化")
            viz_mode = st.radio(
                "可視化モード", ["PCA 2D", "PCA 3D"],
                key="bo_viz_mode", horizontal=True,
            )

            if feature_cols:
                X_ex = df[feature_cols].values
                X_cand = suggestions[feature_cols].values if isinstance(suggestions, pd.DataFrame) else suggestions[:, :len(feature_cols)]
                y_ex = df[target_col].values if target_col in df.columns else None

                if viz_mode == "PCA 2D":
                    fig, pca_info = plot_pca_2d(
                        X_ex, X_cand,
                        feature_names=feature_cols,
                        y_existing=y_ex,
                    )
                else:
                    fig, pca_info = plot_pca_3d(
                        X_ex, X_cand,
                        feature_names=feature_cols,
                        y_existing=y_ex,
                    )
                st.plotly_chart(fig, use_container_width=True)

                # 累積寄与率の詳細
                with st.expander("PCA 累積寄与率の詳細"):
                    cum = pca_info["cumulative"]
                    for i, c in enumerate(cum):
                        st.markdown(f"- PC{i+1}: {c:.1%}")

            # 多目的: パレートフロント
            multi_obj = st.session_state.get("bo_multi_obj", False)
            if multi_obj and optimizer and isinstance(suggestions, pd.DataFrame):
                obj_cols = st.session_state.get("_bo_obj_cols", [])
                dirs = st.session_state.get("_bo_obj_dirs", [])
                if len(obj_cols) == 2:
                    st.markdown("#### パレートフロント")
                    Y_ex = df[obj_cols].values
                    mu_cand, _ = optimizer.predict(suggestions[feature_cols])
                    fig_pareto = plot_pareto_front(
                        Y_ex, mu_cand,
                        objective_names=obj_cols,
                        directions=dirs,
                    )
                    st.plotly_chart(fig_pareto, use_container_width=True)

            # ── 実験結果インポート ──
            st.markdown("---")
            st.markdown("#### 実験結果のインポート")
            st.caption(
                "オフラインで実験を実施後、結果CSVをアップロードしてデータに追加できます。"
            )
            uploaded = st.file_uploader(
                "結果CSV", type=["csv"], key="bo_result_upload",
            )
            if uploaded:
                try:
                    new_data = pd.read_csv(uploaded)
                    st.dataframe(new_data.head(), use_container_width=True)
                    if st.button("データに追加", key="bo_append"):
                        current_df = st.session_state.get("df")
                        if current_df is not None:
                            combined = pd.concat(
                                [current_df, new_data], ignore_index=True,
                            )
                            st.session_state["df"] = combined
                            st.success(
                                f"データ更新完了！ {len(current_df)} → {len(combined)} 行"
                            )
                            st.rerun()
                except Exception as e:
                    st.error(f"CSVの読み込みに失敗しました: {e}")
