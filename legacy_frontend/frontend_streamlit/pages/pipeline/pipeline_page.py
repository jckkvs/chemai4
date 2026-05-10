"""
frontend_streamlit/pages/pipeline/pipeline_page.py

ML パイプライン設定・実行ページ。

設計方針:
  - 初心者: 目的変数選択済みなら「実行」ボタン1クリックで完了（デフォルト設定）
  - 上級者: 全ステップ・全引数を expander 内で詳細設定可能
  - グリッドサーチ: 最下部の「上級設定」内に自然に存在（メインではない）
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from backend.pipeline import (
    PipelineConfig,
    PipelineGridConfig,
    PipelineCombination,
    ColPreprocessConfig,
    FeatureGenConfig,
    FeatureSelectorConfig,
    ColumnMeta,
    build_pipeline,
    generate_pipeline_grid,
    count_combinations,
)
import streamlit.components.v1 as components
from sklearn.utils import estimator_html_repr


# ============================================================
# ユーティリティ
# ============================================================

def _resolve_task(df: pd.DataFrame, target_col: str, task_hint: str) -> str:
    if task_hint in ("regression", "classification"):
        return task_hint
    t = df[target_col]
    if t.dtype == object or t.nunique() <= 10:
        return "classification"
    return "regression"


def _cv_score(combo, X, y, cv: int, task: str) -> dict:
    from sklearn.model_selection import cross_val_score
    scoring = "r2" if task == "regression" else "accuracy"
    try:
        t0 = time.perf_counter()
        scores = cross_val_score(combo.pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1)
        return {"name": combo.name, "mean": float(scores.mean()), "std": float(scores.std()),
                "elapsed": time.perf_counter() - t0, "status": "ok", "combo": combo}
    except Exception as e:
        return {"name": combo.name, "mean": np.nan, "std": np.nan, "elapsed": 0.0,
                "status": f"error: {e}", "combo": combo}


def _parse_name(name: str) -> dict[str, str]:
    return {p.split("=")[0]: p.split("=")[1] for p in name.split("|") if "=" in p}


# ============================================================
# メインレンダー
# ============================================================

def render() -> None:
    st.markdown(
        '<div style="font-size:1.5rem;font-weight:700;'
        'background:linear-gradient(90deg,#00d4ff,#7b2ff7);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
        '🔬 ML パイプライン</div>',
        unsafe_allow_html=True,
    )

    df = st.session_state.get("df")
    if df is None:
        st.warning("⚠️ まずデータを読み込んでください。")
        return

    target_col = st.session_state.get("target_col") or df.columns[-1]
    task_raw   = st.session_state.get("task", "auto")
    if "（" in task_raw:
        task_raw = task_raw.split("（")[0]
    task = _resolve_task(df, target_col, task_raw)
    feat_cols = [c for c in df.columns if c != target_col]

    st.caption(f"🎯 目的変数: **{target_col}**  |  タスク: **{task}**  |  特徴量: {len(feat_cols)} 列")
    st.divider()

    # ============================================================
    # ── STEP 1: 推定器選択（初心者向けメイン） ──────────────────
    # ============================================================
    if task == "regression":
        est_options = {
            "ridge（Ridge 回帰）": "ridge",
            "lasso_est（Lasso 回帰）": "lasso_est",
            "rf（RandomForest）": "rf",
            "xgb（XGBoost）": "xgb",
            "lgbm（LightGBM）": "lgbm",
            "xgbrf（XGBoost RF モード）": "xgbrf",
            "svr（SVR）": "svr",
            "knn_r（KNN Regressor）": "knn_r",
            "dt_r（DecisionTree）": "dt_r",
        }
    else:
        est_options = {
            "logistic（ロジスティック回帰）": "logistic",
            "rf_c（RandomForest）": "rf_c",
            "xgb_c（XGBoost）": "xgb_c",
            "lgbm_c（LightGBM）": "lgbm_c",
            "svc（SVC）": "svc",
            "knn_c（KNN Classifier）": "knn_c",
            "dt_c（DecisionTree）": "dt_c",
        }

    est_label = st.selectbox(
        "🤖 モデルを選んでください",
        list(est_options.keys()),
        key="pp_est_label",
    )
    est_key = est_options[est_label]

    # ── クイック実行ボタン（初心者向け） ─────────────────────
    col_run, col_cv = st.columns([3, 1])
    with col_cv:
        quick_cv = st.slider("CV 分割数", 2, 10, 5, key="pp_quick_cv")
    with col_run:
        st.markdown("")
        if st.button(
            f"🚀 {est_label.split('（')[0]} で実行",
            type="primary",
            use_container_width=True,
            key="pp_quick_run",
        ):
            _run_single(df, target_col, task, est_key, quick_cv, PipelineConfig())

    # 結果がある場合はここに表示
    if "pp_quick_result" in st.session_state:
        r = st.session_state["pp_quick_result"]
        scoring = "R²" if task == "regression" else "Accuracy"
        if r["status"] == "ok":
            st.success(
                f"✅ {scoring}: **{r['mean']:.4f}** ± {r['std']:.4f}  |  ⏱️ {r['elapsed']:.1f}秒"
            )
            
            # パイプライン構造の可視化
            if "pp_pipe_html" in st.session_state:
                with st.expander("🔍 構築されたパイプライン構成（図解）", expanded=False):
                    components.html(st.session_state["pp_pipe_html"], height=400, scrolling=True)
        else:
            st.error(f"❌ {r['status']}")

    st.divider()

    # ============================================================
    # ── 詳細設定（上級者向け expander） ────────────────────────
    # ============================================================
    with st.expander("⚙️ 詳細設定（前処理 / 特徴量 / 推定器パラメータ）", expanded=False):
        _render_advanced_settings(df, feat_cols, task, est_key, quick_cv, target_col)

    # ============================================================
    # ── グリッドサーチ（上級者向け・折り畳み） ─────────────────
    # ============================================================
    with st.expander("🔬 グリッドサーチ（複数候補を一括比較）", expanded=False):
        _render_grid_search(df, target_col, task, feat_cols)


# ============================================================
# クイック実行
# ============================================================

def _run_single(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    est_key: str,
    cv: int,
    config_override: PipelineConfig | None = None,
) -> None:
    # 除外列・weight列・info列を考慮
    _drop = [target_col]
    _drop.extend(st.session_state.get("col_role_exclude", []))
    _drop.extend(st.session_state.get("col_role_info", []))
    _w = st.session_state.get("col_role_weight")
    if _w: _drop.append(_w)
    _drop = [c for c in _drop if c in df.columns]
    X = df.drop(columns=_drop)
    y = df[target_col].values

    cfg = config_override or PipelineConfig()
    cfg.task = task
    cfg.estimator_key = est_key

    try:
        pipe = build_pipeline(cfg)
        html_repr = estimator_html_repr(pipe)
        st.session_state["pp_pipe_html"] = html_repr
    except Exception as e:
        st.session_state["pp_quick_result"] = {"status": f"Pipeline構築エラー: {e}", "mean": np.nan, "std": np.nan, "elapsed": 0.0}
        return

    from sklearn.model_selection import cross_val_score
    scoring = "r2" if task == "regression" else "accuracy"
    try:
        t0 = time.perf_counter()
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        st.session_state["pp_quick_result"] = {
            "status": "ok",
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "elapsed": time.perf_counter() - t0,
        }
    except Exception as e:
        st.session_state["pp_quick_result"] = {"status": f"CV エラー: {e}", "mean": np.nan, "std": np.nan, "elapsed": 0.0}


# ============================================================
# 詳細設定 expander 内部
# ============================================================

def _render_advanced_settings(
    df: pd.DataFrame,
    feat_cols: list[str],
    task: str,
    est_key: str,
    cv: int,
    target_col: str,
) -> None:
    """全引数を設定可能な詳細パネル。"""

    # ── Step 1: 入力列制御 ──────────────────────────────────
    st.markdown("#### 1️⃣ 入力列制御")
    col_mode = st.radio(
        "モード",
        ["all（全列）", "include（指定列のみ）", "exclude（除外列を指定）"],
        horizontal=True, key="pp_adv_mode",
    )
    mode_key = col_mode.split("（")[0]
    sel_cols: list[str] = []
    col_range = None
    if mode_key in ("include", "exclude"):
        sel_cols = st.multiselect(f"対象列（{mode_key}）", feat_cols, key="pp_adv_cols")
        use_range = st.checkbox("インデックス範囲で指定", key="pp_adv_userange")
        if use_range:
            c1, c2 = st.columns(2)
            r0 = c1.number_input("開始", 0, len(feat_cols) - 1, 0, key="pp_r0")
            r1 = c2.number_input("終了(exclusive)", 1, len(feat_cols), len(feat_cols), key="pp_r1")
            col_range = (int(r0), int(r1))

    # ── ColumnMeta（単調性） ────────────────────────────────
    st.markdown("##### 変数メタ情報（単調性制約）")
    st.caption("XGBoost / LightGBM / HistGB にのみ自動反映。他のモデルは無視されます。")
    meta_rows: dict[str, ColumnMeta] = {}
    show_cols = (sel_cols if sel_cols else feat_cols)[:20]
    for col in show_cols:
        c1, c2, c3 = st.columns([2, 1, 2])
        c1.markdown(f"`{col}`")
        mono = c2.selectbox("単調", [0, 1, -1], key=f"pp_mono_{col}",
                            format_func=lambda x: {0: "なし", 1: "↑増", -1: "↓減"}[x])
        grp  = c3.text_input("グループ", key=f"pp_grp_{col}", placeholder="空=なし")
        meta_rows[col] = ColumnMeta(monotonic=mono, group=grp or None)
    if len(feat_cols) > 20:
        st.caption(f"（残り {len(feat_cols) - 20} 列は省略）")

    st.divider()

    # ── Step 2: 前処理 ────────────────────────────────────
    st.markdown("#### 2️⃣ 前処理")
    c1, c2, c3 = st.columns(3)
    with c1:
        num_imputer = st.selectbox("📥 数値欠損補間",
            ["mean", "median", "knn", "iterative", "constant"], key="pp_imputer")
        num_scaler = st.selectbox("📏 数値スケーラー",
            ["standard", "minmax", "robust", "maxabs", "power_yj",
             "quantile_normal", "quantile_uniform", "log", "none"], key="pp_scaler")
    with c2:
        cat_low_enc = st.selectbox("🏷️ カテゴリ（低Card）エンコーダー",
            ["onehot", "ordinal", "target", "binary", "woe", "hashing", "leaveoneout"],
            key="pp_cat_low")
        cat_high_enc = st.selectbox("🏷️ カテゴリ（高Card）エンコーダー",
            ["ordinal", "target", "hashing", "binary", "woe"],
            key="pp_cat_high")
    with c3:
        cat_imputer = st.selectbox("📥 カテゴリ欠損補間",
            ["most_frequent", "constant"], key="pp_cat_imp")
        with st.container():
            add_indicator = st.checkbox("欠損フラグ列を追加", key="pp_add_ind")
            cardinality = st.number_input("高Card閾値", 2, 200, 20, key="pp_card")

    # OHEの追加引数
    with st.expander("OneHotEncoder 詳細設定", expanded=False):
        cc1, cc2, cc3 = st.columns(3)
        ohe_drop = cc1.selectbox("drop", ["first", "if_binary", None], key="pp_ohe_drop")
        ohe_handle = cc2.selectbox("handle_unknown", ["ignore", "infrequent_if_exist", "error"], key="pp_ohe_unk")
        ohe_max_cat = cc3.number_input("max_categories（0=制限なし）", 0, 500, 0, key="pp_ohe_max")

    pre_config = ColPreprocessConfig(
        numeric_imputer=num_imputer,
        numeric_scaler=num_scaler,
        cat_low_encoder=cat_low_enc,
        cat_high_encoder=cat_high_enc,
        categorical_imputer=cat_imputer,
        add_missing_indicator=add_indicator,
        cardinality_threshold=int(cardinality),
        onehot_drop=ohe_drop,
        onehot_handle_unknown=ohe_handle,
        onehot_max_categories=int(ohe_max_cat) if ohe_max_cat else None,
    )

    st.divider()

    # ── Step 3: 特徴量生成 ─────────────────────────────────
    st.markdown("#### 3️⃣ 特徴量生成")
    c1, c2 = st.columns([2, 1])
    gen_method_raw = c1.selectbox("メソッド",
        ["none（スキップ）", "polynomial（多項式）", "interaction_only（交互作用のみ）"],
        key="pp_gen_method")
    gen_method = gen_method_raw.split("（")[0]
    gen_degree = c2.number_input("次数", 2, 5, 2, key="pp_gen_degree") if gen_method != "none" else 2

    gen_config = FeatureGenConfig(method=gen_method, degree=int(gen_degree))

    st.divider()

    # ── Step 4: 特徴量選択 ─────────────────────────────────
    st.markdown("#### 4️⃣ 特徴量選択")
    sel_method_raw = st.selectbox("手法",
        ["none（スキップ）", "lasso", "ridge", "rfr（RandomForest）", "xgb（XGBoost）",
         "select_percentile", "select_kbest", "relieff", "boruta", "group_lasso"],
        key="pp_sel_method")
    sel_method = sel_method_raw.split("（")[0]

    # 手法ごとの追加パラメータ
    sel_threshold = "mean"
    sel_max_feat = None
    sel_percentile = 50
    sel_k = 10
    sel_score_func = "f_regression" if task == "regression" else "f_classif"

    if sel_method in ("lasso", "ridge", "rfr", "xgb"):
        c1, c2 = st.columns(2)
        sel_threshold = c1.text_input("threshold（例: mean, 0.01）", "mean", key="pp_sel_thr")
        sel_max_feat_raw = c2.number_input("max_features（0=制限なし）", 0, 500, 0, key="pp_sel_maxf")
        sel_max_feat = int(sel_max_feat_raw) if sel_max_feat_raw else None
    elif sel_method == "select_percentile":
        sel_percentile = st.slider("percentile", 1, 100, 50, key="pp_sel_perc")
        sel_score_func = st.selectbox("score_func",
            ["f_regression", "mutual_info_regression"] if task == "regression"
            else ["f_classif", "mutual_info_classif", "chi2"], key="pp_score_func")
    elif sel_method == "select_kbest":
        sel_k = st.number_input("k（選択する特徴量数）", 1, len(feat_cols), min(10, len(feat_cols)), key="pp_sel_k")
        sel_score_func = st.selectbox("score_func",
            ["f_regression", "mutual_info_regression"] if task == "regression"
            else ["f_classif", "mutual_info_classif", "chi2"], key="pp_score_func2")

    sel_config = FeatureSelectorConfig(
        method=sel_method,
        task=task,
        threshold=sel_threshold,
        max_features=sel_max_feat,
        percentile=int(sel_percentile),
        k=int(sel_k),
        score_func=sel_score_func,
    )

    st.divider()

    # ── Step 5: 推定器詳細パラメータ (Dynamic UI) ─────────────
    st.markdown("#### 5️⃣ 推定器パラメータ")
    from backend.models.factory import get_model_registry
    import inspect
    
    # task を基にレジストリ取得
    registry = get_model_registry(task)
    entry = registry.get(est_key, {})
    m_item = entry.get("class") or entry.get("factory")
    
    est_params = {}
    if m_item:
        with st.container(border=True):
            st.markdown(f"**{entry.get('name', est_key)} の詳細設定**")
            target_func = m_item.__init__ if inspect.isclass(m_item) else m_item
            try:
                msig = inspect.signature(target_func)
                default_ps = entry.get("default_params", {})
                m_cols = st.columns(3)
                m_idx = 0
                for pname, pinfo in msig.parameters.items():
                    if pname in ("self", "kwargs", "args"): continue
                    dval = default_ps.get(pname, pinfo.default if pinfo.default is not inspect.Parameter.empty else None)
                    anno = pinfo.annotation
                    with m_cols[m_idx % 3]:
                        key_p = f"pipe_adv_{est_key}_{pname}"
                        if isinstance(dval, bool) or anno is bool:
                            est_params[pname] = st.checkbox(pname, value=bool(dval), key=key_p)
                        elif isinstance(dval, int) or anno is int:
                            est_params[pname] = st.number_input(pname, value=int(dval) if dval is not None and not isinstance(dval, str) else 0, key=key_p)
                        elif isinstance(dval, float) or anno is float:
                            est_params[pname] = st.number_input(pname, value=float(dval) if dval is not None else 0.0, format="%.4f", key=key_p)
                        else:
                            est_params[pname] = st.text_input(pname, value=str(dval) if dval is not None else "", key=key_p)
                    m_idx += 1
            except Exception as e:
                st.warning(f"パラメータの取得に失敗しました: {e}")
    else:
        st.warning(f"モデル {est_key} の情報がレジストリに見つかりません。")

    apply_mono = st.checkbox(
        "単調性制約を自動反映（monotonic系パラメータを持つモデルのみ）",
        value=True, key="pp_apply_mono",
    )

    # ── 詳細設定で実行 ───────────────────────────────────
    adv_cv = st.slider("CV 分割数", 2, 10, 5, key="pp_adv_cv")
    if st.button("🚀 このパイプラインで実行", type="primary", use_container_width=True, key="pp_adv_run"):
        config = PipelineConfig(
            task=task,
            col_select_mode=mode_key,
            col_select_columns=sel_cols or None,
            col_select_range=col_range,
            column_meta=meta_rows,
            preprocessor_config=pre_config,
            feature_gen_config=gen_config,
            feature_sel_config=sel_config,
            estimator_key=est_key,
            estimator_params=est_params,
            apply_monotonic=apply_mono,
        )
        _run_single(df, target_col, task, est_key, adv_cv, config)
        st.rerun()


# ============================================================
# グリッドサーチ expander 内部
# ============================================================

def _render_grid_search(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    feat_cols: list[str],
) -> None:
    """複数候補の全組み合わせを比較するグリッドサーチ。"""
    st.caption("各ステップで複数候補を選ぶと、全組み合わせをCVで比較します。")

    c1, c2, c3 = st.columns(3)
    with c1:
        g_imputers = st.multiselect("欠損補間", ["mean", "median", "knn"], default=["mean"], key="pg_g_imp")
        g_scalers  = st.multiselect("スケーラー", ["standard", "robust", "minmax"], default=["standard"], key="pg_g_scl")
    with c2:
        g_gen_methods = st.multiselect("特徴量生成",
            ["none", "polynomial", "interaction_only"],
            default=["none"], key="pg_g_gen")
        g_sel_methods = st.multiselect("特徴量選択",
            ["none", "lasso", "rfr", "select_kbest"],
            default=["none"], key="pg_g_sel")
    with c3:
        if task == "regression":
            g_ests = st.multiselect("推定器",
                ["ridge", "rf", "xgb", "lgbm"], default=["ridge", "rf"], key="pg_g_est")
        else:
            g_ests = st.multiselect("推定器",
                ["logistic", "rf_c", "xgb_c", "lgbm_c"], default=["logistic", "rf_c"], key="pg_g_est")

    g_cv       = st.slider("CV 分割数", 2, 10, 5, key="pg_g_cv")
    g_max_comb = st.slider("最大組み合わせ数", 1, 100, 30, key="pg_g_max")

    # 組み合わせ数プレビュー
    gc = PipelineGridConfig(
        task=task,
        numeric_imputers=g_imputers or ["mean"],
        numeric_scalers=g_scalers or ["standard"],
        feature_gen_methods=g_gen_methods or ["none"],
        feature_sel_methods=g_sel_methods or ["none"],
        estimator_keys=g_ests or ["ridge"],
    )
    n_all = count_combinations(gc)
    n_run = min(n_all, g_max_comb)
    st.markdown(f"理論: **{n_all}** 件 → 実行: **{n_run}** 件")

    if st.button("🔬 グリッドサーチ実行", use_container_width=True, key="pg_g_run"):
        # 除外列・weight列・info列を考慮
        _drop_g = [target_col]
        _drop_g.extend(st.session_state.get("col_role_exclude", []))
        _drop_g.extend(st.session_state.get("col_role_info", []))
        _wg = st.session_state.get("col_role_weight")
        if _wg: _drop_g.append(_wg)
        _drop_g = [c for c in _drop_g if c in df.columns]
        X = df.drop(columns=_drop_g)
        y = df[target_col].values

        with st.spinner("パイプライン生成中..."):
            combos = generate_pipeline_grid(gc, max_combinations=g_max_comb)

        prog = st.progress(0, text="評価中...")
        results = []
        for i, combo in enumerate(combos):
            r = _cv_score(combo, X, y, g_cv, task)
            results.append(r)
            prog.progress((i + 1) / len(combos), text=f"{i+1}/{len(combos)}")

        prog.empty()
        results.sort(key=lambda r: -r["mean"] if not np.isnan(r["mean"]) else -np.inf)
        st.session_state["pg_grid_results"] = results
        st.rerun()

    # 結果表示
    if "pg_grid_results" in st.session_state:
        results = st.session_state["pg_grid_results"]
        ok = [r for r in results if r["status"] == "ok"]
        err = [r for r in results if r["status"] != "ok"]
        scoring = "R²" if task == "regression" else "Accuracy"

        if ok:
            best = ok[0]
            parts = _parse_name(best["name"])
            st.success(
                f"🏆 Best: est=**{parts.get('est','?')}** | sel=**{parts.get('sel','?')}** | "
                f"scl=**{parts.get('scl','?')}** → {scoring}: **{best['mean']:.4f}** ± {best['std']:.4f}"
            )
            rows = []
            for rank, r in enumerate(ok, 1):
                p = _parse_name(r["name"])
                rows.append({
                    "Rank": rank, scoring: round(r["mean"], 4), "±": round(r["std"], 4),
                    "推定器": p.get("est", "-"), "スケーラー": p.get("scl", "-"),
                    "特徴量選択": p.get("sel", "-"), "⏱️": round(r["elapsed"], 1),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ CSV ダウンロード", csv, "grid_results.csv", "text/csv")

        if err:
            with st.expander(f"⚠️ エラー {len(err)} 件"):
                for r in err:
                    st.markdown(f"- `{r['name'].split('|est=')[-1]}` → {r['status']}")
