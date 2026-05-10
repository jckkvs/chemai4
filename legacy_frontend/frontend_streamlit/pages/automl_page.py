"""
frontend_streamlit/pages/automl_page.py

AutoML 完全パイプライン実行ページ。
EDA → 前処理 → モデル学習 → 評価 → 次元削減 → SHAP解析まで
ワンクリックで実行する。
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
import streamlit as st

from backend.models.automl import AutoMLEngine, AutoMLResult
from backend.data.preprocessor import PreprocessConfig
from backend.data.type_detector import TypeDetector
from backend.data.eda import summarize_dataframe, compute_column_stats, detect_outliers
from backend.data.dim_reduction import run_pca
from backend.data.benchmark import evaluate_regression, evaluate_classification


# ─────────────────────────────────────────────────────────────
# データクラス: フルパイプライン結果
# ─────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """フルパイプライン実行結果を保持するデータクラス。"""
    eda_summary: dict[str, Any] = field(default_factory=dict)
    col_stats: list = field(default_factory=list)
    outlier_results: list = field(default_factory=list)
    automl_result: AutoMLResult | None = None
    model_score: Any | None = None          # ModelScore (回帰/分類)
    pca_df: pd.DataFrame | None = None
    pca_evr: np.ndarray | None = None
    shap_importances: dict[str, float] = field(default_factory=dict)
    task: str = "regression"
    elapsed: float = 0.0
    warnings: list[str] = field(default_factory=list)
    smiles_transformer: Any = None
    smiles_correlations: dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# メイン render 関数
# ─────────────────────────────────────────────────────────────

def render(run_config: dict | None = None) -> None:
    st.markdown("## 🤖 解析結果")

    df = st.session_state.get("df")
    target_col = st.session_state.get("target_col")

    if df is None:
        st.warning("⚠️ まずデータを読み込んでください。")
        if st.button("🏠 ホームへ", key="goto_home_a"):
            st.session_state["page"] = "home"
            st.rerun()
        return

    if not target_col:
        st.warning("⚠️ 目的変数が選択されていません。ホームで設定してください。")
        if st.button("🏠 ホームへ", key="goto_home_b"):
            st.session_state["page"] = "home"
            st.rerun()
        return

    # ── ホームから渡された設定で即実行 ──────────────────────
    if run_config is not None:
        _df_for_run = df.copy()

        # P0-1/P1-1: 除外列+Info列をdrop
        _exclude_cols = run_config.get("exclude_cols", [])
        _cols_to_drop = [c for c in _exclude_cols if c in _df_for_run.columns]
        if _cols_to_drop:
            _df_for_run = _df_for_run.drop(columns=_cols_to_drop)
            st.caption(f"🚫 除外列を解析から除去: {', '.join(_cols_to_drop)}")

        # P0-2: グループ列（手動設定 or リーケージ検出）
        _cv_groups_col = None
        _manual_group_col = run_config.get("cv_groups_col")
        if _manual_group_col and _manual_group_col in _df_for_run.columns:
            _cv_groups_col = _manual_group_col
        else:
            # リーケージ検出のグループラベルがある場合はdfに一時列を追加
            _leakage_groups = run_config.get("leakage_group_labels")
            if _leakage_groups is not None and len(_leakage_groups) == len(_df_for_run):
                _df_for_run["_leakage_group"] = _leakage_groups
                _cv_groups_col = "_leakage_group"

        # リーケージ推奨CV or 手動グループ列によるCV選択
        _leakage_cv = run_config.get("leakage_recommended_cv")
        _cv_key = "auto"
        _cv_folds = run_config.get("cv_folds", 5)
        if _manual_group_col and _cv_groups_col:
            # グループ数チェック: GroupKFoldにはグループ数 >= n_splits が必須
            _n_groups = _df_for_run[_cv_groups_col].nunique()
            if _n_groups >= _cv_folds:
                _cv_key = "group_kfold"
                st.info(f"📋 グループ列 `{_manual_group_col}` を使用して GroupKFold を適用します。({_n_groups}グループ)")
            else:
                st.warning(
                    f"⚠️ グループ列 `{_manual_group_col}` のグループ数({_n_groups})が "
                    f"CV分割数({_cv_folds})未満のため、通常のKFoldを適用します。"
                )
                _cv_groups_col = None
        elif _leakage_cv in ("GroupKFold", "LeaveOneGroupOut") and _cv_groups_col:
            # リーケージ検出によるGroupKFold: グループ数チェック
            _n_groups = _df_for_run[_cv_groups_col].nunique()
            if _n_groups >= _cv_folds:
                if _leakage_cv == "LeaveOneGroupOut":
                    _cv_key = "leave_one_group_out"
                    st.info(f"📋 **リーケージ検出** により LeaveOneGroupOut を自動適用しました。({_n_groups}グループ)")
                else:
                    _cv_key = "group_kfold"
                    st.info(f"📋 **リーケージ検出** により GroupKFold を自動適用しました。({_n_groups}グループ)")
            else:
                st.warning(
                    f"⚠️ リーケージ検出でGroupKFoldが推奨されましたが、グループ数({_n_groups})が "
                    f"CV分割数({_cv_folds})未満のため、通常のKFoldを適用します。"
                )
                # リーケージグループ列を削除
                if _cv_groups_col == "_leakage_group" and _cv_groups_col in _df_for_run.columns:
                    _df_for_run = _df_for_run.drop(columns=[_cv_groups_col])
                _cv_groups_col = None

        # P1-2: 時系列列の推奨
        _time_col = run_config.get("col_role_time")
        if _time_col and _time_col in _df_for_run.columns and _cv_key == "auto":
            st.info(f"📅 時系列列 `{_time_col}` が設定されています。TimeSeriesSplit の使用を推奨します（詳細ツール→パイプライン設計で変更可能）。")

        # P0-3: Weight列を説明変数から除外（sample_weightとして使用予定）
        _weight_col = run_config.get("sample_weight_col")
        if _weight_col and _weight_col in _df_for_run.columns:
            _df_for_run = _df_for_run.drop(columns=[_weight_col])
            st.caption(f"⚖️ Weight列 `{_weight_col}` を説明変数から除外しました。")

        _run_full_pipeline(
            df          = _df_for_run,
            target_col  = run_config["target_col"],
            smiles_col  = run_config.get("smiles_col"),
            numeric_scaler = run_config.get("scaler", "auto"),
            task_override  = run_config.get("task", "auto"),
            cv_key      = _cv_key,
            cv_folds    = run_config.get("cv_folds", 5),
            cv_groups_col = _cv_groups_col,
            models      = run_config.get("models", []),
            timeout     = run_config.get("timeout", 300),
            do_eda      = run_config.get("do_eda", True),
            do_prep     = run_config.get("do_prep", True),
            do_ml       = True,
            do_eval     = run_config.get("do_eval", True),
            do_pca      = run_config.get("do_pca", True),
            do_shap     = run_config.get("do_shap", True),
            selected_descriptors = run_config.get("selected_descriptors", None),
            monotonic_constraints_dict = run_config.get("monotonic_constraints_dict", {}),
        )
        return

    # ── データ概要バー ────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    metric_items = [
        (c1, str(df.shape[0]), "サンプル数"),
        (c2, str(df.shape[1] - 1), "特徴量数"),
        (c3, target_col, "目的変数"),
        (c4, str(df.select_dtypes(include="number").shape[1]), "数値列数"),
    ]
    for col, val, lbl in metric_items:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value" style="font-size:1.2rem;">'
                f'{val}</div><div class="metric-label">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── 既存結果の表示 ─────────────────────────────────────
    existing_pl = st.session_state.get("pipeline_result")
    existing_ml = st.session_state.get("automl_result")

    # タブ表示 (結果がある場合)
    if existing_pl is not None:
        _show_pipeline_result(existing_pl)
        st.markdown("---")
        if st.button("🔄 設定を変えて再解析", use_container_width=True, key="rerun_pl"):
            st.session_state["pipeline_result"] = None
            st.session_state["automl_result"] = None
            st.session_state["active_tab_idx"] = 0
            st.rerun()
        return

    elif existing_ml is not None:
        # 旧形式 (AutoMLのみ) 結果表示
        st.success(f"✅ 前回の結果: 最良モデル = **{existing_ml.best_model_key}** | "
                   f"スコア = `{existing_ml.best_score:.4f}`")
        _show_leaderboard(existing_ml)
        if st.button("🔄 設定を変えて再解析", use_container_width=True, key="rerun_old"):
            st.session_state["automl_result"] = None
            st.session_state["active_tab_idx"] = 0
            st.rerun()
        return

    # ── 未実行 → ① データ設定タブで設定してください ─────────────
    # 設定は全て「①データ設定」タブのパイプライン設計で行う（二重設定廃止）
    adv = st.session_state.get("_adv", {})

    # Tab1の「解析開始」ボタンから遷移していない場合
    st.markdown("""
<div style="text-align:center; padding:3rem; color:#555;">
  <div style="font-size:3rem;">🔬</div>
  <div style="margin-top:1rem; color:#8888aa;">
    「① データ設定」タブで設定を完了し、<br>下部の「🚀 解析開始」ボタンを押してください。
  </div>
  <div style="margin-top:0.5rem; font-size:0.85rem; color:#666;">
    設定が完了していれば、以下のボタンからも直接実行できます。
  </div>
</div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        run_clicked = st.button(
            "🚀 現在の設定で実行",
            use_container_width=True,
            key="run_full",
            type="primary",
        )

    if not run_clicked:
        return

    # ── adv設定からパイプライン実行 ─────────────────────────────
    smiles_col = st.session_state.get("smiles_col")
    task_override = st.session_state.get("task", "auto")
    _run_full_pipeline(
        df=df,
        target_col=target_col,
        smiles_col=smiles_col,
        numeric_scaler=adv.get("scaler", "auto"),
        task_override=task_override,
        cv_folds=adv.get("cv_folds", 5),
        models=adv.get("models", []),
        timeout=adv.get("timeout", 300),
        do_eda=adv.get("do_eda", True),
        do_prep=adv.get("do_prep", True),
        do_ml=True,
        do_eval=adv.get("do_eval", True),
        do_pca=adv.get("do_pca", True),
        do_shap=adv.get("do_shap", True),
        selected_descriptors=adv.get("selected_descriptors"),
        monotonic_constraints_dict=st.session_state.get("_monotonic_constraints_dict", {}),
    )


# ─────────────────────────────────────────────────────────────
# フルパイプライン実行
# ─────────────────────────────────────────────────────────────

def _run_full_pipeline(
    df: pd.DataFrame,
    target_col: str,
    *,
    smiles_col: str | None,
    # 前処理設定
    numeric_scaler: str = "auto",
    numeric_imputer: str = "mean",
    add_missing_indicator: bool = True,
    cat_low_encoder: str = "onehot",
    cat_high_encoder: str = "ordinal",
    categorical_imputer: str = "most_frequent",
    # CV設定
    task_override: str = "auto",
    cv_key: str = "auto",
    cv_params: dict | None = None,
    cv_groups_col: str | None = None,
    cv_folds: int = 5,
    # モデル設定
    models: list[str] | None = None,
    model_params: dict[str, dict] | None = None,
    # その他
    timeout: int = 300,
    do_eda: bool = True,
    do_prep: bool = True,
    do_ml: bool = True,
    do_eval: bool = True,
    do_pca: bool = True,
    do_shap: bool = True,
    selected_descriptors: list[str] | None = None,
    extra_params: dict[str, Any] | None = None,
    monotonic_constraints_dict: dict[str, int] | None = None,
) -> None:
    """Full pipeline. All CV, preprocessor, and model settings are fully configurable."""
    extra_params = extra_params or {}
    TOTAL_PHASES = sum([do_eda, do_prep, do_ml, do_eval, do_pca, do_shap])
    # SMILES変換はAutoMLEngine.run()内で処理するため、独立フェーズとして数えない

    result = PipelineResult()
    start_time = time.time()

    progress_bar = st.progress(0.0)
    phase_status = st.empty()
    log_area     = st.empty()
    log_lines: list[str] = []
    phase_done   = 0

    def _log(msg: str) -> None:
        log_lines.append(msg)
        log_area.markdown(
            "<br>".join(
                f'<span style="color:#8888aa;font-size:0.82rem;">{l}</span>'
                for l in log_lines[-6:]
            ),
            unsafe_allow_html=True,
        )

    def _advance(phase_name: str) -> None:
        nonlocal phase_done
        phase_done += 1
        progress_bar.progress(phase_done / max(TOTAL_PHASES, 1))
        phase_status.markdown(f"**▶ {phase_name}**")
        _log(f"✅ {phase_name} 完了")

    try:
        # ── Phase 0: SMILES記述子変換 ────────────────────
        # NOTE: Phase 0でSMILES変換は行わない。
        # AutoMLEngine.run()内でSmilesDescriptorTransformerを使って変換する。
        # 二重変換（Phase 0 + engine.run()内）が解析失敗の主因だったため廃止。
        # smiles_col はそのままengine.run()に渡す。

        # ── Phase 1: EDA ─────────────────────────────────
        if do_eda:
            phase_status.markdown("**Phase 1/6 — EDA 実行中…**")
            result.eda_summary  = summarize_dataframe(df)
            result.col_stats    = compute_column_stats(df)
            result.outlier_results = detect_outliers(df, method="iqr")
            _advance("Phase 1: EDA")

        # ── Phase 2: 前処理確認 ───────────────────────────
        if do_prep:
            phase_status.markdown("**Phase 2/6 — 前処理パイプライン構築中…**")
            from backend.data.type_detector import TypeDetector as _TypeDetector
            detector = _TypeDetector()
            dr = detector.detect(df)
            st.session_state["detection_result"] = dr
            st.session_state["step_preprocess_done"] = True
            _advance("Phase 2: 前処理")

        # ── Phase 3: AutoML ──────────────────────────────
        if do_ml:
            phase_status.markdown("**Phase 3/6 — AutoML 実行中（しばらくお待ちください）…**")
            automl_log: list[str] = []

            def _cb(step: int, total: int, msg: str) -> None:
                progress_bar.progress(
                    (phase_done / max(TOTAL_PHASES, 1))
                    + (step / total) / max(TOTAL_PHASES, 1)
                )
                _log(f"  [{step}/{total}] {msg}")

            cfg = PreprocessConfig(
                numeric_scaler=numeric_scaler,
                numeric_imputer=numeric_imputer,
                add_missing_indicator=add_missing_indicator,
                cat_low_encoder=cat_low_encoder,
                cat_high_encoder=cat_high_encoder,
                categorical_imputer=categorical_imputer,
            )
            engine = AutoMLEngine(
                task=task_override,
                cv_folds=cv_folds,
                cv_key=cv_key,
                cv_groups_col=cv_groups_col,
                model_keys=models if models else None,
                model_params=model_params or {},
                timeout_seconds=timeout,
                progress_callback=_cb,
                selected_descriptors=selected_descriptors,
                monotonic_constraints_dict=monotonic_constraints_dict or {},
            )
            automl_res = engine.run(
                df, target_col=target_col, smiles_col=smiles_col,
                preprocess_config=cfg,
                cv_extra_params=cv_params or {},
            )
            result.automl_result = automl_res
            result.task = automl_res.task if hasattr(automl_res, "task") else task_override
            st.session_state["automl_result"] = automl_res
            # SMILES相関を結果に保持（AutoMLEngine内で計算済みの場合）
            if hasattr(automl_res, 'smiles_transformer') and automl_res.smiles_transformer is not None:
                result.smiles_transformer = automl_res.smiles_transformer
            if hasattr(automl_res, 'smiles_correlations') and automl_res.smiles_correlations:
                result.smiles_correlations = automl_res.smiles_correlations
            if automl_res.warnings:
                result.warnings.extend(automl_res.warnings)
            _advance("Phase 3: AutoML")

        # X_base: AutoML側で目的変数の欠損除外等が適用された「実際に学習に使われたデータ」
        if result.automl_result and result.automl_result.processed_X is not None:
            X_base = result.automl_result.processed_X
            y_base = result.automl_result.oof_true if result.automl_result.oof_true is not None else df.loc[X_base.index, target_col].values
        else:
            X_base = df.drop(columns=[target_col])
            y_base = df[target_col].values

        # ── Phase 4: 評価 ────────────────────────────────
        if do_eval and result.automl_result is not None:
            phase_status.markdown("**Phase 4/6 — モデル評価中…**")
            try:
                ar = result.automl_result
                bp: Pipeline = ar.best_pipeline
                y = y_base

                y_pred = bp.predict(X_base)
                task_str = result.task if result.task in ("regression", "classification") else "regression"
                if task_str == "regression":
                    result.model_score = evaluate_regression(y, y_pred,
                                                             model_key=ar.best_model_key,
                                                             cv_mean=ar.best_score)
                else:
                    y_prob = bp.predict_proba(X_base) if hasattr(bp, "predict_proba") else None
                    result.model_score = evaluate_classification(y, y_pred, y_prob=y_prob,
                                                                 model_key=ar.best_model_key,
                                                                 cv_mean=ar.best_score)
            except Exception as e:
                result.warnings.append(f"評価エラー: {e}")
                _log(f"⚠️ 評価スキップ: {e}")
            _advance("Phase 4: 評価")

        # ── Phase 5: PCA 次元削減 ────────────────────────
        if do_pca:
            phase_status.markdown("**Phase 5/6 — PCA 次元削減中…**")
            try:
                # 数値列が2列以上あれば実行可能（SMILES記述子も含める）
                if X_base.select_dtypes(include="number").shape[1] >= 2:
                    from backend.data.dim_reduction import DimReductionConfig as _DimReductionConfig
                    from backend.data.dim_reduction import DimReducer as _DimReducer
                    
                    pca_cfg = _DimReductionConfig(
                        method="pca", n_components=2, 
                        method_params=extra_params.get("pca", {})
                    )
                    reducer = _DimReducer(pca_cfg)
                    emb_df = reducer.fit_transform(X_base)
                    
                    result.pca_df  = pd.DataFrame(emb_df, columns=["PC1", "PC2"], index=X_base.index)
                    result.pca_evr = reducer.explained_variance_ratio_
            except Exception as e:
                result.warnings.append(f"PCAエラー: {e}")
                _log(f"⚠️ PCAスキップ: {e}")
            _advance("Phase 5: 次元削減")

        # ── Phase 6: SHAP 解析 ──────────────────────────
        if do_shap and result.automl_result is not None:
            phase_status.markdown("**Phase 6/6 — SHAP 特徴量重要度計算中…**")
            try:
                ar = result.automl_result
                bp: Pipeline = ar.best_pipeline
                # Pipelineの最終ステップ(model)から重要度を取得
                final_estimator = bp[-1] if hasattr(bp, '__getitem__') else bp

                # 変換後の特徴量名を取得
                try:
                    preprocessor_step = bp[:-1]  # model以外
                    X_transformed = preprocessor_step.transform(X_base)
                    try:
                        feat_names = bp[:-1].get_feature_names_out().tolist()
                    except Exception:
                        feat_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
                except Exception:
                    feat_names = X_base.select_dtypes(
                        include="number"
                    ).columns.tolist()

                if hasattr(final_estimator, "feature_importances_"):
                    imps = final_estimator.feature_importances_
                    names = feat_names[:len(imps)]
                    result.shap_importances = dict(
                        sorted(zip(names, imps), key=lambda x: x[1], reverse=True)
                    )
                elif hasattr(final_estimator, "coef_"):
                    coefs = np.abs(final_estimator.coef_.ravel())
                    names = feat_names[:len(coefs)]
                    result.shap_importances = dict(
                        sorted(zip(names, coefs), key=lambda x: x[1], reverse=True)
                    )
                else:
                    # SHAP KernelExplainer（低速・フォールバック）
                    import shap
                    if 'X_transformed' in locals() and X_transformed is not None:
                        # 変換済みの特徴量と最終モデルを使う
                        X_s = pd.DataFrame(X_transformed, columns=feat_names[:X_transformed.shape[1]]) if isinstance(X_transformed, np.ndarray) else X_transformed.copy()
                        if not hasattr(X_s, "columns"):
                            X_shape = X_s.shape[1] if hasattr(X_s, 'shape') else len(X_s[0])
                            X_s = pd.DataFrame(X_s, columns=[f"f{i}" for i in range(X_shape)])
                        
                        predict_fn = final_estimator.predict
                    else:
                        X_s = X_base.select_dtypes(include="number").fillna(0)
                        
                        # Pipelineに渡す場合、SHAPが内部でNumPy化するため元に戻すラッパーを噛ませる
                        def _wrapped_predict(X_arr):
                            df_in = pd.DataFrame(X_arr, columns=X_s.columns)
                            return bp.predict(df_in)
                            
                        predict_fn = _wrapped_predict

                    X_s = X_s.fillna(0)
                    explainer = shap.KernelExplainer(
                        predict_fn, shap.sample(X_s, min(50, len(X_s)))
                    )
                    shap_vals = explainer.shap_values(X_s.head(50))
                    
                    if isinstance(shap_vals, list):
                        # 分類の場合、クラス1の重要度
                        val_arr = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
                    else:
                        val_arr = shap_vals
                    mean_abs = np.abs(val_arr).mean(axis=0)
                    
                    cols = X_s.columns.tolist() if hasattr(X_s, "columns") else feat_names[:len(mean_abs)]
                    result.shap_importances = dict(
                        sorted(zip(cols, mean_abs), key=lambda x: x[1], reverse=True)
                    )
            except Exception as e:
                result.warnings.append(f"SHAP計算エラー: {e}")
                _log(f"⚠️ SHAPスキップ: {e}")
            _advance("Phase 6: SHAP")

    except Exception as e:
        progress_bar.empty()
        phase_status.empty()
        st.error(f"❌ パイプライン実行中にエラーが発生しました: {e}")
        st.info("データに予測可能な特徴量が含まれていない、もしくは全ての列が定数などにより前処理で除外された可能性があります。")
        
        with st.expander("🛠️ エラー詳細（スタックトレース）"):
            st.code(traceback.format_exc(), language="python")
            
        # エラー発生時点でのパイプライン構成図（もし構築されていれば）
        ar = result.automl_result
        if ar and hasattr(ar, "best_pipeline") and ar.best_pipeline:
            with st.expander("🔍 エラー時の暫定パイプライン構成図"):
                try:
                    from sklearn.utils import estimator_html_repr
                    import streamlit.components.v1 as components
                    html_repr = estimator_html_repr(ar.best_pipeline)
                    components.html(html_repr, height=400, scrolling=True)
                except Exception:
                    st.warning("パイプライン構成図の生成に失敗しました。")
        return

    result.elapsed = time.time() - start_time
    st.session_state["pipeline_result"] = result
    st.session_state["step_eda_done"] = do_eda
    st.session_state["step_preprocess_done"] = do_prep

    progress_bar.progress(1.0)
    phase_status.empty()
    log_area.empty()
    st.balloons()
    st.rerun()


# ─────────────────────────────────────────────────────────────
# 結果表示
# ─────────────────────────────────────────────────────────────

def _show_pipeline_result(pr: PipelineResult) -> None:
    """PipelineResult を6フェーズタブで表示する。"""
    ar = pr.automl_result

    # ヘッダー
    best_info = (
        f"最良モデル: **{ar.best_model_key}** | スコア: `{ar.best_score:.4f}`"
        if ar else "AutoML は実行されていません"
    )
    st.success(f"✅ パイプライン完了！ ({pr.elapsed:.1f}秒) | {best_info}")

    for w in pr.warnings:
        st.warning(f"⚠️ {w}")

    # タブ
    tabs = st.tabs(["📊 EDA", "⚙️ 前処理", "🤖 AutoML", "📈 評価", "📐 次元削減", "💡 SHAP", "🧪 SMILES相関"])

    # ── Tab 0: EDA ──────────────────────────────────────────
    with tabs[0]:
        st.markdown("### 📊 データ探索サマリー")
        if pr.eda_summary:
            summ = pr.eda_summary
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("行数", f"{summ.get('n_rows', '-'):,}")
            c2.metric("列数", summ.get('n_cols', '-'))
            c3.metric("欠損セル数", summ.get('n_missing', '-'))
            c4.metric("重複行数", summ.get('n_duplicates', '-'))

        if pr.col_stats:
            st.markdown("#### 列ごとの統計")
            rows = []
            for cs in pr.col_stats:
                rows.append({
                    "列名": cs.name,
                    "型": cs.dtype,
                    "ユニーク数": cs.n_unique,
                    "欠損率": f"{cs.null_rate:.1%}",
                    "平均": f"{cs.mean:.3g}" if cs.mean is not None else "-",
                    "標準偏差": f"{cs.std:.3g}" if cs.std is not None else "-",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if pr.outlier_results:
            n_out_cols = sum(1 for r in pr.outlier_results if r.n_outliers > 0)
            st.info(f"IQR外れ値検出: **{n_out_cols}列** に外れ値あり")
        else:
            st.info("EDA は実行されていません。⚙️ 設定で有効化してください。")

    # ── Tab 1: 前処理 ────────────────────────────────────────
    with tabs[1]:
        st.markdown("### ⚙️ 前処理パイプライン")
        dr = st.session_state.get("detection_result")
        if dr:
            col_types: dict[str, list[str]] = {}
            for col, info in dr.column_info.items():
                t = info.col_type.value if hasattr(info.col_type, "value") else str(info.col_type)
                col_types.setdefault(t, []).append(col)

            rows = [{"列型": t, "列数": len(cols), "列名": ", ".join(cols[:5]) + ("..." if len(cols) > 5 else "")}
                    for t, cols in col_types.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("TypeDetector によって自動で各列の型が判定され、最適なスケーラー/エンコーダが選定されます。")
        else:
            st.info("前処理は実行されていませんでした。⚙️ 設定で有効化してください。")

    # ── Tab 2: AutoML ────────────────────────────────────────
    with tabs[2]:
        if ar:
            _show_leaderboard(ar)
        else:
            st.info("AutoML は実行されていません。⚙️ 設定で有効化してください。")

    # ── Tab 3: 評価 ─────────────────────────────────────────
    with tabs[3]:
        st.markdown("### 📈 モデル評価")
        score = pr.model_score
        if score is not None:
            # to_dict() はNoneフィールドを自動除外する
            score_dict = {
                k: v for k, v in score.to_dict().items()
                if k not in ("model_key", "task")
            }
            label_map = {
                "rmse": "RMSE", "mae": "MAE", "r2": "R²",
                "accuracy": "Accuracy", "f1_weighted": "F1 (weighted)",
                "roc_auc": "ROC-AUC", "cv_mean": "CV Mean",
                "cv_std": "CV Std", "train_time": "学習時間(s)",
            }
            c_cols = st.columns(min(len(score_dict), 4))
            for i, (k, v) in enumerate(score_dict.items()):
                with c_cols[i % 4]:
                    st.metric(label_map.get(k, k.upper()),
                              f"{v:.4f}" if isinstance(v, float) else str(v))
        else:
            st.info("評価は実行されていません。⚙️ 設定で有効化するか、先にAutoMLを実行してください。")

    # ── Tab 4: 次元削減 ─────────────────────────────────────
    with tabs[4]:
        st.markdown("### 📐 PCA 次元削減")
        if pr.pca_df is not None and pr.pca_evr is not None:
            c_p1, c_p2 = st.columns([1, 2])
            with c_p1:
                st.info(f"PC1 寄与率: {pr.pca_evr[0]:.1%}\nPC2 寄与率: {pr.pca_evr[1]:.1%}")
            with c_p2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pr.pca_df["PC1"], y=pr.pca_df["PC2"],
                    mode="markers",
                    marker=dict(size=8, color="#2E86B1", line=dict(width=1, color="white"))
                ))
                fig.update_layout(title="PCA Projection", xaxis_title="PC1", yaxis_title="PC2")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("次元削減は実行されていません。⚙️ 設定で有効化するか、2列以上の特徴量が必要です。")

    # ── Tab 5: SHAP ─────────────────────────────────────────
    with tabs[5]:
        st.markdown("### 💡 SHAP / 特徴量重要度")
        if pr.shap_importances:
            imp_df = pd.DataFrame(list(pr.shap_importances.items()), columns=["Feature", "Importance"])
            imp_df = imp_df.sort_values(by="Importance", ascending=True).tail(20) # Top 20
            
            fig = go.Figure(go.Bar(
                x=imp_df["Importance"], y=imp_df["Feature"],
                orientation='h', marker_color="#EF7C8E"
            ))
            fig.update_layout(title="Top 20 最重要特徴量", xaxis_title="Mean |SHAP Value| or Feature Importance", height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("SHAP解析は実行されていません。⚙️ 設定で有効化するか、サポート外のモデルです。")
            
    # ── Tab 6: SMILES 相関 ──────────────────────────────────
    with tabs[6]:
        if pr and hasattr(pr, "smiles_correlations") and pr.smiles_correlations:
            st.markdown("### 🧪 SMILES 記述子と目的変数の相関ランキング")
            st.caption("SMILESから抽出された各記述子と目的変数とのピアソン相関係数（絶対値順にランキング表示）")
            
            # DataFrame化して絶対値でソート
            corr_dict = pr.smiles_correlations
            df_corr = pd.DataFrame([
                {"記述子": k, "相関係数": v, "絶対値": abs(v)}
                for k, v in corr_dict.items()
            ]).sort_values("絶対値", ascending=False).drop(columns=["絶対値"]).reset_index(drop=True)
            
            # ランキングを見やすく表示
            df_corr.index = df_corr.index + 1
            st.dataframe(df_corr.style.background_gradient(subset=["相関係数"], cmap="coolwarm", vmin=-1.0, vmax=1.0), use_container_width=True)
        else:
            st.info("SMILES抽出が行われていないか、相関係数の計算ができませんでした。")


    # ── 次のアクションボタン ─────────────────────────────────
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📊 モデル評価（詳細）", use_container_width=True, key="goto_eval"):
            st.session_state["page"] = "evaluation"
            st.rerun()
    with c2:
        if st.button("💡 SHAP 詳細解析", use_container_width=True, key="goto_shap"):
            st.session_state["page"] = "interpret"
            st.rerun()
    with c3:
        if st.button("📐 次元削減（詳細）", use_container_width=True, key="goto_dimred"):
            st.session_state["page"] = "dim_reduction"
            st.rerun()


# ─────────────────────────────────────────────────────────────
# リーダーボード表示（共通）
# ─────────────────────────────────────────────────────────────

def _show_leaderboard(result: AutoMLResult) -> None:
    """モデルリーダーボードとスコアバーチャートを表示する。"""
    st.markdown("### 🏆 モデルリーダーボード")
    scores  = result.model_scores
    details = result.model_details

    df_lb = pd.DataFrame([
        {
            "ランク": i + 1,
            "モデル": k,
            "スコア（平均）": f"{v:.4f}",
            "標準偏差": f"{details[k]['std']:.4f}" if k in details else "-",
            "学習時間(s)": f"{details[k]['fit_time']:.2f}" if k in details else "-",
            "最良": "🏆" if k == result.best_model_key else "",
        }
        for i, (k, v) in enumerate(
            sorted(scores.items(), key=lambda x: x[1], reverse=True)
        )
    ])
    st.dataframe(df_lb, use_container_width=True, hide_index=True)

    st.markdown("### 📊 スコア比較")
    sorted_items = sorted(scores.items(), key=lambda x: x[1])
    keys = [k for k, _ in sorted_items]
    vals = [v for _, v in sorted_items]
    colors = ["#7b2ff7" if k == result.best_model_key else "#00d4ff" for k in keys]

    fig = go.Figure(go.Bar(
        x=vals, y=keys, orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0f0"),
        xaxis=dict(gridcolor="#333", title=result.scoring),
        yaxis=dict(gridcolor="#333"),
        height=max(300, len(keys) * 40),
        margin=dict(l=120, r=50, t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── CVバイアス評価（オプショナル） ─────────────────────────
    # fold別スコアが存在する場合のみTT法バイアス推定を表示
    has_fold_scores = False
    if len(scores) >= 2:
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        param_labels = [k for k, _ in sorted_scores]
        fold_curves = []
        for key in param_labels:
            d = details.get(key, {})
            if "fold_scores" in d and d["fold_scores"]:
                fold_curves.append(d["fold_scores"])
        has_fold_scores = (
            len(fold_curves) == len(param_labels)
            and all(len(f) == len(fold_curves[0]) for f in fold_curves)
        )

    if has_fold_scores:
        with st.expander("📐 CVバイアス推定", expanded=False):
            try:
                from backend.models.cv_bias_evaluator import estimate_tibshirani_bias
                import numpy as _np

                curves = _np.array(fold_curves).T  # (K folds, P models)
                tt_result = estimate_tibshirani_bias(
                    curves, param_values=None, higher_is_better=True
                )
                st.markdown(f"""
| 指標 | 値 |
|------|-----|
| **CVスコア** | `{tt_result.raw_score:.4f}` |
| **推定バイアス** | `{tt_result.bias_estimate:+.4f}` |
| **補正後スコア** | `{tt_result.corrected_score:.4f}` |
""")
                st.caption(
                    "複数モデルから最良を選ぶ際に生じるスコアの過大評価を補正した値です。"
                )
            except Exception as e:
                st.caption(f"バイアス推定をスキップしました: {e}")

