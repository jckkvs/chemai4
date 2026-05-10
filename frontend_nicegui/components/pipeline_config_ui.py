# -*- coding: utf-8 -*-
"""
frontend_nicegui/components/pipeline_config_ui.py

Pipeline 全設定UI — NiceGUI版
7ステップのタブ形式で全パイプラインを設定。
複数選択 = 全組み合わせを評価 / 未選択 = 適切なデフォルトを自動適用。

Streamlit版のpipeline_config_ui.pyと機能等価。
"""
from __future__ import annotations

from typing import Any
from nicegui import ui
from frontend_nicegui.components.monotonicity_config import render_monotonicity_config


# ═══════════════════════════════════════════════════════════
# ヘルパー
# ═══════════════════════════════════════════════════════════

def _section(icon: str, title: str, desc: str = "") -> None:
    """セクションヘッダー。"""
    ui.label(f"{icon} {title}").classes("text-subtitle1 text-bold q-mt-sm")
    if desc:
        ui.label(desc).classes("text-caption text-grey q-mb-xs").style("font-size:0.75rem;")


def _glass_card():
    """ガラスカードのコンテキストマネージャー。"""
    return ui.card().classes("full-width q-pa-sm q-mb-xs").style(
        "border:1px solid rgba(0,188,212,0.2); border-radius:8px;"
        "background:rgba(0,20,40,0.25);"
    )


# ═══════════════════════════════════════════════════════════
# モデル分類
# ═══════════════════════════════════════════════════════════

_MODEL_CATEGORIES = [
    ("📐 線形系", "linear_scale"),
    ("🌲 決定木/アンサンブル", "park"),
    ("🔮 カーネル系", "blur_on"),
    ("🧩 その他", "extension"),
]


def _classify_model(m: dict) -> str:
    """モデル辞書からカテゴリ名を判定する。

    分類優先順位:
    1. カーネル系を先に判定（SVR linear等はkernelを使うため）
    2. 決定木系（LinearTree/LinearForestを含む）
    3. 線形系（TheilSen等含む）
    4. その他（KNN, MLP, RANSAC等）
    """
    k = (m.get("key", "") + m.get("name", "")).lower()

    # カーネル系: SVR, SVC, GPR, KernelRidge（SVR linearも含む）
    if any(x in k for x in ["svr", "svc", "gpr", "gaussian", "kernel_ridge", "kernelridge"]):
        return "🔮 カーネル系"

    # 決定木/アンサンブル: tree, forest, boost, gbm, bagging 等
    # LinearTree, LinearForest も木系に含む
    if any(x in k for x in [
        "tree", "forest", "boost", "gbm", "gradient", "rgf",
        "figs", "rule", "hist", "catboost", "lineartree", "linearforest",
        "linear_tree", "linear_forest", "bagging",
    ]):
        return "🌲 決定木/アンサンブル"

    # 線形系: Ridge, Lasso, ElasticNet, TheilSen, ARD, Huber, PLS, Bayesian 等
    if any(x in k for x in [
        "ridge", "lasso", "elastic", "logistic", "ard", "huber",
        "pls", "bayesian", "theilsen", "theil", "linear",
    ]):
        return "📐 線形系"

    # その他: KNN, MLP, RANSAC 等
    return "🧩 その他"


# モデルメタデータ: 短縮名 / 計算量 / ライブラリ
# key → (short_name, complexity, library)
_MODEL_META: dict[str, tuple[str, str, str]] = {
    # ─── 線形系 ───
    "linear":       ("OLS",          "O(nd²)",        "sklearn"),
    "ridge":        ("Ridge",        "O(nd²)",        "sklearn"),
    "ridge_cv":     ("RidgeCV",      "O(nd²·k)",      "sklearn"),
    "lasso":        ("Lasso",        "O(nd)",         "sklearn"),
    "lasso_cv":     ("LassoCV",      "O(nd·k)",       "sklearn"),
    "elasticnet":   ("EN",           "O(nd)",         "sklearn"),
    "elasticnet_cv":("EN-CV",        "O(nd·k)",       "sklearn"),
    "bayesian_ridge":("BayesRidge",  "O(nd²)",        "sklearn"),
    "ard":          ("ARD",          "O(nd²)",        "sklearn"),
    "huber":        ("Huber",        "O(nd)",         "sklearn"),
    "theilsen":     ("TheilSen",     "O(n²d)",        "sklearn"),
    "pls":          ("PLS",          "O(ndk)",        "sklearn"),
    "logistic":     ("Logistic",     "O(nd)",         "sklearn"),
    # ─── 決定木/アンサンブル ───
    "dt":           ("DT",           "O(nd·log n)",   "sklearn"),
    "rf":           ("RF",           "O(T·nd·log n)", "sklearn"),
    "et":           ("ExtraTrees",   "O(T·nd)",       "sklearn"),
    "gbm":          ("GBM",          "O(T·nd·log n)", "sklearn"),
    "hgbm":         ("HistGBM",      "O(T·n·log n)",  "sklearn"),
    "xgb":          ("XGBoost",      "O(T·nd·log n)", "xgboost"),
    "lgbm":         ("LightGBM",     "O(T·n·log n)",  "lightgbm"),
    "catboost":     ("CatBoost",     "O(T·nd)",       "catboost"),
    "xgbrf":        ("XGB-RF",       "O(T·nd·log n)", "xgboost"),
    "adaboost":     ("AdaBoost",     "O(T·nd)",       "sklearn"),
    "bagging":      ("Bagging",      "O(T·nd·log n)", "sklearn"),
    "lineartree":   ("LinearTree",   "O(nd²·log n)",  "scratch"),
    "linearforest": ("LinForest",    "O(T·nd²)",      "scratch"),
    "linearboost":  ("LinBoost",     "O(T·nd²)",      "scratch"),
    "ridgetree":    ("RidgeTree",    "O(nd²·log n)",  "scratch"),
    "rgf":          ("RGF",          "O(n·d·T)",      "scratch"),
    "figs":         ("FIGS",         "O(nd·R)",       "imodels"),
    "hstree":       ("HS-Tree",      "O(nd·log n)",   "imodels"),
    "rulefit":      ("RuleFit",      "O(nd·R)",       "imodels"),
    "greedytree":   ("GreedyTree",   "O(nd·log n)",   "imodels"),
    # ─── カーネル系 ───
    "svr_rbf":      ("SVR-RBF",      "O(n²d~n³)",     "sklearn"),
    "svr_linear":   ("SVR-Lin",      "O(n²d)",        "sklearn"),
    "gp":           ("GPR",          "O(n³)",         "sklearn"),
    "svc_rbf":      ("SVC-RBF",      "O(n²d~n³)",     "sklearn"),
    "linearsvc":    ("LinearSVC",    "O(nd)",         "sklearn"),
    # ─── その他 ───
    "knn":          ("KNN",          "O(nd)予測時",    "sklearn"),
    "knn_c":        ("KNN",          "O(nd)予測時",    "sklearn"),
    "mlp":          ("MLP",          "O(n·d·h·e)",    "sklearn"),
    "mlp_c":        ("MLP",          "O(n·d·h·e)",    "sklearn"),
    "ransac":       ("RANSAC",       "O(k·n·d)",      "sklearn"),
}


def _get_model_meta(key: str) -> tuple[str, str, str]:
    """モデルキーからメタデータを取得。未登録の場合はデフォルト値。"""
    return _MODEL_META.get(key, (key, "不明", "?"))


def _categorize_models(available: list[dict]) -> dict[str, list]:
    """モデルリストをカテゴリ辞書に分類する。factory.pyの登録順を維持。"""
    categories: dict[str, list] = {name: [] for name, _ in _MODEL_CATEGORIES}
    for m in available:
        cat = _classify_model(m)
        categories[cat].append(m)
    return categories


# ═══════════════════════════════════════════════════════════
# Tab 0: Excluder
# ═══════════════════════════════════════════════════════════

def _tab_excluder(state: dict) -> None:
    _section("🚫", "Excluder（解析除外列）",
             "解析に使わない列を選択。目的変数・SMILES列は自動除外済み。")
    df = state.get("df")
    if df is None:
        ui.label("データ未読み込み").classes("text-caption text-grey")
        return
    target_col = state.get("target_col", "")
    smiles_col = state.get("smiles_col", "")
    skip = {c for c in (target_col, smiles_col) if c}
    opts = [c for c in df.columns if c not in skip]
    if not opts:
        ui.label("除外できる列がありません").classes("text-caption text-grey")
        return
    prev = state.get("exclude_cols", [])

    def _on_change(e):
        state["exclude_cols"] = list(e.value)
    ui.select(
        opts, multiple=True, value=[c for c in prev if c in opts],
        label="除外列を選択",
        on_change=_on_change,
    ).props("use-chips outlined dense").classes("full-width")


# ═══════════════════════════════════════════════════════════
# Tab 1: 数値前処理
# ═══════════════════════════════════════════════════════════

_NUM_IMPUTERS = [
    ("mean", "Mean（平均）"),
    ("median", "Median（中央値）"),
    ("knn", "KNN Imputer"),
    ("iterative", "Iterative Imputer"),
    ("constant", "Constant（固定値）"),
]

_NUM_SCALERS = [
    ("standard", "StandardScaler"),
    ("minmax", "MinMaxScaler"),
    ("robust", "RobustScaler"),
    ("maxabs", "MaxAbsScaler"),
    ("power_yj", "PowerTransformer [YJ]"),
    ("power_bc", "PowerTransformer [BC]"),
    ("quantile_normal", "QuantileTransformer→正規"),
    ("quantile_uniform", "QuantileTransformer→一様"),
    ("none", "スケーリングなし"),
]


def _tab_numeric(state: dict) -> None:
    _section("🔢", "数値列前処理（Imputer × Scaler）",
             "選択した Imputer と Scaler の全組み合わせを評価します。")

    # ── Imputer ──
    ui.label("📊 Imputer（欠損補間）").classes("text-body2 text-bold q-mt-sm")
    imp_keys = state.get("_pg_num_imputers", ["mean"])
    imp_options = [k for k, _ in _NUM_IMPUTERS]
    imp_labels = {k: v for k, v in _NUM_IMPUTERS}

    def _on_imp(e):
        state["_pg_num_imputers"] = list(e.value)
    ui.select(
        imp_options, multiple=True, value=imp_keys,
        label="Imputer（複数選択可）",
        on_change=_on_imp,
    ).props("use-chips outlined dense").classes("full-width")

    ui.separator().classes("q-my-xs")

    # ── Scaler ──
    ui.label("📏 Scaler").classes("text-body2 text-bold")
    scl_keys = state.get("_pg_num_scalers", ["standard"])
    scl_options = [k for k, _ in _NUM_SCALERS]

    def _on_scl(e):
        state["_pg_num_scalers"] = list(e.value)
    ui.select(
        scl_options, multiple=True, value=scl_keys,
        label="Scaler（複数選択可）",
        on_change=_on_scl,
    ).props("use-chips outlined dense").classes("full-width")


# ═══════════════════════════════════════════════════════════
# Tab 2: カテゴリ前処理
# ═══════════════════════════════════════════════════════════

_CAT_IMPUTERS = [
    ("most_frequent", "Most Frequent（最頻値）"),
    ("constant", "Constant（指定文字列）"),
    ("knn", "KNN Imputer"),
]

_LOW_ENCODERS = [
    ("onehot", "OneHotEncoder"),
    ("ordinal", "OrdinalEncoder"),
    ("target", "TargetEncoder"),
    ("binary", "BinaryEncoder"),
]

_HIGH_ENCODERS = [
    ("ordinal", "OrdinalEncoder"),
    ("target", "TargetEncoder"),
    ("hashing", "HashingEncoder"),
    ("binary", "BinaryEncoder"),
    ("leaveoneout", "LeaveOneOut"),
]


def _tab_categorical(state: dict) -> None:
    _section("🏷️", "カテゴリ列前処理（Imputer × Encoder）",
             "低カーディナリティと高カーディナリティで別設定可能。")

    # Imputer
    ui.label("🔤 Categorical Imputer").classes("text-body2 text-bold q-mt-sm")
    ci_keys = state.get("_pg_cat_imputers", ["most_frequent"])
    ci_options = [k for k, _ in _CAT_IMPUTERS]

    def _on_ci(e):
        state["_pg_cat_imputers"] = list(e.value)
    ui.select(ci_options, multiple=True, value=ci_keys,
              label="Imputer", on_change=_on_ci,
              ).props("use-chips outlined dense").classes("full-width")

    ui.separator().classes("q-my-xs")

    # Low / High Encoder
    with ui.row().classes("full-width q-gutter-sm"):
        with ui.column().classes("col"):
            ui.label("🔻 低カーディナリティ").classes("text-body2 text-bold")
            le_keys = state.get("_pg_low_encoders", ["onehot"])
            le_options = [k for k, _ in _LOW_ENCODERS]

            def _on_le(e):
                state["_pg_low_encoders"] = list(e.value)
            ui.select(le_options, multiple=True, value=le_keys,
                      label="Encoder", on_change=_on_le,
                      ).props("use-chips outlined dense").classes("full-width")

        with ui.column().classes("col"):
            ui.label("🔺 高カーディナリティ").classes("text-body2 text-bold")
            he_keys = state.get("_pg_high_encoders", ["ordinal"])
            he_options = [k for k, _ in _HIGH_ENCODERS]

            def _on_he(e):
                state["_pg_high_encoders"] = list(e.value)
            ui.select(he_options, multiple=True, value=he_keys,
                      label="Encoder", on_change=_on_he,
                      ).props("use-chips outlined dense").classes("full-width")


# ═══════════════════════════════════════════════════════════
# Tab 3: バイナリ前処理
# ═══════════════════════════════════════════════════════════

def _tab_binary(state: dict) -> None:
    _section("⚡", "バイナリ列前処理",
             "0/1, True/False などの2値列の処理設定。")
    bi_imps = state.get("_pg_bin_imputers", ["most_frequent"])

    def _on_bi(e):
        state["_pg_bin_imputers"] = list(e.value)
    ui.select(
        ["most_frequent", "constant", "knn"], multiple=True, value=bi_imps,
        label="Imputer", on_change=_on_bi,
    ).props("use-chips outlined dense").classes("full-width")

    be_keys = state.get("_pg_bin_encoders", ["ordinal"])

    def _on_be(e):
        state["_pg_bin_encoders"] = list(e.value)
    ui.select(
        ["ordinal", "passthrough"], multiple=True, value=be_keys,
        label="Encoder", on_change=_on_be,
    ).props("use-chips outlined dense").classes("full-width")


# ═══════════════════════════════════════════════════════════
# Tab 4: 特徴生成
# ═══════════════════════════════════════════════════════════

def _tab_engineer(state: dict) -> None:
    _section("🔧", "Feature Engineering",
             "複数選択で全パターンを評価。未選択 → none。")
    eng_keys = state.get("_pg_engineer", ["none"])

    def _on_eng(e):
        state["_pg_engineer"] = list(e.value)
    ui.select(
        ["none", "polynomial", "interaction_only"],
        multiple=True, value=eng_keys,
        label="生成手法", on_change=_on_eng,
    ).props("use-chips outlined dense").classes("full-width")

    # Polynomial パラメータ
    if "polynomial" in state.get("_pg_engineer", []):
        with _glass_card():
            ui.label("PolynomialFeatures 設定").classes("text-caption text-bold")
            with ui.row().classes("q-gutter-sm"):
                ui.number(
                    "degree", value=state.get("_pg_poly_degree", 2),
                    min=2, max=5, step=1,
                    on_change=lambda e: state.update({"_pg_poly_degree": int(e.value)}),
                ).props("outlined dense").classes("col-4")
                ui.checkbox(
                    "interaction_only",
                    value=state.get("_pg_poly_ia", False),
                    on_change=lambda e: state.update({"_pg_poly_ia": e.value}),
                )


# ═══════════════════════════════════════════════════════════
# Tab 5: 特徴選択
# ═══════════════════════════════════════════════════════════

_SELECTORS = [
    ("none", "なし（全特徴量使用）"),
    ("lasso", "Lasso（線形ペナルティ）"),
    ("rfr", "RF重要度（SelectFromModel）"),
    ("select_kbest", "SelectKBest"),
    ("select_percentile", "SelectPercentile"),
    ("boruta", "Boruta"),
]


def _tab_selector(state: dict) -> None:
    _section("🎯", "Feature Selector",
             "複数選択で全組み合わせを評価。未選択 → none。")
    sel_keys = state.get("_pg_selectors", ["none"])
    sel_options = [k for k, _ in _SELECTORS]

    def _on_sel(e):
        state["_pg_selectors"] = list(e.value)
    ui.select(
        sel_options, multiple=True, value=sel_keys,
        label="特徴選択手法", on_change=_on_sel,
    ).props("use-chips outlined dense").classes("full-width")

    # Lasso パラメータ
    if "lasso" in state.get("_pg_selectors", []):
        with _glass_card():
            ui.label("Lasso 設定").classes("text-caption text-bold")
            with ui.row().classes("q-gutter-sm"):
                ui.number(
                    "alpha", value=state.get("_pg_lasso_alpha", 0.01),
                    min=1e-6, max=10.0, step=0.001, format="%.6f",
                    on_change=lambda e: state.update({"_pg_lasso_alpha": float(e.value)}),
                ).props("outlined dense").classes("col-4")
                ui.number(
                    "max_iter", value=state.get("_pg_lasso_mi", 1000),
                    min=100, max=10000, step=100,
                    on_change=lambda e: state.update({"_pg_lasso_mi": int(e.value)}),
                ).props("outlined dense").classes("col-4")

    # SelectKBest パラメータ
    if "select_kbest" in state.get("_pg_selectors", []):
        with _glass_card():
            ui.label("SelectKBest 設定").classes("text-caption text-bold")
            with ui.row().classes("q-gutter-sm"):
                ui.number(
                    "k", value=state.get("_pg_kbest_k", 10),
                    min=1, max=500, step=1,
                    on_change=lambda e: state.update({"_pg_kbest_k": int(e.value)}),
                ).props("outlined dense").classes("col-4")
                ui.label("score_func").classes("text-caption text-bold")
                ui.radio(
                    {"f_regression": "f_regression", "mutual_info_regression": "mutual_info_regression",
                     "r_regression": "r_regression", "f_classif": "f_classif",
                     "mutual_info_classif": "mutual_info_classif"},
                    value=state.get("_pg_kbest_sf", "f_regression"),
                    on_change=lambda e: state.update({"_pg_kbest_sf": e.value}),
                ).props("dense inline")

    # Boruta パラメータ
    if "boruta" in state.get("_pg_selectors", []):
        with _glass_card():
            ui.label("Boruta 設定").classes("text-caption text-bold")
            with ui.row().classes("q-gutter-sm"):
                ui.number(
                    "n_estimators", value=state.get("_pg_boruta_n", 100),
                    min=10, max=500, step=10,
                    on_change=lambda e: state.update({"_pg_boruta_n": int(e.value)}),
                ).props("outlined dense").classes("col-4")
                ui.number(
                    "max_iter", value=state.get("_pg_boruta_mi", 100),
                    min=10, max=500, step=10,
                    on_change=lambda e: state.update({"_pg_boruta_mi": int(e.value)}),
                ).props("outlined dense").classes("col-4")


# ═══════════════════════════════════════════════════════════
# Tab 6: 変数メタ情報エディタ
# ═══════════════════════════════════════════════════════════

def _tab_column_meta(state: dict) -> None:
    """変数ごとのメタ情報（単調性・線形性・グループ・スケーラーHint）を設定するタブ。"""
    _section("🔖", "変数メタ情報",
             "各説明変数の単調性制約・線形性ヒント・グループ設定などを指定。\n"
             "設定はスケーラー推奨・特徴量選択・推定器の単調性制約に自動反映されます。")

    from frontend_nicegui.components.column_meta_editor import render_column_meta_editor
    df = state.get("df")
    render_column_meta_editor(state, df)

    # 設定サマリー
    meta = state.get("column_meta", {})
    n_mono = sum(1 for m in meta.values() if m.get("monotonic", 0) != 0)
    n_group = len({m.get("group") for m in meta.values() if m.get("group")})
    n_fixed = sum(1 for m in meta.values() if m.get("fixed"))
    n_scale = sum(1 for m in meta.values() if m.get("scale_hint"))
    if n_mono > 0 or n_group > 0 or n_fixed > 0:
        ui.separator().classes("q-my-xs")
        with ui.row().classes("q-gutter-sm text-caption text-grey q-mt-xs"):
            if n_mono > 0:
                ui.badge(f"単調性制約: {n_mono}変数", color="teal").props("dense")
            if n_group > 0:
                ui.badge(f"グループ: {n_group}種", color="blue").props("dense")
            if n_fixed > 0:
                ui.badge(f"常時保持: {n_fixed}変数", color="amber").props("dense")
            if n_scale > 0:
                ui.badge(f"スケーラーHint: {n_scale}変数", color="purple").props("dense")


# ═══════════════════════════════════════════════════════════
# Tab 7: JLランダム射影 (Johnson-Lindenstrauss)
# ═══════════════════════════════════════════════════════════


def _tab_jl_rp(state: dict) -> None:
    """JL補題に基づくランダム射影設定UI。"""
    _section("🎲", "JLランダム射影（Johnson-Lindenstrauss）",
             "n_features > jl_min_dim(n_samples, ε) の場合のみ自動適用。條件不成立時は完全スキップ。")

    enabled = state.get("_pg_rp_enable", False)
    eps = state.get("_pg_rp_eps", 0.1)
    method = state.get("_pg_rp_method", "auto")

    # 有効/無効トグル
    ui.switch(
        "ランダム射影を有効化",
        value=enabled,
        on_change=lambda e: state.update({"_pg_rp_enable": e.value}),
    ).props("color=indigo")

    # JL条件のリアルタイム表示
    df = state.get("df")
    precalc_df = state.get("precalc_df")
    n_samples = len(df) if df is not None else 0
    # 現在の発構済み記述子＋データDFの列数で計算
    if precalc_df is not None and df is not None:
        n_features_est = precalc_df.shape[1] + max(0, df.shape[1] - 2)  # SMILES・目的変数を除いた導入
    elif df is not None:
        n_features_est = max(0, df.shape[1] - 2)
    else:
        n_features_est = 0

    if n_samples > 0 and n_features_est > 0:
        try:
            from sklearn.random_projection import johnson_lindenstrauss_min_dim
            jl_min = int(johnson_lindenstrauss_min_dim(n_samples, eps=eps))
            should_apply = n_features_est > jl_min
            status_color = "green" if should_apply else "grey"
            status_text = (
                f"✅ 適用溈: {n_features_est} → {jl_min} 次元（刂減 {n_features_est-jl_min}次元、{(1-jl_min/n_features_est)*100:.0f}%圧縮）"
                if should_apply else
                f"⏩ 不要（n_features={n_features_est} ≤ jl_min_dim={jl_min}） — 無効で自動スキップ"
            )
        except Exception:
            jl_min = 0
            status_color = "grey"
            status_text = "計算中にエラー"

        with ui.card().classes("full-width q-pa-sm q-my-sm").style(
            f"border:1px solid rgba(0,188,212,0.3); border-radius:8px;"
        ):
            ui.label("📊 JL補題による自動判定（現在のデータで予測）").classes("text-caption text-bold text-cyan q-mb-xs")
            with ui.row().classes("q-gutter-sm items-center"):
                ui.badge(f"n_samples={n_samples}", color="blue-grey").props("outline")
                ui.badge(f"n_features≈{n_features_est}", color="blue-grey").props("outline")
                ui.badge(f"ε={eps}", color="blue-grey").props("outline")
            ui.label(status_text).classes(f"text-body2 text-bold text-{status_color} q-mt-xs")
            ui.label(
                "jl_min_dim = 4 log(n) / (ε²/2 - ε³/3)「これ以上の次元は副作用なしに刢減できることが保証される"
            ).classes("text-caption text-grey q-mt-xs").style("font-size:0.72rem;")
    else:
        ui.label("データ読み込後に自動判定結果を表示します").classes("text-caption text-grey")

    ui.separator().classes("q-my-sm")

    # 詳細設定
    _section("⚙️", "パラメータ設定")
    with ui.row().classes("q-gutter-md items-center flex-wrap"):
        ui.number(
            "ε（歪み許容誤差）",
            value=eps, min=0.01, max=0.5, step=0.01, format="%.2f",
            on_change=lambda e: state.update({"_pg_rp_eps": float(e.value or 0.1)}),
        ).props("outlined dense").style("width:160px;").tooltip(
            "小さいほど距離保全性↑・次元↑、大きいほど圧縮率↑・距離誤差↑。\n"
            "推奨: 0.05～0.2。\n"
            "eps=0.1 → 各点間距離の誤差を最大±10%に抖えることを保証。"
        )
        ui.select(
            {
                "auto": "auto（d>1000→sparse, それ以下→gaussian）",
                "sparse": "Sparse RP（メモリ効率↑, 超高次元向き）",
                "gaussian": "Gaussian RP（理論的保証厳密, 中規模向き）",
            },
            value=method,
            label="射影手法",
            on_change=lambda e: state.update({"_pg_rp_method": e.value}),
        ).props("outlined dense").style("min-width:300px;")

    # 理論的根拠の表示
    with ui.expansion("📚 理論的根拠（JL Lemma）", icon="info").classes("full-width q-mt-sm"):
        ui.html("""
        <div style='font-size:0.82rem; color:#aaa; line-height:1.7;'>
        <b style='color:#00bcd4;'>Johnson-Lindenstrauss Lemma (1984):</b><br>
        n点のデータを．──────────────────────────────<br>
        &emsp;<i>d_jl = O(log(n) / &epsilon;&sup2;)</i> 次元の空間に射影するとき、<br>
        &emsp;任意の2点間距離を (1±&epsilon;) 倍の精度で保全できる。<br><br>
        <b>適用条件:</b> n_features &gt; d_jl のときのみ RP を適用<br>
        <b style='color:#4ade80;'>利点:</b> 計算時間素, メモリ削減, 次元の呪い緩和<br>
        <b style='color:#facc15;'>注意:</b> モデル解釈性が低下する。解釈性高いモデル（線形回帰等）には非推奨。
        </div>
        """)



def _tab_estimator(state: dict) -> None:
    task = state.get("task_type", "regression")
    is_reg = task == "regression"
    _section("🤖", f"Estimator（{'回帰' if is_reg else '分類'}）",
             "使用するモデルを選択。⚙️ で各モデルのパラメータ・Grid/Optuna探索範囲を設定。")

    try:
        from backend.models.factory import list_models, get_default_automl_models
        available = list_models(task=task, available_only=True)
        defaults = get_default_automl_models(task=task)
    except Exception as ex:
        ui.label(f"⚠️ モデル一覧取得エラー: {ex}").classes("text-caption text-red")
        return

    # model_configs 初期化
    if "model_configs" not in state:
        state["model_configs"] = {}

    # カテゴリ分類（共通関数使用）
    categories = _categorize_models(available)

    selected_models = state.get("selected_models", [])
    if not selected_models:
        selected_models = list(defaults)

    # 一括操作
    with ui.row().classes("q-gutter-xs q-mb-sm"):
        def _select_all():
            all_keys = [m["key"] for m in available]
            state["selected_models"] = all_keys
        def _select_defaults():
            state["selected_models"] = list(defaults)
        def _select_none():
            state["selected_models"] = []
        ui.button("全選択", on_click=_select_all).props("flat dense size=xs color=cyan no-caps")
        ui.button("推奨のみ", on_click=_select_defaults).props("flat dense size=xs color=teal no-caps")
        ui.button("全解除", on_click=_select_none).props("flat dense size=xs color=grey no-caps")

    # カテゴリごとに展開パネル
    for cat_name, models in categories.items():
        if not models:
            continue
        n_selected = sum(1 for m in models if m["key"] in selected_models)
        with ui.expansion(
            f"{cat_name}  【選択: {n_selected} / 全{len(models)}個】",
        ).classes("full-width q-mb-xs").props("dense"):
            for m in models:
                mkey = m["key"]
                mname = m["name"]
                mcls = m.get("class")
                is_checked = mkey in selected_models

                def _toggle(e, key=mkey):
                    sm = state.get("selected_models", list(defaults))
                    if e.value:
                        if key not in sm:
                            sm.append(key)
                    else:
                        sm = [k for k in sm if k != key]
                    state["selected_models"] = sm

                with ui.row().classes("items-center q-gutter-xs"):
                    ui.checkbox(
                        mname, value=is_checked,
                        on_change=_toggle,
                    )
                    if mkey in defaults:
                        ui.badge("推奨", color="teal").props("dense").style("font-size: 0.78rem;")

                    # ⚙️ パラメータ設定ボタン（estimatorクラスがある場合のみ）
                    if mcls is not None:
                        def _open_config(key=mkey, name=mname, cls=mcls):
                            from frontend_nicegui.components.estimator_config_dialog import (
                                EstimatorConfigDialog,
                            )
                            existing = state["model_configs"].get(key)
                            dialog = EstimatorConfigDialog(
                                model_key=key,
                                model_cls=cls,
                                model_name=name,
                                initial_config=existing,
                                on_save=lambda cfg, k=key: state["model_configs"].update({k: cfg}),
                            )
                            dialog.open()

                        btn = ui.button(
                            icon="tune", on_click=_open_config,
                        ).props("flat dense round size=xs color=cyan")
                        btn.tooltip(f"{mname}: デフォルト値 / GridSearch / Optuna 探索範囲を設定")

                    # 設定済みバッジ
                    if mkey in state.get("model_configs", {}):
                        ui.badge("⚙設定済", color="amber").props("outline dense").style("font-size:0.68rem;")


# ═══════════════════════════════════════════════════════════
# 組み合わせ数サマリー
# ═══════════════════════════════════════════════════════════

def _render_combo_summary(state: dict) -> None:
    """リアルタイムで「何通りのパイプラインか」を表示。"""
    n_imp = max(1, len(state.get("_pg_num_imputers", ["mean"])))
    n_scl = max(1, len(state.get("_pg_num_scalers", ["standard"])))
    n_ci = max(1, len(state.get("_pg_cat_imputers", ["most_frequent"])))
    n_le = max(1, len(state.get("_pg_low_encoders", ["onehot"])))
    n_bi = max(1, len(state.get("_pg_bin_imputers", ["most_frequent"])))
    n_eng = max(1, len(state.get("_pg_engineer", ["none"])))
    n_sel = max(1, len(state.get("_pg_selectors", ["none"])))
    n_est = max(1, len(state.get("selected_models", [])))
    n_total = n_imp * n_scl * n_ci * n_le * n_bi * n_eng * n_sel * n_est

    status_color = "teal" if n_total <= 50 else ("amber" if n_total <= 200 else "red")
    status_text = "✅ 適切" if n_total <= 50 else ("⚠️ やや多い" if n_total <= 200 else "🔴 多すぎ")

    with ui.card().classes("full-width q-pa-sm").style(
        f"border:2px solid var(--q-{status_color});border-radius:8px;"
        "background:rgba(0,20,40,0.3);"
    ):
        with ui.row().classes("items-center justify-between full-width"):
            ui.label(f"🔢 評価パイプライン数: {n_total:,} 通り").classes("text-body1 text-bold")
            ui.badge(status_text, color=status_color)
        ui.label(
            f"imp×{n_imp} · scl×{n_scl} · cat×{n_ci} · enc×{n_le} "
            f"· bin×{n_bi} · eng×{n_eng} · sel×{n_sel} · est×{n_est}"
        ).classes("text-caption text-grey").style("font-size:0.7rem;")


# ═══════════════════════════════════════════════════════════
# メインエントリーポイント
# ═══════════════════════════════════════════════════════════

def render_pipeline_config(state: dict) -> None:
    """パイプライン全設定UIをインラインでレンダリングする。

    【UI設計方針】
    - 推定器選択 = 最上段・常時表示・大面積・1クリック選択
    - 前処理設定 = 折りたたみ内・コンパクト・目立たない
    """
    task = state.get("task_type", "regression")
    is_reg = task == "regression"

    try:
        from backend.models.factory import list_models, get_default_automl_models
        available = list_models(task=task, available_only=True)
        defaults = get_default_automl_models(task=task)
    except Exception as ex:
        ui.label(f"⚠️ モデル一覧取得エラー: {ex}").classes("text-caption text-red")
        return

    if "model_configs" not in state:
        state["model_configs"] = {}

    # カテゴリ分類（共通関数使用）
    categories = _categorize_models(available)
    cat_icons = {name: icon for name, icon in _MODEL_CATEGORIES}

    selected_models = state.get("selected_models", [])
    if not selected_models:
        selected_models = list(defaults)
        state["selected_models"] = selected_models

    # ═══════════════════════════════════════════════════════
    # 🤖 推定器セレクター（トップレベル・最大面積・常時表示）
    # ═══════════════════════════════════════════════════════
    @ui.refreshable
    def _render_estimator_selector():
        selected_models = state.get("selected_models", [])
        
        with ui.card().classes("full-width q-pa-md q-mb-sm").style(
            "border: 2px solid rgba(0,188,212,0.4); border-radius: 12px;"
            "background: linear-gradient(135deg, rgba(0,30,60,0.4), rgba(0,20,50,0.3));"
        ):
            # ヘッダー行
            with ui.row().classes("items-center justify-between full-width q-mb-sm"):
                with ui.row().classes("items-center q-gutter-sm"):
                    ui.icon("smart_toy", color="cyan", size="md")
                    ui.label(f"🤖 推定器セレクター（{'回帰' if is_reg else '分類'}）").classes(
                        "text-h6 text-bold"
                    )
                with ui.row().classes("q-gutter-xs"):
                    n_sel = len(selected_models)
                    n_all = len(available)
                    ui.badge(f"{n_sel}/{n_all} 選択中", color="cyan").props("dense")

            # 一括操作ボタン
            with ui.row().classes("q-gutter-xs q-mb-md"):
                def _select_all():
                    all_keys = [m["key"] for m in available]
                    state["selected_models"] = all_keys
                    ui.notify(f"✅ 全{len(all_keys)}モデルを選択しました", type="positive", timeout=2000)
                    _render_estimator_selector.refresh()
                
                def _select_defaults():
                    state["selected_models"] = list(defaults)
                    ui.notify(f"⭐ 推奨{len(defaults)}モデルを選択しました", type="info", timeout=2000)
                    _render_estimator_selector.refresh()
                
                def _select_none():
                    state["selected_models"] = []
                    ui.notify("🚫 全モデルを解除しました", type="warning", timeout=2000)
                    _render_estimator_selector.refresh()

                ui.button("✅ 全選択", on_click=_select_all).props(
                    "outline color=cyan size=sm no-caps"
                ).style("border-radius: 20px;")
                ui.button("⭐ 推奨のみ", on_click=_select_defaults).props(
                    "outline color=teal size=sm no-caps"
                ).style("border-radius: 20px;")
                ui.button("🚫 全解除", on_click=_select_none).props(
                    "outline color=grey size=sm no-caps"
                ).style("border-radius: 20px;")

            # カテゴリごとのチップグリッド（展開不要・即クリック）
            for cat_name, models in categories.items():
                if not models:
                    continue
                n_cat_sel = sum(1 for m in models if m["key"] in selected_models)
                icon = cat_icons.get(cat_name, "")

                with ui.row().classes("items-center q-gutter-xs q-mb-xs"):
                    ui.icon(icon, color="cyan-4", size="xs")
                    ui.label(cat_name).classes("text-body2 text-bold text-cyan-3")
                    ui.badge(f"{n_cat_sel}/{len(models)}").props("dense").style(
                        "font-size:0.7rem;"
                    )

                with ui.row().classes("q-gutter-xs q-mb-sm flex-wrap"):
                    for m in models:
                        mkey = m["key"]
                        mname = m["name"]  # フルネーム（tooltip用）
                        mcls = m.get("class")
                        is_on = mkey in selected_models
                        is_default = mkey in defaults
                        has_config = mkey in state.get("model_configs", {})
                        short, complexity, lib = _get_model_meta(mkey)

                        # チップスタイル決定
                        if is_on:
                            chip_style = (
                                "background: rgba(0,188,212,0.25); border: 1.5px solid rgba(0,188,212,0.6);"
                                "color: #e0f7fa; border-radius: 20px; cursor: pointer;"
                            )
                        else:
                            chip_style = (
                                "background: rgba(60,60,80,0.3); border: 1px solid rgba(100,100,120,0.3);"
                                "color: #888; border-radius: 20px; cursor: pointer;"
                            )

                        def _toggle_model(key=mkey):
                            sm = list(state.get("selected_models", []))
                            if key in sm:
                                sm.remove(key)
                            else:
                                sm.append(key)
                            state["selected_models"] = sm
                            _render_estimator_selector.refresh()

                        # 短縮ラベル（スペース節約）
                        chip_label = f"{'✓' if is_on else ''}{short}"
                        if is_default:
                            chip_label += "⭐"
                        if has_config:
                            chip_label += "⚙"

                        # リッチツールチップ: 正式名 / 計算量 / ライブラリ
                        tip = f"{mname}\n計算量: {complexity}\nlib: {lib}"
                        if is_on:
                            tip += "\n（クリックで除外）"
                        else:
                            tip += "\n（クリックで追加）"

                        btn = ui.button(
                            chip_label,
                            on_click=_toggle_model,
                        ).props("flat dense no-caps size=sm").style(
                            chip_style + "padding: 1px 8px; font-size: 0.78rem; min-height: 26px;"
                        )
                        btn.tooltip(tip)

            # 選択済みモデルの詳細パネル（⚙️設定ボタン付き）
            sel_models = [m for m in available if m["key"] in selected_models]
            if sel_models:
                ui.separator().classes("q-my-sm")
                ui.label("🔧 選択済みモデルの設定").classes("text-body2 text-bold text-grey-4 q-mb-xs")
                with ui.element("div").classes("full-width").style(
                    "display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 6px;"
                ):
                    for m in sel_models:
                        mkey = m["key"]
                        mname = m["name"]
                        mcls = m.get("class")
                        has_config = mkey in state.get("model_configs", {})

                        with ui.card().classes("q-pa-xs").style(
                            "border: 1px solid rgba(0,188,212,0.15); border-radius: 8px;"
                            "background: rgba(0,20,40,0.2); min-height: 40px;"
                        ):
                            with ui.row().classes("items-center justify-between no-wrap"):
                                ui.label(mname).classes("text-caption text-grey-3 ellipsis").style(
                                    "max-width: 140px;"
                                )
                                with ui.row().classes("items-center q-gutter-xs no-wrap"):
                                    if has_config:
                                        ui.badge("⚙設定済", color="amber").props(
                                            "outline dense"
                                        ).style("font-size:0.6rem;")
                                    if mcls is not None:
                                        def _open_config(key=mkey, name=mname, cls=mcls):
                                            from frontend_nicegui.components.estimator_config_dialog import (
                                                EstimatorConfigDialog,
                                            )
                                            existing = state["model_configs"].get(key)
                                            dialog = EstimatorConfigDialog(
                                                model_key=key,
                                                model_cls=cls,
                                                model_name=name,
                                                initial_config=existing,
                                                on_save=lambda cfg, k=key: state["model_configs"].update({k: cfg}),
                                            )
                                            dialog.open()

                                        ui.button(
                                            icon="tune", on_click=_open_config,
                                        ).props("flat dense round size=xs color=cyan").tooltip(
                                            f"{mname}: デフォルト値 / Grid / Optuna"
                                        )

    _render_estimator_selector()

    # ═══════════════════════════════════════════════════════
    # 🔧 前処理設定（折りたたみ・コンパクト・目立たない）
    # ═══════════════════════════════════════════════════════
    _render_preprocess_section(state, defaults)

    # ═══════════════════════════════════════════════════════
    # 逆解析連携（設定は post_analysis_config に一元化済み）
    # ═══════════════════════════════════════════════════════
    ui.separator().classes("q-my-sm")
    with ui.card().classes("full-width q-pa-sm").style(
        "border: 1px solid rgba(156,39,176,0.2); border-radius: 8px;"
        "background: rgba(30,0,40,0.15);"
    ):
        ui.html(
            '<div style="font-size:0.82rem; color:#aaa;">'
            '💡 <b>自動実行設定について</b><br>'
            '「順解析完了後に逆解析を自動実行」の設定は、'
            'このタブの下部「🔮 解析後の自動処理」セクションに集約されています。'
            '</div>'
        )


def _render_preprocess_section(state: dict, defaults: list) -> None:
    """前処理設定（折りたたみ内にコンパクト配置）。"""
    # 組み合わせ数計算
    n_imp = max(1, len(state.get("_pg_num_imputers", ["mean"])))
    n_scl = max(1, len(state.get("_pg_num_scalers", ["standard"])))
    n_ci = max(1, len(state.get("_pg_cat_imputers", ["most_frequent"])))
    n_le = max(1, len(state.get("_pg_low_encoders", ["onehot"])))
    n_bi = max(1, len(state.get("_pg_bin_imputers", ["most_frequent"])))
    n_eng = max(1, len(state.get("_pg_engineer", ["none"])))
    n_sel = max(1, len(state.get("_pg_selectors", ["none"])))
    n_est = max(1, len(state.get("selected_models", [])))
    n_total = n_imp * n_scl * n_ci * n_le * n_bi * n_eng * n_sel * n_est

    status_color = "teal" if n_total <= 50 else ("amber" if n_total <= 200 else "red")

    with ui.card().classes("full-width q-pa-sm q-mb-sm").style(
        "border: 1px solid rgba(100,120,140,0.2); border-radius: 8px;"
        "background: rgba(0,15,30,0.2);"
    ):
        with ui.expansion(
            f"🔧 前処理設定（{n_total:,}通り）", icon="build",
        ).classes("full-width").props("dense header-class=text-grey-5"):
            ui.label(
                "通常はデフォルトのままで十分です。"
                "複数選択すると全組み合わせを評価します。"
            ).classes("text-caption text-grey-6 q-mb-sm").style("font-size:0.72rem;")

            with ui.tabs().classes("full-width").props(
                "dense no-caps active-color=blue-grey indicator-color=blue-grey"
            ) as pg_tabs:
                ui.tab("pg_excl", label="🚫 除外")
                ui.tab("pg_num", label="🔢 数値")
                ui.tab("pg_cat", label="🏷️ カテゴリ")
                ui.tab("pg_bin", label="⚡ バイナリ")
                ui.tab("pg_eng", label="🔧 特徴生成")
                ui.tab("pg_sel", label="🎯 特徴選択")
                ui.tab("pg_meta", label="🔖 変数メタ")

            with ui.tab_panels(pg_tabs, value="pg_excl").classes("full-width"):
                with ui.tab_panel("pg_excl"):
                    _tab_excluder(state)
                with ui.tab_panel("pg_num"):
                    _tab_numeric(state)
                with ui.tab_panel("pg_cat"):
                    _tab_categorical(state)
                with ui.tab_panel("pg_bin"):
                    _tab_binary(state)
                with ui.tab_panel("pg_eng"):
                    _tab_engineer(state)
                with ui.tab_panel("pg_sel"):
                    _tab_selector(state)
                with ui.tab_panel("pg_meta"):
                    _tab_column_meta(state)


            # JL-RP（次元削減）
            _tab_jl_rp(state)

            ui.separator().classes("q-my-xs")
            _render_combo_summary(state)


# _render_inverse_link は削除済み — 設定は post_analysis_config.py に一元化

