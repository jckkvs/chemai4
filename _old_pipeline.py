# -*- coding: utf-8 -*-
"""
frontend_nicegui/components/pipeline_config_ui.py

Pipeline 蜈ｨ險ｭ螳啅I 窶・NiceGUI迚・7繧ｹ繝・ャ繝励・繧ｿ繝門ｽ｢蠑上〒蜈ｨ繝代う繝励Λ繧､繝ｳ繧定ｨｭ螳壹・隍・焚驕ｸ謚・= 蜈ｨ邨・∩蜷医ｏ縺帙ｒ隧穂ｾ｡ / 譛ｪ驕ｸ謚・= 驕ｩ蛻・↑繝・ヵ繧ｩ繝ｫ繝医ｒ閾ｪ蜍暮←逕ｨ縲・
Streamlit迚医・pipeline_config_ui.py縺ｨ讖溯・遲我ｾ｡縲・"""
from __future__ import annotations

from typing import Any
from nicegui import ui


# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・# 繝倥Ν繝代・
# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・
def _section(icon: str, title: str, desc: str = "") -> None:
    """繧ｻ繧ｯ繧ｷ繝ｧ繝ｳ繝倥ャ繝繝ｼ縲・""
    ui.label(f"{icon} {title}").classes("text-subtitle1 text-bold q-mt-sm")
    if desc:
        ui.label(desc).classes("text-caption text-grey q-mb-xs").style("font-size:0.75rem;")


def _glass_card():
    """繧ｬ繝ｩ繧ｹ繧ｫ繝ｼ繝峨・繧ｳ繝ｳ繝・く繧ｹ繝医・繝阪・繧ｸ繝｣繝ｼ縲・""
    return ui.card().classes("full-width q-pa-sm q-mb-xs").style(
        "border:1px solid rgba(0,188,212,0.2); border-radius:8px;"
        "background:rgba(0,20,40,0.25);"
    )


# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・# Tab 0: Excluder
# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・
def _tab_excluder(state: dict) -> None:
    _section("圻", "Excluder・郁ｧ｣譫宣勁螟門・・・,
             "隗｣譫舌↓菴ｿ繧上↑縺・・繧帝∈謚槭ら岼逧・､画焚繝ｻSMILES蛻励・閾ｪ蜍暮勁螟匁ｸ医∩縲・)
    df = state.get("df")
    if df is None:
        ui.label("繝・・繧ｿ譛ｪ隱ｭ縺ｿ霎ｼ縺ｿ").classes("text-caption text-grey")
        return
    target_col = state.get("target_col", "")
    smiles_col = state.get("smiles_col", "")
    skip = {c for c in (target_col, smiles_col) if c}
    opts = [c for c in df.columns if c not in skip]
    if not opts:
        ui.label("髯､螟悶〒縺阪ｋ蛻励′縺ゅｊ縺ｾ縺帙ｓ").classes("text-caption text-grey")
        return
    prev = state.get("exclude_cols", [])

    def _on_change(e):
        state["exclude_cols"] = list(e.value)
    ui.select(
        opts, multiple=True, value=[c for c in prev if c in opts],
        label="髯､螟門・繧帝∈謚・,
        on_change=_on_change,
    ).props("use-chips outlined dense").classes("full-width")


# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・# Tab 1: 謨ｰ蛟､蜑榊・逅・# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・
_NUM_IMPUTERS = [
    ("mean", "Mean・亥ｹｳ蝮・ｼ・),
    ("median", "Median・井ｸｭ螟ｮ蛟､・・),
    ("knn", "KNN Imputer"),
    ("iterative", "Iterative Imputer"),
    ("constant", "Constant・亥崋螳壼､・・),
]

_NUM_SCALERS = [
    ("standard", "StandardScaler"),
    ("minmax", "MinMaxScaler"),
    ("robust", "RobustScaler"),
    ("maxabs", "MaxAbsScaler"),
    ("power_yj", "PowerTransformer [YJ]"),
    ("power_bc", "PowerTransformer [BC]"),
    ("quantile_normal", "QuantileTransformer竊呈ｭ｣隕・),
    ("quantile_uniform", "QuantileTransformer竊剃ｸ讒・),
    ("none", "繧ｹ繧ｱ繝ｼ繝ｪ繝ｳ繧ｰ縺ｪ縺・),
]


def _tab_numeric(state: dict) -> None:
    _section("箸", "謨ｰ蛟､蛻怜燕蜃ｦ逅・ｼ・mputer ﾃ・Scaler・・,
             "驕ｸ謚槭＠縺・Imputer 縺ｨ Scaler 縺ｮ蜈ｨ邨・∩蜷医ｏ縺帙ｒ隧穂ｾ｡縺励∪縺吶・)

    # 笏笏 Imputer 笏笏
    ui.label("投 Imputer・域ｬ謳崎｣憺俣・・).classes("text-body2 text-bold q-mt-sm")
    imp_keys = state.get("_pg_num_imputers", ["mean"])
    imp_options = [k for k, _ in _NUM_IMPUTERS]
    imp_labels = {k: v for k, v in _NUM_IMPUTERS}

    def _on_imp(e):
        state["_pg_num_imputers"] = list(e.value)
    ui.select(
        imp_options, multiple=True, value=imp_keys,
        label="Imputer・郁､・焚驕ｸ謚槫庄・・,
        on_change=_on_imp,
    ).props("use-chips outlined dense").classes("full-width")

    ui.separator().classes("q-my-xs")

    # 笏笏 Scaler 笏笏
    ui.label("棟 Scaler").classes("text-body2 text-bold")
    scl_keys = state.get("_pg_num_scalers", ["standard"])
    scl_options = [k for k, _ in _NUM_SCALERS]

    def _on_scl(e):
        state["_pg_num_scalers"] = list(e.value)
    ui.select(
        scl_options, multiple=True, value=scl_keys,
        label="Scaler・郁､・焚驕ｸ謚槫庄・・,
        on_change=_on_scl,
    ).props("use-chips outlined dense").classes("full-width")


# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・# Tab 2: 繧ｫ繝・ざ繝ｪ蜑榊・逅・# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・
_CAT_IMPUTERS = [
    ("most_frequent", "Most Frequent・域怙鬆ｻ蛟､・・),
    ("constant", "Constant・域欠螳壽枚蟄怜・・・),
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
    _section("捷・・, "繧ｫ繝・ざ繝ｪ蛻怜燕蜃ｦ逅・ｼ・mputer ﾃ・Encoder・・,
             "菴弱き繝ｼ繝・ぅ繝翫Μ繝・ぅ縺ｨ鬮倥き繝ｼ繝・ぅ繝翫Μ繝・ぅ縺ｧ蛻･險ｭ螳壼庄閭ｽ縲・)

    # Imputer
    ui.label("筈 Categorical Imputer").classes("text-body2 text-bold q-mt-sm")
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
            ui.label("判 菴弱き繝ｼ繝・ぅ繝翫Μ繝・ぅ").classes("text-body2 text-bold")
            le_keys = state.get("_pg_low_encoders", ["onehot"])
            le_options = [k for k, _ in _LOW_ENCODERS]

            def _on_le(e):
                state["_pg_low_encoders"] = list(e.value)
            ui.select(le_options, multiple=True, value=le_keys,
                      label="Encoder", on_change=_on_le,
                      ).props("use-chips outlined dense").classes("full-width")

        with ui.column().classes("col"):
            ui.label("伴 鬮倥き繝ｼ繝・ぅ繝翫Μ繝・ぅ").classes("text-body2 text-bold")
            he_keys = state.get("_pg_high_encoders", ["ordinal"])
            he_options = [k for k, _ in _HIGH_ENCODERS]

            def _on_he(e):
                state["_pg_high_encoders"] = list(e.value)
            ui.select(he_options, multiple=True, value=he_keys,
                      label="Encoder", on_change=_on_he,
                      ).props("use-chips outlined dense").classes("full-width")


# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・# Tab 3: 繝舌う繝翫Μ蜑榊・逅・# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・
def _tab_binary(state: dict) -> None:
    _section("笞｡", "繝舌う繝翫Μ蛻怜燕蜃ｦ逅・,
             "0/1, True/False 縺ｪ縺ｩ縺ｮ2蛟､蛻励・蜃ｦ逅・ｨｭ螳壹・)
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


# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・# Tab 4: 迚ｹ蠕ｴ逕滓・
# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・
def _tab_engineer(state: dict) -> None:
    _section("肌", "Feature Engineering",
             "隍・焚驕ｸ謚槭〒蜈ｨ繝代ち繝ｼ繝ｳ繧定ｩ穂ｾ｡縲よ悴驕ｸ謚・竊・none縲・)
    eng_keys = state.get("_pg_engineer", ["none"])

    def _on_eng(e):
        state["_pg_engineer"] = list(e.value)
    ui.select(
        ["none", "polynomial", "interaction_only"],
        multiple=True, value=eng_keys,
        label="逕滓・謇区ｳ・, on_change=_on_eng,
    ).props("use-chips outlined dense").classes("full-width")

    # Polynomial 繝代Λ繝｡繝ｼ繧ｿ
    if "polynomial" in state.get("_pg_engineer", []):
        with _glass_card():
            ui.label("PolynomialFeatures 險ｭ螳・).classes("text-caption text-bold")
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


# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・# Tab 5: 迚ｹ蠕ｴ驕ｸ謚・# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・
_SELECTORS = [
    ("none", "縺ｪ縺暦ｼ亥・迚ｹ蠕ｴ驥丈ｽｿ逕ｨ・・),
    ("lasso", "Lasso・育ｷ壼ｽ｢繝壹リ繝ｫ繝・ぅ・・),
    ("rfr", "RF驥崎ｦ∝ｺｦ・・electFromModel・・),
    ("select_kbest", "SelectKBest"),
    ("select_percentile", "SelectPercentile"),
    ("boruta", "Boruta"),
]


def _tab_selector(state: dict) -> None:
    _section("識", "Feature Selector",
             "隍・焚驕ｸ謚槭〒蜈ｨ邨・∩蜷医ｏ縺帙ｒ隧穂ｾ｡縲よ悴驕ｸ謚・竊・none縲・)
    sel_keys = state.get("_pg_selectors", ["none"])
    sel_options = [k for k, _ in _SELECTORS]

    def _on_sel(e):
        state["_pg_selectors"] = list(e.value)
    ui.select(
        sel_options, multiple=True, value=sel_keys,
        label="迚ｹ蠕ｴ驕ｸ謚樊焔豕・, on_change=_on_sel,
    ).props("use-chips outlined dense").classes("full-width")

    # Lasso 繝代Λ繝｡繝ｼ繧ｿ
    if "lasso" in state.get("_pg_selectors", []):
        with _glass_card():
            ui.label("Lasso 險ｭ螳・).classes("text-caption text-bold")
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

    # SelectKBest 繝代Λ繝｡繝ｼ繧ｿ
    if "select_kbest" in state.get("_pg_selectors", []):
        with _glass_card():
            ui.label("SelectKBest 險ｭ螳・).classes("text-caption text-bold")
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

    # Boruta 繝代Λ繝｡繝ｼ繧ｿ
    if "boruta" in state.get("_pg_selectors", []):
        with _glass_card():
            ui.label("Boruta 險ｭ螳・).classes("text-caption text-bold")
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


# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・# Tab 7: JL繝ｩ繝ｳ繝繝蟆・ｽｱ (Johnson-Lindenstrauss)
# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・
def _tab_jl_rp(state: dict) -> None:
    """JL陬憺｡後↓蝓ｺ縺･縺上Λ繝ｳ繝繝蟆・ｽｱ險ｭ螳啅I縲・""
    _section("軸", "JL繝ｩ繝ｳ繝繝蟆・ｽｱ・・ohnson-Lindenstrauss・・,
             "n_features > jl_min_dim(n_samples, ﾎｵ) 縺ｮ蝣ｴ蜷医・縺ｿ閾ｪ蜍暮←逕ｨ縲よ｢昜ｻｶ荳肴・遶区凾縺ｯ螳悟・繧ｹ繧ｭ繝・・縲・)

    enabled = state.get("_pg_rp_enable", False)
    eps = state.get("_pg_rp_eps", 0.1)
    method = state.get("_pg_rp_method", "auto")

    # 譛牙柑/辟｡蜉ｹ繝医げ繝ｫ
    ui.switch(
        "繝ｩ繝ｳ繝繝蟆・ｽｱ繧呈怏蜉ｹ蛹・,
        value=enabled,
        on_change=lambda e: state.update({"_pg_rp_enable": e.value}),
    ).props("color=indigo")

    # JL譚｡莉ｶ縺ｮ繝ｪ繧｢繝ｫ繧ｿ繧､繝陦ｨ遉ｺ
    df = state.get("df")
    precalc_df = state.get("precalc_df")
    n_samples = len(df) if df is not None else 0
    # 迴ｾ蝨ｨ縺ｮ逋ｺ讒区ｸ医∩險倩ｿｰ蟄撰ｼ九ョ繝ｼ繧ｿDF縺ｮ蛻玲焚縺ｧ險育ｮ・    if precalc_df is not None and df is not None:
        n_features_est = precalc_df.shape[1] + max(0, df.shape[1] - 2)  # SMILES繝ｻ逶ｮ逧・､画焚繧帝勁縺・◆蟆主・
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
                f"笨・驕ｩ逕ｨ貅・ {n_features_est} 竊・{jl_min} 谺｡蜈・ｼ亥・貂・{n_features_est-jl_min}谺｡蜈・＋(1-jl_min/n_features_est)*100:.0f}%蝨ｧ邵ｮ・・
                if should_apply else
                f"竢ｩ 荳崎ｦ・ｼ・_features={n_features_est} 竕､ jl_min_dim={jl_min}・・窶・辟｡蜉ｹ縺ｧ閾ｪ蜍輔せ繧ｭ繝・・"
            )
        except Exception:
            jl_min = 0
            status_color = "grey"
            status_text = "險育ｮ嶺ｸｭ縺ｫ繧ｨ繝ｩ繝ｼ"

        with ui.card().classes("full-width q-pa-sm q-my-sm").style(
            f"border:1px solid rgba(0,188,212,0.3); border-radius:8px;"
        ):
            ui.label("投 JL陬憺｡後↓繧医ｋ閾ｪ蜍募愛螳夲ｼ育樟蝨ｨ縺ｮ繝・・繧ｿ縺ｧ莠域ｸｬ・・).classes("text-caption text-bold text-cyan q-mb-xs")
            with ui.row().classes("q-gutter-sm items-center"):
                ui.badge(f"n_samples={n_samples}", color="blue-grey").props("outline")
                ui.badge(f"n_features竕・n_features_est}", color="blue-grey").props("outline")
                ui.badge(f"ﾎｵ={eps}", color="blue-grey").props("outline")
            ui.label(status_text).classes(f"text-body2 text-bold text-{status_color} q-mt-xs")
            ui.label(
                "jl_min_dim = 4 log(n) / (ﾎｵﾂｲ/2 - ﾎｵﾂｳ/3)縲後％繧御ｻ･荳翫・谺｡蜈・・蜑ｯ菴懃畑縺ｪ縺励↓蛻｢貂帙〒縺阪ｋ縺薙→縺御ｿ晁ｨｼ縺輔ｌ繧・
            ).classes("text-caption text-grey q-mt-xs").style("font-size:0.72rem;")
    else:
        ui.label("繝・・繧ｿ隱ｭ縺ｿ霎ｼ蠕後↓閾ｪ蜍募愛螳夂ｵ先棡繧定｡ｨ遉ｺ縺励∪縺・).classes("text-caption text-grey")

    ui.separator().classes("q-my-sm")

    # 隧ｳ邏ｰ險ｭ螳・    _section("笞呻ｸ・, "繝代Λ繝｡繝ｼ繧ｿ險ｭ螳・)
    with ui.row().classes("q-gutter-md items-center flex-wrap"):
        ui.number(
            "ﾎｵ・域ｭｪ縺ｿ險ｱ螳ｹ隱､蟾ｮ・・,
            value=eps, min=0.01, max=0.5, step=0.01, format="%.2f",
            on_change=lambda e: state.update({"_pg_rp_eps": float(e.value or 0.1)}),
        ).props("outlined dense").style("width:160px;").tooltip(
            "蟆上＆縺・⊇縺ｩ霍晞屬菫晏・諤ｧ竊代・谺｡蜈・・縲∝､ｧ縺阪＞縺ｻ縺ｩ蝨ｧ邵ｮ邇・・繝ｻ霍晞屬隱､蟾ｮ竊代・n"
            "謗ｨ螂ｨ: 0.05・・.2縲・n"
            "eps=0.1 竊・蜷・せ髢楢ｷ晞屬縺ｮ隱､蟾ｮ繧呈怙螟ｧﾂｱ10%縺ｫ謚悶∴繧九％縺ｨ繧剃ｿ晁ｨｼ縲・
        )
        ui.select(
            {
                "auto": "auto・・>1000竊痴parse, 縺昴ｌ莉･荳銀・gaussian・・,
                "sparse": "Sparse RP・医Γ繝｢繝ｪ蜉ｹ邇・・, 雜・ｫ俶ｬ｡蜈・髄縺搾ｼ・,
                "gaussian": "Gaussian RP・育炊隲也噪菫晁ｨｼ蜴ｳ蟇・ 荳ｭ隕乗ｨ｡蜷代″・・,
            },
            value=method,
            label="蟆・ｽｱ謇区ｳ・,
            on_change=lambda e: state.update({"_pg_rp_method": e.value}),
        ).props("outlined dense").style("min-width:300px;")

    # 逅・ｫ也噪譬ｹ諡縺ｮ陦ｨ遉ｺ
    with ui.expansion("答 逅・ｫ也噪譬ｹ諡・・L Lemma・・, icon="info").classes("full-width q-mt-sm"):
        ui.html("""
        <div style='font-size:0.82rem; color:#aaa; line-height:1.7;'>
        <b style='color:#00bcd4;'>Johnson-Lindenstrauss Lemma (1984):</b><br>
        n轤ｹ縺ｮ繝・・繧ｿ繧抵ｼ寂楳笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏<br>
        &emsp;<i>d_jl = O(log(n) / &epsilon;&sup2;)</i> 谺｡蜈・・遨ｺ髢薙↓蟆・ｽｱ縺吶ｋ縺ｨ縺阪・br>
        &emsp;莉ｻ諢上・2轤ｹ髢楢ｷ晞屬繧・(1ﾂｱ&epsilon;) 蛟阪・邊ｾ蠎ｦ縺ｧ菫晏・縺ｧ縺阪ｋ縲・br><br>
        <b>驕ｩ逕ｨ譚｡莉ｶ:</b> n_features &gt; d_jl 縺ｮ縺ｨ縺阪・縺ｿ RP 繧帝←逕ｨ<br>
        <b style='color:#4ade80;'>蛻ｩ轤ｹ:</b> 險育ｮ玲凾髢鍋ｴ, 繝｡繝｢繝ｪ蜑頑ｸ・ 谺｡蜈・・蜻ｪ縺・ｷｩ蜥・br>
        <b style='color:#facc15;'>豕ｨ諢・</b> 繝｢繝・Ν隗｣驥域ｧ縺御ｽ惹ｸ九☆繧九りｧ｣驥域ｧ鬮倥＞繝｢繝・Ν・育ｷ壼ｽ｢蝗槫ｸｰ遲会ｼ峨↓縺ｯ髱樊耳螂ｨ縲・        </div>
        """)



def _tab_estimator(state: dict) -> None:
    task = state.get("task_type", "regression")
    is_reg = task == "regression"
    _section("､・, f"Estimator・・'蝗槫ｸｰ' if is_reg else '蛻・｡・}・・,
             "菴ｿ逕ｨ縺吶ｋ繝｢繝・Ν繧帝∈謚槭Ｇactory.py 縺九ｉ閾ｪ蜍墓､懷・縲・)

    try:
        from backend.models.factory import list_models, get_default_automl_models
        available = list_models(task=task, available_only=True)
        defaults = get_default_automl_models(task=task)
    except Exception as ex:
        ui.label(f"笞・・繝｢繝・Ν荳隕ｧ蜿門ｾ励お繝ｩ繝ｼ: {ex}").classes("text-caption text-red")
        return

    # 繧ｫ繝・ざ繝ｪ蛻・｡・    categories: dict[str, list] = {
        "盗 邱壼ｽ｢邉ｻ": [],
        "鹸 豎ｺ螳壽惠/繧｢繝ｳ繧ｵ繝ｳ繝悶Ν": [],
        "笞呻ｸ・繧ｫ繝ｼ繝阪Ν/縺昴・莉・: [],
    }
    for m in available:
        k = (m["key"] + m["name"]).lower()
        if any(x in k for x in ["linear", "ridge", "lasso", "elastic", "logistic", "ard", "huber", "pls", "bayesian"]):
            categories["盗 邱壼ｽ｢邉ｻ"].append(m)
        elif any(x in k for x in ["tree", "forest", "boost", "gbm", "gradient", "rgf", "figs", "rule", "hist", "catboost"]):
            categories["鹸 豎ｺ螳壽惠/繧｢繝ｳ繧ｵ繝ｳ繝悶Ν"].append(m)
        else:
            categories["笞呻ｸ・繧ｫ繝ｼ繝阪Ν/縺昴・莉・].append(m)

    selected_models = state.get("selected_models", [])
    if not selected_models:
        selected_models = list(defaults)

    # 荳諡ｬ謫堺ｽ・    with ui.row().classes("q-gutter-xs q-mb-sm"):
        def _select_all():
            all_keys = [m["key"] for m in available]
            state["selected_models"] = all_keys
        def _select_defaults():
            state["selected_models"] = list(defaults)
        def _select_none():
            state["selected_models"] = []
        ui.button("蜈ｨ驕ｸ謚・, on_click=_select_all).props("flat dense size=xs color=cyan no-caps")
        ui.button("謗ｨ螂ｨ縺ｮ縺ｿ", on_click=_select_defaults).props("flat dense size=xs color=teal no-caps")
        ui.button("蜈ｨ隗｣髯､", on_click=_select_none).props("flat dense size=xs color=grey no-caps")

    # 繧ｫ繝・ざ繝ｪ縺斐→縺ｫ螻暮幕繝代ロ繝ｫ
    for cat_name, models in categories.items():
        if not models:
            continue
        n_selected = sum(1 for m in models if m["key"] in selected_models)
        with ui.expansion(
            f"{cat_name}  縲宣∈謚・ {n_selected} / 蜈ｨ{len(models)}蛟九・,
        ).classes("full-width q-mb-xs").props("dense"):
            for m in models:
                mkey = m["key"]
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
                        m["name"], value=is_checked,
                        on_change=_toggle,
                    )
                    if mkey in defaults:
                        ui.badge("謗ｨ螂ｨ", color="teal").props("dense").style("font-size: 0.78rem;")


# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・# 邨・∩蜷医ｏ縺帶焚繧ｵ繝槭Μ繝ｼ
# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・
def _render_combo_summary(state: dict) -> None:
    """繝ｪ繧｢繝ｫ繧ｿ繧､繝縺ｧ縲御ｽ暮壹ｊ縺ｮ繝代う繝励Λ繧､繝ｳ縺九阪ｒ陦ｨ遉ｺ縲・""
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
    status_text = "笨・驕ｩ蛻・ if n_total <= 50 else ("笞・・繧・ｄ螟壹＞" if n_total <= 200 else "閥 螟壹☆縺・)

    with ui.card().classes("full-width q-pa-sm").style(
        f"border:2px solid var(--q-{status_color});border-radius:8px;"
        "background:rgba(0,20,40,0.3);"
    ):
        with ui.row().classes("items-center justify-between full-width"):
            ui.label(f"箸 隧穂ｾ｡繝代う繝励Λ繧､繝ｳ謨ｰ: {n_total:,} 騾壹ｊ").classes("text-body1 text-bold")
            ui.badge(status_text, color=status_color)
        ui.label(
            f"impﾃ養n_imp} ﾂｷ sclﾃ養n_scl} ﾂｷ catﾃ養n_ci} ﾂｷ encﾃ養n_le} "
            f"ﾂｷ binﾃ養n_bi} ﾂｷ engﾃ養n_eng} ﾂｷ selﾃ養n_sel} ﾂｷ estﾃ養n_est}"
        ).classes("text-caption text-grey").style("font-size:0.7rem;")


# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・# 繝｡繧､繝ｳ繧ｨ繝ｳ繝医Μ繝ｼ繝昴う繝ｳ繝・# 笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊絶武笊・
def render_pipeline_config(state: dict) -> None:
    """繝代う繝励Λ繧､繝ｳ蜈ｨ險ｭ螳啅I繧偵Ξ繝ｳ繝繝ｪ繝ｳ繧ｰ縺吶ｋ縲・
    繝｡繧､繝ｳ逕ｻ髱｢縺ｫ縺ｯ繧ｵ繝槭Μ繝ｼ繧ｫ繝ｼ繝会ｼ九ム繧､繧｢繝ｭ繧ｰ襍ｷ蜍輔・繧ｿ繝ｳ繧定｡ｨ遉ｺ縲・    7繧ｹ繝・ャ繝励・隧ｳ邏ｰ險ｭ螳壹・繝繧､繧｢繝ｭ繧ｰ蜀・↓驟咲ｽｮ縲・    """
    from frontend_nicegui.components.dialog_manager import (
        create_settings_dialog,
        render_settings_summary,
    )

    # 邨・∩蜷医ｏ縺帶焚險育ｮ・    n_imp = max(1, len(state.get("_pg_num_imputers", ["mean"])))
    n_scl = max(1, len(state.get("_pg_num_scalers", ["standard"])))
    n_ci = max(1, len(state.get("_pg_cat_imputers", ["most_frequent"])))
    n_le = max(1, len(state.get("_pg_low_encoders", ["onehot"])))
    n_bi = max(1, len(state.get("_pg_bin_imputers", ["most_frequent"])))
    n_eng = max(1, len(state.get("_pg_engineer", ["none"])))
    n_sel = max(1, len(state.get("_pg_selectors", ["none"])))
    n_est = max(1, len(state.get("selected_models", [])))
    n_total = n_imp * n_scl * n_ci * n_le * n_bi * n_eng * n_sel * n_est

    # 繧ｹ繝・・繧ｿ繧ｹ
    if n_total <= 50:
        status_text = "笨・驕ｩ蛻・
        status_color = "teal"
    elif n_total <= 200:
        status_text = "笞・・繧・ｄ螟壹＞"
        status_color = "amber"
    else:
        status_text = "閥 螟壹☆縺・
        status_color = "red"

    # 繧ｵ繝槭Μ繝ｼ陦・    summary = [
        f"箸 隧穂ｾ｡繝代う繝励Λ繧､繝ｳ謨ｰ: {n_total:,} 騾壹ｊ ({status_text})",
        f"謨ｰ蛟､: impﾃ養n_imp} sclﾃ養n_scl} / 繧ｫ繝・ざ繝ｪ: encﾃ養n_le} / 迚ｹ蠕ｴ: engﾃ養n_eng} selﾃ養n_sel}",
        f"謗ｨ螳壼勣: {n_est}蛟矩∈謚・,
    ]
    excl = state.get("exclude_cols", [])
    if excl:
        summary.append(f"髯､螟門・: {len(excl)}蛟・)

    def _build_dialog_content():
        ui.label(
            "繧ｹ繝・ャ繝励ｒ繧ｿ繝悶〒蛻・崛縺・竊・蜷・せ繝・ャ繝励〒繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝繧帝∈謚槭・
            "隍・焚驕ｸ謚槭＠縺溷ｴ蜷医・蜈ｨ邨・∩蜷医ｏ縺帙ｒ閾ｪ蜍戊ｩ穂ｾ｡縲・
        ).classes("text-caption text-grey q-mb-sm")

        # 7繧ｹ繝・ャ繝励ち繝・        with ui.tabs().classes("full-width").props(
            "dense no-caps active-color=cyan indicator-color=cyan"
        ) as pg_tabs:
            ui.tab("pg_excl", label="圻 髯､螟・)
            ui.tab("pg_num", label="箸 謨ｰ蛟､")
            ui.tab("pg_cat", label="捷・・繧ｫ繝・ざ繝ｪ")
            ui.tab("pg_bin", label="笞｡ 繝舌う繝翫Μ")
            ui.tab("pg_eng", label="肌 迚ｹ蠕ｴ逕滓・")
            ui.tab("pg_sel", label="識 迚ｹ蠕ｴ驕ｸ謚・)
            ui.tab("pg_est", label="､・謗ｨ螳壼勣")

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
            with ui.tab_panel("pg_est"):
                _tab_estimator(state)

        # 邨・∩蜷医ｏ縺帶焚繧ｵ繝槭Μ繝ｼ
        ui.separator().classes("q-my-sm")
        _render_combo_summary(state)

    def _open_dialog():
        snapshot_keys = [
            "exclude_cols",
            "_pg_num_imputers", "_pg_num_scalers",
            "_pg_cat_imputers", "_pg_low_encoders", "_pg_high_encoders",
            "_pg_bin_imputers", "_pg_bin_encoders",
            "_pg_engineer", "_pg_poly_degree", "_pg_poly_ia",
            "_pg_selectors", "_pg_lasso_alpha", "_pg_lasso_mi",
            "_pg_kbest_k", "_pg_kbest_sf",
            "_pg_boruta_n", "_pg_boruta_mi",
            "selected_models",
        ]
        dlg = create_settings_dialog(
            title="笞呻ｸ・Pipeline 蜈ｨ險ｭ螳夲ｼ・TEP 0縲・・・,
            icon="tune",
            width="90vw",
            max_width="1100px",
            content_builder=_build_dialog_content,
            state=state,
            snapshot_keys=snapshot_keys,
        )
        dlg.open()

    render_settings_summary(
        icon="tune",
        title="Pipeline 蜈ｨ險ｭ螳・,
        summary_lines=summary,
        button_label="笞呻ｸ・繝代う繝励Λ繧､繝ｳ險ｭ螳壹ｒ螟画峩",
        on_click=_open_dialog,
        badge_text=f"{n_total:,}騾壹ｊ",
        badge_color=status_color,
    )
