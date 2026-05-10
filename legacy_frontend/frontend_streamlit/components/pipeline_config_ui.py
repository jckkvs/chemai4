"""
frontend_streamlit/components/pipeline_config_ui.py

Pipeline 全設定UI ── デザイナー＋プログラマー観点で設計

情報設計:
  ・パイプラインは 7 ステップの順次フロー → タブで一発切替
  ・各ステップでアルゴリズムを選択 → 選択後に全パラメータ入力欄がコンテキスト展開
  ・複数選択 = 全組み合わせを評価 / 未選択 = 適切なデフォルトを自動適用
  ・常に「何通りのPipelineになるか」をリアルタイム表示

永続ルール:
  ─ 全引数を UI から設定可能にする（一度決めた原則は永続）
  ─ 選択したアルゴリズムのパラメータは必ず表示する
"""
from __future__ import annotations
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# 汎用ヘルパー
# ─────────────────────────────────────────────────────────────────────────────

def _k(*parts) -> str:            return "pg_" + "_".join(str(p) for p in parts)
def _get(key, default=None):      return st.session_state.get(key, default)
def _cb(lbl, key, default=False): return st.checkbox(lbl, value=_get(key, default), key=key)

def _num(lbl, key, default, *, min_val=None, max_val=None, step=None, fmt=None):
    kw = dict(label=lbl, value=_get(key, default), key=key)
    if min_val is not None: kw["min_value"] = min_val
    if max_val is not None: kw["max_value"] = max_val
    if step    is not None: kw["step"]      = step
    if fmt     is not None: kw["format"]    = fmt
    return st.number_input(**kw)

def _sel(lbl, key, options, default=None):
    d = _get(key, default if default is not None else options[0])
    idx = options.index(d) if d in options else 0
    return st.selectbox(lbl, options, index=idx, key=key)

def _txt(lbl, key, default="", placeholder=""):
    return st.text_input(lbl, value=_get(key, default), key=key, placeholder=placeholder)

def _section_header(icon: str, title: str, summary: str = "") -> None:
    """ステップのセクションヘッダー（アイコン + タイトル + 説明）。"""
    st.markdown(f"**{icon} {title}**")
    if summary:
        st.caption(summary)


# ─────────────────────────────────────────────────────────────
# 単調性制約 per-feature UI
# ─────────────────────────────────────────────────────────────

_MONOTONIC_OPTS = ["なし (0)", "増加 (+1)", "減少 (-1)"]
_MONOTONIC_MAP  = {"なし (0)": 0, "増加 (+1)": 1, "減少 (-1)": -1}
_MONOTONIC_REV  = {0: "なし (0)", 1: "増加 (+1)", -1: "減少 (-1)"}

# ── 有機化学/ADMET向け単調性プリセット定義 ─────────────────────
# {プリセット名: {部分一致キーワード: 単調値} }
# キーワードはcase-insensitiveで列名に部分一致
_ADMET_PRESETS: dict[str, dict[str, int]] = {
    "沸点/融点予測": {
        # 分子量↑・重原子数↑・回転可能結合数↑ → 沸点/融点↑
        "molwt": 1, "mw": 1, "mol_wt": 1, "molecularweight": 1,
        "heavyatom": 1, "heavyatoms": 1, "nha": 1,
        "rotbond": 1, "rotatable": 1, "nrotb": 1,
        "ringcount": 1, "ring": 1,
    },
    "水溶性予測": {
        # LogP↑ → 溶解度↓ / HBD・HBA↑ → 溶解度↑
        "logp": -1, "alogp": -1, "xlogp": -1, "clogp": -1,
        "hbd": 1, "hbond_donor": 1, "nhbd": 1, "numhdonor": 1,
        "hba": 1, "hbond_acceptor": 1, "nhba": 1, "numhacceptor": 1,
        "tpsa": 1,
    },
    "膜透過性/BBB": {
        # LogP増加→膜透過↑ / TPSA増加→膜透過↓ / 分子量増加→BBB通過↓
        "logp": 1, "alogp": 1, "xlogp": 1,
        "tpsa": -1,
        "molwt": -1, "mw": -1,
        "hbd": -1, "numhdonor": -1,
    },
    "毒性予測": {
        # 一般的に毒性はLogPに正相関、水溶性に負相関
        "logp": 1, "alogp": 1,
        "molwt": 1,
        "ring": 1, "aromatic": 1,
    },
}


def _apply_admet_preset(
    feature_cols: list[str],
    preset_name: str,
) -> None:
    """指定プリセットを列名の部分一致で適用する。"""
    preset = _ADMET_PRESETS.get(preset_name, {})
    for col in feature_cols:
        col_lower = col.lower()
        matched = 0
        for kw, val in preset.items():
            if kw in col_lower:
                matched = val
                break
        if matched != 0:
            st.session_state[_k("mono", col)] = _MONOTONIC_REV[matched]


def render_monotonic_constraints_ui(
    feature_cols: list[str],
    *,
    n_cols: int = 4,
) -> dict[str, int]:
    """
    各特徴量ごとに単調性制約（増加/減少/なし）を選択するUIを描画する。

    単調性制約はカーネル系モデル（SVR/GPR/KernelRidge/SVC）では
    ソフト制約（MonotonicKernelWrapper）として、
    XGBoost/LightGBM/HistGBではネイティブ制約として適用される。

    Args:
        feature_cols: 特徴量列名のリスト
        n_cols: 1行あたりの表示列数

    Returns:
        {列名: 0|1|-1} の辞書
    """
    if not feature_cols:
        return {}

    _section_header(
        "📐", "特徴量ごとの単調性制約",
        "各変数に対して増加・減少・なしの単調性制約を設定します。\n"
        "カーネル系モデル(SVR/GPR等)はソフト近似制約、XGBoost/LGBMはネイティブ制約として適用されます。"
    )

    # ── 一括操作ボタン ─────────────────────────────────────────
    st.caption("**一括操作**")
    c_all, c_inc, c_dec, _ = st.columns([1, 1, 1, 3])
    if c_all.button("🔄 全て解除", key="mono_all_reset", use_container_width=True):
        for col in feature_cols:
            st.session_state[_k("mono", col)] = "なし (0)"
    if c_inc.button("⬆️ 全て増加", key="mono_all_inc", use_container_width=True):
        for col in feature_cols:
            st.session_state[_k("mono", col)] = "増加 (+1)"
    if c_dec.button("⬇️ 全て減少", key="mono_all_dec", use_container_width=True):
        for col in feature_cols:
            st.session_state[_k("mono", col)] = "減少 (-1)"

    # ── 有機化学/ADMET プリセット ───────────────────────────────
    st.caption("**🧪 有機化学プリセット（列名を部分一致でADMET単調性を自動設定）**")
    preset_cols = st.columns(len(_ADMET_PRESETS))
    for i, preset_name in enumerate(_ADMET_PRESETS.keys()):
        if preset_cols[i].button(
            f"🎯 {preset_name}", key=f"mono_preset_{i}", use_container_width=True
        ):
            _apply_admet_preset(feature_cols, preset_name)
            st.rerun()

    st.divider()

    result: dict[str, int] = {}
    cols_iter = st.columns(n_cols)
    for i, feat in enumerate(feature_cols):
        key = _k("mono", feat)
        saved = _get(key, "なし (0)")
        if saved not in _MONOTONIC_OPTS:
            saved = "なし (0)"
        choice = cols_iter[i % n_cols].selectbox(
            label=feat,
            options=_MONOTONIC_OPTS,
            index=_MONOTONIC_OPTS.index(saved),
            key=key,
        )
        result[feat] = _MONOTONIC_MAP[choice]

    # サマリー表示
    n_constrained = sum(1 for v in result.values() if v != 0)
    if n_constrained > 0:
        constrained_feats = [f for f, v in result.items() if v != 0]
        st.success(
            f"✅ **{n_constrained}変数** に単調性制約が設定されています: "
            f"{', '.join(constrained_feats[:5])}{'...' if len(constrained_feats) > 5 else ''}"
        )
    else:
        st.info("ℹ️ 単調性制約なし（全変数に適用しない）")

    return result



# ─────────────────────────────────────────────────────────────────────────────
# 各アルゴリズムのパラメータUI（コンテキスト展開）
# ─────────────────────────────────────────────────────────────────────────────

def _params_knn_imputer(prefix: str) -> dict:
    c1, c2 = st.columns(2)
    n = int(c1.number_input("n_neighbors", min_value=1, max_value=50,
                             value=_get(_k(prefix,"nn"), 5), step=1, key=_k(prefix,"nn")))
    w = c2.selectbox("weights", ["uniform", "distance"], key=_k(prefix,"w"))
    return {"n_neighbors": n, "weights": w}

def _params_iterative_imputer(prefix: str) -> dict:
    c1, c2, c3 = st.columns(3)
    max_iter = int(c1.number_input("max_iter",    min_value=1, max_value=200,
                                    value=_get(_k(prefix,"mi"), 10), step=1, key=_k(prefix,"mi")))
    tol      = float(c2.number_input("tol",        min_value=0.0,
                                      value=_get(_k(prefix,"tol"), 0.001), step=0.0001, format="%.6f", key=_k(prefix,"tol")))
    rs       = int(c3.number_input("random_state", min_value=0,
                                    value=_get(_k(prefix,"rs"), 0), step=1, key=_k(prefix,"rs")))
    return {"max_iter": max_iter, "tol": tol, "random_state": rs}


# ─────────────────────────────────────────────────────────────────────────────
# TAB 0: Excluder
# ─────────────────────────────────────────────────────────────────────────────

def _tab_excluder(all_cols, target_col, smiles_col) -> list[str]:
    skip = {c for c in (target_col, smiles_col) if c}
    opts = [c for c in all_cols if c not in skip]
    _section_header("🚫", "Excluder（解析除外列）",
                    "解析に使わない列を選択。未選択なら除外なし。目的変数・SMILES列は自動除外済み。")
    if not opts:
        st.info("除外できる列がありません（全列が目的変数 or SMILES）")
        return []
    prev = [c for c in _get("pg_excl", []) if c in opts]
    return st.multiselect("除外列", options=opts, default=prev, key="pg_excl",
                          label_visibility="collapsed")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: 数値列前処理
# ─────────────────────────────────────────────────────────────────────────────

def _tab_numeric() -> dict:
    result = {"imputers": [], "scalers": []}
    _section_header("🔢", "数値列前処理（Imputer ✕ Scaler）",
                    "選択した Imputer と Scaler の全組み合わせを評価します。未選択 → `mean` / `standard` を自動適用。")

    # ─── Imputer ───
    st.markdown("#### 📊 Imputer（欠損補間）")

    IMPUTERS = [
        ("mean",      "Mean（平均）",         False),
        ("median",    "Median（中央値）",      False),
        ("knn",       "KNN Imputer",           True),
        ("iterative", "Iterative Imputer",      True),
        ("constant",  "Constant（固定値）",    True),
    ]
    for key, label, has_params in IMPUTERS:
        checked = _cb(label, _k("ni", key), default=(key == "mean"))
        if checked:
            if has_params:
                with st.container(border=True):
                    if key == "knn":
                        p = _params_knn_imputer(_k("ni","knn"))
                    elif key == "iterative":
                        p = _params_iterative_imputer(_k("ni","iter"))
                    else:  # constant
                        fv = float(_num("fill_value", _k("ni","const","fv"), 0.0, step=0.1, fmt="%.4f"))
                        p = {"fill_value": fv}
            else:
                p = {}
            result["imputers"].append((key, p))

    if not result["imputers"]:
        st.caption("→ デフォルト: `mean`")
        result["imputers"] = [("mean", {})]

    st.divider()

    # ─── Scaler ───
    st.markdown("#### 📏 Scaler")

    SCALERS = [
        ("standard",         "StandardScaler",            True),
        ("minmax",           "MinMaxScaler",               True),
        ("robust",           "RobustScaler",               True),
        ("maxabs",           "MaxAbsScaler",               False),
        ("power_yj",         "PowerTransformer [YJ]",     False),
        ("power_bc",         "PowerTransformer [BC]",     False),
        ("quantile_normal",  "QuantileTransformer→正規",   True),
        ("quantile_uniform", "QuantileTransformer→一様",   True),
        ("none",             "スケーリングなし",             False),
    ]
    for key, label, has_params in SCALERS:
        checked = _cb(label, _k("scl", key), default=(key == "standard"))
        if checked:
            p = {}
            if has_params:
                with st.container(border=True):
                    if key == "standard":
                        c1, c2 = st.columns(2)
                        p["with_mean"] = c1.checkbox("with_mean", key=_k("scl","std","wm"), value=_get(_k("scl","std","wm"), True))
                        p["with_std"]  = c2.checkbox("with_std",  key=_k("scl","std","ws"), value=_get(_k("scl","std","ws"), True))
                    elif key == "minmax":
                        c1, c2 = st.columns(2)
                        lo = float(c1.number_input("min", value=_get(_k("scl","mm","lo"), 0.0), key=_k("scl","mm","lo")))
                        hi = float(c2.number_input("max", value=_get(_k("scl","mm","hi"), 1.0), key=_k("scl","mm","hi")))
                        p["feature_range"] = (lo, hi)
                    elif key == "robust":
                        c1, c2 = st.columns(2)
                        qlo = float(c1.number_input("quantile_range low%",  value=_get(_k("scl","rb","lo"), 25.0), key=_k("scl","rb","lo")))
                        qhi = float(c2.number_input("quantile_range high%", value=_get(_k("scl","rb","hi"), 75.0), key=_k("scl","rb","hi")))
                        p["quantile_range"] = (qlo, qhi)
                    elif key in ("quantile_normal", "quantile_uniform"):
                        nq = int(_num("n_quantiles", _k("scl","qt",key,"nq"), 1000, min_val=10, max_val=10000, step=100))
                        p["n_quantiles"] = nq
            result["scalers"].append((key, p))

    if not result["scalers"]:
        st.caption("→ デフォルト: `standard`")
        result["scalers"] = [("standard", {})]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: カテゴリ列前処理
# ─────────────────────────────────────────────────────────────────────────────

def _tab_categorical() -> dict:
    result = {"imputers": [], "low_encoders": [], "high_encoders": []}
    _section_header("🏷️", "カテゴリ列前処理（Imputer ✕ Encoder）",
                    "低カーディナリティ（少種類）と高カーディナリティ（多種類）で Encoder を別設定できます。")

    # ─ Imputer ─
    st.markdown("#### 🔤 Categorical Imputer")
    CI = [("most_frequent", "Most Frequent（最頻値）", False),
          ("constant",       "Constant（指定文字列）",  True),
          ("knn",            "KNN Imputer",              True)]
    for key, label, has_params in CI:
        if _cb(label, _k("ci", key), default=(key == "most_frequent")):
            p = {}
            if has_params:
                with st.container(border=True):
                    if key == "constant":
                        p["fill_value"] = _txt("fill_value", _k("ci","const","fv"), default="missing")
                    elif key == "knn":
                        p = _params_knn_imputer(_k("ci","knn"))
            result["imputers"].append((key, p))
    if not result["imputers"]:
        st.caption("→ デフォルト: `most_frequent`")
        result["imputers"] = [("most_frequent", {})]

    st.divider()

    # ─ タブで Low / High を切替（適材適所のタブ活用）─
    enc_tab_low, enc_tab_high = st.tabs(["🔻 低カーディナリティ Encoder", "🔺 高カーディナリティ Encoder"])

    with enc_tab_low:
        LOW_ENC = [
            ("onehot",  "OneHotEncoder",  True),
            ("ordinal", "OrdinalEncoder", False),
            ("target",  "TargetEncoder",  False),
            ("binary",  "BinaryEncoder",  False),
            ("woe",     "WOE Encoder",    False),
        ]
        for key, label, has_params in LOW_ENC:
            if _cb(label, _k("le", key), default=(key == "onehot")):
                p = {}
                if has_params:
                    with st.container(border=True):
                        c1, c2, c3 = st.columns(3)
                        p["drop"]           = c1.selectbox("drop", ["first","if_binary","None"], key=_k("ohe","drop"))
                        p["handle_unknown"] = c2.selectbox("handle_unknown", ["ignore","infrequent_if_exist","error"], key=_k("ohe","hu"))
                        mc = int(c3.number_input("max_categories (0=無制限)", min_value=0,
                                                  value=_get(_k("ohe","mc"), 0), step=1, key=_k("ohe","mc")))
                        p["max_categories"] = mc if mc > 0 else None
                result["low_encoders"].append((key, p))
        if not result["low_encoders"]:
            st.caption("→ デフォルト: `onehot`")
            result["low_encoders"] = [("onehot", {})]

    with enc_tab_high:
        HIGH_ENC = [
            ("ordinal",     "OrdinalEncoder", False),
            ("target",      "TargetEncoder",  False),
            ("hashing",     "HashingEncoder", True),
            ("binary",      "BinaryEncoder",  False),
            ("leaveoneout", "LeaveOneOut",     False),
        ]
        for key, label, has_params in HIGH_ENC:
            if _cb(label, _k("he", key), default=(key == "ordinal")):
                p = {}
                if has_params:
                    with st.container(border=True):
                        nc = int(_num("n_components", _k("he","hash","nc"), 8, min_val=1, max_val=256, step=1))
                        p["n_components"] = nc
                result["high_encoders"].append((key, p))
        if not result["high_encoders"]:
            st.caption("→ デフォルト: `ordinal`")
            result["high_encoders"] = [("ordinal", {})]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: バイナリ列前処理
# ─────────────────────────────────────────────────────────────────────────────

def _tab_binary() -> dict:
    result = {"imputers": [], "encoders": []}
    _section_header("⚡", "バイナリ列前処理（Imputer & Encoder）",
                    "0/1・True/False などの2値列の処理設定。")

    st.markdown("#### 🔤 Binary Imputer")
    BI = [("most_frequent", "Most Frequent", False),
          ("constant",       "Constant",      True),
          ("knn",            "KNN",           True)]
    for key, label, has_params in BI:
        if _cb(label, _k("bi", key), default=(key == "most_frequent")):
            p = {}
            if has_params:
                with st.container(border=True):
                    if key == "constant":
                        p["fill_value"] = float(_num("fill_value", _k("bi","const","fv"), 0.0, step=1.0))
                    elif key == "knn":
                        p = _params_knn_imputer(_k("bi","knn"))
            result["imputers"].append((key, p))
    if not result["imputers"]:
        result["imputers"] = [("most_frequent", {})]

    st.divider()
    st.markdown("#### 🔢 Binary Encoder")
    if _cb("OrdinalEncoder",          _k("benc","ord"),  default=True):  result["encoders"].append("ordinal")
    if _cb("passthrough (そのまま)", _k("benc","pass"), default=False): result["encoders"].append("passthrough")
    if not result["encoders"]:
        result["encoders"] = ["ordinal"]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def _tab_engineer() -> dict:
    result = {"methods": []}
    _section_header("🔧", "Feature Engineering",
                    "複数選択した場合、全パターンを評価します。未選択 → none（生成なし）。")

    if _cb("none（生成なし）", _k("eng","none"), default=True):
        result["methods"].append({"method": "none"})

    if _cb("PolynomialFeatures", _k("eng","poly"), default=False):
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            deg = int(c1.number_input("degree", min_value=2, max_value=5,
                                       value=_get(_k("eng","poly","deg"), 2), step=1, key=_k("eng","poly","deg")))
            ia  = c2.checkbox("interaction_only", key=_k("eng","poly","ia"), value=_get(_k("eng","poly","ia"), False))
            ib  = c3.checkbox("include_bias",     key=_k("eng","poly","ib"), value=_get(_k("eng","poly","ib"), True))
        result["methods"].append({"method": "polynomial", "degree": deg, "interaction_only": ia, "include_bias": ib})

    if _cb("Interaction Only", _k("eng","ia"), default=False):
        with st.container(border=True):
            deg = int(_num("degree", _k("eng","ia","deg"), 2, min_val=2, max_val=4, step=1))
        result["methods"].append({"method": "interaction_only", "degree": deg})

    if not result["methods"]:
        st.caption("→ デフォルト: `none`")
        result["methods"] = [{"method": "none"}]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: Feature Selector
# ─────────────────────────────────────────────────────────────────────────────

def _tab_selector() -> dict:
    result = {"methods": []}
    _section_header("🎯", "Feature Selector",
                    "複数選択すると全組み合わせを評価。未選択 → none（全特徴量を使用）。")

    SCORE_F = ["f_regression", "mutual_info_regression", "r_regression",
               "f_classif", "mutual_info_classif", "chi2"]

    if _cb("なし（全特徴量を使用）", _k("sel","none"), default=True):
        result["methods"].append({"method": "none"})

    if _cb("Lasso（線形ペナルティ）", _k("sel","lasso"), default=False):
        with st.container(border=True):
            c1, c2 = st.columns(2)
            a  = float(c1.number_input("alpha", min_value=1e-6, value=_get(_k("sel","lasso","a"), 0.01), format="%.6f", key=_k("sel","lasso","a")))
            mi = int(c2.number_input("max_iter", min_value=100, max_value=10000, value=_get(_k("sel","lasso","mi"), 1000), step=100, key=_k("sel","lasso","mi")))
        result["methods"].append({"method": "lasso", "alpha": a, "max_iter": mi})

    if _cb("RF 重要度（SelectFromModel）", _k("sel","rfr"), default=False):
        with st.container(border=True):
            c1, c2 = st.columns(2)
            thr = c1.selectbox("threshold", ["mean", "median", "0.01", "0.001"], key=_k("sel","rfr","thr"))
            mf  = int(c2.number_input("max_features (0=なし)", min_value=0, value=_get(_k("sel","rfr","mf"), 0), step=1, key=_k("sel","rfr","mf")))
        result["methods"].append({"method": "rfr", "threshold": thr, "max_features": mf or None})

    if _cb("SelectKBest", _k("sel","kb"), default=False):
        with st.container(border=True):
            c1, c2 = st.columns(2)
            k  = int(c1.number_input("k", min_value=1, max_value=500, value=_get(_k("sel","kb","k"), 10), step=1, key=_k("sel","kb","k")))
            sf = c2.selectbox("score_func", SCORE_F, key=_k("sel","kb","sf"))
        result["methods"].append({"method": "select_kbest", "k": k, "score_func": sf})

    if _cb("SelectPercentile", _k("sel","pct"), default=False):
        with st.container(border=True):
            c1, c2 = st.columns(2)
            pct = int(c1.number_input("percentile %", min_value=1, max_value=99, value=_get(_k("sel","pct","p"), 50), step=1, key=_k("sel","pct","p")))
            sf  = c2.selectbox("score_func", SCORE_F, key=_k("sel","pct","sf"))
        result["methods"].append({"method": "select_percentile", "percentile": pct, "score_func": sf})

    if _cb("Boruta", _k("sel","boruta"), default=False):
        with st.container(border=True):
            c1, c2 = st.columns(2)
            bn  = int(c1.number_input("n_estimators", min_value=10, max_value=500, value=_get(_k("sel","boruta","n"), 100), step=10, key=_k("sel","boruta","n")))
            bmi = int(c2.number_input("max_iter",     min_value=10, max_value=500, value=_get(_k("sel","boruta","mi"), 100), step=10, key=_k("sel","boruta","mi")))
        result["methods"].append({"method": "boruta", "n_estimators": bn, "boruta_max_iter": bmi})

    if not result["methods"]:
        st.caption("→ デフォルト: `none`")
        result["methods"] = [{"method": "none"}]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6: Estimator
# ─────────────────────────────────────────────────────────────────────────────

def _tab_estimator(task: str = "regression") -> dict:
    """
    Estimatorタブ — 自動パラメータUI生成版。

    factory.pyのモデルレジストリからモデルクラスを取得し、
    introspect_params()で__init__パラメータを自動検出、
    auto_params_ui.render_param_editorで自動UIを生成する。

    新モデル追加時にUIコード変更は不要。
    """
    result = {"estimators": []}
    REG = (task == "regression")
    _section_header("🤖", f"Estimator（{'回帰' if REG else '分類'}）",
                    "使用するモデルを選択し、パラメータを設定してください。\n"
                    "各モデルの引数はクラスから自動検出されるため、新モデル追加時にUI変更は不要です。")

    try:
        from backend.models.factory import list_models, get_default_automl_models
        from backend.ui.param_schema import introspect_params, apply_params
        from frontend_streamlit.components.auto_params_ui import render_param_editor

        available = list_models(task=task, available_only=True)
        defaults = get_default_automl_models(task=task)

        # カテゴリ分類
        categories: dict[str, list] = {"線形系": [], "決定木/アンサンブル": [], "カーネル/その他": []}
        for m in available:
            k = m["key"].lower() + m["name"].lower()
            if any(x in k for x in ["linear", "ridge", "lasso", "elastic", "logistic", "ard", "huber", "pls", "bayesian"]):
                categories["線形系"].append(m)
            elif any(x in k for x in ["tree", "forest", "boost", "gbm", "gradient", "rgf", "figs", "rule", "hist", "catboost"]):
                categories["決定木/アンサンブル"].append(m)
            else:
                categories["カーネル/その他"].append(m)

        # タブで表示
        tab_labels = [f"📐 {cat} ({len(models)})" for cat, models in categories.items() if models]
        tab_data = [(cat, models) for cat, models in categories.items() if models]

        if tab_labels:
            tabs = st.tabs(tab_labels)
            for tab, (cat_name, models) in zip(tabs, tab_data):
                with tab:
                    for m in models:
                        mkey = m["key"]
                        is_default = mkey in defaults
                        checked = _cb(
                            m["name"],
                            _k("est", mkey),
                            default=is_default,
                        )
                        if checked:
                            model_cls = m.get("class")
                            if model_cls is not None:
                                with st.expander(f"⚙️ {m['name']} パラメータ", expanded=False):
                                    specs = introspect_params(model_cls)
                                    if specs:
                                        user_vals = render_param_editor(
                                            specs,
                                            title=m["name"],
                                            key_prefix=f"est_{mkey}_",
                                        )
                                        params = apply_params(specs, user_vals)
                                    else:
                                        params = {}
                                        st.caption("ℹ️ パラメータ設定なし")
                            else:
                                params = {}
                            result["estimators"].append((mkey, params))

    except Exception as ex:
        st.error(f"モデル一覧取得エラー: {ex}")
        st.info("フォールバック: factoryの代わりに基本モデルを表示")
        # フォールバック: 基本的なモデルのみ
        _FALLBACK = [
            ("RandomForestRegressor", "rf", "sklearn.ensemble", True),
            ("Ridge", "ridge", "sklearn.linear_model", False),
            ("SVR", "svr", "sklearn.svm", False),
        ] if REG else [
            ("RandomForestClassifier", "rf_c", "sklearn.ensemble", True),
            ("LogisticRegression", "logistic", "sklearn.linear_model", False),
        ]
        for cls_name, mkey, mod_path, is_default in _FALLBACK:
            checked = _cb(cls_name, _k("est", mkey), default=is_default)
            if checked:
                try:
                    import importlib
                    mod = importlib.import_module(mod_path)
                    cls = getattr(mod, cls_name)
                    from backend.ui.param_schema import introspect_params, apply_params
                    from frontend_streamlit.components.auto_params_ui import render_param_editor
                    specs = introspect_params(cls)
                    if specs:
                        with st.expander(f"⚙️ {cls_name} パラメータ"):
                            user_vals = render_param_editor(specs, title=cls_name, key_prefix=f"est_{mkey}_")
                            params = apply_params(specs, user_vals)
                    else:
                        params = {}
                except Exception:
                    params = {}
                result["estimators"].append((mkey, params))

    if not result["estimators"]:
        st.caption("→ デフォルト: RandomForest")
        result["estimators"].append(("rf", {}))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# メインエントリーポイント
# ─────────────────────────────────────────────────────────────────────────────

def render_pipeline_config_ui(
    all_cols: list[str],
    target_col: str | None,
    smiles_col: str | None,
    task: str = "regression",
) -> dict:
    """
    Pipeline 全設定UIを描画し、設定辞書を返す。

    情報設計:
      - タブでステップを切替（パイプラインの流れに沿った自然なナビゲーション）
      - アルゴリズム選択後にパラメータ入力欄がコンテキスト展開
      - 全引数設定可能（永続ルール）
      - 未設定は適切なデフォルトを自動適用
    """
    st.markdown("---")
    st.markdown("### ⚙️ Pipeline 全設定（STEP 0〜6）")
    st.caption(
        "**ステップをタブで切替え** → 各ステップでアルゴリズムを選択するとパラメータ入力欄が展開します。  "
        "複数選択した場合は **全組み合わせを自動評価**。未選択はデフォルト値を自動適用。"
    )

    t0, t1, t2, t3, t4, t5, t6 = st.tabs([
        "🚫 除外",
        "🔢 数値前処理",
        "🏷️ カテゴリ",
        "⚡ バイナリ",
        "🔧 特徴生成",
        "🎯 特徴選択",
        "🤖 推定器",
    ])

    cfg: dict = {}
    with t0: cfg["exclude_columns"] = _tab_excluder(all_cols, target_col, smiles_col)
    with t1: cfg["numeric"]         = _tab_numeric()
    with t2: cfg["categorical"]     = _tab_categorical()
    with t3: cfg["binary"]          = _tab_binary()
    with t4: cfg["engineer"]        = _tab_engineer()
    with t5: cfg["selector"]        = _tab_selector()
    with t6: cfg["estimator"]       = _tab_estimator(task=task)

    # ── 組合せサマリー（常時表示）──
    n_imp = len(cfg["numeric"]["imputers"])
    n_scl = len(cfg["numeric"]["scalers"])
    n_ci  = len(cfg["categorical"]["imputers"])
    n_le  = len(cfg["categorical"]["low_encoders"])
    n_bi  = len(cfg["binary"]["imputers"])
    n_eng = len(cfg["engineer"]["methods"])
    n_sel = len(cfg["selector"]["methods"])
    n_est = len(cfg["estimator"]["estimators"])
    n_total = n_imp * n_scl * n_ci * n_le * n_bi * n_eng * n_sel * n_est

    with st.container(border=True):
        cols = st.columns([3, 1])
        cols[0].markdown(
            f"**🔢 評価パイプライン数：{n_total:,} 通り**  \n"
            f"<small>imp×{n_imp} · scl×{n_scl} · cat_imp×{n_ci} · enc×{n_le} "
            f"· bin×{n_bi} · eng×{n_eng} · sel×{n_sel} · est×{n_est}</small>",
            unsafe_allow_html=True,
        )
        status = "✅ 評価実行可能" if n_total <= 50 else f"⚠️ {n_total:,}通りは多い可能性があります"
        cols[1].markdown(f"**{status}**")

    st.session_state["_pipeline_full_config"] = cfg
    return cfg
