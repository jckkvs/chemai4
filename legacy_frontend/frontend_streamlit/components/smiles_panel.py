"""
frontend_streamlit/components/smiles_panel.py

SMILES記述子変換パネル — 解析者目線UI。

「どのエンジンを使うか」ではなく
「どのような記述子が自分の解析に有効か」
を軸にUIを構成する。

カテゴリ:
  🧪 物理化学的性質   : 溶解度・分配係数・物性予測に有効（RDKit, GroupContrib）
  🔑 分子フィンガープリント : 活性予測・構造検索・類似度に有効（scikit-FP, Molfeat）
  📐 包括的記述子      : 網羅的QSAR/QSPR（Mordred, DescriptaStorus, PaDEL）
  🤖 深層学習表現      : 非線形な構造-活性関係（MolAI, Mol2Vec, Chemprop）
  ⚛️ 量子化学        : 電子状態・反応性予測（XTB, COSMO-RS, UniPKa, UMA）
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


# ── アダプタ読み込み ────────────────────────────────────────
def _load_adapters() -> dict:
    """各アダプタをbackend.chemから読み込む。失敗時もNoneにしない。"""
    import importlib
    adapters: dict = {}
    mod = importlib.import_module("backend.chem")

    _names = [
        "RDKitAdapter", "MordredAdapter", "DescriptaStorusAdapter",
        "PaDELAdapter", "GroupContribAdapter", "SkfpAdapter",
        "MolfeatAdapter", "MolAIAdapter", "Mol2VecAdapter",
        "ChempropAdapter", "XTBAdapter", "CosmoAdapter",
        "UniPkaAdapter", "UMAAdapter",
    ]
    for cls_name in _names:
        try:
            cls = getattr(mod, cls_name)
            adapters[cls_name] = cls()
        except Exception:
            adapters[cls_name] = None
    return adapters


# ── 解析目的別カテゴリ ──────────────────────────────────────
_PURPOSE_CATEGORIES = [
    {
        "icon": "🧪",
        "name": "物理化学的性質",
        "purpose": "溶解度・LogP・融点・沸点など物性の予測に最適",
        "level": "🟢 初級",
        "default_open": True,
        "engines": [
            {"cls": "RDKitAdapter",      "label": "RDKit 基本記述子",
             "dims": "~200種", "speed": "⚡高速",
             "desc": "MW, LogP, TPSA, HBA/HBD 等。まずこれだけで十分",
             "default_on": True},
            {"cls": "GroupContribAdapter","label": "基団寄与法",
             "dims": "~15種", "speed": "⚡高速",
             "desc": "Crippen LogP, MR, TPSA の分解"},
        ],
    },
    {
        "icon": "🔑",
        "name": "分子フィンガープリント",
        "purpose": "化合物の活性予測・構造類似度・バーチャルスクリーニングに有効",
        "level": "🟢 初級",
        "default_open": False,
        "engines": [
            {"cls": "SkfpAdapter",       "label": "scikit-fingerprints",
             "dims": "~2,200bit", "speed": "⚡高速",
             "desc": "ECFP, MACCS, Avalon 等 30+種類のFP"},
            {"cls": "MolfeatAdapter",    "label": "Molfeat",
             "dims": "可変", "speed": "⚡高速",
             "desc": "統合FPフレームワーク"},
        ],
    },
    {
        "icon": "📐",
        "name": "包括的記述子（QSAR/QSPR）",
        "purpose": "網羅的に記述子を生成し、特徴量選択と組み合わせて高精度モデルを構築",
        "level": "🟡 中級",
        "default_open": False,
        "engines": [
            {"cls": "MordredAdapter",    "label": "Mordred",
             "dims": "~1,800種", "speed": "🟡 中速",
             "desc": "2D/3D記述子を網羅的に計算"},
            {"cls": "DescriptaStorusAdapter", "label": "DescriptaStorus",
             "dims": "~200種", "speed": "⚡高速",
             "desc": "Merck開発の高速記述子"},
            {"cls": "PaDELAdapter",      "label": "PaDEL",
             "dims": "~1,800種", "speed": "🟡 中速",
             "desc": "PaDEL-Descriptor互換（Java必要）"},
        ],
    },
    {
        "icon": "🤖",
        "name": "深層学習ベースの分子表現",
        "purpose": "従来の記述子では捉えにくい非線形な構造-活性関係を学習",
        "level": "🟡 中級",
        "default_open": False,
        "engines": [
            {"cls": "MolAIAdapter",      "label": "MolAI (CNN+PCA)",
             "dims": "指定可", "speed": "🟡 中速",
             "desc": "CNN潜在ベクトル → PCAで次元圧縮",
             "has_pca": True},
            {"cls": "Mol2VecAdapter",    "label": "Mol2Vec",
             "dims": "300次元", "speed": "🟡 中速",
             "desc": "SMILES部分構造のWord2Vec分散表現"},
            {"cls": "ChempropAdapter",   "label": "Chemprop (D-MPNN)",
             "dims": "可変", "speed": "🔴 低速",
             "desc": "Directed Message Passing GNN"},
        ],
    },
    {
        "icon": "⚛️",
        "name": "量子化学・電子状態",
        "purpose": "HOMO-LUMO, 双極子モーメント, pKa等。反応性・触媒設計に",
        "level": "🔴 上級",
        "default_open": False,
        "engines": [
            {"cls": "XTBAdapter",        "label": "xTB (GFN2-xTB)",
             "dims": "~20種", "speed": "🔴 低速",
             "desc": "HOMO, LUMO, 双極子, 分極率…"},
            {"cls": "CosmoAdapter",      "label": "COSMO-RS",
             "dims": "~10種", "speed": "🔴 低速",
             "desc": "溶媒和自由エネルギー, σプロファイル"},
            {"cls": "UniPkaAdapter",     "label": "UniPKa",
             "dims": "~5種", "speed": "🟡 中速",
             "desc": "酸解離定数 pKa の予測"},
            {"cls": "UMAAdapter",        "label": "UMA (Meta FAIR)",
             "dims": "~7種", "speed": "🔴 低速",
             "desc": "DFTレベル分子物性"},
        ],
    },
]

# 目的別おすすめプリセット
# ※ 名称は化学的に正確に。溶解度は物性の一種であり並列しない。
# ※ QSAR/QSPRは手法名であり目的分類としては不適切。使い分ける用途で区分。
_PRESETS = {
    "🧪 基本物性（沸点・密度等）": ["RDKitAdapter", "GroupContribAdapter"],
    "🔑 構造活性相関（FP中心）": ["RDKitAdapter", "SkfpAdapter"],
    "📐 網羅的記述子（特徴量選択前提）": ["RDKitAdapter", "MordredAdapter"],
    "🧠 深層学習表現": ["MolAIAdapter", "Mol2VecAdapter"],
    "⚛️ 量子化学込み": ["RDKitAdapter", "XTBAdapter", "CosmoAdapter"],
}


def render_smiles_panel_content(smiles_col: str, df: pd.DataFrame) -> None:
    """SMILES記述子変換パネルの本体を描画する。"""

    # === アダプタ読み込み（キャッシュ + リフレッシュ） ===
    if "_chem_adapters" not in st.session_state:
        st.session_state["_chem_adapters"] = _load_adapters()
    adapters = st.session_state["_chem_adapters"]

    # === 利用可能状況サマリ ===
    def _is_avail(cls_name: str) -> bool:
        adp = adapters.get(cls_name)
        if adp is None:
            return False
        try:
            return adp.is_available()
        except Exception:
            return False

    _n_ok = sum(1 for c in adapters if _is_avail(c))
    _n_all = len(adapters)

    # === 計算済み？ ===
    if st.session_state.get("precalc_done"):
        _precalc_df = st.session_state.get("precalc_smiles_df")
        n_descs = len(_precalc_df.columns) if _precalc_df is not None else 0
        st.success(f"✅ {n_descs} 個の記述子が計算済みです。")
        if st.button("🔄 記述子を再計算する", key="recalc_smiles"):
            st.session_state["precalc_done"] = False
            st.session_state["precalc_smiles_df"] = None
            st.session_state.pop("_chem_adapters", None)
            st.rerun()
    else:
        st.info(
            f"⚙️ SMILES列 **{smiles_col}** から分子記述子を計算します。"
        )

    # ════════════════════════════════════════════════════════════
    # 🎯 目的別プリセット（ワンクリック設定）
    # ════════════════════════════════════════════════════════════
    with st.expander("🎯 目的別おすすめセット（ワンクリック）", expanded=not st.session_state.get("precalc_done", False)):
        st.caption("解析の目的に合わせてエンジンをまとめてON。あとから個別変更も可能です。")
        _preset_cols = st.columns(len(_PRESETS))
        for _pi, (_pname, _pengines) in enumerate(_PRESETS.items()):
            with _preset_cols[_pi]:
                _all_avail = all(_is_avail(e) for e in _pengines)
                _btn_label = _pname if _all_avail else f"{_pname} ⚠️"
                if st.button(_btn_label, key=f"preset_{_pi}", use_container_width=True,
                             disabled=not _all_avail):
                    # 全エンジンOFF → 選択されたエンジンだけON
                    for cat in _PURPOSE_CATEGORIES:
                        for eng in cat["engines"]:
                            _k = f"use_{eng['cls'].replace('Adapter','').lower()}"
                            st.session_state[_k] = eng["cls"] in _pengines
                    st.session_state["precalc_done"] = False
                    st.session_state["precalc_smiles_df"] = None
                    st.rerun()

    # ════════════════════════════════════════════════════════════
    # 📋 記述子カテゴリ別選択（解析目的ベース）
    # ════════════════════════════════════════════════════════════
    with st.expander(f"📋 記述子を個別に選択（{_n_ok}/{_n_all} エンジン利用可能）",
                     expanded=False):
        # リフレッシュボタン
        if st.button("🔄 エンジン状態を再チェック", key="refresh_adapters"):
            st.session_state.pop("_chem_adapters", None)
            st.rerun()

        for _cat in _PURPOSE_CATEGORIES:
            with st.expander(
                f"{_cat['icon']} {_cat['name']}  —  {_cat['purpose']}",
                expanded=_cat["default_open"],
            ):
                st.caption(f"レベル: {_cat['level']}")
                _eng_cols = st.columns(min(3, len(_cat["engines"])))
                for _ei, _eng in enumerate(_cat["engines"]):
                    with _eng_cols[_ei % len(_eng_cols)]:
                        _cls_name = _eng["cls"]
                        _avail = _is_avail(_cls_name)
                        _use_key = f"use_{_cls_name.replace('Adapter','').lower()}"
                        _default_val = _eng.get("default_on", False) and _avail
                        _curr_val = st.session_state.get(_use_key, _default_val)

                        # ステータス表示
                        if _avail:
                            _status = '<span style="color:#4ade80; font-size:0.75em">✅ 利用可</span>'
                        else:
                            _status = '<span style="color:#888; font-size:0.75em">🚫 未インストール</span>'

                        _border = "#4c9be8" if (_curr_val and _avail) else ("#555" if _avail else "#333")
                        _bg = "#0d1f2d" if (_curr_val and _avail) else "#0e1117"
                        st.markdown(
                            f'<div style="border:1px solid {_border}; border-radius:8px; '
                            f'padding:8px 12px; margin-bottom:4px; background:{_bg}">'
                            f'<b>{_eng["label"]}</b> {_status}<br>'
                            f'<span style="font-size:0.75em; color:#888">'
                            f'{_eng["speed"]} | {_eng["dims"]} | {_eng["desc"]}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        _new_val = st.checkbox(
                            "使用中 ✓" if _curr_val else "使用する",
                            value=_curr_val,
                            disabled=not _avail,
                            key=f"chk_{_use_key}",
                        )
                        if _new_val != _curr_val:
                            st.session_state[_use_key] = _new_val
                            st.session_state["precalc_done"] = False
                            st.session_state["precalc_smiles_df"] = None
                            st.rerun()

                        # MolAI PCA次元数
                        if _eng.get("has_pca") and _new_val and _avail:
                            _molai_n = st.slider(
                                "PCA次元数", 1, 256,
                                value=st.session_state.get("molai_n_components", 32),
                                step=1, key="slider_molai_n",
                            )
                            if _molai_n != st.session_state.get("molai_n_components", 32):
                                st.session_state["molai_n_components"] = _molai_n
                                st.session_state["precalc_done"] = False
                                st.rerun()

    # ════════════════════════════════════════════════════════════
    # ▶ 計算実行ボタン
    # ════════════════════════════════════════════════════════════
    if not st.session_state.get("precalc_done", False):
        # どのエンジンが選択されているか表示
        _selected = []
        for _cat in _PURPOSE_CATEGORIES:
            for _eng in _cat["engines"]:
                _k = f"use_{_eng['cls'].replace('Adapter','').lower()}"
                if st.session_state.get(_k, _eng.get("default_on", False)):
                    _selected.append(_eng["label"])
        if _selected:
            st.caption(f"選択中: {', '.join(_selected)}")
        else:
            st.caption("⚠️ エンジンが未選択です。上のプリセットから選ぶか、個別選択をしてください。")

        if st.button("▶ 記述子を計算", key="btn_precalc", type="primary", use_container_width=True):
            try:
                import importlib, sys
                _mod_key = "backend.chem.smiles_transformer"
                if _mod_key in sys.modules:
                    _mod = importlib.reload(sys.modules[_mod_key])
                else:
                    _mod = importlib.import_module(_mod_key)
                precalculate_all_descriptors = _mod.precalculate_all_descriptors
            except Exception as _imp_err:
                st.error(f"⚠️ 記述子計算モジュールの読み込みに失敗: {_imp_err}")
                return

            smiles_series = df[smiles_col]
            valid_mask = smiles_series.notna()
            smiles_list = smiles_series[valid_mask].tolist()
            valid_idx = smiles_series[valid_mask].index
            n = len(smiles_list)

            _engine_flags = {}
            for _cat in _PURPOSE_CATEGORIES:
                for _eng in _cat["engines"]:
                    _k = f"use_{_eng['cls'].replace('Adapter','').lower()}"
                    _engine_flags[_k] = st.session_state.get(_k, _eng.get("default_on", False))

            target_name = st.session_state.get("target_col", "")
            molai_n = st.session_state.get("molai_n_components", 32)

            _progress = st.empty()
            def _ui_progress(step: int, total: int, msg: str) -> None:
                _progress.caption(f"🔄 [{step}/{total}] {msg}")

            with st.spinner(f"⚙️ {n} 件の記述子を計算中..."):
                df_result, molai_variance = precalculate_all_descriptors(
                    smiles_list=smiles_list,
                    target_col_name=target_name,
                    engine_flags=_engine_flags,
                    molai_n_components=molai_n,
                    progress_callback=_ui_progress,
                )

            df_result.index = valid_idx
            _progress.empty()
            st.session_state["molai_explained_variance"] = molai_variance
            st.session_state["precalc_smiles_df"] = df_result
            st.session_state["precalc_done"] = True
            st.rerun()

    # ════════════════════════════════════════════════════════════
    # 🔴 上級者向け: 電荷・スピン設定
    # ════════════════════════════════════════════════════════════
    with st.expander("🔴 上級者向け: 分子電荷・スピン設定", expanded=False):
        st.caption(
            "XTBやCOSMO等の量子化学計算で使用する電荷・スピン多重度を設定します。"
            "通常は変更不要です。"
        )
        try:
            from frontend_streamlit.components.charge_config_ui import render_charge_config_panel
            render_charge_config_panel(smiles_col, df)
        except ImportError:
            st.info("電荷設定パネルは利用できません。")

    # ════════════════════════════════════════════════════════════
    # MolAI PCA寄与率（計算完了後）
    # ════════════════════════════════════════════════════════════
    _mev = st.session_state.get("molai_explained_variance")
    if _mev and _mev.get("ratio"):
        import plotly.graph_objects as _go_ev
        _evr = _mev["ratio"]; _evc = _mev["cumulative"]; _nc = _mev["n_components"]
        _pcs = [f"PC{i+1}" for i in range(len(_evr))]
        with st.expander(f"📊 MolAI PCA 寄与率（n={_nc}）", expanded=False):
            _fig_ev = _go_ev.Figure()
            _fig_ev.add_bar(x=_pcs, y=[v*100 for v in _evr], name="寄与率 (%)", marker_color="#4c9be8")
            _fig_ev.add_scatter(x=_pcs, y=[v*100 for v in _evc], name="累積寄与率 (%)",
                                mode="lines+markers", yaxis="y2",
                                line=dict(color="#f4a261", width=2), marker=dict(size=5))
            _fig_ev.update_layout(
                yaxis=dict(title="寄与率 (%)", range=[0, max(v*100 for v in _evr)*1.15]),
                yaxis2=dict(title="累積寄与率 (%)", overlaying="y", side="right", range=[0,105], showgrid=False),
                legend=dict(orientation="h", y=1.15), height=300, margin=dict(l=10, r=10, t=40, b=40),
            )
            st.plotly_chart(_fig_ev, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # 記述子の選択・絞り込み（計算完了後のみ）
    # ════════════════════════════════════════════════════════════
    if st.session_state.get("precalc_smiles_df") is not None:
        with st.expander("🔬 記述子の絞り込み", expanded=False):
            st.caption("計算済みの記述子から解析に使う記述子を絞り込めます。")

            _precalc_df = st.session_state["precalc_smiles_df"]
            _target_col = st.session_state.get("target_col")
            _target_s = (df[_target_col]
                         if _target_col and _target_col in df.columns
                         and pd.api.types.is_numeric_dtype(df[_target_col])
                         else None)
            _corr: dict = {}
            if _target_s is not None:
                try:
                    _al = _target_s.loc[_precalc_df.index]
                    _corr = _precalc_df.corrwith(_al, method="pearson").abs().dropna().to_dict()
                except Exception:
                    pass

            _all_descs = list(_precalc_df.columns)
            _cur_sel = set(st.session_state.get("adv_desc", _all_descs))

            if _corr:
                _sorted = sorted(_all_descs, key=lambda d: _corr.get(d, 0), reverse=True)
                c1, c2 = st.columns(2)
                if c1.button("相関上位10件を選択", key="sel_top10"):
                    st.session_state["adv_desc"] = _sorted[:10]
                    st.rerun()
                if c2.button("相関上位30件を選択", key="sel_top30"):
                    st.session_state["adv_desc"] = _sorted[:30]
                    st.rerun()
                _selected_descs = st.multiselect(
                    "記述子を選択（|r|降順）",
                    options=_sorted,
                    default=[d for d in _sorted if d in _cur_sel],
                    format_func=lambda d: f"{d}  |r|={_corr.get(d, 0):.3f}",
                    key="desc_sel_corr",
                )
            else:
                if _target_col:
                    st.warning("相関係数の計算には数値型の目的変数が必要です。")
                _selected_descs = st.multiselect(
                    "記述子を選択",
                    options=_all_descs,
                    default=[d for d in _all_descs if d in _cur_sel],
                    key="desc_sel_all",
                )

            st.session_state["adv_desc"] = _selected_descs
