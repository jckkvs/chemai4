"""
frontend_streamlit/components/charge_config_ui.py

SMILES 分子への電荷・スピン・プロトン化状態設定 UI コンポーネント。

量子化学計算（XTB, COSMO-RS）および記述子計算（RDKit, Mordred）で
正しい電荷設定を使えるように、ユーザーが分子の電荷状態を指定できる UI を提供する。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def render_charge_config_panel(smiles_col: str, df: pd.DataFrame) -> None:
    """
    SMILES列の化合物に対する電荷・スピン・プロトン化設定パネルを描画する。

    セッションステートのキー:
        smiles_charge_configs : ChargeConfigStore
            電荷設定ストア。設定変更時に更新される。

    Parameters
    ----------
    smiles_col : str
        SMILES が格納されている列名
    df : pd.DataFrame
        データフレーム（ユニーク分子の抽出に使用）
    """
    import streamlit as st
    from backend.chem.charge_config import ChargeConfigStore, MoleculeChargeConfig

    # セッションステートの初期化
    if "smiles_charge_configs" not in st.session_state:
        st.session_state["smiles_charge_configs"] = ChargeConfigStore()

    store: ChargeConfigStore = st.session_state["smiles_charge_configs"]

    st.markdown("### ⚡ 分子電荷・スピン設定")
    st.caption(
        "量子化学計算（XTB）やRDKit記述子計算で正しい電荷・スピン状態を使用するために設定します。"
        "荷電分子（アンモニウム塩・カルボン酸イオン等）ではHOMO/LUMOエネルギーに本質的な影響があります。"
    )

    # ── グローバル設定 ────────────────────────────────────────────────────
    with st.expander("🌐 グローバル設定（全分子共通）", expanded=True):
        _render_global_settings(store)

    # ── 分子ごとの個別設定 ──────────────────────────────────────────────
    with st.expander("🔬 分子ごとの個別設定（上級）", expanded=False):
        _render_per_molecule_settings(store, smiles_col, df)

    # ── プレビュー ──────────────────────────────────────────────────────
    with st.expander("👁️ 設定プレビュー", expanded=False):
        _render_preview(store, smiles_col, df)

    # 変更を保存
    st.session_state["smiles_charge_configs"] = store


def _render_global_settings(store) -> None:
    """グローバル電荷設定の UI を描画する。"""
    import streamlit as st
    from backend.chem.charge_config import MoleculeChargeConfig

    cfg = store.default
    st.markdown("**全ての分子に適用されるデフォルト設定（個別設定で上書き可能）**")

    col1, col2 = st.columns(2)

    with col1:
        # 形式電荷
        st.markdown("**形式電荷 (Formal Charge)**")
        auto_charge = st.checkbox(
            "SMILESから自動読取（推奨）",
            value=cfg.auto_charge_from_smiles,
            key="gc_auto_charge",
            help="[NH4+] → +1、[COO-] → -1 を SMILES から自動検出します（RDKit GetFormalCharge）",
        )
        cfg.auto_charge_from_smiles = auto_charge

        if not auto_charge:
            formal_charge = st.slider(
                "形式電荷 (e)",
                min_value=-5,
                max_value=5,
                value=cfg.formal_charge,
                step=1,
                key="gc_formal_charge",
                help="分子全体の電荷。XTBに --chrg として渡されます。",
            )
            cfg.formal_charge = formal_charge
        else:
            st.info("💡 各SMILESのRDKit形式電荷を自動読取してXTBに渡します")

        # スピン多重度
        st.markdown("**スピン多重度 (Spin Multiplicity)**")
        spin_options = {
            "1 — 閉殻（通常の有機分子）": 1,
            "2 — ラジカル（一重ラジカル, TEMPO等）": 2,
            "3 — 三重項（カルベン, O₂等）": 3,
            "4 — 四重項（稀）": 4,
        }
        spin_key = next(
            (k for k, v in spin_options.items() if v == cfg.spin_multiplicity),
            "1 — 閉殻（通常の有機分子）"
        )
        spin_sel = st.selectbox(
            "スピン多重度 (2S+1)",
            options=list(spin_options.keys()),
            index=list(spin_options.keys()).index(spin_key),
            key="gc_spin",
            help=(
                "1=閉殻（デフォルト）, 2=一重ラジカル（不対電子1つ）, "
                "3=三重項（不対電子2つ）。XTBに --uhf (M-1) として渡されます。"
            ),
        )
        cfg.spin_multiplicity = spin_options[spin_sel]

        if cfg.spin_multiplicity > 1:
            st.warning(
                f"⚠️ スピン多重度 {cfg.spin_multiplicity} を設定しています。"
                f"XTBに `--uhf {cfg.uhf}` が渡されます。"
                f"閉殻分子に対してラジカル計算を行うと誤った結果になります。"
            )

    with col2:
        # プロトン化モード
        st.markdown("**プロトン化モード**")
        protonate_options = {
            "🔵 as_is — SMILESのまま使用": "as_is",
            "🟢 neutral — 全て中性化（塩を除去）": "neutral",
            "🟡 auto_ph — pHで自動プロトン化（UniPKa使用）": "auto_ph",
            "🔴 max_acid — 最大脱プロトン化形": "max_acid",
            "🟣 max_base — 最大プロトン化形": "max_base",
        }
        mode_key = next(
            (k for k, v in protonate_options.items() if v == cfg.protonate_mode),
            "🔵 as_is — SMILESのまま使用"
        )
        mode_sel = st.selectbox(
            "プロトン化の方針",
            options=list(protonate_options.keys()),
            index=list(protonate_options.keys()).index(mode_key),
            key="gc_protonate_mode",
            help=(
                "as_is: 変換なし（最速）\n"
                "neutral: RDKit中性化（塩・対イオン除去）\n"
                "auto_ph: UniPKaのpKa予測 + Henderson-Hasselbalch式で電荷を決定\n"
                "max_acid: 全酸性基を脱プロトン化した形\n"
                "max_base: 全塩基性基をプロトン化した形"
            ),
        )
        cfg.protonate_mode = protonate_options[mode_sel]

        # pH設定（auto_phモード時のみ）
        if cfg.protonate_mode == "auto_ph":
            ph_presets = {
                "pH 7.4（生理的条件）": 7.4,
                "pH 7.0（中性）": 7.0,
                "pH 2.0（強酸性）": 2.0,
                "pH 5.5（弱酸性）": 5.5,
                "pH 9.0（弱塩基性）": 9.0,
                "pH 12.0（強塩基性）": 12.0,
                "カスタム": None,
            }
            ph_preset_sel = st.selectbox(
                "pH プリセット",
                options=list(ph_presets.keys()),
                key="gc_ph_preset",
            )
            if ph_presets[ph_preset_sel] is None:
                ph_val = st.number_input(
                    "pH値",
                    min_value=0.0,
                    max_value=14.0,
                    value=float(cfg.ph) if cfg.ph is not None else 7.4,
                    step=0.1,
                    key="gc_ph_custom",
                )
            else:
                ph_val = ph_presets[ph_preset_sel]
            cfg.ph = ph_val
            st.info(
                f"💡 pH {ph_val:.1f} での各分子のイオン化状態をUniPKaで予測します。"
                f"UniPKaが利用不可の場合は中性化にフォールバックします。"
            )

        # 部分電荷モデル
        st.markdown("**部分電荷モデル**")
        pcharge_options = {
            "gasteiger — Gasteiger-Marsili（高速・RDKit内蔵）": "gasteiger",
            "xtb_mulliken — GFN2-xTB Mulliken（精密・XTB必要）": "xtb_mulliken",
            "none — 部分電荷記述子を使用しない": "none",
        }
        pc_key = next(
            (k for k, v in pcharge_options.items() if v == cfg.partial_charge_model),
            "gasteiger — Gasteiger-Marsili（高速・RDKit内蔵）"
        )
        pc_sel = st.selectbox(
            "部分電荷モデル",
            options=list(pcharge_options.keys()),
            index=list(pcharge_options.keys()).index(pc_key),
            key="gc_partial_charge",
            help=(
                "Gasteiger: q_max, q_min, q_range, q_std, q_abs_mean を記述子として追加\n"
                "XTB Mulliken: xtb_MullikenChargeMax等を記述子として追加（XTBが必要）\n"
                "none: 部分電荷記述子なし"
            ),
        )
        cfg.partial_charge_model = pcharge_options[pc_sel]

        # 互変異性体
        cfg.consider_tautomers = st.checkbox(
            "互変異性体の最安定形を使用（RDKit MolStandardize）",
            value=cfg.consider_tautomers,
            key="gc_tautomers",
            help=(
                "ケト-エノール、アミド-イミドール等の互変異性体の中で"
                "最安定な形を自動選択します。計算コストが若干増加します。"
            ),
        )

    store.default = cfg


def _render_per_molecule_settings(store, smiles_col: str, df: pd.DataFrame) -> None:
    """分子ごとの個別電荷設定テーブルを描画する。"""
    import streamlit as st
    from backend.chem.charge_config import MoleculeChargeConfig

    if smiles_col not in df.columns:
        st.warning("SMILES列が見つかりません。")
        return

    smiles_series = df[smiles_col].dropna().unique()
    # 多すぎる場合は上位50件に絞る
    MAX_SHOW = 50
    if len(smiles_series) > MAX_SHOW:
        st.info(f"⚠️ ユニーク分子が {len(smiles_series)} 件あるため、上位 {MAX_SHOW} 件を表示します。")
        smiles_series = smiles_series[:MAX_SHOW]

    st.caption(
        "個別設定を行う分子のSMILESを選択してください。"
        "グローバル設定からの差分のみ表示されます。"
    )

    # 分子選択
    selected_smi = st.selectbox(
        "設定する分子のSMILES",
        options=["（選択してください）"] + list(smiles_series),
        key="per_mol_select",
    )

    if selected_smi == "（選択してください）":
        return

    # 現在の個別設定を取得（なければデフォルトをコピー）
    if selected_smi in store.per_molecule:
        cur_cfg = store.per_molecule[selected_smi]
    else:
        cur_cfg = MoleculeChargeConfig(
            formal_charge=store.default.formal_charge,
            spin_multiplicity=store.default.spin_multiplicity,
            ph=store.default.ph,
            protonate_mode=store.default.protonate_mode,
            partial_charge_model=store.default.partial_charge_model,
            auto_charge_from_smiles=store.default.auto_charge_from_smiles,
        )

    # 分子のSMIES情報と現在設定を表示
    try:
        from backend.chem.charge_config import _read_smiles_formal_charge
        auto_charge = _read_smiles_formal_charge(selected_smi)
        st.markdown(f"**選択中の分子:** `{selected_smi}`")
        st.markdown(f"**SMILES読取り電荷:** `{auto_charge:+d}e`")
    except Exception:
        pass

    col_a, col_b = st.columns(2)
    with col_a:
        override_auto = st.checkbox(
            "SMILESから自動電荷読取",
            value=cur_cfg.auto_charge_from_smiles,
            key=f"pm_auto_{selected_smi[:10]}",
        )
        cur_cfg.auto_charge_from_smiles = override_auto

        if not override_auto:
            cur_cfg.formal_charge = st.slider(
                "形式電荷 (e)",
                min_value=-5, max_value=5,
                value=cur_cfg.formal_charge,
                key=f"pm_charge_{selected_smi[:10]}",
            )

        spin_map = {1: "1 (閉殻)", 2: "2 (ラジカル)", 3: "3 (三重項)", 4: "4 (四重項)"}
        spin_sel = st.selectbox(
            "スピン多重度",
            options=list(spin_map.values()),
            index=cur_cfg.spin_multiplicity - 1,
            key=f"pm_spin_{selected_smi[:10]}",
        )
        cur_cfg.spin_multiplicity = int(spin_sel.split()[0])

    with col_b:
        pmode_map = {
            "as_is": "SMILESのまま",
            "neutral": "中性化",
            "auto_ph": "pH自動調整",
            "max_acid": "最大脱プロトン",
            "max_base": "最大プロトン化",
        }
        pm_sel = st.selectbox(
            "プロトン化モード",
            options=list(pmode_map.values()),
            index=list(pmode_map.keys()).index(cur_cfg.protonate_mode),
            key=f"pm_mode_{selected_smi[:10]}",
        )
        cur_cfg.protonate_mode = next(k for k, v in pmode_map.items() if v == pm_sel)

        if cur_cfg.protonate_mode == "auto_ph":
            cur_cfg.ph = st.number_input(
                "pH",
                min_value=0.0, max_value=14.0,
                value=float(cur_cfg.ph) if cur_cfg.ph else 7.4,
                step=0.1,
                key=f"pm_ph_{selected_smi[:10]}",
            )

    col_save, col_del = st.columns(2)
    with col_save:
        if st.button("💾 この分子に保存", key=f"pm_save_{selected_smi[:10]}"):
            store.set_per_molecule(selected_smi, cur_cfg)
            st.success(f"✅ 保存しました: `{selected_smi[:30]}`")

    with col_del:
        if selected_smi in store.per_molecule:
            if st.button("🗑️ 個別設定を削除", key=f"pm_del_{selected_smi[:10]}"):
                del store.per_molecule[selected_smi]
                st.info("個別設定を削除しました。グローバル設定が適用されます。")

    # 現在の個別設定一覧
    if store.per_molecule:
        st.markdown("---")
        st.markdown("**📋 登録済みの個別設定**")
        rows = []
        for smi, pcfg in store.per_molecule.items():
            rows.append({
                "SMILES": smi[:40] + "..." if len(smi) > 40 else smi,
                "電荷": f"{pcfg.formal_charge:+d}" if not pcfg.auto_charge_from_smiles else "AUTO",
                "スピン": pcfg.spin_multiplicity,
                "プロトン化": pcfg.protonate_mode,
                "pH": pcfg.ph or "-",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _render_preview(store, smiles_col: str, df: pd.DataFrame) -> None:
    """設定後のプロトン化状態と Gasteiger 電荷をプレビュー表示する。"""
    import streamlit as st
    from backend.chem.protonation import apply_protonation, get_protonation_state_info

    if smiles_col not in df.columns:
        st.warning("SMILES列が見つかりません。")
        return

    smiles_list = df[smiles_col].dropna().unique()[:10]  # 先頭10件のみプレビュー

    st.markdown("**先頭10件の分子について設定を適用した結果をプレビューします。**")

    preview_rows = []
    for smi in smiles_list:
        cfg = store.get_config(smi)
        converted = apply_protonation(smi, cfg)
        changed = "✅" if converted != smi else "—"

        # 形式電荷を解決
        charge = store.resolve_charge(smi)

        row = {
            "元のSMILES": smi[:35] + "..." if len(smi) > 35 else smi,
            "変換後SMILES": converted[:35] + "..." if len(converted) > 35 else converted,
            "変換": changed,
            "形式電荷": f"{charge:+d}e",
            "スピン多重度": cfg.spin_multiplicity,
            "プロトン化モード": cfg.protonate_mode,
        }
        preview_rows.append(row)

    if preview_rows:
        st.dataframe(pd.DataFrame(preview_rows), hide_index=True, use_container_width=True)

    # pH設定時の pKa 情報
    if store.default.protonate_mode == "auto_ph" and store.default.ph is not None:
        try:
            from unipka import UnipKa  # noqa: F401
            st.markdown("---")
            st.markdown(f"**📊 pH {store.default.ph:.1f} でのイオン化状態予測（先頭5件）**")
            for smi in list(smiles_list)[:5]:
                with st.spinner(f"pKa計算中: {smi[:25]}..."):
                    info = get_protonation_state_info(smi, ph=store.default.ph)
                form_icons = {
                    "neutral": "⚪",
                    "anion": "🔵",
                    "cation": "🔴",
                    "zwitterion": "🟣",
                    "unknown": "⬜",
                }
                icon = form_icons.get(info["dominant_form_at_ph"], "⬜")
                st.markdown(
                    f"{icon} `{smi[:30]}` — {info['ionization_note']}"
                )
        except ImportError:
            st.caption("⚠️ UniPKaが未インストールのため pKa プレビューは利用できません。")

    # Gasteiger 電荷の色付き可視化（RDkit SimilarityMapを利用）
    if store.default.partial_charge_model == "gasteiger":
        _render_gasteiger_visualization(store, smiles_col, df)


def _render_gasteiger_visualization(store, smiles_col: str, df: pd.DataFrame) -> None:
    """Gasteiger 部分電荷を色付き原子マップで可視化する。"""
    import streamlit as st

    st.markdown("---")
    st.markdown("**🎨 Gasteiger 部分電荷マップ（赤=負電荷, 青=正電荷）**")
    st.caption(
        "各原子の Gasteiger 電荷を色で可視化します。"
        "電子豊富な原子（求核点）が赤く、電子不足な原子（求電子点）が青く表示されます。"
    )

    smiles_list = df[smiles_col].dropna().unique()[:6]  # 先頭6件

    try:
        from rdkit import Chem
        from rdkit.Chem import rdPartialCharges, Draw
        from rdkit.Chem.Draw import SimilarityMaps
        import io
        import base64
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cols = st.columns(min(3, len(smiles_list)))
        for i, smi in enumerate(smiles_list):
            cfg = store.get_config(smi)
            from backend.chem.protonation import apply_protonation
            smi_converted = apply_protonation(smi, cfg)

            mol = Chem.MolFromSmiles(smi_converted) or Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            try:
                mol_h = Chem.AddHs(mol)
                rdPartialCharges.ComputeGasteigerCharges(mol_h)
                charges = {
                    i: float(mol_h.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge'))
                    for i in range(mol.GetNumAtoms())  # 重水素なしの原子数で
                }

                fig, ax = plt.subplots(figsize=(3, 2.5), facecolor="#0e1117")
                ax.set_facecolor("#0e1117")

                SimilarityMaps.GetSimilarityMapFromWeights(
                    mol,
                    list(charges.values()),
                    colorMap="RdBu_r",
                    contourLines=0,
                    size=(280, 220),
                    ax=ax,
                )
                ax.set_title(
                    smi[:20] + "..." if len(smi) > 20 else smi,
                    fontsize=8, color="white", pad=2
                )

                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=80, bbox_inches="tight",
                            facecolor="#0e1117")
                plt.close(fig)
                buf.seek(0)

                with cols[i % 3]:
                    st.image(buf, use_container_width=True)

            except Exception as e_mol:
                logger.debug("Gasteiger可視化失敗 (%s): %s", smi[:20], e_mol)

    except ImportError as e_imp:
        st.caption(f"可視化に必要なライブラリが利用できません: {e_imp}")
    except Exception as e_gen:
        st.caption(f"可視化エラー: {e_gen}")
