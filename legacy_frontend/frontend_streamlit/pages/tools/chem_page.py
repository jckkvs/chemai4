"""frontend_streamlit/pages/chem_page.py - 化合物解析ページ"""
from __future__ import annotations
import streamlit as st
import pandas as pd


def render() -> None:
    st.markdown("## 🧬 化合物解析 (SMILES)")
    st.caption(
        "💡 このページは個別の化合物をインタラクティブに調べるための探索ツールです。"
        "メインの解析フローで使う記述子の選択は「⚗️ SMILES特徴量設計」タブで行えます。"
    )

    from backend.chem.rdkit_adapter import RDKitAdapter
    rdkit_ok = RDKitAdapter().is_available()

    if not rdkit_ok:
        st.error("❌ RDKitがインストールされていません。`conda install -c conda-forge rdkit` を実行してください。")

    st.markdown("""
<div class="card">
<h4>🧬 化合物記述子計算</h4>
<p style="color:#b0afd0;">SMILESデータから分子記述子・フィンガープリントを計算します。個別のSMILESを入力して結果を確認できます。</p>
</div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📂 データから計算", "✏️ 単一SMILES解析"])

    with tab1:
        df = st.session_state.get("df")
        if df is None:
            st.warning("⚠️ まずデータを読み込んでください。")
            return

        smiles_col = st.selectbox("SMILES列を選択",
            [c for c in df.columns if df[c].dtype == object] or df.columns.tolist())

        col1, col2 = st.columns(2)
        with col1:
            compute_fp = st.checkbox("フィンガープリントも計算", value=False)
            include_maccs = st.checkbox("MACCS Keys", value=False)
        with col2:
            morgan_radius = st.slider("Morgan半径 (ECFP radius)", 1, 4, 2)

        if st.button("🔬 記述子を計算", use_container_width=True, disabled=not rdkit_ok):
            from backend.chem.rdkit_adapter import RDKitAdapter
            adapter = RDKitAdapter(compute_fp=compute_fp, morgan_radius=morgan_radius,
                                   include_maccs=include_maccs)
            with st.spinner("記述子計算中..."):
                try:
                    smiles_list = df[smiles_col].fillna("").tolist()
                    desc_result = adapter.compute(smiles_list)
                    st.session_state["desc_result"] = desc_result

                    st.success(f"✅ {len(smiles_list)}分子 / 失敗={len(desc_result.failed_indices)} / "
                               f"成功率={desc_result.success_rate:.1%} / 記述子数={desc_result.n_descriptors}")
                    st.dataframe(desc_result.descriptors.head(10), use_container_width=True)

                    if st.download_button(
                        "📥 記述子をCSVでダウンロード",
                        desc_result.descriptors.to_csv(index=False).encode("utf-8"),
                        file_name="descriptors.csv", mime="text/csv",
                    ):
                        pass
                except Exception as e:
                    st.error(f"計算エラー: {e}")

    with tab2:
        smi = st.text_input("SMILESを入力", placeholder="例: CCO, c1ccccc1, CC(=O)O")
        if smi and rdkit_ok:
            from backend.chem.rdkit_adapter import RDKitAdapter
            adapter = RDKitAdapter(compute_fp=False)
            try:
                res = adapter.compute([smi])
                if res.failed_indices:
                    st.error(f"❌ 無効なSMILES: {smi}")
                else:
                    st.success("✅ 物理化学的性質")
                    st.dataframe(res.descriptors.T.rename(columns={0: "値"}), use_container_width=True)
                    try:
                        from rdkit import Chem
                        from rdkit.Chem import Draw
                        import io
                        mol = Chem.MolFromSmiles(smi)
                        if mol:
                            img = Draw.MolToImage(mol, size=(300, 200))
                            buf = io.BytesIO()
                            img.save(buf, format="PNG")
                            st.image(buf.getvalue(), caption=f"分子構造: {smi}", width=300)
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"エラー: {e}")
