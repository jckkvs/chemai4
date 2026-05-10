# -*- coding: utf-8 -*-
"""
プリセット記述子の厳選 + 「2/2」バッジ改善 + データ読込リセットバグ修正
"""

# ====== 1. プリセットを厳選記述子リスト化 ======
fp1 = 'C:/Users/horie/chemai2/frontend_nicegui/components/descriptor_plugins_ui.py'
with open(fp1, 'r', encoding='utf-8') as f:
    content1 = f.read()

changes1 = 0

# プリセット定義を厳選記述子リスト付きに置換
old_presets = '''_PRESETS = {
    "\U0001f9ea 基本物性（沸点・密度等）": {
        "engines": ["RDKitAdapter", "GroupContribAdapter"],
        "desc": "MW, LogP, TPSA, 基団寄与法を中心に物性予測",
    },
    "\U0001f511 構造活性相関（FP中心）": {
        "engines": ["RDKitAdapter", "SkfpAdapter"],
        "desc": "ECFP/MACCS等のフィンガープリントで活性予測・QSAR",
    },
    "\U0001f4d0 網羅的記述子（特徴量選択前提）": {
        "engines": ["RDKitAdapter", "MordredAdapter"],
        "desc": "Mordred 1800+記述子を全計算→特徴量選択で絞り込む",
    },
    "\U0001f9e0 深層学習表現": {
        "engines": ["MolAIAdapter", "Mol2VecAdapter"],
        "desc": "CNN潜在ベクトル+Word2Vec分散表現",
    },
    "⚛️ 量子化学込み": {
        "engines": ["RDKitAdapter", "XTBAdapter", "CosmoAdapter"],
        "desc": "HOMO/LUMO, 溶媒和エネルギー等を加えた高精度モデル",
    },
    "\U0001f680 フルセット（全エンジン）": {
        "engines": [e["cls"] for e in _ENGINE_INFO],
        "desc": "利用可能な全エンジンを一括ON（時間がかかります）",
    },
}'''

new_presets = '''_PRESETS = {
    "\U0001f9ea 基本物性（沸点・密度等）": {
        "engines": ["RDKitAdapter", "GroupContribAdapter"],
        "desc": "MW, LogP, TPSA等の主要物性15記述子を厳選",
        "descriptors": [
            # 分子サイズ
            "MolWt", "HeavyAtomCount", "LabuteASA",
            # 極性・溶解性
            "MolLogP", "TPSA", "MolMR",
            # 水素結合
            "NumHAcceptors", "NumHDonors",
            # トポロジー
            "NumRotatableBonds", "RingCount", "NumAromaticRings",
            "FractionCSP3",
            # 基団寄与法（物性推算）
            "joback_Tb", "joback_Tc", "joback_Hf",
        ],
    },
    "\U0001f511 構造活性相関（FP中心）": {
        "engines": ["RDKitAdapter", "SkfpAdapter"],
        "desc": "ECFP + 主要物性20記述子で活性予測・QSAR",
        "descriptors": [
            # 基本物性（少数）
            "MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors",
            "NumRotatableBonds", "RingCount", "FractionCSP3",
            # 電子状態
            "MaxPartialCharge", "MinPartialCharge",
            # トポロジー
            "BertzCT", "HallKierAlpha", "Kappa1", "Kappa2",
            "Chi1n", "Chi1v",
            # EState
            "MaxAbsEStateIndex", "MinAbsEStateIndex",
            # QED
            "qed",
        ],
        "include_fp_prefix": ["Morgan_r2_"],  # Morganフィンガープリントも含む
    },
    "\U0001f4d0 網羅的記述子（特徴量選択前提）": {
        "engines": ["RDKitAdapter", "MordredAdapter"],
        "desc": "RDKit+Mordred全記述子→特徴量選択で自動絞り込み",
        # descriptors未指定 → エンジンの全記述子を使用
    },
    "\U0001f9e0 深層学習表現": {
        "engines": ["MolAIAdapter", "Mol2VecAdapter"],
        "desc": "CNN潜在ベクトル+Word2Vec分散表現",
    },
    "⚛️ 量子化学込み": {
        "engines": ["RDKitAdapter", "XTBAdapter", "CosmoAdapter"],
        "desc": "HOMO/LUMO, 溶媒和エネルギー等を加えた高精度モデル",
        "descriptors": [
            # 基本物性
            "MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors",
            "NumRotatableBonds", "RingCount",
            # 量子化学
            "HomoEnergy", "LumoEnergy", "HomoLumoGap",
            "DipoleMoment", "Polarizability",
            # xTB
            "xtb_total_energy", "xtb_homo", "xtb_lumo", "xtb_gap",
            "xtb_dipole", "xtb_polarizability",
        ],
    },
    "\U0001f680 フルセット（全エンジン）": {
        "engines": [e["cls"] for e in _ENGINE_INFO],
        "desc": "利用可能な全エンジンを一括ON（時間がかかります）",
    },
}'''

if old_presets in content1:
    content1 = content1.replace(old_presets, new_presets, 1)
    changes1 += 1
else:
    print("WARNING: old_presets not found, trying line-by-line search")
    # エスケープ問題の可能性あり、別のアプローチ
    idx = content1.find('_PRESETS = {')
    if idx >= 0:
        end_idx = content1.find('\n}\n', idx)
        if end_idx >= 0:
            print(f"Found _PRESETS at pos {idx} to {end_idx+2}")
        else:
            print("Could not find end of _PRESETS")
    else:
        print("_PRESETS not found at all")

# 2. プリセット適用ロジック修正: descriptorsキーがある場合はそれを使用
# _apply_preset関数の内部でdescriptors指定を優先
old_apply = '''                    def _apply_preset(engines=preset_engines, pname=preset_name):
                        # 全エンジンOFF → 選択エンジンのみON
                        for eng in _ENGINE_INFO:
                            key = f"use_{eng['cls'].replace('Adapter', '').lower()}"
                            state[key] = eng["cls"] in engines
                        # 選択エンジンに属する記述子のみを選択
                        engine_cls_set = set(engines)'''

new_apply = '''                    def _apply_preset(engines=preset_engines, pname=preset_name, pinfo=preset_info):
                        # 全エンジンOFF → 選択エンジンのみON
                        for eng in _ENGINE_INFO:
                            key = f"use_{eng['cls'].replace('Adapter', '').lower()}"
                            state[key] = eng["cls"] in engines

                        # 厳選記述子リストがある場合はそれを優先
                        curated = pinfo.get("descriptors")
                        fp_prefixes = pinfo.get("include_fp_prefix", [])
                        if curated:
                            selected = [d for d in all_descs if d in curated]
                            # FPプレフィックス指定があればそれも追加
                            for pfx in fp_prefixes:
                                selected += [d for d in all_descs if d.startswith(pfx)]
                            state["selected_descriptors"] = selected
                            ui.notify(
                                f"{pname}: {len(selected)}個の厳選記述子を適用",
                                type="positive",
                            )
                            return

                        # 選択エンジンに属する記述子のみを選択
                        engine_cls_set = set(engines)'''

if old_apply in content1:
    content1 = content1.replace(old_apply, new_apply, 1)
    changes1 += 1
else:
    print("WARNING: old_apply not found")

# 3. 「2/2」バッジを改善: descriptors指定がある場合は「N個厳選」表示
old_badge = '''                    n_avail = sum(
                        1 for e in preset_engines if _is_available(adapters, e)
                    )'''

new_badge = '''                    n_avail = sum(
                        1 for e in preset_engines if _is_available(adapters, e)
                    )
                    # 厳選記述子数（表示用）
                    n_curated = len(preset_info.get("descriptors", []))'''

if old_badge in content1:
    content1 = content1.replace(old_badge, new_badge, 1)
    changes1 += 1
else:
    print("WARNING: old_badge not found")

with open(fp1, 'w', encoding='utf-8') as f:
    f.write(content1)

print(f"descriptor_plugins_ui.py: {changes1} changes")

# ====== 2. データ読込タブのリセットバグ修正 ======
fp2 = 'C:/Users/horie/chemai2/frontend_nicegui/components/data_tab.py'
with open(fp2, 'r', encoding='utf-8') as f:
    content2 = f.read()

changes2 = 0

# _render_data_loadの先頭でstateにdfが存在する場合はプレビューを即座に表示
old_render = '''def _render_data_load(state: dict) -> None:
    """ファイルアップロード + サンプル + ベンチマークのデータ読込UI"""

    upload_status = ui.label("").classes("text-grey-5 q-mt-sm")
    preview_container = ui.column().classes("full-width q-mt-md")'''

new_render = '''def _render_data_load(state: dict) -> None:
    """ファイルアップロード + サンプル + ベンチマークのデータ読込UI"""

    # データ読み込み済みの場合はステータスを復元
    df_existing = state.get("df")
    fn_existing = state.get("filename", "")
    if df_existing is not None and not df_existing.empty:
        status_text = f"\\u2705 {fn_existing} ({len(df_existing)}行 \\u00d7 {len(df_existing.columns)}列)"
        upload_status = ui.label(status_text).classes("text-green q-mt-sm")
    else:
        upload_status = ui.label("").classes("text-grey-5 q-mt-sm")
    preview_container = ui.column().classes("full-width q-mt-md")

    # 既存データがある場合はプレビューを即座に表示（タブ切替でリセットされない）
    if df_existing is not None and not df_existing.empty:
        _show_preview(df_existing, preview_container)'''

if old_render in content2:
    content2 = content2.replace(old_render, new_render, 1)
    changes2 += 1
else:
    print("WARNING: old_render not found")

with open(fp2, 'w', encoding='utf-8') as f:
    f.write(content2)

print(f"data_tab.py: {changes2} changes")
