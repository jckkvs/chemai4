#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Make all necessary edits to descriptor_plugins_ui.py"""

filepath = r'C:\Users\horie\chemai2_qwen\frontend_nicegui\components\descriptor_plugins_ui.py'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# ============================================================
# Edit 1: Update docstring to "8つのタブ"
# ============================================================
old1 = '''def _render_target_recommendations(state: dict, adapters: dict) -> None:
    """
    記述子のの選択方法を6つのタブで提供する。
    1. 🎯 プリセットで選ぶ — 目的別ワンクリックプリセット
    2. 🔬 目的変数で選ぶ — recommender.pyの推奨DBから選択
    3. 📈 相関係数で選ぶ — 目的変数との相関上位を選択
    4. ⚙️ エンジンから選ぶ — ライブラリ単位で選択
    5. 🔍 テキスト検索 — 記述子名の部分一致検索
    6. 📊 分散ベース — 情報量が大きい記述子を選択
    """'''

new1 = '''def _render_target_recommendations(state: dict, adapters: dict) -> None:
    """
    記述子のの選択方法を8つのタブで提供する。
    1. 🎯 プリセットで選ぶ — 目的別ワンクリックプリセット
    2. 🔬 目的変数で選ぶ — recommender.pyの推奨DBから選択
    3. 📈 相関係数で選ぶ — 目的変数との相関上位を選択
    4. 🔢 数えあげ系 — カウント系記述子（Num*, RingCount等）
    5. 🤖 LLMによる推奨 — LLMを活用した記述子推奨
    6. ⚙️ エンジンから選ぶ — ライブラリ単位で選択
    7. 🔍 テキスト検索 — 記述子名の部分一致検索
    8. 📊 分散ベース — 情報量が大きい記述子を選択
    """'''

if old1 in content:
    content = content.replace(old1, new1)
    print("Edit 1: Updated docstring to 8つのタブ - SUCCESS")
else:
    print("Edit 1: FAILED - string not found")

# ============================================================
# Edit 2: Add new tabs (tab_count, tab_llm)
# ============================================================
old2 = '''        # ── 6タブ ──
        with ui.tabs().classes("full-width").props(
            "dense no-caps active-color=cyan indicator-color=cyan"
        ) as tabs:
            tab_preset = ui.tab("preset", label="🎯 プリセット", icon="auto_awesome")
            tab_target = ui.tab("target", label="🔬 目的変数", icon="science")
            tab_corr = ui.tab("corr", label="📈 相関係数", icon="trending_up")
            tab_engine = ui.tab("engine", label="⚙️ エンジン", icon="settings")
            tab_search = ui.tab("search", label="🔍 検索", icon="search")
            tab_variance = ui.tab("variance", label="📊 分散", icon="bar_chart")'''

new2 = '''        # ── 8タブ ──
        with ui.tabs().classes("full-width").props(
            "dense no-caps active-color=cyan indicator-color=cyan"
        ) as tabs:
            tab_preset = ui.tab("preset", label="🎯 プリセット", icon="auto_awesome")
            tab_target = ui.tab("target", label="🔬 目的変数", icon="science")
            tab_corr = ui.tab("corr", label="📈 相関係数", icon="trending_up")
            tab_count = ui.tab("count", label="🔢 数えあげ系", icon="calculate")
            tab_llm = ui.tab("llm", label="🤖 LLMによる推奨", icon="smart_toy")
            tab_engine = ui.tab("engine", label="⚙️ エンジン", icon="settings")
            tab_search = ui.tab("search", label="🔍 検索", icon="search")
            tab_variance = ui.tab("variance", label="📊 分散", icon="bar_chart")'''

if old2 in content:
    content = content.replace(old2, new2)
    print("Edit 2: Added new tabs (tab_count, tab_llm) - SUCCESS")
else:
    print("Edit 2: FAILED - string not found")

# ============================================================
# Edit 3: Update "目的変数" tab panel to show all descriptors with selection UI
# ============================================================
old3 = '''                                # ── プロパティをチェックボックスとして表示 ──
                                _valid_dn = [d.name for d in rec.descriptors if d.name in all_descs]
                                _n_valid = len(_valid_dn)
                                _n_total_rec = len(rec.descriptors)
                                _n_sel = sum(1 for dn in _valid_dn if dn in state.get("selected_descriptors", []))

                                # 全て選択されているかを判定
                                _is_all_on = (_n_sel == _n_valid and _n_valid > 0)

                                def _toggle_rec(val, ds=_valid_dn, r=rec):
                                    s = set(state.get("selected_descriptors", []))
                                    if val:
                                        s.update(ds)
                                        ui.notify(f"{r.target_name}: {len(ds)}個の記述子を追加", type="positive")
                                    else:
                                        s -= set(ds)
                                    state["selected_descriptors"] = list(s)

                                with ui.row().classes("full-width items-center q-gutter-x-sm q-py-xs"):
                                    ui.checkbox(
                                        rec.target_name,
                                        value=_is_all_on,
                                        on_change=lambda e, d=_valid_dn, r=rec: _toggle_rec(e.value, d, r)
                                    )
                                    ui.badge(
                                        f"計算済 {_n_valid} / 推奨 {_n_total_rec} 個",
                                        color="teal" if _n_valid == _n_total_rec else "amber",
                                    ).props("outline")'''

new3 = '''                                # ── 推奨記述子の表示 ──
                                _valid_descs = [d for d in rec.descriptors if d.name in all_descs]
                                _n_valid = len(_valid_descs)
                                _n_total_rec = len(rec.descriptors)
                                _valid_dn = [d.name for d in _valid_descs]
                                _n_sel = sum(1 for dn in _valid_dn if dn in state.get("selected_descriptors", []))

                                # 全て選択されているかを判定
                                _is_all_on = (_n_sel == _n_valid and _n_valid > 0)

                                with ui.card().classes("full-width q-pa-xs q-ma-xs").style(
                                    "background: rgba(255,255,255,0.03);"
                                ):
                                    # ヘッダー行: チェックボックス + 名前 + バッジ
                                    with ui.row().classes("full-width items-center q-gutter-x-sm"):
                                        ui.checkbox(
                                            rec.target_name,
                                            value=_is_all_on,
                                            on_change=lambda e, d=_valid_dn, r=rec: _toggle_rec(e.value, d, r)
                                        ).props("dense")

                                        ui.badge(
                                            f"計算済 {_n_valid} / 推奨 {_n_total_rec} 個",
                                            color="teal" if _n_valid == _n_total_rec else "amber",
                                        ).props("outline dense")

                                        if rec.category:
                                            ui.badge(rec.category, color="blue").props("outline dense")

                                    # サマリー
                                    if rec.summary:
                                        ui.label(rec.summary).classes("text-caption text-grey q-ml-lg")

                                    # 全選択/全解除ボタン
                                    if _valid_descs:
                                        with ui.row().classes("q-gutter-xs q-ml-lg q-mt-xs"):
                                            def _select_all_rec(val, names=_valid_dn, r=rec):
                                                s = set(state.get("selected_descriptors", []))
                                                if val:
                                                    s.update(names)
                                                    state["selected_descriptors"] = list(s)
                                                    ui.notify(f"{r.target_name}: {len(names)}件を選択", type="positive")
                                                else:
                                                    s -= set(names)
                                                    state["selected_descriptors"] = list(s)
                                                    ui.notify(f"{r.target_name}: 全解除しました", type="info")

                                            ui.button(
                                                "全選択",
                                                on_click=lambda e, n=_valid_dn, r=rec: _select_all_rec(True, n, r)
                                            ).props("outline size=xs no-caps color=blue")
                                            ui.button(
                                                "全解除",
                                                on_click=lambda e, n=_valid_dn, r=rec: _select_all_rec(False, n, r)
                                            ).props("outline size=xs no-caps color=red")

                                    # 個別記述子のチェックボックスリスト
                                    if _valid_descs:
                                        ui.separator().classes("q-my-xs")
                                        ui.label("個別選択:").classes("text-caption text-grey q-ml-lg")

                                        for desc in _valid_descs:
                                            def _toggle_individual(val, dn=desc.name):
                                                s = set(state.get("selected_descriptors", []))
                                                if val:
                                                    s.add(dn)
                                                else:
                                                    s.discard(dn)
                                                state["selected_descriptors"] = list(s)

                                            _is_sel = desc.name in state.get("selected_descriptors", [])
                                            with ui.row().classes(
                                                "items-center q-gutter-xs q-py-xs"
                                            ).style("border-bottom: 1px solid rgba(255,255,255,0.03);"):
                                                ui.checkbox(
                                                    desc.name, value=_is_sel,
                                                    on_change=lambda e, dn=desc.name: _toggle_individual(e.value, dn),
                                                ).props("dense").style("min-width: 180px;")
                                                ui.label(f"({desc.library})").classes("text-caption text-grey-6")
                                                if desc.meaning:
                                                    ui.label(desc.meaning).classes("text-caption text-grey").style(
                                                        "font-size: 0.8rem;"
                                                    )

                                def _toggle_rec(val, ds=_valid_dn, r=rec):
                                    s = set(state.get("selected_descriptors", []))
                                    if val:
                                        s.update(ds)
                                        ui.notify(f"{r.target_name}: {len(ds)}個の記述子を追加", type="positive")
                                    else:
                                        s -= set(ds)
                                    state["selected_descriptors"] = list(s)'''

if old3 in content:
    content = content.replace(old3, new3)
    print("Edit 3: Updated 目的変数 tab panel - SUCCESS")
else:
    print("Edit 3: FAILED - string not found")

# ============================================================
# Edit 4: Add tab panels for "数えあげ系" and "LLMによる推奨"
# Need to find where to insert them (after tab_variance panel, before the subcategory comment)
# ============================================================
# Find the location after the variance tab panel
marker4 = '                except Exception as ex:\n                    ui.label(f"分散計算エラー: {ex}").classes("text-red")\n\n\n# ════════════════════════'
if marker4 in content:
    insert_pos = content.find(marker4) + len(marker4)
    new4 = '''
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # タブ4: 数えあげ系 — カウント系記述子
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with ui.tab_panel("count"):
                ui.label(
                    "分子内の原子数、環数、官能基数など「数えあげ系」の記述子を選択します。"
                    "NumHDonors、RingCount、FractionCSP3 などが含まれます。"
                ).classes("text-caption text-grey q-mb-sm")

                # 数えあげ系記述子の特定
                _counting_patterns = ("num", "count", "n_", "heavyatom", "ringcount")
                counting_descs = [
                    d for d in all_descs
                    if any(p in d.lower() for p in _counting_patterns)
                ]

                # 代表的数えあげ記述子を優先表示
                _priority_counting = [
                    "MolWt", "HeavyAtomCount", "NumHDonors", "NumHAcceptors",
                    "NumRotatableBonds", "NumAromaticRings", "RingCount",
                    "NumSaturatedRings", "NumAliphaticRings", "NumHeterocycles",
                    "NumAromaticHeterocycles", "FractionCSP3", "BertzCT",
                ]
                priority_set = [d for d in _priority_counting if d in counting_descs]
                other_counting = [d for d in counting_descs if d not in priority_set]
                sorted_counting = priority_set + sorted(other_counting)

                if not sorted_counting:
                    ui.label("数えあげ系記述子が見つかりませんでした。").classes("text-grey")
                else:
                    # 全選択/全解除ボタン
                    with ui.row().classes("q-gutter-sm q-mb-sm"):
                        def _select_all_count():
                            s = set(state.get("selected_descriptors", [])) | set(sorted_counting)
                            state["selected_descriptors"] = list(s)
                            ui.notify(f"数えあげ系 {len(sorted_counting)}件を選択", type="positive")

                        def _deselect_all_count():
                            s = set(state.get("selected_descriptors", [])) - set(sorted_counting)
                            state["selected_descriptors"] = list(s)
                            ui.notify(f"数えあげ系を全解除", type="info")

                        ui.button("全選択", on_click=_select_all_count).props(
                            "outline size=sm no-caps color=blue"
                        )
                        ui.button("全解除", on_click=_deselect_all_count).props(
                            "outline size=sm no-caps color=red"
                        )
                        ui.label(f"計 {len(sorted_counting)} 件").classes("text-caption text-grey q-ml-sm")

                    # 個別選択チェックボックス
                    _count_selected = set(state.get("selected_descriptors", []))
                    for d in sorted_counting:
                        _meaning = _catalog_meanings.get(d, "")
                        with ui.row().classes(
                            "items-center full-width q-py-xs q-gutter-xs"
                        ).style("border-bottom: 1px solid rgba(255,255,255,0.05);"):
                            def _toggle_count(val, dn=d):
                                s = set(state.get("selected_descriptors", []))
                                if val:
                                    s.add(dn)
                                else:
                                    s.discard(dn)
                                state["selected_descriptors"] = list(s)

                            ui.checkbox(
                                d, value=(d in _count_selected),
                                on_change=lambda e, dn=d: _toggle_count(e.value, dn),
                            ).props("dense").style("min-width: 200px;")
                            if _meaning:
                                ui.label(_meaning).classes(
                                    "text-caption text-grey"
                                ).style("font-size: 0.82rem;")

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # タブ5: LLMによる推奨 — LLM推奨記述子
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            with ui.tab_panel("llm"):
                ui.label(
                    "LLM（大規模言語モデル）を活用して、目的変数に適した記述子を推奨します。"
                    "化学的知見に基づいた推奨を行います。"
                ).classes("text-caption text-grey q-mb-sm")

                target_col = state.get("target_col")
                if not target_col:
                    ui.label(
                        "⚠️ 目的変数が設定されていません。「列の役割」タブで目的変数を指定してください。"
                    ).classes("text-amber text-caption q-mb-sm")

                # LLM推奨の取得
                try:
                    from backend.chem.recommender import (
                        get_target_recommendation_by_name, get_all_target_recommendations,
                        get_target_names, get_target_categories,
                    )

                    # 目的変数に基づく推奨
                    rec = None
                    if target_col:
                        rec = get_target_recommendation_by_name(target_col)

                    # 推奨が見つからない場合は全推奨から選択可能にする
                    all_recs = get_all_target_recommendations()
                    target_names = get_target_names()
                    target_categories = get_target_categories()

                    if not rec and target_col:
                        # 目的変数名に部分一致する推奨を探す
                        for r in all_recs:
                            if target_col.lower() in r.target_name.lower() or \
                               r.target_name.lower() in target_col.lower():
                                rec = r
                                break

                    # カテゴリ別に推奨を表示
                    if target_categories:
                        ui.label("物性カテゴリ別推奨記述子").classes("text-subtitle2 q-mt-sm q-mb-xs")

                        for cat in target_categories:
                            cat_recs = [r for r in all_recs if r.category == cat]
                            if not cat_recs:
                                continue

                            with ui.expansion(f"{cat} ({len(cat_recs)}件)", icon="category").classes("full-width q-mb-xs"):
                                for r in cat_recs:
                                    with ui.card().classes("full-width q-pa-xs q-ma-xs").style(
                                        "background: rgba(255,255,255,0.05);"
                                    ):
                                        with ui.row().classes("items-center q-gutter-sm"):
                                            ui.label(f"📌 {r.target_name}").classes("text-bold")
                                            if r.category:
                                                ui.badge(r.category, color="blue").props("outline")

                                        ui.label(r.summary).classes("text-caption text-grey q-mt-xs")

                                        # 推奨記述子の表示
                                        _rec_descs = [d for d in r.descriptors if d.name in available_set]
                                        if _rec_descs:
                                            ui.label("推奨記述子:").classes("text-caption text-bold q-mt-xs")

                                            # 全選択/全解除ボタン
                                            _rec_names = [d.name for d in _rec_descs]
                                            with ui.row().classes("q-gutter-xs q-mb-xs"):
                                                def _select_rec_descs(val, names=_rec_names):
                                                    s = set(state.get("selected_descriptors", []))
                                                    if val:
                                                        s.update(names)
                                                    else:
                                                        s -= set(names)
                                                    state["selected_descriptors"] = list(s)
                                                    ui.notify(f"{r.target_name}の推奨 {len(names)}件を{'選択' if val else '解除'}", type="positive")

                                                ui.button(
                                                    "全選択",
                                                    on_click=lambda e, n=_rec_names: _select_rec_descs(True, n)
                                                ).props("outline size=xs no-caps color=blue")
                                                ui.button(
                                                    "全解除",
                                                    on_click=lambda e, n=_rec_names: _select_rec_descs(False, n)
                                                ).props("outline size=xs no-caps color=red")

                                            for desc in _rec_descs:
                                                def _toggle_rec(val, dn=desc.name):
                                                    s = set(state.get("selected_descriptors", []))
                                                    if val:
                                                        s.add(dn)
                                                    else:
                                                        s.discard(dn)
                                                    state["selected_descriptors"] = list(s)

                                                _is_sel = desc.name in state.get("selected_descriptors", [])
                                                with ui.row().classes("items-center q-gutter-xs").style(
                                                    "border-bottom: 1px solid rgba(255,255,255,0.03);"
                                                ):
                                                    ui.checkbox(
                                                        desc.name, value=_is_sel,
                                                        on_change=lambda e, dn=desc.name: _toggle_rec(e.value, dn),
                                                    ).props("dense").style("min-width: 180px;")
                                                    ui.label(f"({desc.library})").classes("text-caption text-grey-6")
                                                    if desc.meaning:
                                                        ui.label(desc.meaning).classes("text-caption text-grey").style(
                                                            "font-size: 0.8rem;"
                                                        )

                                        # 現在の目的変数に一致する場合はハイライト
                                        if rec and rec.target_name == r.target_name:
                                            ui.label("⭐ 現在の目的変数に一致").classes("text-caption text-cyan q-mt-xs")

                    # 現在の目的変数の推奨をハイライト表示
                    if rec:
                        ui.separator().classes("q-my-sm")
                        with ui.card().classes("full-width q-pa-sm").style(
                            "background: rgba(0,200,255,0.1); border: 1px solid rgba(0,200,255,0.3);"
                        ):
                            ui.label(f"⭐ 現在の目的変数「{target_col}」の推奨").classes("text-bold text-cyan")
                            ui.label(rec.summary).classes("text-caption q-mt-xs")

                            _rec_descs = [d for d in rec.descriptors if d.name in available_set]
                            if _rec_descs:
                                _rec_names = [d.name for d in _rec_descs]

                                with ui.row().classes("q-gutter-sm q-mt-sm"):
                                    def _select_all_llm():
                                        s = set(state.get("selected_descriptors", [])) | set(_rec_names)
                                        state["selected_descriptors"] = list(s)
                                        ui.notify(f"LLM推奨 {len(_rec_names)}件を選択", type="positive")

                                    def _deselect_all_llm():
                                        s = set(state.get("selected_descriptors", [])) - set(_rec_names)
                                        state["selected_descriptors"] = list(s)
                                        ui.notify("LLM推奨を全解除", type="info")

                                    ui.button("全選択", on_click=_select_all_llm).props(
                                        "outline size=sm no-caps color=cyan"
                                    )
                                    ui.button("全解除", on_click=_deselect_all_llm).props(
                                        "outline size=sm no-caps color=red"
                                    )

                                for desc in _rec_descs:
                                    def _toggle_llm(val, dn=desc.name):
                                        s = set(state.get("selected_descriptors", []))
                                        if val:
                                            s.add(dn)
                                        else:
                                            s.discard(dn)
                                        state["selected_descriptors"] = list(s)

                                    _is_sel = desc.name in state.get("selected_descriptors", [])
                                    with ui.row().classes("items-center q-gutter-xs q-py-xs").style(
                                        "border-bottom: 1px solid rgba(255,255,255,0.05);"
                                    ):
                                        ui.checkbox(
                                            desc.name, value=_is_sel,
                                            on_change=lambda e, dn=desc.name: _toggle_llm(e.value, dn),
                                        ).props("dense").style("min-width: 180px;")
                                        ui.label(f"({desc.library})").classes("text-caption text-grey-6")
                                        if desc.meaning:
                                            ui.label(desc.meaning).classes("text-caption text-grey")

                except Exception as e:
                    ui.label(f"LLM推奨の取得エラー: {e}").classes("text-red")
'''
    content = content[:insert_pos] + new4 + content[insert_pos:]
    print("Edit 4: Added tab panels for count and llm - SUCCESS")
else:
    print("Edit 4: FAILED - marker not found")

# ============================================================
# Edit 5: Update _ensure_default_sets to add new sets
# ============================================================
old5 = '''        # 相関セットが作れなかった場合: 分散ベースにフォールバック
        if "📈 相関Top-N" not in sets:
            try:
                variances = precalc_df.var(numeric_only=True).dropna()
                sorted_var = variances.sort_values(ascending=False).index.tolist()
                max_var_n = min(n_samples // 5, 50)
                sets["📊 分散Top-N"] = {
                    "engines": [],
                    "active": True,
                    "descriptors": sorted_var[:max(max_var_n, 5)],
                }
            except Exception as e:
                _logger.warning("分散Top-Nセット生成エラー: %s", e)

        # デフォルトセット: MOLAI+PCA（全記述子は説明変数過多のため非推奨）'''

new5 = '''        # 相関セットが作れなかった場合: 分散ベースにフォールバック
        if "📈 相関Top-N" not in sets:
            try:
                variances = precalc_df.var(numeric_only=True).dropna()
                sorted_var = variances.sort_values(ascending=False).index.tolist()
                max_var_n = min(n_samples // 5, 50)
                sets["📊 分散Top-N"] = {
                    "engines": [],
                    "active": True,
                    "descriptors": sorted_var[:max(max_var_n, 5)],
                }
            except Exception as e:
                _logger.warning("分散Top-Nセット生成エラー: %s", e)

        # ─── セットD: 🤖 LLMによる推奨 ───
        target_col = state.get("target_col")
        if target_col:
            try:
                from backend.chem.recommender import (
                    get_target_recommendation_by_name, get_all_target_recommendations,
                )
                rec = get_target_recommendation_by_name(target_col)
                if rec is None:
                    # 完全一致しない場合は全推奨から該当しそうなものを探す
                    all_recs = get_all_target_recommendations()
                    if all_recs:
                        rec = all_recs[0]  # デフォルトで最初のを使用

                if rec:
                    llm_descs = [d.name for d in rec.descriptors if d.name in available_set]
                    if llm_descs:
                        sets["🤖 LLMによる推奨"] = {
                            "engines": list(set(
                                ["" for d in rec.descriptors if d.library == "RDKit"] +
                                ["MolAI"] if any(d.library == "MolAI" for d in rec.descriptors) else []
                            )),
                            "active": True,
                            "descriptors": llm_descs[:max(min(n_samples // 3, len(llm_descs)), 3)],
                            "summary": rec.summary,
                            "category": rec.category,
                        }
            except Exception as e:
                _logger.warning("LLM推奨セット生成エラー: %s", e)

        # ─── セットE: 🔢 数えあげ系 ───
        _counting_patterns = ("num", "count", "n_", "heavyatom", "ringcount")
        counting_descs = [
            d for d in all_descs
            if any(p in d.lower() for p in _counting_patterns)
        ]
        # 代表的数えあげ記述子を優先
        _priority_counting = [
            "MolWt", "HeavyAtomCount", "NumHDonors", "NumHAcceptors",
            "NumRotatableBonds", "NumAromaticRings", "RingCount",
            "NumSaturatedRings", "NumAliphaticRings", "NumHeterocycles",
            "NumAromaticHeterocycles", "FractionCSP3", "BertzCT",
        ]
        priority_set = [d for d in _priority_counting if d in available_set]
        other_counting = [d for d in counting_descs if d not in priority_set]
        counting_descs = priority_set + other_counting

        if counting_descs:
            max_count = min(n_samples // 3, len(counting_descs))
            sets["🔢 数えあげ系"] = {
                "engines": [],
                "active": True,
                "descriptors": counting_descs[:max(max_count, 5)],
            }

        # デフォルトセット: MOLAI+PCA（全記述子は説明変数過多のため非推奨）'''

if old5 in content:
    content = content.replace(old5, new5)
    print("Edit 5: Updated _ensure_default_sets - SUCCESS")
else:
    print("Edit 5: FAILED - string not found")

# ============================================================
# Edit 6: Update default set priority order
# ============================================================
old6 = '''            for preferred in ["📈 相関Top-N", "📊 分散Top-N", "🎯 汎用QSPR", "🧠 MolAI+PCA"]:'''
new6 = '''            for preferred in ["🤖 LLMによる推奨", "🔢 数えあげ系", "📈 相関Top-N", "📊 分散Top-N", "🎯 汎用QSPR", "🧠 MolAI+PCA"]:'''

if old6 in content:
    content = content.replace(old6, new6)
    print("Edit 6: Updated default set priority - SUCCESS")
else:
    print("Edit 6: FAILED - string not found")

# Write the updated content back to the file
with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("\nAll edits completed!")
