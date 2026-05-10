"""
frontend_nicegui/components/descriptor_selector_dialog.py

記述子の個別選択ダイアログ + 選択済み一覧 + 複数セット管理。

設計:
  1) 各エンジンの「詳細選択」ボタン → ダイアログ（カテゴリ別展開+チェックボックス）
  2) 選択済み記述子の一覧表示 + 個別ON/OFF
  3) 複数の記述子セット(パターン)を定義し、同時に解析を試行

UIが縦長にならないよう、ダイアログ方式を採用。
"""
from __future__ import annotations

import copy
import logging
from typing import Any

from nicegui import ui

from frontend_nicegui.components.descriptor_catalog import (
    ENGINE_CATALOG_MAP,
    SUPPORTED_ENGINES,
    get_catalog,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. 記述子個別選択ダイアログ
# ═══════════════════════════════════════════════════════════

def open_descriptor_detail_dialog(
    engine_name: str,
    state: dict,
) -> None:
    """
    指定エンジンの記述子をカテゴリ別に展開し、
    各記述子をチェックボックスで個別に選択/解除できるダイアログを開く。

    選択結果は state["selected_descriptors"][engine_name] に list[str] として格納。
    """
    catalog = get_catalog(engine_name)
    if not catalog:
        ui.notify(f"⚠️ {engine_name}のカタログは未定義です", type="warning")
        return

    # 選択状態の初期化
    if "selected_descriptors" not in state:
        state["selected_descriptors"] = {}
    if engine_name not in state["selected_descriptors"]:
        # デフォルト: 全記述子をON
        all_names = []
        for descs in catalog.values():
            for d in descs:
                if not d["name"].startswith("_"):
                    all_names.append(d["name"])
        state["selected_descriptors"][engine_name] = list(all_names)

    selected = set(state["selected_descriptors"][engine_name])
    all_desc_names = []
    for descs in catalog.values():
        for d in descs:
            if not d["name"].startswith("_"):
                all_desc_names.append(d["name"])

    # チェックボックスの参照を保持（全選択/全解除で使う）
    checkbox_refs: list[tuple[str, Any]] = []

    with ui.dialog() as dlg, ui.card().classes("q-pa-md").style(
        "width: 85vw; max-width: 1100px; max-height: 85vh;"
    ):
        # ── ヘッダー ──
        with ui.row().classes("items-center justify-between full-width q-mb-sm"):
            ui.label(f"🔬 {engine_name} 記述子選択").classes("text-h6")
            n_sel = len([n for n in all_desc_names if n in selected])
            count_lbl = ui.label(f"{n_sel}/{len(all_desc_names)} 選択中").classes(
                "text-caption text-cyan"
            )

        def _update_count():
            n = len([n for n in all_desc_names if n in selected])
            count_lbl.text = f"{n}/{len(all_desc_names)} 選択中"

        # ── クイックボタン + 検索 (F-09) ──
        with ui.row().classes("q-gutter-sm q-mb-sm items-center full-width"):
            def _select_all():
                for n in all_desc_names:
                    selected.add(n)
                for name, cb in checkbox_refs:
                    cb.value = True
                _update_count()

            def _deselect_all():
                for n in all_desc_names:
                    selected.discard(n)
                for name, cb in checkbox_refs:
                    cb.value = False
                _update_count()

            ui.button("全選択", on_click=_select_all).props(
                "outline size=sm no-caps color=cyan"
            )
            ui.button("全解除", on_click=_deselect_all).props(
                "outline size=sm no-caps color=grey"
            )

            ui.space()

            # F-09: 検索ボックス
            search_input = ui.input(
                placeholder="🔍 記述子を検索...",
            ).props(
                'dense outlined clearable'
            ).style("min-width: 280px;").tooltip(
                "記述子名・カテゴリ名・日本語説明で絞り込み"
            )

            # 検索フィルタ用リスト
            _filter_rows: list[tuple] = []
            _filter_expansions: list[tuple] = []

            def _on_search(e):
                """検索テキストで記述子行の表示/非表示を切り替え"""
                query = (e.value or "").lower().strip()
                for row_el, text in _filter_rows:
                    if not query or query in text:
                        row_el.style(remove="display: none")
                    else:
                        row_el.style(add="display: none")
                for exp_el, cat_text, child_texts in _filter_expansions:
                    if not query:
                        exp_el.style(remove="display: none")
                    elif query in cat_text or any(query in t for t in child_texts):
                        exp_el.style(remove="display: none")
                        if query and query not in cat_text:
                            exp_el.value = True
                    else:
                        exp_el.style(add="display: none")

            search_input.on('update:model-value', _on_search)

        ui.separator()

        # ── カテゴリ別記述子一覧 ──
        with ui.scroll_area().style("max-height: 55vh;"):
            for cat_name, descs in catalog.items():
                cat_actual = [d for d in descs if not d["name"].startswith("_")]
                cat_group = [d for d in descs if d["name"].startswith("_")]
                n_cat_sel = len([d for d in cat_actual if d["name"] in selected])

                exp_panel = ui.expansion(
                    f"{cat_name}  ({n_cat_sel}/{len(cat_actual)})",
                    icon="folder",
                ).classes("full-width q-mb-xs")
                # F-09: カテゴリのフィルタ登録
                cat_search_texts = [d["name"].lower() + " " + d.get("short", "").lower() for d in cat_actual]
                _filter_expansions.append((exp_panel, cat_name.lower(), cat_search_texts))
                with exp_panel:
                    # カテゴリ内全選択/解除
                    cat_names_list = [d["name"] for d in cat_actual]
                    with ui.row().classes("q-gutter-xs q-mb-xs"):
                        def _cat_on(ns=cat_names_list):
                            for n in ns:
                                selected.add(n)
                            for name, cb in checkbox_refs:
                                if name in ns:
                                    cb.value = True
                            _update_count()

                        def _cat_off(ns=cat_names_list):
                            for n in ns:
                                selected.discard(n)
                            for name, cb in checkbox_refs:
                                if name in ns:
                                    cb.value = False
                            _update_count()

                        ui.button("全選択", on_click=_cat_on).props(
                            "flat dense size=xs no-caps color=cyan"
                        )
                        ui.button("全解除", on_click=_cat_off).props(
                            "flat dense size=xs no-caps color=grey"
                        )

                    # グループアイテム（FPなど）
                    for g in cat_group:
                        b = g.get("bits", "")
                        bits_str = f" ({b}bit)" if b else ""
                        ui.label(f"  📦 {g.get('short', g['name'])}{bits_str}").classes(
                            "text-caption text-grey"
                        )

                    # 個別記述子チェックボックス
                    for desc in cat_actual:
                        dname = desc["name"]
                        short = desc.get("short", "")
                        desc_row = ui.row().classes("items-center q-gutter-xs").style(
                            "min-height: 28px;"
                        )
                        # F-09: 行のフィルタ登録
                        _filter_rows.append((desc_row, (dname + " " + short).lower()))
                        with desc_row:
                            cb = ui.checkbox(
                                dname,
                                value=(dname in selected),
                            ).props("dense").style("min-width: 200px;")

                            def _on_change(e, n=dname):
                                if e.value:
                                    selected.add(n)
                                else:
                                    selected.discard(n)
                                _update_count()

                            cb.on_value_change(_on_change)
                            checkbox_refs.append((dname, cb))

                            if short:
                                ui.label(short).classes(
                                    "text-caption text-grey"
                                ).style("font-size: 0.72rem;")

        ui.separator()

        # ── フッター ──
        with ui.row().classes("justify-end q-gutter-sm"):
            def _apply():
                state["selected_descriptors"][engine_name] = [
                    n for n in all_desc_names if n in selected
                ]
                n = len(state["selected_descriptors"][engine_name])
                ui.notify(f"✅ {engine_name}: {n}記述子を選択", type="positive")
                dlg.close()

            ui.button("キャンセル", on_click=dlg.close).props("flat no-caps color=grey")
            ui.button("適用", on_click=_apply).props("no-caps color=cyan")

    dlg.open()


# ═══════════════════════════════════════════════════════════
# 2. 記述子一覧（グループフィルター+相関閾値+検索+ソート）
# ═══════════════════════════════════════════════════════════

def _classify_descriptor_group(col_name: str) -> str:
    """記述子名からグループを自動分類する。"""
    cl = col_name.lower()
    if cl.startswith("morgan_r2") or cl.startswith("morgan2"):
        return "morgan_r2"
    if cl.startswith("morgan_r3") or cl.startswith("morgan3"):
        return "morgan_r3"
    if cl.startswith("morgan"):
        return "morgan_other"
    if cl.startswith("maccs"):
        return "maccs"
    if cl.startswith("avalon"):
        return "avalon"
    if cl.startswith("xtb_"):
        return "xtb"
    if cl.startswith("mu_") or cl.startswith("ln_gamma") or cl.startswith("cosmo_"):
        return "cosmo"
    if cl.startswith("mordred_") or cl.startswith("mrd_"):
        return "mordred"
    if cl.startswith("molfeat_"):
        return "molfeat"
    if cl.startswith("mol2vec_"):
        return "mol2vec"
    if cl.startswith("chemprop_"):
        return "chemprop"
    if cl.startswith("uma_"):
        return "uma"
    if cl.startswith("padel_"):
        return "padel"
    if cl.startswith("ds_"):
        return "descriptastorus"
    if cl.startswith("pka") or cl == "pKa_pred":
        return "unipka"
    if cl.startswith("joback_"):
        return "group_contrib"
    if cl.startswith("fr_"):
        return "rdkit_functional"
    if cl.startswith("gasteiger_"):
        return "rdkit_charge"
    if cl.startswith("molai_") or cl.startswith("cnn_pca_"):
        return "molai"
    # fallback
    return "rdkit_property"


_GROUP_DISPLAY = {
    "rdkit_property":   {"label": "RDKit 物性記述子", "icon": "🧪", "default_on": True},
    "rdkit_functional": {"label": "RDKit 官能基",     "icon": "⚗️", "default_on": True},
    "rdkit_charge":     {"label": "RDKit 電荷",       "icon": "⚡", "default_on": True},
    "molai":            {"label": "MolAI (CNN+PCA)",  "icon": "🧠", "default_on": True},
    "xtb":              {"label": "xTB 量子化学",      "icon": "⚛️", "default_on": True},
    "cosmo":            {"label": "COSMO-RS",          "icon": "🌊", "default_on": True},
    "group_contrib":    {"label": "基団寄与法",         "icon": "🔗", "default_on": True},
    "mordred":          {"label": "Mordred",           "icon": "🔬", "default_on": True},
    "descriptastorus":  {"label": "DescriptaStorus",   "icon": "📊", "default_on": True},
    "molfeat":          {"label": "Molfeat",           "icon": "🧬", "default_on": True},
    "mol2vec":          {"label": "Mol2Vec",           "icon": "📝", "default_on": True},
    "chemprop":         {"label": "Chemprop",          "icon": "🔴", "default_on": True},
    "uma":              {"label": "UMA",               "icon": "🏛️", "default_on": True},
    "padel":            {"label": "PaDEL",             "icon": "📦", "default_on": True},
    "unipka":           {"label": "UniPKa",            "icon": "🔵", "default_on": True},
    "morgan_r2":        {"label": "Morgan FP (r2)",    "icon": "🔵", "default_on": False},
    "morgan_r3":        {"label": "Morgan FP (r3)",    "icon": "🔵", "default_on": False},
    "morgan_other":     {"label": "Morgan FP (他)",    "icon": "🔵", "default_on": False},
    "maccs":            {"label": "MACCS Keys",        "icon": "🟢", "default_on": True},
    "avalon":           {"label": "Avalon FP",         "icon": "🟣", "default_on": True},
}


def render_selected_descriptors_panel(state: dict) -> None:
    """
    現在選択されている記述子の一覧を表示。
    グループフィルター + 相関係数閾値 + 検索 + ソート + ページネーション。
    """
    import numpy as np

    precalc_df = state.get("precalc_df")
    if precalc_df is None:
        return

    all_cols = list(precalc_df.columns)
    if "active_descriptors" not in state:
        state["active_descriptors"] = list(all_cols)

    n_active = len(state["active_descriptors"])
    n_total = len(all_cols)

    # ── 相関係数を計算 ──
    target_col = state.get("target_col", "")
    df_main = state.get("df")
    corr_map: dict[str, float] = {}
    if df_main is not None and target_col and target_col in df_main.columns:
        try:
            target_vals = df_main[target_col].astype(float)
            for col in all_cols:
                if col in precalc_df.columns:
                    try:
                        c = float(precalc_df[col].astype(float).corr(target_vals))
                        if np.isfinite(c):
                            corr_map[col] = c
                    except Exception:
                        corr_map[col] = 0.0
        except Exception:
            pass

    # ── グループ分類 ──
    group_map: dict[str, str] = {}
    group_counts: dict[str, int] = {}
    for col in all_cols:
        g = _classify_descriptor_group(col)
        group_map[col] = g
        group_counts[g] = group_counts.get(g, 0) + 1

    # 状態管理 (Inline化するため、状態が失われないように state に保存)
    if "_filter_state" not in state:
        state["_filter_state"] = {
            "visible_groups": {
                g for g, info in _GROUP_DISPLAY.items()
                if info.get("default_on", True) and g in group_counts
            },
            "corr_threshold": 0.0,
            "search_text": "",
            "sort_by": "corr_desc",
            "page": 1,
            "rows_per_page": 50,
        }
    filter_state = state["_filter_state"]

    with ui.expansion(f"📋 選択中の特徴量一覧・個別管理 ({n_active} / {n_total} 選択中)", icon="checklist").classes("full-width bg-dark q-mt-sm").props("default-opened header-class='text-h6'"):
        with ui.card().classes("full-width q-pa-md").style("border: 1px solid rgba(0,212,255,0.2); background: rgba(0,0,0,0.1)"):
            
            # ── ヘッダー ──
            with ui.row().classes("items-center justify-between full-width q-mb-sm"):
                count_lbl = ui.label(f"{n_active}/{n_total} 使用中").classes("text-subtitle1 text-cyan text-bold")
                visible_lbl = ui.label("").classes("text-caption text-grey")

            # ── グループフィルター ──
            with ui.expansion("🔍 表示フィルター", icon="filter_list").classes("full-width q-mb-md").props("dense"):
                ui.label("グループ表示").classes("text-body2 text-bold q-mb-xs")
                group_cbs: dict[str, Any] = {}
                with ui.element("div").style("display:grid;grid-template-columns:repeat(auto-fill, minmax(150px, 1fr));gap:4px;"):
                    for g_key in sorted(group_counts.keys(), key=lambda k: _GROUP_DISPLAY.get(k, {}).get("label", k)):
                        info = _GROUP_DISPLAY.get(g_key, {"label": g_key, "icon": "📄", "default_on": True})
                        cnt = group_counts.get(g_key, 0)
                        visible = g_key in filter_state["visible_groups"]
                        cb = ui.checkbox(f"{info['icon']} {info['label']} ({cnt})", value=visible).props("dense size=xs")
                        group_cbs[g_key] = cb

                        def _toggle_group(e, gk=g_key):
                            if e.value:
                                filter_state["visible_groups"].add(gk)
                            else:
                                filter_state["visible_groups"].discard(gk)
                            filter_state["page"] = 1
                            _rebuild_table()

                        cb.on_value_change(_toggle_group)

                ui.separator().classes("q-my-sm")

                # 相関係数閾値
                with ui.row().classes("items-center q-gutter-sm"):
                    ui.label("相関係数閾値: |r| ≧").classes("text-body2")
                    corr_slider = ui.slider(min=0, max=0.9, step=0.05, value=filter_state["corr_threshold"]).props("label-always dense").classes("col-4")
                    corr_val_lbl = ui.label(f"{filter_state['corr_threshold']:.2f}").classes("text-body2 text-bold").style("min-width:40px;")

                    def _on_corr(e):
                        filter_state["corr_threshold"] = float(e.value)
                        corr_val_lbl.set_text(f"{float(e.value):.2f}")
                        filter_state["page"] = 1
                        _rebuild_table()

                    corr_slider.on("update:model-value", _on_corr)

                ui.separator().classes("q-my-sm")

                # クイックアクション
                with ui.row().classes("q-gutter-sm"):
                    def _show_all_groups():
                        for gk, cb in group_cbs.items():
                            filter_state["visible_groups"].add(gk)
                            cb.value = True
                        _rebuild_table()

                    def _hide_all_groups():
                        filter_state["visible_groups"].clear()
                        for gk, cb in group_cbs.items():
                            cb.value = False
                        _rebuild_table()

                    def _high_corr_only():
                        filter_state["corr_threshold"] = 0.3
                        corr_slider.value = 0.3
                        corr_val_lbl.set_text("0.30")
                        for gk, cb in group_cbs.items():
                            filter_state["visible_groups"].add(gk)
                            cb.value = True
                        _rebuild_table()

                    ui.button("すべて表示", on_click=_show_all_groups).props("outline size=sm no-caps color=cyan")
                    ui.button("すべて非表示", on_click=_hide_all_groups).props("outline size=sm no-caps color=grey")
                    ui.button("高相関のみ (|r|≧0.3)", on_click=_high_corr_only).props("outline size=sm no-caps color=teal")

            # ── 検索 + ソート + 全ON/OFF ──
            with ui.row().classes("items-center q-gutter-sm full-width q-mb-xs"):
                search_input = ui.input("検索", placeholder="記述子名で絞り込み", value=filter_state["search_text"]).props("dense outlined clearable").style("width: 300px;")

                def _on_search(e):
                    filter_state["search_text"] = (e.value or "").lower()
                    filter_state["page"] = 1
                    _rebuild_table()

                search_input.on("update:model-value", _on_search)

                sort_toggle = ui.toggle(
                    {"corr_desc": "|r| 降順", "corr_asc": "|r| 昇順", "name_asc": "名前順"},
                    value=filter_state["sort_by"],
                ).props("dense no-caps rounded size=sm")

                def _on_sort(e):
                    filter_state["sort_by"] = e.value
                    _rebuild_table()

                sort_toggle.on("update:model-value", _on_sort)

                # 一括操作
                active_set = set(state["active_descriptors"])
                def _all_on():
                    for c in all_cols: active_set.add(c)
                    state["active_descriptors"] = list(active_set)
                    _rebuild_table()

                def _all_off():
                    active_set.clear()
                    state["active_descriptors"] = list(active_set)
                    _rebuild_table()

                ui.button("全表示中ON", on_click=lambda: _toggle_filtered_on_off(True)).props("flat dense size=sm no-caps color=cyan")
                ui.button("全表示中OFF", on_click=lambda: _toggle_filtered_on_off(False)).props("flat dense size=sm no-caps color=grey")
                
                def _toggle_filtered_on_off(turn_on: bool):
                    rows = _get_filtered_rows()
                    for r in rows:
                        if turn_on: active_set.add(r["name"])
                        else: active_set.discard(r["name"])
                    state["active_descriptors"] = list(active_set)
                    _rebuild_table()

            ui.separator().classes("q-my-sm")

            # ── テーブルコンテナ ──
            table_container = ui.column().classes("full-width")
            page_container = ui.row().classes("items-center justify-center q-gutter-sm q-mt-sm")

            def _get_filtered_rows() -> list[dict]:
                rows = []
                search = filter_state["search_text"]
                threshold = filter_state["corr_threshold"]

                for col in all_cols:
                    g = group_map.get(col, "rdkit_property")
                    if g not in filter_state["visible_groups"]: continue
                    r = abs(corr_map.get(col, 0.0))
                    if r < threshold: continue
                    if search and search not in col.lower(): continue

                    g_info = _GROUP_DISPLAY.get(g, {"label": g, "icon": "📄"})
                    rows.append({
                        "name": col,
                        "corr": corr_map.get(col, 0.0),
                        "abs_corr": r,
                        "group": g,
                        "group_label": f"{g_info['icon']} {g_info['label']}",
                        "active": col in active_set,
                    })

                sb = filter_state["sort_by"]
                if sb == "corr_desc": rows.sort(key=lambda r: r["abs_corr"], reverse=True)
                elif sb == "corr_asc": rows.sort(key=lambda r: r["abs_corr"])
                else: rows.sort(key=lambda r: r["name"])
                return rows

            def _rebuild_table():
                table_container.clear()
                page_container.clear()

                rows = _get_filtered_rows()
                n_filtered = len(rows)
                rpp = filter_state["rows_per_page"]
                total_pages = max(1, (n_filtered + rpp - 1) // rpp)
                page = min(filter_state["page"], total_pages)
                filter_state["page"] = page
                
                start = (page - 1) * rpp
                end = start + rpp
                page_rows = rows[start:end]

                n_act = len([c for c in all_cols if c in active_set])
                count_lbl.set_text(f"{n_act}/{n_total} 使用中")
                visible_lbl.set_text(f"表示中: {n_filtered} 件 / {n_total} 件")

                with table_container:
                    # 動的除外理由（VIF代わりの共線性フィルタなど）を取得
                    current_set = state.get("current_set_name", "")
                    ex_reasons = state.get("descriptor_sets", {}).get(current_set, {}).get("exclusion_reasons", {})

                    # AG Gridを使用し高速化と一覧性を向上
                    row_data = []
                    for r in page_rows:
                        reason = ex_reasons.get(r["name"], "")
                        row_data.append({
                            "active": r["active"],
                            "name": r["name"],
                            "corr": f'{r["corr"]:.4f}' if r["corr"] != 0.0 else "-",
                            "group": r["group_label"],
                            "reason": reason
                        })

                    grid_opts = {
                        "columnDefs": [
                            {"headerName": "使用", "field": "active", "checkboxSelection": True, "headerCheckboxSelection": False, "width": 80},
                            {"headerName": "記述子名", "field": "name", "sortable": True, "filter": True, "flex": 2},
                            {"headerName": "相関係数", "field": "corr", "sortable": True, "width": 120},
                            {"headerName": "カテゴリ", "field": "group", "sortable": True, "filter": True, "flex": 1},
                            {"headerName": "除外理由/備考", "field": "reason", "sortable": True, "filter": True, "flex": 2, "cellStyle": {"color": "#fbbf24", "fontSize": "11px"}},
                        ],
                        "rowData": row_data,
                        "rowSelection": "multiple",
                        "suppressRowClickSelection": False,
                    }

                    ag = ui.aggrid(grid_opts).classes("full-width").style("height: 480px;")

                    def _on_selection(e):
                        # AGGridの selectionChanged イベントはNiceGUIでは e.args から取得可能だが、
                        # 今回は行クリックで行うか、AGGrid APIを利用する。
                        pass

                    # 簡易的な行クリックでの切り替え(NiceGUIのaggridは rowClicked をサポート)
                    def _on_row_click(e):
                        row_name = e.args.get("data", {}).get("name")
                        if not row_name: return
                        
                        if row_name in active_set:
                            active_set.discard(row_name)
                        else:
                            active_set.add(row_name)
                        state["active_descriptors"] = list(active_set)
                        _rebuild_table()
                        
                    ag.on("rowClicked", _on_row_click)

                    # AGGridの選択状態とcheckboxSelectionを同期
                    def update_selection():
                        for i, r in enumerate(row_data):
                            if r["active"]:
                                ag.run_grid_method('api.selectIndex', i, True, False)
                    ui.timer(0.1, update_selection, once=True)

                with page_container:
                    if total_pages > 1:
                        def _go_page(p):
                            filter_state["page"] = p
                            _rebuild_table()

                        if page > 1: ui.button("◀", on_click=lambda: _go_page(page - 1)).props("flat dense size=sm")
                        start_p = max(1, page - 3)
                        end_p = min(total_pages, page + 3)
                        for p in range(start_p, end_p + 1):
                            if p == page: ui.badge(str(p), color="cyan").props("rounded")
                            else: ui.button(str(p), on_click=lambda pp=p: _go_page(pp)).props("flat dense size=sm")
                        if page < total_pages: ui.button("▶", on_click=lambda: _go_page(page + 1)).props("flat dense size=sm")
                        ui.label(f"({page}/{total_pages} ページ)").classes("text-caption text-grey")

            # 初期構築
            _rebuild_table()


# ═══════════════════════════════════════════════════════════
# 3. 複数記述子セット（パターン）管理 — カード型ビジュアルUI
# ═══════════════════════════════════════════════════════════

def _get_engine_badges(descriptors: list[str] | None) -> list[str]:
    """記述子名からエンジン名を推定してバッジ用のリストを返す。"""
    if not descriptors:
        return []
    engines = set()
    for d in descriptors:
        dl = d.lower()
        if dl.startswith("xtb_") or d in (
            "HomoEnergy", "LumoEnergy", "HomoLumoGap",
            "DipoleMoment", "Polarizability",
        ):
            engines.add("XTB")
        elif dl.startswith("joback_") or d in (
            "CohesiveEnergy", "CohesiveEnergyDensity", "FreeVolume",
        ):
            engines.add("基団寄与")
        elif dl.startswith("mu_") or dl.startswith("ln_gamma") or dl.startswith("cosmo_"):
            engines.add("COSMO")
        elif dl.startswith("mordred_") or dl.startswith("mrd_"):
            engines.add("Mordred")
        elif dl.startswith("morgan") or dl.startswith("maccs") or dl.startswith("avalon"):
            engines.add("scikit-FP")
        elif dl.startswith("molfeat_"):
            engines.add("Molfeat")
        elif dl.startswith("mol2vec_"):
            engines.add("Mol2Vec")
        elif dl.startswith("chemprop_"):
            engines.add("Chemprop")
        elif dl.startswith("uma_"):
            engines.add("UMA")
        elif dl.startswith("padel_"):
            engines.add("PaDEL")
        elif dl.startswith("ds_"):
            engines.add("DS")
        elif dl.startswith("pka") or d == "pKa_pred":
            engines.add("UniPKa")
        elif dl.startswith("fr_") or d in ("MolWt", "MolLogP", "TPSA", "qed"):
            engines.add("RDKit")
        elif dl.startswith("gasteiger_"):
            engines.add("RDKit")
        else:
            engines.add("RDKit")
    return sorted(engines)[:5]

_ENGINE_BADGE_COLORS = {
    "RDKit": "green", "XTB": "orange", "COSMO": "purple",
    "基団寄与": "teal", "Mordred": "blue", "scikit-FP": "indigo",
    "Molfeat": "pink", "Mol2Vec": "deep-purple", "Chemprop": "red",
    "UMA": "amber", "PaDEL": "light-blue", "DS": "lime",
    "UniPKa": "cyan", "他": "grey",
}


def _open_set_compare_dialog(sets: dict, state: dict) -> None:
    """セット比較ダイアログ: セット間の記述子重複・差分を表示。"""
    set_names = list(sets.keys())
    if len(set_names) < 2:
        ui.notify("⚠️ 比較には2つ以上のセットが必要です", type="warning")
        return

    with ui.dialog() as dlg, ui.card().classes("q-pa-md").style(
        "width: 85vw; max-width: 1000px; max-height: 85vh;"
    ):
        ui.label("📊 セット比較ダッシュボード").classes("text-h6 q-mb-sm")

        # 各セットの記述子セット
        desc_sets = {}
        for name in set_names:
            descs = sets[name].get("descriptors")
            desc_sets[name] = set(descs) if descs else set()

        # 比較テーブル
        rows = []
        for name in set_names:
            d = desc_sets[name]
            others_union = set()
            for n2 in set_names:
                if n2 != name:
                    others_union |= desc_sets[n2]
            unique = d - others_union
            shared = d & others_union
            rows.append({
                "name": name,
                "total": len(d),
                "unique": len(unique),
                "shared": len(shared),
                "engines": ", ".join(_get_engine_badges(list(d))),
            })

        cols = [
            {"name": "name", "label": "セット名", "field": "name"},
            {"name": "total", "label": "記述子数", "field": "total", "sortable": True},
            {"name": "unique", "label": "固有", "field": "unique", "sortable": True},
            {"name": "shared", "label": "共有", "field": "shared", "sortable": True},
            {"name": "engines", "label": "エンジン", "field": "engines"},
        ]
        ui.table(columns=cols, rows=rows, row_key="name").classes(
            "full-width"
        ).props("dense flat bordered")

        # ペアワイズ重複マトリクス
        if len(set_names) >= 2:
            ui.separator().classes("q-my-sm")
            ui.label("🔗 ペアワイズ重複率 (%)").classes("text-subtitle2")
            matrix_rows = []
            for n1 in set_names:
                row_data = {"name": n1}
                for n2 in set_names:
                    if desc_sets[n1] and desc_sets[n2]:
                        overlap = len(desc_sets[n1] & desc_sets[n2])
                        union = len(desc_sets[n1] | desc_sets[n2])
                        pct = round(overlap / union * 100) if union > 0 else 0
                    else:
                        pct = 0
                    row_data[n2] = f"{pct}%"
                matrix_rows.append(row_data)

            m_cols = [{"name": "name", "label": "", "field": "name"}] + [
                {"name": n, "label": n, "field": n} for n in set_names
            ]
            ui.table(columns=m_cols, rows=matrix_rows, row_key="name").classes(
                "full-width"
            ).props("dense flat bordered")

        ui.separator()
        ui.button("閉じる", on_click=dlg.close).props("outline no-caps color=cyan")

    dlg.open()


def _open_rename_dialog(old_name: str, sets: dict, state: dict) -> None:
    """セット名変更ダイアログ。"""
    if old_name == "デフォルト":
        ui.notify("⚠️ デフォルトセットは名前変更できません", type="warning")
        return

    with ui.dialog() as dlg, ui.card().classes("q-pa-md").style("min-width: 350px;"):
        ui.label("✏️ セット名を変更").classes("text-h6 q-mb-sm")
        name_input = ui.input("新しい名前", value=old_name).props(
            "outlined dense autofocus"
        ).classes("full-width")

        with ui.row().classes("justify-end q-gutter-sm q-mt-sm"):
            def _apply():
                new_name = name_input.value.strip()
                if not new_name or new_name == old_name:
                    dlg.close()
                    return
                if new_name in sets:
                    ui.notify(f"⚠️ 「{new_name}」は既に存在します", type="warning")
                    return
                sets[new_name] = sets.pop(old_name)
                if state.get("current_set_name") == old_name:
                    state["current_set_name"] = new_name
                ui.notify(f"✏️ 「{old_name}」→「{new_name}」に変更", type="positive")
                dlg.close()

            ui.button("キャンセル", on_click=dlg.close).props("flat no-caps color=grey")
            ui.button("変更", on_click=_apply).props("no-caps color=cyan")

    dlg.open()


def render_descriptor_sets_panel(state: dict) -> None:
    """
    複数の記述子セット(パターン)をカード型で管理するUI。
    各セットの内容（記述子数・エンジン構成・プログレスバー）が一目瞭然。
    """
    # セット管理の初期化
    if "descriptor_sets" not in state:
        state["descriptor_sets"] = {
            "デフォルト": {
                "engines": [],
                "active": True,
                "descriptors": None,
            }
        }
    if "current_set_name" not in state:
        state["current_set_name"] = "デフォルト"

    sets = state["descriptor_sets"]
    current = state["current_set_name"]

    # 全記述子数を取得（プログレスバー用）
    precalc_df = state.get("precalc_df")
    total_available = precalc_df.shape[1] if precalc_df is not None else 0

    # ── メインカード ──
    with ui.card().classes("full-width q-pa-md q-mb-sm").style(
        "border: 1px solid rgba(0,188,212,0.3); border-radius: 12px;"
        "background: rgba(0,20,40,0.3);"
    ):
        # ── ヘッダー ──
        with ui.row().classes("items-center justify-between full-width q-mb-sm"):
            with ui.row().classes("items-center q-gutter-sm"):
                ui.icon("layers", color="cyan").classes("text-h5")
                ui.label("記述子セット管理").classes("text-h6")
                ui.badge(f"{len(sets)} セット", color="cyan").props("outline")

            with ui.row().classes("q-gutter-xs"):
                def _add_set():
                    idx = len(sets) + 1
                    name = f"セット{idx}"
                    while name in sets:
                        idx += 1
                        name = f"セット{idx}"
                    active = state.get("active_descriptors", [])
                    sets[name] = {
                        "engines": [],
                        "active": True,
                        "descriptors": list(active) if active else None,
                    }
                    state["current_set_name"] = name
                    ui.notify(f"➕ セット「{name}」を作成", type="positive")

                ui.button("➕ 新規セット", on_click=_add_set).props(
                    "unelevated size=sm no-caps color=green-8"
                ).tooltip("現在の記述子選択を新しいセットとして保存")

                if len(sets) >= 2:
                    ui.button(
                        "📊 比較",
                        on_click=lambda: _open_set_compare_dialog(sets, state),
                    ).props("outline size=sm no-caps color=amber")

        # ── プリセットギャラリー ──
        try:
            from backend.chem.recommender import (
                get_target_categories,
                get_targets_by_category,
            )
            categories = get_target_categories()
            if categories:
                with ui.expansion(
                    "🏪 プリセットから作成", icon="collections_bookmark",
                ).classes("full-width q-mb-sm").props("dense"):
                    ui.label(
                        "予測目標に最適化された記述子セットをワンクリックで作成できます"
                    ).classes("text-caption text-grey q-mb-xs")
                    with ui.row().classes("q-gutter-xs flex-wrap"):
                        for cat_name in categories:
                            cat_recs = get_targets_by_category(cat_name)
                            for rec in cat_recs:
                                def _create_preset(r=rec):
                                    preset_name = f"📌 {r.target_name}"
                                    desc_names = [d.name for d in r.descriptors]
                                    sets[preset_name] = {
                                        "engines": sorted(set(d.library for d in r.descriptors)),
                                        "active": True,
                                        "descriptors": desc_names,
                                    }
                                    state["current_set_name"] = preset_name
                                    state["active_descriptors"] = list(desc_names)
                                    state["_applied_recommendation"] = r
                                    ui.notify(
                                        f"✅ プリセット「{r.target_name}」を作成 ({len(desc_names)}記述子)",
                                        type="positive",
                                    )

                                lib_colors = sorted(set(d.library for d in rec.descriptors))
                                with ui.button(
                                    rec.target_name,
                                    on_click=_create_preset,
                                ).props("outline dense size=sm no-caps color=cyan").classes("text-xs"):
                                    pass
        except ImportError:
            pass

        # ── LLM推奨セクション ──
        try:
            from backend.chem.recommender import get_combined_recommendations

            with ui.expansion("🤖 LLM推奨（4000+特徴量から選択）", icon="smart_toy").classes("full-width q-mb-sm").props("dense"):
                ui.label(
                    "LLMに目的変数を伝えて、4000+の特徴量から最適なものを選択させます"
                ).classes("text-caption text-grey q-mb-xs")

                with ui.row().classes("items-center q-gutter-sm full-width"):
                    target_input = ui.input(
                        "目的変数名（例: 屈折率、Tg、粘度など）",
                        placeholder="目的変数を入力...",
                    ).props("dense outlined").classes("col")
                    target_input.value = state.get("target_col", "")

                    provider_select = ui.select(
                        {"stub": "Stub（テスト）", "openai": "OpenAI", "anthropic": "Anthropic"},
                        label="LLMプロバイダー",
                        value="stub",
                    ).props("dense outlined").classes("w-40")

                # 結果表示エリア
                result_container = ui.column().classes("full-width")

                def _run_llm_recommend():
                    target = target_input.value.strip()
                    if not target:
                        ui.notify("⚠️ 目的変数名を入力してください", type="warning")
                        return

                    provider = provider_select.value
                    ui.notify(f"🤖 LLM推奨を実行中... (target={target}, provider={provider})", type="info")

                    # プログレスダイアログを表示
                    with ui.dialog() as progress_dlg, ui.card().classes("q-pa-md"):
                        ui.label("🤖 LLM推奨を実行中...").classes("text-h6")
                        ui.label(f"目的変数: {target}").classes("text-caption")
                        ui.label("4000+の特徴量からLLMが選択中...").classes("text-caption text-grey")
                        progress_spinner = ui.spinner(size="lg", color="cyan")

                    progress_dlg.open()

                    try:
                        result = get_combined_recommendations(
                            target_name=target,
                            provider_name=provider,
                            llm_weight=0.7,
                        )

                        progress_dlg.close()
                        result_container.clear()
                        with result_container:
                            descriptors = result.get("descriptors", [])
                            if not descriptors:
                                ui.label("⚠️ 推奨結果がありません").classes("text-caption text-grey")
                                return

                            ui.label(f"✅ {len(descriptors)}個の特徴量を推奨（{result.get('llm_selected_count', 0)}個がLLM選択、{result.get('rule_based_count', 0)}個がルールベース）").classes("text-caption text-cyan q-mb-xs")

                            # サマリー
                            summary = result.get("summary", "")
                            if summary:
                                ui.label(f"📝 {summary}").classes("text-caption text-grey q-mb-sm").style("max-width: 600px;")

                            # エラー表示
                            errors = result.get("errors")
                            if errors:
                                ui.label(f"⚠️ 一部エラー: {', '.join(errors[:3])}").classes("text-caption text-amber")

                            # 推奨特徴量リスト（プレビュー）
                            with ui.scroll_area().style("max-height: 300px; width: 100%;"):
                                for i, desc in enumerate(descriptors[:50]):  # 最大50個表示
                                    source = desc.get("source", "unknown")
                                    source_icon = "🤖" if source == "llm" else "📌"
                                    name = desc.get("name", "")
                                    category = desc.get("category", "")
                                    meaning = desc.get("meaning", "")
                                    library = desc.get("library", "")

                                    with ui.row().classes("items-center q-gutter-xs no-wrap"):
                                        ui.label(f"{source_icon}").classes("text-xs")
                                        ui.label(name).classes("text-body2 text-bold")
                                        if library:
                                            ui.label(f"({library})").classes("text-caption text-grey")
                                        if meaning and len(meaning) < 50:
                                            ui.label(f"- {meaning}").classes("text-caption text-grey")

                                if len(descriptors) > 50:
                                    ui.label(f"... 他 {len(descriptors) - 50}個").classes("text-caption text-grey")

                            # セットとして保存ボタン
                            def _save_as_set(desc_list=descriptors, t=target):
                                desc_names = [d.get("name", "") for d in desc_list if d.get("name")]
                                if not desc_names:
                                    ui.notify("⚠️ 保存する特徴量がありません", type="warning")
                                    return
                                set_name = f"🤖 LLM: {t}"
                                idx = 2
                                while set_name in sets:
                                    set_name = f"🤖 LLM: {t} ({idx})"
                                    idx += 1
                                sets[set_name] = {
                                    "engines": list(set(d.get("library", "") for d in desc_list if d.get("library"))),
                                    "active": True,
                                    "descriptors": desc_names,
                                }
                                state["current_set_name"] = set_name
                                state["active_descriptors"] = list(desc_names)
                                ui.notify(f"✅ 「{set_name}」を保存 ({len(desc_names)}特徴量)", type="positive")

                            ui.button("💾 この推奨をセットとして保存", on_click=_save_as_set).props("size=sm no-caps color=teal")

                    except Exception as e:
                        progress_dlg.close()
                        ui.notify(f"❌ LLM推奨エラー: {e}", type="negative")
                        import traceback
                        traceback.print_exc()

                ui.button("🤖 LLMに聞く", on_click=_run_llm_recommend).props("size=sm no-caps color=purple")

        except ImportError:
            pass

        ui.separator().classes("q-my-sm")

        # ── セットカード一覧 ──
        with ui.scroll_area().style("max-height: 400px;"):
            with ui.row().classes("q-gutter-md flex-wrap full-width"):
                for set_name, info in sets.items():
                    is_current = (set_name == current)
                    descs = info.get("descriptors")
                    n_descs = len(descs) if descs else 0
                    engines = _get_engine_badges(descs)
                    pct = round(n_descs / total_available * 100) if total_available > 0 and descs else 0
                    is_active = info.get("active", True)

                    # カードの色設定
                    border_color = "rgba(0,212,255,0.7)" if is_current else (
                        "rgba(255,255,255,0.15)" if is_active else "rgba(255,255,255,0.05)"
                    )
                    bg_color = "rgba(0,35,60,0.6)" if is_current else "rgba(25,25,35,0.5)"
                    glow = "box-shadow: 0 0 15px rgba(0,200,255,0.15);" if is_current else ""

                    with ui.card().classes("q-pa-sm").style(
                        f"border: 2px solid {border_color}; border-radius: 10px;"
                        f"background: {bg_color}; min-width: 240px; max-width: 320px;"
                        f"transition: all 0.3s ease; {glow}"
                    ):
                        # ── カードヘッダー: 名前 + 操作ボタン ──
                        with ui.row().classes("items-center justify-between full-width no-wrap"):
                            with ui.row().classes("items-center q-gutter-xs no-wrap"):
                                if is_current:
                                    ui.icon("play_arrow", color="cyan").classes("text-body1")
                                else:
                                    ui.icon("folder", color="grey").classes("text-body1")
                                ui.label(set_name).classes(
                                    "text-body1 text-bold" + (" text-cyan" if is_current else "")
                                ).style("max-width: 160px; overflow: hidden; text-overflow: ellipsis;")

                            with ui.row().classes("q-gutter-none no-wrap"):
                                # 名前変更
                                ui.button(
                                    icon="edit",
                                    on_click=lambda n=set_name: _open_rename_dialog(n, sets, state),
                                ).props("flat round dense size=xs color=grey").tooltip("名前変更")

                                # 複製
                                def _dup(n=set_name):
                                    new_name = f"{n}_コピー"
                                    i = 2
                                    while new_name in sets:
                                        new_name = f"{n}_コピー{i}"
                                        i += 1
                                    sets[new_name] = copy.deepcopy(sets[n])
                                    state["current_set_name"] = new_name
                                    ui.notify(f"📋 「{new_name}」を複製", type="info")

                                ui.button(
                                    icon="content_copy",
                                    on_click=_dup,
                                ).props("flat round dense size=xs color=grey").tooltip("複製")

                                # 削除
                                if set_name != "デフォルト":
                                    def _del(n=set_name):
                                        del sets[n]
                                        if state.get("current_set_name") == n:
                                            state["current_set_name"] = "デフォルト"
                                        ui.notify(f"🗑️ 「{n}」を削除", type="info")

                                    ui.button(
                                        icon="delete_outline",
                                        on_click=_del,
                                    ).props("flat round dense size=xs color=red-4").tooltip("削除")

                        # ── 記述子数 + プログレスバー ──
                        with ui.column().classes("full-width q-mt-xs q-gutter-none"):
                            if descs is not None:
                                with ui.row().classes("items-center justify-between full-width"):
                                    ui.label(f"📐 {n_descs} 記述子").classes(
                                        "text-caption" + (" text-cyan" if is_current else " text-grey")
                                    )
                                    if total_available > 0:
                                        ui.label(f"{pct}%").classes("text-caption text-grey")
                                ui.linear_progress(
                                    value=pct / 100, color="cyan" if is_current else "grey",
                                ).props("rounded instant-feedback").style("height: 4px;")
                            else:
                                ui.label("📐 全記述子（未制限）").classes("text-caption text-amber")
                                ui.linear_progress(
                                    value=1.0, color="amber",
                                ).props("rounded instant-feedback").style("height: 4px;")

                        # ── エンジンバッジ ──
                        if engines:
                            with ui.row().classes("q-gutter-xs q-mt-xs flex-wrap"):
                                for eng in engines:
                                    color = _ENGINE_BADGE_COLORS.get(eng, "grey")
                                    ui.badge(eng, color=color).props("dense outline").classes("text-xs")

                        # ── アクションボタン ──
                        with ui.row().classes("q-gutter-xs q-mt-xs justify-center full-width"):
                            if not is_current:
                                def _switch(n=set_name):
                                    state["current_set_name"] = n
                                    if sets[n].get("descriptors"):
                                        state["active_descriptors"] = list(sets[n]["descriptors"])
                                    ui.notify(f"🔄 セット「{n}」に切替", type="info")

                                ui.button(
                                    "▶ 切替", on_click=_switch,
                                ).props("unelevated size=sm no-caps color=cyan")
                            else:
                                def _save():
                                    active = state.get("active_descriptors", [])
                                    sets[current]["descriptors"] = list(active)
                                    ui.notify(
                                        f"💾 セット「{current}」に {len(active)} 記述子を保存",
                                        type="positive",
                                    )

                                ui.button(
                                    "💾 現在の選択を保存", on_click=_save,
                                ).props("unelevated size=sm no-caps color=teal")

                            # アクティブ切替
                            def _toggle(n=set_name, val=not is_active):
                                sets[n]["active"] = val
                                status = "有効" if val else "無効"
                                ui.notify(f"{'✅' if val else '⏸️'} 「{n}」を{status}に", type="info")

                            if is_active:
                                ui.button(
                                    icon="pause_circle_outline", on_click=_toggle,
                                ).props("flat round dense size=xs color=grey").tooltip("比較解析から除外")
                            else:
                                ui.button(
                                    icon="play_circle_outline", on_click=_toggle,
                                ).props("flat round dense size=xs color=green").tooltip("比較解析に含める")

        # ── フッター: アクティブセット数 ──
        active_sets = [n for n, info in sets.items() if info.get("active", True)]
        if len(active_sets) > 1:
            ui.separator().classes("q-my-xs")
            with ui.row().classes("items-center q-gutter-sm"):
                ui.icon("compare_arrows", color="amber")
                ui.label(
                    f"🔬 {len(active_sets)} セットが比較解析対象（アクティブ）"
                ).classes("text-caption text-amber")
