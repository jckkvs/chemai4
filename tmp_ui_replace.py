# -*- coding: utf-8 -*-
"""プリセットUI（P-B案: アコーディオン詳細表示）への改修スクリプト"""

fp = 'C:/Users/horie/chemai2/frontend_nicegui/components/descriptor_plugins_ui.py'
with open(fp, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

old_ui = '''                    # エンジン名のラベルマップ
                    _eng_label = {e["cls"]: e["label"] for e in _ENGINE_INFO}

                    border = "rgba(0,212,255,0.4)" if all_avail else "rgba(255,200,0,0.3)"
                    with ui.card().classes("full-width q-pa-sm q-mb-xs").style(
                        f"border: 1px solid {border}; border-radius: 8px;"
                        "background: rgba(0,20,40,0.3);"
                    ):
                        with ui.row().classes("items-center full-width justify-between"):
                            with ui.column().classes("q-gutter-none").style("flex: 1;"):
                                ui.label(preset_name).classes("text-body2 text-bold")
                                ui.label(preset_desc).classes("text-caption text-grey").style(
                                    "font-size: 0.7rem;"
                                )
                                # エンジンバッジ
                                with ui.row().classes("q-gutter-xs q-mt-xs"):
                                    for pe in preset_engines:
                                        lbl = _eng_label.get(pe, pe.replace("Adapter", ""))
                                        avl = _is_available(adapters, pe)
                                        ui.badge(
                                            lbl,
                                            color="teal" if avl else "grey",
                                        ).props("outline dense").style("font-size: 0.78rem;")
                                # 含まれる記述子のプレビュー
                                ui.label(
                                    f"📋 {n_preview}記述子: {preview_text}"
                                ).classes("text-caption text-grey-5").style("font-size: 0.82rem;")

                            with ui.row().classes("items-center q-gutter-xs"):
                                ui.badge(
                                    f"{n_avail}/{len(preset_engines)}",
                                    color="green" if all_avail else "amber",
                                ).props("outline")
                                ui.button(
                                    "適用", on_click=_apply_preset,
                                ).props(
                                    f"{'unelevated' if all_avail else 'outline'}"
                                    " size=sm no-caps color=cyan"
                                )'''

new_ui = '''                    # エンジン名のラベルマップ
                    _eng_label = {e["cls"]: e["label"] for e in _ENGINE_INFO}

                    border = "rgba(0,212,255,0.4)" if all_avail else "rgba(255,200,0,0.3)"
                    with ui.card().classes("full-width q-mb-md").style(
                        f"border: 1px solid {border}; border-radius: 8px;"
                        "background: rgba(0,20,40,0.3); padding: 0;"
                    ):
                        # --- P-B案 UI実装 ---
                        with ui.row().classes("items-center full-width justify-between q-pa-sm"):
                            with ui.column().classes("q-gutter-none").style("flex: 1;"):
                                ui.label(preset_name).classes("text-body1 text-bold")
                                ui.label(preset_desc).classes("text-caption text-grey")
                                # エンジンバッジ
                                with ui.row().classes("q-gutter-xs q-mt-xs"):
                                    for pe in preset_engines:
                                        lbl = _eng_label.get(pe, pe.replace("Adapter", ""))
                                        avl = _is_available(adapters, pe)
                                        ui.badge(
                                            lbl, color="teal" if avl else "grey",
                                        ).props("outline dense").style("font-size: 0.78rem;")

                            with ui.row().classes("items-center q-gutter-sm"):
                                # 厳選バッジ（「2/2」等の内部都合を排し、人間がわかる表示へ）
                                display_badge_text = f"{n_curated}個厳選" if n_curated > 0 else f"全{n_preview}個"
                                ui.badge(
                                    display_badge_text,
                                    color="green" if all_avail else "amber",
                                ).props("outline")
                                ui.button(
                                    "適用", on_click=_apply_preset,
                                ).props(
                                    f"{'unelevated' if all_avail else 'outline'}"
                                    " size=md no-caps color=cyan"
                                )
                        
                        # --- アコーディオン展開で透明性確保（研究者・数学者用） ---
                        with ui.expansion("内訳を表示", icon="view_list").classes("full-width bg-transparent"):
                            with ui.row().classes("q-gutter-xs q-pa-sm"):
                                _disp_limit = 50
                                for dp in _pv_non_fp[:_disp_limit]:
                                    _meaning = _catalog_meanings.get(dp, "")
                                    _tt = _meaning if _meaning else "詳細情報なし"
                                    ui.chip(dp, icon="science", color="cyan").props("outline size=sm").tooltip(_tt)
                                
                                for gl, bits in sorted(_pv_fp_groups.items(), key=lambda x: -len(x[1])):
                                    ui.chip(f"{gl}({len(bits)}bit)", icon="fingerprint", color="purple").props("outline size=sm")
                                
                                if len(_pv_non_fp) > _disp_limit:
                                    ui.chip(f"他 {len(_pv_non_fp) - _disp_limit} 個の機能...", color="grey").props("outline size=sm")'''

if old_ui in content:
    content = content.replace(old_ui, new_ui, 1)
    changes += 1
else:
    print("WARNING: Target UI block not found")

with open(fp, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done: {changes} changes applied")
