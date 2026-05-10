#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re

filepath = r'C:\Users\horie\chemai2_qwen\frontend_nicegui\components\descriptor_plugins_ui.py'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the section to replace - use a more flexible pattern
pattern = r'(# ── プロパティをチェックボックスとして表示 ──\s+_valid_dn = \[d\.name for d in rec\.descriptors if d\.name in all_descs\].*?)(# 全て選択されているかを判定)'

# Actually, let me use a simpler approach - find and replace line by line
lines = content.split('\n')
new_lines = []
skip_until = -1
in_target_section = False
replacement_done = False

i = 0
while i < len(lines):
    line = lines[i]

    # Detect start of the section to replace
    if '# ── プロパティをチェックボックスとして表示 ──' in line:
        in_target_section = True
        # Write the new header
        new_lines.append('                                # ── 推奨記述子の表示 ──')
        i += 1
        continue

    # Skip old content until we reach the checkbox row
    if in_target_section and not replacement_done:
        # Check if this is the end of the old section (the row with checkbox)
        if 'ui.checkbox(' in line and 'rec.target_name' in line:
            # Write the new implementation
            indent = '                                '
            new_lines.append(indent + '_valid_descs = [d for d in rec.descriptors if d.name in all_descs]')
            new_lines.append(indent + '_n_valid = len(_valid_descs)')
            new_lines.append(indent + '_n_total_rec = len(rec.descriptors)')
            new_lines.append(indent + '_valid_dn = [d.name for d in _valid_descs]')
            new_lines.append(indent + '_n_sel = sum(1 for dn in _valid_dn if dn in state.get("selected_descriptors", []))')
            new_lines.append('')
            new_lines.append(indent + '# 全て選択されているかを判定')
            new_lines.append(indent + '_is_all_on = (_n_sel == _n_valid and _n_valid > 0)')
            new_lines.append('')
            new_lines.append(indent + 'with ui.card().classes("full-width q-pa-xs q-ma-xs").style(')
            new_lines.append(indent + '    "background: rgba(255,255,255,0.03);"')
            new_lines.append(indent + '):')
            new_lines.append(indent + '    # ヘッダー行: チェックボックス + 名前 + バッジ')
            new_lines.append(indent + '    with ui.row().classes("full-width items-center q-gutter-x-sm"):')
            new_lines.append(indent + '        ui.checkbox(')
            new_lines.append(indent + '            rec.target_name,')
            new_lines.append(indent + '            value=_is_all_on,')
            new_lines.append(indent + '            on_change=lambda e, d=_valid_dn, r=rec: _toggle_rec(e.value, d, r)')
            new_lines.append(indent + '        ).props("dense")')
            new_lines.append('')
            new_lines.append(indent + '        ui.badge(')
            new_lines.append(indent + '            f"計算済 {_n_valid} / 推奨 {_n_total_rec} 個",')
            new_lines.append(indent + '            color="teal" if _n_valid == _n_total_rec else "amber",')
            new_lines.append(indent + '        ).props("outline dense")')
            new_lines.append('')
            new_lines.append(indent + '        if rec.category:')
            new_lines.append(indent + '            ui.badge(rec.category, color="blue").props("outline dense")')
            new_lines.append('')
            new_lines.append(indent + '    # サマリー')
            new_lines.append(indent + '    if rec.summary:')
            new_lines.append(indent + '        ui.label(rec.summary).classes("text-caption text-grey q-ml-lg")')
            new_lines.append('')
            new_lines.append(indent + '    # 全選択/全解除ボタン')
            new_lines.append(indent + '    if _valid_descs:')
            new_lines.append(indent + '        with ui.row().classes("q-gutter-xs q-ml-lg q-mt-xs"):')
            new_lines.append(indent + '            def _select_all_rec(val, names=_valid_dn, r=rec):')
            new_lines.append(indent + '                s = set(state.get("selected_descriptors", []))')
            new_lines.append(indent + '                if val:')
            new_lines.append(indent + '                    s.update(names)')
            new_lines.append(indent + '                    state["selected_descriptors"] = list(s)')
            new_lines.append(indent + '                    ui.notify(f"{r.target_name}: {len(names)}件を選択", type="positive")')
            new_lines.append(indent + '                else:')
            new_lines.append(indent + '                    s -= set(names)')
            new_lines.append(indent + '                    state["selected_descriptors"] = list(s)')
            new_lines.append(indent + '                    ui.notify(f"{r.target_name}: 全解除しました", type="info")')
            new_lines.append('')
            new_lines.append(indent + '            ui.button(')
            new_lines.append(indent + '                "全選択",')
            new_lines.append(indent + '                on_click=lambda e, n=_valid_dn, r=rec: _select_all_rec(True, n, r)')
            new_lines.append(indent + '            ).props("outline size=xs no-caps color=blue")')
            new_lines.append(indent + '            ui.button(')
            new_lines.append(indent + '                "全解除",')
            new_lines.append(indent + '                on_click=lambda e, n=_valid_dn, r=rec: _select_all_rec(False, n, r)')
            new_lines.append(indent + '            ).props("outline size=xs no-caps color=red")')
            new_lines.append('')
            new_lines.append(indent + '    # 個別記述子のチェックボックスリスト')
            new_lines.append(indent + '    if _valid_descs:')
            new_lines.append(indent + '        ui.separator().classes("q-my-xs")')
            new_lines.append(indent + '        ui.label("個別選択:").classes("text-caption text-grey q-ml-lg")')
            new_lines.append('')
            new_lines.append(indent + '        for desc in _valid_descs:')
            new_lines.append(indent + '            def _toggle_individual(val, dn=desc.name):')
            new_lines.append(indent + '                s = set(state.get("selected_descriptors", []))')
            new_lines.append(indent + '                if val:')
            new_lines.append(indent + '                    s.add(dn)')
            new_lines.append(indent + '                else:')
            new_lines.append(indent + '                    s.discard(dn)')
            new_lines.append(indent + '                state["selected_descriptors"] = list(s)')
            new_lines.append('')
            new_lines.append(indent + '            _is_sel = desc.name in state.get("selected_descriptors", [])')
            new_lines.append(indent + '            with ui.row().classes(')
            new_lines.append(indent + '                "items-center q-gutter-xs q-py-xs"')
            new_lines.append(indent + '            ).style("border-bottom: 1px solid rgba(255,255,255,0.03);"):')
            new_lines.append(indent + '                ui.checkbox(')
            new_lines.append(indent + '                    desc.name, value=_is_sel,')
            new_lines.append(indent + '                    on_change=lambda e, dn=desc.name: _toggle_individual(e.value, dn),')
            new_lines.append(indent + '                ).props("dense").style("min-width: 180px;")')
            new_lines.append(indent + '                ui.label(f"({desc.library})").classes("text-caption text-grey-6")')
            new_lines.append(indent + '                if desc.meaning:')
            new_lines.append(indent + '                    ui.label(desc.meaning).classes("text-caption text-grey").style(')
            new_lines.append(indent + '                        "font-size: 0.8rem;"')
            new_lines.append(indent + '                    )')

            replacement_done = True
            in_target_section = False
            # Also add the _toggle_rec function definition that was after the old section
            i += 1
            continue

        # Skip the old lines in the section
        if not replacement_done:
            i += 1
            continue

    # Normal line - add it
    new_lines.append(line)
    i += 1

# Also need to add the _toggle_rec function at the end
# Find where the except ImportError is and add the function before it
result = '\n'.join(new_lines)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(result)

print('Updated successfully!')
