#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Update the 目的変数 tab to show all descriptors with selection UI."""

filepath = r'C:\Users\horie\chemai2_qwen\frontend_nicegui\components\descriptor_plugins_ui.py'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# The section to replace starts after "_apply_rec" function and ends before "except ImportError"
# Let me find the exact location

# Find the start marker
start_marker = '# ── プロパティをチェックボックスとして表示 ──'
if start_marker not in content:
    # Try with different dash characters
    import re
    # Search for similar pattern
    match = re.search(r'# .*プロパティ.*チェックボックス.*表示', content)
    if match:
        start_marker = match.group(0)
        print(f"Found start marker: {repr(start_marker)}")
    else:
        print("Start marker not found!")
        # Let's find what's around line 1334
        lines = content.split('\n')
        print(f"Line 1334: {repr(lines[1333]) if len(lines) > 1333 else 'N/A'}")
        # Search for alternative
        for i, line in enumerate(lines):
            if '推奨記述子の表示' in line:
                print(f"Found '推奨記述子の表示' at line {i+1}: {repr(line)}")
                start_marker = line
                break

print(f"Using start marker: {repr(start_marker)}")

# Now find the end marker (the row with checkbox for rec.target_name)
# After the start marker, we need to find the end of the old section
lines = content.split('\n')
start_idx = None
end_idx = None

for i, line in enumerate(lines):
    if start_marker in line:
        start_idx = i
        break

if start_idx is not None:
    # Find the end of the old section (the except ImportError line or the next major section)
    for i in range(start_idx + 1, min(start_idx + 100, len(lines))):
        if 'except ImportError:' in lines[i]:
            end_idx = i
            break

    print(f"Start: {start_idx}, End: {end_idx}")

    if end_idx is not None:
        # Build the new content
        # Keep everything before start_idx
        new_lines = lines[:start_idx]

        # Add the new implementation
        indent = '                                '
        new_lines.extend([
            indent + '# ── 推奨記述子の表示 ──',
            indent + '_valid_descs = [d for d in rec.descriptors if d.name in all_descs]',
            indent + '_n_valid = len(_valid_descs)',
            indent + '_n_total_rec = len(rec.descriptors)',
            indent + '_valid_dn = [d.name for d in _valid_descs]',
            indent + '_n_sel = sum(1 for dn in _valid_dn if dn in state.get("selected_descriptors", []))',
            '',
            indent + '# 全て選択されているかを判定',
            indent + '_is_all_on = (_n_sel == _n_valid and _n_valid > 0)',
            '',
            indent + 'with ui.card().classes("full-width q-pa-xs q-ma-xs").style(',
            indent + '    "background: rgba(255,255,255,0.03);"',
            indent + '):',
            indent + '    # ヘッダー行: チェックボックス + 名前 + バッジ',
            indent + '    with ui.row().classes("full-width items-center q-gutter-x-sm"):',
            indent + '        ui.checkbox(',
            indent + '            rec.target_name,',
            indent + '            value=_is_all_on,',
            indent + '            on_change=lambda e, d=_valid_dn, r=rec: _toggle_rec(e.value, d, r)',
            indent + '        ).props("dense")',
            '',
            indent + '        ui.badge(',
            indent + '            f"計算済 {_n_valid} / 推奨 {_n_total_rec} 個",',
            indent + '            color="teal" if _n_valid == _n_total_rec else "amber",',
            indent + '        ).props("outline dense")',
            '',
            indent + '        if rec.category:',
            indent + '            ui.badge(rec.category, color="blue").props("outline dense")',
            '',
            indent + '    # サマリー',
            indent + '    if rec.summary:',
            indent + '        ui.label(rec.summary).classes("text-caption text-grey q-ml-lg")',
            '',
            indent + '    # 全選択/全解除ボタン',
            indent + '    if _valid_descs:',
            indent + '        with ui.row().classes("q-gutter-xs q-ml-lg q-mt-xs"):',
            indent + '            def _select_all_rec(val, names=_valid_dn, r=rec):',
            indent + '                s = set(state.get("selected_descriptors", []))',
            indent + '                if val:',
            indent + '                    s.update(names)',
            indent + '                    state["selected_descriptors"] = list(s)',
            indent + '                    ui.notify(f"{r.target_name}: {len(names)}件を選択", type="positive")',
            indent + '                else:',
            indent + '                    s -= set(names)',
            indent + '                    state["selected_descriptors"] = list(s)',
            indent + '                    ui.notify(f"{r.target_name}: 全解除しました", type="info")',
            '',
            indent + '            ui.button(',
            indent + '                "全選択",',
            indent + '                on_click=lambda e, n=_valid_dn, r=rec: _select_all_rec(True, n, r)',
            indent + '            ).props("outline size=xs no-caps color=blue")',
            indent + '            ui.button(',
            indent + '                "全解除",',
            indent + '                on_click=lambda e, n=_valid_dn, r=rec: _select_all_rec(False, n, r)',
            indent + '            ).props("outline size=xs no-caps color=red")',
            '',
            indent + '    # 個別記述子のチェックボックスリスト',
            indent + '    if _valid_descs:',
            indent + '        ui.separator().classes("q-my-xs")',
            indent + '        ui.label("個別選択:").classes("text-caption text-grey q-ml-lg")',
            '',
            indent + '        for desc in _valid_descs:',
            indent + '            def _toggle_individual(val, dn=desc.name):',
            indent + '                s = set(state.get("selected_descriptors", []))',
            indent + '                if val:',
            indent + '                    s.add(dn)',
            indent + '                else:',
            indent + '                    s.discard(dn)',
            indent + '                state["selected_descriptors"] = list(s)',
            '',
            indent + '            _is_sel = desc.name in state.get("selected_descriptors", [])',
            indent + '            with ui.row().classes(',
            indent + '                "items-center q-gutter-xs q-py-xs"',
            indent + '            ).style("border-bottom: 1px solid rgba(255,255,255,0.03);"):',
            indent + '                ui.checkbox(',
            indent + '                    desc.name, value=_is_sel,',
            indent + '                    on_change=lambda e, dn=desc.name: _toggle_individual(e.value, dn),',
            indent + '                ).props("dense").style("min-width: 180px;")',
            indent + '                ui.label(f"({desc.library})").classes("text-caption text-grey-6")',
            indent + '                if desc.meaning:',
            indent + '                    ui.label(desc.meaning).classes("text-caption text-grey").style(',
            indent + '                        "font-size: 0.8rem;"',
            indent + '                    )',
            '',
        ])

        # Add the _toggle_rec function definition
        new_lines.append(indent + 'def _toggle_rec(val, ds=_valid_dn, r=rec):')
        new_lines.append(indent + '    s = set(state.get("selected_descriptors", []))')
        new_lines.append(indent + '    if val:')
        new_lines.append(indent + '        s.update(ds)')
        new_lines.append(indent + '        ui.notify(f"{r.target_name}: {len(ds)}個の記述子を追加", type="positive")')
        new_lines.append(indent + '    else:')
        new_lines.append(indent + '        s -= set(ds)')
        new_lines.append(indent + '    state["selected_descriptors"] = list(s)')
        new_lines.append('')

        # Add the rest of the file (from end_idx onwards)
        new_lines.extend(lines[end_idx:])

        # Write the updated content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))

        print("Update completed successfully!")
    else:
        print("Could not find the end of the section to replace")
else:
    print("Start marker not found in the file")
