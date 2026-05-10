# -*- coding: utf-8 -*-
"""_refresh_tabs修正: loadタブを除外して他タブだけリフレッシュ"""

fp = 'C:/Users/horie/chemai2/frontend_nicegui/components/data_tab.py'
with open(fp, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# 1. _refresh_tabsを修正: loadタブを除外
old = 'state["_refresh_tabs"] = lambda: [rendered.update({k: False}) for k in rendered]'
new = '''# loadタブは除外（サンプルデータ読み込み後にプレビューが消えるのを防止）
    state["_refresh_tabs"] = lambda: [rendered.update({k: False}) for k in rendered if k != "load"]'''

if old in content:
    content = content.replace(old, new, 1)
    changes += 1
else:
    print("WARNING: _refresh_tabs target not found")

with open(fp, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done: {changes} changes")
