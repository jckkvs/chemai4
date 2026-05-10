"""Script to add descriptor recommendation button to main.py."""
# Read the file
with open('frontend_nicegui/main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add import for the dialog function
import_line = 'from frontend_nicegui.ui.descriptor_recommendation_dialog import open_descriptor_recommendation_dialog'
if import_line not in content:
    # Add after the nicegui import
    old = 'from nicegui import ui, app'
    new = f'{old}\n{import_line}'
    content = content.replace(old, new)
    print('Added import')

# Add the button after the settings button
# The settings button is around line 717
old_button = '''        ui.button(icon="settings", on_click=_open_settings).props(
            'flat round size=sm color=grey aria-label="設定" id="sidebar-settings-btn"'
        ).tooltip("⚙️ 設定")'''

new_button = '''        ui.button(icon="settings", on_click=_open_settings).props(
            'flat round size=sm color=grey aria-label="設定" id="sidebar-settings-btn"'
        ).tooltip("⚙️ 設定")
        ui.button(icon="psychology", on_click=_open_descriptor_recommendation).props(
            'flat round size=sm color=grey aria-label="記述子推奨" id="sidebar-descriptor-rec-btn"'
        ).tooltip("🧠 記述子推奨 (LLM)")'''

if old_button in content:
    content = content.replace(old_button, new_button)
    print('Added descriptor recommendation button')
else:
    print('Could not find settings button, trying alternative...')
    # Try to find the exact text
    idx = content.find('sidebar-settings-btn')
    if idx != -1:
        print(f'Found settings button at index {idx}')
        # Find the end of the tooltip
        end_idx = content.find('.tooltip(', idx)
        if end_idx != -1:
            # Find the closing paren
            paren_count = 0
            i = end_idx + len('.tooltip(')
            while i < len(content):
                if content[i] == '(':
                    paren_count += 1
                elif content[i] == ')':
                    if paren_count == 0:
                        break
                    paren_count -= 1
                i += 1
            # Insert after this
            new_code = ''')\\n        ui.button(icon="psychology", on_click=_open_descriptor_recommendation).props(\\n            'flat round size=sm color=grey aria-label="記述子推奨" id="sidebar-descriptor-rec-btn"\\n        ).tooltip("🧠 記述子推奨 (LLM)")'''
            content = content[:i+1] + new_code + content[i+1:]
            print('Added button using alternative method')

# Add the callback function
callback_func = '''

def _open_descriptor_recommendation():
    """記述子推奨ダイアログを開く。"""
    from frontend_nicegui.ui.descriptor_recommendation_dialog import open_descriptor_recommendation_dialog
    from backend.llm.data_analyst import get_data_analyst

    # Get current state
    state = {
        "df": getattr(app, 'current_df', None),
        "target_col": getattr(app, 'current_target_col', ''),
        "smiles_col": getattr(app, 'current_smiles_col', ''),
        "exclude_cols": getattr(app, 'current_exclude_cols', []),
    }

    def on_apply(selected_descriptors):
        """選択された記述子を適用する。"""
        app.current_selected_descriptors = selected_descriptors
        from nicegui import ui
        ui.notify(f"{len(selected_descriptors)}個の記述子を選択しました", type="positive")

    open_descriptor_recommendation_dialog(state, on_apply)
'''

if '_open_descriptor_recommendation' not in content:
    # Add before the main entry point or after other callbacks
    if 'def _open_settings():' in content:
        content = content.replace('def _open_settings():', callback_func + '\\ndef _open_settings():')
        print('Added callback function')
    else:
        print('Could not find where to add callback')

# Write back
with open('frontend_nicegui/main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done!')
