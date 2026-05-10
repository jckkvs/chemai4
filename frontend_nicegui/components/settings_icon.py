"""
frontend_nicegui/components/settings_icon.py

ヘッダー右上に常設する設定ギアアイコン（控えめ・最小限）。
フルダイアログが必要な場合は settings_page.open_settings_dialog() を使用。
"""
from __future__ import annotations

from nicegui import ui
from backend.config.settings_manager import SettingsManager


def create_settings_icon() -> None:
    """
    ギアアイコンボタンを描画する。
    クリックするとインラインメニューで主要設定を素早く変更できる。
    """
    settings = SettingsManager.get_instance()

    with ui.button(icon="settings").props(
        'flat round dense size=sm color=grey aria-label="設定" id="header-settings-icon"'
    ).tooltip("⚙️ クイック設定"):
        with ui.menu().props("anchor='bottom right' self='top right'").style(
            "min-width: 320px;"
        ):
            # ── ヘッダー ──
            with ui.item().classes("q-px-md q-py-xs"):
                ui.label("⚙️ クイック設定").classes("text-subtitle2 text-bold")

            ui.separator()

            # ── HuggingFace トークン ──
            with ui.expansion("🤗 HuggingFace", icon="key").classes("full-width"):
                token_input = ui.input(
                    label="APIトークン",
                    password=True,
                    password_toggle_button=True,
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
                ).props("dense outlined dark").classes("full-width q-pa-sm")
                token_input.set_value(settings.get("huggingface", "token", "") or "")

                def _save_hf():
                    settings.set("huggingface", "token", token_input.value)
                    settings.apply_to_environment()
                    settings.save_config()
                    ui.notify("✅ HuggingFace トークンを保存しました", type="positive", timeout=2000)

                ui.button("💾 保存", on_click=_save_hf).props(
                    "flat dense color=cyan size=sm no-caps"
                ).classes("q-px-sm q-mb-sm")

            # ── プロキシ ──
            with ui.expansion("🌐 プロキシ", icon="public").classes("full-width"):
                proxy_switch = ui.switch(
                    "有効",
                    value=settings.get("proxy", "use_proxy", False),
                )
                http_input = ui.input(
                    label="HTTP プロキシ",
                    placeholder="http://proxy.example.com:8080",
                ).props("dense outlined dark").classes("full-width q-px-sm")
                http_input.set_value(settings.get("proxy", "http_proxy", "") or "")

                https_input = ui.input(
                    label="HTTPS プロキシ",
                    placeholder="https://proxy.example.com:8080",
                ).props("dense outlined dark").classes("full-width q-px-sm")
                https_input.set_value(settings.get("proxy", "https_proxy", "") or "")

                def _save_proxy():
                    settings.set("proxy", "use_proxy",  proxy_switch.value)
                    settings.set("proxy", "http_proxy", http_input.value)
                    settings.set("proxy", "https_proxy", https_input.value)
                    settings.apply_to_environment()
                    settings.save_config()
                    ui.notify("✅ プロキシ設定を保存しました", type="positive", timeout=2000)

                ui.button("💾 保存", on_click=_save_proxy).props(
                    "flat dense color=cyan size=sm no-caps"
                ).classes("q-px-sm q-mb-sm")

            # ── SSL ──
            with ui.expansion("🔒 SSL", icon="lock").classes("full-width"):
                verify_switch = ui.switch(
                    "SSL証明書を検証する",
                    value=settings.get("ssl", "verify", True),
                )
                ca_input = ui.input(
                    label="CAバンドルパス",
                    placeholder="/path/to/ca-bundle.crt",
                ).props("dense outlined dark").classes("full-width q-px-sm")
                ca_input.set_value(settings.get("ssl", "ca_bundle_path", "") or "")

                def _save_ssl():
                    settings.set("ssl", "verify", verify_switch.value)
                    settings.set("ssl", "ca_bundle_path", ca_input.value)
                    settings.apply_to_environment()
                    settings.save_config()
                    ui.notify("✅ SSL設定を保存しました", type="positive", timeout=2000)

                ui.button("💾 保存", on_click=_save_ssl).props(
                    "flat dense color=cyan size=sm no-caps"
                ).classes("q-px-sm q-mb-sm")

            ui.separator()

            # ── 接続テスト ──
            result_label = ui.label("").classes("text-caption text-grey-5 q-px-md")

            def _test_connection():
                result_label.set_text("テスト中...")
                r = settings.test_hf_connection()
                result_label.set_text(r["message"])
                ui.notify(
                    r["message"],
                    type="positive" if r["success"] else "negative",
                    timeout=3000,
                )

            with ui.row().classes("q-px-md q-py-xs q-gutter-sm items-center"):
                ui.button("🔌 接続テスト", on_click=_test_connection).props(
                    "outline dense color=cyan size=sm no-caps"
                )
                # 詳細設定ダイアログを開くリンク
                def _open_full():
                    from frontend_nicegui.pages.settings_page import open_settings_dialog
                    open_settings_dialog()

                ui.button("⚙️ 詳細設定...", on_click=_open_full).props(
                    "flat dense color=grey size=sm no-caps"
                )
