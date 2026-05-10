"""
frontend_nicegui/pages/settings_page.py
アプリ設定ページ（NiceGUI）

右上ギアアイコン → ダイアログ として表示。
ページルート /settings でも独立表示可能。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from nicegui import ui

from backend.config.settings_manager import SettingsManager
from backend.config.auto_downloader import AutoDownloader

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 設定フォームのスキーマ定義
# ─────────────────────────────────────────────
_SECTIONS: Dict[str, dict] = {
    "proxy": {
        "title": "プロキシ設定",
        "icon": "public",
        "color": "blue-6",
        "fields": [
            {
                "key": "use_proxy",
                "type": "toggle",
                "label": "プロキシを使用する",
                "help": "企業ネットワーク等でプロキシが必要な場合に有効化",
            },
            {
                "key": "http_proxy",
                "type": "text",
                "label": "HTTP プロキシ",
                "placeholder": "http://proxy.example.com:8080",
            },
            {
                "key": "https_proxy",
                "type": "text",
                "label": "HTTPS プロキシ",
                "placeholder": "https://proxy.example.com:8080",
            },
            {
                "key": "no_proxy",
                "type": "text",
                "label": "プロキシ除外ホスト",
                "placeholder": "localhost,127.0.0.1",
            },
        ],
    },
    "ssl": {
        "title": "SSL / TLS",
        "icon": "lock",
        "color": "green-6",
        "fields": [
            {
                "key": "verify",
                "type": "toggle",
                "label": "SSL 証明書を検証する",
                "help": "⚠️ オフにするとセキュリティリスクがあります",
            },
            {
                "key": "ca_bundle_path",
                "type": "text",
                "label": "CA バンドルファイルパス",
                "placeholder": "/path/to/ca-bundle.crt",
                "help": "空の場合はシステム証明書を使用",
            },
        ],
    },
    "download": {
        "title": "ダウンロード設定",
        "icon": "download",
        "color": "cyan-6",
        "fields": [
            {
                "key": "auto_download_models",
                "type": "toggle",
                "label": "モデルを自動ダウンロード",
            },
            {
                "key": "auto_download_datasets",
                "type": "toggle",
                "label": "データセットを自動ダウンロード",
            },
            {
                "key": "preferred_source",
                "type": "select",
                "label": "優先ソース",
                "options": ["huggingface", "modelscope", "local"],
                "help": "modelscope は中国ミラーサイト（pip も Tsinghua ミラーを使用）",
            },
            {
                "key": "max_retries",
                "type": "number",
                "label": "最大リトライ回数",
                "min": 0,
                "max": 10,
            },
            {
                "key": "timeout_seconds",
                "type": "number",
                "label": "タイムアウト（秒）",
                "min": 30,
                "max": 3600,
            },
        ],
    },
}


# ─────────────────────────────────────────────
# 設定ダイアログ（メイン実装）
# ─────────────────────────────────────────────

def open_settings_dialog() -> None:
    """設定ダイアログを開く（どこからでも呼べる共通エントリ）"""
    settings = SettingsManager.get_instance()

    with ui.dialog().props("maximized persistent") as dlg, ui.card().classes(
        "q-pa-none bg-dark full-width full-height"
    ).style("max-width: 900px; margin: auto; max-height: 90vh; overflow: hidden; border-radius: 16px;"):

        # ── ヘッダー ──
        with ui.row().classes(
            "items-center justify-between q-px-lg q-py-sm full-width"
        ).style("background: rgba(0,212,255,0.08); border-bottom: 1px solid rgba(255,255,255,0.1);"):
            with ui.row().classes("items-center q-gutter-sm"):
                ui.icon("settings", color="cyan").classes("text-h5")
                ui.label("設定").classes("text-h6 text-bold hero-gradient")
            ui.button(icon="close", on_click=dlg.close).props("flat round size=sm color=grey")

        # ── タブ ──
        with ui.tabs().classes("full-width q-px-md").props(
            "active-color=cyan indicator-color=cyan align=left dense"
        ) as tabs:
            for key, sec in _SECTIONS.items():
                ui.tab(key, icon=sec["icon"], label=sec["title"])
            # 化学モデル・LLM設定タブを追加
            ui.tab("chemical_models", icon="science", label="化学モデル")
            ui.tab("local_llm", icon="download", label="ローカルLLM")
            ui.tab("cloud_llm", icon="cloud", label="クラウドLLM")

        # ── 入力フォームのウィジェット保持 ──
        # { section_key: { field_key: widget } }
        widgets: Dict[str, Dict[str, Any]] = {}

        with ui.tab_panels(tabs, value=list(_SECTIONS.keys())[0]).classes(
            "full-width q-px-lg q-py-md"
        ).style("overflow-y: auto; max-height: 60vh;"):
            for section_key, sec in _SECTIONS.items():
                widgets[section_key] = {}
                with ui.tab_panel(section_key):
                    with ui.column().classes("full-width q-gutter-md"):
                        for field in sec["fields"]:
                            fkey = field["key"]
                            current = settings.get(section_key, fkey)
                            widget = _render_field(field, current)
                            if widget is not None:
                                widgets[section_key][fkey] = widget

            # 化学モデル管理パネル
            with ui.tab_panel("chemical_models"):
                from frontend_nicegui.pages.model_manager import render_chemical_models_content
                render_chemical_models_content()

            # ローカルLLM設定パネル
            with ui.tab_panel("local_llm"):
                from frontend_nicegui.pages.model_manager import render_local_llm_content
                render_local_llm_content()

            # クラウドLLM設定パネル
            with ui.tab_panel("cloud_llm"):
                from frontend_nicegui.pages.model_manager import render_cloud_llm_content
                render_cloud_llm_content()

        # ── フッターボタン ──
        with ui.row().classes(
            "items-center justify-between q-px-lg q-py-sm full-width"
        ).style("border-top: 1px solid rgba(255,255,255,0.1);"):

            # 接続テストボタン
            test_lbl = ui.label("").classes("text-caption text-grey-6")

            async def _test():
                test_lbl.set_text("テスト中...")
                result = await ui.run_javascript(
                    "0",  # ダミー – 実際はサーバーサイドで実行
                    timeout=1,
                    respond_timeout=1,
                )
                r = settings.test_hf_connection()
                test_lbl.set_text(r["message"])
                color = "positive" if r["success"] else "negative"
                ui.notify(r["message"], type=color)

            ui.button("🔌 接続テスト", on_click=_test).props("outline color=cyan size=sm no-caps")

            with ui.row().classes("q-gutter-sm"):
                ui.button("キャンセル", on_click=dlg.close).props("flat color=grey size=sm no-caps")

                async def _save(close: bool = False):
                    _apply_widgets_to_settings(widgets, settings)
                    try:
                        settings.apply_to_environment()
                        settings.save_config()
                        ui.notify("✅ 設定を保存しました", type="positive")
                        if close:
                            dlg.close()
                    except Exception as e:
                        ui.notify(f"❌ 保存失敗: {e}", type="negative")

                ui.button("保存", on_click=lambda: _save(False)).props(
                    "outline color=cyan size=sm no-caps"
                )
                ui.button("保存して閉じる", on_click=lambda: _save(True)).props(
                    "unelevated color=cyan size=sm no-caps"
                )

        # ── 依存パッケージチェック（アコーディオン）──
        with ui.expansion("📦 パッケージ管理", icon="inventory_2").classes(
            "full-width q-px-lg q-pb-sm"
        ).style("border-top: 1px solid rgba(255,255,255,0.08);"):
            _render_package_manager()

    dlg.open()


# ─────────────────────────────────────────────
# フィールドレンダリング
# ─────────────────────────────────────────────

def _render_field(field: dict, current_value: Any) -> Optional[Any]:
    """フィールドタイプに応じて入力ウィジェットを描画し返す"""
    ftype = field["type"]
    label = field.get("label", field["key"])
    help_text = field.get("help", "")

    with ui.card().classes("q-pa-sm glass-card full-width"):
        with ui.row().classes("items-center q-mb-xs"):
            ui.label(label).classes("text-caption text-bold text-grey-3")
            if help_text:
                ui.icon("info_outline", size="xs", color="grey-6").tooltip(help_text)

        widget: Any = None

        if ftype == "password":
            widget = ui.input(
                placeholder=field.get("placeholder", ""),
                password=True,
                password_toggle_button=True,
            ).classes("full-width").props("dense filled dark")
            if current_value:
                widget.set_value(str(current_value))

        elif ftype == "text":
            widget = ui.input(
                placeholder=field.get("placeholder", ""),
            ).classes("full-width").props("dense filled dark")
            if current_value:
                widget.set_value(str(current_value))

        elif ftype == "toggle":
            widget = ui.switch(value=bool(current_value))

        elif ftype == "select":
            options = field.get("options", [])
            widget = ui.select(
                options=options,
                value=current_value if current_value in options else options[0],
            ).classes("full-width").props("dense filled dark")

        elif ftype == "number":
            widget = ui.number(
                min=field.get("min"),
                max=field.get("max"),
                value=current_value if current_value is not None else field.get("min", 0),
            ).classes("full-width").props("dense filled dark")

    return widget


def _apply_widgets_to_settings(
    widgets: Dict[str, Dict[str, Any]],
    settings: SettingsManager,
) -> None:
    """ウィジェットの値を SettingsManager に書き戻す"""
    for section_key, fields in widgets.items():
        for field_key, widget in fields.items():
            try:
                val = widget.value
                settings.set(section_key, field_key, val)
            except Exception as e:
                logger.warning("設定値の取得失敗 [%s.%s]: %s", section_key, field_key, e)


# ─────────────────────────────────────────────
# パッケージマネージャー UI
# ─────────────────────────────────────────────

def _render_package_manager() -> None:
    """依存パッケージチェック＆インストール UI"""
    downloader = AutoDownloader()
    status_label = ui.label("チェックしていません").classes("text-caption text-grey-5")

    rows_holder: Dict[str, list] = {"rows": []}
    pkg_table = ui.table(
        columns=[
            {"name": "category", "label": "カテゴリー", "field": "category", "align": "left"},
            {"name": "package",  "label": "パッケージ",  "field": "package",  "align": "left"},
            {"name": "status",   "label": "状態",       "field": "status",   "align": "center"},
            {"name": "version",  "label": "バージョン",  "field": "version",  "align": "center"},
        ],
        rows=[],
    ).classes("full-width q-mt-sm").props("dense flat bordered dark virtual-scroll rows-per-page=0")

    def _refresh():
        status_label.set_text("チェック中...")
        dep_info = downloader.check_all_dependencies()
        new_rows = []
        for cat, pkgs in dep_info.items():
            for pname, info in pkgs.items():
                new_rows.append({
                    "category": cat,
                    "package": pname,
                    "status": "✅" if info["installed"] else "❌",
                    "version": info["version"] or "—",
                })
        pkg_table.rows = new_rows
        missing = sum(1 for r in new_rows if r["status"] == "❌")
        status_label.set_text(f"完了: {len(new_rows) - missing} / {len(new_rows)} パッケージ インストール済み")

    async def _install_missing():
        status_label.set_text("インストール中...")
        dep_info = downloader.check_all_dependencies()
        for cat, pkgs in dep_info.items():
            for pname, info in pkgs.items():
                if not info["installed"]:
                    spec = info.get("spec", pname)
                    ui.notify(f"インストール中: {spec}", timeout=2000)
                    ok, msg = downloader.install_package(spec)
                    if not ok:
                        ui.notify(f"❌ {pname}: {msg}", type="negative", timeout=5000)
        _refresh()

    with ui.row().classes("q-gutter-sm q-mb-sm"):
        ui.button("🔄 チェック", on_click=_refresh).props("outline color=cyan size=sm no-caps")
        ui.button("⬇️ 不足分をインストール", on_click=_install_missing).props(
            "unelevated color=cyan size=sm no-caps"
        )
        status_label

    pkg_table


# ─────────────────────────────────────────────
# NiceGUI ページルート（/settings から直接アクセス）
# ─────────────────────────────────────────────

def render_settings_page() -> None:
    """
    /settings ルート用エントリーポイント。
    ui.page("/settings") ではなく、main.py 側の tab_panel 内から呼ぶことも可。
    """
    settings = SettingsManager.get_instance()
    downloader = AutoDownloader(settings)

    ui.label("⚙️ 設定").classes("text-h5 text-bold hero-gradient q-mb-md")
    ui.label(
        f"設定ファイル: {settings.config_file}"
    ).classes("text-caption text-grey-6 q-mb-md")

    ui.separator()

    with ui.tabs().classes("full-width").props(
        "active-color=cyan indicator-color=cyan dense"
    ) as tabs:
        for key, sec in _SECTIONS.items():
            ui.tab(key, icon=sec["icon"], label=sec["title"])
        # 化学モデル・LLM設定タブを追加
        ui.tab("chemical_models", icon="science", label="化学モデル")
        ui.tab("local_llm", icon="download", label="ローカルLLM")
        ui.tab("cloud_llm", icon="cloud", label="クラウドLLM")

    widgets: Dict[str, Dict[str, Any]] = {}

    with ui.tab_panels(tabs, value=list(_SECTIONS.keys())[0]).classes("full-width q-mt-md"):
        for section_key, sec in _SECTIONS.items():
            widgets[section_key] = {}
            with ui.tab_panel(section_key):
                with ui.column().classes("full-width q-gutter-md"):
                    for field in sec["fields"]:
                        fkey = field["key"]
                        current = settings.get(section_key, fkey)
                        w = _render_field(field, current)
                        if w is not None:
                            widgets[section_key][fkey] = w

        # 化学モデル管理パネル
        with ui.tab_panel("chemical_models"):
            from frontend_nicegui.pages.model_manager import render_chemical_models_content
            render_chemical_models_content()

        # ローカルLLM設定パネル
        with ui.tab_panel("local_llm"):
            from frontend_nicegui.pages.model_manager import render_local_llm_content
            render_local_llm_content()

        # クラウドLLM設定パネル
        with ui.tab_panel("cloud_llm"):
            from frontend_nicegui.pages.model_manager import render_cloud_llm_content
            render_cloud_llm_content()

    ui.separator().classes("q-my-md")

    with ui.row().classes("q-gutter-sm"):
        async def _test():
            r = settings.test_hf_connection()
            ui.notify(r["message"], type="positive" if r["success"] else "negative")

        def _save():
            _apply_widgets_to_settings(widgets, settings)
            try:
                settings.apply_to_environment()
                settings.save_config()
                ui.notify("✅ 設定を保存しました", type="positive")
            except Exception as e:
                ui.notify(f"❌ 保存失敗: {e}", type="negative")

        ui.button("🔌 接続テスト", on_click=_test).props("outline color=cyan no-caps")
        ui.button("💾 保存", on_click=_save).props("unelevated color=cyan no-caps")

    ui.separator().classes("q-my-md")
    with ui.expansion("📦 パッケージ管理", icon="inventory_2").classes("full-width"):
        _render_package_manager()
