"""
frontend_common/components/settings_panel.py
フレームワーク非依存の設定スキーマ定義（Streamlit/NiceGUI 共用）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from backend.config.settings_manager import SettingsManager
from backend.config.auto_downloader import AutoDownloader


# ─────────────────────────────────────────────
# 設定セクションスキーマ
# ─────────────────────────────────────────────

SECTIONS: Dict[str, dict] = {
    "huggingface": {
        "title": "Hugging Face",
        "icon": "🤗",
        "fields": {
            "token": {
                "type": "password",
                "label": "APIトークン",
                "placeholder": "hf_xxx...",
                "help": "https://huggingface.co/settings/tokens で取得",
            },
            "cache_dir": {
                "type": "text",
                "label": "キャッシュディレクトリ",
                "placeholder": "~/.cache/huggingface",
                "help": "モデル・データの保存先（空=デフォルト）",
            },
        },
    },
    "proxy": {
        "title": "プロキシ設定",
        "icon": "🌐",
        "fields": {
            "use_proxy": {
                "type": "checkbox",
                "label": "プロキシを使用する",
                "default": False,
                "help": "企業ネットワーク等で必要な場合に有効化",
            },
            "http_proxy": {
                "type": "text",
                "label": "HTTP プロキシ",
                "placeholder": "http://proxy.example.com:8080",
                "conditional_on": "use_proxy",
            },
            "https_proxy": {
                "type": "text",
                "label": "HTTPS プロキシ",
                "placeholder": "https://proxy.example.com:8080",
                "conditional_on": "use_proxy",
            },
            "no_proxy": {
                "type": "text",
                "label": "プロキシ除外ホスト",
                "placeholder": "localhost,127.0.0.1",
                "default": "localhost,127.0.0.1",
            },
        },
    },
    "ssl": {
        "title": "SSL / TLS",
        "icon": "🔒",
        "fields": {
            "verify": {
                "type": "checkbox",
                "label": "SSL 証明書を検証する",
                "default": True,
                "help": "⚠️ オフにするとセキュリティリスクがあります",
            },
            "ca_bundle_path": {
                "type": "text",
                "label": "CA バンドルファイルパス",
                "placeholder": "/path/to/ca-bundle.crt",
                "help": "空の場合はシステム証明書を使用",
            },
        },
    },
    "download": {
        "title": "ダウンロード設定",
        "icon": "⬇️",
        "fields": {
            "auto_download_models": {
                "type": "checkbox",
                "label": "モデルを自動ダウンロード",
                "default": True,
            },
            "auto_download_datasets": {
                "type": "checkbox",
                "label": "データセットを自動ダウンロード",
                "default": True,
            },
            "preferred_source": {
                "type": "select",
                "label": "優先ソース",
                "options": ["huggingface", "modelscope", "local"],
                "default": "huggingface",
                "help": "modelscope は中国ミラーサイト（pip も Tsinghua ミラーを使用）",
            },
            "max_retries": {
                "type": "number",
                "label": "最大リトライ回数",
                "default": 3,
                "min": 0,
                "max": 10,
            },
            "timeout_seconds": {
                "type": "number",
                "label": "タイムアウト（秒）",
                "default": 300,
                "min": 30,
                "max": 3600,
            },
        },
    },
}


class SettingsPanelBase:
    """
    フレームワーク非依存の設定パネル基底クラス。
    Streamlit / NiceGUI それぞれで継承・実装する。
    """

    def __init__(self):
        self.settings = SettingsManager.get_instance()
        self.downloader = AutoDownloader(self.settings)
        self.sections = SECTIONS

    # ─── 共通ロジック ───────────────────────────

    def save(self) -> None:
        self.settings.apply_to_environment()
        self.settings.save_config()

    def test_connection(self) -> Dict[str, Any]:
        return self.settings.test_hf_connection()

    def get_dep_status(self) -> Dict[str, Dict[str, dict]]:
        return self.downloader.check_all_dependencies()

    def install_missing_packages(
        self,
        progress_cb: Optional[Any] = None,
    ) -> Dict[str, str]:
        return self.downloader.install_missing(progress_callback=progress_cb)
