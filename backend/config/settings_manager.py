"""
backend/config/settings_manager.py
HuggingFace トークン・プロキシ・SSL・ダウンロード設定の一元管理
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SettingsManager:
    """アプリケーション設定の一元管理（シングルトン）"""

    DEFAULT_CONFIG_FILE = Path.home() / ".chemai" / "config.json"

    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or self.DEFAULT_CONFIG_FILE
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self._config = self._load_config()

    # ─────────────────────────────────────────────
    # ロード / セーブ
    # ─────────────────────────────────────────────

    def _default_config(self) -> Dict[str, Any]:
        return {
            "huggingface": {
                "token": "",
                "cache_dir": "~/.cache/huggingface",
            },
            "llm": {
                "provider": "gguf",
                "api_key": "",
                "api_url": "https://api.anthropic.com",
                "local_model_path": "",
                "local_models_dir": "models/llm",
                "auto_detect_models": True,
            },
            "proxy": {
                "http_proxy": "",
                "https_proxy": "",
                "no_proxy": "localhost,127.0.0.1",
                "use_proxy": False,
            },
            "ssl": {
                "verify": True,
                "ca_bundle_path": "",
                "ssl_cert_file": "",
            },
            "download": {
                "auto_download_models": True,
                "auto_download_datasets": True,
                "preferred_source": "huggingface",
                "max_retries": 3,
                "timeout_seconds": 300,
            },
            "ui": {
                "theme": "coolwarm_balanced",
                "language": "ja",
            },
        }

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み（存在しなければデフォルトを返す）"""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                # デフォルトにないキーをマージ（後方互換）
                default = self._default_config()
                for section, values in default.items():
                    if section not in loaded:
                        loaded[section] = values
                    else:
                        for k, v in values.items():
                            loaded[section].setdefault(k, v)
                return loaded
            except Exception as e:
                logger.warning(f"設定ファイル読み込み失敗: {e} — デフォルト設定を使用")
        return self._default_config()

    def save_config(self) -> None:
        """設定をJSONファイルに保存"""
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            logger.info(f"設定を保存しました: {self.config_file}")
        except Exception as e:
            logger.error(f"設定保存失敗: {e}")
            raise

    # ─────────────────────────────────────────────
    # Get / Set
    # ─────────────────────────────────────────────

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._config.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value

    def get_section(self, section: str) -> Dict[str, Any]:
        return dict(self._config.get(section, {}))

    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        if section not in self._config:
            self._config[section] = {}
        self._config[section].update(values)

    # ─────────────────────────────────────────────
    # 環境変数への適用
    # ─────────────────────────────────────────────

    def apply_to_environment(self) -> None:
        """保存済み設定を OS 環境変数に反映する"""
        # HuggingFace
        hf_token = self.get("huggingface", "token", "")
        hf_cache = self.get("huggingface", "cache_dir", "")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        if hf_cache:
            expanded = str(Path(hf_cache).expanduser())
            os.environ["HF_HOME"] = expanded
            os.environ["TRANSFORMERS_CACHE"] = expanded

        # Proxy
        if self.get("proxy", "use_proxy", False):
            if http_proxy := self.get("proxy", "http_proxy", ""):
                os.environ["HTTP_PROXY"] = http_proxy
            if https_proxy := self.get("proxy", "https_proxy", ""):
                os.environ["HTTPS_PROXY"] = https_proxy
            if no_proxy := self.get("proxy", "no_proxy", ""):
                os.environ["NO_PROXY"] = no_proxy
        else:
            for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
                os.environ.pop(key, None)

        # SSL
        if not self.get("ssl", "verify", True):
            os.environ["REQUESTS_CA_BUNDLE"] = ""
            os.environ["CURL_CA_BUNDLE"] = ""
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except ImportError:
                pass
        else:
            ca_path = self.get("ssl", "ca_bundle_path", "")
            if ca_path and Path(ca_path).exists():
                os.environ["REQUESTS_CA_BUNDLE"] = ca_path
                os.environ["CURL_CA_BUNDLE"] = ca_path

    # ─────────────────────────────────────────────
    # requests 用 kwargs 生成
    # ─────────────────────────────────────────────

    def get_requests_kwargs(self) -> Dict[str, Any]:
        """requests ライブラリに渡す共通キーワード引数"""
        kwargs: Dict[str, Any] = {}
        if not self.get("ssl", "verify", True):
            kwargs["verify"] = False
        else:
            ca_path = self.get("ssl", "ca_bundle_path", "")
            if ca_path and Path(ca_path).exists():
                kwargs["verify"] = ca_path

        if self.get("proxy", "use_proxy", False):
            proxies = {}
            if http_proxy := self.get("proxy", "http_proxy", ""):
                proxies["http"] = http_proxy
            if https_proxy := self.get("proxy", "https_proxy", ""):
                proxies["https"] = https_proxy
            if proxies:
                kwargs["proxies"] = proxies

        return kwargs

    # ─────────────────────────────────────────────
    # 接続テスト
    # ─────────────────────────────────────────────

    def test_hf_connection(self) -> Dict[str, Any]:
        """HuggingFace API への接続テスト"""
        try:
            import requests
        except ImportError:
            return {"success": False, "status_code": 0, "message": "requests ライブラリが未インストールです"}

        url = "https://huggingface.co/api/models?limit=1"
        try:
            kwargs = self.get_requests_kwargs()
            headers: Dict[str, str] = {}
            if token := self.get("huggingface", "token", ""):
                headers["Authorization"] = f"Bearer {token}"
            resp = requests.get(url, headers=headers, timeout=10, **kwargs)
            ok = resp.status_code == 200
            return {
                "success": ok,
                "status_code": resp.status_code,
                "message": "✅ 接続成功" if ok else f"⚠️ HTTP {resp.status_code}",
            }
        except Exception as e:
            return {"success": False, "status_code": 0, "message": f"❌ 接続失敗: {e}"}

    # ─────────────────────────────────────────────
    # シングルトン
    # ─────────────────────────────────────────────

    _instance: Optional["SettingsManager"] = None

    @classmethod
    def get_instance(cls) -> "SettingsManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """テスト用: シングルトンをリセット"""
        cls._instance = None
