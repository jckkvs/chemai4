"""
backend/config/auto_downloader.py
ライブラリ・HuggingFace モデルの自動ダウンロード＆インストール管理
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import logging

from .settings_manager import SettingsManager

logger = logging.getLogger(__name__)


class AutoDownloader:
    """ライブラリ・モデルの自動ダウンロード管理"""

    # カテゴリー別 必須パッケージ
    REQUIRED_PACKAGES: Dict[str, List[str]] = {
        "core": [
            "pandas>=2.0",
            "numpy>=1.24",
            "scikit-learn>=1.3",
        ],
        "ml": [
            "optuna>=3.0",
            "shap>=0.42",
            "xgboost>=2.0",
            "lightgbm>=4.0",
        ],
        "deep_learning": [
            "torch>=2.0",
            "transformers>=4.30",
            "datasets>=2.14",
        ],
        "chemistry": [
            "rdkit>=2023.0",
        ],
        "visualization": [
            "plotly>=5.15",
            "matplotlib>=3.7",
        ],
        "ui": [
            "nicegui>=1.4",
        ],
    }

    def __init__(self, settings_manager: Optional[SettingsManager] = None):
        self.settings = settings_manager or SettingsManager.get_instance()
        self._progress_callback: Optional[Callable[[str, float], None]] = None

    def set_progress_callback(self, callback: Callable[[str, float], None]) -> None:
        self._progress_callback = callback

    def _report(self, message: str, progress: float) -> None:
        if self._progress_callback:
            self._progress_callback(message, progress)
        logger.info("[%3.0f%%] %s", progress * 100, message)

    # ─────────────────────────────────────────────
    # パッケージ確認 / インストール
    # ─────────────────────────────────────────────

    def check_package_installed(self, package_spec: str) -> bool:
        """パッケージがインストール済みかつバージョン要件を満たすか確認"""
        try:
            import pkg_resources
            pkg_resources.require(package_spec)
            return True
        except Exception:
            return False

    def _get_package_version(self, pkg_name: str) -> Optional[str]:
        """インストール済みパッケージのバージョンを取得"""
        # pkg_name に >=, == 等が含まれている場合は除去
        bare = pkg_name.split(">=")[0].split("==")[0].split("<=")[0].strip()
        try:
            import pkg_resources
            return pkg_resources.get_distribution(bare).version
        except Exception:
            pass
        try:
            import importlib
            mod = importlib.import_module(bare.replace("-", "_"))
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return None

    def _build_pip_env(self) -> dict:
        """pip インストール用の環境変数辞書を構築"""
        env = dict(os.environ)
        if self.settings.get("proxy", "use_proxy", False):
            if hp := self.settings.get("proxy", "http_proxy", ""):
                env["HTTP_PROXY"] = hp
            if hp := self.settings.get("proxy", "https_proxy", ""):
                env["HTTPS_PROXY"] = hp
        env["PIP_NO_CACHE_DIR"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        if self.settings.get("download", "preferred_source") == "modelscope":
            env["PIP_INDEX_URL"] = "https://pypi.tuna.tsinghua.edu.cn/simple"
        return env

    def install_package(self, package_spec: str, upgrade: bool = False) -> Tuple[bool, str]:
        """
        単一パッケージをインストール。

        Returns:
            (success, message)
        """
        if package_spec.startswith("huggingface/"):
            model_name = package_spec[len("huggingface/"):]
            ok = self._download_hf_model(model_name)
            return ok, ("ダウンロード成功" if ok else "ダウンロード失敗")

        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(package_spec)

        # SSL 設定
        if not self.settings.get("ssl", "verify", True):
            cmd += ["--trusted-host", "pypi.org", "--trusted-host", "files.pythonhosted.org"]
        elif ca_path := self.settings.get("ssl", "ca_bundle_path", ""):
            if Path(ca_path).exists():
                cmd += ["--cert", ca_path]

        timeout = self.settings.get("download", "timeout_seconds", 300) or 300

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=self._build_pip_env(),
                timeout=timeout,
            )
            if result.returncode == 0:
                logger.info("インストール成功: %s", package_spec)
                return True, "インストール成功"
            else:
                msg = result.stderr.strip()[-500:] if result.stderr else "不明なエラー"
                logger.error("インストール失敗: %s\n%s", package_spec, msg)
                return False, msg
        except subprocess.TimeoutExpired:
            msg = f"タイムアウト ({timeout}秒)"
            logger.error("インストールタイムアウト: %s", package_spec)
            return False, msg
        except Exception as e:
            logger.error("インストールエラー: %s — %s", package_spec, e)
            return False, str(e)

    # ─────────────────────────────────────────────
    # HuggingFace モデルダウンロード
    # ─────────────────────────────────────────────

    def _download_hf_model(self, model_name: str) -> bool:
        """HuggingFace Hub からモデルを snapshot_download でキャッシュ"""
        try:
            from huggingface_hub import snapshot_download, login
        except ImportError:
            logger.error("huggingface_hub が未インストールです")
            return False

        try:
            if token := self.settings.get("huggingface", "token", ""):
                login(token=token, add_to_git_credential=False)
            cache_dir_str = self.settings.get("huggingface", "cache_dir", "")
            cache_dir = Path(cache_dir_str).expanduser() if cache_dir_str else None
            self.settings.get("download", "max_retries", 3)

            snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                local_files_only=False,
                max_workers=4,
            )
            logger.info("モデルダウンロード成功: %s", model_name)
            return True
        except Exception as e:
            logger.error("モデルダウンロード失敗: %s — %s", model_name, e)
            return False

    # ─────────────────────────────────────────────
    # カテゴリー別インストール / 全件チェック
    # ─────────────────────────────────────────────

    def install_category(self, category: str, upgrade: bool = False) -> Dict[str, str]:
        """カテゴリー内の全パッケージをインストール"""
        packages = self.REQUIRED_PACKAGES.get(category, [])
        results: Dict[str, str] = {}
        for i, pkg in enumerate(packages):
            self._report(f"{category}: {pkg}", i / max(len(packages), 1))
            if not self.check_package_installed(pkg) or upgrade:
                ok, _ = self.install_package(pkg, upgrade=upgrade)
                results[pkg] = "installed" if ok else "failed"
            else:
                results[pkg] = "already_installed"
        self._report(f"{category} 完了", 1.0)
        return results

    def check_all_dependencies(self) -> Dict[str, Dict[str, dict]]:
        """
        全カテゴリーのインストール状態を返す。

        Returns:
            {category: {pkg_name: {"installed": bool, "version": str|None}}}
        """
        out: Dict[str, Dict[str, dict]] = {}
        for category, packages in self.REQUIRED_PACKAGES.items():
            out[category] = {}
            for spec in packages:
                bare = spec.split(">=")[0].split("==")[0].strip()
                installed = self.check_package_installed(spec)
                out[category][bare] = {
                    "installed": installed,
                    "spec": spec,
                    "version": self._get_package_version(bare) if installed else None,
                }
        return out

    def install_missing(
        self,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, str]:
        """未インストールパッケージをすべてインストール"""
        if progress_callback:
            self.set_progress_callback(progress_callback)

        all_pkgs = [
            spec
            for pkgs in self.REQUIRED_PACKAGES.values()
            for spec in pkgs
            if not self.check_package_installed(spec)
        ]
        results: Dict[str, str] = {}
        for i, spec in enumerate(all_pkgs):
            self._report(f"インストール中: {spec}", i / max(len(all_pkgs), 1))
            ok, _ = self.install_package(spec)
            results[spec] = "installed" if ok else "failed"
        return results
