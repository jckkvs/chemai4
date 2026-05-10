"""
初回起動検知と Bonsai 8B 自動ダウンロード管理モジュール。

アプリの初回起動時に Bonsai 8B (prism-ml/Bonsai-8B-gguf) を
HuggingFace Hub から自動ダウンロードし、以降の起動で使えるようにする。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from backend.llm.providers.gguf_provider import (
    GGUF_MODEL_CATALOG,
    download_model_async,
    is_model_downloaded,
    load_gguf_config,
    save_gguf_config,
)

logger = logging.getLogger(__name__)

_FIRST_LAUNCH_FLAG = Path(__file__).parent.parent.parent / ".first_launch_done"


def is_first_launch() -> bool:
    """初回起動かどうかを判定。"""
    return not _FIRST_LAUNCH_FLAG.exists()


def mark_first_launch_done() -> None:
    """初回起動完了をマーク。"""
    try:
        _FIRST_LAUNCH_FLAG.touch()
        logger.info("初回起動完了をマークしました")
    except Exception as e:
        logger.error("初回起動フラグの作成に失敗: %s", e)


def get_auto_download_model() -> dict | None:
    """自動ダウンロード対象モデルを取得（auto_download=True）。"""
    # CPU最適化モデルを優先
    for model in GGUF_MODEL_CATALOG:
        if model.get("cpu_optimized") and model.get("auto_download"):
            return model
    # 次に auto_download=True を探す
    for model in GGUF_MODEL_CATALOG:
        if model.get("auto_download", False):
            return model
    # フォールバック: 最初のCPU最適化モデル
    for model in GGUF_MODEL_CATALOG:
        if model.get("cpu_optimized"):
            return model
    return None


def is_auto_download_complete() -> bool:
    """自動ダウンロード対象モデルがキャッシュ済みか確認。"""
    model = get_auto_download_model()
    if model is None:
        return True  # 自動ダウンロード対象がない場合は完了とみなす
    return is_model_downloaded(model["id"], model.get("file", ""))


def trigger_auto_download(callback: Callable | None = None) -> None:
    """
    自動ダウンロードを開始する。

    Args:
        callback: 進捗通知用コールバック（省略時は標準出力）
                シグネチャ: callback(status, fraction, message)
    """
    model = get_auto_download_model()
    if model is None:
        logger.warning("自動ダウンロード対象モデルが見つかりません")
        return

    if is_model_downloaded(model["id"], model.get("file", "")):
        logger.info(f"モデル {model['id']} は既にキャッシュされています")
        mark_first_launch_done()
        return

    logger.info(f"初回起動: モデル {model['id']} の自動ダウンロードを開始します")

    # 設定からトークンを取得
    config = load_gguf_config()
    token = config.get("token") or ""

    # 設定にモデル情報を保存
    config["model_id"] = model["id"]
    config["filename"] = model.get("file", "")
    save_gguf_config(config)

    # 非同期ダウンロードを開始（別スレッド）
    def _on_progress(status, fraction, message):
        logger.info(f"[AutoDownload] {message}")
        if callback:
            callback(status, fraction, message)

    try:
        download_model_async(
            model_id=model["id"],
            filename=model.get("file", ""),
            token=token or None,
            on_progress=_on_progress,
        )
    except Exception as e:
        logger.error("自動ダウンロードの開始に失敗: %s", e)


def check_and_trigger_first_launch(
    callback: Callable | None = None,
    force: bool = False,
) -> bool:
    """
    初回起動をチェックして必要なら自動ダウンロードを開始する。

    Args:
        callback: 進捗通知用コールバック
        force: True の場合、フラグに関わらずダウンロードを試行

    Returns:
        True なら自動ダウンロードを開始した（または完了済み）
    """
    if not force and not is_first_launch():
        # 2回目以降の起動
        if is_auto_download_complete():
            return True
        # モデルが削除されていた場合は再ダウンロード
        logger.info("モデルが見つかりません。再ダウンロードを試行します")
        trigger_auto_download(callback)
        return False

    # 初回起動
    if is_auto_download_complete():
        mark_first_launch_done()
        return True

    trigger_auto_download(callback)
    return False
