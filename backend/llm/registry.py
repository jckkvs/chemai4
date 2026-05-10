"""
backend/llm/registry.py

LLMプロバイダーのレジストリ。
登録・取得・一覧管理を担当する。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class LLMProviderRegistry:
    """
    LLMプロバイダーを名前で管理するレジストリ。

    新しいプロバイダー（OpenAI等）を追加するには:
        registry.register("openai", OpenAIProvider)
    """

    def __init__(self) -> None:
        self._providers: dict[str, type] = {}

    def register(self, name: str, cls: type) -> None:
        """プロバイダークラスを登録する。"""
        self._providers[name] = cls
        logger.debug(f"[LLMRegistry] registered provider: {name!r}")

    def get(self, name: str) -> "LLMProvider":
        """名前でプロバイダーインスタンスを取得する。"""
        if name not in self._providers:
            available = list(self._providers.keys())
            raise KeyError(
                f"LLMプロバイダー {name!r} は未登録です。"
                f"利用可能: {available}"
            )
        return self._providers[name]()

    def list_available(self) -> list[str]:
        """登録済みプロバイダー名の一覧を返す。"""
        return list(self._providers.keys())

    def list_all_with_status(self) -> list[dict]:
        """
        全プロバイダーの名前・利用可能状態を返す。
        UIでのプロバイダー選択に使用する。
        """
        result = []
        for name, cls in self._providers.items():
            try:
                instance = cls()
                result.append({
                    "name": name,
                    "display_name": getattr(instance, "display_name", name),
                    "is_available": instance.is_available,
                    "description": getattr(instance, "description", ""),
                })
            except Exception as e:
                result.append({
                    "name": name,
                    "display_name": name,
                    "is_available": False,
                    "description": f"初期化エラー: {e}",
                })
        return result
