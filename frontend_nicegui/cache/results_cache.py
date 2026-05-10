"""
frontend_nicegui/cache/results_cache.py
解析結果（主にPlotly Figure等のレンダリングコストが高いオブジェクト）のキャッシュ管理
"""
from typing import Dict, Any, Optional

class ResultsCache:
    _instance: Optional["ResultsCache"] = None
    cache: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def is_ready(self) -> bool:
        return bool(self.cache)

    def update_item(self, key: str, value: Any):
        """個別のキャッシュアイテムを更新"""
        self.cache[key] = value

    def clear(self):
        """キャッシュをクリア"""
        self.cache.clear()

    def get(self, key: str) -> Any:
        """キャッシュアイテムを取得"""
        return self.cache.get(key)
