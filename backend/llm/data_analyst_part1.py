"""
backend/llm/data_analyst.py

データ読込後のLLM対話型分析プラン立案サービス。
GGUFProvider (Bonsai 8B) を使用し、多輪対話で分析方針を提案する。
記述子推奨機能付き。
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from backend.llm.descriptor_knowledge import (
    PROPERTY_CATEGORIES,
    build_descriptor_recommendation_prompt,
    find_matching_properties,
)

logger = logging.getLogger(__name__)


class LLMDataAnalyst:
    """
    データ分析対話を管理するサービスクラス。
    GGUFProviderを使用してLLM推論を実行する。
    """

    def __init__(self):
        self.conversation_history: list[dict] = []
        self.last_suggestions: dict = {}
        self._data_summary: str = ""
        self._state_snapshot: dict = {}
