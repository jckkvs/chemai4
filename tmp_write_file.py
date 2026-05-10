"""Script to create the complete correct data_analyst.py file."""
import os

# Build the file content as a list of strings
# Use \\n to represent the two characters \ and n in the output file
lines = []

# Header
lines.append('"""')
lines.append('backend/llm/data_analyst.py')
lines.append('')
lines.append('データ読込後のLLM対話型分析プラン立案サービス。')
lines.append('GGUFProvider (Bonsai 8B) を使用し、多輪対話で分析方針を提案する。')
lines.append('記述子推奨機能付き。')
lines.append('"""')
lines.append('')
lines.append('from __future__ import annotations')
lines.append('')
lines.append('import json')
lines.append('import logging')
lines.append('from typing import Any')
lines.append('')
lines.append('import pandas as pd')
lines.append('')
lines.append('from backend.llm.descriptor_knowledge import (')
lines.append('    PROPERTY_CATEGORIES,')
lines.append('    build_descriptor_recommendation_prompt,')
lines.append('    find_matching_properties,')
lines.append(')')
lines.append('')
lines.append('logger = logging.getLogger(__name__)')
lines.append('')
lines.append('')
lines.append('class LLMDataAnalyst:')
lines.append('    """')
lines.append('    データ分析対話を管理するサービスクラス。')
lines.append('    GGUFProviderを使用してLLM推論を実行する。')
lines.append('    """')
lines.append('')
lines.append('    def __init__(self):')
lines.append('        self.conversation_history: list[dict] = []')
lines.append('        self.last_suggestions: dict = {}')
lines.append('        self._data_summary: str = ""')
lines.append('        self._state_snapshot: dict = {}')
lines.append('')

# Write what we have so far
output_path = 'backend/llm/data_analyst.py'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f"Part 1 written. Total lines: {len(lines)}")
