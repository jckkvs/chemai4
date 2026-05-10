"""
backend/chem/mix_rules.py

混合系の記述子加重平均ルール（mol% or wt%）を管理するモジュール。
全SMILES記述子に対するデフォルト推奨ルールを提供し、
ユーザーがカスタム設定ファイルでオーバーライドできる仕組みを持ちます。
"""
import json
import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

RuleType = Literal["mol", "wt"]

# デフォルトで一部の特徴量を wt% 加重平均とするキーワード
# 通常、分子レベルの特性（トポロジカル、電子状態、原子カウント）は mol% が理にかなう。
WT_KEYWORDS = [
    "density",
    "mass", 
    "specific_volume",
    "specific_heat",
    "wt_fraction",
]

_OVERRIDE_FILE = Path("chemai_data/mix_rules.json")

def get_default_rule(descriptor_name: str) -> RuleType:
    """
    記述子名から推奨の加重平均ルール（"mol" または "wt"）を推定する。
    """
    name_lower = descriptor_name.lower()
    for kw in WT_KEYWORDS:
        if kw in name_lower:
            return "wt"
    return "mol"


class MixRulesManager:
    """
    記述子名ごとの加重平均ルールを管理するクラス。
    ユーザーによるカスタムファイル(JSON)がある場合はそれを使用し、
    無い場合はデフォルトルール(mol主体)を返す。
    """
    def __init__(self, override_file: Path | str | None = None):
        if override_file is None:
            override_file = _OVERRIDE_FILE
        self.override_file = Path(override_file)
        self.user_rules: dict[str, RuleType] = {}
        self._load_overrides()

    def _load_overrides(self):
        if self.override_file.exists():
            try:
                with open(self.override_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.user_rules = data
            except Exception as e:
                logger.error(f"Failed to load mix_rules override file: {e}")

    def save_overrides(self):
        self.override_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.override_file, "w", encoding="utf-8") as f:
            json.dump(self.user_rules, f, indent=4, ensure_ascii=False)

    def get_rule(self, descriptor_name: str) -> RuleType:
        """
        記述子の加重平均ルールを取得する。
        ユーザー設定があれば優先し、なければデフォルトルールを返す。
        """
        if descriptor_name in self.user_rules:
            return self.user_rules[descriptor_name]
        return get_default_rule(descriptor_name)

    def set_rule(self, descriptor_name: str, rule: RuleType):
        """ルールの手動設定（保存は save_overrides を呼ぶこと）"""
        self.user_rules[descriptor_name] = rule

    def batch_get_rules(self, descriptor_names: list[str]) -> dict[str, RuleType]:
        return {n: self.get_rule(n) for n in descriptor_names}


# グローバルなシングルトンインスタンス
default_rules_manager = MixRulesManager()
