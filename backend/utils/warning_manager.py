"""
backend/utils/warning_manager.py

システム拡張やUI上の設定において生じた矛盾や軽微なエラーを収集し、
処理を停止するのではなくWarningとして扱うための管理クラス。
"""
from typing import List, Dict

class WarningCollector:
    """
    システムのロード時や実行時に発生した警告を収集・管理するクラス。
    破壊的エラーを避け、なるべくGraceful Degradationを実現するために用いる。
    """
    def __init__(self):
        self.warnings: List[Dict[str, str]] = []

    def add(self, level: str, module: str = "Unknown", message: str = "", descriptor: str = None) -> None:
        """
        警告を追加する。
        
        Args:
            level: 'INFO', 'WARNING', 'ERROR' などの重要度
            module: 発生元のモジュール名やファイル名
            message: エラーや警告の詳細なメッセージ
            descriptor: 対象の記述子やコンポーネント名（任意）
        """
        warn_info = {
            "level": level,
            "module": module,
            "message": message,
        }
        if descriptor:
            warn_info["descriptor"] = descriptor
            
        self.warnings.append(warn_info)

    def get_all(self) -> List[Dict[str, str]]:
        """収集したすべての警告を返す"""
        return self.warnings

    def clear(self) -> None:
        """警告リストをクリアする"""
        self.warnings.clear()

    def has_warnings(self) -> bool:
        """警告が1つ以上あるかどうかを返す"""
        return len(self.warnings) > 0
