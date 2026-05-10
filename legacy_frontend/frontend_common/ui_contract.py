"""
frontend_common/ui_contract.py

複数のフロントエンド（NiceGUI, Streamlit, Django/HTMX）において
同一のJSON Schemaを解釈し、同じ体験を提供するためのインターフェース。
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

class UIContract(ABC):
    """フレームワーク横断UI仕様"""
    
    @abstractmethod
    def render_auto_form(self, schema: Dict, vals: Dict, on_change: Callable) -> Any:
        """JSON Schemaから動的フォームを描画し、バインディングを行う"""
        pass
        
    @abstractmethod
    def render_progress(self, task_id: str, poll_url: str = "") -> Any:
        """重いタスク（XTB等）の非同期進捗バーを描画する"""
        pass
        
    @abstractmethod
    def emit_state(self) -> Dict:
        """現在のフォームの入力状態を（必要に応じて）ディクショナリで返す"""
        pass
