"""
backend/pipeline/tasks/manager.py

計算リソース管理と非同期実行基盤。
現状は ProcessPoolExecutor を用いたインメモリキューと状態保持を提供し、
将来的な Celery / Redis などの導入を見越したアダプタとして機能する。
"""

import uuid
from typing import Callable, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, Future
import logging

logger = logging.getLogger(__name__)

class TaskManager:
    """
    アプリケーション全体で共有される軽量非同期タスクマネージャー。
    注意: サーバー再起動で状態は消滅するインメモリ設計。
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self, max_workers: int = 4):
        if self._initialized:
            return
            
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self._initialized = True
        logger.info(f"TaskManager initialized with {max_workers} workers.")
        
    def submit(self, func: Callable, *args, **kwargs) -> str:
        """
        タスクをキューに登録し、タスクIDを返す。
        
        Returns:
            str: UUID形式のタスクID
        """
        task_id = str(uuid.uuid4())
        
        # 状態登録
        self.tasks[task_id] = {
            "status": "queued",
            "result": None,
            "error": None
        }
        
        future = self.executor.submit(func, *args, **kwargs)
        future.add_done_callback(lambda f: self._on_task_done(task_id, f))
        
        return task_id
        
    def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        タスクの現在の状態を取得する。
        """
        if task_id not in self.tasks:
            return None
        return self.tasks[task_id]
        
    def _on_task_done(self, task_id: str, future: Future):
        """
        タスク完了時のコールバック。
        ステータスと結果を更新する。
        """
        try:
            result = future.result()
            self.tasks[task_id]["status"] = "success"
            self.tasks[task_id]["result"] = result
        except BaseException as e:
            logger.error(f"Task {task_id} failed: {e}")
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["error"] = str(e)
            
    def shutdown(self):
        """シャットダウンフック"""
        self.executor.shutdown(wait=False)

def get_task_manager() -> TaskManager:
    """設定済みのシングルトン TaskManager を取得する"""
    return TaskManager()
