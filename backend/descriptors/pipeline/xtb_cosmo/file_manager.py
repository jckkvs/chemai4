"""
backend/descriptors/pipeline/xtb_cosmo/file_manager.py

一時ファイル・ディレクトリの安全な管理。
"""
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class TempFileManager:
    def __init__(self, base_dir: Optional[Path] = None, keep_on_error: bool = False):
        self.base_dir = base_dir or Path(tempfile.mkdtemp(prefix="chemai_pipeline_"))
        self.keep_on_error = keep_on_error
        self._created_dirs: Dict[str, Path] = {}
        self._cleanup_done = False
    
    def create_subdir(self, name: str) -> Path:
        if name in self._created_dirs:
            return self._created_dirs[name]
        
        subdir = self.base_dir / name
        subdir.mkdir(parents=True, exist_ok=True)
        self._created_dirs[name] = subdir
        return subdir
    
    def create_temp_file(self, suffix: str = "", prefix: str = "") -> Path:
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=self.base_dir)
        Path(path).chmod(0o644)
        return Path(path)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cleanup_done:
            return False
            
        if exc_type is not None and self.keep_on_error:
            logger.info(f"エラー発生のため一時ファイルを保持: {self.base_dir}")
            self._cleanup_done = True
            return False
            
        try:
            if self.base_dir.exists():
                shutil.rmtree(self.base_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"一時ディレクトリの削除に失敗: {e}")
            
        self._cleanup_done = True
        return False
