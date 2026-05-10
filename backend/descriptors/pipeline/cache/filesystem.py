"""
backend/descriptors/pipeline/cache/filesystem.py

ファイルシステムベースのキャッシュ
"""
import json
import hashlib
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class FileSystemCache:
    def __init__(self, cache_dir: str, ttl_hours: int = 168, max_cache_size_bytes: int = 10 * 1024 * 1024 * 1024):
        self.cache_dir = Path(cache_dir)
        self.ttl = timedelta(hours=ttl_hours)
        self.max_cache_size_bytes = max_cache_size_bytes
        self.entries_dir = self.cache_dir / "entries"
        self.metadata_file = self.cache_dir / "metadata.json"
        
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        self._load_metadata()
    
    def _load_metadata(self):
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    self._metadata = json.load(f)
            except Exception:
                self._metadata = {"created": datetime.now().isoformat(), "count": 0}
        else:
            self._metadata = {"created": datetime.now().isoformat(), "count": 0}
            self._save_metadata()
            
    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self._metadata, f, indent=2)
            
    def _make_path(self, key: str) -> Path:
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.entries_dir / f"{key_hash}.json"
        
    def get(self, key: str) -> Optional[Any]:
        path = self._make_path(key)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                entry = json.load(f)
            created = datetime.fromisoformat(entry["created"])
            if datetime.now() - created > self.ttl:
                path.unlink(missing_ok=True)
                return None
            return entry["value"]
        except Exception as e:
            logger.warning(f"キャッシュ読み込みエラー {key}: {e}")
            return None
            
    def set(self, key: str, value: Any):
        path = self._make_path(key)
        entry = {
            "key": key,
            "value": value,
            "created": datetime.now().isoformat(),
            "version": "1.0",
        }
        try:
            with open(path, 'w') as f:
                json.dump(entry, f)
            self._metadata["count"] = self._metadata.get("count", 0) + 1
            self._metadata["last_updated"] = datetime.now().isoformat()
            self._save_metadata()
            self._enforce_size_limit()
        except Exception as e:
            logger.warning(f"キャッシュ保存エラー {key}: {e}")
            
    def _enforce_size_limit(self):
        """ディスク容量が上限を超えている場合、古いエントリを削除する(LRU)"""
        # 現在の合計サイズを計算
        entries = []
        total_size = 0
        for entry_file in self.entries_dir.glob("*.json"):
            if not entry_file.is_file():
                continue
            stat = entry_file.stat()
            size = stat.st_size
            total_size += size
            # modification time でソート (Windowsではatimeが更新されない場合があるため)
            entries.append((entry_file, size, stat.st_mtime))
            
        if total_size <= self.max_cache_size_bytes:
            return
            
        # 古いものから削除して上限の80%まで減らす
        target_size = self.max_cache_size_bytes * 0.8
        # st_atime(アクセス時刻)の昇順(古い順)でソート
        entries.sort(key=lambda x: x[2])
        
        deleted_count = 0
        for entry_file, size, _atime in entries:
            try:
                entry_file.unlink()
                total_size -= size
                deleted_count += 1
                if total_size <= target_size:
                    break
            except Exception:
                pass
                
        if deleted_count > 0:
            logger.info(f"キャッシュ容量制限のため {deleted_count} 件削除し、使用量を削減しました。")
            self._metadata["count"] = max(0, self._metadata.get("count", 0) - deleted_count)
            self._save_metadata()
            
    def clear(self, older_than_hours: Optional[int] = None):
        cutoff = datetime.now() - timedelta(hours=older_than_hours) if older_than_hours else None
        deleted = 0
        for entry_file in self.entries_dir.glob("*.json"):
            try:
                with open(entry_file) as f:
                    entry = json.load(f)
                created = datetime.fromisoformat(entry["created"])
                if cutoff is None or created < cutoff:
                    entry_file.unlink()
                    deleted += 1
            except Exception:
                continue
        logger.info(f"キャッシュクリア: {deleted} 件削除")
        self._metadata["count"] = max(0, self._metadata.get("count", 0) - deleted)
        self._save_metadata()
