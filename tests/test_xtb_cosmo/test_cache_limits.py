"""
tests/test_xtb_cosmo/test_cache_limits.py
"""
import pytest
import time
from pathlib import Path
from backend.descriptors.pipeline.cache.filesystem import FileSystemCache

def test_cache_size_limit(tmp_path):
    # Set max cache size to 200 bytes (key1 ~70 bytes, key2 ~140 bytes)
    cache = FileSystemCache(str(tmp_path), max_cache_size_bytes=200)

    # Store items
    cache.set("key1", "A" * 10)  # Around 70 bytes when serialized to JSON dict
    time.sleep(0.01)  # to ensure different access time

    # Check that key1 was saved
    assert cache.get("key1") is not None

    # Save another one, this should trigger the enforce_size_limit
    cache.set("key2", "B" * 20)  # Around 140 bytes

    # Wait to guarantee file system operations finalize (usually not strictly needed but safe)
    time.sleep(0.05)

    # Total size ~210 bytes > 200, target is 80% (160 bytes)
    # key1 should have been deleted (LRU policy) since it was stored earlier.
    assert cache.get("key1") is None
    assert cache.get("key2") is not None
