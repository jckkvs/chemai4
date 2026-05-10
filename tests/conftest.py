"""
tests/conftest.py

pytest共通フィクスチャと設定。
"""
from __future__ import annotations

import os

# ---------- MKL DLL クラッシュ回避 (Windows) ----------
# Intel MKL の threadpoolctl が MKL_Get_Max_Threads() 呼出時に
# SEH exception (WinError 0xc06d007f) を起こす環境固有バグの回避策。
# テスト起動前に環境変数でスレッド数を制限する。
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import pytest
from backend.utils.optional_import import probe_all_optional_libraries

# テストセッション開始時に可用性キャッシュを初期化
probe_all_optional_libraries()



@pytest.fixture(scope="session")
def random_seed() -> int:
    return 42


@pytest.fixture(scope="session")
def small_regression_df():
    import numpy as np
    import pandas as pd
    """セッションスコープの小さい回帰DataFrameフィクスチャ。"""
    np.random.seed(42)
    n = 150
    return pd.DataFrame({
        "numeric_a": np.random.randn(n),
        "numeric_b": np.random.exponential(5, n),
        "cat_a": np.random.choice(["X", "Y", "Z"], n),
        "binary": np.random.randint(0, 2, n).astype(float),
        "target": np.random.randn(n),
    })


@pytest.fixture(scope="session")
def small_classification_df():
    import numpy as np
    import pandas as pd
    """セッションスコープの分類DataFrameフィクスチャ。"""
    np.random.seed(42)
    n = 150
    return pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.choice(["A", "B", "C"], n),
        "target": np.random.randint(0, 2, n),
    })
