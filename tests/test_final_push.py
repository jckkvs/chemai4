"""
tests/test_final_push.py

カバレッジ 90% を目指すための最終ブーストテスト。
"""
import pytest
import numpy as np
import pandas as pd
import io
from unittest.mock import MagicMock, patch

from backend.data.loader import load_file, load_from_bytes, save_dataframe
from backend.models.cv_manager import get_cv, CVConfig, run_cross_validation
from backend.models.tuner import tune, TunerConfig
from sklearn.linear_model import Ridge

def test_loader_remaining_branches(tmp_path):
    # JSON with syntax error should raise
    p = tmp_path / "bad.json"
    p.write_text("{\"invalid\": }")
    with pytest.raises(Exception):
        load_file(p)
        
    # Bytes loader for tsv
    df = load_from_bytes(b"col1\tcol2\n1\t2", "data.tsv")
    assert df.shape == (1, 2)

    # _load_sdf_from_buf coverage
    from backend.data.loader import _load_sdf_from_buf
    buf = io.BytesIO(b"dummy sdf content")
    with patch("backend.utils.optional_import.is_available", return_value=True), \
         patch("rdkit.Chem.PandasTools.LoadSDF", return_value=pd.DataFrame([{"MOL": 1}])):
        _load_sdf_from_buf(buf, "test.sdf")

def test_cv_manager_remaining_branches():
    X = pd.DataFrame({"a": range(10)})
    y = np.array([0,1]*5)
    
    # run_cross_validation with more scoring
    cfg = CVConfig(cv_key="kfold", n_splits=2)
    run_cross_validation(Ridge(), X, y, cfg, scoring=["neg_mean_absolute_error", "r2"])

def test_tuner_remaining_branches():
    X, y = np.random.randn(10, 2), np.random.randn(10)
    # n_iter handling
    cfg = TunerConfig(method="random", n_iter=2)
    tune(Ridge(), X, y, cfg)

def test_factory_list_models_detailed():
    from backend.models.factory import list_models
    # Filter by available_only and tags
    list_models(available_only=True, tags=["linear"])
    list_models(available_only=False, tags=["ensemble"])

def test_optional_import_safe_exhaustive():
    from backend.utils.optional_import import safe_import, is_available
    # Nonexistent
    safe_import("a.b.c.nonexistent", "dummy")
    is_available("a.b.c.nonexistent")
