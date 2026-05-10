import numpy as np
import pandas as pd
import pytest
from backend.data.dim_reduction import DimReducer

def test_pca_reconstruction_error():
    # テストデータ生成 (低ランク)
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10)
    
    # 手動で config をエミュレート (DimReducer は config.method 等を参照する)
    class MockConfig:
        method = "pca"
        n_components = 5
        whiten = False
        random_state = 42
        scale = True
        method_params = {}
    
    reducer = DimReducer(MockConfig())
    X_reduced = reducer.fit_transform(X)
    
    error = reducer.reconstruction_error_
    assert error is not None
    assert len(error) == 100
    assert np.all(error >= 0)
    
    # 全次元保持
    class MockConfigFull:
        method = "pca"
        n_components = 10
        whiten = False
        random_state = 42
        scale = True
        method_params = {}
        
    reducer_full = DimReducer(MockConfigFull())
    reducer_full.fit_transform(X)
    assert np.all(reducer_full.reconstruction_error_ < 1e-10)

def test_dim_reduction_extra_params():
    class MockConfigParams:
        method = "pca"
        n_components = 2
        whiten = False
        random_state = 42
        scale = True
        method_params = {"svd_solver": "full"}
        
    reducer = DimReducer(MockConfigParams())
    X = np.random.randn(20, 5)
    reducer.fit(X)
    assert reducer._reducer.svd_solver == "full"
