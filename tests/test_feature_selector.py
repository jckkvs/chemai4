import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from backend.pipeline.feature_selector import FeatureSelector, FeatureSelectorConfig, _GroupLassoSelector

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5],
        "f2": [5, 4, 3, 2, 1],
        "f3": [0, 0, 0, 0, 0],
        "f4": [1, 0, 1, 0, 1],
    })
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return X, y

@pytest.fixture
def sample_class_data():
    X = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5],
        "f2": [5, 4, 3, 2, 1],
        "f3": [0, 0, 0, 0, 0],
        "f4": [1, 0, 1, 0, 1],
    })
    y = np.array([0, 1, 0, 1, 0])
    return X, y

class DummyMeta:
    def __init__(self, fixed=False, group=None):
        self.fixed = fixed
        self.group = group

def test_feature_selector_none(sample_data):
    X, y = sample_data
    cfg = FeatureSelectorConfig(method="none")
    fs = FeatureSelector(cfg)
    fs.fit(X, y)
    X_trans = fs.transform(X)
    assert X_trans.shape == X.shape
    assert len(fs.get_feature_names_out()) == 4

def test_feature_selector_lasso(sample_data):
    X, y = sample_data
    cfg = FeatureSelectorConfig(method="lasso", task="regression")
    fs = FeatureSelector(cfg)
    fs.fit(X, y)
    X_trans = fs.transform(X)
    assert X_trans.shape[1] <= X.shape[1]

def test_feature_selector_ridge(sample_data):
    X, y = sample_data
    cfg = FeatureSelectorConfig(method="ridge", task="regression")
    fs = FeatureSelector(cfg)
    fs.fit(X, y)
    X_trans = fs.transform(X)
    assert X_trans.shape[1] <= X.shape[1]

def test_feature_selector_rf(sample_data, sample_class_data):
    X, y = sample_data
    # Regression
    cfg = FeatureSelectorConfig(method="rfr", task="regression")
    fs = FeatureSelector(cfg)
    fs.fit(X, y)
    X_trans = fs.transform(X)
    assert X_trans.shape[1] <= X.shape[1]

    # Classification
    X_c, y_c = sample_class_data
    cfg_c = FeatureSelectorConfig(method="rfc", task="classification")
    fs_c = FeatureSelector(cfg_c)
    fs_c.fit(X_c, y_c)
    X_trans_c = fs_c.transform(X_c)
    assert X_trans_c.shape[1] <= X_c.shape[1]

def test_feature_selector_xgb_fallback(sample_data):
    X, y = sample_data
    cfg = FeatureSelectorConfig(method="xgb", task="regression")
    # xgboostが無い環境をモック
    with patch("backend.pipeline.feature_selector._xgb", False):
        fs = FeatureSelector(cfg)
        fs.fit(X, y)
        X_trans = fs.transform(X)
        assert X_trans.shape[1] <= X.shape[1]

def test_feature_selector_select_percentile(sample_data):
    X, y = sample_data
    cfg = FeatureSelectorConfig(method="select_percentile", percentile=50, score_func="f_regression")
    fs = FeatureSelector(cfg)
    fs.fit(X, y)
    X_trans = fs.transform(X)
    assert X_trans.shape[1] == 2  # 50% of 4 features

def test_feature_selector_select_kbest_classification(sample_class_data):
    X, y = sample_class_data
    cfg = FeatureSelectorConfig(method="select_kbest", k=2, task="classification", score_func="f_classif")
    fs = FeatureSelector(cfg)
    fs.fit(X, y)
    X_trans = fs.transform(X)
    assert X_trans.shape[1] == 2

def test_feature_selector_score_func_correction(sample_class_data):
    X, y = sample_class_data
    # f_regressionを指定するが分類なので修正されるはず
    cfg = FeatureSelectorConfig(method="select_kbest", k=2, task="classification", score_func="f_regression")
    fs = FeatureSelector(cfg)
    fs.fit(X, y)
    assert fs._resolve_score_func(cfg).__name__ == "f_classif"

def test_feature_selector_invalid_method(sample_data):
    X, y = sample_data
    cfg = FeatureSelectorConfig(method="invalid_method")
    fs = FeatureSelector(cfg)
    fs.fit(X, y)
    X_trans = fs.transform(X)
    # invalid method falls back to SelectFromModel(RandomForest), so it selects <= 4 features
    assert X_trans.shape[1] <= X.shape[1]

def test_feature_selector_fixed_columns(sample_data):
    X, y = sample_data
    meta = {
        "f3": DummyMeta(fixed=True),
        "f4": DummyMeta(fixed=True)
    }
    cfg = FeatureSelectorConfig(method="select_kbest", k=1, score_func="f_regression")
    fs = FeatureSelector(cfg, column_meta=meta)
    fs.fit(X, y)
    X_trans = fs.transform(X)
    
    # Normally k=1 would choose 1 feature, but f3 and f4 are fixed
    assert X_trans.shape[1] >= 2
    
def test_group_lasso_selector_wrapper(sample_data):
    X, y = sample_data
    X_arr = X.values
    # Test internal _GroupLassoSelector if group_lasso is mocked
    gl = _GroupLassoSelector(alpha=0.05, groups=[[0, 1], [2, 3]])
    
    import sys
    mock_group_lasso = MagicMock()
    sys.modules["group_lasso"] = mock_group_lasso
    
    try:
        with patch("group_lasso.GroupLasso") as mock_gl:
            mock_instance = MagicMock()
            mock_instance.coef_ = np.array([0.1, 0.0, 0.5, 0.0])
            mock_gl.return_value = mock_instance
            
            gl.fit(X_arr, y)
            X_trans = gl.transform(X_arr)
            assert X_trans.shape[1] == 2  # indices 0 and 2 have non-zero coef
    finally:
        del sys.modules["group_lasso"]

def test_build_group_lasso(sample_data):
    X, y = sample_data
    meta = {
        "f1": DummyMeta(group="G1"),
        "f2": DummyMeta(group="G1"),
        "f3": DummyMeta(group="G2"),
        "f4": DummyMeta(group="G2"),
    }
    cfg = FeatureSelectorConfig(method="group_lasso", group_lasso_groups=None)
    
    import sys
    mock_group_lasso_module = MagicMock()
    sys.modules["group_lasso"] = mock_group_lasso_module
    
    try:
        with patch("backend.pipeline.feature_selector._group_lasso", True), \
             patch("backend.pipeline.feature_selector._GroupLassoSelector") as mock_wrapper:
            fs = FeatureSelector(cfg, column_meta=meta)
            mock_instance = MagicMock()
            mock_wrapper.return_value = mock_instance
            fs.fit(X, y)
            mock_wrapper.assert_called_once()
            args, kwargs = mock_wrapper.call_args
    finally:
        del sys.modules["group_lasso"]
        assert kwargs["groups"] == [[0, 1], [2, 3]]

def test_build_relieff_fallback(sample_data):
    X, y = sample_data
    cfg = FeatureSelectorConfig(method="relieff")
    with patch("backend.pipeline.feature_selector._skrebate", False):
        fs = FeatureSelector(cfg)
        fs.fit(X, y)
        assert fs._selector.__class__.__name__ == "SelectFromModel"

def test_build_boruta_fallback(sample_data):
    X, y = sample_data
    cfg = FeatureSelectorConfig(method="boruta")
    with patch("backend.pipeline.feature_selector._boruta", False):
        fs = FeatureSelector(cfg)
        fs.fit(X, y)
        assert fs._selector.__class__.__name__ == "SelectFromModel"

def test_build_genetic_fallback(sample_data):
    X, y = sample_data
    cfg = FeatureSelectorConfig(method="genetic")
    with patch("backend.pipeline.feature_selector._genetic", False):
        fs = FeatureSelector(cfg)
        fs.fit(X, y)
        assert fs._selector.__class__.__name__ == "SelectFromModel"

def test_select_from_model_custom(sample_data):
    X, y = sample_data
    cfg = FeatureSelectorConfig(method="select_from_model", estimator_key="ridge")
    
    with patch("backend.models.factory.get_model") as mock_get_model:
        mock_ridge = MagicMock()
        mock_get_model.return_value = mock_ridge
        fs = FeatureSelector(cfg)
        fs.fit(X, y)
        mock_get_model.assert_called_with("ridge", task="regression")
