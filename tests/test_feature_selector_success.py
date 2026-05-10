import sys
import pytest
from unittest.mock import patch, MagicMock

from backend.pipeline.feature_selector import FeatureSelector, FeatureSelectorConfig

@pytest.fixture
def sample_data():
    import numpy as np
    import pandas as pd
    X = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5],
        "f2": [5, 4, 3, 2, 1],
        "f3": [0, 0, 0, 0, 0],
        "f4": [1, 0, 1, 0, 1],
    })
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return X, y

def test_build_relieff_success(sample_data):
    X, y = sample_data
    cfg = FeatureSelectorConfig(method="relieff", relieff_n_features=2, relieff_n_neighbors=2)
    
    mock_skrebate = MagicMock()
    mock_relieff_class = MagicMock()
    mock_skrebate.ReliefF = mock_relieff_class
    
    sys.modules["skrebate"] = mock_skrebate
    try:
        with patch("backend.pipeline.feature_selector._skrebate", True):
            fs = FeatureSelector(cfg)
            # The fit should call ReliefF
            fs.fit(X, y)
            mock_relieff_class.assert_called_once_with(n_features_to_select=2, n_neighbors=2)
    finally:
        del sys.modules["skrebate"]

def test_build_boruta_success(sample_data):
    X, y = sample_data
    cfg = FeatureSelectorConfig(method="boruta", boruta_n_estimators=10, boruta_max_iter=50, task="regression")
    
    mock_boruta = MagicMock()
    mock_boruta_class = MagicMock()
    mock_boruta.BorutaPy = mock_boruta_class
    
    sys.modules["boruta"] = mock_boruta
    try:
        with patch("backend.pipeline.feature_selector._boruta", True):
            fs = FeatureSelector(cfg)
            fs.fit(X, y)
            mock_boruta_class.assert_called_once()
            args, kwargs = mock_boruta_class.call_args
            assert kwargs["max_iter"] == 50
    finally:
        del sys.modules["boruta"]

def test_build_genetic_success(sample_data):
    X, y = sample_data
    cfg = FeatureSelectorConfig(method="genetic", genetic_cv=2, genetic_n_population=5, genetic_n_generations=5, task="classification")
    
    mock_genetic = MagicMock()
    mock_ga_class = MagicMock()
    mock_genetic.GAFeatureSelectionCV = mock_ga_class
    
    sys.modules["sklearn_genetic"] = mock_genetic
    try:
        with patch("backend.pipeline.feature_selector._genetic", True):
            fs = FeatureSelector(cfg)
            fs.fit(X, y)
            mock_ga_class.assert_called_once()
            args, kwargs = mock_ga_class.call_args
            assert kwargs["cv"] == 2
            assert kwargs["scoring"] == "accuracy"
    finally:
        del sys.modules["sklearn_genetic"]

def test_build_rf_xgb_classifier_success(sample_data):
    import numpy as np
    # Cover the classification and regression branches for xgboost
    X, y = sample_data
    # Convert y to int for classification
    y_cls = np.array([0, 1, 0, 1, 0])
    
    cfg = FeatureSelectorConfig(method="xgb", task="classification")
    
    mock_xgboost = MagicMock()
    sys.modules["xgboost"] = mock_xgboost
    try:
        with patch("backend.pipeline.feature_selector._xgb", True):
            fs = FeatureSelector(cfg)
            fs.fit(X, y_cls)
            mock_xgboost.XGBClassifier.assert_called_once()
    finally:
        del sys.modules["xgboost"]

def test_group_lasso_no_groups(sample_data):
    X, y = sample_data
    cfg = FeatureSelectorConfig(method="group_lasso", group_lasso_groups=None)
    
    mock_group_lasso_module = MagicMock()
    sys.modules["group_lasso"] = mock_group_lasso_module
    try:
        with patch("backend.pipeline.feature_selector._group_lasso", True):
            fs = FeatureSelector(cfg, column_meta=None)
            # Groups will be None, so it should fallback to RandomForest
            fs.fit(X, y)
            assert fs._selector.__class__.__name__ == "SelectFromModel"
    finally:
        del sys.modules["group_lasso"]
