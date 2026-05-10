import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from domainml.analysis.metrics import constraint_satisfaction_score
from domainml.analysis.constrained_cv import ConstrainedCV
from domainml.analysis.uncertainty import UncertaintyEstimator

def test_metrics_satisfaction_score():
    X = np.array([[1], [2], [3], [4]])
    
    # Perfectly increasing
    y_inc = np.array([2, 4, 6, 8])
    score1 = constraint_satisfaction_score(X, y_inc, feature_idx=0, constraint_type='increasing')
    assert score1 == 1.0
    
    # Perfectly decreasing
    score2 = constraint_satisfaction_score(X, y_inc, feature_idx=0, constraint_type='decreasing')
    assert score2 == 0.0

    # Flat returns 1.0
    y_flat = np.array([5, 5, 5, 5])
    assert constraint_satisfaction_score(X, y_flat) == 1.0

    with pytest.raises(ValueError):
        constraint_satisfaction_score(X, y_inc, constraint_type='bad')

def test_constrained_cv():
    X = np.random.rand(20, 2)
    y = X[:, 0] * 2 + np.random.randn(20) * 0.1
    
    base = LinearRegression()
    # cv=2 for speed
    ccv = ConstrainedCV(estimator=base, cv=2, scoring='r2', constraint_func=constraint_satisfaction_score)
    res = ccv.evaluate(X, y)
    
    assert 'mean_score' in res
    assert 'mean_constraint_score' in res
    assert 'objective' in res
    assert isinstance(res['mean_score'], float)
    
    # Test without constraint_func (fallback)
    ccv_no_const = ConstrainedCV(estimator=base, cv=2, scoring='r2')
    res2 = ccv_no_const.evaluate(X, y)
    assert res2['mean_constraint_score'] == 1.0

def test_uncertainty_estimator():
    np.random.seed(42)
    X = np.random.rand(30, 2)
    y = X[:, 0] * 2 + X[:, 1]
    
    base = LinearRegression()
    # Use low bootstrap count for speed
    ue = UncertaintyEstimator(base, n_bootstraps=5, random_state=42)
    ue.fit(X, y)
    
    preds, stds = ue.predict(X, return_std=True)
    assert preds.shape == (30,)
    assert stds.shape == (30,)
    assert np.all(stds >= 0)
    
    ue_unfitted = UncertaintyEstimator(base)
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        ue_unfitted.predict(X)
        
    # Pandas DataFrame coverage
    import pandas as pd
    X_df = pd.DataFrame(X)
    y_df = pd.Series(y)
    ue.fit(X_df, y_df)
    preds2 = ue.predict(X_df, return_std=False)
    assert preds2.shape == (30,)
