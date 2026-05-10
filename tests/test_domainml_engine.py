import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from domainml.constraints.engine import MonotonicityEngine, LazyConstraintEvaluator

def test_lazy_evaluator():
    evaluator = LazyConstraintEvaluator()
    arr = np.array([1, 2, 3])
    x_hash = evaluator._hash_array(arr)
    
    # First call
    count = [0]
    def compute():
        count[0] += 1
        return "result"
        
    res1 = evaluator.get_or_evaluate(x_hash, compute)
    assert res1 == "result"
    assert count[0] == 1
    
    # Second call uses cache
    res2 = evaluator.get_or_evaluate(x_hash, compute)
    assert res2 == "result"
    assert count[0] == 1

def test_monotonicity_engine():
    X = np.array([[1], [3], [2], [4], [5]])
    y = np.array([1, 2, 1.5, 3, 5])
    
    base = LinearRegression()
    # Test increasing
    engine = MonotonicityEngine(base, constraint_type='increasing')
    engine.fit(X, y)
    
    X_test = np.array([[1.5], [2.5]])
    preds = engine.predict(X_test)
    assert preds.shape == (2,)
    # Should be monotonically increasing
    assert preds[0] <= preds[1]
    
    # Test decreasing
    engine_dec = MonotonicityEngine(base, constraint_type='decreasing')
    engine_dec.fit(X, -y)
    preds_dec = engine_dec.predict(X_test)
    assert preds_dec[0] >= preds_dec[1]
    
    # Invalid constraint type
    with pytest.raises(ValueError):
        bad_engine = MonotonicityEngine(base, constraint_type='invalid')
        bad_engine.fit(X, y)
    
    # Check invalid inputs for predict
    engine_unfitted = MonotonicityEngine(base)
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        engine_unfitted.predict(X_test)
