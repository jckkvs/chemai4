import pytest
import numpy as np
from domainml.constraints.kernel_opt import KernelMonotonicity

def test_kernel_monotonicity_increasing():
    np.random.seed(42)
    # Generate monotonic data with some noise
    X = np.sort(np.random.rand(20, 1) * 10, axis=0)
    y = (X ** 2).ravel() + np.random.randn(20) * 2
    
    model = KernelMonotonicity(gamma=0.1, alpha=0.1, constraint_type='increasing')
    model.fit(X, y)
    
    # Test strict monotonicity on training data
    preds = model.predict(X)
    diffs = np.diff(preds)
    # Give small tolerance for numerical precision in CVXPY
    assert np.all(diffs >= -1e-5)
    
def test_kernel_monotonicity_decreasing():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([5, 4, 3, 2, 1])
    
    model = KernelMonotonicity(gamma=0.1, alpha=0.1, constraint_type='decreasing')
    model.fit(X, y)
    
    preds = model.predict(X)
    diffs = np.diff(preds)
    assert np.all(diffs <= 1e-5)

def test_kernel_invalid_inputs():
    model = KernelMonotonicity()
    # 2D features should raise ValueError
    X_2d = np.random.rand(10, 2)
    y = np.random.rand(10)
    with pytest.raises(ValueError, match="exactly 1D features"):
        model.fit(X_2d, y)
        
    model_bad = KernelMonotonicity(constraint_type='weird')
    X_1d = np.random.rand(10, 1)
    with pytest.raises(ValueError, match="Invalid constraint type"):
        model_bad.fit(X_1d, y)

    model_unfitted = KernelMonotonicity()
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        model_unfitted.predict(X_1d)
        
    model.fit(X_1d, y)
    with pytest.raises(ValueError, match="Predict input must be exactly 1D"):
        model.predict(X_2d)
