"""
Implements: F-202
論文: None specific (QP based monotonic kernel ridge regression)
注意点: pass/NotImplementedErrorは禁止。cvxpyを用いて厳密な単調性制約を二次計画問題として解く。
"""
import numpy as np
import cvxpy as cp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class KernelMonotonicity(BaseEstimator, RegressorMixin):
    """
    Implements: F-202
    Kernel Ridge Regression with strict Monotonicity constraints using CVXPY.
    Assumes 1D input X for simplicity of the exact monotonic derivative constraint,
    or enforces non-decreasing relationships between ordered training points.
    """
    def __init__(self, gamma=1.0, alpha=1.0, constraint_type='increasing'):
        self.gamma = gamma
        self.alpha = alpha
        self.constraint_type = constraint_type

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if X.shape[1] != 1:
            raise ValueError("KernelMonotonicity currently supports exactly 1D features for strict constraints.")
            
        if self.constraint_type not in ['increasing', 'decreasing']:
            raise ValueError(f"Invalid constraint type: {self.constraint_type}")

        N = X.shape[0]
        self.X_fit_ = X
        
        # Compute Kernel matrix
        K = rbf_kernel(X, X, gamma=self.gamma)
        
        # Optimization variables: weights alpha for kernel expansion
        w = cp.Variable(N)
        
        P = 2 * (K.T @ K + self.alpha * K)
        # Ensure symmetric positive semi-definite (numerical stability)
        P = (P + P.T) / 2
        P_psd = cp.psd_wrap(P)
        q = -2 * K.T @ y
        
        objective = cp.Minimize(0.5 * cp.quad_form(w, P_psd) + q.T @ w)
        
        # Constraints: 
        # For increasing monotonic, the derivative of f(x) = sum w_i K(x, x_i) must be >= 0 everywhere.
        # RBF Kernel derivative w.r.t x: K(x, x_i) * (-2 * gamma * (x - x_i))
        X_flat = X.ravel()
        diff = X_flat[:, np.newaxis] - X_flat[np.newaxis, :]  # shape (N, N)
        D = K * (-2 * self.gamma * diff)
        
        if self.constraint_type == 'increasing':
            constraints = [D @ w >= 0]
        else:
            constraints = [D @ w <= 0]
            
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)
        
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise RuntimeError(f"CVXPY Optimization failed. Status: {prob.status}")
            
        self.weights_ = w.value
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, ['is_fitted_', 'X_fit_', 'weights_'])
        X = check_array(X)
        if X.shape[1] != 1:
            raise ValueError("Predict input must be exactly 1D.")
        
        K_pred = rbf_kernel(X, self.X_fit_, gamma=self.gamma)
        return K_pred @ self.weights_
