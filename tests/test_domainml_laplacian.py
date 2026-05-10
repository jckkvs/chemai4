import pytest
import numpy as np
from scipy import sparse
from domainml.constraints.laplacian import SparseLaplacianBuilder, ManifoldValidityEstimator

def test_sparse_laplacian_builder():
    np.random.seed(42)
    X = np.random.rand(20, 2)
    
    builder = SparseLaplacianBuilder(n_neighbors=3)
    L = builder.build_laplacian(X)
    
    # Laplacian should be sparse
    assert sparse.issparse(L)
    # Shape should be (N, N)
    assert L.shape == (20, 20)
    
    # Row sums of a graph laplacian are typically 0
    row_sums = np.array(L.sum(axis=1)).flatten()
    assert np.allclose(row_sums, 0, atol=1e-5)

def test_manifold_validity_estimator():
    np.random.seed(42)
    # Create two disconnected clusters
    X1 = np.random.rand(10, 2) + np.array([0, 0])
    X2 = np.random.rand(10, 2) + np.array([10, 10])
    X = np.vstack((X1, X2))
    
    builder = SparseLaplacianBuilder(n_neighbors=3)
    L = builder.build_laplacian(X)
    
    estimator = ManifoldValidityEstimator(k_eigenvalues=3)
    score = estimator.estimate_validity(L)
    
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    
    # Empty / small case
    estimator2 = ManifoldValidityEstimator(k_eigenvalues=1)
    L_small = builder.build_laplacian(np.random.rand(4, 2))
    score_small = estimator2.estimate_validity(L_small)
    assert 0.0 <= score_small <= 1.0
