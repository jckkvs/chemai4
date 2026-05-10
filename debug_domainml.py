import numpy as np
import sys
import traceback

def debug_kernel():
    try:
        from domainml.constraints.kernel_opt import KernelMonotonicity
        X = np.array([[1], [2], [3], [4], [5]], dtype=float)
        y = np.array([5, 4, 3, 2, 1], dtype=float)

        model = KernelMonotonicity(gamma=1.0, alpha=0.1, constraint_type='decreasing')
        model.fit(X, y)
        preds = model.predict(X)
        print("KERNEL PREDS:", preds)
        print("KERNEL DIFFS:", np.diff(preds))
    except Exception as e:
        print("KERNEL ERROR:")
        traceback.print_exc()

def debug_laplacian():
    try:
        from domainml.constraints.laplacian import SparseLaplacianBuilder, ManifoldValidityEstimator
        np.random.seed(42)
        X1 = np.random.rand(10, 2) + np.array([0, 0])
        X2 = np.random.rand(10, 2) + np.array([10, 10])
        X = np.vstack((X1, X2))
        builder = SparseLaplacianBuilder(n_neighbors=3)
        L = builder.build_laplacian(X)
        
        estimator = ManifoldValidityEstimator(k_eigenvalues=3)
        score = estimator.estimate_validity(L)
        print("LAPLACIAN SCORE:", score)
    except Exception as e:
        print("LAPLACIAN ERROR:")
        traceback.print_exc()

debug_kernel()
debug_laplacian()
