import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import chi2

def compute_ad_and_uncertainty(X_new: np.ndarray, X_train: np.ndarray, predictions: np.ndarray):
    """
    1. 適用領域距離（PCA空間Mahalanobis）
    2. 予測不確実性（軽量アンサンブル分散）

    Returns:
        dict with "ad_distance", "in_domain", "uncertainty"
    """
    n_features = X_train.shape[1]
    n_components = min(10, n_features)
    
    # NaN/infinity fallback
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_new = np.nan_to_num(X_new, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        pca = PCA(n_components=n_components).fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_new_pca = pca.transform(X_new)
        
        mean = X_train_pca.mean(axis=0)
        cov = np.cov(X_train_pca.T)
        
        # Add small epsilon to diagonal for pseudo-inverse stability if needed
        cov += np.eye(n_components) * 1e-6
        cov_inv = np.linalg.pinv(cov)
        
        def mahalanobis(x, mean, cov_inv):
            delta = x - mean
            return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))
            
        ad_dist = np.array([mahalanobis(x, mean, cov_inv) for x in X_new_pca])
        # Using 95% confidence bounds of chi2 distribution
        threshold = np.sqrt(chi2.ppf(0.95, n_components))
        in_domain = ad_dist < threshold
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"AD calculation failed: {e}")
        in_domain = np.ones(len(X_new), dtype=bool)
        ad_dist = np.zeros(len(X_new))
        
    uncertainty = np.std(predictions, axis=0) if isinstance(predictions, np.ndarray) and predictions.ndim > 1 else np.zeros(len(X_new))
    
    return {
        "ad_distance": ad_dist,
        "in_domain": in_domain,
        "uncertainty": uncertainty
    }
