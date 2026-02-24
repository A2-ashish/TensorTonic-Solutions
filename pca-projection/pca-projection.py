import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    X: 2D NumPy array of shape (n_samples, n_features)
    k: number of principal components to keep
    """
    # Center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors by descending eigenvalues
    sorted_idx = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, sorted_idx[:k]]
    
    # Project data onto top-k components
    X_projected = np.dot(X_centered, top_eigenvectors)
    
    return X_projected
