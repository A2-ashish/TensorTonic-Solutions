import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    X: 2D NumPy array of shape (n_samples, n_features)
    y: 1D NumPy array of shape (n_samples,)
    """
    X=np.array(X)
    y=np.array(y)
    # Normal equation: w = (X^T X)^(-1) X^T y
    xtx = np.dot(X.T, X)
    xtx_inv = np.linalg.inv(xtx)
    xty = np.dot(X.T, y)
    w = np.dot(xtx_inv, xty)
    return w
