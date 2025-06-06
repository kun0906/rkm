"""" Regularised_k_means

    Reference:
        Kushagra, Shrinu, Yaoliang Yu, and Shai Ben-David. "Provably noise-robust, regularised $ k $-means clustering."
    arXiv preprint arXiv:1711.11247 (2017).
"""

import numpy as np
import cvxpy as cp
from sklearn.cluster import KMeans


def compute_z_y(X, k, lambda_):
    """
    Solve the SDP (equation 6) in the reference for robust k-means clustering.

    Parameters:
    - X: (n x d) data matrix
    - k: number of clusters
    - lambda_: regularization parameter

    Returns:
    - Z: optimal clustering matrix (n x n)
    - y: noise indicator vector (n,)
    """
    n = X.shape[0]
    # Compute squared Euclidean distance matrix D
    D = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)

    # Define optimization variables
    Z = cp.Variable((n, n), PSD=True)  # Positive semidefinite matrix
    y = cp.Variable(n, nonneg=True)    # Nonnegative noise variables, y >= 0
    one = np.ones(n)        # all one vector
    # Constraints
    constraints = [
        cp.trace(Z) == k,                               # Trace equals number of clusters
        0.5 * (Z + Z.T) @ one + y == one, # (Z + Z^T)/2 * 1 + y = 1
        Z >= 0                                         # Element-wise nonnegativity of Z
    ]

    # Objective function
    objective = cp.Minimize(cp.trace(D @ Z) + lambda_ * (one @ y))

    # Solve SDP
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, warm_start=True)

    print("Optimal value:", prob.value)
    return Z.value, y.value


def filter_noise(Z, y, X, threshold):
    """
    Remove points considered noise and adjust Z and X accordingly.

    Parameters:
    - Z: (n x n) clustering matrix
    - y: (n,) noise vector
    - X: (n x d) data points
    - threshold: threshold to determine noise points

    Returns:
    - X_reduced: data points without noise
    - Z_reduced: clustering matrix without noise rows/cols
    - C_k_plus_1: noise points cluster
    """
    y = y.flatten()
    indices_to_remove = np.where(y > threshold)[0]

    # Noise cluster
    C_k_plus_1 = X[indices_to_remove, :]

    # Mask for non-noise points
    mask = np.ones(X.shape[0], dtype=bool)
    mask[indices_to_remove] = False

    X_reduced = X[mask, :]
    Z_reduced = Z[np.ix_(mask, mask)]

    return X_reduced, Z_reduced, C_k_plus_1


def regularised_k_means(X, k, lambda_=0.5, threshold=0.5, random_state=0):
    """
    Perform robust regularised k-means clustering.

    Parameters:
    - X: data matrix (n x d)
    - k: number of clusters
    - lambda_: regularization parameter
    - threshold: threshold to identify noise points
    - random_state: random seed for k-means

    Returns:
    - clusters: list of np.arrays, clusters C1,...,Ck and noise cluster C_{k+1}
    """
    Z, y = compute_z_y(X, k, lambda_)
    X_reduced, Z_reduced, C_k_plus_1 = filter_noise(Z, y, X, threshold)

    # Compute X^T Z_reduced and transpose for clustering columns
    X2 = (X_reduced.T @ Z_reduced).T

    # k-means clustering on columns of X^T Z_reduced
    kmeans = KMeans(n_clusters=k, random_state=random_state).fit(X2)
    labels = kmeans.labels_

    clusters = []
    for cluster_id in range(k):
        clusters.append(X_reduced[labels == cluster_id, :])

    # Append noise cluster as the (k+1)-th cluster
    clusters.append(C_k_plus_1)

    return clusters


if __name__ == "__main__":
    # Parameters
    n = 10  # number of data points
    k = 3   # number of clusters
    random_state=42

    # Generate example data points
    np.random.seed(random_state)
    X = np.random.randn(n, 2)

    clusters = regularised_k_means(X, k, lambda_=0.5, threshold=0.5, random_state=random_state)

    for i, cluster in enumerate(clusters, 1):
        print(f"Cluster C{i} has {cluster.shape[0]} points")
