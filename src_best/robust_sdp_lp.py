"""
    Srivastava, Prateek R., Purnamrita Sarkar, and Grani A. Hanasusanto. "A robust spectral clustering algorithm
    for sub-Gaussian mixture models with outliers." Operations Research 71, no. 1 (2023): 224-244.

    Algorithm 1 Robust Spectral Clustering / Robust-SDP

"""

import cvxpy as cp
import numpy as np
from numpy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def choose_theta_gamma(Y, beta=0.06, alpha=0.2, reduce_dim=False, d_reduced=10):
    """
    Compute the scaling parameter θ and the offset parameter γ
    based on robust neighborhood-based distance quantiles.

    Parameters:
    - Y: np.ndarray of shape (N, d), dataset of N points in d-dimensional space
    - beta: float, fraction for local neighborhood distance quantile (default 0.06)
    - alpha: float, fraction of outliers to tolerate (default 0.2)
    - reduce_dim: bool, whether to reduce dimensionality using PCA
    - d_reduced: int, target dimensionality if reduce_dim is True

    Returns:
    - theta: float, kernel scaling parameter
    - gamma: float, kernel offset parameter
    """
    N, d = Y.shape

    if reduce_dim and d > d_reduced:
        pca = PCA(n_components=d_reduced)
        Y = pca.fit_transform(Y)
        d = d_reduced

    # Compute pairwise distances: l2 distances
    D = pairwise_distances(Y, Y, metric='euclidean')

    # Compute q_i: the beta-quantile of each row (excluding self-distance)
    q = np.array([
        np.quantile(np.delete(D[i], i), beta)
        for i in range(N)
    ])

    # θ is the 1 - α quantile of q_i
    q_alpha = np.quantile(q, 1 - alpha)
    theta = q_alpha / np.sqrt(chi2.ppf(1 - alpha, df=d))

    # Compute γ based on quantile of chi-squared distribution
    t_alpha = chi2.ppf(1 - alpha, df=d)
    gamma = np.exp(-t_alpha / 2)

    return theta, gamma


def solve_lp_sdp(K, N, gamma, is_sdp=False):
    # Construct E_N: matrix of all ones
    E_N = np.ones((N, N))

    # Define optimization variable
    if is_sdp:
        X = cp.Variable((N, N), PSD=True)
    else:
        X = cp.Variable((N, N))

    # Define objective function
    objective = cp.Maximize(cp.trace((K - gamma * E_N) @ X))

    # Define constraints
    constraints = [X >= 0, X <= 1]

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Get the optimized X matrix
    return X.value


def k_means(points, centroids_input, k, max_iterations=300, tolerance=1e-4):
    new_centroids = np.copy(centroids_input)

    for i in range(max_iterations):
        # Assign each point to the closest centroid
        distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
        labels = np.argmin(distances, axis=1)

        pre_centroids = np.copy(new_centroids)
        # Update the centroids to be the mean of the points in each cluster
        for j in range(k):
            if sum(labels == j) == 0:
                # new_centroids[j] use the previous centroid
                continue
            new_centroids[j] = np.mean(points[labels == j], axis=0)

        if np.sum((new_centroids - pre_centroids) ** 2) / k < tolerance:
            break
    # L2
    distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    labels = np.argmin(distances, axis=1)
    # print(f'k_means iterations: {i}')
    return new_centroids, labels


def robust_spectral_clustering(Y, r, theta, gamma, is_sdp=False, random_state=42):
    """
    Parameters:
        Y     : numpy array of shape (N, d) - input observations
        r     : int - number of clusters
        theta : float - scaling parameter for the Gaussian kernel
        gamma : float in (0, 1) - offset parameter
    Returns:
        new_centroids:
        labels:
    """

    N = Y.shape[0]

    # Step 1: Construct Gaussian kernel matrix
    pairwise_sq_dists = cdist(Y, Y, 'sqeuclidean')
    K = np.exp(-pairwise_sq_dists / (2 * theta ** 2))

    # Step 2: Solve Robust-LP/SDP
    X_hat = solve_lp_sdp(K, N, gamma, is_sdp=is_sdp)

    # Step 3: Eigen-decomposition
    # Compute top-r eigenvectors
    vals, vecs = eigh(X_hat)
    print(f'top {r} eigenvalues: {vals[-r:]}')
    U_hat = vecs[:, -r:]  # top r eigenvectors

    # # Step 4: Apply k-means clustering on rows of U_hat
    print('Apply k-means clustering')
    rng = np.random.RandomState(seed=random_state)
    indices = rng.choice(range(len(U_hat)), size=r, replace=False)
    centroids_input = U_hat[indices, :]
    new_centroids, labels = k_means(U_hat, centroids_input, k=r, max_iterations=300, tolerance=1e-4)

    # # Step 5: Thresholding to identify inliers and outliers
    # degrees = X_hat.sum(axis=1)
    # noise_ratio = 0.1     # noise_ratio=0.1,
    # tau = np.quantile(degrees, q=noise_ratio)  # degree threshold
    # I_hat = np.where(degrees >= tau)[0]  # inliers
    # O_hat = np.where(degrees < tau)[0]  # outliers

    return new_centroids, labels


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import collections

    random_state = 42
    Y, _ = make_blobs(n_samples=200, centers=3, n_features=2, random_state=random_state)
    print('Y.shape: ', Y.shape)
    r = 3  # number of clusters
    theta, gamma = choose_theta_gamma(Y, beta=0.06, alpha=0.2)
    print(f'theta: {theta}, gamma: {gamma}')
    new_centroids, labels = robust_spectral_clustering(Y, r, theta, gamma,
                                                       is_sdp=False, random_state=random_state)

    print(collections.Counter(labels))
