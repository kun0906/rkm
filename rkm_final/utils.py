from functools import wraps
import time

import numpy as np
import copy
import itertools
import matplotlib.pyplot as plt
import warnings
import scipy.stats as stats
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

from robust_spectral_clustering import RSC


def timer(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # print(f'{func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        print(f'{func.__name__} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def compute_bandwidth(X):
    pd = pairwise_distances(X, Y=None, metric='euclidean')
    beta = 0.3
    qs = np.quantile(pd, q=beta, axis=1)
    alpha = 0.01
    n, d = X.shape
    df = d  # degrees of freedom
    denominator = np.sqrt(stats.chi2.ppf((1 - alpha), df))
    bandwidth = np.quantile(qs, (1 - alpha)) / denominator

    return bandwidth


# @timer
def sc_projection(points, k, random_state):
    from sklearn.metrics import pairwise_kernels
    params = {}  # default value in slkearn
    # https://github.com/scikit-learn/scikit-learn/blob/872124551/sklearn/cluster/_spectral.py#L667
    # Number of eigenvectors to use for the spectral embedding, default=n_clusters
    n_components = k
    eigen_tol = 0.0
    eigen_solver = None
    affinity = 'rbf'  # affinity str or callable, default =’rbf’
    if affinity == 'rbf':
        # params["gamma"] = 1.0  # ?
        sigma = compute_bandwidth(points)
        params["gamma"] = 1 / (2 * sigma ** 2)

        params["degree"] = 3
        params["coef0"] = 1
        # eigen_solver{‘arpack’, ‘lobpcg’, ‘amg’}, default=None
        # The eigenvalue decomposition strategy to use. AMG requires pyamg to be installed.
        # It can be faster on very large, sparse problems, but may also lead to instabilities.
        # If None, then 'arpack' is used. See [4] for more details regarding 'lobpcg'.
        affinity_matrix_ = pairwise_kernels(
            points, metric=affinity, filter_params=True, **params,
        )
    else:
        # if affinity == "nearest_neighbors":
        connectivity = kneighbors_graph(
            points, n_neighbors=10, include_self=True, n_jobs=None
        )
        affinity_matrix_ = 0.5 * (connectivity + connectivity.T).toarray()

    # We now obtain the real valued solution matrix to the
    # relaxed Ncut problem, solving the eigenvalue problem
    # L_sym x = lambda x  and recovering u = D^-1/2 x.
    # The first eigenvector is constant only for fully connected graphs
    # and should be kept for spectral clustering (drop_first = False)
    # See spectral_embedding documentation.
    from sklearn.manifold import spectral_embedding
    maps = spectral_embedding(
        affinity_matrix_,  # n xn
        n_components=n_components,
        eigen_solver=eigen_solver,
        random_state=random_state,
        eigen_tol=eigen_tol,
        drop_first=False,
    )
    # print(np.min(points), np.max(points), np.min(affinity_matrix_), np.max(affinity_matrix_), np.min(maps), np.max(maps), flush=True)
    # MAX=1e+5
    # maps[maps > MAX] = MAX  # avoid overflow in np.square()
    # maps[maps < -MAX] = -MAX
    # print(np.min(points), np.max(points), np.min(affinity_matrix_), np.max(affinity_matrix_), np.min(maps),
    #       np.max(maps), flush=True)
    return maps


def robust_sc_projection(points, k, n_neighbours=15, random_state=42):
    """ Robust Spectral clustering
        https://github.com/abojchevski/rsc/tree/master

        RSC(k=k, nn=n_neighbours, theta=60*400, m=0.5,laplacian=1,  verbose=False, random_state=random_state)

        theta = number of corrupted edges we want to remove. E.g., if we want to remove 10, then theta = 5.
        m is minimum  percentage of neighbours will be removed for each node (omega_i constraints)

        """
    rsc = RSC(k=k, nn=n_neighbours, theta=50, m=0.5,laplacian=1,  normalize=True, verbose=False, random_state=random_state)
    # y_rsc = rsc.fit_predict(X)
    Ag, Ac, H = rsc._RSC__latent_decomposition(points)
    # # Ag: similarity matrix of good points
    # # Ac: similarity matrix of corruption points
    # # A = Ag + Ac
    # rsc.Ag = Ag
    # rsc.Ac = Ac

    if rsc.normalize:
        rsc.H = H / np.linalg.norm(H, axis=1)[:, None]
    else:
        rsc.H = H

    projected_points = rsc.H
    return projected_points
