"""
Robust Spectral Clustering

Source:
    https://github.com/abojchevski/rsc/tree/master
"""
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.stats as stats
from scipy.linalg import eigh, eig
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, eigs
from sklearn.cluster import k_means
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_kernels
from sklearn.neighbors import kneighbors_graph


def compute_bandwidth(X, q = 0.3):
    pd = pairwise_distances(X, Y=None, metric='euclidean')
    qs = np.quantile(pd, q=q, axis=1)
    alpha = 0.01
    n, d = X.shape
    df = d  # degrees of freedom
    denominator = np.sqrt(stats.chi2.ppf((1 - alpha), df))
    bandwidth = np.quantile(qs, (1 - alpha)) / denominator

    return bandwidth


def rbf_graph(points, q = 0.3):

    sigma = compute_bandwidth(points, q=q)
    gamma = 1 / (2 * sigma ** 2)

    # eigen_solver{‘arpack’, ‘lobpcg’, ‘amg’}, default=None
    # The eigenvalue decomposition strategy to use. AMG requires pyamg to be installed.
    # It can be faster on very large, sparse problems, but may also lead to instabilities.
    # If None, then 'arpack' is used. See [4] for more details regarding 'lobpcg'.
    affinity_matrix_ = pairwise_kernels(
        points, metric='rbf', filter_params=True, gamma=gamma,
    )
    # Set the diagonal values to NaN or 0
    np.fill_diagonal(affinity_matrix_, 0)  # Use np.nan or 0 to exclude, inplace = True
    return affinity_matrix_


def plot_matrix(adjacency_matrix):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # # Convert the sparse matrix to a dense format (adjacency matrix)
    # adjacency_matrix = connectivity.toarray()

    # Visualize the adjacency matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(adjacency_matrix, annot=True, cmap='viridis', cbar=False)
    plt.title('k-Nearest Neighbors Graph as Adjacency Matrix')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    plt.show()


def is_symmetric(matrix):
    """
    Check if a given csr_matrix is symmetric.

    Parameters:
    matrix (csr_matrix): The matrix to check for symmetry.

    Returns:
    bool: True if the matrix is symmetric, False otherwise.
    """
    if not isinstance(matrix, csr_matrix):
        raise ValueError("Input must be a csr_matrix")

    # Convert the matrix to its transpose
    matrix_transpose = matrix.transpose()

    # Check if the matrix is equal to its transpose
    is_symmetric = (matrix != matrix_transpose).nnz == 0

    return is_symmetric

import numpy as np

def verify_laplacian(L):
    # Check symmetry
    if not np.allclose(L, L.T):
        print("Matrix is not symmetric.")
        return False

    # Check non-positivity of off-diagonal elements
    if not np.all(L[np.triu_indices_from(L, 1)] <= 0):
        print("Off-diagonal elements are not non-positive.")
        return False

    # Check sum of rows (should be zero)
    row_sums = np.sum(L, axis=1)
    if not np.allclose(row_sums, 0):
        print("Row sums are not zero.")
        return False

    # # Check positive semi-definiteness
    # eigenvalues = np.linalg.eigvalsh(L)
    # if not np.all(eigenvalues >= -1e-10):  # allow small numerical errors
    #     print("Matrix is not positive semi-definite.")
    #     return False

    # print("Matrix is a valid Laplacian matrix.")
    return True


class RSC:
    """
    Implementation of the method proposed in the paper:
    'Robust Spectral Clustering for Noisy Data: Modeling Sparse Corruptions Improves Latent Embeddings'

    If you publish material based on algorithms or evaluation measures obtained from this code,
    then please note this in your acknowledgments and please cite the following paper:
        Aleksandar Bojchevski, Yves Matkovic, and Stephan Günnemann.
        2017. Robust Spectral Clustering for Noisy Data.
        In Proceedings of KDD’17, August 13–17, 2017, Halifax, NS, Canada.

    Copyright (C) 2017
    Aleksandar Bojchevski
    Yves Matkovic
    Stephan Günnemann
    Technical University of Munich, Germany
    """

    def __init__(self, k, nn=15, theta=20, m=0.5, laplacian=1, n_iter=50, affinity='knn', q = 0.3,
                 normalize=False, verbose=False,
                 random_state=42):
        """
        :param k: number of clusters
        :param nn: number of neighbours to consider for constructing the KNN graph (excluding the node itself)
        :param theta: number of corrupted edges to remove
        :param m: minimum percentage of neighbours to keep per node (omega_i constraints)
        :param n_iter: number of iterations of the alternating optimization procedure
        :param laplacian: which graph Laplacian to use: 0: L, 1: L_rw, 2: L_sym
        :param normalize: whether to row normalize the eigen vectors before performing k_means
        :param verbose: verbosity
        """

        self.k = k
        self.nn = nn
        self.theta = theta
        self.m = m
        self.n_iter = n_iter
        self.normalize = normalize
        self.verbose = verbose
        self.laplacian = laplacian
        self.random_state = random_state
        self.affinity=affinity
        self.q = q

        if laplacian == 0:
            if self.verbose:
                print('Using unnormalized Laplacian L')
        elif laplacian == 1:
            if self.verbose:
                print('Using random walk based normalized Laplacian L_rw')
        elif laplacian == 2:
            raise NotImplementedError('The symmetric normalized Laplacian L_sym is not implemented yet.')
        else:
            raise ValueError('Choice of graph Laplacian not valid. Please use 0, 1 or 2.')

    def __latent_decomposition(self, X):
        # Set random seed for reproducibility
        # rng = np.random.seed(self.random_state)   # set globally for all random number generation operations
        rng = np.random.RandomState(
            seed=self.random_state)  # creates a new instance of the random number generator with the specified seed value.

        if self.affinity not in ['rbf', 'knn']:
            raise ValueError(f"{self.affinity} is not correct!")
        if self.affinity == 'rbf':
            # this method doesn't work for rbf when we do Ag = A - Ac
            A = rbf_graph(X, q=self.q)
            # Convert the numpy array to a csr_matrix
            A = csr_matrix(A)
            # raise ValueError("This method doesn't work for rbf when we do Ag = A - Ac")
        elif self.affinity == 'knn':
            # compute the KNN graph
            A = kneighbors_graph(X=X, n_neighbors=self.nn, metric='euclidean', include_self=False, mode='connectivity')
        else:
            raise NotImplementedError(self.affinity)

        A = A.maximum(A.T)  # make the graph undirected
        # plot_matrix(A.toarray())

        N = A.shape[0]  # number of nodes
        deg = A.sum(0).A1  # node degrees,  Return `self` as a flattened `ndarray`.

        prev_trace = np.inf  # keep track of the trace for convergence
        Ag = A.copy()
        Ac = None
        pre_h, pre_H, pre_Ac, pre_Ag = None, None, None, None
        for it in range(self.n_iter):
            if self.verbose: print(f'it:{it}, Ag: {Ag.toarray()}')
            # form the unnormalized Laplacian
            D = sp.diags(Ag.sum(0).A1).tocsr()
            L = D - Ag
            if not verify_laplacian(L.toarray()):
                warnings.warn(f'{it}th iteration, L is not a valid Laplacian matrix.')

            # Avoid random initialization for eigsh(),
            # generate random samples from a uniform distribution over [0, 1).
            v0 = rng.rand(min(L.shape))

            if self.laplacian == 0:
                # solve the eigenvalue problem: L@H = \lambda@H
                h, H = eigsh(L, min(self.k, N), which='SM', v0=v0)  # eigsh can involve random initialization
            elif self.laplacian == 1:
                # solve the generalized eigenvalue problem: L@H = \lambda*D@H
                try:
                    h, H = eigsh(L, min(self.k, N), D, which='SM', v0=v0)
                    # print(self.k*2, flush=True)
                except Exception as e:
                    # warnings.warn(f'{it}th iteration, eigsh() fails, {e}, so we use the previous results: Ag, Ac, H.')
                    # h, H = pre_h, pre_H
                    # Ac, Ag = pre_Ac, pre_Ag
                    # break
                    h, H = eig(L.toarray(), b=D.toarray())
                if self.verbose:
                    print(list(h), sorted(h, key=lambda x: abs(x), reverse=False))
                    print('h[i] - h[i-1] diff： ', [h[i] - h[i - 1] for i in range(1, len(h))])
                # is_non_zero_eigen = False  # if True, we only use non_zero_eigenvalues and eigenvectors
                # if is_non_zero_eigen:
                #     # Find top 2 non-zero eigenvalues
                #     top_k_nonzero_indices = []
                #     for idx in range(len(h)):
                #         if len(top_k_nonzero_indices) >= self.k:
                #             break
                #         if not np.isclose(h[idx], 0, atol=1.e-10):  # Check if the eigenvalue is non-zero
                #             top_k_nonzero_indices.append(idx)
                #     # print(it, top_k_nonzero_indices, h, flush=True)
                #     h, H = h[top_k_nonzero_indices], H[:, top_k_nonzero_indices]
                # else:
                #     h, H = h[:self.k], H[:, :self.k]
                self.h = h
            else:
                raise NotImplementedError(f'{self.laplacian} is not implemented!')
            trace = h.sum()
            pre_h, pre_H, pre_Ac, pre_Ag = h, H, Ac, Ag
            if self.verbose:
                print('Iter: {}, prev_trace - trace: {}, Trace: {:.4f} h: {}'.format(it, prev_trace - trace, trace, h))

            if self.theta == 0:
                # no edges are removed
                break

            if prev_trace - trace < 1e-10:
                # we have converged
                break

            if self.affinity == 'rbf':
                allowed_to_remove_per_node = (deg * self.m)
            elif self.affinity == 'knn':
                allowed_to_remove_per_node = (deg * self.m).astype(np.int64)
            else:
                raise NotImplementedError(self.affinity)
            prev_trace = trace

            # consider only the edges on the lower triangular part since we are symmetric
            edges = sp.tril(A).nonzero()
            removed_edges = []

            if self.laplacian == 1:
                # fix for potential numerical instability of the eigenvalues computation
                h[np.isclose(h, 0)] = 0

                # equation (5) in the paper
                p = np.linalg.norm(H[edges[0]] - H[edges[1]], axis=1) ** 2 \
                    - np.linalg.norm(H[edges[0]] * np.sqrt(h), axis=1) ** 2 \
                    - np.linalg.norm(H[edges[1]] * np.sqrt(h), axis=1) ** 2
            else:
                # equation (4) in the paper
                p = np.linalg.norm(H[edges[0]] - H[edges[1]], axis=1) ** 2

            # Greedy remove the worst edges
            for ind in p.argsort()[::-1]:
                e_i, e_j, p_e = edges[0][ind], edges[1][ind], p[ind]

                # remove the edge if it satisfies the constraints
                if allowed_to_remove_per_node[e_i] > 0 and allowed_to_remove_per_node[e_j] > 0 and p_e > 0:
                    if self.affinity == 'rbf':
                        a_ij = A[e_i,e_j]
                    elif self.affinity =='knn':
                        a_ij = 1
                    else:
                        raise NotImplementedError(self.affinity)
                    allowed_to_remove_per_node[e_i] -= a_ij
                    allowed_to_remove_per_node[e_j] -= a_ij
                    removed_edges.append((e_i, e_j))
                    if len(removed_edges) == self.theta:
                        break

            removed_edges = np.array(removed_edges)
            if removed_edges.shape[0] > 0:
                if self.affinity == 'rbf':
                    vs = []
                    for edge in removed_edges:
                        a, b = edge
                        if self.verbose: print(a, b, A[a, b], Ag[a, b])
                        vs.append(A[a, b])
                    Ac = sp.coo_matrix((vs, (removed_edges[:, 0], removed_edges[:, 1])),
                                       shape=(N, N)).tocsr()
                elif self.affinity == 'knn':
                    Ac = sp.coo_matrix((np.ones(len(removed_edges)), (removed_edges[:, 0], removed_edges[:, 1])),
                                       shape=(N, N)).tocsr()
                else:
                    raise NotImplementedError(self.affinity)
                Ac = Ac.maximum(Ac.T)
                Ag = A - Ac
                if np.min(Ag) < 0:
                    warnings.warn(f'iter:{it}, np.min(Ag):{np.min(Ag)} < 0')
                    # Replace negative values with 0
                    Ag.data[Ag.data < 0] = 0
            else:
                # use the previous results
                if self.verbose:
                    print(removed_edges.shape, flush=True)
                break

        return Ag, Ac, H

    def fit_predict(self, X, init="k-means++"):
        """
        :param X: array-like or sparse matrix, shape (n_samples, n_features)
        :return: cluster labels ndarray, shape (n_samples,)
        """

        Ag, Ac, H = self.__latent_decomposition(X)
        self.Ag = Ag
        self.Ac = Ac

        if self.normalize:
            self.H = H / np.linalg.norm(H, axis=1)[:, None]
        else:
            self.H = H

        # from _kmeans import k_means
        centroids, labels, *_ = k_means(X=self.H, n_clusters=self.k,
                                        init=init, n_init=1,
                                        random_state=self.random_state)

        self.centroids = centroids
        self.labels = labels

        return labels
