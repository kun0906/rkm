"""
    https://github.com/crisbodnar/regularised-spectral-clustering/blob/master/spectral_clustering.ipynb
    Understanding Regularized Spectral Clustering via Graph Conductance
"""

import numpy as np
import scipy as sp
from scipy.cluster.vq import whiten, kmeans
from numpy import linalg as LA
# from networkx.algorithms.cuts import conductance
from scipy.sparse.linalg import eigsh


def eig_laplacian(A, k=2):
    n = np.shape(A)[0]
    D = np.diag(1 / np.sqrt(np.ravel(A.sum(axis=0))))
    L = np.identity(n) - D.dot(A).dot(D)
    return eigsh(L, k, which='SM')

def spectral_clust(A, k=2):
    n = np.shape(A)[0]
    V, Z = eig_laplacian(A, k)

    rows_norm = np.linalg.norm(Z, axis=1, ord=2)
    Y = (Z.T / rows_norm).T
    centroids, distortion = kmeans(Y, k)

    y_hat = np.zeros(n, dtype=int)
    for i in range(n):
        dists = np.array([np.linalg.norm(Y[i] - centroids[c]) for c in range(k)])
        y_hat[i] = np.argmin(dists)
    return y_hat

def spectral_clust_chauduri(A, tau, k=2):
    n = np.shape(A)[0]
    At = A + tau / n
    D = np.diag(1 / np.sqrt(np.ravel(At.sum(axis=0))))
    L = np.identity(n) - D.dot(A).dot(D)
    V, Z = eigsh(L, k, which='SM')

    rows_norm = np.linalg.norm(Z, axis=1, ord=2)
    Y = (Z.T / rows_norm).T
    centroids, distortion = kmeans(Y, k)

    y_hat = np.zeros(n, dtype=int)
    for i in range(n):
        dists = np.array([np.linalg.norm(Y[i] - centroids[c]) for c in range(k)])
        y_hat[i] = np.argmin(dists)
    return y_hat


# Compute the size of the smallest partition
def get_min_part_size(labels):
    return min(np.sum(labels), np.size(labels) - np.sum(labels))


def get_avg_min_part_size(map_entry):
    van_part, reg_part = [], []
    for seed in range(0, 1):
        np.random.seed(seed)
        train, _ = read_graph(graph_map[map_entry])

        N = train.number_of_nodes()
        S = nx.to_numpy_matrix(train)
        tao = train.number_of_edges() * 2 / N
        SR = S + tao / N

        van_labels = spectral_clust(S)
        reg_labels = spectral_clust(SR)

        van_part.append(get_min_part_size(van_labels))
        reg_part.append(get_min_part_size(reg_labels))

    return np.mean(van_part), np.mean(reg_part), N


def graphs_part_sizes():
    van_sizes, reg_sizes, graph_sizes = [], [], []
    for graph_key in graph_map:
        print('Processing ' + graph_map[graph_key])
        van, reg, size = get_avg_min_part_size(graph_key)
        van_sizes.append(van)
        reg_sizes.append(reg)
        graph_sizes.append(size)
    return van_sizes, reg_sizes, graph_sizes


def get_conductance(map_entry):
    van_cond_train, van_cond_test = [], []
    reg_cond_train, reg_cond_test = [], []
    for seed in range(0, 1):
        np.random.seed(seed)
        train, test = read_graph(graph_map[map_entry])

        N = train.number_of_nodes()
        S = nx.to_numpy_matrix(train)
        tao = train.number_of_edges() * 2 / N
        SR = S + tao / N

        van_labels = spectral_clust(S)
        reg_labels = spectral_clust(SR)

        nodes = np.array(list(train.nodes()))
        van_cond_train.append(conductance(train, nodes[van_labels == 1]))
        van_cond_test.append(conductance(test, nodes[van_labels == 1]))
        reg_cond_train.append(conductance(train, nodes[reg_labels == 1]))
        reg_cond_test.append(conductance(test, nodes[reg_labels == 1]))

    return np.mean(van_cond_train), np.mean(reg_cond_train), np.mean(van_cond_test), np.mean(reg_cond_test), N


def graphs_conductances():
    van_train, reg_train, van_test, reg_test, sizes = [], [], [], [], []
    for graph_key in graph_map:
        print('Processing graph ' + graph_map[graph_key])
        vt, rt, vtes, rtes, s = get_conductance(graph_key)
        van_train.append(vt)
        reg_train.append(rt)
        van_test.append(vtes)
        reg_test.append(rtes)
        sizes.append(s)

    return van_train, reg_train, van_test, reg_test, sizes
