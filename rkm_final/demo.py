from clustering import plot_centroids_diff
from robust_spectral_clustering import RSC
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
def robust_sc_projection2(points, k, n_neighbours=15, random_state=42):
    """ Robust Spectral clustering
        https://github.com/abojchevski/rsc/tree/master
        """
    rsc = RSC(k=k, nn=n_neighbours, theta=20, m=0.5,laplacian=1,  verbose=False)
    # y_rsc = rsc.fit_predict(X)
    Ag, Ac, H = rsc._RSC__latent_decomposition(points)
    # Ag: similarity matrix of good points
    # Ac: similarity matrix of corruption points
    # A = Ag + Ac
    rsc.Ag = Ag
    rsc.Ac = Ac

    if rsc.normalize:
        rsc.H = H / np.linalg.norm(H, axis=1)[:, None]
    else:
        rsc.H = H

    projected_points = rsc.H
    return projected_points


import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph
def robust_sc_projection(points, k, n_neighbours=5, random_state=42):
    """ Robust Spectral clustering
        https://github.com/abojchevski/rsc/tree/master
        """
    # np.random.seed(random_state)

    # compute the KNN graph
    A = kneighbors_graph(X=points, n_neighbors=n_neighbours, metric='euclidean', include_self=False, mode='connectivity')
    A = A.maximum(A.T)  # make the graph undirected

    N = A.shape[0]  # number of nodes
    deg = A.sum(0).A1  # node degrees
    print(max(deg))

    prev_trace = np.inf  # keep track of the trace for convergence
    Ag = A.copy()
    n_iter = 50
    for it in range(n_iter):
        # form the unnormalized Laplacian
        D = sp.diags(Ag.sum(0).A1).tocsc()
        L = D - Ag
        np.random.seed(random_state)
        # v0 = np.ones((min(L.shape),))
        v0 = np.random.rand(min(L.shape))
        h, H = eigsh(L, k, D, which='SM', v0=v0)  # get random results. add 1e-10 to the eigsh()

    projected_points = H
    # projected_points = A.toarray()[:, :k]
    return projected_points


def plot_data(points, random_state=42):
    # Create a color map for 5 classes
    colors = ['red', 'green', 'blue', 'purple', 'orange']

    # Plot the figures
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    x1, x2 = points[:, 0], points[:,1]
    axes[0, 0].scatter(x1, x2, color=colors[i], label=f'Class {i}')
    axes[0, 0].set_title('Points without outliers')
    axes[0, 0].set_xlabel('X axis')
    axes[0, 0].set_ylabel('Y axis')
    axes[0, 0].legend()

    plt.show()



for i in range(1):
    seed = i
    rng = np.random.RandomState(seed=seed)
    # points = rng.normal(size=(10, 2))
    points = np.asarray([[1, 0],
                         [0, 1],
                         [1, 1],
                         [5, 5]])
    plot_data(points)

    for clustering_method in [1, 2, 3]:
        random_state = seed
        k = 2
        n_neighbours = 2
        # clustering_method = 'robust_spectral_clustering'
        # np.random.seed(42)
        projected_points = robust_sc_projection(points, k, n_neighbours=n_neighbours, random_state=random_state)

        plot_centroids_diff(points, projected_points, cluster_size=100, clustering_method=clustering_method, random_state=random_state)



