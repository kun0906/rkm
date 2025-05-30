from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import pairwise_distances

from robust_spectral_clustering import RSC, compute_bandwidth
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import kneighbors_graph


def plot_graph(A, X):
    import networkx as nx

    # Convert to a NetworkX graph
    # G = nx.from_scipy_sparse_matrix(A)
    G = nx.from_scipy_sparse_array(A)

    # Draw the graph
    plt.figure(figsize=(8, 6))
    # # Define positions for nodes
    pos = {i: (x1, x2) for i, (x1, x2) in enumerate(X)}

    # nx.draw(G, node_size=50)
    nx.draw_networkx(G, pos, node_size=50)
    plt.show()
# @timer
def sc_embedding(points, k, n_neighbors=10, affinity='knn', q=1, normalize=False, random_state=42):
    from sklearn.metrics import pairwise_kernels
    params = {}  # default value in slkearn
    # https://github.com/scikit-learn/scikit-learn/blob/872124551/sklearn/cluster/_spectral.py#L667
    # Number of eigenvectors to use for the spectral embedding, default=n_clusters
    n_components = k
    eigen_tol = 0.0
    eigen_solver = None
    # affinity = 'rbf'  # affinity str or callable, default =’rbf’
    if affinity == 'rbf':
        # params["gamma"] = 1.0  # ?
        # sigma = compute_bandwidth(points)
        # params["gamma"] = 1 / (2 * sigma ** 2)

        # pd = pairwise_distances(points, Y=None, metric='euclidean')
        # # v = np.quantile(pd, q=q)
        # # params["gamma"] = 1/(2*v**2)
        # # Step 2: Calculate the standard deviation of pairwise distances
        # sigma = np.std(pd)
        # params["gamma"] = 1 / (2 * sigma ** 2) * q

        # pd = pairwise_distances(points, Y=None, metric='euclidean')
        # beta = q
        # qs = np.quantile(pd, q=beta, axis=1)
        # alpha = 0.01
        # n, d = points.shape
        # df = d  # degrees of freedom
        # denominator = np.sqrt(stats.chi2.ppf((1 - alpha), df))
        # bandwidth = np.quantile(qs, (1 - alpha)) / denominator
        # params["gamma"] = 1 / (2 * bandwidth ** 2)

        # params["degree"] = 3
        # params["coef0"] = 1
        # # eigen_solver{‘arpack’, ‘lobpcg’, ‘amg’}, default=None
        # The eigenvalue decomposition strategy to use. AMG requires pyamg to be installed.
        # It can be faster on very large, sparse problems, but may also lead to instabilities.
        # If None, then 'arpack' is used. See [4] for more details regarding 'lobpcg'.

        sigma = compute_bandwidth(points, q=q)
        gamma = 1 / (2 * sigma ** 2)
        affinity_matrix_ = pairwise_kernels(
            points, metric=affinity, filter_params=True, gamma=gamma,
        )
        np.fill_diagonal(affinity_matrix_, 0)
    else:  # if affinity == "nearest_neighbors":
        mode = "connectivity" # "distance"
        connectivity = kneighbors_graph(
            points, n_neighbors=n_neighbors, metric='euclidean', include_self=False, mode=mode)
        # affinity_matrix_ = 0.5 * (connectivity + connectivity.T).toarray()
        affinity_matrix_ = connectivity.maximum(connectivity.T).toarray()  # make the graph undirected
        if mode == 'distance':
            # affinity_matrix_ = 1/affinity_matrix_
            # the bigger the distance, the smaller the similarity
            affinity_matrix_ = np.where(affinity_matrix_ != 0, 1 / affinity_matrix_, 0)
    # print(affinity_matrix_)
    # We now obtain the real valued solution matrix to the
    # relaxed Ncut problem, solving the eigenvalue problem
    # L_sym x = lambda x  and recovering u = D^-1/2 x.
    # The first eigenvector is constant only for fully connected graphs
    # and should be kept for spectral clustering (drop_first = False)
    # See spectral_embedding documentation.
    from sklearn.manifold import spectral_embedding
    n_connected_components, _ = connected_components(affinity_matrix_)
    print(n_connected_components)
    plot_graph( csr_matrix(affinity_matrix_), points)

    H = spectral_embedding(
        affinity_matrix_,  # n xn
        n_components=n_components,
        eigen_solver=eigen_solver,
        random_state=random_state,
        eigen_tol=eigen_tol,
        norm_laplacian=True,
        drop_first=False,
    )
    # print(np.min(points), np.max(points), np.min(affinity_matrix_), np.max(affinity_matrix_), np.min(maps), np.max(maps), flush=True)
    # MAX=1e+5
    # maps[maps > MAX] = MAX  # avoid overflow in np.square()
    # maps[maps < -MAX] = -MAX
    # print(np.min(points), np.max(points), np.min(affinity_matrix_), np.max(affinity_matrix_), np.min(maps),
    #       np.max(maps), flush=True)
    if normalize:
        projected_points = H / np.linalg.norm(H, axis=1)[:, None]
    else:
        projected_points = H
    return projected_points


def plot_data(points, title='', nrows=1, ncols=2, random_state=42):
    # Create a color map for 5 classes
    colors = ['red', 'green', 'blue', 'purple', 'orange']

    # Plot the figures
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6))
    axes = axes.reshape((nrows, ncols))
    i = 0
    x1, x2 = points[:, 0], points[:, 1]
    axes[0, 0].scatter(x1, x2, color=colors[i], label=f'{i}')
    axes[0, 0].set_title(title)
    axes[0, 0].set_xlabel('X axis')
    axes[0, 0].set_ylabel('Y axis')
    axes[0, 0].legend()

    plt.show()


def plot_Xs(Xs, title='', nrows=1, ncols=5, random_state=42):
    # Create a color map for 5 classes
    colors = ['red', 'green', 'blue', 'purple', 'orange']

    # Plot the figures
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4))
    axes = axes.reshape((nrows, ncols))

    for i, X in enumerate(Xs):
        # print(i, X)
        x1, x2 = X[:, 0], X[:, 1]
        axes[0, i].scatter(x1, x2, color=colors[i], label=f'{i}')
        axes[0, i].set_title(title)
        # axes[0, i].set_xlabel('X axis')
        # axes[0, i].set_ylabel('Y axis')
        # axes[0, i].legend()

    plt.show()

def precision(x, p = 3):

    return [round(v, ndigits=p) for v in x]

def plot_eigen(embedded_Xs, title='', nrows=2, ncols=5, random_state=42):
    # Create a color map for 5 classes
    colors = ['red', 'green', 'blue', 'purple', 'orange']

    # Plot the figures
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4))
    axes = axes.reshape((nrows, ncols))
    n, d = embedded_Xs.shape
    for j in range(ncols):
        x = range(n)
        y = precision(embedded_Xs[:, j], p=20)
        axes[0, j].scatter(x, y, label=f'{j}-th eigenvector')
        title = f'{j}-th eigenvector'
        axes[0, j].set_title(title)

        # 1 histgram of y
        print(j, np.unique(y, return_counts=True))
        axes[1, j].hist(y, label=f'{j}-th eigenvector')
        title = f'{j}-th eigenvector histgram'
        axes[1, j].set_title(title)

        # axes[0, i].set_xlabel('X axis')
        # axes[0, i].set_ylabel('Y axis')
        # axes[0, i].legend()
    plt.tight_layout()
    plt.show()

def analyze_rsc():
    from sklearn.manifold import SpectralEmbedding
    from sklearn.datasets import make_blobs
    import numpy as np

    # Example data
    # X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    rng = np.random.RandomState(seed=42)
    # points = np.asarray([[1, 0], [0, 1], [5, 6], [1, 5]])  # [10, 20], [10, 15]
    X1 = rng.normal(0, 1, size=(10, 2))
    X2 = rng.normal(5, 1, size=(10, 2))
    # X3 = np.asarray([[10,10], [20, 10], [-100, -50], [-80, -60]])
    X3 = np.asarray([[100000, 100000]])
    points = np.vstack((X1, X2, X3))
    plot_data(points, title='data')
    # outliers = np.asarray([[10, 10], [20, 20]])
    # find the projected centroids
    # k = 2
    k = points.shape[0]
    n_neighbors = 2
    # true_centroids = np.asarray([[0, 0], [5, 5]])
    # X = np.concatenate([true_centroids, points], axis=0)
    X = points
    X_projs = []
    for q in [0.1,]:    #  0.25, 0.5,
        print(f'\n\nq: {q}')
        X_sc = sc_embedding(X, k, affinity='knn', q=q, n_neighbors=2, normalize=False, random_state=42)
        plot_eigen(X_sc, title='sc')
        plot_Xs([X_sc, ], title=f'knn with q:{q}')

        # rsc = RSC(k, nn=n_neighbors, affinity='rbf', q=q, theta=4, m=0.5, random_state=42)
        # Ag, Ac, X_rsc = rsc._RSC__latent_decomposition(X)
        #
        # plot_Xs([X_sc, X_rsc], title=f'rbf with q:{q}')
        # print(f'X_sc: {X_sc}')
        # print(f'X_rsc: {X_rsc}')

        # for nn in [1, 2, 3]:
        #     X_ = sc_embedding(X, k, affinity='knn', q=0.2, n_neighbors=nn, normalize=False, random_state=42)
        # X_projs.append(X_)
    # plot_Xs(X_projs, title='sc+rbf')
    # print(X_projected)


if __name__ == '__main__':
    analyze_rsc()
