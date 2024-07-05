import warnings

from scipy.sparse import csr_matrix

# from base import sc_projection
from base import *
from robust_spectral_clustering import RSC, rbf_graph
import matplotlib.pyplot as plt
from base import plot_projected_data
import numpy as np
np.random.seed(42)


# # Example usage:
# A = np.array([[4, 2],
#               [2, 3]], dtype=float)

# eigenvalues, eigenvectors = qr_algorithm(A)
# print("Eigenvalues:", eigenvalues)
# print("Eigenvectors:\n", eigenvectors)



def robust_sc_projection2(points, k, n_neighbors=15, random_state=42):
    """ Robust Spectral clustering
        https://github.com/abojchevski/rsc/tree/master
        """
    rsc = RSC(k=k, nn=n_neighbors, theta=20, m=0.5,laplacian=1,  verbose=False)
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
def robust_sc_projection(points, k, n_neighbors=5, random_state=42):
    """ Robust Spectral clustering
        https://github.com/abojchevski/rsc/tree/master
        """
    # np.random.seed(random_state)

    # compute the KNN graph
    A = kneighbors_graph(X=points, n_neighbors=n_neighbors, metric='euclidean', include_self=False, mode='connectivity')
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


def plot_data(points, title='', nrows=1, ncols=2, random_state=42):
    # Create a color map for 5 classes
    colors = ['red', 'green', 'blue', 'purple', 'orange']

    # Plot the figures
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6))
    axes = axes.reshape((nrows, ncols))
    i = 0
    x1, x2 = points[:, 0], points[:,1]
    axes[0, 0].scatter(x1, x2, color=colors[i], label=f'{i}')
    axes[0, 0].set_title(title)
    axes[0, 0].set_xlabel('X axis')
    axes[0, 0].set_ylabel('Y axis')
    axes[0, 0].legend()

    plt.show()


def test1():
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
            n_neighbors = 2
            # clustering_method = 'robust_spectral_clustering'
            # np.random.seed(42)
            projected_points = robust_sc_projection(points, k, n_neighbors=n_neighbors, random_state=random_state)

            # plot_centroids_diff(points, projected_points, cluster_size=100, clustering_method=clustering_method, random_state=random_state)

def plot_data2():

    from sklearn.manifold import SpectralEmbedding
    from sklearn.datasets import make_blobs
    import numpy as np

    # Example data
    # X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

    points = np.asarray([[1, 0], [0,1], [5,6], [6, 6],[10,20], [10, 15]])
    # outliers = np.asarray([[10, 10], [20, 20]])
    # find the projected centroids
    k = 2
    n_neighbors  = 5
    true_centroids = np.asarray([[0, 0], [5, 5]])
    X = np.concatenate([true_centroids, points], axis=0)
    X_projected = sc_projection(X, k, affinity='knn', n_neighbors=n_neighbors, normalize=False, random_state=42)
    # X_projected = rsc_projection(X, k, n_neighbors=n_neighbors, theta=0, random_state=42)

    for i in range(X_projected.shape[1]-1):
        projected_true_centroids = X_projected[:k, i:i+2]
        projected_points = X_projected[k:, i:i+2]
        print('\n', X_projected)
        # if clustering_method == 'sc_k_medians_l2':
        plot_projected_data(points, projected_points, cluster_size=100, clustering_method= f'rsc, {i}:{i+2}',
                            centroids=true_centroids, projected_centroids=projected_true_centroids,
                            n_clusters=k, out_dir = "", x_axis= 'location', random_state=42)


    # Fit the model
    # X_transformed = embedding.fit_transform(X)


    import matplotlib.pyplot as plt
    import networkx as nx
    from sklearn.neighbors import kneighbors_graph

    # Create a k-neighbors graph
    A = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=False)
    A = A.maximum(A.T)
    # Convert to a NetworkX graph
    # G = nx.from_scipy_sparse_matrix(A)
    G = nx.from_scipy_sparse_array(A)

    # Draw the graph
    plt.figure(figsize=(8, 6))
    # Define positions for nodes
    pos = {i:(x1, x2) for i, (x1, x2) in enumerate(X)}

    # nx.draw(G, node_size=50)
    nx.draw_networkx(G, pos, node_size=50)
    plt.show()




    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances
    from sklearn.manifold import spectral_embedding
    from scipy.sparse import csgraph
    from scipy.sparse.linalg import eigsh

    # Generate sample data
    X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

    # Compute the affinity matrix (similarity matrix)
    affinity_matrix = pairwise_distances(X, metric='euclidean')
    sigma = np.mean(affinity_matrix)
    affinity_matrix = np.exp(-affinity_matrix ** 2 / (2. * sigma ** 2))

    # Compute the graph Laplacian
    laplacian = csgraph.laplacian(affinity_matrix, normed=True)

    # Compute the first k eigenvalues and eigenvectors of the Laplacian
    n_components = 2
    eigenvalues, eigenvectors = eigsh(laplacian, k=n_components + 1, which='SM')
    eigenvectors = eigenvectors[:, 1:]  # Drop the first eigenvector corresponding to the zero eigenvalue

    # Normalize the embedded points
    normalize = False
    if normalize:
        projected_points = eigenvectors / np.linalg.norm(eigenvectors, axis=1, keepdims=True)
    else:
        projected_points = eigenvectors

    print(normalize, projected_points)

    # Perform k-means clustering on the projected points
    kmeans = KMeans(n_clusters=n_components, random_state=42)
    labels = kmeans.fit_predict(projected_points)

    # Plot the results
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
    plt.title('Spectral Clustering with Normalization' if normalize else 'Spectral Clustering without Normalization')
    plt.show()


def plot_data2():
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.neighbors import kneighbors_graph

    # Generate sample data
    np.random.seed(0)
    X = np.random.rand(10, 2)  # 10 points in 2D space

    # Create a k-nearest neighbors graph
    k = 3  # Number of neighbors
    connectivity = kneighbors_graph(X, n_neighbors=k,
                                    include_self=True)  # Including self-loops for better visibility in the matrix

    # Convert the sparse matrix to a dense format (adjacency matrix)
    adjacency_matrix = connectivity.toarray()

    # Visualize the adjacency matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(adjacency_matrix, annot=True, cmap='viridis', cbar=False)
    plt.title('k-Nearest Neighbors Graph as Adjacency Matrix')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    plt.show()


# plot_data2()

def compare_median_means():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    # Sample dataset
    X = np.array([
        [1, 1], [2, 1], [1, 2],  # Cluster 1
        [5, 5], [6, 5], [5, 6],  # Cluster 2
        [9, 9], [10, 9], [9, 10]  # Cluster 3
    ])

    seed = 0
    np.random.seed(seed)
    k = 3
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    print(centroids)

    # Apply K-means
    from _kmeans import KMeans
    kmeans = KMeans(n_clusters=k, init=centroids, n_init=1, random_state=seed).fit(X)
    kmeans_centroids = kmeans.cluster_centers_

    # Function to calculate Manhattan distance
    def manhattan_distance(a, b):
        return np.abs(a - b).sum(axis=1)

    # Function to apply K-medians
    def kmedians(X, k, max_iters=100, init=None):
        centroids = init

        for _ in range(max_iters):
            # Assign points to the nearest centroid based on Manhattan distance
            labels = np.argmin(np.array([manhattan_distance(X, centroid) for centroid in centroids]).T, axis=1)

            # Calculate the median for each cluster
            new_centroids = np.array([np.median(X[labels == i], axis=0) for i in range(k)])

            # If centroids don't change, break
            if np.all(centroids == new_centroids):
                break

            centroids = new_centroids

        return centroids

    # Apply K-medians
    kmedians_centroids = kmedians(X, k, init=centroids)

    # Plotting the results
    plt.scatter(X[:, 0], X[:, 1], c='blue', label='Data Points')
    plt.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], c='red', label='K-means Centroids', marker='x')
    plt.scatter(kmedians_centroids[:, 0], kmedians_centroids[:, 1], c='green', label='K-medians Centroids', marker='o')
    plt.legend()
    plt.title('K-means vs K-medians Centroids')
    plt.show()

def compare_sc_rsc():
    # compare_median_means()
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_circles, make_moons
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse import csgraph

    # Generate a dataset of two concentric circles
    n_samples = 500
    # X, y = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    X, y = make_moons(n_samples=n_samples, noise=0.1)

    # Visualize the dataset
    plt.figure(figsize=(8, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
    plt.title("Original Data")
    plt.show()

    for q in range(1, 10):
        q = q * 0.05
        pair_dists = pairwise_distances(X, Y=None, metric='euclidean')
        # beta = 0.3
        # qs = np.quantile(pd, q=q, axis=1)
        # alpha = 0.01
        # n, d = X.shape
        # df = d  # degrees of freedom
        # denominator = np.sqrt(stats.chi2.ppf((1 - alpha), df))
        # sigma = np.quantile(qs, (1 - alpha)) / denominator
        sigma = np.quantile(pair_dists, q)
        gamma = 1 / (2*sigma ** 2)
        # RBF (Gaussian) kernel similarity matrix
        rbf_similarity = np.exp(-gamma * pairwise_distances(X, squared=True))
        # Perform spectral clustering using RBF kernel
        sc_rbf = SpectralClustering(n_clusters=2, affinity='precomputed')
        labels_rbf = sc_rbf.fit_predict(rbf_similarity)

        # Visualize the results of spectral clustering with RBF kernel
        plt.figure(figsize=(8, 4))
        plt.scatter(X[:, 0], X[:, 1], c=labels_rbf, cmap='viridis', s=50)
        plt.title(f"Spectral Clustering with RBF Kernel with gamma:{gamma} and q:{q}")
        plt.show()
    #
    m = X.shape[0]//2
    for k in [1, 2, 5, 10, 20, 50, 100, 200, 400]:
        # kNN similarity matrix
        # k = 10  # Number of neighbors
        knn_similarity = kneighbors_graph(X, n_neighbors=k, include_self=False).toarray()
        knn_similarity = 0.5 * (knn_similarity + knn_similarity.T)  # Make the graph symmetric

        # Perform spectral clustering using kNN
        sc_knn = SpectralClustering(n_clusters=2, affinity='precomputed')
        labels_knn = sc_knn.fit_predict(knn_similarity)

        # Visualize the results of spectral clustering with kNN
        plt.figure(figsize=(6,6))
        plt.scatter(X[:, 0], X[:, 1], c=labels_knn, cmap='viridis', s=50)
        plt.title(f"Spectral Clustering with kNN with k={k}")
        plt.show()



def plot_Xs(Xs, title='', nrows=1, ncols=5, random_state=42):
    # Create a color map for 5 classes
    colors = ['red', 'green', 'blue', 'purple', 'orange']

    # Plot the figures
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4))
    axes = axes.reshape((nrows, ncols))

    for i, X in enumerate(Xs):
        print(i, X)
        x1, x2 = X[:, 0], X[:,1]
        axes[0, i].scatter(x1, x2, color=colors[i], label=f'{i}')
        axes[0, i].set_title(title)
        # axes[0, i].set_xlabel('X axis')
        # axes[0, i].set_ylabel('Y axis')
        # axes[0, i].legend()

    plt.show()



# @timer
def sc_embedding(points, k, n_neighbors=10, affinity = 'knn', q=1, normalize=False, random_state=42):
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

        pd = pairwise_distances(points, Y=None, metric='euclidean')
        # v = np.quantile(pd, q=q)
        # params["gamma"] = 1/(2*v**2)
        # Step 2: Calculate the standard deviation of pairwise distances
        sigma = np.std(pd)
        params["gamma"] = 1 / (2 * sigma ** 2) * q


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
        affinity_matrix_ = pairwise_kernels(
            points, metric=affinity, filter_params=True, **params,
        )
        np.fill_diagonal(affinity_matrix_, 0)
    else: # if affinity == "nearest_neighbors":
        mode = "distance"
        connectivity = kneighbors_graph(
            points, n_neighbors=n_neighbors, metric='euclidean', include_self=False, mode=mode)
        # affinity_matrix_ = 0.5 * (connectivity + connectivity.T).toarray()
        affinity_matrix_ = connectivity.maximum(connectivity.T).toarray()  # make the graph undirected
        if mode == 'distance':
            # affinity_matrix_ = 1/affinity_matrix_
            # the bigger the distance, the smaller the similarity
            affinity_matrix_ = np.where(affinity_matrix_ != 0, 1 / affinity_matrix_, 0)
    print(affinity_matrix_)
    # We now obtain the real valued solution matrix to the
    # relaxed Ncut problem, solving the eigenvalue problem
    # L_sym x = lambda x  and recovering u = D^-1/2 x.
    # The first eigenvector is constant only for fully connected graphs
    # and should be kept for spectral clustering (drop_first = False)
    # See spectral_embedding documentation.
    from sklearn.manifold import spectral_embedding
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


def analyze_rsc():
    from sklearn.manifold import SpectralEmbedding
    from sklearn.datasets import make_blobs
    import numpy as np

    # Example data
    # X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

    points = np.asarray([[1, 0], [0, 1], [5, 6], [1000, 500]])   # [10, 20], [10, 15]
    plot_data(points,title='data')
    # outliers = np.asarray([[10, 10], [20, 20]])
    # find the projected centroids
    k = 2
    n_neighbors = 2
    # true_centroids = np.asarray([[0, 0], [5, 5]])
    # X = np.concatenate([true_centroids, points], axis=0)
    X = points
    X_projs = []
    for q in [0.1, 0.25, 0.5, 1, 5]:
        # X_ = sc_embedding(X, k, affinity='rbf', q=q, n_neighbors=0, normalize=False, random_state=42)
        rsc = RSC(X, nn=n_neighbors, affinity='rbf', theta=4, m=0.5, random_state=42)
        Ag, Ac, X_ = rsc._RSC__latent_decomposition(X)
        # for nn in [1, 2, 3]:
        #     X_ = sc_embedding(X, k, affinity='knn', q=0.2, n_neighbors=nn, normalize=False, random_state=42)
        X_projs.append(X_)
    plot_Xs(X_projs, title='sc+rbf')
    # print(X_projected)



    # Ag, Ac, H = rsc_embedding(X, k, affinity='knn', n_neighbors=n_neighbors, theta=4, m=0.5, random_state=42)
    # plot_data(H, title='rsc')
    # print(Ag.toarray(), H)
    #
    # for i in range(X_projected.shape[1] - 1):
    #     projected_true_centroids = X_projected[:k, i:i + 2]
    #     projected_points = X_projected[k:, i:i + 2]
    #     print('\n', X_projected)
    #     # if clustering_method == 'sc_k_medians_l2':
    #     plot_projected_data(points, projected_points, cluster_size=100, clustering_method=f'rsc, {i}:{i + 2}',
    #                         centroids=true_centroids, projected_centroids=projected_true_centroids,
    #                         n_clusters=k, out_dir="", x_axis='location', random_state=42)


# analyze_rsc()

def eigen_decomposition():
    import numpy as np
    import matplotlib.pyplot as plt

    # Seed for reproducibility
    np.random.seed(0)

    # Generate a clean dataset
    mean1 = [0, 0]
    cov1 = [[1, 0.], [0., 1]]  # diagonal covariance
    data1 = np.random.multivariate_normal(mean1, cov1, 50)

    mean2 = [0, 5]
    cov2 = [[1, 0.], [0., 1]]  # diagonal covariance
    data2 = np.random.multivariate_normal(mean2, cov2, 50)

    # Combine the datasets
    clean_data = np.vstack((data1, data2))

    # Plot the clean data
    plt.scatter(clean_data[:, 0], clean_data[:, 1])
    plt.title('Clean Data')
    plt.show()

    from sklearn.decomposition import PCA

    # Perform PCA
    pca_clean = PCA(n_components=2)
    pca_clean.fit(clean_data)
    components_clean = pca_clean.components_
    print(pca_clean.singular_values_, pca_clean.components_, np.linalg.norm(pca_clean.components_, axis=1))

    # Plot PCA components
    plt.scatter(clean_data[:, 0], clean_data[:, 1])
    for component in components_clean:
        plt.quiver(0, 0, component[0], component[1], angles='xy', scale_units='xy', scale=1, color='r')
    plt.title('PCA on Clean Data')
    plt.show()

    # Add outliers
    outliers = np.array([[10, 10], [12, 12], [15, 15]])
    data_with_outliers = np.vstack((clean_data, outliers))

    # # Add noise
    # noise = np.random.normal(0, 0.5, data_with_outliers.shape)
    # data_with_noise = data_with_outliers + noise

    # Plot data with outliers and noise
    plt.scatter(data_with_outliers[:, 0], data_with_outliers[:, 1])
    plt.title('Data with Outliers and Noise')
    plt.show()

    # Perform PCA
    pca_noisy = PCA(n_components=2)
    pca_noisy.fit(data_with_outliers)
    components_noisy = pca_noisy.components_
    print(pca_noisy.singular_values_, pca_noisy.components_,np.linalg.norm(pca_noisy.components_, axis=1))

    # Plot PCA components
    plt.scatter(data_with_outliers[:, 0], data_with_outliers[:, 1])
    for component in components_noisy:
        plt.quiver(0, 0, component[0], component[1], angles='xy', scale_units='xy', scale=1, color='r')
    plt.title('PCA on Data with Outliers and Noise')
    plt.show()

eigen_decomposition()