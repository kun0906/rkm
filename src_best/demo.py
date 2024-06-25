
# from base import sc_projection
from base import *
from robust_spectral_clustering import RSC
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

def plot_data():

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

# compare_median_means()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic dataset
n_samples = 300
n_clusters = 3
X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=0.60, random_state=0)

# Apply k-means clustering to the original dataset
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
original_inertia = kmeans.inertia_

# Plot original clusters
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')
plt.title(f'Original Clusters with Inertia: {original_inertia:.2f}')
plt.show()

# Add outliers to the dataset
outliers = np.array([[8, 8], [9, 9], [100, 10], [11, 11]])
X_with_outliers = np.vstack([X, outliers])

# Apply k-means clustering to the dataset with outliers
kmeans_with_outliers = KMeans(n_clusters=n_clusters, random_state=0).fit(X_with_outliers)
outliers_inertia = kmeans_with_outliers.inertia_

# Plot clusters with outliers
plt.scatter(X_with_outliers[:, 0], X_with_outliers[:, 1], c=kmeans_with_outliers.labels_, s=50, cmap='viridis')
plt.scatter(kmeans_with_outliers.cluster_centers_[:, 0], kmeans_with_outliers.cluster_centers_[:, 1], s=200, c='red', marker='X')
plt.title(f'Clusters with Outliers and Inertia: {outliers_inertia:.2f}')
plt.show()

print(f"Original Inertia: {original_inertia:.2f}")
print(f"Inertia with Outliers: {outliers_inertia:.2f}")
