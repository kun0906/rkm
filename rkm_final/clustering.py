import collections
import warnings

import numpy as np

tot_iterate = 50
tolerance = 1e-4
def kmeans(points, k, centroids_input, max_iterations=tot_iterate):
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
    # print(f'kmeans iterations: {i}')
    return new_centroids, labels


def sc_kmeans(projected_points, points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None, true_labels=None):
    new_centroids = np.copy(centroids_input)

    for i in range(max_iterations):
        # Assign each point to the closest centroid
        try:
            distances = np.sqrt(np.sum((projected_points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
        except RuntimeWarning as e:
            # Code to handle the RuntimeWarning
            print("Overflow encountered in square operation")
            print(i, e)
        labels = np.argmin(distances, axis=1)

        pre_centroids = np.copy(new_centroids)
        # Update the centroids to be the mean of the points in each cluster
        for j in range(k):
            if sum(labels == j) == 0:
                # new_centroids[j] use the previous centroid
                continue
            new_centroids[j] = np.mean(projected_points[labels == j], axis=0)

        if np.sum((new_centroids - pre_centroids) ** 2) / k < tolerance:
            break

    # find the labels on the projected data first. Here should be L2
    distances = np.sqrt(np.sum((projected_points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    labels = np.argmin(distances, axis=1)

    # according to labels, find the final centroids on the original centroids.
    new_centroids = np.zeros((k, points.shape[1]))
    # Update the centroids to be the median of the points in each cluster
    for j in range(k):
        if sum(labels == j) == 0:
            # new_centroids[j] use the previous centroid
            raise ValueError('no data to the cluster')
            # continue
        new_centroids[j] = np.mean(points[labels == j], axis=0)  # original data points.

    # L2
    distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    labels = np.argmin(distances, axis=1)

    return new_centroids, labels

def kmed(points, k, centroids_input, max_iterations=tot_iterate):
    new_centroids = np.copy(centroids_input)

    for i in range(max_iterations):
        # Assign each point to the closest centroid
        distances = np.sum(np.abs(points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
        labels = np.argmin(distances, axis=1)

        pre_centroids = np.copy(new_centroids)
        # Update the centroids to be the median of the points in each cluster
        for j in range(k):
            if sum(labels == j) == 0:
                # new_centroids[j] use the previous centroid
                continue
            new_centroids[j] = np.median(points[labels == j], axis=0)

        if np.sum((new_centroids - pre_centroids) ** 2) / k < tolerance:
            break

    # It should be L1, not L2
    # distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    distances = np.sum(np.abs(points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
    labels = np.argmin(distances, axis=1)
    # print(f'kmedL1 iterations: {i}')
    return new_centroids, labels


def sc_kmed(projected_points, points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None, true_labels=None):
    new_centroids = np.copy(centroids_input)

    for i in range(max_iterations ):
        # Assign each point to the closest centroid
        distances = np.sum(np.abs(projected_points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
        labels = np.argmin(distances, axis=1)

        pre_centroids = np.copy(new_centroids)
        # Update the centroids to be the median of the points in each cluster
        for j in range(k):
            if sum(labels == j) == 0:
                # new_centroids[j] use the previous centroid
                continue
            new_centroids[j] = np.median(projected_points[labels == j], axis=0)

        if np.sum((new_centroids - pre_centroids) ** 2) / k < tolerance:
            break


    # find the labels on the projected data first. Here should be L2
    distances = np.sum(np.abs(projected_points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
    labels = np.argmin(distances, axis=1)

    # according to labels, find the final centroids on the original centroids.
    new_centroids = np.zeros((k, points.shape[1]))
    # Update the centroids to be the median of the points in each cluster
    for j in range(k):
        if sum(labels == j) == 0:
            # new_centroids[j] use the previous centroid
            raise ValueError('no data to the cluster')
            # continue
        new_centroids[j] = np.median(points[labels == j], axis=0)  # original data points.

    # note that it should be L1, not L2
    # distances2 = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    distances = np.sum(np.abs(points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
    labels = np.argmin(distances, axis=1)

    return new_centroids, labels
def lloydL1(points, k, centroids_input, max_iterations=tot_iterate):

    new_centroids = np.copy(centroids_input)

    for i in range(max_iterations):
        # Assign each point to the closest centroid
        distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
        labels = np.argmin(distances, axis=1)

        pre_centroids = np.copy(new_centroids)
        # Update the centroids to be the median of the points in each cluster
        for j in range(k):
            if sum(labels == j) == 0:
                # new_centroids[j] use the previous centroid
                continue
            new_centroids[j] = np.median(points[labels == j], axis=0)
        # print(i, collections.Counter(labels))
        if np.sum((new_centroids - pre_centroids) ** 2) / k < tolerance:
            break

    # it should be L2
    distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    labels = np.argmin(distances, axis=1)
    return new_centroids, labels


def sc_lloydL1(projected_points, points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None, true_labels=None):
    new_centroids = np.copy(centroids_input)

    for i in range(max_iterations):
        # Assign each point to the closest centroid
        try:
            distances = np.sqrt(np.sum((projected_points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
        except RuntimeWarning as e:
            print(i, e)
        labels = np.argmin(distances, axis=1)

        pre_centroids = np.copy(new_centroids)
        # Update the centroids to be the median of the points in each cluster
        for j in range(k):
            if sum(labels == j) == 0:
                # new_centroids[j] use the previous centroid
                continue
            new_centroids[j] = np.median(projected_points[labels == j], axis=0)

        if np.sum((new_centroids - pre_centroids) ** 2) / k < tolerance:
            break

    # find the labels on the projected data first. Here should be L2
    distances = np.sqrt(np.sum((projected_points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    labels = np.argmin(distances, axis=1)

    # according to labels, find the final centroids on the original centroids.
    new_centroids = np.zeros((k, points.shape[1]))
    # Update the centroids to be the median of the points in each cluster
    for j in range(k):
        if sum(labels == j) == 0:
            # new_centroids[j] use the previous centroid
            raise ValueError('no data to the cluster')
            # continue
        new_centroids[j] = np.median(points[labels == j], axis=0)  # original data points.

    # find the labels on the projected data first. Here should be L2
    distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    labels = np.argmin(distances, axis=1)
    return new_centroids, labels

def geom_kmed(points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None, true_labels=None):
    new_centroids = np.copy(centroids_input)
    from geom_median.numpy import compute_geometric_median  # NumPy API

    for i in range(max_iterations ):
        # Assign each point to the closest centroid
        distances = np.sum(np.abs(points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
        labels = np.argmin(distances, axis=1)

        pre_centroids = np.copy(new_centroids)
        # Update the centroids to be the median of the points in each cluster
        for j in range(k):
            if sum(labels == j) == 0:
                # new_centroids[j] use the previous centroid
                continue
            # new_centroids[j] = np.median(points[labels == j], axis=0)
            # https://github.com/krishnap25/geom_median/blob/main/README.md
            _out = compute_geometric_median(points[labels == j])
            # Access the median via `out.median`, which has the same shape as the points, i.e., (d,)
            new_centroids[j] = _out.median

        if np.sum((new_centroids - pre_centroids) ** 2) / k < tolerance:
            break

    # note that it should be L1, not L2
    # distances2 = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    distances = np.sum(np.abs(points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
    labels = np.argmin(distances, axis=1)

    return new_centroids, labels



def sc_projection(points, k, random_state):
    from sklearn.metrics import pairwise_kernels
    params = {}  # default value in slkearn
    # https://github.com/scikit-learn/scikit-learn/blob/872124551/sklearn/cluster/_spectral.py#L667
    params["gamma"] = 1.0  # ?
    params["degree"] = 3
    params["coef0"] = 1

    # Number of eigenvectors to use for the spectral embedding, default=n_clusters
    n_components = k
    affinity = 'rbf'  # affinity str or callable, default =’rbf’
    eigen_tol = 0.0
    eigen_solver = None
    # eigen_solver{‘arpack’, ‘lobpcg’, ‘amg’}, default=None
    # The eigenvalue decomposition strategy to use. AMG requires pyamg to be installed.
    # It can be faster on very large, sparse problems, but may also lead to instabilities.
    # If None, then 'arpack' is used. See [4] for more details regarding 'lobpcg'.

    affinity_matrix_ = pairwise_kernels(
        points, metric=affinity, filter_params=True, **params,
    )

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
    maps[maps > 1e+30] = 1e+30  # avoid overflow in np.square()
    return maps

def sc_omniscient(points, centroids_input, k, max_iterations=tot_iterate, clustering_method='kmeans', random_state=42):
    """ Spectral clustering in sklearn

    Parameters
    ----------
    points
    k
    centroids_input
    max_iterations

    Returns
    -------

    """

    projected_points = sc_projection(points, k, random_state=random_state)

    # X = np.concatenate([true_centroids, points], axis=0)
    # X_projected = sc_projection(X, k, random_state=random_state)
    # projected_true_centroids = X_projected[:k, :]

    # random select initial centroids
    rng = np.random.RandomState(seed=random_state)
    indices = rng.choice(range(len(points)), size=k, replace=False)
    projected_init_centroids = projected_points[indices, :]
    if clustering_method == 'kmeans':
        centroids, labels = sc_kmeans(projected_points, points, k, projected_init_centroids, max_iterations=max_iterations,
                            true_centroids=centroids_input, true_labels=None)
    elif clustering_method == 'Kmed':
        centroids, labels =sc_kmed(projected_points, points, k, projected_init_centroids, max_iterations=max_iterations,
                            true_centroids=centroids_input, true_labels=None)
    elif clustering_method == 'lloydL1':
        centroids, labels = sc_lloydL1(projected_points, points, k, projected_init_centroids, max_iterations=max_iterations,
                            true_centroids=centroids_input, true_labels=None)
    else:
        raise NotImplemented(f'{clustering_method}')


    return centroids, labels



#
# def robust_sc_kmeans(points, k, centroids_input, max_iterations=tot_iterate, random_state=42):
#     """ Robust Spectral clustering
#     https://github.com/abojchevski/rsc/tree/master
#     """
#     from robust_spectral_clustering import RSC
#     n_neighbors = 15
#     rsc = RSC(k=k, nn=n_neighbors, theta=10, verbose=True)
#     # y_rsc = rsc.fit_predict(X)
#     Ag, Ac, H = rsc.__latent_decomposition(points)
#     # Ag: similarity matrix of good points
#     # Ac: similarity matrix of corruption points
#     # A = Ag + Ac
#     rsc.Ag = Ag
#     rsc.Ac = Ac
#
#     if rsc.normalize:
#         rsc.H = H / np.linalg.norm(H, axis=1)[:, None]
#     else:
#         rsc.H = H
#
#     projected_points = rsc.H
#     projected_init_centroids = ?#
#     # Find the labels
#     # Method 1: Use classic kmeans to find the centroids and labels on the projected data.
#     _, kmeans_labels = kmeans(projected_points, centroids_input=projected_init_centroids,
#                               max_iterations=max_iterations, k=k)
#     # Method 2: Use cluster_qr method directly extract clusters from eigenvectors in spectral clustering.
#     # Todo
#
#     # Compute the final centroids based on the labels.
#     kmeans_centroids = np.copy(centroids_input)
#     unique_labels = np.unique(kmeans_labels)  # Returns the sorted unique elements of an array.
#     if k != len(unique_labels):
#         warnings.warn(f'the number of unique final labels ({unique_labels}) ?= k:{k}')
#     for i, label in enumerate(unique_labels):
#         cluster_points = points[kmeans_labels == label]  # on the original data points
#         centroid = np.mean(cluster_points, axis=0)
#         kmeans_centroids[i, :] = centroid
#
#     return kmeans_centroids, kmeans_labels

#
# def reg_sc_kmeans(points, k, centroids_input, max_iterations=tot_iterate, random_state=42):
#     """ Regularized Spectral clustering
#     https://github.com/crisbodnar/regularised-spectral-clustering/tree/master?tab=readme-ov-file
#     """
#     pass