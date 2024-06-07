import collections
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import kneighbors_graph

from utils import *

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


def sc_kmeans(projected_points, points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None,
              true_labels=None):
    new_centroids = np.copy(centroids_input)

    for i in range(max_iterations):
        # Assign each point to the closest centroid
        distances = np.sqrt(np.sum((projected_points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
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
    # TODO: we don't ACD now
    # # Update the centroids to be the median of the points in each cluster
    # for j in range(k):
    #     if sum(labels == j) == 0:
    #         # random select initial centroids
    #         rng = np.random.RandomState(seed=j)
    #         indices = rng.choice(range(len(points)), size=1, replace=False)
    #         # raise ValueError('no data to the cluster')
    #         new_centroids[j] = points[indices[0]]
    #         continue
    #     new_centroids[j] = np.mean(points[labels == j], axis=0)  # original data points.
    #
    # # L2
    # distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    # labels = np.argmin(distances, axis=1)

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


def sc_kmed(projected_points, points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None,
            true_labels=None):
    new_centroids = np.copy(centroids_input)

    for i in range(max_iterations):
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
    # TODO: we don't ACD now
    # # Update the centroids to be the median of the points in each cluster
    # for j in range(k):
    #     if sum(labels == j) == 0:
    #         # new_centroids[j] use the previous centroid
    #         # random select initial centroids
    #         rng = np.random.RandomState(seed=j)
    #         indices = rng.choice(range(len(points)), size=1, replace=False)
    #         # raise ValueError('no data to the cluster')
    #         new_centroids[j] = points[indices[0]]
    #         continue
    #         # continue
    #     new_centroids[j] = np.median(points[labels == j], axis=0)  # original data points.
    #
    # # note that it should be L1, not L2
    # # distances2 = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    # distances = np.sum(np.abs(points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
    # labels = np.argmin(distances, axis=1)

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


def sc_lloydL1(projected_points, points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None,
               true_labels=None):
    new_centroids = np.copy(centroids_input)

    for i in range(max_iterations):
        # Assign each point to the closest centroid
        try:
            distances = np.sqrt(
                np.sum((projected_points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
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
    # TODO: we don't ACD now
    # # Update the centroids to be the median of the points in each cluster
    # for j in range(k):
    #     if sum(labels == j) == 0:
    #         # new_centroids[j] use the previous centroid
    #         print('be careful: ', np.unique(labels), k)
    #         # random select initial centroids
    #         rng = np.random.RandomState(seed=j)
    #         indices = rng.choice(range(len(points)), size=1, replace=False)
    #         # raise ValueError('no data to the cluster')
    #         new_centroids[j] = points[indices[0]]
    #         continue
    #     new_centroids[j] = np.median(points[labels == j], axis=0)  # original data points.
    #
    # # find the labels on the projected data first. Here should be L2
    # distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    # labels = np.argmin(distances, axis=1)
    #
    return new_centroids, labels


def geom_kmed(points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None, true_labels=None):
    new_centroids = np.copy(centroids_input)
    from geom_median.numpy import compute_geometric_median  # NumPy API

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

    # find the projected initial centroids
    X = np.concatenate([centroids_input, points], axis=0)
    X_projected = sc_projection(X, k, random_state=random_state)
    projected_true_centroids = X_projected[:k, :]

    # # random select initial centroids
    # rng = np.random.RandomState(seed=random_state)
    # indices = rng.choice(range(len(points)), size=k, replace=False)
    # projected_init_centroids = projected_points[indices, :]
    #
    if clustering_method == 'kmeans':
        centroids, labels = sc_kmeans(projected_points, points, k, projected_true_centroids,
                                      max_iterations=max_iterations,
                                      true_centroids=centroids_input, true_labels=None)
    elif clustering_method == 'Kmed':
        centroids, labels = sc_kmed(projected_points, points, k, projected_true_centroids,
                                    max_iterations=max_iterations,
                                    true_centroids=centroids_input, true_labels=None)
    elif clustering_method == 'lloydL1':
        centroids, labels = sc_lloydL1(projected_points, points, k, projected_true_centroids,
                                       max_iterations=max_iterations,
                                       true_centroids=centroids_input, true_labels=None)
    else:
        raise NotImplemented(f'{clustering_method}')

    return centroids, labels

def plot_centroids_diff(points, projected_points, cluster_size=100, clustering_method='kmeans', random_state=42):
    # Create a color map for 5 classes
    colors = ['red', 'green', 'blue', 'purple', 'orange']

    # Plot the figures
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    # Plot points
    for i in range(4):
        x1 = points[i*cluster_size: (i+1)*cluster_size, 0]
        x2 = points[i*cluster_size: (i+1)*cluster_size, 1]
        axes[0, 0].scatter(x1, x2, color=colors[i], label=f'Class {i}')
    axes[0, 0].set_title('Points without outliers')
    axes[0, 0].set_xlabel('X axis')
    axes[0, 0].set_ylabel('Y axis')
    axes[0, 0].legend()

    # Plot points
    for i in range(5):
        x1 = points[i*cluster_size: (i+1)*cluster_size, 0]
        x2 = points[i*cluster_size: (i+1)*cluster_size, 1]
        axes[0, 1].scatter(x1, x2, color=colors[i], label=f'Class {i}')
    axes[0, 1].set_title('Points')
    axes[0, 1].set_xlabel('X axis')
    axes[0, 1].set_ylabel('Y axis')
    axes[0, 1].legend()


    # Plot projected points
    for i in range(4):
        x1 = projected_points[i*cluster_size: (i+1)*cluster_size, 0]
        x2 = projected_points[i*cluster_size: (i+1)*cluster_size, 1]
        axes[1, 0].scatter(x1, x2, color=colors[i], label=f'Class {i}')
    axes[1, 0].set_title('Projected Points without outliers')
    axes[1, 0].set_xlabel('X axis')
    axes[1, 0].set_ylabel('Y axis')
    axes[1, 0].legend()

    # Plot projected points
    for i in range(5):
        x1 = projected_points[i*cluster_size: (i+1)*cluster_size, 0]
        x2 = projected_points[i*cluster_size: (i+1)*cluster_size, 1]
        axes[1, 1].scatter(x1, x2, color=colors[i], label=f'Class {i}')
    axes[1, 1].set_title('Projected Points')
    axes[1, 1].set_xlabel('X axis')
    axes[1, 1].set_ylabel('Y axis')
    axes[1, 1].legend()

    fig.suptitle(f'Seed: {random_state},{clustering_method}')
    plt.tight_layout()
    plt.savefig('diff.png')
    plt.show()

def robust_sc_omniscient(points, k, centroids_input, max_iterations=tot_iterate, random_state=42,
                     clustering_method='kmeans', n_neighbours=15):

    projected_points = robust_sc_projection(points, k, n_neighbours=n_neighbours, random_state=random_state)

    # plot_centroids_diff(points, projected_points, cluster_size=100, clustering_method=clustering_method, random_state=random_state)

    # find the projected initial centroids
    X = np.concatenate([centroids_input, points], axis=0)
    X_projected = robust_sc_projection(X, k, n_neighbours=n_neighbours, random_state=random_state)
    projected_true_centroids = X_projected[:k, :]

    # # random select initial centroids
    # rng = np.random.RandomState(seed=random_state)
    # indices = rng.choice(range(len(points)), size=k, replace=False)
    # projected_init_centroids = projected_points[indices, :]
    #
    if clustering_method == 'kmeans':
        centroids, labels = sc_kmeans(projected_points, points, k, projected_true_centroids,
                                      max_iterations=max_iterations,
                                      true_centroids=centroids_input, true_labels=None)
    elif clustering_method == 'Kmed':
        centroids, labels = sc_kmed(projected_points, points, k, projected_true_centroids,
                                    max_iterations=max_iterations,
                                    true_centroids=centroids_input, true_labels=None)
    elif clustering_method == 'lloydL1':
        centroids, labels = sc_lloydL1(projected_points, points, k, projected_true_centroids,
                                       max_iterations=max_iterations,
                                       true_centroids=centroids_input, true_labels=None)
    else:
        raise NotImplemented(f'{clustering_method}')

    return centroids, labels

#
# def reg_sc_kmeans(points, k, centroids_input, max_iterations=tot_iterate, random_state=42):
#     """ Regularized Spectral clustering
#     https://github.com/crisbodnar/regularised-spectral-clustering/tree/master?tab=readme-ov-file
#     """
#     pass
