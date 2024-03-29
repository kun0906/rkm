import collections

import numpy as np
import copy
import itertools
import matplotlib.pyplot as plt


def plot_cluster(points, labels, new_label1, new_label2):
    labels = np.asarray(labels)
    new_label1 = np.asarray(new_label1)
    new_label2 = np.asarray(new_label2)

    ls_lst = [labels, new_label1, new_label2]
    names = ['true labels', 'before alignment', 'after alignment']
    colors = ['g', 'purple', 'b']
    markers = ['*', 'o', '+']
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for i in range(3):
        for l in [0, 1]:
            ls = ls_lst[i]
            data = points[ls == l]
            m = len(data)
            axes[i].scatter(data[:, 0], data[:, 1], marker=markers[l], color=colors[l], s=100)
        axes[i].set_title(names[i])
    plt.tight_layout()
    plt.show()


def align_centroids(centroids, true_centroids, method='name'):
    # print(f'centroids before: {centroids}')
    # print(f"{len(centroids)} centroids include {len(list(itertools.permutations(centroids)))} permutations")
    c1 = copy.deepcopy(true_centroids)
    # check which point is close to which true centroids.
    min_d = np.inf
    indices = range(len(centroids))
    for _indices in list(itertools.permutations(indices)):
        c = centroids[_indices, :]
        d = np.sum(np.sum(np.square(c - c1), axis=1), axis=0)
        if d < min_d:
            # print(method, d, min_d)
            min_d = np.copy(d) # here is just a float, so there is no need to copy()
            best_centroids = np.asarray(copy.deepcopy(c))
            # print(method, d, min_d, best_centroids)
    # print(f'centroids after: {best_centroids}')
    return best_centroids


tot_iterate = 50
sub_iterate = 4
tolerance = 1e-4


def kmeans(points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None, true_labels=None):
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

    before_centroids = np.copy(new_centroids)
    new_centroids = align_centroids(new_centroids, true_centroids, method='kmeans')
    # if np.sum(np.square(before_centroids-new_centroids)) > 0:
    #     print(f'kmeans true: ', true_centroids)
    #     print('kmeans before: ', before_centroids)
    #     print('kmeans after: ',new_centroids)
    # L2
    distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    labels = np.argmin(distances, axis=1)

    return new_centroids, labels


def kmed(points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None, true_labels=None):
    new_centroids = np.copy(centroids_input)

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
            new_centroids[j] = np.median(points[labels == j], axis=0)

        if np.sum((new_centroids - pre_centroids) ** 2) / k < tolerance:
            break
    new_centroids = align_centroids(new_centroids, true_centroids, method='Kmed')
    # note that it should be L1, not L2
    # distances2 = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    distances = np.sum(np.abs(points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
    labels = np.argmin(distances, axis=1)

    return new_centroids, labels


def lloydL1(points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None, true_labels=None):
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

        if np.sum((new_centroids - pre_centroids) ** 2) / k < tolerance:
            break

    new_centroids = align_centroids(new_centroids, true_centroids, method='lloydL1')
    # Here should be L2
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
    new_centroids = align_centroids(new_centroids, true_centroids, method='Kmed')
    # note that it should be L1, not L2
    # distances2 = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    distances = np.sum(np.abs(points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
    labels = np.argmin(distances, axis=1)

    return new_centroids, labels
