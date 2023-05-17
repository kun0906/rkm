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


def align_centroids(centroids, true_centroids):
    # print(f'centroids before: {centroids}')
    # print(f"{len(centroids)} centroids include {len(list(itertools.permutations(centroids)))} permutations")
    c1 = copy.deepcopy(true_centroids)
    # check which point is close to which true centroids.
    min_d = np.inf
    for c in list(itertools.permutations(centroids)):
        d = np.sum(np.sum(np.square(c - c1), axis=1), axis=0)
        if d < min_d:
            min_d = d # here is just a float, so there is no need to copy()
            best_centroids = np.asarray(copy.deepcopy(c))
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
        # tmp = sum([len(set(row)) if len(set(row)) != k else 0 for row in distances])/len(distances)
        # if tmp > 0:
        #     print(i,  ' same distances: ', tmp)
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

    # print(f'kmeans iterations: {i}')
    # tmp = collections.Counter(labels)
    # tmp = dict(sorted(tmp.items()))
    # print(f'Kmeans, before labels: {tmp}')
    new_centroids = align_centroids(new_centroids, true_centroids)

    distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    labels = np.argmin(distances, axis=1)
    # tmp = sum([len(set(row)) if len(set(row)) != k else 0 for row in distances]) / len(distances)
    # if tmp > 0:
    #     print(i, ' same distances: ', tmp)
    # tmp2 = collections.Counter(labels)
    # tmp2 = dict(sorted(tmp2.items()))
    # # print(f'Kmeans, after labels: {tmp2}')
    #
    # for k1, k2 in zip(tmp.keys(), tmp2.keys()):
    #     if k1 !=k2 or tmp[k1] !=tmp2[k2]:
    #         print('kmeans: ', k1, k2, tmp[k1], tmp2[k2])

    return new_centroids, labels


def kmed(points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None, true_labels=None):
    new_centroids = np.copy(centroids_input)

    for i in range(max_iterations ):
        # Assign each point to the closest centroid
        distances = np.sum(np.abs(points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
        tmp = sum([len(set(row)) if len(set(row)) != k else 0 for row in distances]) / len(distances)
        if tmp > 0:
            print(i, ' same distances: ', tmp)
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
    # print(f'kmedL1 iterations: {i}')
    # tmp = collections.Counter(labels)
    # tmp = dict(sorted(tmp.items()))
    # print(f'Kmed, before labels: {tmp}', labels)
    # print()
    new_centroids2 = align_centroids(new_centroids, true_centroids)
    # note that it should be L1, not L2
    # distances2 = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    distances2 = np.sum(np.abs(points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
    labels2 = np.argmin(distances2, axis=1)
    # tmp = sum([len(set(row)) if len(set(row)) != k else 0 for row in distances]) / len(distances)
    # if tmp > 0:
    #     print(i, ' same distances: ', tmp)
    # tmp2 = collections.Counter(labels2)
    # tmp2 = dict(sorted(tmp2.items()))
    #
    # for k1, k2 in zip(tmp.keys(), tmp2.keys()):
    #     if k1 != k2 or tmp[k1] !=tmp2[k2]:
    #         print('Kmed: ', k1, k2, tmp[k1], tmp2[k2], dict(tmp), dict(tmp2))
    #         print(f'Kmed, before labels: {tmp}', labels)
    #         print(f'Kmed, after labels: {tmp2}', labels2)
    #         # print(set(new_centroids2) == set(true_centroids))
    #         # plot_cluster(points, true_labels, labels, labels2)

    return new_centroids2, labels2


def lloydL1(points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None, true_labels=None):
    new_centroids = np.copy(centroids_input)

    for i in range(max_iterations):
        # Assign each point to the closest centroid
        distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
        tmp = sum([len(set(row)) if len(set(row)) != k else 0 for row in distances]) / len(distances)
        if tmp > 0:
            print(i, ' same distances: ', tmp)
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

    # tmp = collections.Counter(labels)
    # tmp = dict(sorted(tmp.items()))
    # print(f'lloydL1, before labels: {tmp}')
    new_centroids = align_centroids(new_centroids, true_centroids)
    # for i in range(sub_iterate):
    #     # Assign each point to the closest centroid
    #     distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    #     labels = np.argmin(distances, axis=1)
    #
    #     pre_centroids = np.copy(new_centroids)
    #     # Update the centroids to be the mean of the points in each cluster
    #     for j in range(k):
    #         if sum(labels == j) == 0:
    #             # new_centroids[j] use the previous centroid
    #             continue
    #         new_centroids[j] = np.median(points[labels == j], axis=0)
    # Here should be L2
    distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    # tmp = sum([len(set(row)) if len(set(row)) != k else 0 for row in distances]) / len(distances)
    # if tmp > 0:
    #     print(i, ' same distances: ', tmp)
    labels = np.argmin(distances, axis=1)
    # tmp2 = collections.Counter(labels)
    # tmp2 = dict(sorted(tmp2.items()))
    # # print(f'Kmed, after labels: {tmp2}')
    # for k1, k2 in zip(tmp.keys(), tmp2.keys()):
    #     if k1 != k2 or tmp[k1] != tmp2[k2]:
    #         print('lloydL1: ', k1, k2, tmp[k1], tmp2[k2])
    return new_centroids, labels
