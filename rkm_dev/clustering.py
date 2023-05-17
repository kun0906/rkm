import collections
import os
import matplotlib.pyplot as plt

import numpy as np

tot_iterate = 50
tolerance = 1e-4
def kmeans(points, k, centroids_input, max_iterations=tot_iterate):
    new_centroids = np.copy(centroids_input)

    # diff1 = []  # the difference between pre_centroids and current_centroids
    history_centroids = []
    history_labels = []
    for i in range(max_iterations):
        # Assign each point to the closest centroid
        distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
        labels = np.argmin(distances, axis=1)

        pre_centroids = np.copy(new_centroids)
        history_centroids.append(pre_centroids)
        history_labels.append(labels)
        # Update the centroids to be the mean of the points in each cluster
        for j in range(k):
            if sum(labels == j) == 0:
                # new_centroids[j] use the previous centroid
                continue
            new_centroids[j] = np.mean(points[labels == j], axis=0)

        _diff = np.sum((new_centroids - pre_centroids) ** 2) / k
        if _diff < tolerance:
            break

    # print(f'kmeans iterations: {i}')
    res = {'centroids': new_centroids, 'labels':labels, 'history_centroids':history_centroids, 'history_labels': history_labels}
    return res

def kmed(points, k, centroids_input, max_iterations=tot_iterate):
    new_centroids = np.copy(centroids_input)
    history_centroids = []
    history_labels = []
    for i in range(max_iterations):
        # Assign each point to the closest centroid
        distances = np.sum(np.abs(points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
        labels = np.argmin(distances, axis=1)

        pre_centroids = np.copy(new_centroids)
        history_centroids.append(pre_centroids)
        history_labels.append(labels)
        # Update the centroids to be the median of the points in each cluster
        for j in range(k):
            if sum(labels == j) == 0:
                # new_centroids[j] use the previous centroid
                continue
            new_centroids[j] = np.median(points[labels == j], axis=0)

        if np.sum((new_centroids - pre_centroids) ** 2) / k < tolerance:
            break
    # print(f'kmedL1 iterations: {i}')
    res = {'centroids': new_centroids, 'labels':labels, 'history_centroids':history_centroids, 'history_labels': history_labels}
    return res


def lloydL1(points, k, centroids_input,max_iterations=tot_iterate):

    new_centroids = np.copy(centroids_input)

    history_centroids = []
    history_labels = []
    for i in range(max_iterations):
        # Assign each point to the closest centroid
        distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
        labels = np.argmin(distances, axis=1)

        pre_centroids = np.copy(new_centroids)
        history_centroids.append(pre_centroids)
        history_labels.append(labels)
        # Update the centroids to be the median of the points in each cluster
        for j in range(k):
            if sum(labels == j) == 0:
                # new_centroids[j] use the previous centroid
                continue
            new_centroids[j] = np.median(points[labels == j], axis=0)

        if np.sum((new_centroids - pre_centroids) ** 2) / k < tolerance:
            break

    res = {'centroids': new_centroids, 'labels':labels, 'history_centroids':history_centroids, 'history_labels': history_labels}
    return res




def plot_diff_i(lloydL1_his, kmed_his, kmeans_his, ith_repeat, title='', out_dir=''):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    colors = ['g', 'purple', 'b']
    fmts = ['*-', '^-', 'o-']
    labels = ['lloydL1', 'kmedian', 'kmean']
    for i, his in enumerate([lloydL1_his, kmed_his, kmeans_his]):
        ds = []
        m = len(his['history_centroids'])
        for j in range(m):
            d = np.sum((his['history_centroids'][j]-his['history_centroids'][0])**2)
            ds.append(d)
            tmp = dict(sorted(collections.Counter(list(his['history_labels'][j])).items()))
            # print(i, j, tmp)
        plt.plot(range(len(ds)), ds, fmts[i], color=colors[i], label=f'{labels[i]}:{tmp}')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Centroid differnce')
    plt.title(title)
    f = f'{out_dir}/{title}.png'
    plt.savefig(f, dpi=100)
    print(f)
    plt.show()


