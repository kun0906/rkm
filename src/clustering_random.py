import copy
import itertools

from base import *


# Filter or ignore the RuntimeWarning
# warnings.filterwarnings("error", category=RuntimeWarning)

def get_ith_results_random(points, init_centroids, true_centroids, true_labels, true_single_cluster_size, n_centroids,
                           n_neighbors, out_dir='.', x_axis='', random_state=42):
    results = {}
    for clustering_method in CLUSTERING_METHODS:

        if clustering_method.startswith('sc_'):
            # find the projected centroids
            k = n_centroids
            projected_points = sc_projection(points, k, affinity='knn', n_neighbors=n_neighbors, random_state=random_state)
            # random select initial centroids
            rng = np.random.RandomState(seed=random_state)
            indices = rng.choice(range(len(points)), size=k, replace=False)
            projected_init_centroids = projected_points[indices, :]
            # if clustering_method == 'sc_k_medians_l2':
            #     plot_projected_data(points, projected_points, cluster_size=100, clustering_method=clustering_method,
            #                         centroids=true_centroids, projected_centroids=projected_init_centroids,
            #                         out_dir = out_dir, x_axis = x_axis, random_state=random_state)
            # align the labels with true_labels
            if clustering_method == 'sc_k_medians_l2':
                centroids_, labels_ = sc_k_medians_l2(projected_points, points, k, projected_init_centroids,
                                                      max_iterations=50,
                                                      true_centroids=None, true_labels=true_labels)
            elif clustering_method == 'sc_k_medians_l1':
                centroids_, labels_ = sc_k_medians_l1(projected_points, points, k, projected_init_centroids,
                                                      max_iterations=50,
                                                      true_centroids=None, true_labels=true_labels)
            elif clustering_method == 'sc_k_means':
                centroids_, labels_ = sc_k_means(projected_points, points, k, projected_init_centroids,
                                                 max_iterations=50,
                                                 true_centroids=None, true_labels=true_labels)
            else:
                raise NotImplementedError()
        elif clustering_method.startswith('rsc_'):
            # find the projected centroids
            k = n_centroids
            projected_points = rsc_projection(points, k, n_neighbors, random_state=random_state)
            # random select initial centroids
            rng = np.random.RandomState(seed=random_state)
            indices = rng.choice(range(len(points)), size=k, replace=False)
            projected_init_centroids = projected_points[indices, :]
            # if clustering_method == 'rsc_k_medians_l2':
            #     plot_projected_data(points, projected_points, cluster_size=100, clustering_method=clustering_method,
            #                         centroids=true_centroids, projected_centroids=projected_init_centroids,
            #                         out_dir = out_dir, x_axis = x_axis, random_state=random_state)
            # align the labels with true_labels
            if clustering_method == 'rsc_k_medians_l2':
                centroids_, labels_ = sc_k_medians_l2(projected_points, points, k, projected_init_centroids,
                                                      max_iterations=50,
                                                      true_centroids=None, true_labels=true_labels)
            elif clustering_method == 'rsc_k_medians_l1':
                centroids_, labels_ = sc_k_medians_l1(projected_points, points, k, projected_init_centroids,
                                                      max_iterations=50,
                                                      true_centroids=None, true_labels=true_labels)
            elif clustering_method == 'rsc_k_means':
                centroids_, labels_ = sc_k_means(projected_points, points, k, projected_init_centroids,
                                                 max_iterations=50,
                                                 true_centroids=None, true_labels=true_labels)
            elif clustering_method == 'rsc_k_means_orig':  # robust k_means from the original api
                rsc = RSC(k=k, nn=n_neighbors, theta=50, m=0.5, laplacian=1, normalize=True, verbose=False,
                          random_state=random_state)
                labels_ = rsc.fit_predict(points, init=projected_init_centroids)
                # labels_ = align_labels(labels_, true_labels)    # must align the labels with true_labels
                centroids_ = np.zeros((k, points.shape[1]))
            else:
                raise NotImplementedError()
        else:
            # align the labels with true_centroids
            if clustering_method == 'k_medians_l2':
                centroids_, labels_ = k_medians_l2(points, centroids_input=init_centroids,
                                                   k=n_centroids, true_centroids=true_centroids)
            elif clustering_method == 'k_medians_l1':
                centroids_, labels_ = k_medians_l1(points, centroids_input=init_centroids,
                                                   k=n_centroids, true_centroids=true_centroids)
            elif clustering_method == 'k_means':
                centroids_, labels_ = k_means(points, centroids_input=init_centroids,
                                              k=n_centroids, true_centroids=true_centroids)
            else:
                raise NotImplementedError()

        # TODO: double check if we can align the labels for omniscient initialization.
        # After sc_project, it's better to align the labels with true_labels.
        labels_ = align_labels(labels_, true_labels)

        # print(clustering_method, len(labels_), flush=True)
        mp = sum(labels_[range(n_centroids * true_single_cluster_size)] != true_labels) / len(true_labels)
        acd = 0  # np.sum((centroids_ - true_centroids) ** 2) / n_centroids
        results[clustering_method] = {'centroids': centroids_, 'labels': labels_,
                                      'mp': mp, 'acd': acd, }

    return results


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
            min_d = np.copy(d)  # here is just a float, so there is no need to copy()
            best_centroids = np.asarray(copy.deepcopy(c))
            # print(method, d, min_d, best_centroids)
    # print(f'centroids after: {best_centroids}')
    return best_centroids


def align_labels(labels, true_labels, method='name'):
    # print(f'labels before: {labels}')
    # print(f"{len(labels)} labels include {len(list(itertools.permutations(labels)))} permutations")
    min_mp = np.inf
    for _labs in list(itertools.permutations(np.unique(true_labels))):
        new_labels = np.zeros_like(labels)
        for i, l in enumerate(_labs):
            new_labels[labels == l] = i  # align the labels
        # compute the misclutering labeling
        mp = sum(new_labels[:len(true_labels)] != true_labels) / len(true_labels)
        if mp < min_mp:
            # print(method, mp, min_mp)
            min_mp = mp
            best_labels = np.asarray(copy.deepcopy(new_labels))
            # print(method, mp, min_mp, best_labels)
    # print(f'mp={min_mp}, labels after: {best_labels}')
    return best_labels


tot_iterate = 50
sub_iterate = 4
tolerance = 1e-4


def k_means(points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None, true_labels=None):
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
    new_centroids = align_centroids(new_centroids, true_centroids, method='k_means')
    # if np.sum(np.square(before_centroids-new_centroids)) > 0:
    #     print(f'k_means true: ', true_centroids)
    #     print('k_means before: ', before_centroids)
    #     print('k_means after: ',new_centroids)
    # L2
    distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    labels = np.argmin(distances, axis=1)

    return new_centroids, labels


def sc_k_means(projected_points, points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None,
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
    labels = align_labels(labels, true_labels)

    return new_centroids, labels


def k_medians_l1(points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None, true_labels=None):
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
    new_centroids = align_centroids(new_centroids, true_centroids, method='k_medians_l1')
    # note that it should be L1, not L2
    # distances2 = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    distances = np.sum(np.abs(points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
    labels = np.argmin(distances, axis=1)

    return new_centroids, labels


def sc_k_medians_l1(projected_points, points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None,
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
    labels = align_labels(labels, true_labels)

    # # according to labels, find the final centroids on the original centroids.
    new_centroids = np.zeros((k, points.shape[1]))

    return new_centroids, labels


def k_medians_l2(points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None, true_labels=None):
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

    new_centroids = align_centroids(new_centroids, true_centroids, method='k_medians_l2')
    # Here should be L2
    distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    labels = np.argmin(distances, axis=1)

    return new_centroids, labels


def sc_k_medians_l2(projected_points, points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None,
                    true_labels=None):
    new_centroids = np.copy(centroids_input)

    for i in range(max_iterations):
        # Assign each point to the closest centroid
        distances = np.sqrt(np.sum((projected_points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
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
    labels = align_labels(labels, true_labels)

    # according to labels, find the final centroids on the original centroids.
    new_centroids = np.zeros((k, points.shape[1]))

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
    new_centroids = align_centroids(new_centroids, true_centroids, method='k_medians_l1')
    # note that it should be L1, not L2
    # distances2 = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    distances = np.sum(np.abs(points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
    labels = np.argmin(distances, axis=1)

    return new_centroids, labels


def sc_random(points, k, max_iterations=tot_iterate, clustering_method='k_means', random_state=42,
              true_centroids=None,
              true_labels=None,
              n_neighbors=None):
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
    if clustering_method == 'k_means':
        centroids, labels = sc_k_means(projected_points, points, k, projected_init_centroids,
                                       max_iterations=max_iterations,
                                       true_centroids=true_centroids, true_labels=true_labels)
    elif clustering_method == 'k_medians_l1':
        centroids, labels = sc_k_medians_l1(projected_points, points, k, projected_init_centroids,
                                            max_iterations=max_iterations,
                                            true_centroids=true_centroids, true_labels=true_labels)
    elif clustering_method == 'k_medians_l2':
        centroids, labels = sc_k_medians_l2(projected_points, points, k, projected_init_centroids,
                                            max_iterations=max_iterations,
                                            true_centroids=true_centroids, true_labels=true_labels)
    else:
        raise NotImplemented(f'{clustering_method}')

    return centroids, labels


def robust_sc_random(points, k, max_iterations=tot_iterate, clustering_method='k_means', random_state=42,
                     true_centroids=None,
                     true_labels=None,
                     n_neighbors=15):
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

    projected_points = rsc_projection(points, k, n_neighbors, random_state=random_state)

    # X = np.concatenate([true_centroids, points], axis=0)
    # X_projected = sc_projection(X, k, random_state=random_state)
    # projected_true_centroids = X_projected[:k, :]

    # random select initial centroids
    rng = np.random.RandomState(seed=random_state)
    indices = rng.choice(range(len(points)), size=k, replace=False)
    projected_init_centroids = projected_points[indices, :]
    if clustering_method == 'k_means':
        centroids, labels = sc_k_means(projected_points, points, k, projected_init_centroids,
                                       max_iterations=max_iterations,
                                       true_centroids=true_centroids, true_labels=true_labels)
    elif clustering_method == 'k_medians_l1':
        centroids, labels = sc_k_medians_l1(projected_points, points, k, projected_init_centroids,
                                            max_iterations=max_iterations,
                                            true_centroids=true_centroids, true_labels=true_labels)
    elif clustering_method == 'k_medians_l2':
        centroids, labels = sc_k_medians_l2(projected_points, points, k, projected_init_centroids,
                                            max_iterations=max_iterations,
                                            true_centroids=true_centroids, true_labels=true_labels)
    else:
        raise NotImplemented(f'{clustering_method}')

    return centroids, labels
