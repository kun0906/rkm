"""

"""

from base import *


def get_ith_results(points, true_centroids, true_labels, true_single_cluster_size, n_centroids,
                    n_neighbors=15, theta= 50, m=0.5, out_dir='', x_axis='', random_state=42):
    results = {}
    for clustering_method in CLUSTERING_METHODS:
        if clustering_method.startswith('sc_'):
            # find the projected centroids
            k = n_centroids
            X = np.concatenate([true_centroids, points], axis=0)
            X_projected = sc_projection(X, k, affinity='knn', n_neighbors=n_neighbors, random_state=random_state)
            projected_true_centroids = X_projected[:k, :]
            projected_points = X_projected[k:, :]
            # if clustering_method == 'sc_k_medians_l2':
            #     plot_projected_data(points, X_projected, cluster_size=100, clustering_method=clustering_method,
            #                     centroids=true_centroids, projected_centroids=projected_true_centroids,
            #                     n_clusters=k, out_dir = out_dir, x_axis=x_axis, random_state=random_state)
            if clustering_method == 'sc_k_medians_l2':
                centroids_, labels_ = sc_k_medians_l2(projected_points, points, k, projected_true_centroids,
                                                      max_iterations=50,
                                                      true_centroids=None, true_labels=true_labels)
            elif clustering_method == 'sc_k_medians_l1':
                centroids_, labels_ = sc_k_medians_l1(projected_points, points, k, projected_true_centroids,
                                                      max_iterations=50,
                                                      true_centroids=None, true_labels=true_labels)
            elif clustering_method == 'sc_k_means':
                centroids_, labels_ = sc_k_means(projected_points, points, k, projected_true_centroids,
                                                 max_iterations=50,
                                                 true_centroids=None, true_labels=true_labels)
            else:
                raise NotImplementedError(clustering_method)
        elif clustering_method.startswith('rsc_'):
            # find the projected centroids
            k = n_centroids
            X = np.concatenate([true_centroids, points], axis=0)
            X_projected = rsc_projection(X, k, n_neighbors, theta=theta, m=m,random_state=random_state)
            projected_true_centroids = X_projected[:k, :]
            projected_points = X_projected[k:, :]
            # if clustering_method == 'rsc_k_medians_l2':
            #     plot_projected_data(points, X_projected, cluster_size=100, clustering_method=clustering_method,
            #                     centroids=true_centroids, projected_centroids=projected_true_centroids,
            #                     n_clusters=k, out_dir=out_dir, x_axis=x_axis, random_state=random_state)
            if clustering_method == 'rsc_k_medians_l2':
                centroids_, labels_ = sc_k_medians_l2(projected_points, points, k, projected_true_centroids,
                                                      max_iterations=50,
                                                      true_centroids=None, true_labels=true_labels)
            elif clustering_method == 'rsc_k_medians_l1':
                centroids_, labels_ = sc_k_medians_l1(projected_points, points, k, projected_true_centroids,
                                                      max_iterations=50,
                                                      true_centroids=None, true_labels=true_labels)
            elif clustering_method == 'rsc_k_means':
                centroids_, labels_ = sc_k_means(projected_points, points, k, projected_true_centroids,
                                                 max_iterations=50,
                                                 true_centroids=None, true_labels=true_labels)
            elif clustering_method == 'rsc_k_means_orig':  # robust k_means from the original api
                rsc = RSC(k=k, nn=n_neighbors, theta=theta, m=m, laplacian=1, normalize=True, verbose=False,
                          random_state=random_state)
                labels_ = rsc.fit_predict(points, init=projected_true_centroids)
                # # rsc.fit_predict uses k-means with random initialization on the eigenvectors,
                # # so we should align the final labels
                # from clustering_random import align_labels
                # must align the labels with true_labels, we did this at the bottom
                # labels_ = align_labels(labels_, true_labels)
                centroids_ = np.zeros((k, points.shape[1]))
                # X_projected = rsc.H
                # plot_projected_data(points, X_projected, cluster_size=100, clustering_method=clustering_method,
                #                     centroids=true_centroids, projected_centroids=projected_true_centroids,
                #                     random_state=random_state)
            else:
                raise NotImplementedError(clustering_method)

        else:
            if clustering_method == 'k_medians_l2':
                centroids_, labels_ = k_medians_l2(points, centroids_input=true_centroids,
                                                   k=n_centroids)
            elif clustering_method == 'k_medians_l1':
                centroids_, labels_ = k_medians_l1(points, centroids_input=true_centroids,
                                                   k=n_centroids)
            elif clustering_method == 'k_means':
                centroids_, labels_ = k_means(points, centroids_input=true_centroids,
                                              k=n_centroids)
            else:
                raise NotImplementedError(clustering_method)

        # TODO: double check if we can align the labels for omniscient initialization.
        # After sc_project, it's better to align the labels with true_labels.
        from clustering_random import align_labels
        labels_ = align_labels(labels_, true_labels)

        # print(clustering_method, len(labels_), flush=True)
        mp = sum(labels_[range(n_centroids * true_single_cluster_size)] != true_labels) / len(true_labels)
        acd = 0  # np.sum((centroids_ - true_centroids) ** 2) / n_centroids
        results[clustering_method] = {'centroids': centroids_, 'labels': labels_,
                                      'mp': mp, 'acd': acd, }

    return results


def k_means(points, k, centroids_input, max_iterations=tot_iterate):
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
    # print(f'k_means iterations: {i}')
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

    # according to labels, find the final centroids on the original centroids.
    new_centroids = np.zeros((k, points.shape[1]))
    return new_centroids, labels


def k_medians_l1(points, k, centroids_input, max_iterations=tot_iterate):
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

    # according to labels, find the final centroids on the original centroids.
    new_centroids = np.zeros((k, points.shape[1]))

    return new_centroids, labels


def k_medians_l2(points, k, centroids_input, max_iterations=tot_iterate):
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


def sc_k_medians_l2(projected_points, points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None,
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
