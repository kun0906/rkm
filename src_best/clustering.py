"""

"""
import copy
import itertools

import numpy as np

from base import *


def get_ith_results(datasets, out_dir='', x_axis='', tuning=0, show=0, affinity='rbf', n_init=1):
    results = {}
    for clustering_method in CLUSTERING_METHODS:
        if clustering_method.startswith('sc_'):
            if tuning:
                # find the best one
                # nns = [5, 10]
                nns = [10, 50, 100, 200]
                qs = [0.1, 0.3, 0.5]
            else:
                nns = [100]
                qs = [0.3]
            best_mp = np.inf
            best_ = (np.inf, np.inf)
            for q in qs:
                n_neighbors = 0
                mps = []
                for data in datasets:
                    try:
                        true_centroids = data['true_centroids']
                        n_centroids = data['n_centroids']
                        points = data['points']
                        true_labels = data['true_labels']
                        true_single_cluster_size = data['true_single_cluster_size']
                        random_state = data['random_state']
                        # find the projected centroids
                        k = n_centroids
                        using_label_in_initialization = True
                        if using_label_in_initialization:
                            projected_points = sc_projection(points, k, affinity=affinity, q=q, n_neighbors=n_neighbors,
                                                             random_state=random_state)
                            projected_true_centroids = np.zeros((n_centroids, k))
                            ls, cs = np.unique(true_labels, return_counts=True)
                            for i, l in enumerate(ls):
                                mask = true_labels == l
                                cluster = projected_points[:len(true_labels), :][mask]
                                projected_true_centroids[i] = np.mean(cluster, axis=0)
                        else:
                            X = np.concatenate([true_centroids, points], axis=0)
                            X_projected = sc_projection(X, k, affinity=affinity, n_neighbors=n_neighbors, q=q,
                                                        random_state=random_state)
                            # X_projected = np.concatenate([true_centroids, points], axis=0)
                            projected_true_centroids = X_projected[:k, :]
                            projected_points = X_projected[k:, :]

                        if show and clustering_method == 'sc_k_medians_l2':
                            plot_projected_data(points, projected_points, cluster_size=100,
                                                clustering_method=clustering_method,
                                                centroids=true_centroids,
                                                projected_centroids=projected_true_centroids,
                                                title=f'{clustering_method}, nn:{n_neighbors}, q:{q}',
                                                out_dir=out_dir, x_axis=x_axis, random_state=random_state)

                        if clustering_method == 'sc_k_medians_l2':
                            centroids_, labels_ = sc_k_medians_l2(projected_points, points, k, projected_true_centroids,
                                                                  max_iterations=300,
                                                                  true_centroids=None, true_labels=true_labels)
                        elif clustering_method == 'sc_k_medians_l1':
                            centroids_, labels_ = sc_k_medians_l1(projected_points, points, k, projected_true_centroids,
                                                                  max_iterations=300,
                                                                  true_centroids=None, true_labels=true_labels)
                        elif clustering_method == 'sc_k_means':
                            centroids_, labels_ = sc_k_means(projected_points, points, k, projected_true_centroids,
                                                             max_iterations=300,
                                                             true_centroids=None, true_labels=true_labels)
                        else:
                            raise NotImplementedError(clustering_method)

                        # TODO: double check if we can align the labels for omniscient initialization.
                        # After sc_project, it's better to align the labels with true_labels.
                        from clustering_random import align_labels
                        labels_ = align_labels(labels_, true_labels)

                        # print(clustering_method, len(labels_), flush=True)
                        mp = sum(labels_[range(n_centroids * true_single_cluster_size)] != true_labels) / len(
                            true_labels)
                        acd = 0  # np.sum((centroids_ - true_centroids) ** 2) / n_centroids
                        mps.append(mp)
                    except Exception as e:
                        print(n_neighbors, clustering_method, e)
                print(clustering_method, mps, x_axis)
                mean_, std_ = np.mean(mps), 1.96 * np.std(mps) / np.sqrt(len(mps))
                if len(mps) != len(datasets):
                    print(clustering_method, len(mps), len(datasets))
                if best_mp > mean_:
                    best_mp = mean_
                    best_ = (mean_, std_)
                    best_params = {'neighbors': n_neighbors, }
            results[clustering_method] = {f'mp_mu': best_[0],
                                          f'mp_std': best_[1],
                                          f'acd_mu': 0,
                                          f'acd_std': 1,
                                          'params': best_params}
        elif clustering_method.startswith('rsc_'):
            if tuning:
                # find the best one
                # nns = [5, 10]
                # thetas = [10]
                # ms = [0.1]
                nns = [10, 50, 100, 200]
                qs = [0.1, 0.3, 0.5]
                thetas = [50]
                ms = [0.5]
            else:
                nns = [100]
                qs = [0.3]
                thetas = [50]
                ms = [0.5]
            # Generate all combinations using itertools.product
            combinations = list(itertools.product(qs, thetas, ms))
            best_mp = np.inf
            best_ = (np.inf, np.inf)
            for q, theta, m in combinations:
                n_neighbors = 0
                mps = []
                for data in datasets:
                    try:
                        true_centroids = data['true_centroids']
                        n_centroids = data['n_centroids']
                        points = data['points']
                        true_labels = data['true_labels']
                        true_single_cluster_size = data['true_single_cluster_size']
                        random_state = data['random_state']
                        # find the projected centroids
                        k = n_centroids
                        using_label_in_initialization = True
                        if using_label_in_initialization:
                            projected_points = rsc_projection(points, k, n_neighbors, theta=theta, m=m,
                                                              affinity=affinity,
                                                              q=q, random_state=random_state)
                            projected_true_centroids = np.zeros((n_centroids, k))
                            ls, cs = np.unique(true_labels, return_counts=True)
                            for i, l in enumerate(ls):
                                mask = true_labels == l
                                cluster = projected_points[:len(true_labels), :][mask]
                                projected_true_centroids[i] = np.mean(cluster, axis=0)
                        else:
                            X = np.concatenate([true_centroids, points], axis=0)
                            X_projected = rsc_projection(X, k, n_neighbors, theta=theta, m=m, affinity=affinity,
                                                         q=q, random_state=random_state)
                            projected_true_centroids = X_projected[:k, :]
                            projected_points = X_projected[k:, :]
                        if show and clustering_method == 'rsc_k_medians_l2':
                            plot_projected_data(points, projected_points, cluster_size=100,
                                                clustering_method=clustering_method,
                                                centroids=true_centroids,
                                                projected_centroids=projected_true_centroids,
                                                title=f'x_axis:{x_axis},  nn:{n_neighbors}, theta:{theta}, m:{m}, q:{q}',
                                                out_dir=out_dir, x_axis=x_axis, random_state=random_state)
                        if clustering_method == 'rsc_k_medians_l2':
                            centroids_, labels_ = sc_k_medians_l2(projected_points, points, k, projected_true_centroids,
                                                                  max_iterations=300,
                                                                  true_centroids=None, true_labels=true_labels)
                        elif clustering_method == 'rsc_k_medians_l1':
                            centroids_, labels_ = sc_k_medians_l1(projected_points, points, k, projected_true_centroids,
                                                                  max_iterations=300,
                                                                  true_centroids=None, true_labels=true_labels)
                        elif clustering_method == 'rsc_k_means':
                            centroids_, labels_ = sc_k_means(projected_points, points, k, projected_true_centroids,
                                                             max_iterations=300,
                                                             true_centroids=None, true_labels=true_labels)
                        elif clustering_method == 'rsc_k_means_orig':  # robust k_means from the original api
                            # rsc = RSC(k=k, nn=n_neighbors, theta=theta, m=m, laplacian=1, normalize=False,
                            #           verbose=False,
                            #           random_state=random_state)
                            # labels_ = rsc.fit_predict(points, init=projected_true_centroids)
                            # # # rsc.fit_predict uses k-means with random initialization on the eigenvectors,
                            # # # so we should align the final labels
                            # # from clustering_random import align_labels
                            # # must align the labels with true_labels, we did this at the bottom
                            # # labels_ = align_labels(labels_, true_labels)

                            from _kmeans import k_means as sklearn_k_means
                            centroids_, labels_, inertia_, *_ = sklearn_k_means(X=projected_points, n_clusters=k,
                                                                                init=projected_true_centroids,
                                                                                n_init=1,
                                                                                random_state=random_state)
                            # X_projected = rsc.H
                            # plot_projected_data(points, X_projected, cluster_size=100, clustering_method=clustering_method,
                            #                     centroids=true_centroids, projected_centroids=projected_true_centroids,
                            #                     random_state=random_state)
                        else:
                            raise NotImplementedError(clustering_method)

                        # TODO: double check if we can align the labels for omniscient initialization.
                        # After sc_project, it's better to align the labels with true_labels.
                        from clustering_random import align_labels
                        labels_ = align_labels(labels_, true_labels)

                        # print(clustering_method, len(labels_), flush=True)
                        mp = sum(labels_[range(n_centroids * true_single_cluster_size)] != true_labels) / len(
                            true_labels)
                        acd = 0  # np.sum((centroids_ - true_centroids) ** 2) / n_centroids
                        mps.append(mp)
                    except Exception as e:
                        print(n_neighbors, theta, m, clustering_method, e)
                print(clustering_method, mps, x_axis)
                mean_, std_ = np.mean(mps), 1.96 * np.std(mps) / np.sqrt(len(mps))
                if len(mps) != len(datasets):
                    print(clustering_method, len(mps), len(datasets))
                if best_mp > mean_:
                    best_mp = mean_
                    best_ = (mean_, std_)
                    best_params = {'neighbors': n_neighbors, 'theta': theta, 'm': m}
            results[clustering_method] = {f'mp_mu': best_[0],
                                          f'mp_std': best_[1],
                                          f'acd_mu': 0,
                                          f'acd_std': 1,
                                          'params': best_params}
        else:
            mps = []
            for data in datasets:
                true_centroids = data['true_centroids']
                n_centroids = data['n_centroids']
                points = data['points']
                true_labels = data['true_labels']
                true_single_cluster_size = data['true_single_cluster_size']
                random_state = data['random_state']
                # init_centroids = data['init_centroids']
                # align the labels with true_centroids

                # if show and clustering_method.startswith('k_'):
                #     # only align the labels, do we need to align the centroids too for plotting.
                #     plot_centroids(points, cluster_size=100, clustering_method=clustering_method,
                #                    init_centroids=true_centroids, title=f'data',
                #                    final_centroids=true_centroids, final_labels=true_labels,
                #                    out_dir=out_dir, x_axis=x_axis, random_state=random_state)

                best_inertia = np.inf
                # rng = np.random.RandomState(seed=random_state * seed)
                # indices = rng.choice(range(len(points)), size=n_centroids, replace=False)
                init_centroids_0 = true_centroids
                if clustering_method == 'k_medians_l2':
                    centroids_0, labels_0, inertia_0 = k_medians_l2(points, centroids_input=init_centroids_0,
                                                                    k=n_centroids, true_centroids=true_centroids)
                elif clustering_method == 'k_medians_l1':
                    centroids_0, labels_0, inertia_0 = k_medians_l1(points, centroids_input=init_centroids_0,
                                                                    k=n_centroids, true_centroids=true_centroids)
                elif clustering_method == 'k_means':
                    centroids_0, labels_0, inertia_0 = k_means(points, centroids_input=init_centroids_0,
                                                               k=n_centroids, true_centroids=true_centroids)
                # elif clustering_method == 'k_means_sdp':
                #     centroids_0, labels_0, inertia_0 = regularised_k_means_SDP(points,
                #                                                                centroids_input=init_centroids_0,
                #                                                                k=n_centroids,
                #                                                                true_centroids=true_centroids)
                elif clustering_method == 'k_means_robust_lp':
                    centroids_0, labels_0, inertia_0 = robust_k_means_LP_SDP(points,
                                                                             centroids_input=None,
                                                                             k=n_centroids,
                                                                             true_centroids=None,
                                                                             true_labels=true_labels,
                                                                             is_sdp=False)
                else:
                    raise NotImplementedError()
                # if show and clustering_method.startswith('k_'):
                #     plot_centroids(points, cluster_size=100, clustering_method=clustering_method,
                #                    init_centroids=init_centroids_0,
                #                    title=f'{clustering_method}, seed: {random_state*seed}, inertia_0:{inertia_0}',
                #                    final_centroids=centroids_0, final_labels=labels_0,
                #                    out_dir=out_dir, x_axis=x_axis, random_state=random_state*seed)
                if best_inertia > inertia_0:
                    best_inertia = inertia_0
                    centroids_ = copy.deepcopy(centroids_0)
                    labels_ = copy.deepcopy(labels_0)
                    init_centroids_ = copy.deepcopy(init_centroids_0)
                    # print(f'{clustering_method}, best_inertia: {best_inertia}')
                # if show: print(clustering_method, seed, random_state, centroids_0, inertia_0, -1, best_inertia)
                # TODO: double check if we can align the labels for omniscient initialization.
                # it's better to align the labels with true_labels.
                from clustering_random import align_labels
                labels_ = align_labels(labels_, true_labels)

                # print(clustering_method, len(labels_), flush=True)
                mp = sum(labels_[range(n_centroids * true_single_cluster_size)] != true_labels) / len(true_labels)
                acd = 0  # np.sum((centroids_ - true_centroids) ** 2) / n_centroids
                mps.append(mp)
            print(clustering_method, mps, x_axis)
            mean_, std_ = np.mean(mps), 1.96 * np.std(mps) / np.sqrt(len(mps))
            best_ = (mean_, std_)
            results[clustering_method] = {f'mp_mu': best_[0],
                                          f'mp_std': best_[1],
                                          f'acd_mu': 0,
                                          f'acd_std': 1,
                                          'params': {}}

    return results


def k_means(points, centroids_input, k, max_iterations=tot_iterate, true_centroids=None, true_labels=None):
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
    inertia = np.sum(np.min(distances, axis=1))
    return new_centroids, labels, inertia


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
    inertia = np.sum(np.min(distances, axis=1))
    return new_centroids, labels, inertia


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

    # It should be L1, not L2
    # distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    distances = np.sum(np.abs(points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
    labels = np.argmin(distances, axis=1)
    # print(f'kmedL1 iterations: {i}')
    inertia = np.sum(np.min(distances, axis=1))
    return new_centroids, labels, inertia


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

    inertia = np.sum(np.min(distances, axis=1))
    return new_centroids, labels, inertia


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
        # print(i, collections.Counter(labels))
        if np.sum((new_centroids - pre_centroids) ** 2) / k < tolerance:
            break

    # it should be L2
    distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    labels = np.argmin(distances, axis=1)
    inertia = np.sum(np.min(distances, axis=1))
    return new_centroids, labels, inertia


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

    inertia = np.sum(np.min(distances, axis=1))
    return new_centroids, labels, inertia


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

    inertia = np.sum(np.min(distances, axis=1))
    return new_centroids, labels, inertia


def regularised_k_means_SDP(X, centroids_input, k, max_iterations=tot_iterate,
                            lambda_=0.5, threshold=0.5):
    from sdp_lp_relaxation import compute_z_y, filter_noise
    Z, y = compute_z_y(X, k, lambda_)
    X_reduced, Z_reduced, C_k_plus_1 = filter_noise(Z, y, X, threshold)

    # Compute X^T Z_reduced and transpose for clustering columns
    X2 = (X_reduced.T @ Z_reduced).T

    # run Kmeans only on the reduced X
    # # k-means clustering on columns of X^T Z_reduced
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=k, random_state=random_state).fit(X2)
    # labels = kmeans.labels_
    new_centroids, labels = k_means(X2, k, centroids_input, max_iterations)

    # reassign all the points to its nearest centroid.
    distances = np.sqrt(np.sum((X[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    labels = np.argmin(distances, axis=1)
    inertia = np.sum(np.min(distances, axis=1))
    return new_centroids, labels, inertia


def robust_k_means_LP_SDP(X, centroids_input, k,
                          true_centroids=None, true_labels=None,
                          is_sdp=False, random_state=42):
    import numpy as np
    from numpy.linalg import eigh
    from scipy.spatial.distance import cdist
    from robust_sdp_lp import choose_theta_gamma, solve_lp_sdp

    Y = X
    theta, gamma = choose_theta_gamma(Y, beta=0.06, alpha=0.2)
    print(f'theta: {theta}, gamma: {gamma}')
    N = Y.shape[0]

    # Step 1: Construct Gaussian kernel matrix
    pairwise_sq_dists = cdist(Y, Y, 'sqeuclidean')
    K = np.exp(-pairwise_sq_dists / (2 * theta ** 2))

    # Step 2: Solve Robust-LP/SDP
    X_hat = solve_lp_sdp(K, N, gamma, is_sdp=is_sdp)

    # Step 3: Eigen-decomposition
    # Compute top-r eigenvectors
    vals, vecs = eigh(X_hat)
    r = k
    print(f'top {r} eigenvalues: {vals[-r:]}')
    U_hat = vecs[:, -r:]  # top r eigenvectors

    # # Step 4: Apply k-means clustering on rows of U_hat
    print('Apply k-means clustering')
    # rng = np.random.RandomState(seed=random_state)
    # indices = rng.choice(range(len(U_hat)), size=r, replace=False)
    # indices = centroids_input

    # Compute the true centroids based on U_hat
    new_centroids = np.zeros((k, k))
    for j in range(k):
        if sum(true_labels == j) == 0:
            # new_centroids[j] use the previous centroid
            continue
        new_centroids[j] = np.mean(U_hat[:len(true_labels)][true_labels == j], axis=0)

    new_centroids, labels, inertia = k_means(U_hat, k=r, centroids_input=new_centroids,
                                             max_iterations=tot_iterate, true_centroids=new_centroids)
    return new_centroids, labels, inertia
