import copy
import itertools
import traceback

from sklearn.utils.sparsefuncs import mean_variance_axis

from base import *


# Filter or ignore the RuntimeWarning
# warnings.filterwarnings("error", category=RuntimeWarning)

def get_ith_results_random(datasets, out_dir='', x_axis='', affinity='rbf', tuning=0, show=0, n_init=1,
                           normalize_project=0):
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
                n_neighbors = 15
                mps = []
                for data in datasets:
                    try:
                        true_centroids = data['true_centroids']
                        n_centroids = data['n_centroids']
                        points = data['points']
                        true_labels = data['true_labels']
                        true_single_cluster_size = data['true_single_cluster_size']
                        random_state = data['random_state']
                        # init_centroids = data['init_centroids']
                        # find the projected centroids
                        k = n_centroids
                        projected_points = sc_projection(points, k, affinity=affinity, n_neighbors=n_neighbors,
                                                         normalize=normalize_project, q=q,
                                                         random_state=random_state)
                        # projected_points = points
                        best_inertia = np.inf
                        for seed in range(1, n_init + 1):
                            # random select initial centroids
                            rng = np.random.RandomState(seed=random_state * seed)
                            indices = rng.choice(range(len(points)), size=k, replace=False)
                            projected_init_centroids_0 = projected_points[indices, :]
                            # projected_init_centroids = init_centroids
                            if show and clustering_method == 'sc_k_medians_l2' and seed == 1:
                                plot_projected_data(points, projected_points, cluster_size=100,
                                                    clustering_method=clustering_method,
                                                    centroids=true_centroids,
                                                    projected_centroids=projected_init_centroids_0,
                                                    title=f'{clustering_method}, seed: {random_state * seed}, inertia_0:{inertia_0},nn:{n_neighbors}, q:{q}',
                                                    out_dir=out_dir, x_axis=x_axis, random_state=random_state)
                            # align the labels with true_labels
                            if clustering_method == 'sc_k_medians_l2':
                                centroids_0, labels_0, inertia_0 = sc_k_medians_l2(projected_points, points, k,
                                                                                   projected_init_centroids_0,
                                                                                   max_iterations=300,
                                                                                   true_centroids=None,
                                                                                   true_labels=true_labels)
                            elif clustering_method == 'sc_k_medians_l1':
                                centroids_0, labels_0, inertia_0 = sc_k_medians_l1(projected_points, points, k,
                                                                                   projected_init_centroids_0,
                                                                                   max_iterations=300,
                                                                                   true_centroids=None,
                                                                                   true_labels=true_labels)
                            elif clustering_method == 'sc_k_means':
                                centroids_0, labels_0, inertia_0 = sc_k_means(projected_points, points, k,
                                                                              projected_init_centroids_0,
                                                                              max_iterations=300,
                                                                              true_centroids=None,
                                                                              true_labels=true_labels)
                            else:
                                raise NotImplementedError()
                            if best_inertia > inertia_0:
                                best_inertia = inertia_0
                                centroids_ = copy.deepcopy(centroids_0)
                                labels_ = copy.deepcopy(labels_0)
                                projected_init_centroids_ = copy.deepcopy(projected_init_centroids_0)
                                # print(f'{clustering_method}, best_inertia: {best_inertia}')
                            if show: print(clustering_method, ' n_init:', seed, random_state, inertia_0, -1,
                                           best_inertia, n_neighbors)
                        # TODO: double check if we can align the labels for omniscient initialization.
                        # After sc_project, it's better to align the labels with true_labels.
                        from clustering_random import align_labels
                        labels_ = align_labels(labels_, true_labels)
                        if show and clustering_method == 'sc_k_medians_l2' and seed == 1:
                            y_sc = labels_
                            X_sc = points
                            plt.scatter(X_sc[:, 0], X_sc[:, 1], c=y_sc, cmap='Accent', linewidths=0)
                            plt.title(f'labels obtained by {clustering_method}, seed: {seed}')
                            plt.show()

                        # print(clustering_method, len(labels_), flush=True)
                        mp = sum(labels_[range(n_centroids * true_single_cluster_size)] != true_labels) / len(
                            true_labels)
                        acd = 0  # np.sum((centroids_ - true_centroids) ** 2) / n_centroids
                        mps.append(mp)
                    except Exception as e:
                        print(n_neighbors, clustering_method, e)
                print(clustering_method, mps, x_axis)
                # if show and clustering_method.startswith('sc_'):
                #     plot_centroids(projected_points, cluster_size=100, clustering_method=clustering_method,
                #                    init_centroids=projected_init_centroids_,
                #                    final_centroids=centroids_, final_labels=labels_,
                #                    out_dir=out_dir, x_axis=x_axis, random_state=random_state)
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
                n_neighbors = 15
                mps = []
                for data in datasets:
                    try:
                        true_centroids = data['true_centroids']
                        n_centroids = data['n_centroids']
                        points = data['points']
                        true_labels = data['true_labels']
                        true_single_cluster_size = data['true_single_cluster_size']
                        random_state = data['random_state']
                        init_centroids = data['init_centroids']
                        # find the projected centroids
                        k = n_centroids
                        projected_points = rsc_projection(points, k, n_neighbors, theta=theta, m=m, q=q,
                                                          normalize=normalize_project, affinity=affinity,
                                                          random_state=random_state)
                        # projected_points  = points
                        best_inertia = np.inf
                        for seed in range(1, n_init + 1):
                            # random select initial centroids
                            rng = np.random.RandomState(seed=random_state * seed)
                            indices = rng.choice(range(len(points)), size=k, replace=False)
                            projected_init_centroids_0 = projected_points[indices, :]
                            # projected_init_centroids = init_centroids
                            if show and clustering_method == 'rsc_k_medians_l2' and seed == 1:
                                plot_projected_data(points, projected_points, cluster_size=100,
                                                    clustering_method=clustering_method,
                                                    centroids=true_centroids,
                                                    projected_centroids=projected_init_centroids_0,
                                                    title=f'x_axis:{x_axis}, n_init:{seed}, nn:{n_neighbors}, theta:{theta}, m:{m}, q:{q}',
                                                    out_dir=out_dir, x_axis=x_axis, random_state=random_state)
                            # align the labels with true_labels
                            if clustering_method == 'rsc_k_medians_l2':
                                centroids_0, labels_0, inertia_0 = sc_k_medians_l2(projected_points, points, k,
                                                                                   projected_init_centroids_0,
                                                                                   max_iterations=300,
                                                                                   true_centroids=None,
                                                                                   true_labels=true_labels)
                            elif clustering_method == 'rsc_k_medians_l1':
                                centroids_0, labels_0, inertia_0 = sc_k_medians_l1(projected_points, points, k,
                                                                                   projected_init_centroids_0,
                                                                                   max_iterations=300,
                                                                                   true_centroids=None,
                                                                                   true_labels=true_labels)
                            elif clustering_method == 'rsc_k_means':
                                centroids_0, labels_0, inertia_0 = sc_k_means(projected_points, points, k,
                                                                              projected_init_centroids_0,
                                                                              max_iterations=300,
                                                                              true_centroids=None,
                                                                              true_labels=true_labels)
                            elif clustering_method == 'rsc_k_means_orig':  # robust k_means from the original api
                                # rsc = RSC(k=k, nn=n_neighbors, theta=theta, m=m, laplacian=1, normalize=False, verbose=False,
                                #           random_state=random_state)
                                # # _kmeans() will recompute self._tol depending on the input data,
                                # # which will be smaller than self.tol.
                                # labels_ = rsc.fit_predict(points, init=projected_init_centroids)
                                # # labels_ = align_labels(labels_, true_labels)    # must align the labels with true_labels
                                # centroids_ = rsc.centroids
                                from _kmeans import k_means as sklearn_k_means
                                centroids_0, labels_0, inertia_0, *_ = sklearn_k_means(X=projected_points, n_clusters=k,
                                                                                       init=projected_init_centroids_0,
                                                                                       n_init=n_init,
                                                                                       random_state=random_state)
                            else:
                                raise NotImplementedError()
                            # if show and clustering_method.startswith('rsc_'):
                            #     plot_centroids(projected_points, cluster_size=100, clustering_method=clustering_method,
                            #                    init_centroids=projected_init_centroids_, title = f'n_init_seed: {seed*random_state}, {inertia_0}',
                            #                    final_centroids=centroids_, final_labels=labels_,
                            #                    out_dir=out_dir, x_axis=x_axis, random_state=random_state)
                            if best_inertia > inertia_0:
                                best_inertia = inertia_0
                                centroids_ = copy.deepcopy(centroids_0)
                                labels_ = copy.deepcopy(labels_0)
                                projected_init_centroids_ = copy.deepcopy(projected_init_centroids_0)
                                # print(f'{clustering_method}, best_inertia: {best_inertia}')
                            if show: print(clustering_method, ' n_init:', seed, random_state, inertia_0, -1,
                                           best_inertia, n_neighbors, theta, m)
                        # TODO: double check if we can align the labels for omniscient initialization.
                        # After sc_project, it's better to align the labels with true_labels.
                        from clustering_random import align_labels
                        labels_ = align_labels(labels_, true_labels)

                        if show and clustering_method == 'rsc_k_medians_l2' and seed == 1:
                            y_rsc = labels_
                            X_rsc = points
                            plt.scatter(X_rsc[:, 0], X_rsc[:, 1], c=y_rsc, cmap='Accent', linewidths=0)
                            plt.title(f'labels obtained by {clustering_method}, seed: {seed}')
                            plt.show()

                        # print(clustering_method, len(labels_), flush=True)
                        mp = sum(labels_[range(n_centroids * true_single_cluster_size)] != true_labels) / len(
                            true_labels)
                        acd = 0  # np.sum((centroids_ - true_centroids) ** 2) / n_centroids
                        mps.append(mp)
                    except Exception as e:
                        print(n_neighbors, theta, m, clustering_method, e)
                        # traceback.print_exc()

                    # if show and clustering_method.startswith('rsc_'):
                    #     plot_centroids(projected_points, cluster_size=100, clustering_method=clustering_method,
                    #                    init_centroids=projected_init_centroids_,
                    #                    final_centroids=centroids_, final_labels=labels_, title='final',
                    #                    out_dir=out_dir, x_axis=x_axis, random_state=random_state)
                print(clustering_method, mps, x_axis)
                mean_, std_ = np.mean(mps), 1.96 * np.std(mps) / np.sqrt(len(mps))
                if len(mps) != len(datasets):
                    print(clustering_method, len(mps), len(datasets))
                if best_mp > mean_:
                    best_mp = mean_
                    best_ = (mean_, std_)
                    best_params = {'neighbors': n_neighbors, 'theta': theta, 'm': m}

            # print(clustering_method, mps, best_)
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
                init_centroids = data['init_centroids']
                indices = data['init_centroids_indices']
                # align the labels with true_centroids

                # if show and clustering_method.startswith('k_'):
                #     # only align the labels, do we need to align the centroids too for plotting.
                #     plot_centroids(points, cluster_size=100, clustering_method=clustering_method,
                #                    init_centroids=true_centroids, title=f'data',
                #                    final_centroids=true_centroids, final_labels=true_labels,
                #                    out_dir=out_dir, x_axis=x_axis, random_state=random_state)

                best_inertia = np.inf
                init_centroids_0 = init_centroids
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
                # elif clustering_method == 'k_means_robust_lp':
                #     centroids_0, labels_0, inertia_0 = robust_k_means_LP_SDP(points,
                #                                                              centroids_input=indices,
                #                                                              k=n_centroids,
                #                                                              true_centroids=true_centroids,
                #                                                              true_labels=true_labels,
                #                                                              is_sdp=False)

                elif clustering_method.startswith('robust_lp'):
                    U_hat, new_true_centroids = robust_LP_SDP(points, k=n_centroids,  true_labels=true_labels,
                                                                            is_sdp=False)
                    init_centroids_0 = U_hat[indices, :]
                    if clustering_method == 'robust_lp_k_medians_l2':
                        centroids_0, labels_0, inertia_0 = k_medians_l2(U_hat, centroids_input=init_centroids_0,
                                                                        k=n_centroids, true_centroids=new_true_centroids)
                    elif clustering_method == 'robust_lp_k_medians_l1':
                        centroids_0, labels_0, inertia_0 = k_medians_l1(U_hat, centroids_input=init_centroids_0,
                                                                        k=n_centroids, true_centroids=new_true_centroids)
                    elif clustering_method == 'robust_lp_k_means':
                        centroids_0, labels_0, inertia_0 = k_means(U_hat, centroids_input=init_centroids_0,
                                                                   k=n_centroids, true_centroids=new_true_centroids)
                    else:
                        raise NotImplementedError()

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
            # if show and clustering_method.startswith('k_'):
            #     # only align the labels, do we need to align the centroids too for plotting.
            #     plot_centroids(points, cluster_size=100, clustering_method=clustering_method,
            #                    init_centroids=init_centroids_, title=f'{best_inertia}',
            #                    final_centroids=centroids_, final_labels=labels_,
            #                    out_dir=out_dir, x_axis=x_axis, random_state=random_state)
            mean_, std_ = np.mean(mps), 1.96 * np.std(mps) / np.sqrt(len(mps))
            best_ = (mean_, std_)
            # print(clustering_method, best_, x_axis)
            results[clustering_method] = {f'mp_mu': best_[0],
                                          f'mp_std': best_[1],
                                          f'acd_mu': 0,
                                          f'acd_std': 1,
                                          'params': {}}

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


# tolerance = 1e-4


def _tolerance(X, tol):
    import scipy.sparse as sp
    """Return a tolerance which is dependent on the dataset."""
    if tol == 0:
        return 0
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
    else:
        variances = np.var(X, axis=0)
    return np.mean(variances) * tol


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
    new_centroids = align_centroids(new_centroids, true_centroids, method='k_medians_l1')
    # note that it should be L1, not L2
    # distances2 = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    distances = np.sum(np.abs(points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
    labels = np.argmin(distances, axis=1)

    inertia = np.sum(np.min(distances, axis=1))
    return new_centroids, labels, inertia


def sc_k_medians_l1(projected_points, points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None,
                    true_labels=None):
    new_centroids = np.copy(centroids_input)
    # labels = np.copy(true_labels)
    # new_tolerance = _tolerance(projected_points, tolerance)
    for i in range(max_iterations):
        # pre_labels = np.copy(labels)
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

        # plot_centroids(projected_points, final_labels=labels, final_centroids=new_centroids, init_centroids=centroids_input,
        #                clustering_method=f'k_medians_l1, iter:{i}')

        if np.sum((new_centroids - pre_centroids) ** 2) / k < tolerance:
            break
        # if np.sum(pre_labels == labels) == len(labels):
        #     break
    # find the labels on the projected data first. Here should be L2
    distances = np.sum(np.abs(projected_points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
    labels = np.argmin(distances, axis=1)

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

        if np.sum((new_centroids - pre_centroids) ** 2) / k < tolerance:
            break

    new_centroids = align_centroids(new_centroids, true_centroids, method='k_medians_l2')
    # Here should be L2
    distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    labels = np.argmin(distances, axis=1)

    inertia = np.sum(np.min(distances, axis=1))
    return new_centroids, labels, inertia


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


def regularised_k_means_SDP(X, centroids_input, k, max_iterations=tot_iterate,
                            true_centroids=None, true_labels=None,
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
    new_centroids, labels, _ = k_means(X2, k, centroids_input, max_iterations, true_centroids)

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
    # print(f'theta: {theta}, gamma: {gamma}')
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
    # print(f'top {r} eigenvalues: {vals[-r:]}')
    U_hat = vecs[:, -r:]  # top r eigenvectors

    # # Step 4: Apply k-means clustering on rows of U_hat
    # print('Apply k-means clustering')
    # rng = np.random.RandomState(seed=random_state)
    # indices = rng.choice(range(len(U_hat)), size=r, replace=False)
    indices = centroids_input

    # Compute the true centroids based on U_hat
    new_centroids = np.zeros((k, k))
    for j in range(k):
        if sum(true_labels == j) == 0:
            # new_centroids[j] use the previous centroid
            continue
        new_centroids[j] = np.mean(U_hat[:len(true_labels)][true_labels == j], axis=0)

    new_centroids, labels, inertia = k_means(U_hat, k=r, centroids_input=U_hat[indices, :],
                                             max_iterations=tot_iterate, true_centroids=new_centroids)
    return new_centroids, labels, inertia


def robust_LP_SDP(X, k,
                          true_labels=None,
                          is_sdp=False, random_state=42):
    import numpy as np
    from numpy.linalg import eigh
    from scipy.spatial.distance import cdist
    from robust_sdp_lp import choose_theta_gamma, solve_lp_sdp

    Y = X
    theta, gamma = choose_theta_gamma(Y, beta=0.06, alpha=0.2)
    # print(f'theta: {theta}, gamma: {gamma}')
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
    # print(f'top {r} eigenvalues: {vals[-r:]}')
    U_hat = vecs[:, -r:]  # top r eigenvectors

    # # Step 4: Apply k-means clustering on rows of U_hat
    # print('Apply k-means clustering')
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

    return U_hat, new_centroids

