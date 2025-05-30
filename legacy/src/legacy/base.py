import matplotlib.pyplot as plt
import numpy as np
CLUSTERING_METHODS = ['k_medians_l2', 'k_medians_l1', 'k_means',
                      'sc_k_medians_l2', 'sc_k_medians_l1', 'sc_k_means',
                      'rsc_k_medians_l2', 'rsc_k_medians_l1', 'rsc_k_means']

tot_iterate = 50
tolerance = 1e-4
def plot_result(df, out_dir, xlabel='', ylabel='', title=''):
    # Plot the line plot with error bars
    fig, ax = plt.subplots(figsize=(8, 6))

    X_axis = df['X-axis']
    LINESTYLES_COLORS_LABELS = {
        'k_medians_l2': ('-.', 'green', 'Lloyd-$L_1$'),  # linestyle, color, label
        'k_medians_l1': ('--', 'purple', 'k-median'),
        'k_means': ('-', 'blue', 'Llyod (k_means)'),

        'sc_k_medians_l2': ('-o', 'lightgreen', 'SC-Lloyd-$L_1$'),
        'sc_k_medians_l1': ('-^', 'violet', 'SC-k-median'),
        'sc_k_means': ('-s', 'skyblue', 'SC-Llyod (k_means)'),

        'rsc_k_medians_l2': ('-+', 'lime', 'RSC-Lloyd-$L_1$'),
        'rsc_k_medians_l1': ('-x', 'fuchsia', 'RSC-k-median'),
        'rsc_k_means': ('-p', 'steelblue', 'RSC-Llyod (k_means)')

    }
    for clustering_method in CLUSTERING_METHODS:
        ls, color, label = LINESTYLES_COLORS_LABELS[clustering_method]
        y, yerr = df[f'{clustering_method}_mp_mu'], df[f'{clustering_method}_mp_std']
        plt.plot(X_axis, y, ls, label=label, color=color)
        plt.errorbar(X_axis, y, yerr=yerr, fmt='none', ecolor='black', capsize=3)

    ax.set_xticks(X_axis)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    # temp = time.time()
    # temp = 0
    plt.savefig(f'{out_dir}/mp.png', dpi=300)
    plt.show()
    # plt.pause(2)


def get_ith():
    for clustering_method in CLUSTERING_METHODS:

        if clustering_method == 'k_medians_l2':
            centroids_, labels_ = k_medians_l2(points, centroids_input=copy.deepcopy(true_centroids),
                                          k=n_centroids)
        elif clustering_method == 'k_medians_l1':
            centroids_, labels_ = k_medians_l1(points, centroids_input=copy.deepcopy(true_centroids),
                                       k=n_centroids)
        elif clustering_method == 'k_means':
            centroids_, labels_ = k_means(points, centroids_input=copy.deepcopy(true_centroids),
                                         k=n_centroids)

        elif clustering_method == 'sc_k_medians_l2':
            # find the projected centroids
            k = n_centroids
            X = np.concatenate([true_centroids, points], axis=0)
            X_projected = sc_projection(X, k, random_state=seed)
            projected_true_centroids = X_projected[:k, :]
            projected_points = X_projected[k:, :]

            centroids_, labels_ = sc_k_medians_l2(projected_points, points, k, projected_true_centroids,
                                             max_iterations=50,
                                             true_centroids=true_centroids, true_labels=None)

        elif clustering_method == 'sc_k_medians_l1':
            # find the projected centroids
            k = n_centroids
            X = np.concatenate([true_centroids, points], axis=0)
            X_projected = sc_projection(X, k, random_state=seed)
            projected_true_centroids = X_projected[:k, :]
            projected_points = X_projected[k:, :]

            centroids_, labels_ = sc_k_medians_l1(projected_points, points, k, projected_true_centroids,
                                          max_iterations=50,
                                          true_centroids=true_centroids, true_labels=None)

        elif clustering_method == 'sc_k_means':
            # find the projected centroids
            k = n_centroids
            X = np.concatenate([true_centroids, points], axis=0)
            X_projected = sc_projection(X, k, random_state=seed)
            projected_true_centroids = X_projected[:k, :]
            projected_points = X_projected[k:, :]

            centroids_, labels_ = sc_k_means(projected_points, points, k, projected_true_centroids,
                                            max_iterations=50,
                                            true_centroids=true_centroids, true_labels=None)

        elif clustering_method == 'rsc_k_medians_l2':
            # find the projected centroids
            k = n_centroids
            X = np.concatenate([true_centroids, points], axis=0)
            X_projected = robust_sc_projection(X, k, random_state=seed)
            projected_true_centroids = X_projected[:k, :]
            projected_points = X_projected[k:, :]

            centroids_, labels_ = sc_k_medians_l2(projected_points, points, k, projected_true_centroids,
                                             max_iterations=50,
                                             true_centroids=true_centroids, true_labels=None)

        elif clustering_method == 'rsc_k_medians_l1':
            # find the projected centroids
            k = n_centroids
            X = np.concatenate([true_centroids, points], axis=0)
            X_projected = robust_sc_projection(X, k, random_state=seed)
            projected_true_centroids = X_projected[:k, :]
            projected_points = X_projected[k:, :]

            centroids_, labels_ = sc_k_medians_l1(projected_points, points, k, projected_true_centroids,
                                          max_iterations=50,
                                          true_centroids=true_centroids, true_labels=None)
        elif clustering_method == 'rsc_k_means':
            # find the projected centroids
            k = n_centroids
            X = np.concatenate([true_centroids, points], axis=0)
            X_projected = robust_sc_projection(X, k, random_state=seed)
            projected_true_centroids = X_projected[:k, :]
            projected_points = X_projected[k:, :]

            centroids_, labels_ = sc_k_means(projected_points, points, k, projected_true_centroids,
                                            max_iterations=50,
                                            true_centroids=true_centroids, true_labels=None)

        else:
            raise NotImplementedError()

        mp = sum(labels_[range(n_centroids * true_single_cluster_size)] != true_labels) / len(true_labels)
        acd = np.sum((centroids_ - true_centroids) ** 2) / n_centroids
        ith_dim_repeat_results[clustering_method] = {'centroids': centroids_, 'labels': labels_,
                                                     'mp': mp, 'acd': acd, }


        return ith_dim_repeat_results