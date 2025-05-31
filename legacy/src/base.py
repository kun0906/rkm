"""

"""
import os

import numpy as np
import scipy.stats as stats
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

from robust_spectral_clustering import RSC

tot_iterate = 50
tolerance = 1e-4

import matplotlib.pyplot as plt

# for testing
CLUSTERING_METHODS = ['k_medians_l2', 'k_medians_l1', 'k_means',
                      # 'rsc_k_means_orig' # robust k_means from the original api
                      ]
#
# CLUSTERING_METHODS = ['k_medians_l2', 'k_medians_l1', 'k_means', 'k_means_sdp',
#                       'sc_k_medians_l2', 'sc_k_medians_l1', 'sc_k_means',
#                       'rsc_k_medians_l2', 'rsc_k_medians_l1', 'rsc_k_means',
#                       'rsc_k_means_orig'  # robust k_means from the original api
#                       ]

LINESTYLES_COLORS_LABELS = {
    'k_medians_l2': ('-.', 'green', '$k$-medians-hybrid'),  # linestyle, color, label
    'k_medians_l1': ('--', 'purple', '$k$-medians-$\ell_1$'),
    'k_means': ('-', 'blue', '$k$_means'),

    'sc_k_medians_l2': ('-o', 'lightgreen', 'SC-$k$-medians-hybrid'),
    'sc_k_medians_l1': ('-^', 'violet', 'SC-$k$-medians-$\ell_1$'),
    'sc_k_means': ('-s', 'skyblue', 'SC-$k$_means'),

    'rsc_k_medians_l2': ('-+', 'lime', 'RSC-$k$-medians-hybrid'),
    'rsc_k_medians_l1': ('-x', 'fuchsia', 'RSC-$k$-medians-$\ell_1$'),
    'rsc_k_means': ('-p', 'steelblue', 'RSC-$k$_means'),

    'rsc_k_means_orig': ('-*', 'red', 'RSC-k_means-orig')
}


def plot_result(df, out_dir, out_name='mp', xlabel='', ylabel='', title='', show=False):
    os.makedirs(out_dir, exist_ok=True)
    # Plot the line plot with error bars
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

    X_axis = df['x_axis']

    for clustering_method in ['k_medians_l2', 'k_medians_l1', 'k_means']:
        ls, color, label = LINESTYLES_COLORS_LABELS[clustering_method]
        y, yerr = df[f'{clustering_method}_mp_mu'], df[f'{clustering_method}_mp_std']
        ax[0, 0].plot(X_axis, y, ls, label=label, color=color)
        ax[0, 0].errorbar(X_axis, y, yerr=yerr, fmt='none', ecolor='black', capsize=3)
        ax[0, 0].set_xticks(X_axis)
        # ax[0, 0].set_title('Original Points')
        ax[0, 0].set_xlabel(xlabel)
        ax[0, 0].set_ylabel(ylabel)
        ax[0, 0].legend(loc='upper left')

    # for clustering_method in ['sc_k_medians_l2', 'sc_k_medians_l1', 'sc_k_means']:
    #     ls, color, label = LINESTYLES_COLORS_LABELS[clustering_method]
    #     y, yerr = df[f'{clustering_method}_mp_mu'], df[f'{clustering_method}_mp_std']
    #     ax[0, 1].plot(X_axis, y, ls, label=label, color=color)
    #     ax[0, 1].errorbar(X_axis, y, yerr=yerr, fmt='none', ecolor='black', capsize=3)
    #     ax[0, 1].set_xticks(X_axis)
    #     # ax[0, 1].set_title('Projected Points with SC')
    #     ax[0, 1].set_xlabel(xlabel)
    #     ax[0, 1].set_ylabel(ylabel)
    #     ax[0, 1].legend(loc='upper left')
    #
    # for clustering_method in ['rsc_k_medians_l2', 'rsc_k_medians_l1', 'rsc_k_means', 'rsc_k_means_orig']:
    #     ls, color, label = LINESTYLES_COLORS_LABELS[clustering_method]
    #     y, yerr = df[f'{clustering_method}_mp_mu'], df[f'{clustering_method}_mp_std']
    #     ax[1, 0].plot(X_axis, y, ls, label=label, color=color)
    #     ax[1, 0].errorbar(X_axis, y, yerr=yerr, fmt='none', ecolor='black', capsize=3)
    #     ax[1, 0].set_xticks(X_axis)
    #     # ax[1, 0].set_title('Projected Points with RSC')
    #     ax[1, 0].set_xlabel(xlabel)
    #     ax[1, 0].set_ylabel(ylabel)
    #     ax[1, 0].legend(loc='upper left')

    for clustering_method in CLUSTERING_METHODS:
        ls, color, label = LINESTYLES_COLORS_LABELS[clustering_method]
        y, yerr = df[f'{clustering_method}_mp_mu'], df[f'{clustering_method}_mp_std']
        ax[1, 1].plot(X_axis, y, ls, label=label, color=color)
        ax[1, 1].errorbar(X_axis, y, yerr=yerr, fmt='none', ecolor='black', capsize=3)
        ax[1, 1].set_xticks(X_axis)
        # ax[1, 1].set_title('Points')
        ax[1, 1].set_xlabel(xlabel)
        ax[1, 1].set_ylabel(ylabel)
        ax[1, 1].legend(loc='upper left')

    fig.suptitle(title)
    # plt.legend()
    plt.tight_layout()
    # temp = time.time()
    # temp = 0
    plt.savefig(f'{out_dir}/{out_name}.png', dpi=300)
    if show: plt.show()
    # plt.pause(2)
    plt.close()

def compute_bandwidth(X):
    pd = pairwise_distances(X, Y=None, metric='euclidean')
    beta = 0.3
    qs = np.quantile(pd, q=beta, axis=1)
    alpha = 0.01
    n, d = X.shape
    df = d  # degrees of freedom
    denominator = np.sqrt(stats.chi2.ppf((1 - alpha), df))
    bandwidth = np.quantile(qs, (1 - alpha)) / denominator

    return bandwidth


# @timer
def sc_projection(points, k, n_neighbors=10, affinity = 'knn', normalize=True, random_state=42):
    from sklearn.metrics import pairwise_kernels
    params = {}  # default value in slkearn
    # https://github.com/scikit-learn/scikit-learn/blob/872124551/sklearn/cluster/_spectral.py#L667
    # Number of eigenvectors to use for the spectral embedding, default=n_clusters
    n_components = k
    eigen_tol = 0.0
    eigen_solver = None
    # affinity = 'rbf'  # affinity str or callable, default =’rbf’
    if affinity == 'rbf':
        # params["gamma"] = 1.0  # ?
        sigma = compute_bandwidth(points)
        params["gamma"] = 1 / (2 * sigma ** 2)

        params["degree"] = 3
        params["coef0"] = 1
        # eigen_solver{‘arpack’, ‘lobpcg’, ‘amg’}, default=None
        # The eigenvalue decomposition strategy to use. AMG requires pyamg to be installed.
        # It can be faster on very large, sparse problems, but may also lead to instabilities.
        # If None, then 'arpack' is used. See [4] for more details regarding 'lobpcg'.
        affinity_matrix_ = pairwise_kernels(
            points, metric=affinity, filter_params=True, **params,
        )
    else:
        # if affinity == "nearest_neighbors":
        connectivity = kneighbors_graph(
            points, n_neighbors=n_neighbors, metric = 'euclidean', include_self = False, mode = 'connectivity')
        # affinity_matrix_ = 0.5 * (connectivity + connectivity.T).toarray()
        affinity_matrix_ = connectivity.maximum(connectivity.T).toarray()  # make the graph undirected

    # We now obtain the real valued solution matrix to the
    # relaxed Ncut problem, solving the eigenvalue problem
    # L_sym x = lambda x  and recovering u = D^-1/2 x.
    # The first eigenvector is constant only for fully connected graphs
    # and should be kept for spectral clustering (drop_first = False)
    # See spectral_embedding documentation.
    from sklearn.manifold import spectral_embedding
    H = spectral_embedding(
        affinity_matrix_,  # n xn
        n_components=n_components,
        eigen_solver=eigen_solver,
        random_state=random_state,
        eigen_tol=eigen_tol,
        drop_first=False,
    )
    # print(np.min(points), np.max(points), np.min(affinity_matrix_), np.max(affinity_matrix_), np.min(maps), np.max(maps), flush=True)
    # MAX=1e+5
    # maps[maps > MAX] = MAX  # avoid overflow in np.square()
    # maps[maps < -MAX] = -MAX
    # print(np.min(points), np.max(points), np.min(affinity_matrix_), np.max(affinity_matrix_), np.min(maps),
    #       np.max(maps), flush=True)
    if normalize:
        projected_points = H / np.linalg.norm(H, axis=1)[:, None]
    else:
        projected_points = H
    return projected_points


def rsc_projection(points, k, n_neighbors=15, theta=50, m=0.5, random_state=42):
    """ Robust Spectral clustering
        https://github.com/abojchevski/rsc/tree/master

        RSC(k=k, nn=n_neighbors, theta=60*400, m=0.5,laplacian=1,  verbose=False, random_state=random_state)

        theta = number of corrupted edges we want to remove. E.g., if we want to remove 10, then theta = 5.
        m is minimum  percentage of neighbours will be removed for each node (omega_i constraints)

        """
    rsc = RSC(k=k, nn=n_neighbors, theta=theta, m=m, laplacian=1, normalize=True, verbose=False,
              random_state=random_state)
    # y_rsc = rsc.fit_predict(X)
    Ag, Ac, H = rsc._RSC__latent_decomposition(points)
    # # Ag: similarity matrix of good points
    # # Ac: similarity matrix of corruption points
    # # A = Ag + Ac
    # rsc.Ag = Ag
    # rsc.Ac = Ac

    if rsc.normalize:
        rsc.H = H / np.linalg.norm(H, axis=1)[:, None]
    else:
        rsc.H = H

    projected_points = rsc.H
    return projected_points


def compute_ith_avg(ith_prop_results, prop):
    ith_avg_results = {'x_axis': prop}
    for clustering_method in CLUSTERING_METHODS:
        for metric in ['mp', 'acd']:
            scores_ = [res[clustering_method][metric] for res in ith_prop_results]
            mu_ = np.mean(scores_)
            std_ = 1.96 * np.std(scores_) / np.sqrt(len(scores_))
            if mu_ >= 0.5:
                print(clustering_method, prop, scores_, mu_, std_,
                      [res[clustering_method]['labels'] for res in ith_prop_results])
            column_name = f'{clustering_method}_{metric}_mu'
            ith_avg_results[column_name] = mu_
            column_name = f'{clustering_method}_{metric}_std'
            ith_avg_results[column_name] = std_

    return ith_avg_results


def plot_projected_data(points, projected_points, cluster_size=100, clustering_method='k_means',
                        centroids=None, projected_centroids=None, n_clusters=4,
                        out_dir='', x_axis=None, random_state=42):
    # Create a color map for 5 classes
    colors = ['red', 'green', 'blue', 'purple', 'orange']

    # Plot the figures
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # fig, axes = plt.subplots(2, 2)

    # Plot points without outliers
    for i in range(n_clusters):
        x1 = points[i * cluster_size: (i + 1) * cluster_size, 0]
        x2 = points[i * cluster_size: (i + 1) * cluster_size, 1]
        axes[0, 0].scatter(x1, x2, color=colors[i], label=f'Class {i}', alpha=0.4)
        if centroids is not None and i < centroids.shape[0]:
            axes[0, 0].scatter(centroids[i, 0], centroids[i, 1], color='black', marker='x', s=100, label=f'{i}')
        # if projected_centroids is not None and i < projected_centroids.shape[0]:
        #     axes[0, 0].scatter(projected_centroids[i, 0], projected_centroids[i, 1], color='black', marker='s',  s = 100, label=f'{i}')
    axes[0, 0].set_title('Points without outliers')
    axes[0, 0].set_xlabel('X axis')
    axes[0, 0].set_ylabel('Y axis')
    # axes[0, 0].legend()
    #
    # Plot points with outliers
    for i in range(n_clusters + 1):
        x1 = points[i * cluster_size: (i + 1) * cluster_size, 0]
        x2 = points[i * cluster_size: (i + 1) * cluster_size, 1]
        axes[0, 1].scatter(x1, x2, color=colors[i], label=f'Class {i}', alpha=0.4)
        # print(i, x1, x2)
        if centroids is not None and i < centroids.shape[0]:
            axes[0, 1].scatter(centroids[i, 0], centroids[i, 1], color='black', marker='x', s=100, label=f'{i}')
        # if projected_centroids is not None and i < projected_centroids.shape[0]:
        #     axes[0, 1].scatter(projected_centroids[i, 0], projected_centroids[i, 1], color='yellow', marker='s', s=100,
        #                        label=f'{i}')
    axes[0, 1].set_title('Points')
    axes[0, 1].set_xlabel('X axis')
    axes[0, 1].set_ylabel('Y axis')
    # axes[0, 1].legend()

    # Plot projected points without outliers
    for i in range(n_clusters):
        x1 = projected_points[i * cluster_size: (i + 1) * cluster_size, 0]
        x2 = projected_points[i * cluster_size: (i + 1) * cluster_size, 1]
        axes[1, 0].scatter(x1, x2, color=colors[i], label=f'Class {i}', alpha=0.4)
        # if centroids is not None and i < centroids.shape[0]:
        #     axes[1, 0].scatter(centroids[i, 0], centroids[i, 1], color='black', marker='x',  s = 100, label=f'{i}')
        if projected_centroids is not None and i < projected_centroids.shape[0]:
            axes[1, 0].scatter(projected_centroids[i, 0], projected_centroids[i, 1], color='black', marker='s', s=100,
                               label=f'{i}')
    axes[1, 0].set_title('Projected Points without outliers')
    axes[1, 0].set_xlabel('X axis')
    axes[1, 0].set_ylabel('Y axis')
    # axes[1, 0].legend()

    # Plot projected points with outliers
    for i in range(n_clusters + 1):
        x1 = projected_points[i * cluster_size: (i + 1) * cluster_size, 0]
        x2 = projected_points[i * cluster_size: (i + 1) * cluster_size, 1]
        axes[1, 1].scatter(x1, x2, color=colors[i], label=f'Class {i}', alpha=0.4)
        # if centroids is not None and i < centroids.shape[0]:
        #     axes[1, 1].scatter(centroids[i, 0], centroids[i, 1], color='black', marker='x', s=100, label=f'{i}')
        if projected_centroids is not None and i < projected_centroids.shape[0]:
            axes[1, 1].scatter(projected_centroids[i, 0], projected_centroids[i, 1], color='black', marker='s', s=100,
                               label=f'{i}')
    axes[1, 1].set_title('Projected Points')
    axes[1, 1].set_xlabel('X axis')
    axes[1, 1].set_ylabel('Y axis')
    # axes[1, 1].legend()

    fig.suptitle(f'Seed: {random_state},{clustering_method}')
    plt.tight_layout()
    out_dir = os.path.join(out_dir, 'tmp')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/{x_axis}-{clustering_method}-Seed_{random_state}.png')
    plt.show()
