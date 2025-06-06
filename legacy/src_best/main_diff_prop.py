"""The impact of different proportions of outliers for different clustering methods.

"""
import math
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from base import compute_ith_avg, plot_result
from utils import parse_arguments


def main():
    args = parse_arguments()
    args.add_outlier = False if args.add_outlier == 'False' else True
    print(args)

    n_repetitions = args.n_repetitions
    init_method = args.init_method
    true_single_cluster_size = args.true_single_cluster_size
    add_outlier = args.add_outlier
    n_neighbors = args.n_neighbors
    theta = args.theta
    m = args.m
    # out_dir = f'{args.out_dir}/diffdim/{init_method}/R_{n_repetitions}-S_{true_single_cluster_size}'
    out_dir = args.out_dir
    print(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if init_method == 'random' or init_method == 'robust_init':
        from clustering_random import get_ith_results_random
    else:
        from clustering import get_ith_results

    for n_centroids in range(4, 9, 5):
        # True labels
        true_labels = np.concatenate([np.ones(true_single_cluster_size) * i for i in range(n_centroids)]).astype(int)
        dim = 10
        props = [0, 0.2, 0.4, 0.6, 0.8]
        prop_results = []
        for prop in tqdm(props):
            # Generate data first
            datasets = []
            for i in range(n_repetitions):
                # random seed
                seed = i
                rng = np.random.RandomState(seed=seed)

                # True centroids
                true_centroids = rng.normal(size=(n_centroids, dim))
                true_centroids /= np.linalg.norm(true_centroids, axis=1)[:, np.newaxis]
                # centroids /= max(np.linalg.norm(centroids, axis=1)[:, np.newaxis])
                radius = args.radius
                sigma = args.cluster_std
                true_centroids *= radius

                # True points
                cov = np.identity(dim)
                true_points = np.concatenate(
                    [rng.multivariate_normal(mean, cov * (sigma ** 2), size=true_single_cluster_size) for mean in
                     true_centroids])


                # Fraction of outliers
                adding_partial_direction_oultiers = False
                if adding_partial_direction_oultiers:
                    outlier_std = 10
                    m = math.floor(true_single_cluster_size * prop)
                    outliers = np.zeros((m, dim))
                    m_cols = dim // 2 + 1
                    partial_outliers = rng.normal(loc=0, scale=outlier_std, size=(m, m_cols))
                    outliers[:, :m_cols] = partial_outliers
                else:
                    # Fraction of outliers
                    # prop = 0.60
                    outlier_std = 10
                    # outlier_std = 2
                    outliers = rng.multivariate_normal(np.ones(dim) * 0,
                                                       np.eye(dim) * outlier_std ** 2,
                                                       size=math.floor(true_single_cluster_size * prop))
                # Final points
                if add_outlier:
                    points = np.concatenate((true_points, outliers), axis=0)
                    # labels = np.concatenate([true_labels, 10 * np.ones((outliers.shape[0],))])
                else:
                    # Without outliers
                    points = true_points

                if init_method == 'random':
                    indices = rng.choice(range(len(points)), size=n_centroids, replace=False)
                    init_centroids = points[indices, :]
                elif init_method == 'robust_init':
                    import init_k_cent
                    init_centroids, _ = init_k_cent.iodk(points, n_centroids, m1=20, m=10, beta=0.1)
                else:
                    init_centroids = true_centroids
                data = {
                    "true_centroids": true_centroids, "true_labels": true_labels,
                    "true_single_cluster_size": true_single_cluster_size,
                    "n_centroids": n_centroids,
                    'points': points,
                    'init_centroids': init_centroids,
                    "random_state": seed
                }
                datasets.append(data)

            # Get avg result
            if init_method == 'random' or init_method == 'random_init':
                ith_prop_results = get_ith_results_random(datasets, out_dir=out_dir, x_axis=prop)
            else:
                ith_prop_results = get_ith_results(datasets, out_dir=out_dir, x_axis=prop)

            prop_results.append(ith_prop_results)

        # Collect all the results togather
        avg_results = {'x_axis': props}
        for cluster_method in prop_results[0].keys():
            for metric in ['mp', 'acd']:
                mu_ = f'{cluster_method}_{metric}_mu'
                avg_results[mu_] = [prop_results[i_][cluster_method][f'{metric}_mu'] for i_ in range(len(props))]
                std_ = f'{cluster_method}_{metric}_std'
                avg_results[std_] = [prop_results[i_][cluster_method][f'{metric}_std'] for i_ in range(len(props))]

        df = pd.DataFrame(avg_results)
        # Save data to CSV file
        df.to_csv(f'{out_dir}/data_%g_clusters.csv' % n_centroids, index=False)

        title = "Plot of mp: %g_clusters_rad_%g_out_%g_sigma_%g" % (n_centroids, radius, prop, sigma)
        plot_result(df, out_dir, xlabel="Noise Proportion", ylabel="Misclustering proportion (MP)", title=title)


if __name__ == '__main__':
    main()
