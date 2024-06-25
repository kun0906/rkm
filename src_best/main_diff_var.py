"""The impact of different standard deviation (variances) for different clustering methods.

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

    if init_method == 'random':
        from clustering_random import get_ith_results_random
    else:
        from clustering import get_ith_results

    for n_centroids in range(4, 9, 5):
        # True labels
        true_labels = np.concatenate([np.ones(true_single_cluster_size) * i for i in range(n_centroids)]).astype(int)
        dim = 10
        # dim = 5
        sigma_out_vec = np.trunc(np.linspace(1, 20, 11))

        sigma_results = []
        for sigma_out in tqdm(sigma_out_vec):
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

                # True points
                radius = args.radius     # 5
                sigma = args.cluster_std  # 2
                true_centroids *= radius
                # Set means and covariance matrices
                cov = np.identity(dim)
                true_points = np.concatenate(
                    [rng.multivariate_normal(mean, cov * (sigma ** 2), size=true_single_cluster_size) for mean in
                     true_centroids])

                # Fraction of outliers
                prop = 0.60
                outliers = rng.multivariate_normal(np.ones(dim) * 0,
                                                   np.eye(dim) * sigma_out ** 2,
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
            if init_method == 'random':
                ith_sigma_results = get_ith_results_random(datasets, out_dir=out_dir, x_axis=sigma_out)
            else:
                ith_sigma_results = get_ith_results(datasets, out_dir=out_dir, x_axis=sigma_out)

            sigma_results.append(ith_sigma_results)

        # Collect all the results togather
        avg_results = {'x_axis': sigma_out_vec}
        for cluster_method in sigma_results[0].keys():
            for metric in ['mp', 'acd']:
                mu_ = f'{cluster_method}_{metric}_mu'
                avg_results[mu_] = [sigma_results[i_][cluster_method][f'{metric}_mu'] for i_ in range(len(sigma_out_vec))]
                std_ = f'{cluster_method}_{metric}_std'
                avg_results[std_] = [sigma_results[i_][cluster_method][f'{metric}_std'] for i_ in range(len(sigma_out_vec))]

        df = pd.DataFrame(avg_results)
        # Save data to CSV file
        df.to_csv(f'{out_dir}/data_%g_clusters.csv' % n_centroids, index=False)

        title = "Plot of mp: %g_clusters_rad_%g_out_%g_dim_%g" % (n_centroids, radius, prop, dim)
        plot_result(df, out_dir, xlabel='Outlier std', ylabel="Misclustering proportion (MP)", title=title)


if __name__ == '__main__':
    main()
