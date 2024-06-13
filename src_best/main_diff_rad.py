"""The impact of different locations of outliers for different clustering methods.

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

    for n_centroids in range(4, 9, 6):
        # True labels
        true_labels = np.concatenate([np.ones(true_single_cluster_size) * i for i in range(n_centroids)]).astype(int)
        dim = 10
        rad_out_vec = np.trunc(np.linspace(0, 100, 11)) # choose 11 values from [0, 100]
        rad_results = []
        for rad_out in tqdm(rad_out_vec):
            ith_rad_results = []
            for i in range(n_repetitions):
                # random seed
                seed = i
                rng = np.random.RandomState(seed=seed)

                # True centroids
                true_centroids = rng.normal(size=(n_centroids, dim))
                true_centroids /= np.linalg.norm(true_centroids, axis=1)[:, np.newaxis]
                # centroids /= max(np.linalg.norm(centroids, axis=1)[:, np.newaxis])
                radius = 5
                # sigma = args.cluster_std
                sigma = 1.0   # in the paper, we use sigma=1.0 for this case.
                true_centroids *= radius

                # True points
                # Set means and covariance matrices
                true_points = np.concatenate(
                    [rng.multivariate_normal(mean, np.identity(dim) * sigma ** 2, size=true_single_cluster_size) for
                     mean in
                     true_centroids])

                # Fraction of outliers
                # prop = 0.6
                # sigma_out = 10      # for testing
                prop = 0.4
                sigma_out = 2
                # outliers = rad_out/np.sqrt(dim) + sigma_out * rng.multivariate_normal(np.zeros(dim), np.eye(dim),
                #                                                      size = math.floor(true_single_cluster_size * prop))
                centroids_out_dir = rng.multivariate_normal(np.zeros(dim), np.eye(dim), size=1)
                centroids_out_dir /= np.linalg.norm(centroids_out_dir, axis=1)[:, np.newaxis]
                # outlier_mean = rad_out / np.sqrt(dim) * centroids_out_dir[0]
                outlier_mean = rad_out * centroids_out_dir[0]
                outliers = outlier_mean + rng.multivariate_normal(np.zeros(dim), np.eye(dim) * sigma_out ** 2,
                                                                  size=math.floor(true_single_cluster_size * prop))

                # Final points
                if add_outlier:
                    points = np.concatenate((true_points, outliers), axis=0)
                else:
                    # Without outliers
                    points = true_points

                if init_method == 'random':
                    indices = rng.choice(range(len(points)), size=n_centroids, replace=False)
                    init_centroids = points[indices, :]
                    ith_repeat_results = get_ith_results_random(points, init_centroids,
                                                                true_centroids, true_labels, true_single_cluster_size,
                                                                n_centroids,
                                                                n_neighbors=n_neighbors, theta=theta, m=m,
                                                                out_dir=out_dir, x_axis=rad_out, random_state=seed)
                else:
                    ith_repeat_results = get_ith_results(points, true_centroids, true_labels, true_single_cluster_size,
                                                         n_centroids,
                                                         n_neighbors=n_neighbors, theta=theta, m=m,
                                                         out_dir=out_dir, x_axis=rad_out, random_state=seed)
                ith_rad_results.append(ith_repeat_results)

            # Compute mean and error bar for ith_rad_results
            ith_avg_results = compute_ith_avg(ith_rad_results, rad_out)
            rad_results.append(ith_avg_results)

        # Collect all the results togather
        avg_results = {}
        for key in rad_results[0].keys():
            avg_results[key] = [rad_results[i_][key] for i_ in range(len(rad_out_vec))]
        df = pd.DataFrame(avg_results)
        # Save data to CSV file
        df.to_csv(f'{out_dir}/data_%g_clusters.csv' % n_centroids, index=False)

        title = "Plot of mp: %g_clusters_rad_%g_out_%g_dim_%g" % (n_centroids, radius, prop, dim)
        plot_result(df, out_dir, xlabel="Outlier rad", ylabel="Misclustering proportion (MP)", title=title)


if __name__ == '__main__':
    main()
