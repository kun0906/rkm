"""The impact of different variances for different clustering methods.

"""
import argparse
import math
import os

import pandas as pd
from tqdm import tqdm

from clustering_random import *


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--force', default=False,   # whether overwrite the previous results or not?
    #                     action='store_true', help='force')
    parser.add_argument("--n_repetitions", type=int, default=2)  #
    parser.add_argument("--true_single_cluster_size", type=int, default=100)
    parser.add_argument("--init_method", type=str, default='random')
    parser.add_argument("--add_outlier", type=str, default='True')
    parser.add_argument("--out_dir", type=str, default='out')
    parser.add_argument("--cluster_std", type=float, default=1)
    parser.add_argument("--n_neighbors", type=int, default=15)
    args = parser.parse_args()
    args.add_outlier = False if args.add_outlier == 'False' else True
    print(args)

    n_repetitions = args.n_repetitions
    init_method = args.init_method
    true_single_cluster_size = args.true_single_cluster_size
    add_outlier = args.add_outlier
    n_neighbors = args.n_neighbors
    # out_dir = f'{args.out_dir}/diffdim/{init_method}/R_{num_repeat}-S_{true_single_cluster_size}'
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    results = {'mp': {}, 'acd': {}}
    for n_centroids in range(4, 9, 5):
        # True labels
        true_labels = np.concatenate([np.ones(true_single_cluster_size) * i for i in range(n_centroids)]).astype(int)
        dim = 10
        sigma_out_vec = np.trunc(np.linspace(1, 20, 11))
        sigma_results = []
        for sigma_out in tqdm(sigma_out_vec):
            ith_sigma_results = []
            for i in range(n_repetitions):

                # random seed
                seed = i
                rng = np.random.RandomState(seed=seed)

                # True centroids
                true_centroids = rng.normal(size=(n_centroids, dim))
                true_centroids /= np.linalg.norm(true_centroids, axis=1)[:, np.newaxis]
                # centroids /= max(np.linalg.norm(centroids, axis=1)[:, np.newaxis])

                # True points
                radius = 5
                sigma = args.cluster_std  # 1
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
                else:
                    # Without outliers
                    points = true_points

                if init_method == 'random':
                    indices = rng.choice(range(len(points)), size=n_centroids, replace=False)
                    init_centroids = points[indices, :]
                    # init_centroids = np.copy(centroids)
                else:
                    pass

                ith_repeat_results = get_ith_results_random(points, init_centroids,
                                                            true_centroids, true_labels, true_single_cluster_size,
                                                            n_centroids, n_neighbors=n_neighbors,
                                                            random_state=seed)
                ith_sigma_results.append(ith_repeat_results)

            # Compute mean and error bar for ith_sigma_results
            ith_avg_results = compute_ith_avg(ith_sigma_results, sigma_out)
            sigma_results.append(ith_avg_results)

        # Collect all the results togather
        avg_results = {}
        for key in sigma_results[0].keys():
            avg_results[key] = [sigma_results[i_][key] for i_ in range(len(sigma_out_vec))]
        df = pd.DataFrame(avg_results)
        # Save data to CSV file
        df.to_csv(f'{out_dir}/data_%g_clusters.csv' % n_centroids, index=False)

        title = "Plot of mp: %g_clusters_rad_%g_out_%g_dim_%g" % (n_centroids, radius, prop, dim)
        plot_result(df, out_dir, xlabel='Outlier std', ylabel="Misclustering proportion (MP)", title=title)


if __name__ == '__main__':
    main()
