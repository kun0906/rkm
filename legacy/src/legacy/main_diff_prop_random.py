"""The impact of different proportions of outliers for different clustering methods.

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

    # num_repeat = 400
    n_repetitions = args.n_repetitions
    init_method = args.init_method
    true_single_cluster_size = args.true_single_cluster_size
    add_outlier = args.add_outlier
    n_neighbors = args.n_neighbors
    # out_dir = f'{args.out_dir}/diffdim/{init_method}/R_{num_repeat}-S_{true_single_cluster_size}'
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for n_centroids in range(4, 9, 5):
        # True labels
        true_labels = np.concatenate([np.ones(true_single_cluster_size) * i for i in range(n_centroids)]).astype(int)

        # dim = 50 ?
        dim = 10
        props = [0, 0.2, 0.4, 0.6, 0.8]
        prop_results = []
        for prop in tqdm(props):
            ith_prop_results = []
            for i in range(n_repetitions):
                # random seed
                seed = i
                rng = np.random.RandomState(seed=seed)

                # True centroids
                true_centroids = rng.normal(size=(n_centroids, dim))
                true_centroids /= np.linalg.norm(true_centroids, axis=1)[:, np.newaxis]
                # centroids /= max(np.linalg.norm(centroids, axis=1)[:, np.newaxis])
                radius = 5
                sigma = args.cluster_std
                true_centroids *= radius

                # True points
                cov = np.identity(dim)
                true_points = np.concatenate(
                    [rng.multivariate_normal(mean, cov * (sigma ** 2), size=true_single_cluster_size) for mean in
                     true_centroids])

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
                ith_prop_results.append(ith_repeat_results)

            # Compute mean and error bar for ith_prop_results
            ith_avg_results = compute_ith_avg(ith_prop_results, prop)
            prop_results.append(ith_avg_results)

        # Collect all the results togather
        avg_results = {}
        for key in prop_results[0].keys():
            avg_results[key] = [prop_results[i_][key] for i_ in range(len(props))]
        df = pd.DataFrame(avg_results)
        # Save data to CSV file
        df.to_csv(f'{out_dir}/data_%g_clusters.csv' % n_centroids, index=False)

        title = "Plot of mp: %g_clusters_rad_%g_out_%g_sigma_%g" % (n_centroids, radius, prop, sigma)
        plot_result(df, out_dir, xlabel="Noise Proportion", ylabel="Misclustering proportion (MP)", title=title)


if __name__ == '__main__':
    main()
