"""The impact of different proportions of outliers for different clustering methods.

"""
import argparse
import os

import pandas as pd
from tqdm import tqdm

from clustering_random import *
from data.gen_data import gen_data


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--force', default=False,   # whether overwrite the previous results or not?
    #                     action='store_true', help='force')
    parser.add_argument("--n_repetitions", type=int, default=2)  #
    parser.add_argument("--true_single_cluster_size", type=int, default=100)
    parser.add_argument("--init_method", type=str, default='random')
    parser.add_argument("--add_outlier", type=str, default='True')
    parser.add_argument("--out_dir", type=str, default='out')
    parser.add_argument("--data_name", type=str, default='pen_digits')
    parser.add_argument("--fake_label", type=str, default='special')
    parser.add_argument("--cluster_std", type=float, default=0)
    parser.add_argument("--n_neighbors", type=int, default=15)
    args = parser.parse_args()
    args.add_outlier = False if args.add_outlier == 'False' else True
    print(args)

    # num_repeat = 400
    n_repetitions = args.n_repetitions
    init_method = args.init_method
    true_single_cluster_size = args.true_single_cluster_size
    add_outlier = args.add_outlier
    data_name = args.data_name
    fake_label = args.fake_label
    n_neighbors = args.n_neighbors
    # out_dir = f'{args.out_dir}/diffdim/{init_method}/R_{num_repeat}-S_{true_single_cluster_size}'
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for n_centroids in range(3, 9, 9):
        # True labels
        true_labels = np.concatenate([np.ones(true_single_cluster_size) * i for i in range(n_centroids)]).astype(int)

        # dim = 50
        props = [0.0, 0.2, 0.4, 0.6, 0.8]
        prop_results = []
        for prop in tqdm(props):
            ith_prop_results = []
            for i in range(n_repetitions):
                # random seed
                seed = i
                rng = np.random.RandomState(seed=seed)

                radius = 0
                sigma = args.cluster_std
                # data_name = 'iot_intrusion' #'pen_digits' # 'biocoin_heist' #'letter_recognition'
                data = gen_data(data_name, fake_label, n_centroids, true_single_cluster_size, prop, add_outlier,
                                random_state=i)
                true_points = data['X']
                true_labels = data['Y']
                true_centroids = data['centroids']
                outliers = data['outliers']

                # Final points
                if add_outlier:
                    points = np.concatenate((true_points, outliers), axis=0)
                else:
                    # Without outliers
                    points = true_points

                # normalize the data
                if True:
                    # only normalize inliers to mean=0 and std=1
                    dim = points.shape[1]
                    mu = np.mean(points, axis=0)
                    scale = np.std(points, axis=0)
                    for j in range(dim):
                        points[:, j] = (points[:, j] - mu[j])
                        if scale[j] == 0: continue
                        points[:, j] = points[:, j] / scale[j]

                    # recompute the centroids after standardization.
                    j = 0
                    for idx in range(0, len(points), true_single_cluster_size):
                        if j >= n_centroids: break
                        true_centroids[j] = np.mean(points[idx:idx + true_single_cluster_size], axis=0)
                        j += 1

                # plot_xy(points, np.concatenate([true_labels, [max(true_labels)+1] * len(outliers)]),
                #         random_state=i, true_centroids= copy.deepcopy(centroids),
                #         title=f'prop: {prop} after std')

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
