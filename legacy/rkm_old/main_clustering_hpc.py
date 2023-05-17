"""
    python3 main_clustering.py > log.txt 2>&1 &
"""
import argparse
import datetime
import os
import time
import traceback

import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import pandas as pd
import copy
from node_info import *

tot_iterate = 50
np.random.seed(42)
tolerance = 1e-4
OUT_DIR = 'out'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


def kmeans(points, k, centroids_input, max_iterations=tot_iterate):
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

    # print(f'kmeans iterations: {i}')
    return new_centroids, labels


def kmed(points, k, centroids_input, max_iterations=tot_iterate):
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
    # print(f'kmed iterations: {i}')
    return new_centroids, labels


def kmedL1(points, k, centroids_input, max_iterations=tot_iterate):
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
    # print(f'kmedL1 iterations: {i}')
    return new_centroids, labels


def main(num_repeat=5, radius=3, noise_mean=0, noise_cov=5, prop=0.80, initial_method='omnisicent'):
    true_cluster_size = 100

    for num_centroids in tqdm(range(5, 30 + 1, 5)):
        # True labels

        true_labels = np.concatenate([np.ones(true_cluster_size) * i for i in range(num_centroids)]).astype(int)

        tot_dims = range(5, 30 + 1, 5)
        # tot_dims = [50, 100, 500, 750, 1000]

        # temp = time.time()

        kmed_misc_avg = []
        kmed_misc_err = []
        kmedL1_misc_avg = []
        kmedL1_misc_err = []
        kmeans_misc_avg = []
        kmeans_misc_err = []

        # acd variables

        kmed_acd_avg = []
        kmed_acd_err = []
        kmedL1_acd_avg = []
        kmedL1_acd_err = []
        kmeans_acd_avg = []
        kmeans_acd_err = []

        for dim in tot_dims:

            kmed_misc = []
            kmedL1_misc = []
            kmeans_misc = []

            kmed_acd = []
            kmedL1_acd = []
            kmeans_acd = []

            for i in range(num_repeat):
                centroids = np.random.normal(size=(num_centroids, dim))
                centroids /= np.linalg.norm(centroids, axis=1)[:, np.newaxis]
                centroids *= radius

                # Set means and covariance matrices
                cov = np.identity(dim)

                true_points = np.concatenate(
                    [np.random.multivariate_normal(mean, cov, size=true_cluster_size) for mean in centroids])

                if initial_method == 'random':
                    indices = np.random.choice(range(len(true_points)), size=num_centroids, replace=False)
                    centroids = true_points[indices, :]
                else:
                    pass

                # Fraction of outliers

                # prop = 0.8

                # error_std = 4

                # outliers = error_std * np.random.multivariate_normal(np.zeros(dim), np.eye(dim),
                #                                                      size=math.floor(true_cluster_size * prop))

                outliers = np.random.multivariate_normal(np.ones(dim) * noise_mean, np.eye(dim) * noise_cov,
                                                         size=math.floor(true_cluster_size * num_centroids * prop))

                # Final points

                points = np.concatenate((true_points, outliers), axis=0)
                # print(f"K:{num_centroids}, dim:{dim}, points:{points.shape}, {initial_method}", flush=True)
                ## Without outliers
                # points = true_points

                # Perform k-means clustering with k clusters
                kmed_centroids, kmed_labels = kmed(points, centroids_input=copy.deepcopy(centroids), k=num_centroids)
                kmedL1_centroids, kmedL1_labels = kmedL1(points, centroids_input=copy.deepcopy(centroids),
                                                         k=num_centroids)
                kmeans_centroids, kmeans_labels = kmeans(points, centroids_input=copy.deepcopy(centroids),
                                                         k=num_centroids)

                # print(kmed_labels)
                #
                # print(true_labels)

                # acd computations

                kmed_acd.append(np.sum((kmed_centroids - centroids) ** 2) / num_centroids)
                kmedL1_acd.append(np.sum((kmedL1_centroids - centroids) ** 2) / num_centroids)
                kmeans_acd.append(np.sum((kmeans_centroids - centroids) ** 2) / num_centroids)

                # Misclustering label estimation

                kmed_misc.append(
                    sum(kmed_labels[range(num_centroids * true_cluster_size)] != true_labels) / len(true_labels))

                kmedL1_misc.append(
                    sum(kmedL1_labels[range(num_centroids * true_cluster_size)] != true_labels) / len(true_labels))

                kmeans_misc.append(
                    sum(kmeans_labels[range(num_centroids * true_cluster_size)] != true_labels) / len(true_labels))

            # acd average and error bar

            kmed_acd_avg.append(np.mean(kmed_acd))
            kmed_acd_err.append(1.96 * np.std(kmed_acd) / np.sqrt(len(kmed_acd)))

            kmedL1_acd_avg.append(np.mean(kmedL1_acd))
            kmedL1_acd_err.append(1.96 * np.std(kmedL1_acd) / np.sqrt(len(kmedL1_acd)))

            kmeans_acd_avg.append(np.mean(kmeans_acd))
            kmeans_acd_err.append(1.96 * np.std(kmeans_acd) / np.sqrt(len(kmeans_acd)))

            # Misclustering proportion avg and error bar

            kmed_misc_avg.append(np.mean(kmed_misc))
            kmed_misc_err.append(1.96 * np.std(kmed_misc) / np.sqrt(len(kmed_misc)))

            kmedL1_misc_avg.append(np.mean(kmedL1_misc))
            kmedL1_misc_err.append(1.96 * np.std(kmedL1_misc) / np.sqrt(len(kmedL1_misc)))

            kmeans_misc_avg.append(np.mean(kmeans_misc))
            kmeans_misc_err.append(1.96 * np.std(kmeans_misc) / np.sqrt(len(kmeans_misc)))

        # create data frame with the misclustering
        out_dir = f'{OUT_DIR}/R_{num_repeat}-K_{num_centroids}-r_{radius}-Nmean_{noise_mean}-Ncov_{noise_cov}-p_{prop}/{initial_method}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        data = {'dimensions': tot_dims, 'kmedians misc': kmed_misc_avg, 'kmedians misc err_bar': kmed_misc_err,
                'kmedians-L1 misc': kmedL1_misc_avg, 'kmedians-L1 misc err_bar': kmedL1_misc_err,
                'kmeans misc': kmeans_misc_avg, 'kmeans missc err_bar': kmeans_misc_err,
                'kmedians acd': kmed_acd_avg, 'kmedians acd err_bar': kmed_acd_err,
                'kmedians-L1 acd': kmedL1_acd_avg, 'kmedians-L1 acd err_bar': kmedL1_acd_err,
                'kmeans acd': kmeans_acd_avg, 'kmeans acd err_bar': kmeans_acd_err
                }
        df = pd.DataFrame(data)

        # Save data to CSV file
        df.to_csv(f'{out_dir}/data_%g_clusters.csv' % num_centroids, index=False)

        # Plot the line plot with error bars

        fig, ax = plt.subplots(figsize=(8, 6))

        plt.plot(tot_dims, kmed_misc_avg, '-.', label='k-median', color="green")
        plt.errorbar(tot_dims, kmed_misc_avg, yerr=kmed_misc_err, fmt='none', ecolor='black', capsize=3)

        plt.plot(tot_dims, kmedL1_misc_avg, '--', label='k-medianL1', color="purple")
        plt.errorbar(tot_dims, kmedL1_misc_avg, yerr=kmedL1_misc_err, fmt='none', ecolor='black', capsize=3)

        plt.plot(tot_dims, kmeans_misc_avg, '-', label='k-means', color="blue")
        plt.errorbar(tot_dims, kmeans_misc_avg, yerr=kmeans_misc_err, fmt='none', ecolor='black', capsize=3)

        # plt.ylim(0,0.5)
        ax.set_xticks(tot_dims)

        plt.xlabel("Dimensions")
        plt.ylabel("Misclustering proportion")
        plt.title("Plot of misclustering proportion for %g clusters" % num_centroids)

        # Add a legend and show the plot
        plt.legend()

        # save the figure
        plt.savefig(f'{out_dir}/plot_%g_clusters_rad_%g_out_%g_misc.png' % (num_centroids, radius, prop), dpi=300)

        # plt.show(block=False)
        # plt.pause(2)
        plt.close()

        # Plot the line plot with error bars

        fig, ax = plt.subplots(figsize=(8, 6))

        plt.plot(tot_dims, kmed_acd_avg, '-.', label='k-median', color="green")
        plt.errorbar(tot_dims, kmed_acd_avg, yerr=kmed_acd_err, fmt='none', ecolor='black', capsize=3)

        plt.plot(tot_dims, kmedL1_acd_avg, '--', label='k-medianL1', color="purple")
        plt.errorbar(tot_dims, kmedL1_acd_avg, yerr=kmedL1_acd_err, fmt='none', ecolor='black', capsize=3)

        plt.plot(tot_dims, kmeans_acd_avg, '-', label='k-means', color="blue")
        plt.errorbar(tot_dims, kmeans_acd_avg, yerr=kmeans_acd_err, fmt='none', ecolor='black', capsize=3)

        # plt.ylim(0, max(np.array(kmeans_acd_avg,kmed_acd_avg,kmedL1_acd_avg))+0.3)
        ax.set_xticks(tot_dims)

        plt.xlabel("Dimensions")
        plt.ylabel("acd")
        plt.title("Plot of acd for %g clusters" % num_centroids)

        # Add a legend and show the plot
        plt.legend()

        # save the figure

        plt.savefig(f'{out_dir}/plot_%g_clusters_rad_%g_out_%g_acd.png' % (num_centroids, radius, prop), dpi=300)

        # plt.show(block=False)
        # plt.pause(2)
        plt.close()

#
# def main_call(args):
#     try:
#         num_repeat, radius, noise_mean, noise_cov, prop, initial_method = args[0], args[1], args[2], args[3], args[4], \
#         args[5]
#         print(
#             f'num_repeat: {num_repeat}, radius: {radius}, noise_mean: {noise_mean}, noise_cov: {noise_cov}, prop: {prop}, '
#             f'initial_method: {initial_method}', flush=True)
#         main(num_repeat, radius, noise_mean, noise_cov, prop, initial_method)
#     except Exception as e:
#         traceback.print_exc()
#

parser = argparse.ArgumentParser()
parser.add_argument('--force', default=False,   # whether overwrite the previous results or not?
                    action='store_true', help='force')
parser.add_argument("--n_repeats", type=int, default=50)  # -1 for debugging
parser.add_argument("--radius", type=int, default=1)  # -1 for debugging
parser.add_argument("--nmean", type=int, default=0)
parser.add_argument("--ncov", type=int, default=25)
parser.add_argument("--prop", type=float, default=0.01)
parser.add_argument("-d", type=int, default=2)
parser.add_argument("--init_method", type=str, default='omniscient')
args = parser.parse_args()


if __name__ == '__main__':
    print(datetime.datetime.now())
    st = time.time()

    # is_debug = False
    # if is_debug:
    #     radiuses = [3]  # [1, 2, 3, 4, 5, 6]
    #     noise_covs = [25]  # [1, 4, 9, 16, 25]  #[25]
    #     noise_means = [0]  # [1, 4, 9, 16, 25]
    #     props = [0.01]  # [0.01, 0.05, 0.1, 0.15, 0.2]
    # else:
    #     radiuses = [1, 2, 3, 4, 5, 6]
    #     noise_covs = [1, 4, 9, 16, 25]  # [25]
    #     noise_means = [1, 4, 9, 16, 25]
    #     props = [0.01, 0.05, 0.1, 0.15, 0.2]  # [0.1, 0.5, 1, 2, 3]
    n_repeats = args.n_repeats
    radius = args.radius
    noise_mean = args.nmean
    noise_cov = args.ncov
    prop = args.prop
    initial_method = args.init_method

    print(args)
    main(n_repeats, radius, noise_mean, noise_cov, prop, initial_method)

    ed = time.time()
    print(datetime.datetime.now())
    print(f'Total time: {(ed - st) / 3600} hours.')
