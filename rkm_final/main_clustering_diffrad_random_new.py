import argparse
import pickle
import shutil
import time
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import pandas as pd
import copy


def remove_outliers(lst, q):  # q is trimming percent
    arr = np.array(lst)
    # lower_percentile = np.percentile(arr, q)
    lower_percentile = 0
    upper_percentile = np.percentile(arr, 100 - q)

    filtered_lst = [x for x in lst if lower_percentile <= x <= upper_percentile]

    return filtered_lst


import os
# from clustering_random import *

import collections

import numpy as np
import copy
import itertools
import matplotlib.pyplot as plt


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


def find_minimal_mp(centroids, true_centroids, points, num_centroids, true_cluster_size, true_labels,
                    method='kmeans'):
    # print(f'centroids before: {centroids}')
    # print(f"{len(centroids)} centroids include {len(list(itertools.permutations(centroids)))} permutations")
    # c1 = copy.deepcopy(true_centroids)
    # check which point is close to which true centroids.
    min_mp = np.inf
    indices = range(len(centroids))
    for _indices in list(itertools.permutations(indices)):
        new_centroids = centroids[_indices, :]
        # d = np.sum(np.sum(np.square(c - c1), axis=1), axis=0) # aligned by centroids
        # aligned by MP
        if method == 'kmeans':
            # L2
            distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
        elif method == 'kmed':
            # L1
            distances = np.sum(np.abs(points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
        elif method == 'lloydL1':
            # L2
            distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
        else:
            raise ValueError(method)

        labels = np.argmin(distances, axis=1)
        mp = sum(labels[range(num_centroids * true_cluster_size)] != true_labels) / len(true_labels)
        if mp < min_mp:
            # print(method, d, min_d)
            min_mp = mp  # here is just a float, so there is no need to copy()
            best_centroids = np.asarray(copy.deepcopy(new_centroids))
            best_labels = np.asarray(copy.deepcopy(labels))
            # print(method, d, min_d, best_centroids)
    # print(f'centroids after: {best_centroids}')

    return best_centroids, best_labels, min_mp


tot_iterate = 50
sub_iterate = 4
tolerance = 1e-4


def kmeans(points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None, true_labels=None):
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
    # L2
    distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    labels = np.argmin(distances, axis=1)

    return new_centroids, labels


def kmed(points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None, true_labels=None):
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

    # note that it should be L1, not L2
    # distances2 = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    distances = np.sum(np.abs(points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]), axis=2)
    labels = np.argmin(distances, axis=1)

    return new_centroids, labels


def lloydL1(points, k, centroids_input, max_iterations=tot_iterate, true_centroids=None, true_labels=None):
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

    # Here should be L2
    distances = np.sqrt(np.sum((points[:, np.newaxis, :] - new_centroids[np.newaxis, :, :]) ** 2, axis=2))
    labels = np.argmin(distances, axis=1)

    return new_centroids, labels


parser = argparse.ArgumentParser()
# parser.add_argument('--force', default=False,   # whether overwrite the previous results or not?
#                     action='store_true', help='force')
parser.add_argument("--n_repeats", type=int, default=2)  #
parser.add_argument("--true_cluster_size", type=int, default=100)
parser.add_argument("--init_method", type=str, default='random')
parser.add_argument("--with_outlier", type=str, default='True')
parser.add_argument("--out_dir", type=str, default='out')
parser.add_argument("--std", type=float, default=1)
args = parser.parse_args()
args.with_outlier = False if args.with_outlier == 'False' else True
print(args)

num_repeat = args.n_repeats
init_method = args.init_method
true_cluster_size = args.true_cluster_size
with_outlier = args.with_outlier

save_steps = max(1, num_repeat//5)
# out_dir = f'{args.out_dir}/diffdim/{init_method}/R_{num_repeat}-S_{true_cluster_size}'
# out_dir = os.path.join(args.out_dir, f'std_{args.std}')
out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

dim = 10
rad_out_vec = np.trunc(np.linspace(0, 100, 11))

for num_centroids in range(4, 10, 6):
    # True labels

    true_labels = np.concatenate([np.ones(true_cluster_size) * i for i in range(num_centroids)]).astype(int)

    # tot_dims = range(2,15,2)

    lloydL1_misc_avg = []
    lloydL1_misc_err = []
    kmed_misc_avg = []
    kmed_misc_err = []
    kmeans_misc_avg = []
    kmeans_misc_err = []

    # acd variables

    lloydL1_acd_avg = []
    lloydL1_acd_err = []
    kmed_acd_avg = []
    kmed_acd_err = []
    kmeans_acd_avg = []
    kmeans_acd_err = []

    for rad_out in tqdm(rad_out_vec):

        lloydL1_misc = []
        kmed_misc = []
        kmeans_misc = []

        lloydL1_acd = []
        kmed_acd = []
        kmeans_acd = []

        # save centroids
        true_centroids_list = []
        initial_centroids_list = []
        # estimated_centroids_list = []
        lloydL1_estc = []
        kmed_estc = []
        kmeans_estc = []

        lloydL1_estc_mp = []
        kmed_estc_mp = []
        kmeans_estc_mp = []

        for i in range(num_repeat):

            rng = np.random.RandomState(seed=i)
            centroids = rng.normal(size=(num_centroids, dim))
            centroids /= np.linalg.norm(centroids, axis=1)[:, np.newaxis]

            # centroids /= max(np.linalg.norm(centroids, axis=1)[:, np.newaxis])

            radius = 5

            sigma = args.std
            # sigma = 2
            # sigma = 0.5
            # sigma = 0.25
            # sigma = 0.1

            centroids *= radius

            # Set means and covariance matrices
            true_points = np.concatenate(
                [rng.multivariate_normal(mean, np.identity(dim) * sigma ** 2, size=true_cluster_size) for mean in
                 centroids])

            if init_method == 'random':
                indices = rng.choice(range(len(true_points)), size=num_centroids, replace=False)
                init_centroids = true_points[indices, :]
            else:
                pass

            # Fraction of outliers

            prop = 0.6

            sigma_out = 10

            # outliers = rad_out/np.sqrt(dim) + sigma_out * rng.multivariate_normal(np.zeros(dim), np.eye(dim),
            #                                                      size = math.floor(true_cluster_size * prop))

            centroids_out_dir = rng.multivariate_normal(np.zeros(dim), np.eye(dim), size=1)

            centroids_out_dir /= np.linalg.norm(centroids_out_dir, axis=1)[:, np.newaxis]

            outlier_mean = rad_out / np.sqrt(dim) * centroids_out_dir[0]

            outliers = outlier_mean + rng.multivariate_normal(np.zeros(dim), np.eye(dim) * sigma_out ** 2,
                                                              size=math.floor(true_cluster_size * prop))

            # Final points
            if with_outlier:
                points = np.concatenate((true_points, outliers), axis=0)
            else:
                # Without outliers
                points = true_points

            # Perform k-means clustering with k clusters
            lloydL1_centroids, lloydL1_labels = lloydL1(points, centroids_input=init_centroids,
                                                        k=num_centroids, true_centroids=centroids)
            kmed_centroids, kmed_labels = kmed(points, centroids_input=init_centroids, k=num_centroids,
                                               true_centroids=centroids)
            kmeans_centroids, kmeans_labels = kmeans(points, centroids_input=init_centroids, k=num_centroids,
                                                     true_centroids=centroids)
            true_centroids_list.append(copy.deepcopy(centroids))
            initial_centroids_list.append(copy.deepcopy(init_centroids))
            lloydL1_estc.append(copy.deepcopy(lloydL1_centroids))
            kmed_estc.append(copy.deepcopy(kmed_centroids))
            kmeans_estc.append(copy.deepcopy(kmeans_centroids))

            # we should align the estimated centroids to true centroids by minimizing the MP
            lloydL1_centroids, lloydL1_labels, lloydL1_mp = find_minimal_mp(lloydL1_centroids, centroids, points,
                                                                            num_centroids,
                                                                            true_cluster_size, true_labels,
                                                                            method='lloydL1')
            kmed_centroids, kmed_labels, kmed_mp = find_minimal_mp(kmed_centroids, centroids, points, num_centroids,
                                                                   true_cluster_size, true_labels, method='kmed')
            kmeans_centroids, kmeans_labels, kmeans_mp = find_minimal_mp(kmeans_centroids, centroids, points,
                                                                         num_centroids,
                                                                         true_cluster_size, true_labels,
                                                                         method='kmeans')
            lloydL1_estc_mp.append(copy.deepcopy(lloydL1_centroids))
            kmed_estc_mp.append(copy.deepcopy(kmed_centroids))
            kmeans_estc_mp.append(copy.deepcopy(kmeans_centroids))

            # lloydL1_acd.append(np.median((lloydL1_centroids - centroids) ** 2))
            # kmed_acd.append(np.median((kmed_centroids - centroids) ** 2))
            # kmeans_acd.append(np.sum((kmeans_centroids - centroids) ** 2) / num_centroids)
            lloydL1_acd.append(np.sum((lloydL1_centroids - centroids) ** 2) / num_centroids)
            kmed_acd.append(np.sum((kmed_centroids - centroids) ** 2) / num_centroids)
            kmeans_acd.append(np.sum((kmeans_centroids - centroids) ** 2) / num_centroids)

            # Misclustering label estimation
            # all the normal data are the first k*100 points.
            lloydL1_misc.append(lloydL1_mp)
            kmed_misc.append(kmed_mp)
            kmeans_misc.append(kmeans_mp)


        # Save centroids to disk
        centroid_file = f'{out_dir}/seeds/K_{num_centroids}-R_{rad_out}.csv'
        tmp_dir = os.path.dirname(centroid_file)
        # if os.path.exists(tmp_dir):
        #     shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        centroids_data = {
            'true_centroids': true_centroids_list,
            'initial_centroids': initial_centroids_list,
            'lloydL1_estc': lloydL1_estc, 'lloydL1_estc_mp': lloydL1_estc_mp,
            'kmed_estc': kmed_estc_mp, 'kmed_estc_mp': kmed_estc_mp,
            'kmeans_estc': kmeans_estc_mp, 'kmeans_estc_mp': kmeans_estc_mp,
        }
        df = pd.DataFrame(centroids_data)
        # Save data to disk
        df.to_csv(centroid_file, index=False)

        # acd average and error bar
        # trimming top and bottom q % data top and bottom
        q = 10
        # lloydL1_acd_temp = remove_outliers(lloydL1_acd,q)
        lloydL1_acd_temp = lloydL1_acd.copy()
        lloydL1_acd_avg.append(np.mean(lloydL1_acd_temp))
        lloydL1_acd_err.append(1.96 * np.std(lloydL1_acd_temp) / np.sqrt(len(lloydL1_acd_temp)))

        # kmed_acd_temp = remove_outliers(kmed_acd,q)
        kmed_acd_temp = kmed_acd.copy()
        kmed_acd_avg.append(np.mean(kmed_acd_temp))
        kmed_acd_err.append(1.96 * np.std(kmed_acd_temp) / np.sqrt(len(kmed_acd_temp)))

        # kmeans_acd_temp = remove_outliers(kmeans_acd,q)
        kmeans_acd_temp = kmeans_acd.copy()
        kmeans_acd_avg.append(np.mean(kmeans_acd_temp))
        kmeans_acd_err.append(1.96 * np.std(kmeans_acd_temp) / np.sqrt(len(kmeans_acd_temp)))

        # Misclustering proportion avg and error bar
        # lloydL1_misc_temp = remove_outliers(lloydL1_misc,q)
        lloydL1_misc_temp = lloydL1_misc.copy()
        lloydL1_misc_avg.append(np.mean(lloydL1_misc_temp))
        lloydL1_misc_err.append(1.96 * np.std(lloydL1_misc_temp) / np.sqrt(len(lloydL1_misc_temp)))

        # kmed_misc_temp = remove_outliers(kmed_misc,q)
        kmed_misc_temp = kmed_misc.copy()
        kmed_misc_avg.append(np.mean(kmed_misc_temp))
        kmed_misc_err.append(1.96 * np.std(kmed_misc_temp) / np.sqrt(len(kmed_misc_temp)))

        # kmeans_misc_temp = remove_outliers(kmeans_misc,q)
        kmeans_misc_temp = kmeans_misc.copy()
        kmeans_misc_avg.append(np.mean(kmeans_misc_temp))
        kmeans_misc_err.append(1.96 * np.std(kmeans_misc_temp) / np.sqrt(len(kmeans_misc_temp)))

        # break

    # create data frame with the misclustering
    data = {
        'outlier rad': rad_out_vec,
        'lloydL1ians misc': lloydL1_misc_avg, 'lloydL1ians misc err_bar': lloydL1_misc_err,
        'lloydL1ians-L1 misc': kmed_misc_avg, 'lloydL1ians-L1 misc err_bar': kmed_misc_err,
        'kmeans misc': kmeans_misc_avg, 'kmeans misc err_bar': kmeans_misc_err,
        'lloydL1ians acd': lloydL1_acd_avg, 'lloydL1ians acd err_bar': lloydL1_acd_err,
        'lloydL1ians-L1 acd': kmed_acd_avg, 'lloydL1ians-L1 acd err_bar': kmed_acd_err,
        'kmeans acd': kmeans_acd_avg, 'kmeans acd err_bar': kmeans_acd_err,

    }
    df = pd.DataFrame(data)

    # Save data to CSV file
    df.to_csv(f'{out_dir}/data_%g_clusters.csv' % num_centroids, index=False)

    # Plot the line plot with error bars

    # break

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.plot(rad_out_vec, lloydL1_misc_avg, '-.', label='Lloyd-$L_1$', color="green")
    plt.errorbar(rad_out_vec, lloydL1_misc_avg, yerr=lloydL1_misc_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(rad_out_vec, kmed_misc_avg, '--', label='k-median', color="purple")
    plt.errorbar(rad_out_vec, kmed_misc_avg, yerr=kmed_misc_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(rad_out_vec, kmeans_misc_avg, '-', label='Llyod (k-means)', color="blue")
    plt.errorbar(rad_out_vec, kmeans_misc_avg, yerr=kmeans_misc_err, fmt='none', ecolor='black', capsize=3)

    # plt.ylim(0,0.5)
    ax.set_xticks(rad_out_vec)

    plt.xlabel("Outlier rad")
    plt.ylabel("Misclustering proportion")
    plt.title("Plot of misc_prop: %g_clusters_rad_%g_out_%g_dim_%g" % (num_centroids, radius, prop, dim))

    # Add a legend and show the plot
    plt.legend()

    # save the figure

    temp = time.time()
    temp = 0
    plt.savefig(f'{out_dir}/misc_prop_%g_clusters_%f.png' % (num_centroids, temp), dpi=300)

    plt.show(block=False)
    plt.pause(2)

    # Plot the line plot with error bars

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.plot(rad_out_vec, lloydL1_acd_avg, '-.', label='Lloyd-$L_1$', color="green")
    plt.errorbar(rad_out_vec, lloydL1_acd_avg, yerr=lloydL1_acd_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(rad_out_vec, kmed_acd_avg, '--', label='k-median', color="purple")
    plt.errorbar(rad_out_vec, kmed_acd_avg, yerr=kmed_acd_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(rad_out_vec, kmeans_acd_avg, '-', label='Lloyd ($k$-means)', color="blue")
    plt.errorbar(rad_out_vec, kmeans_acd_avg, yerr=kmeans_acd_err, fmt='none', ecolor='black', capsize=3)

    # plt.ylim(0, max(np.array(kmeans_acd_avg,lloydL1_acd_avg,kmed_acd_avg))+0.3)
    ax.set_xticks(rad_out_vec)

    plt.xlabel("Outlier rad")
    plt.ylabel("acd")
    plt.title("Plot of acd: %g_clusters_rad_%g_out_%g_dim_%g" % (num_centroids, radius, prop, dim))

    # Add a legend and show the plot
    plt.legend()

    # save the figure

    plt.savefig(f'{out_dir}/acd_%g_clusters_%f.png' % (num_centroids, temp), dpi=300)

    plt.show(block=False)
    plt.pause(2)
    plt.close()
