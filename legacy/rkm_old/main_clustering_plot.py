"""
    python3 main_clustering.py > log.txt 2>&1 &
"""
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

tot_iterate = 50
np.random.seed(42)
tolerance = 1e-4

out_dir='out_plot'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def plot_res(df, xs, x_label, title='', num_centroids=5, radius=3):
    plt.clf()

    ######### misclustering proportion
    # Plot the line plot with error bars
    fig, ax = plt.subplots(figsize=(8, 6))

    kmed_misc_avg = df['kmedians misc']
    kmed_misc_err = df['kmedians misc err_bar']
    plt.plot(xs, kmed_misc_avg, '-.', label='k-median', color="green")
    plt.errorbar(xs, kmed_misc_avg, yerr=kmed_misc_err, fmt='none', ecolor='black', capsize=3)

    kmedL1_misc_avg = df['kmedians-L1 misc']
    kmedL1_misc_err = df['kmedians-L1 misc err_bar']
    plt.plot(xs, kmedL1_misc_avg, '--', label='k-medianL1', color="purple")
    plt.errorbar(xs, kmedL1_misc_avg, yerr=kmedL1_misc_err, fmt='none', ecolor='black', capsize=3)

    kmeans_misc_avg = df['kmeans misc']
    kmeans_misc_err = df['kmeans missc err_bar']
    plt.plot(xs, kmeans_misc_avg, '-', label='k-means', color="blue")
    plt.errorbar(xs, kmeans_misc_avg, yerr=kmeans_misc_err, fmt='none', ecolor='black', capsize=3)

    # plt.ylim(0,0.5)
    ax.set_xticks(xs)

    plt.xlabel(x_label)
    plt.ylabel("Misclustering Proportion")
    plt.title("Plot of misclustering proportion for %g clusters" % num_centroids)

    # Add a legend and show the plot
    plt.legend()

    # save the figure
    f = f'{out_dir}/{x_label}_%g_clusters_rad_%g_out_%g_misc.png' % (num_centroids, radius, prop)
    print(f)
    plt.savefig(f, dpi=300)

    plt.show(block=False)
    # plt.pause(2)
    plt.close()


    ######### ACD
    # Plot the line plot with error bars
    fig, ax = plt.subplots(figsize=(8, 6))

    kmed_acd_avg = df['kmedians acd']
    kmed_acd_err = df['kmedians acd err_bar']
    plt.plot(xs, kmed_acd_avg, '-.', label='k-median', color="green")
    plt.errorbar(xs, kmed_acd_avg, yerr=kmed_acd_err, fmt='none', ecolor='black', capsize=3)

    kmedL1_acd_avg = df['kmedians-L1 acd']
    kmedL1_acd_err = df['kmedians-L1 acd err_bar']
    plt.plot(xs, kmedL1_acd_avg, '--', label='k-medianL1', color="purple")
    plt.errorbar(xs, kmedL1_acd_avg, yerr=kmedL1_acd_err, fmt='none', ecolor='black', capsize=3)

    kmeans_acd_avg = df['kmeans acd']
    kmeans_acd_err = df['kmeans acd err_bar']
    plt.plot(xs, kmeans_acd_avg, '-', label='k-means', color="blue")
    plt.errorbar(xs, kmeans_acd_avg, yerr=kmeans_acd_err, fmt='none', ecolor='black', capsize=3)

    # plt.ylim(0, max(np.array(kmeans_acd_avg,kmed_acd_avg,kmedL1_acd_avg))+0.3)
    ax.set_xticks(xs)

    plt.xlabel(x_label)
    plt.ylabel("ACD")
    plt.title("Plot of acd for %g clusters" % num_centroids)

    # Add a legend and show the plot
    plt.legend()

    # save the figure
    f=f'{out_dir}/{x_label}_%g_clusters_rad_%g_out_%g_acd.png' % (num_centroids, radius, prop)
    print(f)
    plt.savefig(f, dpi=300)
    plt.show(block=False)
    # plt.pause(2)
    plt.close()


if __name__ == '__main__':

    init_method = 'omniscient'
    R = 50
    in_dir = 'out'

    # # 1. X-axis: varies dimensions (fixed K and noise_cov)
    # print('*** 1. varies dimensions')
    # K=5
    # r=3
    # noise_mean=1
    # noise_cov=25
    # prop=0.2
    # _out_file = f'{in_dir}/R_{R}-K_{K}-r_{r}-Nmean_{noise_mean}-Ncov_{noise_cov}-p_{prop}/{init_method}/data_{K}_clusters.csv'
    # df = pd.read_csv(_out_file)
    # plot_res(df, xs=range(5, 30+1, 5), x_label='Dimension', num_centroids=K)

    # # 2. X-axis: varies noise covariances (fixed K and noise_cov)
    # print('\n*** 2. varies noise covariances')
    # K=5
    # r=3
    # prop= 0.4
    # noise_mean= 0
    # dim = 3
    # noise_covs = [i ** 2 for i in range(1, 6, 1)]
    # for i, noise_cov in enumerate(noise_covs):
    #     _out_file = f'{in_dir}/R_{R}-K_{K}-r_{r}-Nmean_{noise_mean}-Ncov_{noise_cov}-p_{prop}/{init_method}/data_{K}_clusters.csv'
    #     if i==0:
    #         df = pd.read_csv(_out_file)
    #         df = df[df['dimensions']==dim]
    #     else:
    #         _df = pd.read_csv(_out_file)
    #         _df = _df[_df['dimensions'] == dim]
    #         df = pd.concat([df, _df], axis=0)
    # plot_res(df, xs = noise_covs, x_label='Noise Covariance', num_centroids=K)

    # 3. X-axis: varies noise mean/location (fixed K and noise_cov)
    print('\n*** 3. varies noise means')
    K = 5
    r = 3
    noise_means = [i ** 2 for i in range(1, 6, 1)]
    noise_cov= 25
    prop = 0.4
    dim = 3
    for i, noise_mean in enumerate(noise_means) :
        _out_file = f'{in_dir}/R_{R}-K_{K}-r_{r}-Nmean_{noise_mean}-Ncov_{noise_cov}-p_{prop}/{init_method}/data_{K}_clusters.csv'
        if i==0:
            df = pd.read_csv(_out_file)
            df = df[df['dimensions'] == dim]
        else:
            _df = pd.read_csv(_out_file)
            _df = _df[_df['dimensions'] == dim]
            df = pd.concat([df, _df], axis=0)
    plot_res(df, xs=noise_means, x_label='Noise Location', num_centroids=K)

    # # 4. X-axis: varies noise percent (fixed K and noise_cov)
    # print('\n*** 4. varies noise percents')
    # K = 5
    # r = 3
    # dim = 10
    # noise_mean = 0
    # noise_cov= 25
    # # props = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    # props = [0.01, 0.05, 0.1, 0.15, 0.2]
    # for i, prop in enumerate(props):
    #     _out_file = f'{in_dir}/R_{R}-K_{K}-r_{r}-Nmean_{noise_mean}-Ncov_{noise_cov}-p_{prop}/{init_method}/data_{K}_clusters.csv'
    #     if i==0:
    #         df = pd.read_csv(_out_file)
    #         df = df[df['dimensions'] == dim]
    #     else:
    #         _df = pd.read_csv(_out_file)
    #         _df = _df[_df['dimensions'] == dim]
    #         df = pd.concat([df, _df], axis=0)
    # plot_res(df, xs=props, x_label='Percent of Noise', num_centroids=K)

    # # 5. X-axis: varies radius  (fixed K and noise_cov)
    # print('\n*** 4. varies radius')
    # K = 5
    # dim = 10
    # noise_mean = 10
    # noise_cov = 25
    # radiuses = [1, 2, 3, 4, 5, 6]
    # prop = 0.2
    # for i, r in enumerate(radiuses):
    #     _out_file = f'{in_dir}/R_{R}-K_{K}-r_{r}-Nmean_{noise_mean}-Ncov_{noise_cov}-p_{prop}/{init_method}/data_{K}_clusters.csv'
    #     if i == 0:
    #         df = pd.read_csv(_out_file)
    #         df = df[df['dimensions'] == dim]
    #     else:
    #         _df = pd.read_csv(_out_file)
    #         _df = _df[_df['dimensions'] == dim]
    #         df = pd.concat([df, _df], axis=0)
    # plot_res(df, xs=radiuses, x_label='Radius', num_centroids=K)