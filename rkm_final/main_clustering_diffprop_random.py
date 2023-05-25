import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import pandas as pd
import copy


import os
from clustering_random import *


parser = argparse.ArgumentParser()
# parser.add_argument('--force', default=False,   # whether overwrite the previous results or not?
#                     action='store_true', help='force')
parser.add_argument("--n_repeats", type=int, default=2)  #
parser.add_argument("--true_cluster_size", type=int, default=100)
parser.add_argument("--init_method", type=str, default='random')
parser.add_argument("--with_outlier", type=str, default='True')
parser.add_argument("--out_dir", type=str, default='out')
args = parser.parse_args()
args.with_outlier = False if args.with_outlier == 'False' else True
print(args)

# num_repeat = 400
num_repeat = args.n_repeats
init_method = args.init_method
true_cluster_size = args.true_cluster_size
with_outlier=args.with_outlier
# out_dir = f'{args.out_dir}/diffdim/{init_method}/R_{num_repeat}-S_{true_cluster_size}'
out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


for num_centroids in range(4,9,9):
    # True labels

    true_labels = np.concatenate([np.ones(true_cluster_size)*i for i in range(num_centroids)]).astype(int)

    dim = 10
    props = [0, 0.2, 0.4, 0.6, 0.8]



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

    for prop in tqdm(props):

        lloydL1_misc = []
        kmed_misc = []
        kmeans_misc = []

        lloydL1_acd = []
        kmed_acd = []
        kmeans_acd = []

        for i in range(num_repeat):
            rng = np.random.RandomState(seed=i)
            centroids = rng.normal(size=(num_centroids, dim))
            centroids /= np.linalg.norm(centroids, axis=1)[:, np.newaxis]

            # centroids /= max(np.linalg.norm(centroids, axis=1)[:, np.newaxis])

            radius = 5

            sigma = 2

            centroids *= radius

            # Set means and covariance matrices
            cov = np.identity(dim)

            true_points = np.concatenate(
                [rng.multivariate_normal(mean, cov * (sigma**2), size=true_cluster_size) for mean in centroids])

            if init_method == 'random':
                indices = rng.choice(range(len(true_points)), size=num_centroids, replace=False)
                init_centroids = true_points[indices, :]
                # init_centroids = np.copy(centroids)
            else:
                pass


            # Fraction of outliers
            # prop = 0.60

            outlier_std = 10

            outliers = rng.multivariate_normal(np.ones(dim)*0,
                                           np.eye(dim)*outlier_std**2, size = math.floor(true_cluster_size * prop))


            # Final points
            if with_outlier:
                points = np.concatenate((true_points, outliers), axis=0)
            else:
                # Without outliers
                points = true_points

            # Perform k-means clustering with k clusters
            lloydL1_centroids, lloydL1_labels = lloydL1(points,centroids_input=init_centroids,k=num_centroids, true_centroids=centroids)
            kmed_centroids, kmed_labels = kmed(points,centroids_input=init_centroids,k=num_centroids,  true_centroids=centroids)
            kmeans_centroids, kmeans_labels = kmeans(points,centroids_input=init_centroids,k=num_centroids,  true_centroids=centroids)

            # print(lloydL1_labels)
            #
            # print(true_labels)

            # acd computations


            lloydL1_acd.append(np.sum((lloydL1_centroids-centroids)**2)/num_centroids)
            kmed_acd.append(np.sum((kmed_centroids - centroids) ** 2) / num_centroids)
            kmeans_acd.append(np.sum((kmeans_centroids - centroids) ** 2) / num_centroids)

            # Misclustering label estimation

            lloydL1_misc.append(sum(lloydL1_labels[range(num_centroids*true_cluster_size)]!=true_labels)/len(true_labels))

            kmed_misc.append(sum(kmed_labels[range(num_centroids*true_cluster_size)]!=true_labels)/len(true_labels))

            kmeans_misc.append(sum(kmeans_labels[range(num_centroids*true_cluster_size)]!=true_labels)/len(true_labels))

        # acd average and error bar

        lloydL1_acd_avg.append(np.mean(lloydL1_acd))
        lloydL1_acd_err.append(1.96 * np.std(lloydL1_acd) / np.sqrt(len(lloydL1_acd)))

        kmed_acd_avg.append(np.mean(kmed_acd))
        kmed_acd_err.append(1.96 * np.std(kmed_acd) / np.sqrt(len(kmed_acd)))

        kmeans_acd_avg.append(np.mean(kmeans_acd))
        kmeans_acd_err.append(1.96 * np.std(kmeans_acd) / np.sqrt(len(kmeans_acd)))


        # Misclustering proportion avg and error bar

        lloydL1_misc_avg.append(np.mean(lloydL1_misc))
        lloydL1_misc_err.append(1.96*np.std(lloydL1_misc)/np.sqrt(len(lloydL1_misc)))

        kmed_misc_avg.append(np.mean(kmed_misc))
        kmed_misc_err.append(1.96*np.std(kmed_misc)/np.sqrt(len(kmed_misc)))

        kmeans_misc_avg.append(np.mean(kmeans_misc))
        kmeans_misc_err.append(1.96*np.std(kmeans_misc)/np.sqrt(len(kmeans_misc)))


    # create data frame with the misclustering

    data = {'props':props,'lloydL1ians misc':lloydL1_misc_avg,'lloydL1ians misc err_bar':lloydL1_misc_err,
            'lloydL1ians-L1 misc':kmed_misc_avg,'lloydL1ians-L1 misc err_bar':kmed_misc_err,
            'kmeans misc':kmeans_misc_avg,'kmeans missc err_bar':kmeans_misc_err,
            'lloydL1ians acd': lloydL1_acd_avg, 'lloydL1ians acd err_bar': lloydL1_acd_err,
            'lloydL1ians-L1 acd': kmed_acd_avg, 'lloydL1ians-L1 acd err_bar': kmed_acd_err,
            'kmeans acd': kmeans_acd_avg, 'kmeans acd err_bar': kmeans_acd_err
            }
    df = pd.DataFrame(data)

    # Save data to CSV file
    df.to_csv(f'{out_dir}/data_%g_clusters.csv'%num_centroids, index=False)


    # Plot the line plot with error bars

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.plot(props, lloydL1_misc_avg, '-.', label='Lloyd-$L_1$',color="green")
    plt.errorbar(props, lloydL1_misc_avg, yerr=lloydL1_misc_err, fmt='none', ecolor='black', capsize=3)


    plt.plot(props, kmed_misc_avg, '--', label='k-median',color="purple")
    plt.errorbar(props, kmed_misc_avg, yerr=kmed_misc_err, fmt='none', ecolor='black', capsize=3)


    plt.plot(props, kmeans_misc_avg, '-', label='Llyod (k-means)',color="blue")
    plt.errorbar(props, kmeans_misc_avg, yerr=kmeans_misc_err, fmt='none', ecolor='black', capsize=3)


    # plt.ylim(0,0.5)
    ax.set_xticks(props)

    plt.xlabel("Noise Proportion")
    plt.ylabel("Misclustering proportion")
    plt.title("Plot of misc_prop: %g_clusters_rad_%g_out_%g_sigma_%g" % (num_centroids,radius,prop,sigma))


    # Add a legend and show the plot
    plt.legend()

    #save the figure

    temp = time.time()
    temp = 0
    plt.savefig(f'{out_dir}/misc_prop_%g_clusters_%f.png'%(num_centroids,temp), dpi=300)

    plt.show(block=False)
    plt.pause(2)

    # Plot the line plot with error bars

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.plot(props, lloydL1_acd_avg, '-.', label='Lloyd-$L_1$', color="green")
    plt.errorbar(props, lloydL1_acd_avg, yerr=lloydL1_acd_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(props, kmed_acd_avg, '--', label='k-median', color="purple")
    plt.errorbar(props, kmed_acd_avg, yerr=kmed_acd_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(props, kmeans_acd_avg, '-', label='Lloyd ($k$-means)', color="blue")
    plt.errorbar(props, kmeans_acd_avg, yerr=kmeans_acd_err, fmt='none', ecolor='black', capsize=3)

    # plt.ylim(0, max(np.array(kmeans_acd_avg,lloydL1_acd_avg,kmed_acd_avg))+0.3)
    ax.set_xticks(props)

    plt.xlabel("Noise Proportion")
    plt.ylabel("acd")
    plt.title("Plot of acd: %g_clusters_rad_%g_out_%g_sigma_%g" % (num_centroids,radius,prop,sigma))

    # Add a legend and show the plot
    plt.legend()

    # save the figure


    plt.savefig(f'{out_dir}/acd_%g_clusters_%f.png' % (num_centroids,temp), dpi=300)

    plt.show(block=False)
    plt.pause(2)
    plt.close()

