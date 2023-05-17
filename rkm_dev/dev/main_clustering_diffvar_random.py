import time

import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import pandas as pd
import copy
import os
from clustering import *


num_repeat = 1000

# num_centroids = 5

true_cluster_size = 200

dim = 10
sigma_out_vec = np.trunc(np.linspace(1,10,10))

init_method = 'random'
out_dir = f'out/diffvar/{init_method}/R_{num_repeat}'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for num_centroids in [5]: #range(5,9,4):
    # True labels

    true_labels = np.concatenate([np.ones(true_cluster_size)*i for i in range(num_centroids)]).astype(int)

    # tot_dims = range(2,15,2)




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

    for sigma_out in tqdm(sigma_out_vec):

        kmed_misc = []
        kmedL1_misc = []
        kmeans_misc = []

        kmed_acd = []
        kmedL1_acd = []
        kmeans_acd = []

        for i in range(num_repeat):
            r = np.random.RandomState(seed=i)
            centroids = r.normal(size=(num_centroids, dim))
            centroids /= np.linalg.norm(centroids, axis=1)[:, np.newaxis]

            # centroids /= max(np.linalg.norm(centroids, axis=1)[:, np.newaxis])

            radius = 5

            sigma = 2

            centroids *= radius

            # Set means and covariance matrices
            cov = np.identity(dim)

            true_points = np.concatenate(
                [r.multivariate_normal(mean,  cov*sigma**2, size=true_cluster_size) for mean in centroids])

            # Fraction of outliers

            # prop = 0.8

            # error_std = 4


            # outliers = error_std * r.multivariate_normal(np.zeros(dim), np.eye(dim),
            #                                                      size=math.floor(true_cluster_size * prop))



            prop = 0.80

            outliers = r.multivariate_normal(np.ones(dim)*0,
                                           np.eye(dim)*sigma_out**2, size = math.floor(true_cluster_size * prop))


            # Final points

            points = np.concatenate((true_points, outliers), axis=0)

            ## Without outliers
            # points = true_points

            # Perform k-means clustering with k clusters
            kmed_centroids, kmed_labels = kmed(points,centroids_input=copy.deepcopy(centroids),k=num_centroids)
            kmedL1_centroids, kmedL1_labels = kmedL1(points,centroids_input=copy.deepcopy(centroids),k=num_centroids)
            kmeans_centroids, kmeans_labels = kmeans(points,centroids_input=copy.deepcopy(centroids),k=num_centroids)
            # lloyd_order_centroids,lloyd_order_labels = lloyd_order(points,centroids_input=copy.deepcopy(centroids),k=num_centroids)
            # print(kmed_labels)
            #
            # print(true_labels)

            # acd computations




            kmed_acd.append(np.sum((kmed_centroids-centroids)**2)/num_centroids)
            kmedL1_acd.append(np.sum((kmedL1_centroids - centroids) ** 2) / num_centroids)
            kmeans_acd.append(np.sum((kmeans_centroids - centroids) ** 2) / num_centroids)

            # Misclustering label estimation

            kmed_misc.append(sum(kmed_labels[range(num_centroids*true_cluster_size)]!=true_labels)/len(true_labels))

            kmedL1_misc.append(sum(kmedL1_labels[range(num_centroids*true_cluster_size)]!=true_labels)/len(true_labels))

            kmeans_misc.append(sum(kmeans_labels[range(num_centroids*true_cluster_size)]!=true_labels)/len(true_labels))

        # acd average and error bar

        kmed_acd_avg.append(np.mean(kmed_acd))
        kmed_acd_err.append(1.96 * np.std(kmed_acd) / np.sqrt(len(kmed_acd)))

        kmedL1_acd_avg.append(np.mean(kmedL1_acd))
        kmedL1_acd_err.append(1.96 * np.std(kmedL1_acd) / np.sqrt(len(kmedL1_acd)))

        kmeans_acd_avg.append(np.mean(kmeans_acd))
        kmeans_acd_err.append(1.96 * np.std(kmeans_acd) / np.sqrt(len(kmeans_acd)))


        # Misclustering proportion avg and error bar

        kmed_misc_avg.append(np.mean(kmed_misc))
        kmed_misc_err.append(1.96*np.std(kmed_misc)/np.sqrt(len(kmed_misc)))

        kmedL1_misc_avg.append(np.mean(kmedL1_misc))
        kmedL1_misc_err.append(1.96*np.std(kmedL1_misc)/np.sqrt(len(kmedL1_misc)))

        kmeans_misc_avg.append(np.mean(kmeans_misc))
        kmeans_misc_err.append(1.96*np.std(kmeans_misc)/np.sqrt(len(kmeans_misc)))


    # create data frame with the misclustering

    data = {'outlier std':sigma_out_vec,'kmedians misc':kmed_misc_avg,'kmedians misc err_bar':kmed_misc_err,
            'kmedians-L1 misc':kmedL1_misc_avg,'kmedians-L1 misc err_bar':kmedL1_misc_err,
            'kmeans misc':kmeans_misc_avg,'kmeans missc err_bar':kmeans_misc_err,
            'kmedians acd': kmed_acd_avg, 'kmedians acd err_bar': kmed_acd_err,
            'kmedians-L1 acd': kmedL1_acd_avg, 'kmedians-L1 acd err_bar': kmedL1_acd_err,
            'kmeans acd': kmeans_acd_avg, 'kmeans acd err_bar': kmeans_acd_err
            }
    df = pd.DataFrame(data)

    # Save data to CSV file
    df.to_csv(f'{out_dir}/data_%g_clusters.csv'%num_centroids, index=False)


    # Plot the line plot with error bars

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.plot(sigma_out_vec, kmed_misc_avg, '-.', label='Lloyd-$L_1$',color="green")
    plt.errorbar(sigma_out_vec, kmed_misc_avg, yerr=kmed_misc_err, fmt='none', ecolor='black', capsize=3)


    plt.plot(sigma_out_vec, kmedL1_misc_avg, '--', label='k-median',color="purple")
    plt.errorbar(sigma_out_vec, kmedL1_misc_avg, yerr=kmedL1_misc_err, fmt='none', ecolor='black', capsize=3)


    plt.plot(sigma_out_vec, kmeans_misc_avg, '-', label='Llyod (k-means)',color="blue")
    plt.errorbar(sigma_out_vec, kmeans_misc_avg, yerr=kmeans_misc_err, fmt='none', ecolor='black', capsize=3)


    # plt.ylim(0,0.5)
    ax.set_xticks(sigma_out_vec)

    plt.xlabel("Outlier std")
    plt.ylabel("Misclustering proportion")
    plt.title("Plot of misc_prop: %g_clusters_rad_%g_out_%g_dim_%g" % (num_centroids,radius,prop,dim))


    # Add a legend and show the plot
    plt.legend()

    #save the figure

    temp = time.time()

    plt.savefig(f'{out_dir}/misc_prop_%g_clusters.png'%(num_centroids), dpi=300)

    plt.show(block=False)
    # plt.pause(2)
    plt.clf()

    # Plot the line plot with error bars

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.plot(sigma_out_vec, kmed_acd_avg, '-.', label='Lloyd-$L_1$', color="green")
    plt.errorbar(sigma_out_vec, kmed_acd_avg, yerr=kmed_acd_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(sigma_out_vec, kmedL1_acd_avg, '--', label='k-median', color="purple")
    plt.errorbar(sigma_out_vec, kmedL1_acd_avg, yerr=kmedL1_acd_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(sigma_out_vec, kmeans_acd_avg, '-', label='Lloyd ($k$-means)', color="blue")
    plt.errorbar(sigma_out_vec, kmeans_acd_avg, yerr=kmeans_acd_err, fmt='none', ecolor='black', capsize=3)

    # plt.ylim(0, max(np.array(kmeans_acd_avg,kmed_acd_avg,kmedL1_acd_avg))+0.3)
    ax.set_xticks(sigma_out_vec)

    plt.xlabel("Outlier std")
    plt.ylabel("acd")
    plt.title("Plot of acd: %g_clusters_rad_%g_out_%g_dim_%g" % (num_centroids,radius,prop,dim))

    # Add a legend and show the plot
    plt.legend()

    # save the figure


    plt.savefig(f'{out_dir}/acd_%g_clusters.png' % (num_centroids), dpi=300)

    plt.show(block=False)
    # plt.pause(2)
    plt.clf()
    plt.close()

