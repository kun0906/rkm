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
from data.gen_data import gen_data, plot_xy

parser = argparse.ArgumentParser()
# parser.add_argument('--force', default=False,   # whether overwrite the previous results or not?
#                     action='store_true', help='force')
parser.add_argument("--n_repeats", type=int, default=2)  #
parser.add_argument("--true_cluster_size", type=int, default=100)
parser.add_argument("--init_method", type=str, default='random')
parser.add_argument("--with_outlier", type=str, default='True')
parser.add_argument("--out_dir", type=str, default='out')
parser.add_argument("--data_name", type=str, default='pen_digits')
parser.add_argument("--fake_label", type=str, default='synthetic')
parser.add_argument("--std", type=float, default=0)
args = parser.parse_args()
args.with_outlier = False if args.with_outlier == 'False' else True
print(args)

# num_repeat = 400
num_repeat = args.n_repeats
init_method = args.init_method
true_cluster_size = args.true_cluster_size
with_outlier=args.with_outlier
data_name = args.data_name
fake_label = args.fake_label
# out_dir = f'{args.out_dir}/diffdim/{init_method}/R_{num_repeat}-S_{true_cluster_size}'
out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


for num_centroids in range(3,9,9):
    # True labels

    # true_labels = np.concatenate([np.ones(true_cluster_size)*i for i in range(num_centroids)]).astype(int)

    # dim = 10
    props = [0, 0.2, 0.4, 0.6, 0.8]

    lloydL1_misc_avg = []
    lloydL1_misc_err = []
    kmed_misc_avg = []
    kmed_misc_err = []
    kmeans_misc_avg = []
    kmeans_misc_err = []

    sc_lloydL1_misc_avg = []
    sc_lloydL1_misc_err = []
    sc_kmed_misc_avg = []
    sc_kmed_misc_err = []
    sc_kmeans_misc_avg = []
    sc_kmeans_misc_err = []
    # acd variables

    lloydL1_acd_avg = []
    lloydL1_acd_err = []
    kmed_acd_avg = []
    kmed_acd_err = []
    kmeans_acd_avg = []
    kmeans_acd_err = []

    sc_lloydL1_acd_avg = []
    sc_lloydL1_acd_err = []
    sc_kmed_acd_avg = []
    sc_kmed_acd_err = []
    sc_kmeans_acd_avg = []
    sc_kmeans_acd_err = []

    for prop in tqdm(props):

        lloydL1_misc = []
        kmed_misc = []
        kmeans_misc = []
        sc_lloydL1_misc = []
        sc_kmed_misc = []
        sc_kmeans_misc = []

        lloydL1_acd = []
        kmed_acd = []
        kmeans_acd = []

        sc_lloydL1_acd = []
        sc_kmed_acd = []
        sc_kmeans_acd = []

        for i in range(num_repeat):
            # print(prop, i, flush=True)
            seed = i
            rng = np.random.RandomState(seed=seed)
            radius = 5

            sigma = args.std
            # data_name = 'biocoin_heist'  # 'letter_recognition'
            data = gen_data(data_name, fake_label, num_centroids, true_cluster_size, prop, with_outlier,
                            random_state=i)
            true_points = data['X']
            true_labels = data['Y']
            centroids = data['centroids']
            outliers = data['outliers']

            # Final points
            if with_outlier:
                points = np.concatenate((true_points, outliers), axis=0)
            else:
                # Without outliers
                points = true_points

            # plot_xy(np.concatenate([data['X'], outliers], axis=0),
            #         np.concatenate([data['Y'], np.asarray(['100'] * len(outliers))], axis=0),
            #         random_state=i,
            #         title=f'prop: {prop}')

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
                for i in range(0, len(points), true_cluster_size):
                    if j >= num_centroids: break
                    centroids[j] = np.mean(points[i:i + true_cluster_size], axis=0)
                    j += 1

            if init_method == 'random':
                indices = rng.choice(range(len(points)), size=num_centroids, replace=False)
                init_centroids = points[indices, :]
                # init_centroids = np.copy(centroids)
            else:
                pass

            # plot_xy(points, np.concatenate([true_labels, [max(true_labels)+1] * len(outliers)]),
            #         random_state=i, true_centroids= copy.deepcopy(init_centroids),
            #         title=f'prop: {prop} after std')

            # Perform k-means clustering with k clusters
            lloydL1_centroids, lloydL1_labels = lloydL1(points,centroids_input=init_centroids,k=num_centroids, true_centroids=centroids)
            kmed_centroids, kmed_labels = kmed(points,centroids_input=init_centroids,k=num_centroids,  true_centroids=centroids)
            kmeans_centroids, kmeans_labels = kmeans(points,centroids_input=init_centroids,k=num_centroids,  true_centroids=centroids)

            sc_lloydL1_centroids, sc_lloydL1_labels = sc_random(points, k=num_centroids,
                                                                clustering_method='lloydL1',
                                                                random_state=seed, true_centroids=centroids)
            sc_kmed_centroids, sc_kmed_labels = sc_random(points, k=num_centroids, clustering_method='Kmed',
                                                          random_state=seed, true_centroids=centroids)
            sc_kmeans_centroids, sc_kmeans_labels = sc_random(points, k=num_centroids, clustering_method='kmeans',
                                                              random_state=seed, true_centroids=centroids)


            # print(lloydL1_labels)
            #
            # print(true_labels)

            # acd computations


            lloydL1_acd.append(np.sum((lloydL1_centroids-centroids)**2)/num_centroids)
            kmed_acd.append(np.sum((kmed_centroids - centroids) ** 2) / num_centroids)
            kmeans_acd.append(np.sum((kmeans_centroids - centroids) ** 2) / num_centroids)

            sc_lloydL1_acd.append(np.sum((sc_lloydL1_centroids - centroids) ** 2) / num_centroids)
            sc_kmed_acd.append(np.sum((sc_kmed_centroids - centroids) ** 2) / num_centroids)
            sc_kmeans_acd.append(np.sum((sc_kmeans_centroids - centroids) ** 2) / num_centroids)

            # Misclustering label estimation

            lloydL1_misc.append(sum(lloydL1_labels[range(num_centroids*true_cluster_size)]!=true_labels)/len(true_labels))

            kmed_misc.append(sum(kmed_labels[range(num_centroids*true_cluster_size)]!=true_labels)/len(true_labels))

            kmeans_misc.append(sum(kmeans_labels[range(num_centroids*true_cluster_size)]!=true_labels)/len(true_labels))

            sc_lloydL1_misc.append(
                sum(sc_lloydL1_labels[range(num_centroids * true_cluster_size)] != true_labels) / len(true_labels))
            sc_kmed_misc.append(
                sum(sc_kmed_labels[range(num_centroids * true_cluster_size)] != true_labels) / len(true_labels))
            sc_kmeans_misc.append(
                sum(sc_kmeans_labels[range(num_centroids * true_cluster_size)] != true_labels) / len(true_labels))
        # acd average and error bar

        lloydL1_acd_avg.append(np.mean(lloydL1_acd))
        lloydL1_acd_err.append(1.96 * np.std(lloydL1_acd) / np.sqrt(len(lloydL1_acd)))

        kmed_acd_avg.append(np.mean(kmed_acd))
        kmed_acd_err.append(1.96 * np.std(kmed_acd) / np.sqrt(len(kmed_acd)))

        kmeans_acd_avg.append(np.mean(kmeans_acd))
        kmeans_acd_err.append(1.96 * np.std(kmeans_acd) / np.sqrt(len(kmeans_acd)))

        sc_lloydL1_acd_avg.append(np.mean(sc_lloydL1_acd))
        sc_lloydL1_acd_err.append(1.96 * np.std(sc_lloydL1_acd) / np.sqrt(len(sc_lloydL1_acd)))

        sc_kmed_acd_avg.append(np.mean(sc_kmed_acd))
        sc_kmed_acd_err.append(1.96 * np.std(sc_kmed_acd) / np.sqrt(len(sc_kmed_acd)))

        sc_kmeans_acd_avg.append(np.mean(sc_kmeans_acd))
        sc_kmeans_acd_err.append(1.96 * np.std(sc_kmeans_acd) / np.sqrt(len(sc_kmeans_acd)))

        # Misclustering proportion avg and error bar

        lloydL1_misc_avg.append(np.mean(lloydL1_misc))
        lloydL1_misc_err.append(1.96*np.std(lloydL1_misc)/np.sqrt(len(lloydL1_misc)))

        kmed_misc_avg.append(np.mean(kmed_misc))
        kmed_misc_err.append(1.96*np.std(kmed_misc)/np.sqrt(len(kmed_misc)))

        kmeans_misc_avg.append(np.mean(kmeans_misc))
        kmeans_misc_err.append(1.96*np.std(kmeans_misc)/np.sqrt(len(kmeans_misc)))

        sc_lloydL1_misc_avg.append(np.mean(sc_lloydL1_misc))
        sc_lloydL1_misc_err.append(1.96 * np.std(sc_lloydL1_misc) / np.sqrt(len(sc_lloydL1_misc)))

        sc_kmed_misc_avg.append(np.mean(sc_kmed_misc))
        sc_kmed_misc_err.append(1.96 * np.std(sc_kmed_misc) / np.sqrt(len(sc_kmed_misc)))

        sc_kmeans_misc_avg.append(np.mean(sc_kmeans_misc))
        sc_kmeans_misc_err.append(1.96 * np.std(sc_kmeans_misc) / np.sqrt(len(sc_kmeans_misc)))

    # create data frame with the misclustering

    data = {'props':props,'lloydL1ians misc':lloydL1_misc_avg,'lloydL1ians misc err_bar':lloydL1_misc_err,
            'lloydL1ians-L1 misc':kmed_misc_avg,'lloydL1ians-L1 misc err_bar':kmed_misc_err,
            'kmeans misc':kmeans_misc_avg,'kmeans misc err_bar':kmeans_misc_err,
            'sc_lloydL1ians misc': sc_lloydL1_misc_avg, 'sc_lloydL1ians misc err_bar': sc_lloydL1_misc_err,
            'sc_lloydL1ians-L1 misc': sc_kmed_misc_avg, 'sc_lloydL1ians-L1 misc err_bar': sc_kmed_misc_err,
            'sc_kmeans misc': sc_kmeans_misc_avg, 'sc_kmeans misc err_bar': sc_kmeans_misc_err,

            'lloydL1ians acd': lloydL1_acd_avg, 'lloydL1ians acd err_bar': lloydL1_acd_err,
            'lloydL1ians-L1 acd': kmed_acd_avg, 'lloydL1ians-L1 acd err_bar': kmed_acd_err,
            'kmeans acd': kmeans_acd_avg, 'kmeans acd err_bar': kmeans_acd_err,
            'sc_lloydL1ians acd': sc_lloydL1_acd_avg, 'sc_lloydL1ians acd err_bar': sc_lloydL1_acd_err,
            'sc_lloydL1ians-L1 acd': sc_kmed_acd_avg, 'sc_lloydL1ians-L1 acd err_bar': sc_kmed_acd_err,
            'sc_kmeans acd': sc_kmeans_acd_avg, 'sc_kmeans acd err_bar': sc_kmeans_acd_err,
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

    plt.plot(props, sc_lloydL1_misc_avg, '-.', label='SC-Lloyd-$L_1$', color="lightgreen")
    plt.errorbar(props, sc_lloydL1_misc_avg, yerr=sc_lloydL1_misc_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(props, sc_kmed_misc_avg, '--', label='SC-k-median', color="violet")
    plt.errorbar(props, sc_kmed_misc_avg, yerr=sc_kmed_misc_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(props, sc_kmeans_misc_avg, '-', label='SC-Llyod (k-means)', color="skyblue")
    plt.errorbar(props, sc_kmeans_misc_avg, yerr=sc_kmeans_misc_err, fmt='none', ecolor='black', capsize=3)

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

    plt.plot(props, sc_lloydL1_acd_avg, '-.', label='SC-Lloyd-$L_1$', color="lightgreen")
    plt.errorbar(props, sc_lloydL1_acd_avg, yerr=sc_lloydL1_acd_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(props, sc_kmed_acd_avg, '--', label='SC-k-median', color="violet")
    plt.errorbar(props, sc_kmed_acd_avg, yerr=sc_kmed_acd_err, fmt='none', ecolor='black', capsize=3)

    plt.plot(props, sc_kmeans_acd_avg, '-', label='SC-Lloyd ($k$-means)', color="skyblue")
    plt.errorbar(props, sc_kmeans_acd_avg, yerr=sc_kmeans_acd_err, fmt='none', ecolor='black', capsize=3)

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

