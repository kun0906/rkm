import copy

import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import pandas as pd

np.random.seed(42)

# true_labels = []
#
# true_labels = [np.append(true_labels,np.ones(size))]
#
# # Set means and covariance matrices
# means = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]])
# cov = np.identity(4)
#
# # Generate random samples
# samples = []
# samples = np.append([np.random.multivariate_normal(mean, cov) for mean in means])
#
#

tot_iterate = 10

def kmeans(points, k, centroids, max_iterations=tot_iterate):
    # Randomly initialize k cluster centroids

    for i in range(max_iterations):
        # Assign each point to the closest centroid
        distances = np.sqrt(np.sum((points[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2))
        labels = np.argmin(distances, axis=1)

        # Update the centroids to be the mean of the points in each cluster
        for j in range(k):
            centroids[j] = np.mean(points[labels == j], axis=0)

    return centroids, labels


def kmed(points, k, centroids,max_iterations=tot_iterate):

    for i in range(max_iterations):
        # Assign each point to the closest centroid
        distances = np.sqrt(np.sum((points[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2))
        labels = np.argmin(distances, axis=1)

        # Update the centroids to be the mean of the points in each cluster
        for j in range(k):
            centroids[j] = np.median(points[labels == j], axis=0)

    return centroids, labels

def kmedL1(points, k, centroids,max_iterations=tot_iterate):

    for i in range(max_iterations):
        # Assign each point to the closest centroid
        distances = np.sum(np.abs(points[:, np.newaxis, :] - centroids[np.newaxis, :, :]), axis=2)
        labels = np.argmin(distances, axis=1)

        # Update the centroids to be the mean of the points in each cluster
        for j in range(k):
            centroids[j] = np.median(points[labels == j], axis=0)

    return centroids, labels


# # Generate random data points in 5 dimensions
# points = np.random.rand(100, 5)

num_repeat = 100

num_centroids = 5

true_cluster_size = 200


def acd(centroids, true_centroids):
    return np.sum((centroids - true_centroids) ** 2) / len(centroids)

for num_centroids in range(5,11,2):
    # True labels
    # N = 1000
    # true_cluster_size = N//num_centroids

    true_labels = np.concatenate([np.ones(true_cluster_size)*i for i in range(num_centroids)]).astype(int)

    kmed_dim = []

    kmedL1_dim = []

    kmeans_dim = []

    tot_dims = range(3, 20, 3) # range(6,1000,100) # range(6, 29, 2)

    # kmed_final = []
    # kmedL1_final = []
    # kmeans_final = []

    kmed_dim_avg = []
    kmed_dim_err = []
    kmedL1_dim_avg = []
    kmedL1_dim_err = []
    kmeans_dim_avg = []
    kmeans_dim_err = []

    metric_method = 'Misclustering proportion'
    metric_method = 'ACD'

    for dim in tqdm(tot_dims):
        kmed_final = []
        kmedL1_final = []
        kmeans_final = []

        for i in range(num_repeat):
            centroids = np.random.normal(size=(num_centroids, dim))
            centroids /= np.linalg.norm(centroids, axis=1)[:, np.newaxis]

            centroids *= 3

            # Set means and covariance matrices
            cov = np.identity(dim)

            true_points = np.concatenate(
                [np.random.multivariate_normal(mean, cov, size=true_cluster_size) for mean in centroids])

            # # Fraction of outliers
            prop = 0.10

            outliers = np.random.multivariate_normal(np.zeros(dim), np.eye(dim)*5.0,
                                                                 size=math.floor(true_cluster_size*num_centroids * prop))

            prop = 0.8

            error_std = 6

            # outliers = np.random.multivariate_normal(np.zeros(dim), np.eye(dim)*(error_std**2),
            #                                                      size=math.floor(true_cluster_size * prop))
            # print(np.std(outliers, axis=0))

            outliers = error_std * np.random.multivariate_normal(np.zeros(dim), np.eye(dim),
                                                                 size=math.floor(true_cluster_size * prop))
            # print(np.std(outliers, axis=0))

            # Final points

            points = np.concatenate((true_points, outliers), axis=0)
            # points = true_points

            # Perform k-means clustering with k clusters

            kmed_centroids, kmed_labels = kmed(points,centroids=copy.deepcopy(centroids),k=num_centroids)
            kmedL1_centroids, kmedL1_labels = kmedL1(points,centroids=copy.deepcopy(centroids),k=num_centroids)
            kmeans_centroids, kmeans_labels = kmeans(points,centroids=copy.deepcopy(centroids),k=num_centroids)

            # print(kmed_labels)
            #
            # print(true_labels)
            if metric_method == 'Misclustering proportion':
                kmed_final.append(sum(kmed_labels[range(num_centroids*true_cluster_size)]!=true_labels)/len(true_labels))

                kmedL1_final.append(sum(kmedL1_labels[range(num_centroids*true_cluster_size)]!=true_labels)/len(true_labels))

                kmeans_final.append(sum(kmeans_labels[range(num_centroids*true_cluster_size)]!=true_labels)/len(true_labels))

            else:
                true_centroids = centroids
                kmed_final.append(acd(kmed_centroids, true_centroids))

                kmedL1_final.append(acd(kmedL1_centroids, true_centroids))

                kmeans_final.append(acd(kmeans_centroids, true_centroids))


        kmed_dim_avg.append(np.mean(kmed_final))
        kmed_dim_err.append(1.96*np.std(kmed_final)/np.sqrt(len(kmed_final)))

        kmedL1_dim_avg.append(np.mean(kmedL1_final))
        kmedL1_dim_err.append(1.96*np.std(kmedL1_final)/np.sqrt(len(kmedL1_final)))

        kmeans_dim_avg.append(np.mean(kmeans_final))
        kmeans_dim_err.append(1.96*np.std(kmeans_final)/np.sqrt(len(kmeans_final)))

    # create data frame with the misclustering

    data = {'dimensions':tot_dims,'kmedians missc':kmed_dim_avg,'kmedians missc err_bar':kmed_dim_err,
            'kmedians-L1 missc':kmedL1_dim_avg,'kmedians-L1 missc err_bar':kmedL1_dim_err,
            'kmeans missc':kmeans_dim_avg,'kmeans missc err_bar':kmeans_dim_err}
    df = pd.DataFrame(data)

    # Save data to CSV file
    df.to_csv('data_%g_clusters.csv'%num_centroids, index=False)


    # Plot the line plot with error bars

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.plot(tot_dims, kmed_dim_avg, '-.', label='k-median',color="green")
    plt.errorbar(tot_dims, kmed_dim_avg, yerr=kmed_dim_err, fmt='none', ecolor='black', capsize=3)


    plt.plot(tot_dims, kmedL1_dim_avg, '--', label='k-medianL1',color="purple")
    plt.errorbar(tot_dims, kmedL1_dim_avg, yerr=kmedL1_dim_err, fmt='none', ecolor='black', capsize=3)


    plt.plot(tot_dims, kmeans_dim_avg, '-', label='k-means',color="blue")
    plt.errorbar(tot_dims, kmeans_dim_avg, yerr=kmeans_dim_err, fmt='none', ecolor='black', capsize=3)


    # plt.ylim(0,0.5)
    ax.set_xticks(tot_dims)

    plt.xlabel("Dimensions")
    if metric_method == 'Misclustering proportion':
        plt.ylabel("Misclustering proportion")
        plt.title("Plot of misclustering proportion for %g clusters"%num_centroids)
    else:
        plt.ylabel("ACD")
        plt.title("ACD for %g clusters" % num_centroids)


    # Add a legend and show the plot
    plt.legend()

    #save the figure

    plt.savefig('plot_%g_clusters.png'%num_centroids, dpi=300)

    plt.show(block=False)
    plt.pause(2)
    plt.close()


