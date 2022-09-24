import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
import collections

def gaussian3_diff_outliers(args, random_state=42, **kwargs):
    """
    # two clusters ((-3,0), (3, 0)) with same covariance matrix and size in R^2, i.e., n1 = n2 = 500
    # 10% of outliers, i.e., (n1+n2)*0.1.
    # Move the outliers away from n1 over time.
    # e.g., 'r:0.1|mu:-10,0|cov:0.1,0.1', which means we draw 10% outliers from a Gaussian with mu=(-10, 0) and cov=(0.1, 0.1)

    params['p1'] == 'diff_outliers':
    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """
    # r:0.1|mu:-3,0|cov:0.1,0.1|diff_outliers
    dataset_detail = args['DATASET']['detail']
    tmp = dataset_detail.split('|')
    ratio = float(tmp[0].split(':')[1])

    mu = tmp[1].split(':')[1].split(',')
    mu = np.asarray([float(mu[0]), float(mu[1])])

    cov = tmp[2].split(':')[1].split(',')
    cov = np.asarray([[float(cov[0]), 0], [0,float(cov[1])]])

    r = np.random.RandomState(random_state)
    def get_xy(ratio, mu, cov):

        ############
        # cluster 1
        n1 = 500
        mu1 = np.asarray([-3, 0])
        cov1 = np.asarray([[0.1, 0], [0, 0.1]])
        X1 = r.multivariate_normal(mu1, cov1, size=n1)
        y1 = np.asarray([-1] * n1)

        ############
        # cluster 2
        n2 = 500
        mu2 = np.asarray([3, 0])
        cov2 = np.asarray([[0.1, 0], [0, 0.1]])
        X2 = r.multivariate_normal(mu2, cov2, size=n2)
        y2 = np.asarray([1] * n2)

        # obtain ground truth centroids
        true_centroids = np.zeros((2, 2))
        true_centroids[0] = np.mean(X1, axis=0)
        true_centroids[1] = np.mean(X2, axis=0)

        # obtain initial centroids, i.e., random select 2 data points from cluster 1
        indices = r.choice(range(0, n1), size=2, replace=False)  # without replacement and random
        init_centroids = X1[indices]

        ############
        # outliers
        n_outliers = int((n1+n2)*ratio)
        X_outliers = r.multivariate_normal(mu, cov, size=n_outliers)
        y_outliers = np.asarray([2] * n_outliers)

        # Combine them togather
        X = np.concatenate([X1, X2, X_outliers], axis=0)
        y = np.concatenate([y1, y2, y_outliers], axis=0)

        delta_X = abs(mu[0])
        return X, y, true_centroids, init_centroids, delta_X

    X,y, true_centroids, init_centroids, delta_X= get_xy(ratio, mu, cov)

    is_show = args['IS_SHOW']
    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        ax.scatter(X[:, 0], X[:, 1], c=y, marker="x", s=10, alpha=0.3, label='$G_1$')
        p = np.mean(X, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] - offset, p[1] - offset)
        # print(xytext)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
                    ha='center', va='center',  # textcoords='offset points',
                    bbox=dict(facecolor='none', edgecolor='b', pad=1),
                    arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))

        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right', fontsize = 13)
        if args['SHOW_TITLE']:
            plt.title(dataset_detail.replace(':', '\n'))

        if 'xlim' in kwargs:
            plt.xlim(kwargs['xlim'])
        else:
            plt.xlim([-6, 6])
        if 'ylim' in kwargs:
            plt.ylim(kwargs['ylim'])
        else:
            plt.ylim([-6, 6])

        fontsize = 13
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        plt.tight_layout()
        # if not os.path.exists(params['OUT_DIR']):
        #     os.makedirs(params['OUT_DIR'])
        # f = os.path.join(args['OUT_DIR'], dataset_detail+'.png')
        f = args['data_file'] + '.png'
        print(f)
        plt.savefig(f, dpi=600, bbox_inches='tight')
        plt.show()


    return {'X':X, 'y': y, 'true_centroids': true_centroids, 'init_centroids': init_centroids, 'delta_X': delta_X}




def gaussian3_mixed_clusters(args, random_state=42, **kwargs):
    """
    # two clusters ((-3,0), (3, 0)) with same covariance matrix and size in R^2, i.e., n1 = n2 = 500
    # mix the two clusters with different ratio
    # e.g., 'r:0.4|mixed_clusters', which means we draw 40% data from cluster 1 and add them to cluster2, and vice versa for cluster 2.

    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """
    # d:2|r:0.1|mixed_clusters
    dataset_detail = args['DATASET']['detail']
    tmp = dataset_detail.split('|')

    d = float(tmp[0].split(':')[1])

    ratio = float(tmp[1].split(':')[1])

    r = np.random.RandomState(random_state)
    def get_xy(d, ratio):

        ############
        # cluster 1
        n1 = 500
        mu1 = np.asarray([-1*d, 0])
        cov1 = np.asarray([[0.1, 0], [0, 0.1]])
        X1 = r.multivariate_normal(mu1, cov1, size=n1)
        y1 = np.asarray([-1] * n1)

        ############
        # cluster 2
        n2 = 500
        mu2 = np.asarray([1*d, 0])
        cov2 = np.asarray([[0.1, 0], [0, 0.1]])
        X2 = r.multivariate_normal(mu2, cov2, size=n2)
        y2 = np.asarray([1] * n2)

        # obtain  ground truth centroids
        true_centroids = np.zeros((2, 2))
        true_centroids[0] = np.mean(X1, axis=0)
        true_centroids[1] = np.mean(X2, axis=0)

        X = np.concatenate([X1, X2], axis=0)
        y = np.concatenate([y1, y2], axis=0)

        # obtain initial centroids after mixing the data
        X11, X12, y11, y12 = train_test_split(X1, y1, test_size=ratio, shuffle=True, random_state=random_state)
        X21, X22, y21, y22 = train_test_split(X2, y2, test_size=ratio, shuffle=True, random_state=random_state)
        # Mix them togather
        X1_ = np.concatenate([X11, X22], axis=0)
        # y1_ = np.concatenate([y11, y22], axis=0)
        X2_ = np.concatenate([X21, X12], axis=0)
        # y2_ = np.concatenate([y21, y12], axis=0)

        init_centroids = np.zeros((2, 2))
        init_centroids[0] = np.mean(X1_, axis=0)
        init_centroids[1] = np.mean(X2_, axis=0)

        delta_X = 2 * d

        return X, y, true_centroids, init_centroids, delta_X

    X, y, true_centroids, init_centroids, delta_X = get_xy(d, ratio)

    return {'X': X, 'y': y, 'true_centroids': true_centroids, 'init_centroids': init_centroids, 'delta_X': delta_X}


