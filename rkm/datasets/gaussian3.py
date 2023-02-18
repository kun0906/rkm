import copy
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
    # e.g., 'r:0.1|mu:10,0|cov:0.1,0.1', which means we draw 10% outliers from a Gaussian with mu=(10, 0) and cov=(0.1, 0.1)

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
        mu1 = np.asarray([-1, 0])
        sigma = 1.0
        cov1 = np.asarray([[sigma, 0], [0, sigma]])
        X1 = r.multivariate_normal(mu1, cov1, size=n1)
        y1 = np.asarray(['c1'] * n1)

        ############
        # cluster 2
        n2 = 500
        mu2 = np.asarray([1, 0])
        cov2 = np.asarray([[sigma, 0], [0, sigma]])
        X2 = r.multivariate_normal(mu2, cov2, size=n2)
        y2 = np.asarray(['c2'] * n2)

        # obtain ground truth centroids
        true_centroids = np.zeros((2, 2))
        # true_centroids[0] = np.mean(X1, axis=0)
        # true_centroids[1] = np.mean(X2, axis=0)

        true_centroids[0] = mu1 # ground-truth
        true_centroids[1] = mu2 # ground-truth

        # # obtain initial centroids, i.e., random select 2 data points from cluster 1
        # indices = r.choice(range(0, n1), size=2, replace=False)  # without replacement and random
        # init_centroids = X1[indices]
        init_centroids = tuple([copy.deepcopy(X1), copy.deepcopy(X2)])

        ############
        # outliers
        n_outliers = int(np.round((n1+n2)*ratio))
        X_outliers = r.multivariate_normal(mu, cov, size=n_outliers)
        y_outliers = np.asarray(['noise'] * n_outliers)

        # Combine them togather
        X = np.concatenate([X1, X2, X_outliers], axis=0)
        y = np.concatenate([y1, y2, y_outliers], axis=0)

        delta_X = abs(mu[0])
        return X, y, true_centroids, init_centroids, delta_X

    X,y, true_centroids, init_centroids, delta_X= get_xy(ratio, mu, cov)

    is_show = args['IS_SHOW']
    # is_show=True
    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        label2color= {'c1':'b', 'c2':'g', 'noise': 'r'}
        for l, name in [('c1', '$G_{11}$'), ('c2', '$G_{12}$'), ('noise', '$G_{13}$')]:
            mask = y==l
            ax.scatter(X[mask, 0], X[mask, 1], c=[label2color[l] for l in y[mask]], marker="x", s=10, alpha=0.3, label=name)
        # ax.scatter(X[:, 0], X[:, 1], c = [label2color[l] for l in y], marker="x", s=10, alpha=0.3, label='')
        p = np.mean(X, axis=0)
        # ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        # xytext = (p[0] - offset, p[1] - offset)
        # # print(xytext)
        # ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
        #             ha='center', va='center',  # textcoords='offset points',
        #             bbox=dict(facecolor='none', edgecolor='b', pad=1),
        #             arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
        #                             connectionstyle="angle3, angleA=90,angleB=0"))

        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right', fontsize = 13)
        if args['SHOW_TITLE']:
            plt.title(dataset_detail.replace(':', ':'))

        if 'xlim' in kwargs:
            plt.xlim(kwargs['xlim'])
        else:
            plt.xlim([-5, 8])
        if 'ylim' in kwargs:
            plt.ylim(kwargs['ylim'])
        else:
            plt.ylim([-4, 4])

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
        if not args['IS_SHOW']: plt.show()


    return {'X':X, 'y': y, 'true_centroids': true_centroids, 'init_centroids': init_centroids, 'delta_X': delta_X}




def gaussian3_diff2_outliers(args, random_state=42, **kwargs):
    """
    # two clusters ((-3,0), (3, 0)) with same covariance matrix and size in R^2, i.e., n1 = n2 = 500
    # 10% of outliers, i.e., (n1+n2)*0.1.
    # Move the outliers away from n1 over time.
    # e.g., 'r:0.1|mu:10,0|cov:0.1,0.1', which means we draw 10% outliers from a Gaussian with mu=(10, 0) and cov=(0.1, 0.1)

    params['p1'] == 'diff_outliers':
    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """
    # r:0.1|mu:0,0|cov:0.1,0.1|diff_outliers
    dataset_detail = args['DATASET']['detail']
    tmp = dataset_detail.split('|')
    ratio = float(tmp[0].split(':')[1])

    mu = tmp[1].split(':')[1].split(',')
    mu = np.asarray([float(mu[0]), float(mu[1])])

    cov = tmp[2].split(':')[1].split(',')
    cov_outliers = np.asarray([[float(cov[0]), 0], [0,float(cov[1])]])

    r = np.random.RandomState(random_state)
    def get_xy(ratio, mu, cov):

        ############
        # cluster 1
        n1 = 500
        mu1 = np.asarray([-1, 0])
        sigma = 1.0
        cov1 = np.asarray([[sigma, 0], [0, sigma]])
        X1 = r.multivariate_normal(mu1, cov1, size=n1)
        y1 = np.asarray(['c1'] * n1)

        ############
        # cluster 2
        n2 = 500
        mu2 = np.asarray([1, 0])
        cov2 = np.asarray([[sigma, 0], [0, sigma]])
        X2 = r.multivariate_normal(mu2, cov2, size=n2)
        y2 = np.asarray(['c2'] * n2)

        # obtain ground truth centroids
        true_centroids = np.zeros((2, 2))
        # true_centroids[0] = np.mean(X1, axis=0)
        # true_centroids[1] = np.mean(X2, axis=0)
        true_centroids[0] = mu1  # ground-truth
        true_centroids[1] = mu2  # ground-truth

        # # obtain initial centroids, i.e., random select 2 data points from cluster 1
        # indices = r.choice(range(0, n1), size=2, replace=False)  # without replacement and random
        # init_centroids = X1[indices]
        init_centroids = tuple([copy.deepcopy(X1), copy.deepcopy(X2)])

        ############
        # outliers
        n_outliers = int(np.round((n1+n2)*ratio))
        X_outliers = r.multivariate_normal(np.asarray([0, 0]), cov_outliers, size=n_outliers)
        y_outliers = np.asarray(['noise'] * n_outliers)

        # Combine them togather
        X = np.concatenate([X1, X2, X_outliers], axis=0)
        y = np.concatenate([y1, y2, y_outliers], axis=0)

        delta_X = abs(cov_outliers[0][0])
        return X, y, true_centroids, init_centroids, delta_X

    X,y, true_centroids, init_centroids, delta_X= get_xy(ratio, mu, cov)

    is_show = args['IS_SHOW']
    # is_show=True
    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        label2color= {'c1':'b', 'c2':'g', 'noise': 'r'}
        for l, name in [('c1', '$G_{21}$'), ('c2', '$G_{22}$'), ('noise', '$G_{23}$')]:
            mask = y==l
            ax.scatter(X[mask, 0], X[mask, 1], c=[label2color[l] for l in y[mask]], marker="x", s=10, alpha=0.3, label=name)
        # ax.scatter(X[:, 0], X[:, 1], c = [label2color[l] for l in y], marker="x", s=10, alpha=0.3, label='')
        p = np.mean(X, axis=0)
        # ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        # xytext = (p[0] - offset, p[1] - offset)
        # # print(xytext)
        # ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
        #             ha='center', va='center',  # textcoords='offset points',
        #             bbox=dict(facecolor='none', edgecolor='b', pad=1),
        #             arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
        #                             connectionstyle="angle3, angleA=90,angleB=0"))

        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right', fontsize = 13)
        if args['SHOW_TITLE']:
            plt.title(dataset_detail.replace(':', ':'))

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
        if not args['IS_SHOW']: plt.show()


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
        y1 = np.asarray(['c1'] * n1)

        ############
        # cluster 2
        n2 = 500
        mu2 = np.asarray([1*d, 0])
        cov2 = np.asarray([[0.1, 0], [0, 0.1]])
        X2 = r.multivariate_normal(mu2, cov2, size=n2)
        y2 = np.asarray(['c2'] * n2)

        # obtain  ground truth centroids
        true_centroids = np.zeros((2, 2))
        # true_centroids[0] = np.mean(X1, axis=0)
        # true_centroids[1] = np.mean(X2, axis=0)
        true_centroids[0] = mu1  # ground-truth
        true_centroids[1] = mu2  # ground-truth

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

        # init_centroids = np.zeros((2, 2))
        # init_centroids[0] = np.mean(X1_, axis=0)
        # init_centroids[1] = np.mean(X2_, axis=0)
        init_centroids = tuple([copy.deepcopy(X1_), copy.deepcopy(X2_)])

        delta_X = 2 * d

        return X, y, true_centroids, init_centroids, delta_X

    X, y, true_centroids, init_centroids, delta_X = get_xy(d, ratio)

    return {'X': X, 'y': y, 'true_centroids': true_centroids, 'init_centroids': init_centroids, 'delta_X': delta_X}



def gaussian3_constructed_clusters(args, random_state=42, **kwargs):
    """
        p:0.4|constructed_3gaussians
        Constructed example, in which mean doesn't work; however, median works.
        3 Gaussians:
            the first Gaussian with mu= [-1, 0] and covariance = [1, 1], n1=1000;
            the second Gaussian with mu= [1, 0] and covariance = [1, 1], n2=1000; and
            the last Gaussian with mu= [10, 0] and covariance = [3, 3], n3=2000;

            # For the initialization,
            The first cluster with the mean of Gaussian 1 as its centroid;
            The second cluster with the mean of (100% Gaussian 2 + p (e.g., 20%) of data from Gaussian 3) as its centroid; and
            The third cluster with the mean of (1-p) of data from Gaussian 3 as its centroid.

    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """
    # p:0.4|constructed_3gaussians
    dataset_detail = args['DATASET']['detail']
    tmp = dataset_detail.split('|')

    p = float(tmp[0].split(':')[1])

    r = np.random.RandomState(random_state)
    def get_xy(p):

        ############
        # cluster 1
        n0 = 500
        mu0 = np.asarray([-1, 0])
        sigma = 1.0
        cov0 = np.asarray([[sigma, 0], [0, sigma]])
        X0 = r.multivariate_normal(mu0, cov0, size=n0)
        y0 = np.asarray(['c1'] * n0)

        ############
        # cluster 2
        n1 = 500
        mu1 = np.asarray([1, 0])
        cov1 = np.asarray([[sigma, 0], [0, sigma]])
        X1 = r.multivariate_normal(mu1, cov1, size=n1)
        y1 = np.asarray(['c2'] * n1)

        ############
        # cluster 3
        n2 = 1000
        mu2 = np.asarray([10, 0])
        cov2 = np.asarray([[sigma, 0], [0, sigma]])
        X2 = r.multivariate_normal(mu2, cov2, size=n2)
        y2 = np.asarray(['c3'] * n2)

        # obtain  ground truth centroids
        true_centroids = np.zeros((3, 2))
        # true_centroids[0] = np.mean(X0, axis=0)
        # true_centroids[1] = np.mean(X1, axis=0)
        # true_centroids[2] = np.mean(X2, axis=0)
        true_centroids[0] = mu0  # ground-truth
        true_centroids[1] = mu1  # ground-truth
        true_centroids[2] = mu2  # ground-truth

        n3 = int(np.floor((n0 + n1 + n2) * 0.1))  # for noise

        # obtain initial centroids after mixing the data
        X2, X12, y2, y12 = train_test_split(X2, y2, test_size=p, shuffle=True, random_state=random_state)
        # Mix them togather
        X1 = np.concatenate([X1, X12], axis=0)
        y1 = np.concatenate([y1, y12], axis=0)

        is_noise = False
        if is_noise:
            # noise
            X3 = r.uniform(low=[-5, -5], high=[15, 5], size=(n3, 2))
            y3 = np.asarray(['noise'] * n3)

            X = np.concatenate([X0, X1, X2, X3], axis=0)
            y = np.concatenate([y0, y1, y2, y3], axis=0)
        else:
            X = np.concatenate([X0, X1, X2], axis=0)
            y = np.concatenate([y0, y1, y2], axis=0)

        init_centroids = tuple([copy.deepcopy(X0), copy.deepcopy(X1), copy.deepcopy(X2)])
        # if args['ALGORITHM']['py_name'] == 'kmeans':
        #     init_centroids[0] = np.mean(X0, axis=0)
        #     init_centroids[1] = np.mean(X1, axis=0)
        #     init_centroids[2] = np.mean(X2, axis=0)
        # elif args['ALGORITHM']['py_name']== 'kmedian':
        #     init_centroids[0] = np.median(X0, axis=0)
        #     init_centroids[1] = np.median(X1, axis=0)
        #     init_centroids[2] = np.median(X2, axis=0)
        # else:
        #     raise NotImplementedError(args['ALGORITHM']['py_name'])

        delta_X = p

        return X, y, true_centroids, init_centroids, delta_X

    X, y, true_centroids, init_centroids, delta_X = get_xy(p)

    is_show =  args['IS_SHOW']
    # is_show = True
    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        label2color = {'c1': 'b', 'c2': 'g', 'c3':'m', 'noise': 'r'}
        for l, name in [('c1', '$G_{41}$'), ('c2', '$G_{42}$'), ('c3', '$G_{43}$')]:
            mask = y==l
            ax.scatter(X[mask, 0], X[mask, 1], c=[label2color[l] for l in y[mask]], marker="x", s=10, alpha=0.3, label=name)
        # ax.scatter(X[:, 0], X[:, 1], c=[label2color[l] for l in y], marker="x", s=10, alpha=0.3, label='')
        p = np.mean(X, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] - offset, p[1] - offset)
        # # print(xytext)
        # ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
        #             ha='center', va='center',  # textcoords='offset points',
        #             bbox=dict(facecolor='none', edgecolor='b', pad=1),
        #             arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
        #                             connectionstyle="angle3, angleA=90,angleB=0"))

        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right', fontsize=13)
        if args['SHOW_TITLE']:
            plt.title(dataset_detail.replace(':', ':'))

        # if 'xlim' in kwargs:
        #     plt.xlim(kwargs['xlim'])
        # else:
        #     plt.xlim([-6, 6])
        if 'ylim' in kwargs:
            plt.ylim(kwargs['ylim'])
        else:
            plt.ylim([-5, 5])

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

    return {'X': X, 'y': y, 'true_centroids': true_centroids, 'init_centroids': init_centroids, 'delta_X': delta_X}




def gaussian3_constructed2_clusters(args, random_state=42, **kwargs):
    """
        p:0.4|constructed_3gaussians
        Constructed example, in which mean doesn't work; however, median works.
        3 Gaussians:
            the first Gaussian with mu= [-1, 0] and covariance = [1, 1], n1=1000;
            the second Gaussian with mu= [1, 0] and covariance = [1, 1], n2=1000; and
            the last Gaussian with mu= [np.sqrt(3), 0] and covariance = [1, 1], n3=2000;

            # For the initialization,
            The first cluster with the mean of Gaussian 1 as its centroid;
            The second cluster with the mean of (100% Gaussian 2 + p (e.g., 20%) of data from Gaussian 3) as its centroid; and
            The third cluster with the mean of (1-p) of data from Gaussian 3 as its centroid.

    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """
    # p:0.4|constructed_3gaussians
    dataset_detail = args['DATASET']['detail']
    tmp = dataset_detail.split('|')

    p = float(tmp[0].split(':')[1])

    r = np.random.RandomState(random_state)
    def get_xy(p):

        ############
        # cluster 1
        n0 = 500
        mu0 = np.asarray([-3, 0])
        sigma = 1.0
        cov0 = np.asarray([[sigma, 0], [0, sigma]])
        X0 = r.multivariate_normal(mu0, cov0, size=n0)
        y0 = np.asarray(['c1'] * n0)

        ############
        # cluster 2
        n1 = 500
        mu1 = np.asarray([3, 0])
        cov1 = np.asarray([[sigma, 0], [0, sigma]])
        X1 = r.multivariate_normal(mu1, cov1, size=n1)
        y1 = np.asarray(['c2'] * n1)

        ############
        # cluster 3
        n2 = 500
        mu2 = np.asarray([0,3*np.sqrt(3)])
        cov2 = np.asarray([[sigma, 0], [0, sigma]])
        X2 = r.multivariate_normal(mu2, cov2, size=n2)
        y2 = np.asarray(['c3'] * n2)

        # obtain  ground truth centroids
        true_centroids = np.zeros((3, 2))
        # true_centroids[0] = np.mean(X0, axis=0)
        # true_centroids[1] = np.mean(X1, axis=0)
        # true_centroids[2] = np.mean(X2, axis=0)
        true_centroids[0] = mu0  # ground-truth
        true_centroids[1] = mu1  # ground-truth
        true_centroids[2] = mu2  # ground-truth

        X = np.concatenate([X0, X1, X2], axis=0)
        y = np.concatenate([y0, y1, y2], axis=0)

        # obtain initial centroids after mixing the data
        X0, X00, y0, y00 = train_test_split(X0, y0, test_size=p, shuffle=True, random_state=random_state)
        X1, X11, y1, y11 = train_test_split(X1, y1, test_size=p, shuffle=True, random_state=random_state)
        X2, X22, y2, y22 = train_test_split(X2, y2, test_size=p, shuffle=True, random_state=random_state)
        # Mix them togather
        X0 = np.concatenate([X0, X11], axis=0)
        y0 = np.concatenate([y0, y11], axis=0)
        X1 = np.concatenate([X1, X22], axis=0)
        y1 = np.concatenate([y1, y22], axis=0)
        X2 = np.concatenate([X2, X00], axis=0)
        y2 = np.concatenate([y2, y00], axis=0)

        init_centroids = tuple([copy.deepcopy(X0), copy.deepcopy(X1), copy.deepcopy(X2)])
        # if args['ALGORITHM']['py_name'] == 'kmeans':
        #     init_centroids[0] = np.mean(X0, axis=0)
        #     init_centroids[1] = np.mean(X1, axis=0)
        #     init_centroids[2] = np.mean(X2, axis=0)
        # elif args['ALGORITHM']['py_name']== 'kmedian':
        #     init_centroids[0] = np.median(X0, axis=0)
        #     init_centroids[1] = np.median(X1, axis=0)
        #     init_centroids[2] = np.median(X2, axis=0)
        # else:
        #     raise NotImplementedError(args['ALGORITHM']['py_name'])

        delta_X = p

        return X, y, true_centroids, init_centroids, delta_X

    X, y, true_centroids, init_centroids, delta_X = get_xy(p)

    is_show =  args['IS_SHOW']
    # is_show = True
    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        label2color = {'c1': 'b', 'c2': 'g', 'c3':'m', 'noise': 'r'}

        for l, name in [('c1', '$G_{31}$'), ('c2', '$G_{32}$'), ('c3', '$G_{33}$')]:
            mask = y==l
            ax.scatter(X[mask, 0], X[mask, 1], c=[label2color[l] for l in y[mask]], marker="x", s=10, alpha=0.3, label=name)
        # p = np.mean(X, axis=0)
        # ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        # xytext = (p[0] - offset, p[1] - offset)
        # # print(xytext)
        # ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
        #             ha='center', va='center',  # textcoords='offset points',
        #             bbox=dict(facecolor='none', edgecolor='b', pad=1),
        #             arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
        #                             connectionstyle="angle3, angleA=90,angleB=0"))

        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right', fontsize=13)
        if args['SHOW_TITLE']:
            plt.title(dataset_detail.replace(':', ':'))

        if 'xlim' in kwargs:
            plt.xlim(kwargs['xlim'])
        else:
            plt.xlim([-7, 7])
        if 'ylim' in kwargs:
            plt.ylim(kwargs['ylim'])
        else:
            plt.ylim([-4, 8])

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

    return {'X': X, 'y': y, 'true_centroids': true_centroids, 'init_centroids': init_centroids, 'delta_X': delta_X}
