import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
import collections

def gaussian3_1client_1cluster(params, random_state=42):
    """
    # 2 clusters ((-1,0), (1, 0)) in R^2, each client has one cluster. 2 clusters has overlaps.
    params['p1'] == '1client_1cluster':
    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """

    def get_xy(n=10000):

        # client 1
        mus = [-1, 0]
        sigma = 0.1
        n1 = 10000
        cov = np.asarray([[sigma, 0], [0, sigma]])
        # cov = np.asarray([[sigma, sigma], [sigma, sigma]])
        r = np.random.RandomState(random_state)
        X1 = r.multivariate_normal(mus, cov, size=n1)
        y1 = np.asarray([0] * n1)

        # client 2
        mus = [1, 0]
        sigma = 0.1
        n2 = 10000
        cov = np.asarray([[sigma, 0], [0, sigma]])
        # cov = np.asarray([[sigma, -sigma], [-sigma, sigma]])
        X2 = r.multivariate_normal(mus, cov, size=n2)
        y2 = np.asarray([1] * n2)

        mus = [0, 3]
        n3 = 10000
        sigma = 0.1
        cov = np.asarray([[sigma + 0.9, 0], [0, sigma]])
        X3 = r.multivariate_normal(mus, cov, size=n3)
        y3 = np.asarray([2] * n3)

        return X1, y1, X2, y2, X3, y3

    X1, y1, X2, y2, X3, y3 = get_xy(n=10000)

    is_show = params['is_show']
    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='Centroid 1')
        p = np.mean(X1, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] - offset, p[1] - offset)
        print(xytext)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
                    ha='center', va='center',  # textcoords='offset points',
                    bbox=dict(facecolor='none', edgecolor='b', pad=1),
                    arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        # angleA : starting angle of the path
        # angleB : ending angle of the path

        ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='Centroid 2')
        p = np.mean(X2, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))

        ax.scatter(X3[:, 0], X3[:, 1], c=colors[2], marker="o", s=10, alpha=0.3, label='Centroid 3')
        p = np.mean(X3, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))

        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right')
        plt.title(params['p1'])
        # # plt.xlim([-2, 15])
        # # plt.ylim([-2, 15])
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
        # # plt.xticks([])
        # # plt.yticks([])
        if not os.path.exists(params['out_dir']):
            os.makedirs(params['out_dir'])
        f = os.path.join(params['out_dir'], params['p1']+'.png')
        plt.savefig(f, dpi=600, bbox_inches='tight')
        plt.show()

    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    for i, (x, y) in enumerate([(X1, y1), (X2, y2), (X3, y3)]):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=True,
                                                            random_state=random_state)
        clients_train_x.append(train_x)
        clients_train_y.append(train_y)
        clients_test_x.append(test_x)
        clients_test_y.append(test_y)

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    return x, labels


# gaussian3_1client_1cluster({})


def gaussian3_random_ratio(params, random_state=42):
    ratio = params['ratio']
    def get_xy(n=10000):

        # client 1
        mus = [-1, 0]
        sigma = 0.1
        n1 = 10000
        cov = np.asarray([[sigma, 0], [0, sigma]])
        # cov = np.asarray([[sigma, sigma], [sigma, sigma]])
        r = np.random.RandomState(random_state)
        X1 = r.multivariate_normal(mus, cov, size=n1)
        y1 = np.asarray([0] * n1)

        # client 2
        mus = [1, 0]
        sigma = 0.1
        n2 = 10000
        cov = np.asarray([[sigma, 0], [0, sigma]])
        # cov = np.asarray([[sigma, -sigma], [-sigma, sigma]])
        X2 = r.multivariate_normal(mus, cov, size=n2)
        y2 = np.asarray([1] * n2)

        # client 3
        mus = [0, 3]
        n3 = 10000
        sigma = 0.1
        cov = np.asarray([[sigma + 0.9, 0], [0, sigma]])
        X3 = r.multivariate_normal(mus, cov, size=n3)
        y3 = np.asarray([2] * n3)

        # # mus = [0, -3]
        # cov = np.asarray([[sigma, 0], [0, sigma]])
        # X4 = r.multivariate_normal(mus, cov, size=n1)
        # y4 = np.asarray([1] * n1)

        # X1 = np.concatenate([X1, X2], axis=0)
        # y1 = np.concatenate([y1, y2], axis=0)
        # X3 = np.concatenate([X3, X4], axis=0)
        # y3 = np.concatenate([y3, y4], axis=0)

        return X1, y1, X2, y2, X3, y3

    n = 10000
    X1, y1, X2, y2, X3, y3 = get_xy(n=10000)
    X = np.concatenate([X1, X2, X3], axis=0)
    Y = np.concatenate([y1, y2, y3], axis=0)
    # # new_y1 = np.concatenate([y1[indices1], y2[indices2]], axis=0)
    # new_y1 = np.zeros((new_X1.shape[0],))
    # new_X2 = np.concatenate([X1[indices3], X2[indices4]], axis=0)
    # # new_y2 = np.concatenate([y1[indices3], y2[indices4]], axis=0)
    # new_y2 = np.ones((new_X2.shape[0],))
    # X1, y1, X2, y2 = new_X1, new_y1, new_X2, new_y2

    # client 1: 90% cluster1, 10 % cluster2, 10 % cluster3
    # client 2: 10% cluster1, 90 % cluster2, 10 % cluster3
    # client 3: 10% cluster1, 10 % cluster2, 90 % cluster3
    X, X1, Y, y1 = train_test_split(X, Y, test_size=n, shuffle=True,
                                                  random_state=random_state)
    y1 = np.zeros((X1.shape[0],))
    X3, X2, y3, y2 = train_test_split(X, Y, test_size=ratio, shuffle=True,
                                                  random_state=random_state)
    y2 = np.ones((X2.shape[0],))
    y3 = np.ones((X3.shape[0],)) * 2

    is_show = params['is_show']
    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='Centroid 1')
        p = np.mean(X1, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] - offset, p[1] - offset)
        print(xytext)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
                    ha='center', va='center',  # textcoords='offset points',
                    bbox=dict(facecolor='none', edgecolor='b', pad=1),
                    arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        # angleA : starting angle of the path
        # angleB : ending angle of the path

        ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='Centroid 2')
        p = np.mean(X2, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))

        ax.scatter(X3[:, 0], X3[:, 1], c=colors[2], marker="o", s=10, alpha=0.3, label='Centroid 3')
        p = np.mean(X3, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))

        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right')
        plt.title(params['p1'])
        # # plt.xlim([-2, 15])
        # # plt.ylim([-2, 15])
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
        # # plt.xticks([])
        # # plt.yticks([])
        if not os.path.exists(params['out_dir']):
            os.makedirs(params['out_dir'])
        f = os.path.join(params['out_dir'], params['p1']+'.png')
        plt.savefig(f, dpi=600, bbox_inches='tight')
        plt.show()

    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    for i, (x, y) in enumerate([(X1, y1), (X2, y2), (X3, y3)]):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=True,
                                                            random_state=random_state)
        clients_train_x.append(train_x)
        clients_train_y.append(train_y)
        clients_test_x.append(test_x)
        clients_test_y.append(test_y)

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    return x, labels



def gaussian3_mix_clusters_per_client(params, random_state=42, **kwargs):
    """
     if params['p1'] == 'mix_clusters_per_client':
    # 2 clusters in R^2:
    # 1) client1 has 70% data from cluster 1 and 30% data from cluster2
    # 2) client2 has 30% data from cluster 1 and 70% data from cluster2
    return gaussian3_mix_clusters_per_client(params, random_state=seed)


    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """
    p1 = params['p1'].split(':')    # ratio_0.1:mix_clusters_per_client
    ratio = float(p1[0].split('_')[1])
    def get_xy(n=10000):

        # client 1
        mus = [-1, 0]
        sigma = 0.1
        n1 = 10000
        cov = np.asarray([[sigma, 0], [0, sigma]])
        # cov = np.asarray([[sigma, sigma], [sigma, sigma]])
        r = np.random.RandomState(random_state)
        X1 = r.multivariate_normal(mus, cov, size=n1)
        y1 = np.asarray([0] * n1)

        # client 2
        mus = [1, 0]
        sigma = 0.1
        n2 = 10000
        cov = np.asarray([[sigma, 0], [0, sigma]])
        # cov = np.asarray([[sigma, -sigma], [-sigma, sigma]])
        X2 = r.multivariate_normal(mus, cov, size=n2)
        y2 = np.asarray([1] * n2)

        # client 3
        mus = [0, 3]
        n3 = 10000
        sigma = 0.1
        cov = np.asarray([[sigma + 0.9, 0], [0, sigma]])
        X3 = r.multivariate_normal(mus, cov, size=n3)
        y3 = np.asarray([2] * n3)

        # # mus = [0, -3]
        # cov = np.asarray([[sigma, 0], [0, sigma]])
        # X4 = r.multivariate_normal(mus, cov, size=n1)
        # y4 = np.asarray([1] * n1)

        # X1 = np.concatenate([X1, X2], axis=0)
        # y1 = np.concatenate([y1, y2], axis=0)
        # X3 = np.concatenate([X3, X4], axis=0)
        # y3 = np.concatenate([y3, y4], axis=0)

        return X1, y1, X2, y2, X3, y3

    X1, y1, X2, y2, X3, y3 = get_xy(n=10000)
    # indices1 = np.where(X1[:, 0] < 0)
    # indices2 = np.where(X2[:, 0] < 0)
    # indices3 = np.where(X1[:, 0] >= 0)
    # indices4 = np.where(X2[:, 0] >= 0)
    # new_X1 = np.concatenate([X1[indices1], X2[indices2]], axis=0)
    # # new_y1 = np.concatenate([y1[indices1], y2[indices2]], axis=0)
    # new_y1 = np.zeros((new_X1.shape[0],))
    # new_X2 = np.concatenate([X1[indices3], X2[indices4]], axis=0)
    # # new_y2 = np.concatenate([y1[indices3], y2[indices4]], axis=0)
    # new_y2 = np.ones((new_X2.shape[0],))
    # X1, y1, X2, y2 = new_X1, new_y1, new_X2, new_y2
    if 2 * ratio <= 0 or 2 * ratio >= 1:
        pass
    else:
        # client 1: 90% cluster1, 10 % cluster2, 10 % cluster3
        # client 2: 10% cluster1, 90 % cluster2, 10 % cluster3
        # client 3: 10% cluster1, 10 % cluster2, 90 % cluster3
        train_x1, X1, train_y1, y1 = train_test_split(X1, y1, test_size=2 * ratio, shuffle=True,
                                                      random_state=random_state)  # train set = 1-ratio
        test_x11, test_x12, test_y11, test_y12 = train_test_split(X1, y1, test_size=0.5, shuffle=True,
                                                                  random_state=random_state)  # each test set = 50% of rest data

        train_x2, X2, train_y2, y2 = train_test_split(X2, y2, test_size=2 * ratio, shuffle=True,
                                                      random_state=random_state)
        test_x21, test_x22, test_y21, test_y22 = train_test_split(X2, y2, test_size=0.5, shuffle=True,
                                                                  random_state=random_state)

        train_x3, X3, train_y3, y3 = train_test_split(X3, y3, test_size=2 * ratio, shuffle=True,
                                                      random_state=random_state)
        test_x31, test_x32, test_y31, test_y32 = train_test_split(X3, y3, test_size=0.5, shuffle=True,
                                                                  random_state=random_state)

        X1 = np.concatenate([train_x1, test_x21, test_x31], axis=0)
        # y1 = np.concatenate([train_y1, test_y2], axis=0) # be careful of this
        y1 = np.zeros((X1.shape[0],))

        X2 = np.concatenate([test_x11, train_x2, test_x32], axis=0)
        # y2 = np.concatenate([test_y1, train_y2], axis=0)
        y2 = np.ones((X2.shape[0],))

        X3 = np.concatenate([test_x12, test_x22, train_x3], axis=0)
        y3 = np.ones((X3.shape[0],)) * 2

    is_show = params['is_show']
    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='$G_{11}$')
        p = np.mean(X1, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] - offset, p[1] - offset)
        print(xytext)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
                    ha='center', va='center',  # textcoords='offset points',
                    bbox=dict(facecolor='none', edgecolor='b', pad=1),
                    arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        # angleA : starting angle of the path
        # angleB : ending angle of the path

        ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='$G_{12}$')
        p = np.mean(X2, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))

        ax.scatter(X3[:, 0], X3[:, 1], c=colors[2], marker="o", s=10, alpha=0.3, label='$G_{13}$')
        p = np.mean(X3, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))

        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right', fontsize = 13)
        if params['show_title']:
            plt.title(params['p1'])

        if 'xlim' in kwargs:
            plt.xlim(kwargs['xlim'])
        else:
            plt.xlim([-6, 6])
        if 'ylim' in kwargs:
            plt.ylim(kwargs['ylim'])
        else:
            plt.ylim([-6, 6])
        fontsize = 13
        plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)

        plt.tight_layout()
        if not os.path.exists(params['out_dir']):
            os.makedirs(params['out_dir'])
        f = os.path.join(params['out_dir'], params['p1']+'.png')
        print(f)
        plt.savefig(f, dpi=600, bbox_inches='tight')
        plt.show()

    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    for i, (x, y) in enumerate([(X1, y1), (X2, y2), (X3, y3)]):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=True,
                                                            random_state=random_state)
        clients_train_x.append(train_x)
        clients_train_y.append(train_y)
        clients_test_x.append(test_x)
        clients_test_y.append(test_y)

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    return x, labels


# gaussian3_mix_clusters_per_client({}, random_state=42)


def gaussian3_1client_ylt0(params, random_state=42):
    """
     if params['p1'] == '1client_ylt0':
    # lt0 means all 'y's are larger than 0
    # 2 clusters in R^2
    # 1) client 1 has all data (y>0) from cluster1 and cluster2
    # 1) client 2 has all data (y<=0) from cluster1 and cluster2
    return gaussian3_1client_ylt0(params, random_state=seed)


    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """

    def get_xy(n=10000):

        # client 1
        mus = [-1, 0]
        sigma = 0.1
        n1 = 10000
        cov = np.asarray([[sigma, 0], [0, sigma]])
        # cov = np.asarray([[sigma, sigma], [sigma, sigma]])
        r = np.random.RandomState(random_state)
        X1 = r.multivariate_normal(mus, cov, size=n1)
        y1 = np.asarray([0] * n1)

        # client 2
        mus = [1, 0]
        sigma = 0.1
        n2 = 10000
        cov = np.asarray([[sigma, 0], [0, sigma]])
        # cov = np.asarray([[sigma, -sigma], [-sigma, sigma]])
        X2 = r.multivariate_normal(mus, cov, size=n2)
        y2 = np.asarray([1] * n2)

        # client 3
        mus = [0, 3]
        n3 = 10000
        sigma = 0.1
        cov = np.asarray([[sigma + 0.9, 0], [0, sigma]])
        X3 = r.multivariate_normal(mus, cov, size=n3)
        y3 = np.asarray([2] * n3)

        # # mus = [0, -3]
        # cov = np.asarray([[sigma, 0], [0, sigma]])
        # X4 = r.multivariate_normal(mus, cov, size=n1)
        # y4 = np.asarray([1] * n1)

        # X1 = np.concatenate([X1, X2], axis=0)
        # y1 = np.concatenate([y1, y2], axis=0)
        # X3 = np.concatenate([X3, X4], axis=0)
        # y3 = np.concatenate([y3, y4], axis=0)

        return X1, y1, X2, y2, X3, y3

    X1, y1, X2, y2, X3, y3 = get_xy(n=10000)
    indices11 = np.where(X1[:, 1] < 0)
    indices21 = np.where(X2[:, 1] < 0)
    indices31 = np.where(X3[:, 1] < 3)
    indices12 = np.where(X1[:, 1] >= 0)
    indices22 = np.where(X2[:, 1] >= 0)
    indices32 = np.where(X3[:, 1] >= 3)
    new_X1 = np.concatenate([X1[indices11], X2[indices21], X3[indices31]], axis=0)
    # new_y1 = np.concatenate([y1[indices1], y2[indices2]], axis=0)
    new_y1 = np.zeros((new_X1.shape[0],))
    new_X2 = np.concatenate([X1[indices12], X2[indices22]], axis=0)
    # new_y2 = np.concatenate([y1[indices3], y2[indices4]], axis=0)
    new_y2 = np.ones((new_X2.shape[0],))
    new_X3 = X3[indices32]
    new_y3 = np.ones((new_X3.shape[0],)) * 2
    X1, y1, X2, y2, X3, y3 = new_X1, new_y1, new_X2, new_y2, new_X3, new_y3

    is_show = params['is_show']
    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='Centroid 1')
        p = np.mean(X1, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] - offset, p[1] - offset)
        print(xytext)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
                    ha='center', va='center',  # textcoords='offset points',
                    bbox=dict(facecolor='none', edgecolor='b', pad=1),
                    arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        # angleA : starting angle of the path
        # angleB : ending angle of the path

        ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='Centroid 2')
        p = np.mean(X2, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))

        ax.scatter(X3[:, 0], X3[:, 1], c=colors[2], marker="o", s=10, alpha=0.3, label='Centroid 3')
        p = np.mean(X3, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))

        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right')
        plt.title(params['p1'])
        # # plt.xlim([-2, 15])
        # # plt.ylim([-2, 15])
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
        # # plt.xticks([])
        # # plt.yticks([])
        if not os.path.exists(params['out_dir']):
            os.makedirs(params['out_dir'])
        f = os.path.join(params['out_dir'], params['p1']+'.png')
        plt.savefig(f, dpi=600, bbox_inches='tight')
        plt.show()

    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    for i, (x, y) in enumerate([(X1, y1), (X2, y2), (X3, y3)]):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=True,
                                                            random_state=random_state)
        clients_train_x.append(train_x)
        clients_train_y.append(train_y)
        clients_test_x.append(test_x)
        clients_test_y.append(test_y)

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    return x, labels


# gaussian3_1client_ylt0({}, random_state=42)

def gaussian3_1client_xlt0(params, random_state=42):
    """
     if params['p1'] == '1client_xlt0':
    # lt0 means all 'x's are larger than 0
    # 2 clusters in R^2
    # 1) client 1 has all data (x>0) from cluster1 and cluster2
    # 1) client 2 has all data (x<=0) from cluster1 and cluster2
    return gaussian3_1client_xlt0(params, random_state=seed)

    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """

    def get_xy(n=10000):

        # client 1
        mus = [-1, 0]
        sigma = 0.1
        n1 = 10000
        cov = np.asarray([[sigma, 0], [0, sigma]])
        # cov = np.asarray([[sigma, sigma], [sigma, sigma]])
        r = np.random.RandomState(random_state)
        X1 = r.multivariate_normal(mus, cov, size=n1)
        y1 = np.asarray([0] * n1)

        # client 2
        mus = [1, 0]
        sigma = 0.1
        n2 = 10000
        cov = np.asarray([[sigma, 0], [0, sigma]])
        # cov = np.asarray([[sigma, -sigma], [-sigma, sigma]])
        X2 = r.multivariate_normal(mus, cov, size=n2)
        y2 = np.asarray([1] * n2)

        # client 3
        mus = [0, 3]
        n3 = 10000
        sigma = 0.1
        cov = np.asarray([[sigma + 0.9, 0], [0, sigma]])
        X3 = r.multivariate_normal(mus, cov, size=n3)
        y3 = np.asarray([2] * n3)

        # # mus = [0, -3]
        # cov = np.asarray([[sigma, 0], [0, sigma]])
        # X4 = r.multivariate_normal(mus, cov, size=n1)
        # y4 = np.asarray([1] * n1)

        # X1 = np.concatenate([X1, X2], axis=0)
        # y1 = np.concatenate([y1, y2], axis=0)
        # X3 = np.concatenate([X3, X4], axis=0)
        # y3 = np.concatenate([y3, y4], axis=0)

        return X1, y1, X2, y2, X3, y3

    X1, y1, X2, y2, X3, y3 = get_xy(n=10000)
    indices11 = np.where(X1[:, 0] < 0)
    indices21 = np.where(X2[:, 0] < 0)
    indices31 = np.where(X3[:, 1] < 3)
    indices12 = np.where(X1[:, 0] >= 0)
    indices22 = np.where(X2[:, 0] >= 0)
    indices32 = np.where(X3[:, 1] >= 3)
    new_X1 = np.concatenate([X1[indices11], X2[indices21], X3[indices31]], axis=0)
    # new_y1 = np.concatenate([y1[indices1], y2[indices2]], axis=0)
    new_y1 = np.zeros((new_X1.shape[0],))
    new_X2 = np.concatenate([X1[indices12], X2[indices22]], axis=0)
    # new_y2 = np.concatenate([y1[indices3], y2[indices4]], axis=0)
    new_y2 = np.ones((new_X2.shape[0],))
    new_X3 = X3[indices32]
    new_y3 = np.ones((new_X3.shape[0],)) * 2
    X1, y1, X2, y2, X3, y3 = new_X1, new_y1, new_X2, new_y2, new_X3, new_y3

    is_show = params['is_show']
    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='Centroid 1')
        p = np.mean(X1, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] - offset, p[1] - offset)
        print(xytext)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
                    ha='center', va='center',  # textcoords='offset points',
                    bbox=dict(facecolor='none', edgecolor='b', pad=1),
                    arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        # angleA : starting angle of the path
        # angleB : ending angle of the path

        ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='Centroid 2')
        p = np.mean(X2, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))

        ax.scatter(X3[:, 0], X3[:, 1], c=colors[2], marker="o", s=10, alpha=0.3, label='Centroid 3')
        p = np.mean(X3, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))

        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right')
        plt.title(params['p1'])
        # # plt.xlim([-2, 15])
        # # plt.ylim([-2, 15])
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
        # # plt.xticks([])
        # # plt.yticks([])
        if not os.path.exists(params['out_dir']):
            os.makedirs(params['out_dir'])
        f = os.path.join(params['out_dir'], params['p1']+'.png')
        plt.savefig(f, dpi=600, bbox_inches='tight')
        plt.show()

    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    for i, (x, y) in enumerate([(X1, y1), (X2, y2), (X3, y3)]):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=True,
                                                            random_state=random_state)
        clients_train_x.append(train_x)
        clients_train_y.append(train_y)
        clients_test_x.append(test_x)
        clients_test_y.append(test_y)

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    return x, labels


# gaussian3_1client_xlt0({}, random_state=42)


def gaussian3_1client_1cluster_diff_sigma(params, random_state=42):
    """
    # 2 clusters ((-1,0), (1, 0)) in R^2, each client has one cluster. 2 clusters has no overlaps.
    cluster 1: sigma = 0.5
    cluster 2: sigma = 1
    params['p1'] == '1client_1cluster_diff_sigma':
    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """

    def get_xy(n=10000):

        # client 1
        mus = [-1, 0]
        sigma = 0.1
        n1 = 10000
        cov = np.asarray([[sigma, 0], [0, sigma]])
        # cov = np.asarray([[sigma, sigma], [sigma, sigma]])
        r = np.random.RandomState(random_state)
        X1 = r.multivariate_normal(mus, cov, size=n1)
        y1 = np.asarray([0] * n1)

        # client 2
        mus = [1, 0]
        sigma = 0.2
        n2 = 10000
        cov = np.asarray([[sigma, 0], [0, sigma]])
        # cov = np.asarray([[sigma, -sigma], [-sigma, sigma]])
        X2 = r.multivariate_normal(mus, cov, size=n2)
        y2 = np.asarray([1] * n2)

        # client 3
        mus = [0, 3]
        n3 = 10000
        sigma = 0.3
        cov = np.asarray([[sigma, 0], [0, sigma]])
        X3 = r.multivariate_normal(mus, cov, size=n3)
        y3 = np.asarray([2] * n3)

        # # mus = [0, -3]
        # cov = np.asarray([[sigma, 0], [0, sigma]])
        # X4 = r.multivariate_normal(mus, cov, size=n1)
        # y4 = np.asarray([1] * n1)

        # X1 = np.concatenate([X1, X2], axis=0)
        # y1 = np.concatenate([y1, y2], axis=0)
        # X3 = np.concatenate([X3, X4], axis=0)
        # y3 = np.concatenate([y3, y4], axis=0)

        return X1, y1, X2, y2, X3, y3

    X1, y1, X2, y2, X3, y3 = get_xy(n=10000)
    # indices11 = np.where(X1[:, 1] < 0)
    # indices21 = np.where(X2[:, 1] < 0)
    # indices31 = np.where(X3[:, 1] < 3)
    # indices12 = np.where(X1[:, 1] >= 0)
    # indices22 = np.where(X2[:, 1] >= 0)
    # indices32 = np.where(X3[:, 1] >= 3)
    # new_X1 = np.concatenate([X1[indices11], X2[indices21], X3[indices31]], axis=0)
    # # new_y1 = np.concatenate([y1[indices1], y2[indices2]], axis=0)
    # new_y1 = np.zeros((new_X1.shape[0],))
    # new_X2 = np.concatenate([X1[indices12], X2[indices22]], axis=0)
    # # new_y2 = np.concatenate([y1[indices3], y2[indices4]], axis=0)
    # new_y2 = np.ones((new_X2.shape[0],))
    # new_X3 = X3[indices32]
    # new_y3 = np.ones((new_X3.shape[0],))
    # X1, y1, X2, y2, X3, y3 = new_X1, new_y1, new_X2, new_y2, new_X3, new_y3

    is_show = params['is_show']
    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='Centroid 1')
        p = np.mean(X1, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] - offset, p[1] - offset)
        print(xytext)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
                    ha='center', va='center',  # textcoords='offset points',
                    bbox=dict(facecolor='none', edgecolor='b', pad=1),
                    arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        # angleA : starting angle of the path
        # angleB : ending angle of the path

        ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='Centroid 2')
        p = np.mean(X2, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))

        ax.scatter(X3[:, 0], X3[:, 1], c=colors[2], marker="o", s=10, alpha=0.3, label='Centroid 3')
        p = np.mean(X3, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))

        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right')
        plt.title(params['p1'])
        # # plt.xlim([-2, 15])
        # # plt.ylim([-2, 15])
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
        # # plt.xticks([])
        # # plt.yticks([])
        if not os.path.exists(params['out_dir']):
            os.makedirs(params['out_dir'])
        f = os.path.join(params['out_dir'], params['p1']+'.png')
        plt.savefig(f, dpi=600, bbox_inches='tight')
        plt.show()

    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    for i, (x, y) in enumerate([(X1, y1), (X2, y2), (X3, y3)]):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=True,
                                                            random_state=random_state)
        clients_train_x.append(train_x)
        clients_train_y.append(train_y)
        clients_test_x.append(test_x)
        clients_test_y.append(test_y)

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    return x, labels


# gaussian3_1client_1cluster({})


def gaussian3_diff_sigma_n(args, random_state=42, **kwargs):
    """
    # 2 clusters ((-1,0), (1, 0)) in R^2, each client has one cluster. 2 clusters has no overlaps.
    cluster 1: sigma = 0.1 and n_points = 5000
    cluster 2: sigma = 1    and n_points = 15000
    params['p1'] == 'diff_sigma_n':
    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """
    dataset_detail = args['DATASET']['detail']   # 'n1_100-sigma1_0.1+n2_5000-sigma2_0.2+n3_10000-sigma3_0.3:ratio_0.1:diff_sigma_n'
    p1 = dataset_detail.split(':')
    ratio = float(p1[1].split('_')[1])

    p1_0 = p1[0].split('+')     # 'n1_100-sigma1_0.1, n2_1000-sigma2_0.2, n3_10000-sigma3_0.3
    p1_0_c1 = p1_0[0].split('-')
    n1 = int(p1_0_c1[0].split('_')[1])
    tmp = p1_0_c1[1].split('_')
    sigma1_0, sigma1_1 = float(tmp[1]), float(tmp[2])

    p1_0_c2 = p1_0[1].split('-')
    n2 = int(p1_0_c2[0].split('_')[1])
    tmp = p1_0_c2[1].split('_')
    sigma2_0, sigma2_1 = float(tmp[1]), float(tmp[2])

    p1_0_c3 = p1_0[2].split('-')
    n3 = int(p1_0_c3[0].split('_')[1])
    tmp = p1_0_c3[1].split('_')
    sigma3_0, sigma3_1 = float(tmp[1]), float(tmp[2])

    def get_xy(n=0):

        ############
        # client 1
        mus = [-1, 0]
        # sigma = 0.1
        # n1 = 2000
        cov = np.asarray([[sigma1_0, 0], [0, sigma1_1]])
        # cov = np.asarray([[sigma, sigma], [sigma, sigma]])
        r = np.random.RandomState(random_state)
        X1 = r.multivariate_normal(mus, cov, size=n1)
        y1 = np.asarray([1] * n1)
        # noise 1 
        mus = [-10, 0]
        cov = np.asarray([[0.1, 0], [0, 0.1]])
        X_noise = r.multivariate_normal(mus, cov, size=int(n1*0.01))
        y_noise = np.asarray([11] * X_noise.shape[0])
        X1 = np.concatenate([X1, X_noise], axis=0)
        # y1 = np.concatenate([y1, y_noise], axis=0)
        y1 = np.asarray([1] * X1.shape[0])

        ############
        # client 2
        mus = [1, 0]
        # sigma = 0.2
        # n2 = 5000
        cov = np.asarray([[sigma2_0, 0], [0, sigma2_1]])
        # cov = np.asarray([[sigma, -sigma], [-sigma, sigma]])
        X2 = r.multivariate_normal(mus, cov, size=n2)
        y2 = np.asarray([2] * n2)
        # noise 2 
        mus = [10, 0]
        cov = np.asarray([[0.1, 0], [0, 0.1]])
        X_noise = r.multivariate_normal(mus, cov, size=int(n2*0.01))
        y_noise = np.asarray([22] * X_noise.shape[0])
        X2 = np.concatenate([X2, X_noise], axis=0)
        # y2 = np.concatenate([y2, y_noise], axis=0)
        y2 = np.asarray([2] * X2.shape[0])

        ############
        # client 3
        mus = [0, 3]
        # n3 = 10000
        # sigma = 0.3
        cov = np.asarray([[sigma3_0, 0], [0, sigma3_1]])
        X3 = r.multivariate_normal(mus, cov, size=n3)
        y3 = np.asarray([3] * n3)
         # noise 3
        mus = [0, 10]
        cov = np.asarray([[0.1, 0], [0, 0.1]])
        X_noise = r.multivariate_normal(mus, cov, size=int(n3*0.01))
        y_noise = np.asarray([33] * X_noise.shape[0])
        X3 = np.concatenate([X3, X_noise], axis=0)
        # y3 = np.concatenate([y3, y_noise], axis=0)
        y3 = np.asarray([3] * X3.shape[0])
        # X1 = np.concatenate([X1, X2], axis=0)
        # y1 = np.concatenate([y1, y2], axis=0)
        # X3 = np.concatenate([X3, X4], axis=0)
        # y3 = np.concatenate([y3, y4], axis=0)

        return X1, y1, X2, y2, X3, y3

    X1, y1, X2, y2, X3, y3 = get_xy()
    # X = np.concatenate([X1, X2, X3], axis=0)
    # y = np.concatenate([y1, y2, y3])
    # import collections
    # print(ratio)
    # print(np.mean(X, axis=0), np.std(X, axis=0))
    # print(collections.Counter(y).items())
    # print(np.quantile(X, q = [0, 0.1, 0.3, 0.5, 0.8, 0.9, 0.99], axis=0))
    # print('after')
    if (2* ratio <= 0) or (2* ratio >= 1) or ('centralized' in args['ALGORITHM']['py_name']):
        # centralized_kmeans does not need to consider the ratio.
        pass
    else:
        # client 1: 90% cluster1, 10 % cluster2, 10 % cluster3
        # client 2: 10% cluster1, 90 % cluster2, 10 % cluster3
        # client 3: 10% cluster1, 10 % cluster2, 90 % cluster3
        # random_state = 0
        train_x1, X1, train_y1, y1 = train_test_split(X1, y1, test_size=2 * ratio, shuffle=True, stratify=y1,
                                                      random_state=random_state)  # train set = 1-ratio
        test_x11, test_x12, test_y11, test_y12 = train_test_split(X1, y1, test_size=0.5, shuffle=True, stratify=y1,
                                                                  random_state=random_state)  # each test set = 50% of rest data
        print(train_x1.shape, collections.Counter(train_y1),
              test_x11.shape, collections.Counter(test_y11), test_x12.shape, collections.Counter(test_y12))
        train_x2, X2, train_y2, y2 = train_test_split(X2, y2, test_size=2* ratio, shuffle=True, stratify=y2,
                                                      random_state=random_state)
        test_x21, test_x22, test_y21, test_y22 = train_test_split(X2, y2, test_size=0.5, shuffle=True, stratify=y2,
                                                                  random_state=random_state)
        print(train_x2.shape, collections.Counter(train_y2),
              test_x21.shape, collections.Counter(test_y21), test_x22.shape, collections.Counter(test_y22))

        train_x3, X3, train_y3, y3 = train_test_split(X3, y3, test_size= 2* ratio, shuffle=True, stratify=y3,
                                                      random_state=random_state)
        test_x31, test_x32, test_y31, test_y32 = train_test_split(X3, y3, test_size=0.5, shuffle=True, stratify=y3,
                                                                  random_state=random_state)
        print(train_x3.shape, collections.Counter(train_y3),
              test_x31.shape, collections.Counter(test_y31), test_x32.shape, collections.Counter(test_y32))

        X1 = np.concatenate([train_x1, test_x21, test_x31], axis=0)
        y1 = np.concatenate([train_y1, test_y21, test_y31], axis=0)  # be careful of this
        # y1 = np.zeros((X1.shape[0],))

        X2 = np.concatenate([test_x11, train_x2, test_x32], axis=0)
        y2 = np.concatenate([test_y11, train_y2, test_y32], axis=0)
        # y2 = np.ones((X2.shape[0],))

        X3 = np.concatenate([test_x12, test_x22, train_x3], axis=0)
        y3 = np.concatenate([test_y12, test_y22, train_y3], axis=0)
        # y3 = np.ones((X3.shape[0],)) * 2
        print(X1.shape, collections.Counter(y1),
              X2.shape, collections.Counter(y2), X3.shape, collections.Counter(y3))

    # X = np.concatenate([X1, X2, X3], axis=0)
    # y = np.concatenate([y1, y2, y3])
    # print(ratio)
    # print(np.mean(X, axis=0), np.std(X, axis=0))
    # print(np.quantile(X, q=[0, 0.1, 0.3, 0.5, 0.8, 0.9, 0.99], axis=0))
    # print(collections.Counter(y).items())

    is_show = args['IS_SHOW']
    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='$G_1$')
        p = np.mean(X1, axis=0)
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
        # angleA : starting angle of the path
        # angleB : ending angle of the path

        ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='$G_2$')
        p = np.mean(X2, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))

        ax.scatter(X3[:, 0], X3[:, 1], c=colors[2], marker="o", s=10, alpha=0.3, label='$G_3$')
        p = np.mean(X3, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
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

    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    for i, (x, y) in enumerate([(X1, y1), (X2, y2), (X3, y3)]):
        print(x.shape, collections.Counter(y))
        # # will be a minor issue for small size:
        # e.g., one class is 503 data points, and another one is 505; however, we expect both is 504.
        # train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0, shuffle=True, stratify=y,
        #                                                     random_state=random_state)
        train_x, train_y = x, y
        test_x, test_y = x[:2, :], y[:2]
        clients_train_x.append(train_x)
        clients_train_y.append(train_y)
        clients_test_x.append(test_x)
        clients_test_y.append(test_y)
        print(train_y.shape, collections.Counter(train_y).items())
        # print(test_y.shape, collections.Counter(test_y).items())

    # X = np.concatenate([X1, X2, X3], axis=0)
    # y = np.concatenate([y1, y2, y3])
    # print(ratio)
    # print(np.mean(X, axis=0), np.std(X, axis=0))
    # print(np.quantile(X, q=[0, 0.1, 0.3, 0.5, 0.8, 0.9, 0.99], axis=0))
    # print(collections.Counter(y).items())

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    return x, labels


def gaussian3_1client_xlt0_2(params, random_state=42):
    """
     if params['p1'] == '1client_xlt0':
    # lt0 means all 'x's are larger than 0
    # 2 clusters in R^2
    # 1) client 1 has all data (x>0) from cluster1 and cluster2
    # 1) client 2 has all data (x<=0) from cluster1 and cluster2
    return gaussian3_1client_xlt0(params, random_state=seed)

    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """
    ratio = params['ratio']
    def get_xy(n=10000):

        # client 1
        mus = [-1, 0]
        sigma = 0.1
        n1 = 1000
        cov = np.asarray([[sigma, 0], [0, sigma]])
        # cov = np.asarray([[sigma, sigma], [sigma, sigma]])
        r = np.random.RandomState(random_state)
        X1 = r.multivariate_normal(mus, cov, size=n1)
        y1 = np.asarray([0] * n1)

        # client 2
        mus = [1, 0]
        sigma = 0.1
        n2 = 5000
        cov = np.asarray([[sigma, 0], [0, sigma]])
        # cov = np.asarray([[sigma, -sigma], [-sigma, sigma]])
        X2 = r.multivariate_normal(mus, cov, size=n2)
        y2 = np.asarray([1] * n2)

        mus = [0, 3]
        n3 = 10000
        sigma = 0.1
        cov = np.asarray([[sigma + 0.9, 0], [0, sigma]])
        X3 = r.multivariate_normal(mus, cov, size=n3)
        y3 = np.asarray([2] * n3)

        # # mus = [0, -3]
        # cov = np.asarray([[sigma, 0], [0, sigma]])
        # X4 = r.multivariate_normal(mus, cov, size=n1)
        # y4 = np.asarray([1] * n1)

        # X1 = np.concatenate([X1, X2], axis=0)
        # y1 = np.concatenate([y1, y2], axis=0)
        # X3 = np.concatenate([X3, X4], axis=0)
        # y3 = np.concatenate([y3, y4], axis=0)

        return X1, y1, X2, y2, X3, y3

    X1, y1, X2, y2, X3, y3 = get_xy(n=10000)
    # client 1: 90% cluster1, 10 % cluster2, 10 % cluster3
    # client 2: 10% cluster1, 90 % cluster2, 10 % cluster3
    # client 3: 10% cluster1, 10 % cluster2, 90 % cluster3
    X1, test_x11, y1, test_y11 = train_test_split(X1, y1, test_size=ratio, shuffle=True,
                                                  random_state=random_state)
    train_x1, test_x12, train_y1, test_y12 = train_test_split(X1, y1, test_size=ratio / (1 - ratio), shuffle=True,
                                                              random_state=random_state)

    X2, test_x21, y2, test_y21 = train_test_split(X2, y2, test_size=ratio, shuffle=True,
                                                  random_state=random_state)
    train_x2, test_x22, train_y2, test_y22 = train_test_split(X2, y2, test_size=ratio / (1 - ratio), shuffle=True,
                                                              random_state=random_state)

    X3, test_x31, y3, test_y31 = train_test_split(X3, y3, test_size=ratio, shuffle=True,
                                                  random_state=random_state)
    train_x3, test_x32, train_y3, test_y32 = train_test_split(X3, y3, test_size=ratio / (1 - ratio), shuffle=True,
                                                              random_state=random_state)

    X1 = np.concatenate([train_x1, test_x21, test_x31], axis=0)
    # y1 = np.concatenate([train_y1, test_y2], axis=0) # be careful of this
    y1 = np.zeros((X1.shape[0],))

    X2 = np.concatenate([test_x11, train_x2, test_x32], axis=0)
    # y2 = np.concatenate([test_y1, train_y2], axis=0)
    y2 = np.ones((X2.shape[0],))

    X3 = np.concatenate([test_x12, test_x22, train_x3], axis=0)
    y3 = np.ones((X3.shape[0],)) * 2

    is_show = params['is_show']
    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='Centroid 1')
        p = np.mean(X1, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] - offset, p[1] - offset)
        print(xytext)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
                    ha='center', va='center',  # textcoords='offset points',
                    bbox=dict(facecolor='none', edgecolor='b', pad=1),
                    arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        # angleA : starting angle of the path
        # angleB : ending angle of the path

        ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='Centroid 2')
        p = np.mean(X2, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))

        ax.scatter(X3[:, 0], X3[:, 1], c=colors[2], marker="o", s=10, alpha=0.3, label='Centroid 3')
        p = np.mean(X3, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))

        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right')
        plt.title(params['p1'])
        # # plt.xlim([-2, 15])
        # # plt.ylim([-2, 15])
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
        # # plt.xticks([])
        # # plt.yticks([])
        if not os.path.exists(params['out_dir']):
            os.makedirs(params['out_dir'])
        f = os.path.join(params['out_dir'], params['p1']+'.png')
        plt.savefig(f, dpi=600, bbox_inches='tight')
        plt.show()

    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    for i, (x, y) in enumerate([(X1, y1), (X2, y2), (X3, y3)]):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=True,
                                                            random_state=random_state)
        clients_train_x.append(train_x)
        clients_train_y.append(train_y)
        clients_test_x.append(test_x)
        clients_test_y.append(test_y)

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    return x, labels
