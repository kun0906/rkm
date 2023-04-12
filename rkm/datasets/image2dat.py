import collections

import sklearn.preprocessing
from matplotlib.image import imread

from matplotlib import pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import rkm

import numpy as np

from rkm import utils
from rkm.utils import common
import copy


def show_clusters(img2):
    img = np.copy(img2)
    # https://stackoverflow.com/questions/20368413/draw-grid-lines-over-an-image-in-matplotlib
    # Grid lines at these intervals (in pixels)
    # dx and dy can be different
    dx, dy = 50, 50

    img2 = np.zeros(img2.shape, dtype=int)

    # Custom (rgb) grid color
    grid_color = [255, 255, 255]

    # Modify the image to include the grid
    img2[:, ::dy, :] = grid_color
    img2[::dx, :, :] = grid_color

    # img2[0:100, 0:100, :] = [255, 255, 255]  # white for debug

    img2[320:680, 250:600, :] = [0, 50, 0]  # (y, x), red, the first cluster
    img2[375:575, 625:1100, :] = [0, 50, 0]  # (y, x), red, the second cluster

    # show the centers
    img2[520:530, 440:450, :] = [255, 0, 0]  # (y, x), red, the first cluster center
    img2[480:490, 810:820, :] = [255, 0, 0]  # (y, x), red, the second cluster center

    # Show the result
    plt.imshow(img)  # plot original image
    plt.imshow(img2, alpha=0.2)  # add the mask onto the original image with alpha=0.2
    plt.show()


def image2csv(img_file='datasets/two_galaxy_clusters.jpg'):
    """
    https://www.eso.org/public/images/potw1821a/
    Returns
    -------

    """
    img = imread(img_file)
    h, w, d = img.shape  # h, w, depth
    print(h, w, d)
    plt.imshow(img)
    plt.show()

    show_clusters(np.copy(img))

    # https://stackoverflow.com/questions/39554660/np-arrays-being-immutable-assignment-destination-is-read-only

    img3 = np.copy(img)

    # centroids estimated by myself.
    centroids = [[525, 445] + img[525, 445, :].tolist(),
                 [485, 815] + img[485, 815, :].tolist()]
    print(centroids)
    data = []  # (y, x, r, g, b, label)
    cluster1 = []
    cluster2 = []
    noise = []
    for i in range(h):
        for j in range(w):
            if sum(img[i][j]) < 255: #and max(img[i][j]) < 256 // 2:
                # print('ignore, ', i, j, img[i][j])
                img3[i][j] = [0, 0, 0]
                continue

            if 320 <= i <= 680 and 250 <= j <= 600:
                label = 1
                cluster1.append([i, j] + img3[i][j].tolist())
            elif 375 <= i <= 575 and 625 <= j <= 1100:
                label = 2
                cluster2.append([i, j] + img3[i][j].tolist())
            else:
                # if i > 525:
                #     img3[i][j] = [0, 0, 0]
                #     continue
                label = -1
                noise.append([i, j] + img3[i][j].tolist())

            # # only keep noise
            # if label != -1:
            #     img3[i][j] = [0, 0, 0]

            # data format: (y, x, r, g, b)
            data.append([i, j] + img3[i][j].tolist() + [label])

    # Show the result
    plt.imshow(img3[:, :, :], )
    plt.show()

    cluster1 = np.asarray(cluster1)
    cluster2 = np.asarray(cluster2)
    noise = np.asarray(noise)
    centroids = np.asarray(centroids)

    data = np.asarray(data)
    print(collections.Counter(data[:, -1]))

    out_file = img_file + '.csv'
    np.savetxt(out_file, data, fmt='%d', delimiter=',')

    res = {'data': data, 'shape': img.shape, 'cluster1': cluster1, 'cluster2': cluster2, 'noise': noise,
           'centroids': centroids}
    common.dump(res, img_file + '.dat')


def two_galaxy_clusters_n_backup(args={'DATASET':{'detail':'n:1000|two_galaxy_clusters'}, 'data_file':'datasets'},
                        random_state=42, **kwargs):
    """

    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """
    # n:1000|two_galaxy_clusters
    dataset_detail = args['DATASET']['detail']
    tmp = dataset_detail.split('|')

    p = int(tmp[0].split(':')[1])
    r = np.random.RandomState(random_state)

    def get_xy(p):
        # from numpy import genfromtxt
        # # csv_file = 'datasets/two_galaxy_clusters.jpg.csv'
        # # my_data = genfromtxt(csv_file, delimiter=',')
        in_file = 'datasets/two_galaxy_clusters.jpg.dat'
        res = common.load(in_file)
        shape = res['shape']
        print(f'{in_file}, shape(h,w,d):{shape}')
        print(len(res['cluster1']), len(res['cluster2']), len(res['noise']))

        ############
        # cluster 1:
        n0 = p
        # X0 = res['cluster1'][:, :2] # only coordinate
        X0 = res['cluster1']
        # indics = r.randint(0, len(X0), size=n0) # has duplicates
        indices = r.choice(range(len(X0)), size=n0, replace=False)
        X0 = X0[indices][:, :2]
        print(len(set(indices)))
        y0 = np.asarray(['c1'] * n0)

        ############
        # cluster 2:
        n1 = p
        X1 = res['cluster2']
        indices = r.choice(range(len(X1)), size=n1, replace=False)
        X1 = X1[indices][:, :2]
        print(len(set(indices)))
        y1 = np.asarray(['c2'] * n1)

        ############
        # noise
        n_noise = int(np.round((n0+n1)*0.1))
        X_noise = res['noise'][:, :2]
        indices = r.choice(range(len(X_noise)), size=n_noise, replace=False)
        X_noise = X_noise[indices]
        print(len(set(indices)))
        y_noise = np.asarray(['noise'] * n_noise)

        true_centroids = res['centroids'][:, :2]

        init_centroids = tuple([copy.deepcopy(X0), copy.deepcopy(X1)])
        X = np.concatenate([X0, X1, X_noise], axis=0)
        y = np.concatenate([y0, y1, y_noise])

        # is_show = args['IS_SHOW']
        is_show = False
        if is_show:
            s = {}
            for i in range(len(X)):
                v = X[i]
                key = (v[0], v[1])  # the coordinates in image (h, w)
                s[key] = v[2:5]     # (r, g, b)

            h, w, d = shape
            img = np.zeros(shape, dtype=int)
            for i in range(h):
                for j in range(w):
                    key = (i, j)
                    if key in s.keys():
                        img[i][j] = s[key]
                        # print(i, j, s[key])

            # show data
            for i in range(len(X0)):
                _h, _w = X0[i][0], X0[i][1]
                plt.scatter(_w, _h, marker="^", s=10, linewidths=3, color="g", zorder=10)
            for i in range(len(X1)):
                _h, _w = X1[i][0], X1[i][1]
                plt.scatter(_w, _h, marker="v", s=10, linewidths=3, color="b", zorder=10)

            # Show noise
            for i in range(len(X_noise)):
                _h, _w = X_noise[i][0], X_noise[i][1]
                plt.scatter(_w, _h, marker="*", s=10, linewidths=3, color="r", zorder=10)

            # Show the result
            for i in range(len(true_centroids)):
                _h,_w = true_centroids[i][0], true_centroids[i][1]
                initial = np.mean(init_centroids[i][:2], axis=0)
                _hi, _wi = initial[0], initial[1]
                plt.scatter(_w, _h, marker="*", s=30, linewidths=3, color="r", zorder=10)
                plt.scatter(_wi, _hi, marker="o", s=30, linewidths=3, color="w", zorder=10)

            # plot the results
            f = args['data_file'] + '.png'
            print(f)
            plt.title(f'C1:{len(X0)}, C2:{len(X1)}, noise:{len(X_noise)}\n{f}')
            plt.tight_layout()
            plt.imshow(img)
            plt.savefig(f, dpi=600, bbox_inches='tight')
            plt.show()

        #
        # # obtain initial centroids after mixing the data, p = 0.1, 10%
        # X0, X00, y0, y00 = train_test_split(X0, y0, test_size=p, shuffle=True, random_state=random_state)
        # X1, X11, y1, y11 = train_test_split(X1, y1, test_size=p, shuffle=True, random_state=random_state)
        # X2, X22, y2, y22 = train_test_split(X2, y2, test_size=p, shuffle=True, random_state=random_state)
        # # Mix them togather
        # X0 = np.concatenate([X0, X11], axis=0)
        # y0 = np.concatenate([y0, y11], axis=0)
        # X1 = np.concatenate([X1, X22], axis=0)
        # y1 = np.concatenate([y1, y22], axis=0)
        # X2 = np.concatenate([X2, X00], axis=0)
        # y2 = np.concatenate([y2, y00], axis=0)
        #
        # init_centroids = tuple([copy.deepcopy(X0), copy.deepcopy(X1), copy.deepcopy(X2)])
        # # if args['ALGORITHM']['py_name'] == 'kmeans':
        # #     init_centroids[0] = np.mean(X0, axis=0)
        # #     init_centroids[1] = np.mean(X1, axis=0)
        # #     init_centroids[2] = np.mean(X2, axis=0)
        # # elif args['ALGORITHM']['py_name']== 'kmedian':
        # #     init_centroids[0] = np.median(X0, axis=0)
        # #     init_centroids[1] = np.median(X1, axis=0)
        # #     init_centroids[2] = np.median(X2, axis=0)
        # # else:
        # #     raise NotImplementedError(args['ALGORITHM']['py_name'])

        delta_X = p

        return X, y, true_centroids, init_centroids, delta_X

    X, y, true_centroids, init_centroids, delta_X = get_xy(p)

    return {'X': X, 'y': y, 'true_centroids': true_centroids, 'init_centroids': init_centroids, 'delta_X': delta_X}




def two_galaxy_clusters_n(args={'DATASET':{'detail':'n:1000|two_galaxy_clusters'}, 'data_file':'datasets'},
                        random_state=42, **kwargs):
    """

    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """
    # n:1000|two_galaxy_clusters
    dataset_detail = args['DATASET']['detail']
    tmp = dataset_detail.split('|')

    p = int(tmp[0].split(':')[1])
    r = np.random.RandomState(random_state)

    def get_xy(p):
        # from numpy import genfromtxt
        # # csv_file = 'datasets/two_galaxy_clusters.jpg.csv'
        # # my_data = genfromtxt(csv_file, delimiter=',')
        in_file = 'datasets/two_galaxy_clusters.jpg.dat'
        res = common.load(in_file)
        shape = res['shape']
        print(f'{in_file}, shape(h,w,d):{shape}')
        print(len(res['cluster1']), len(res['cluster2']), len(res['noise']))

        data = res['data']
        n = len(data)
        n0 = int(0.2*n)
        indices = r.choice(range(n), size=n0, replace=False)
        X = data[indices][:, :2]
        print(len(set(indices)))
        y = np.asarray(['c1'] * n0)

        true_centroids = X[:2]
        init_centroids = X[:2]

        delta_X = p

        return X, y, true_centroids, init_centroids, delta_X

    X, y, true_centroids, init_centroids, delta_X = get_xy(p)

    return {'X': X, 'y': y, 'true_centroids': true_centroids, 'init_centroids': init_centroids, 'delta_X': delta_X}


def two_galaxy_clusters_p(args={'DATASET':{'detail':'p:0.49|two_galaxy_clusters'}, 'data_file':'datasets'},
                        random_state=42, **kwargs):
    """

    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """
    # n:1000|two_galaxy_clusters
    dataset_detail = args['DATASET']['detail']
    tmp = dataset_detail.split('|')

    p = float(tmp[0].split(':')[1])
    r = np.random.RandomState(random_state)

    def get_xy(p):
        # from numpy import genfromtxt
        # # csv_file = 'datasets/two_galaxy_clusters.jpg.csv'
        # # my_data = genfromtxt(csv_file, delimiter=',')
        in_file = 'datasets/two_galaxy_clusters.jpg.dat'
        res = common.load(in_file)
        shape = res['shape']
        print(f'{in_file}, shape(h,w,d):{shape}')
        print(len(res['cluster1']), len(res['cluster2']), len(res['noise']))

        ############
        # cluster 1:
        n0 = 7000
        X0 = res['cluster1']
        # indics = r.randint(0, len(X0), size=n0) # has duplicates
        indices = r.choice(range(len(X0)), size=n0, replace=False)
        X0 = X0[indices]
        print(len(set(indices)))
        y0 = np.asarray(['c1'] * n0)

        ############
        # cluster 2:
        n1 = 7000
        X1 = res['cluster2']
        indices = r.choice(range(len(X1)), size=n1, replace=False)
        X1 = X1[indices]
        print(len(set(indices)))
        y1 = np.asarray(['c2'] * n1)

        ############
        # noise
        n_noise = int(np.round((n0+n1)*p))
        X_noise = res['noise']
        indices = r.choice(range(len(X_noise)), size=n_noise, replace=False)
        X_noise = X_noise[indices]
        print(len(set(indices)))
        y_noise = np.asarray(['noise'] * n_noise)

        X = np.concatenate([X0, X1, X_noise], axis=0)
        y = np.concatenate([y0, y1, y_noise])
        true_centroids = res['centroids']
        init_centroids = tuple([copy.deepcopy(X0), copy.deepcopy(X1)])

        # is_normalize = True
        # if is_normalize:
        #     std = sklearn.preprocessing.StandardScaler()
        #     std.fit(X)
        #     X = std.transform(X)
        #     true_centroids = std.transform(true_centroids)
        #     init_centroids = tuple([std.transform(copy.deepcopy(X0)), std.transform(copy.deepcopy(X1))])

        # is_show = args['IS_SHOW']
        is_show = False
        if is_show:
            s = {}
            for i in range(len(X)):
                v = X[i]
                key = (v[0], v[1])  # the coordinates in image (h, w)
                s[key] = v[2:5]     # (r, g, b)

            h, w, d = shape
            img = np.zeros(shape, dtype=int)
            for i in range(h):
                for j in range(w):
                    key = (i, j)
                    if key in s.keys():
                        img[i][j] = s[key]
                        # print(i, j, s[key])

            # Show the result
            for i in range(len(true_centroids)):
                _h,_w = true_centroids[i][0], true_centroids[i][1]
                initial = np.mean(init_centroids[i][:2], axis=0)
                _hi, _wi = initial[0], initial[1]
                plt.scatter(_w, _h, marker="*", s=30, linewidths=3, color="r", zorder=10)
                plt.scatter(_wi, _hi, marker="o", s=30, linewidths=3, color="w", zorder=10)

            # plot the results
            plt.title(f'C1:{len(X0)}, C2:{len(X1)}, noise:{len(X_noise)}')
            plt.imshow(img)
            f = args['data_file'] + '.png'
            print(f)
            plt.savefig(f, dpi=600, bbox_inches='tight')
            plt.show()

        #
        # # obtain initial centroids after mixing the data, p = 0.1, 10%
        # X0, X00, y0, y00 = train_test_split(X0, y0, test_size=p, shuffle=True, random_state=random_state)
        # X1, X11, y1, y11 = train_test_split(X1, y1, test_size=p, shuffle=True, random_state=random_state)
        # X2, X22, y2, y22 = train_test_split(X2, y2, test_size=p, shuffle=True, random_state=random_state)
        # # Mix them togather
        # X0 = np.concatenate([X0, X11], axis=0)
        # y0 = np.concatenate([y0, y11], axis=0)
        # X1 = np.concatenate([X1, X22], axis=0)
        # y1 = np.concatenate([y1, y22], axis=0)
        # X2 = np.concatenate([X2, X00], axis=0)
        # y2 = np.concatenate([y2, y00], axis=0)
        #
        # init_centroids = tuple([copy.deepcopy(X0), copy.deepcopy(X1), copy.deepcopy(X2)])
        # # if args['ALGORITHM']['py_name'] == 'kmeans':
        # #     init_centroids[0] = np.mean(X0, axis=0)
        # #     init_centroids[1] = np.mean(X1, axis=0)
        # #     init_centroids[2] = np.mean(X2, axis=0)
        # # elif args['ALGORITHM']['py_name']== 'kmedian':
        # #     init_centroids[0] = np.median(X0, axis=0)
        # #     init_centroids[1] = np.median(X1, axis=0)
        # #     init_centroids[2] = np.median(X2, axis=0)
        # # else:
        # #     raise NotImplementedError(args['ALGORITHM']['py_name'])

        delta_X = p

        return X, y, true_centroids, init_centroids, delta_X

    X, y, true_centroids, init_centroids, delta_X = get_xy(p)

    return {'X': X, 'y': y, 'true_centroids': true_centroids, 'init_centroids': init_centroids, 'delta_X': delta_X}

if __name__ == '__main__':
    # # generate data
    image2csv()
    # two_galaxy_clusters_n()
    # two_galaxy_clusters_p()
