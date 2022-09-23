from pprint import pprint

import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.preprocessing import StandardScaler

from utils.utils_func import timer


class KM_Base:
    def __init__(
            self,
            n_clusters,
            *,
            init_centroids='k-means++',
            true_centroids= None,
            max_iter=300,
            tol=1e-4,
            verbose=0,
            random_state=42,
            params=None,
    ):
        self.n_clusters = n_clusters
        self.init_centroids = init_centroids
        self.true_centroids = true_centroids
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.params = params

    def do_init_centroids(self, X=None):
        if isinstance(self.init_centroids, str):
            if self.init_centroids == 'random':
                # # assumes data is in range 0-1
                # centroids = np.random.rand(self.n_clusters, self.dim)
                # for dummy data
                # centroids = randomly_init_centroid(0, self.n_clusters + 1, self.dim, self.n_clusters, self.random_state)
                centroids = random_initialize_centroids(X, self.n_clusters, self.random_state)
            elif self.init_centroids == 'kmeans++':
                centroids, _ = kmeans_plusplus(X, n_clusters=self.n_clusters, random_state=self.random_state)
            elif self.init_centroids == 'true':
                centroids = self.true_centroids
            else:
                raise NotImplementedError
        elif self.init_centroids.shape == (self.n_clusters, self.dim):
            centroids = self.init_centroids
        else:
            raise NotImplementedError
        return centroids

    @timer
    def fit(self, X, y=None):
        self.is_train = False
        self.n_points, self.dim = X.shape

        # if in 5 consecutive times, the difference between new and old centroids is less than self.tol,
        # then the model converges and we stop the training.
        self.n_consecutive = 0
        means_record = []
        stds_record = []
        self.history = []
        self.n_training_iterations = self.max_iter
        for iteration in range(0, self.max_iter):
            self.n_training_iterations = iteration
            if self.verbose >= 2:
                print(f'iteration: {iteration}')
            if iteration == 0:
                self.initial_centroids = self.do_init_centroids(X)
                # print(f'initialization method: {self.init_centroids}, centers: {centroids}')
                self.centroids = self.initial_centroids
                print(f'initial_centroids: \n{self.centroids}')
                # testing after each iteration
                scores = self.eval(X, y, verbose=self.verbose)
                
                centroids_diff = self.centroids - self.true_centroids
                centroids_update = self.centroids-np.zeros((self.n_clusters, self.dim))  # centroids(t+1) - centroid(t)
                self.history.append({'iteration': iteration, 'centroids': self.centroids, 'scores': scores,
                                     'centroids_update': centroids_update, 'centroids_diff': centroids_diff})
                if self.verbose >=3:
                    pprint(scores)
                continue
            # compute distances
            # computationally efficient
            # differences = np.expand_dims(x, axis=1) - np.expand_dims(centroids, axis=0)
            # sq_dist = np.sum(np.square(differences), axis=2)
            # memory efficient
            sq_dist = np.zeros((self.n_points, self.n_clusters))
            for i in range(self.n_clusters):
                sq_dist[:, i] = np.sum(np.square(X - self.centroids[i, :]), axis=1)

            labels = np.argmin(sq_dist, axis=1)
            # update centroids
            centroids_update = np.zeros((self.n_clusters, self.dim))
            counts = np.zeros((self.n_clusters,))
            for i in range(self.n_clusters):
                mask = np.equal(labels, i)
                size = np.sum(mask)
                counts[i] = size
                if size > 0:
                    # new_centroids[i, :] = np.mean(X[mask], axis=0)
                    update = np.sum(X[mask] - self.centroids[i, :], axis=0)
                    centroids_update[i, :] = update / size
                # if self.reassign_min is not None:
                #     if size < X.shape[0] * self.reassign_min:
                #         to_reassign[i] += 1
                #     else:
                #         to_reassign[i] = 0

            # np.sum(np.square(centroids - (centroids + centroid_updates)), axis=1)
            # print(iteration, centroid_updates, centroids)
            delta = np.sum(np.square(centroids_update))
            if self.verbose>=2:
                print(f'iteration: {iteration}, np.sum(np.square(centroids_update)): {delta}')
            if delta < self.tol:
                if self.n_consecutive >= self.params['n_consecutive']:
                    self.training_iterations = iteration
                    # training finishes in advance
                    break
                else:
                    self.n_consecutive += 1
            else:
                self.n_consecutive = 0

            # centroids = centroids + centroids_update
            self.centroids = self.centroids + centroids_update
            if self.verbose >= 4:
                print(f'server\'s centroids_update: {centroids_update} and n_points per cluster: {counts}')
                print(f'new centroids: {self.centroids}')

            # testing after each iteration
            scores = self.eval(X, y, verbose=self.verbose )
            centroids_diff = self.centroids - self.true_centroids
            self.history.append({'iteration': iteration, 'centroids': self.centroids, 'scores': scores,
                                 'centroids_update': centroids_update, 'centroids_diff': centroids_diff})
            if self.verbose>=3:
                pprint(scores[split])

        # print(sq_dist.shape)
        # print(labels.shape)
        # print(centroids.shape)
        # self.labels_ = labels

        # print(f'Training result: centers: {centroids}')
        self.is_train = False

        return

    def predict(self, X):
        # before predicting, check if you already preprocessed x (e.g., std).
        # memory efficient
        sq_dist = np.zeros((x.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            sq_dist[:, i] = np.sum(np.square(x - self.centroids[i, :]), axis=1)
        labels = np.argmin(sq_dist, axis=1)
        return labels

    def eval(self, X, y):

        y_true = y 
        y_pred = self.predict(X)

        misclustered_X = {} 
        n_misclustering = 0 
        for i, y_p, y_t in enumerate(zip(y_pred, y_true)):
            if y_p == y_t: 
                n_misclustering +=1


        scores = {'n_misclustering': n_misclustering}

        return scores










