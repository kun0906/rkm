"""


"""
# Email: kun.bj@outllok.com

import numpy as np
from sklearn.cluster import kmeans_plusplus

from rkm.utils.common import timer


class KMBase:
	def __init__(
			self,
			n_clusters,
			*,
			init_centroids='k-means++',
			true_centroids=None,
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
				r = np.random.RandomState(seed=self.random_state)
				centroids = r.choice(X, self.n_clusters, replace=False)
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

	def align_centroids(self, centroids, true_centroids):
		# check which point is close to which true centroids.
		if np.sum(np.square(centroids[0] - true_centroids[0])) < np.sum(
				np.square(centroids[1] - true_centroids[0])):
			pass
		else:
			centroids[0, :], centroids[1, :] = centroids[1, :], centroids[0,:]

		return centroids


	@timer
	def fit(self, X, y=None):
		pass

	def predict(self, X):
		# before predicting, check if you already preprocessed x (e.g., std).
		# memory efficient
		sq_dist = np.zeros((X.shape[0], self.n_clusters))
		for i in range(self.n_clusters):
			sq_dist[:, i] = np.sum(np.square(X - self.centroids[i, :]), axis=1)
		labels = np.argmin(sq_dist, axis=1)
		return labels

	def eval(self, X, y):

		y_true = y
		y_pred = self.predict(X)

		# from rkm.vis.visualize import plot_centroids
		# plot_centroids(X, y_pred, self.init_centroids) # for debugging

		n_misclustered = 0
		_X = []
		_y = []
		misclustered= {}
		n_noise = 0
		for i, (y_p, y_t) in enumerate(zip(y_pred, y_true)):
			if y_t == 2:
				# noise point
				n_noise +=1
				continue
			if y_t == -1:
				y_t = 0
			if y_p != y_t:
				n_misclustered += 1
				if y[i] not in misclustered:
					misclustered[y[i]] = [X[i]]
				else:
					misclustered[y[i]].append(X[i])

		misclustered_error = n_misclustered / (X.shape[0] - n_noise)
		for key in misclustered.keys():
			misclustered[key] = np.asarray(misclustered[key])

		if self.verbose >= 3:
			s = [f'{key}:{len(vs)}' for key, vs in misclustered.items()]
			print(f'misclustered_error: {misclustered_error}, n (excluded noise): {(X.shape[0] - n_noise)}, misclusterd: {s}')

		centroid_diff = np.sum(np.sum(np.square(self.centroids - self.true_centroids), axis=1), axis=0)/self.n_clusters
		if self.verbose >=3:
			print(f'centroid_diff: {centroid_diff}')
		scores = {'misclustered_error': misclustered_error,
		          'misclustered': misclustered, 'centroid_diff': centroid_diff}

		return scores
