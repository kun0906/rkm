"""

"""
# Email: kun.bj@outllok.com
import collections
import copy
from pprint import pprint

import numpy as np

from rkm.cluster._base import KMBase
from rkm.utils.common import timer
from sklearn.cluster import SpectralClustering

class My_SpectralClustering(KMBase, SpectralClustering):
	def __init__(
			self,
			n_clusters,
			*,
			init_centroids= None,
			true_centroids=None,
			max_iter=0,
			tol=0,
			verbose=0,
			random_state=42,
			n_consecutive=0,
			median_method = None,
			params=None,
	):
		self.n_clusters = n_clusters
		# self.init_centroids = init_centroids
		self.true_centroids = true_centroids
		# self.max_iter = max_iter
		# self.tol = tol
		self.verbose = verbose
		self.random_state = random_state
		# self.n_consecutive = n_consecutive
		self.params = params

	@timer
	def fit(self, X, y=None):
		self.history = []
		self.clustering = SpectralClustering(n_clusters=self.n_clusters, assign_labels = 'discretize',random_state = self.random_state)
		self.clustering.fit(X)

		labels = self.clustering.labels_
		self.centroids = np.zeros((self.n_clusters, X.shape[1]))
		for i, v in enumerate(sorted(set(labels))):
			mask = labels == v
			self.centroids[i] = np.mean(X[mask])

		self.centroids = self.align_centroids(self.centroids, self.true_centroids)

		# testing after each iteration
		scores = self.eval(X, y)
		centroids_diff = self.centroids - self.true_centroids
		self.history.append({'iteration': 0, 'centroids': self.centroids, 'scores': copy.deepcopy(scores),
							 'centroids_update': 0, 'centroids_diff': centroids_diff})
		if self.verbose >= 20:
			pprint(scores)

		return

	def predict(self, X):
		# before predicting, check if you already preprocessed x (e.g., std).
		# memory efficient
		sq_dist = np.zeros((X.shape[0], self.n_clusters))
		for i in range(self.n_clusters):
			sq_dist[:, i] = np.sum(np.square(X - self.centroids[i, :]), axis=1)
		labels = np.argmin(sq_dist, axis=1)
		return labels