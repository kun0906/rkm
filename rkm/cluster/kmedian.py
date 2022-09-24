"""

"""
# Email: kun.bj@outllok.com
import copy
from pprint import pprint

import numpy as np

from rkm.cluster._base import KMBase
from rkm.utils.common import timer


class KMedian(KMBase):
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
			n_consecutive=5,
			params=None,
	):
		self.n_clusters = n_clusters
		self.init_centroids = init_centroids
		self.true_centroids = true_centroids
		self.max_iter = max_iter
		self.tol = tol
		self.verbose = verbose
		self.random_state = random_state
		self.n_consecutive = n_consecutive
		self.params = params

	@timer
	def fit(self, X, y=None):
		"""
		if in 5 consecutive times, the difference between new and old centroids is less than self.tol,
		then the model converges and we stop the training.

		Parameters
		----------
		X
		y

		Returns
		-------

		"""
		self.is_train = False
		self.n_points, self.dim = X.shape
		self.history = []
		self.n_training_iterations = self.max_iter
		n_consecutive = 0

		for iteration in range(0, self.max_iter + 1):
			self.n_training_iterations = iteration
			if self.verbose >= 2:
				print(f'iteration: {iteration}')
			if iteration == 0:
				# Use the init_centroids from __init__()
				self.init_centroids = self.do_init_centroids(None)
				self.init_centroids = self.align_centroids(self.init_centroids, self.true_centroids)
				# print(f'initialization method: {self.init_centroids}, centers: {centroids}')
				self.centroids = self.init_centroids
				if self.verbose >=2: print(f'init_centroids: \n{self.centroids}')
				# evaluate the model after each iteration
				scores = self.eval(X, y)

				centroids_diff = self.centroids - self.true_centroids
				# centroids(t+1) - centroid(t)
				centroids_update = self.centroids - np.zeros((self.n_clusters, self.dim))
				self.history.append({'iteration': iteration, 'centroids': self.centroids, 'scores': copy.deepcopy(scores),
				                     'centroids_update': centroids_update, 'centroids_diff': centroids_diff})
				if self.verbose >= 5:
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

			# reassign each data point to the closet centroid
			new_centroids = np.zeros((self.n_clusters, self.dim))
			counts = np.zeros((self.n_clusters,))
			for i in range(self.n_clusters):
				mask = np.equal(labels, i)
				size = np.sum(mask)
				counts[i] = size
				if size > 0:
					new_centroids[i] = np.median(X[mask], axis=0)
					# new_centroids[i, :] = np.mean(X[mask], axis=0)

			# delta is the difference of medians
			centroids_update = new_centroids - self.centroids
			delta = np.sum(np.square(centroids_update))
			# print(delta, np.sum(np.square(centroids_update)))
			if self.verbose >= 2:
				print(f'iteration: {iteration}, np.sum(np.square(centroids_update)): {delta}')
			if delta < self.tol:
				if n_consecutive >= self.params['n_consecutive']:
					self.n_training_iterations = iteration
					# training finishes in advance
					break
				else:
					n_consecutive += 1
			else:
				n_consecutive = 0

			# centroids = centroids + centroids_update
			self.centroids = new_centroids
			self.centroids = self.align_centroids(self.centroids, self.true_centroids)
			if self.verbose >= 4:
				print(f'centroids_update: {centroids_update} and n_points per cluster: {counts}')
				print(f'new centroids: {self.centroids}')

			# testing after each iteration
			scores = self.eval(X, y)
			centroids_diff = self.centroids - self.true_centroids
			self.history.append({'iteration': iteration, 'centroids': self.centroids, 'scores': copy.deepcopy(scores),
			                     'centroids_update': centroids_update, 'centroids_diff': centroids_diff})
			if self.verbose >= 5:
				pprint(scores)

		return
