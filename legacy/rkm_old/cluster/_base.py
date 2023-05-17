"""


"""
# Email: kun.bj@outllok.com
import collections
import copy

import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import confusion_matrix

from rkm.utils.common import timer
import itertools

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

	def do_init_centroids(self, X=None, y=None, with_noise=True):
		if isinstance(self.init_centroids, str):
			if self.init_centroids == 'random':
				r = np.random.RandomState(seed=self.random_state)
				if with_noise:
					n = X.shape[0]
					indices = r.choice(range(n), self.n_clusters, replace=False)
					centroids = X[indices,:]
				else:
					X1 = X[y != 'noise']
					indices = r.choice(range(X1.shape[0]), self.n_clusters, replace=False)
					centroids = X1[indices, :]
			elif self.init_centroids == 'kmeans++':
				if with_noise:
					centroids, _ = kmeans_plusplus(X, n_clusters=self.n_clusters, random_state=self.random_state)
				else:
					X1 = X[y != 'noise']
					centroids, _ = kmeans_plusplus(X1, n_clusters=self.n_clusters, random_state=self.random_state)
			elif self.init_centroids == 'true':
				centroids = self.true_centroids
			elif self.init_centroids == 'noise':
				r = np.random.RandomState(seed=self.random_state)
				mask = y == 'noise'
				data = X[mask]
				n = data.shape[0]
				indices = r.choice(range(n), self.n_clusters, replace=False)
				centroids = data[indices, :]
			else:
				raise NotImplementedError
		# elif self.init_centroids.shape == (self.n_clusters, self.dim):
		# 	centroids = self.init_centroids
		elif len(self.init_centroids) == self.n_clusters:
			centroids = np.zeros((self.n_clusters, self.dim))
			for i, x in enumerate(self.init_centroids):
				if self.params['ALGORITHM']['py_name'] == 'kmeans':
					centroids[i] = np.mean(x, axis=0)
				elif self.params['ALGORITHM']['py_name'] == 'kmedian':
					centroids[i] = np.median(x, axis=0)
				elif self.params['ALGORITHM']['py_name'] == 'kmedian_l1':
					centroids[i] = np.median(x, axis=0)
				else:
					raise ValueError(self.params['ALGORITHM']['py_name'])
		else:
			raise NotImplementedError
		return centroids

	@timer
	def align_centroids(self, centroids, true_centroids):
		c1 = copy.deepcopy(true_centroids)
		# check which point is close to which true centroids.
		min_d = np.inf
		for c in list(itertools.permutations(centroids)):
			d = np.sum(np.sum(np.square(c-c1), axis=1), axis=0)
			if d < min_d:
				min_d = d
				best_centroids = np.asarray(copy.deepcopy(c))
		return best_centroids


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

		if self.verbose >= 100:
			from rkm.vis.visualize import plot_centroids
			plot_centroids(X, y_pred, self.init_centroids, self.centroids, params=self.params, self=self) # for debugging
		n, d = X.shape

		n_misclustered = 0
		_X = []
		_y = []
		label2int = {'c1':0, 'c2':1, 'c3':2, 'c4':3, 'c5':4, 'c6':5, 'c7':6, 'c8':7, 'c9':8, 'c10':9,
					 'noise1': 2000, 'noise2': 2100, 'noise3': 2200, 'noise4': 2300, 'noise5': 2400, 'noise6': 2500,
					 'noise7': 2600, 'noise8': 2700, 'noise9': 2800, 'noise10': 2900,
					 'noise': 3000}
		for i in range(self.n_clusters+10):
			key = f'c{i}'
			if key not in label2int:
				label2int[key] = i

		misclustered= {}
		n_noise = 0
		for i, (y_p, y_t) in enumerate(zip(y_pred, y_true)):
			if y_t == 'noise':
				# noise point
				n_noise +=1
				continue
			# if y_t == 'c1':
			# 	y_t = 0
			# elif y_t == 'c2':
			# 	y_t = 1
			# else:
			# 	y_t = 2
			y_t = label2int[y_t]
			if y_p != y_t:
				n_misclustered += 1
				if y[i] not in misclustered:
					misclustered[y[i]] = [X[i]]
				else:
					misclustered[y[i]].append(X[i])

		misclustered_error = n_misclustered / (n - n_noise)
		for key in misclustered.keys():
			misclustered[key] = np.asarray(misclustered[key])

		# Average: 100%
		# centroids = self.align_centroids(self.centroids, self.true_centroids)
		centroids = copy.deepcopy(self.centroids)  # when you plot the difference between it with ground-truth, you need to align the centroids.

		cm = confusion_matrix([label2int[v] for v in y_true], y_pred)
		if self.verbose >= 200:
			s = [f'{key}:{len(vs)}' for key, vs in misclustered.items()]
			print(f'misclustered_error: {misclustered_error}, n (excluded noise): {(n - n_noise)}, misclusterd: {s}')
			print(f'centroids: {centroids}, \ntrue: {self.true_centroids}')
			print(f'confusion_matrix: \n{cm}')

		# we should align the centroids to the order of true_centroids first.
		# average difference
		centroid_diff = np.sum(np.sum(np.square(centroids - self.true_centroids), axis=1), axis=0)/self.n_clusters
		# max centroid difference
		max_centroid_diff = max(np.sum(np.square(centroids - self.true_centroids), axis=1))
		classes = set(y_true)
		n_clusters = len(classes)
		if 'noise' in classes:
			n_clusters = n_clusters - 1
		else:
			pass
		if self.verbose >= 5:
			print(f'self.n_clusters ({self.n_clusters}) == n_clusters ({n_clusters}): {self.n_clusters==self.n_clusters}')
		# centroids = np.zeros((n_clusters, d))
		# centroids = copy.deepcopy(self.centroids)
		for i, l in enumerate(sorted(set(y_true))):
			mask = list(y_pred == label2int[l])
			if sum(mask) > 1:
				# print(sum(mask), flush=True)
				if self.params['ALGORITHM']['py_name'] == 'kmeans':
					centroids[i] = np.mean(X[mask], axis=0)
				elif self.params['ALGORITHM']['py_name'] == 'kmedian':
					centroids[i] = np.median(X[mask], axis=0)
				elif self.params['ALGORITHM']['py_name'] == 'kmedian_l1':
					centroids[i] = np.median(X[mask], axis=0)
				elif self.params['ALGORITHM']['py_name'] == 'my_spectralclustering':
					centroids[i] = np.mean(X[mask], axis=0)
				else:
					raise ValueError(self.params['ALGORITHM']['py_name'])
			elif sum(mask) == 1:
				centroids[i] = X[mask]
			else:
				# use the previous centroids
				pass
		if self.verbose >=10:
			print(centroids, collections.Counter(y_pred))

		centroid_diff2 = np.sum(np.sum(np.square(centroids - self.true_centroids), axis=1),
		                        axis=0) / self.n_clusters

		if self.verbose >=3:
			print(f'centroid_diff: {centroid_diff}')
		scores = {'misclustered_error': misclustered_error,
				  # 'misclustered': misclustered,
		          'centroid_diff': centroid_diff,   'centroid_diff2': centroid_diff2,
				  'max_centroid_diff': max_centroid_diff,
		          # 'cm':cm,
		          }

		return scores
