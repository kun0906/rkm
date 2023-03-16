""" Run this main file for a single experiment

	Run instruction:
	$pwd
	$rkm/rkm
	$PYTHONPATH='..' python3 main_single.py
"""
# Email: kun.bj@outllok.com
import os.path
from pprint import pprint

from rkm.cluster.spectral_clustering import My_SpectralClustering
from rkm.datasets.dataset import generate_dataset
from rkm import config
from rkm.cluster.kmeans import KMeans
from rkm.cluster.kmedian import KMedian
from rkm.cluster.kmedian_l1 import KMedian_L1
from rkm.utils.common import dump, load


class Framework:

	def __init__(self, args):
		self.args = args

	def run(self):
		# Generate different data and initial centroids based the given DATA_SEED.
		data_file = generate_dataset(self.args)
		if self.args['is_gen_data']: return
		self.data = load(data_file)
		X, y, init_centroids, true_centroids = self.data['X'], self.data['y'], \
		                                       self.data['init_centroids'], self.data['true_centroids']
		delta_X = self.data['delta_X']

		if self.args['ALGORITHM']['init_method'] == 'random':
			init_centroids = 'random'
		elif self.args['ALGORITHM']['init_method'] == 'kmeans++':
			init_centroids = 'kmeans++'
		elif self.args['ALGORITHM']['init_method'] == 'omniscient':
			init_centroids = init_centroids     # use the omniscient centroids
		elif self.args['ALGORITHM']['init_method'] == None:
			init_centroids = None     # use the omniscient centroids
		else:
			raise NotImplementedError(self.args['ALGORITHM']['py_name'])
		ALG2PY = {'kmeans': KMeans,
		          'kmedian': KMedian,
		          'kmedian_l1': KMedian_L1,
		          'kmedian_tukey': KMedian,
				  'my_spectralclustering': My_SpectralClustering,
		          }
		if self.args['ALGORITHM']['py_name'] == 'kmeans':
			self.model = ALG2PY[self.args['ALGORITHM']['py_name']](n_clusters=self.args['N_CLUSTERS'],
			                                                       init_centroids=init_centroids,
			                                                       true_centroids=true_centroids,
			                                                       max_iter=self.args['MAX_ITERATION'],
			                                                       tol=1e-4,
			                                                       verbose=self.args['VERBOSE'],
			                                                       random_state=self.args['SEED'],  # model seed
			                                                       n_consecutive=self.args['n_consecutive'],
			                                                       params=self.args)
		else:
			if self.args['ALGORITHM']['py_name'] == 'kmedian':
				median_method = 'median'
			elif self.args['ALGORITHM']['py_name'] == 'kmedian_l1':
				median_method = 'median_l1'
			elif self.args['ALGORITHM']['py_name'] == 'kmedian_tukey':
				median_method = 'tukey_median'
			elif self.args['ALGORITHM']['py_name'] == 'my_spectralclustering':
				median_method = None
			else:
				raise NotImplementedError(self.args)
			self.model = ALG2PY[self.args['ALGORITHM']['py_name']](n_clusters=self.args['N_CLUSTERS'],
		                                                       init_centroids=init_centroids,
		                                                       true_centroids=true_centroids,
		                                                       max_iter=self.args['MAX_ITERATION'],
		                                                       tol=1e-4,
		                                                       verbose=self.args['VERBOSE'],
		                                                       random_state=self.args['SEED'],  # model seed
		                                                       n_consecutive=self.args['n_consecutive'],
		                                                       median_method = median_method,
		                                                       params=self.args)

		self.model.fit(X, y)

		scores = self.model.eval(X, y)
		self.history = {'scores': scores, 'delta_X': delta_X, 'history': self.model.history, 'data': None}

		# # save results
		# out_file = os.path.join(self.args['OUT_DIR'], 'history.dat')
		# dump(self.history, out_file)
		_dir = self.args['OUT_DIR']
		# if len(os.listdir(_dir)) == 0:
		# 	os.rmdir(_dir)
	def vis(self):
		pass


def main(config_file='config.yaml'):
	"""

	Parameters
	----------
	config_file

	Returns
	-------

	"""

	# args = config.parser(config_file)
	#
	# X, y = datasets.gen_data(args)
	#
	#
	# ALG2PY = {'kmeans': KMeans,
	#           'kmedian': KMedian,
	#           }
	# model = ALG2PY[args['ALGORITHM']['name']]
	#
	# model.fit(X, y)
	#
	# history  = model.history
	#
	# return history

	args = config.parser(config_file)
	if os.path.exists(config_file): os.remove(config_file)
	fw = Framework(args)
	fw.run()

	return fw


if __name__ == '__main__':
	args = main(config_file='config.yaml')
	pprint(args, sort_dicts=False)
