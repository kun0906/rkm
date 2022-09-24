"""

https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_plusplus.html#sphx-glr-auto-examples-cluster-plot-kmeans-plusplus-py

"""
import os
import pickle

from rkm.datasets.gaussian3 import gaussian3_diff_outliers, gaussian3_mixed_clusters
from rkm.utils.common import timer, check_path


@timer
def generate_dataset(args):
	"""

	Parameters
	----------
	args

	Returns
	-------

	"""
	# SEED = args['SEED']
	SEED_DATA = args['SEED_DATA']
	dataset_name = args['DATASET']['name']
	dataset_detail = args['DATASET']['detail']
	data_file = os.path.join(args['IN_DIR'], dataset_name, f'{dataset_detail}.dat')
	# print(data_file)
	check_path(data_file)
	args['data_file'] = data_file
	if args['OVERWRITE'] and os.path.exists(data_file):
		# here could be some issue for multi-tasks, please double-check before calling this function.
		os.remove(data_file)
	elif os.path.exists(data_file):
		return data_file
	else:
		print('Generate dataset')

	print(f'SEED_DATA: {SEED_DATA}')
	if dataset_name == '3GAUSSIANS':
		if 'diff_outliers' in dataset_detail:
			data = gaussian3_diff_outliers(args, random_state=SEED_DATA)
		elif 'mixed_clusters' in dataset_detail:
			data = gaussian3_mixed_clusters(args, random_state=SEED_DATA)
		else:
			msg = f'{dataset_name}, {dataset_detail}'
			raise NotImplementedError(msg)
	else:
		msg = f'{dataset_name}, {dataset_detail}'
		raise NotImplementedError(msg)

	check_path(data_file)

	with open(data_file, 'wb') as f:
		pickle.dump(data, f)

	return data_file
