"""

https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_plusplus.html#sphx-glr-auto-examples-cluster-plot-kmeans-plusplus-py

"""
import os
import pickle

from rkm.datasets.gaussian3 import gaussian3_diff_sigma_n
from rkm.utils.utils_func import timer, check_path


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
	N_CLIENTS = args['N_CLIENTS']
	N_REPEATS = args['N_REPEATS']
	N_CLUSTERS = args['N_CLUSTERS']
	data_file = os.path.join(args['IN_DIR'], dataset_name,  f'{dataset_detail}.dat')
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


	if dataset_name == '3GAUSSIANS':
		if 'diff_sigma_n' in dataset_detail:
			data = gaussian3_diff_sigma_n(args, random_state=SEED_DATA)
		# elif dataset_detail == '1client_1cluster':
		# 	data = gaussian3_1client_1cluster(params, random_state=SEED_DATA)
		# elif dataset_detail.split(':')[-1] == 'mix_clusters_per_client':
		# 	data =  gaussian3_mix_clusters_per_client(params, random_state=SEED_DATA)
		# elif dataset_detail == '1client_ylt0':
		# 	data =  gaussian3_1client_ylt0(params, random_state=SEED_DATA)
		# elif dataset_detail == '1client_xlt0':
		# 	data = gaussian3_1client_xlt0(params, random_state=SEED_DATA)
		# elif dataset_detail == '1client_1cluster_diff_sigma':
		# 	data = gaussian3_1client_1cluster_diff_sigma(params, random_state=SEED_DATA)
		# elif dataset_detail == '1client_xlt0_2':
		# 	data = gaussian3_1client_xlt0_2(params, random_state=SEED_DATA)
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