"""
    run:
        module load anaconda3/2021.5
        cd /scratch/gpfs/ky8517/rkm/rkm
        PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 collect_table_results.py
"""
# Email: kun.bj@outlook.com
import copy
import json
import os
import traceback

import numpy as np
import pandas as pd
import xlsxwriter

from rkm import config
from rkm.main_all import get_datasets_config_lst, get_algorithms_config_lst
from rkm.utils.utils_func import load

algorithm2abbrv = {'Centralized_true': 'True-CKM',
                   'Centralized_random': 'Random-CKM',
                   'Centralized_kmeans++': 'KM++-CKM',
                   'Federated-Server_random_min_max': 'Random-WA-rkm',
                   'Federated-Server_gaussian': 'Gaussian-WA-rkm',
                   'Federated-Server_average-Client_random': 'C-Random-WA-rkm',
                   'Federated-Server_average-Client_kmeans++': 'C-KM++-WA-rkm',
                   'Federated-Server_greedy-Client_random': 'C-Random-GD-rkm',
                   'Federated-Server_greedy-Client_kmeans++': 'C-KM++-GD-rkm',
                   }

metric2abbrv = {'iterations': 'Training iterations',
                'davies_bouldin': 'DB score',
                'silhouette': 'Silhouette',
                'euclidean': 'Euclidean distance'
                }


def save2csv(df, table_l2, py_name, case, column_idx, client_epochs):
	print(py_name, case, column_idx, client_epochs)
	dataset = case['dataset']
	data_details = case['data_details']
	algorithm = case['algorithm']
	params = get_experiment_params(p0=dataset, p1=data_details, p2=algorithm, client_epochs=client_epochs,
	                               p3=py_name)
	# pprint(params)
	out_dir = params['out_dir']
	print(f'out_dir: {out_dir}')
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	try:
		# read scores from out.txt
		server_init_centroids = params['server_init_centroids']
		client_init_centroids = params['client_init_centroids']
		if False:
			print('deprecated')
			out_txt = os.path.join(out_dir, f'varied_clients-Server_{server_init_centroids}-'
			                                f'Client_{client_init_centroids}.txt')
			with open(out_txt, 'r') as f:
				data = json.load(f)
			s = ' '
			for k, vs in data.items():
				for split in vs.keys():
					s += f'{split}:\n'
					for metric, score in vs[split].items():
						s += f'\t{metric}: ' + '+/-'.join(f'{v:.2f}' for v in score) + '\n'
					s += '\n'
		else:
			out_dat = os.path.join(out_dir, f'varied_clients-Server_{server_init_centroids}-'
			                                f'Client_{client_init_centroids}-histories.dat')
			histories = load(out_dat)
			for n_clients, history_res in histories.items():
				results_avg = history_res['results_avg']
				n_clients = history_res['n_clients']
				results = history_res['history']['results']
				s = ''
				c1 = ''
				training_iterations_lst = []
				scores_lst = []
				final_centroids_lst = []
				for vs in results:
					seed = vs['seed']
					# print(f'seed: {seed}')
					training_iterations_lst.append(vs['training_iterations'])
					scores_lst.append(vs['scores'])
				# final_centroids_lst += ['(' + ', '.join(f'{v:.5f}' for v in cen) + ')' for cen
				#                         in vs['final_centroids'].tolist()]

				for split in scores_lst[0].keys():  # ['train', 'test']
					# s += f'{split}:\n'
					# c1 += f'{split}:\n'
					if split == 'train':
						if column_idx == 0:
							c1 += 'iterations\n'
						s += f'{np.mean(training_iterations_lst):.2f} +/- ' \
						     f'{np.std(training_iterations_lst):.2f}\n'
					else:
						c1 += 'iterations\n'
						s += '\n'
					for metric in scores_lst[0][split].keys():
						metric_scores = [scores[split][metric] for scores in scores_lst]
						if column_idx == 0:
							c1 += f'{metric2abbrv[metric]}\n'
						s += f'{np.mean(metric_scores):.2f} +/- {np.std(metric_scores):.2f}\n'
				# s += '\n'

				# # final centroids distribution
				# s += 'final centroids distribution: \n'
				# ss_ = sorted(collections.Counter(final_centroids_lst).items(), key=lambda kv: kv[1], reverse=True)
				# tot_centroids = len(final_centroids_lst)
				# s += '\t\n'.join(f'{cen_}: {cnt_ / tot_centroids * 100:.2f}% - ({cnt_}/{tot_centroids})' for
				#                  cen_, cnt_ in ss_)
				if column_idx == 0:
					df['metric'] = c1.split('\n')
				df[algorithm2abbrv[algorithm]] = s.split('\n')
				break
	except Exception as e:
		print(f'Error: {e}')
		data = '-'


def main(N_REPEATS=1, OVERWRITE=True, IS_DEBUG=False, VERBOSE=5, IS_PCA=False, IS_REMOVE_OUTLIERS=False):
	# get default config.yaml
	config_file = 'config.yaml'
	args = config.load(config_file)
	OUT_DIR = '~/Downloads'
	SEPERTOR = args['SEPERTOR']
	args['N_REPEATS'] = N_REPEATS
	args['OVERWRITE'] = OVERWRITE
	args['VERBOSE'] = VERBOSE
	args['IS_PCA'] = IS_PCA
	args['IS_REMOVE_OUTLIERS'] = IS_REMOVE_OUTLIERS

	tot_cnt = 0
	# ['NBAIOT',  'FEMNIST', 'SENT140', '3GAUSSIANS', '10GAUSSIANS']
	# dataset_names = ['NBAIOT',  'FEMNIST', 'SENT140', '3GAUSSIANS', '10GAUSSIANS'] # ['NBAIOT'] # '3GAUSSIANS', '10GAUSSIANS', 'NBAIOT',  'FEMNIST', 'SENT140'
	dataset_names = ['NBAIOT', '3GAUSSIANS', '10GAUSSIANS', 'SENT140', 'FEMNIST', 'BITCOIN', 'CHARFONT', 'SELFBACK',
	                 'GASSENSOR', 'SELFBACK', 'MNIST']  #
	dataset_names = ['MNIST', 'BITCOIN', 'CHARFONT', 'DRYBEAN', 'GASSENSOR', 'SELFBACK']  #
	dataset_names = ['3GAUSSIANS', '10GAUSSIANS', 'NBAIOT', 'MNIST']  #
	# dataset_names = ['SELFBACK', 'GASSENSOR', 'MNIST', 'DRYBEAN']  # 'NBAIOT', '3GAUSSIANS'
	py_names = [
		'centralized_kmeans',
		'federated_server_init_first',  # server first: min-max per each dimension
		'federated_client_init_first',  # client initialization first : server average
		'federated_greedy_kmeans',  # client initialization first: greedy: server average
		# # 'Our_greedy_center',
		# 'Our_greedy_2K',
		# 'Our_greedy_K_K',
		# 'Our_greedy_concat_Ks',
		# 'Our_weighted_kmeans_initialization',
	]

	sheet_names = set()
	datasets = get_datasets_config_lst(dataset_names)
	for dataset in datasets:
		if dataset['name'] == '3GAUSSIANS' and IS_PCA == True: continue
		if dataset['name'] == '10GAUSSIANS' and IS_PCA == True: continue
		# if dataset['name'] == 'MNIST' and args['IS_PCA'] == True:
		# 	args['IS_PCA'] = 'CNN'
		# if dataset['name'] == 'MNIST' and args['IS_PCA'] == False:
		# 	continue
		# algorithms = get_algorithms_config_lst(py_names, dataset['n_clusters'])
		# print('\n***', dataset['name'], i_repeat, seed_data)
		args1 = copy.deepcopy(args)
		seed_data = 'NONE'
		args1['SEED_DATA'] = seed_data
		args1['DATASET']['name'] = dataset['name']
		args1['DATASET']['detail'] = dataset['detail']
		args1['N_CLIENTS'] = dataset['n_clients']
		args1['N_CLUSTERS'] = dataset['n_clusters']
		N_CLIENTS = args1['N_CLIENTS']
		N_REPEATS = args1['N_REPEATS']
		N_CLUSTERS = args1['N_CLUSTERS']
		NORMALIZE_METHOD = args1['NORMALIZE_METHOD']
		IS_PCA = args1['IS_PCA']
		IS_REMOVE_OUTLIERS = args1['IS_REMOVE_OUTLIERS']
		args1['DATASET']['detail'] = os.path.join(f'{SEPERTOR}'.join(
			[args1['DATASET']['detail'], NORMALIZE_METHOD, f'PCA_{IS_PCA}', f'M_{N_CLIENTS}', f'K_{N_CLUSTERS}',
			 f'REMOVE_OUTLIERS_{IS_REMOVE_OUTLIERS}']), f'SEED_DATA_{seed_data}')
		dataset_detail = args1['DATASET']['detail']

		args2 = copy.deepcopy(args1)
		algorithm = {'IS_FEDERATED':False, 'server_init_method':'Random', 'client_init_method':None, 'py_name':None}
		args2['IS_FEDERATED'] = algorithm['IS_FEDERATED']
		args2['ALGORITHM']['py_name'] = algorithm['py_name']
		# initial_method = args2['ALGORITHM']['initial_method']
		args2['ALGORITHM']['server_init_method'] = algorithm['server_init_method']
		server_init_method = args2['ALGORITHM']['server_init_method']
		args2['ALGORITHM']['client_init_method'] = algorithm['client_init_method']
		client_init_method = args2['ALGORITHM']['client_init_method']
		# args2['ALGORITHM']['name'] = algorithm['py_name'] + '_' + f'{server_init_method}|{client_init_method}'
		TOLERANCE = args2['TOLERANCE']
		NORMALIZE_METHOD = args2['NORMALIZE_METHOD']
		args2['ALGORITHM']['detail'] = f'{SEPERTOR}'.join([f'R_{N_REPEATS}',
		                                                   f'{server_init_method}|{client_init_method}',
		                                                   f'{TOLERANCE}', f'{NORMALIZE_METHOD}'])
		# args2['OUT_DIR'] = os.path.join(OUT_DIR, args2['DATASET']['name'], f'{dataset_detail}',
		#                                 args2['ALGORITHM']['py_name'], args2['ALGORITHM']['detail'])

		ALG2ABBREV = {
			f'centralized_kmeans|R_{N_REPEATS}|random|None|{TOLERANCE}|{NORMALIZE_METHOD}': 'CKM-R',
			f'centralized_kmeans|R_{N_REPEATS}|kmeans++|None|{TOLERANCE}|{NORMALIZE_METHOD}': 'CKM++',
			# f'federated_server_init_first|R_{N_REPEATS}|random|None|{TOLERANCE}|{NORMALIZE_METHOD}': 'Server-Initialized',
			f'federated_server_init_first|R_{N_REPEATS}|min_max|None|{TOLERANCE}|{NORMALIZE_METHOD}': 'Server-MinMax',
			f'federated_client_init_first|R_{N_REPEATS}|average|random|{TOLERANCE}|{NORMALIZE_METHOD}': 'Average-Random',
			f'federated_client_init_first|R_{N_REPEATS}|average|kmeans++|{TOLERANCE}|{NORMALIZE_METHOD}': 'Average-KM++',
			f'federated_greedy_kmeans|R_{N_REPEATS}|greedy|random|{TOLERANCE}|{NORMALIZE_METHOD}': 'Greedy-Random',
			f'federated_greedy_kmeans|R_{N_REPEATS}|greedy|kmeans++|{TOLERANCE}|{NORMALIZE_METHOD}': 'Greedy-KM++',
		}

		csv_file = os.path.join(OUT_DIR, 'xlsx', args2['DATASET']['name'], f'{os.path.dirname(dataset_detail)}',
		                         args2['ALGORITHM']['detail'] + '.csv')

		try:
			df = pd.read_csv(csv_file)
			out_csv = os.path.join(OUT_DIR, 'xlsx', args2['DATASET']['name'], f'{os.path.dirname(dataset_detail)}',
			                         args2['ALGORITHM']['detail'] + '-paper.csv')
			out_csv2 = os.path.join(OUT_DIR, 'xlsx', args2['DATASET']['name'], f'{os.path.dirname(dataset_detail)}',
			                       args2['ALGORITHM']['detail'] + '-paper2.csv')
			df = df.T
			# df.iloc[0, :] = ['CKM-R', 'CKM++', 'Server-MinMax', 'Average-Random',
			#                  'Average-KM++', 'Greedy-Random', 'Greedy-KM++']
			lst = ['CKM-Random', 'CKM++', 'Server-MinMax', 'Average-Random',
			                 'Average-KM++', 'Greedy-Random', 'Greedy-KM++']
			for i in range(7):
				df.iloc[0, i] = lst[i]
			df.iloc[[0, 1, 10, 11], :].to_csv(out_csv, sep = ',')
			# df.iloc[[0, 1, 10, 11], :].to_csv(out_csv2, sep=str('&'))
			tmp = df.iloc[[0, 1, 10, 11], :]
			with open(os.path.expanduser(out_csv2), 'w') as f:
				np.savetxt(f, np.concatenate([tmp.index.to_numpy().reshape((-1, 1)), tmp.values], axis=1), delimiter=' & ', fmt='%s')
		except Exception as e:
			print(e)

		print(out_csv)
	print(f'*** Total cases: {tot_cnt}')

if __name__ == '__main__':
	# tot_cnt = main()
	# print()
	# print(f'*** Total cases: {tot_cnt}')
	for IS_REMOVE_OUTLIERS in [False]:
		for IS_PCA in [False, True]:
			main(N_REPEATS=50, OVERWRITE=False, IS_DEBUG=False, VERBOSE=2, IS_PCA = IS_PCA, IS_REMOVE_OUTLIERS = IS_REMOVE_OUTLIERS)

