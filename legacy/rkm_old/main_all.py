""" Run this main file for all experiments

	#ssh ky8517@tigergpu.princeton.edu
	srun --time=2:00:00 --pty bash -i
	srun --nodes=1  --mem=128G --ntasks-per-node=1 --time=20:00:00 --pty bash -i
	#srun --nodes=1 --gres=gpu:1 --mem=128G --ntasks-per-node=1 --time=20:00:00 --pty bash -i
	cd /scratch/gpfs/ky8517/rkm/rkm
	module load anaconda3/2021.11

	# check why the process is killed
	dmesg -T| grep -E -i -B100 'killed process'

	PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main_all.py

	# delete all empty directories recursively
	https://unix.stackexchange.com/questions/46322/how-can-i-recursively-delete-empty-directories-in-my-home-directory

	find . -type d -empty -print
	find . -type d -empty -delete

"""
# Email: kun.bj@outllok.com
import copy
import os
import shutil
import time
import traceback
from pprint import pprint

import numpy as np

from rkm import config, main_single
from rkm.utils.common import dump

np.set_printoptions(precision=3, suppress=True)

def gen_all_sh(args_lst):
	"""

	Parameters
	----------
	py_name
	case

	Returns
	-------

	"""
	args = args_lst[0]
	# check_arguments()
	dataset_name = args['DATASET']['name']
	dataset_detail = os.path.dirname(args['DATASET']['detail'])
	# n_clients = args['N_CLIENTS']
	# algorithm_py_name = args['ALGORITHM']['py_name']
	# # algorithm_name = args['ALGORITHM']['name']
	# algorithm_detail = args['ALGORITHM']['detail']
	# # n_clusters = args['ALGORITHM']['n_clusters']
	# config_file = args['config_file']
	# OUT_DIR = os.path.join(*args['OUT_DIR'].split('/')[:4])
	OUT_DIR = args['OUT_DIR']

	job_name = f'{dataset_name}-{dataset_detail}'
	# tmp_dir = '~tmp'
	# if not os.path.exists(tmp_dir):
	# 	os.system(f'mkdir {tmp_dir}')
	if '2GAUSSIANS' in dataset_name:
		t = 48
	# elif 'FEMNIST' in dataset_name and 'greedy' in algorithm_py_name:
	# 	t = 48
	else:
		t = 48
	content = fr"""#!/bin/bash
#SBATCH --job-name={OUT_DIR}         # create a short name for your job
#SBATCH --time={t}:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output={OUT_DIR}/out.txt
#SBATCH --error={OUT_DIR}/err.txt
### SBATCH --mail-type=end          # send email when job ends
###SBATCH --mail-user=ky8517@princeton.edu     # which will cause too much email notification. \

## SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
## SBATCH --output={OUT_DIR}/%j-{job_name}-out.txt
## SBATCH --error={OUT_DIR}/%j-{job_name}-err.txt
## SBATCH --nodes=1                # node count
## SBATCH --ntasks=1               # total number of tasks across all nodes
## SBATCH --cpus-per-task=5        # cpu-cores per task (>1 if multi-threaded tasks)
## SBATCH --mem=40G\
### SBATCH --mail-type=begin        # send email when job begins
### SBATCH --mail-user=kun.bj@cloud.com # not work 
### SBATCH --mail-user=<YourNetID>@princeton.edu\

module purge
module load anaconda3/2021.11   # Python 3.9.7
pip3 install nltk 

cd /scratch/gpfs/ky8517/rkm/rkm 
pwd
python3 -V

"""
	content += "\nexport PYTHONPATH=\"${PYTHONPATH}:..\" \n"
	for i, args in enumerate(args_lst):
		config_file_ = args['config_file']
		out_dir_ = args['OUT_DIR']
		# content += '\n' + f"PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 {algorithm_py_name} --dataset '{dataset_name}' " \
		#                   f"--data_details '{dataset_detail}' --algorithm '{algorithm_name}' --n_clusters '{n_clusters}' --n_clients '{n_clients}' \n"
		content += f"\nPYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main_single.py --config_file '{config_file_}' > '{out_dir_}/a.txt' 2>&1  &\n"
		# if i == 0: 	# let the first one run first to generate the data and avoid unknown conflict for the rest.
		# 	content += "echo 'Finish the first one first...'\n"
		content += "\nwait\n"  # must has this line
		content += "echo $!\n"  # stores the background process PID
		content += "echo $?\n"  # $? stores the exit status.
		content += f"echo \'Finish {i}-th args.\'"  # $? stores the exit status.
	# single quote (just treat the content inside it as a string) vs double quote in bash
	# $ echo "$(echo "upg")"
	# upg
	# $ echo '$(echo "upg")'
	# $(echo "upg")
	#
	content += "\nwait\n"  # must has this line
	# The bash wait command is a Shell command that waits for background running processes to complete and returns the exit status.
	# Without any parameters, the wait command waits for all background processes to finish before continuing the script.
	content += "echo $!\n"  # stores the background process PID
	content += "echo $?\n"  # $? stores the exit status.
	content += "\necho \'done\n"
	# sh_file = f'{OUT_DIR}/{dataset_name}-{dataset_detail}-{algorithm_name}-{algorithm_detail}.sh'
	sh_file = f'{OUT_DIR}/sbatch.sh'
	with open(sh_file, 'w') as f:
		f.write(content)
	cmd = f"sbatch '{sh_file}'"
	print(cmd)
	os.system(cmd)


def get_datasets_config_lst(dataset_names=['3GAUSSIANS', '10GAUSSIANS'], case=''):
	datasets = []
	for dataset_name in dataset_names:
		if dataset_name == '3GAUSSIANS':
			# case = 'mixed_clusters'  # 'mixed_clusters'
			if case == 'diff_outliers':
				"""
				  r:0.1|mu:-3,0|cov:1.0.,0.3|diff_outliers
				  
				  2 normal clsuters generated by 2 Gaussians; 
				  1 noise Gaussians with different means mu in [-5, -12.5, -25, -50, -100, -200]
				  
				"""
				n_clusters = 2
				for mu in [2, 3, 4, 5, 6]: #[2, 2.5, 3, 3.5, 4, 4.5, 5]: # -100, -200]:
					detail = f'r:0.1|mu:{mu},0|cov:1.0,1.0|diff_outliers'
					datasets.append({'name': dataset_name, 'detail': detail, 'n_clusters': n_clusters})
			elif case == 'diff2_outliers':
				"""
				  r:0.1|mu:-3,0|cov:1.0.,0.3|diff_outliers

				  2 normal clsuters generated by 2 Gaussians; 
				  1 noise Gaussians with different means mu in [-5, -12.5, -25, -50, -100, -200]

				"""
				n_clusters = 2
				for cov in [0.5, 1.0, 2.0, 3.0, 4.0]: #[0.5, 1, 2, 4, 8]:
					detail = f'r:0.1|mu:0,0|cov:{cov},{cov}|diff2_outliers'
					datasets.append({'name': dataset_name, 'detail': detail, 'n_clusters': n_clusters})
			elif case == 'diff3_outliers':
				"""
				  r:0.1|mu:-3,0|cov:1.0.,0.3|diff_outliers

				  2 normal clsuters generated by 2 Gaussians; 
				  1 noise Gaussians with different means mu in [-5, -12.5, -25, -50, -100, -200]

				"""
				n_clusters = 3
				for mu in [3, 6, 9, 12, 15]:  # [2, 2.5, 3, 3.5, 4, 4.5, 5]: # -100, -200]:
					detail = f'r:0.1|mu:{mu},0|cov:1.0,1.0|diff3_outliers'
					datasets.append({'name': dataset_name, 'detail': detail, 'n_clusters': n_clusters})
			elif case == 'mixed_clusters':
				"""
					d:2|r:0.4|mixed_clusters
					2 normal Gaussians with different differences between the two Gaussians' means
				"""
				n_clusters = 2
				for d in [0.1, 0.5, 1, 2.5, 5, 10]:  # the distance between two clusters is 2*d.
					detail = f'd:{d}|r:0.4|mixed_clusters'
					datasets.append({'name': dataset_name, 'detail': detail, 'n_clusters': n_clusters})
			elif case == 'constructed_3gaussians':
				"""
					p:0.4|constructed_3gaussians
					Constructed example, in which mean doesn't work; however, median works. 
					3 Gaussians: 
						the first Gaussian with mu= [-1, 0] and covariance = [1, 1], n1=1000;
						the second Gaussian with mu= [1, 0] and covariance = [1, 1], n2=1000; and 
						the last Gaussian with mu= [10, 0] and covariance = [3, 3], n3=2000;
						
						# For the initialization, 
						The first cluster with the mean of Gaussian 1 as its centroid;
						The second cluster with the mean of (100% Gaussian 2 + p (e.g., 20%) of data from Gaussian 3) as its centroid; and 
						The third cluster with the mean of (1-p) of data from Gaussian 3 as its centroid. 

				"""
				n_clusters = 3
				ps = [0.05, 0.10, 0.20, 0.35, 0.49]
				# ps = [v * 0.01 for v in range(0, 50+1, 5)]
				print(ps)
				for p in ps:
					detail = f'p:{p:.2f}|constructed_3gaussians'
					datasets.append({'name': dataset_name, 'detail': detail, 'n_clusters': n_clusters})
			elif case == 'constructed2_3gaussians':
				"""
					p:0.4|constructed2_3gaussians
					Constructed example, in which mean doesn't work; however, median works. 
					3 Gaussians: 
						the first Gaussian with mu= [-1, 0] and covariance = [1, 1], n1=1000;
						the second Gaussian with mu= [1, 0] and covariance = [1, 1], n2=1000; and 
						the last Gaussian with mu= [10, 0] and covariance = [3, 3], n3=2000;

						# For the initialization, 
						The first cluster with the mean of Gaussian 1 as its centroid;
						The second cluster with the mean of (100% Gaussian 2 + p (e.g., 20%) of data from Gaussian 3) as its centroid; and 
						The third cluster with the mean of (1-p) of data from Gaussian 3 as its centroid. 

				"""
				n_clusters = 3
				# ps = [0.01, 0.05, 0.10, 0.25, 0.30, 0.40, 0.45, 0.49]
				ps = [0.05, 0.10, 0.20, 0.35, 0.49]
				# ps =[0.49]
				# ps = [v * 0.01 for v in range(1, 50 + 1, 5)]
				print(ps)
				for p in ps:
					detail = f'p:{p:.2f}|constructed2_3gaussians'
					datasets.append({'name': dataset_name, 'detail': detail, 'n_clusters': n_clusters})
			else:
				# raise NotImplementedError
				continue
		elif dataset_name == '10GAUSSIANS':
			if case == 'gaussians10_snr':
				"""
					10 gaussians10_snr
				"""
				n_clusters = 10
				SNRs = [6, 7, 8, 9]
				print(SNRs)
				for SNR in SNRs:
					detail = f'p:{SNR:.2f}|gaussians10_snr'
					datasets.append({'name': dataset_name, 'detail': detail, 'n_clusters': n_clusters})
			elif case == 'gaussians10_covs':
				"""
					10 gaussians10_covs
				  r:0.1|mu:-3,0|cov:1.0.,0.3|diff_outliers

				  2 normal clsuters generated by 2 Gaussians; 
				  1 noise Gaussians with different means mu in [-5, -12.5, -25, -50, -100, -200]

								"""
				n_clusters = 5
				for cov in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]:  # [0.5, 1, 2, 4, 8]:
					detail = f'r:0.1|mu:0,0|cov:{cov},{cov}|gaussians10_covs'
					datasets.append({'name': dataset_name, 'detail': detail, 'n_clusters': n_clusters})
			elif case == 'gaussians10_ds':
				"""
					10 gaussians10_ds
				  d:20|mu:0,0|cov:5.0.,5.0|gaussians10_ds

								"""
				# n_clusters = 2
				# [1, 2, 3, 4, 5, 10]
				for n_clusters in [3, 5]: #[10, 50, 100]:
					# [5, 6, 7, 8, 9, 10]: #
					for dim in [5, 6, 7, 8, 9, 10]: #[5, 6, 7, 8, 9, 10, 20, 50, 100, 200]:  # [0.5, 1, 2, 4, 8]:
						for r in [0.1]: #[0.1, 0.3], [0.05, 0.1, 0.15, 0.2, 0.25]:
							cov = 5.0
							detail = f'd:{dim}_r:{r:.2f}|mu:0,0|cov:{cov},{cov}|gaussians10_ds'
							datasets.append({'name': dataset_name, 'detail': detail, 'n_clusters': n_clusters})

				# for cov in [0.5, 1.0, 5.0, 10.0, 50.0]:
				# 	dim = 10
				# 	r = 0.1
				# 	detail = f'd:{dim}_r:{r:.2f}|mu:0,0|cov:{cov},{cov}|gaussians10_ds'
				# 	datasets.append({'name': dataset_name, 'detail': detail, 'n_clusters': n_clusters})


			elif case == 'gaussians10_random_ds':
				"""
					10 gaussians10_ds_random
				  d:20|mu:0,0|cov:2.0.,2.0|gaussians10_random_ds

								"""
				n_clusters = 5
				#
				for dim in [5, 6, 7, 8, 9, 10, 20]: #[5, 6, 7, 8, 9, 10, 20, 50, 100, 200]:  # [0.5, 1, 2, 4, 8]:
					detail = f'd:{dim}|mu:0,0|cov:5.0,5.0|gaussians10_random_ds'
					datasets.append({'name': dataset_name, 'detail': detail, 'n_clusters': n_clusters})
			elif case == 'gaussians10_ks':
				"""
					10 gaussians10_ks
				"""
				n_clusters_lst = [5, 10, 25, 50]
				print(n_clusters_lst)
				for n_clusters in n_clusters_lst:
					detail = f'p:{n_clusters:.2f}|gaussians10_ks'
					datasets.append({'name': dataset_name, 'detail': detail, 'n_clusters': n_clusters})
			else:
				# raise NotImplementedError
				continue
		elif dataset_name == 'GALAXY':
			if case == 'two_galaxy_clusters_n':
				"""
					two_galaxy_clusters_n
				"""
				ns = [1000, 2000, 3000, 4000, 5000, 35000]
				ns = [5000]
				n_clusters = 2
				for n in ns:
					detail = f'n:{n}|two_galaxy_clusters_n'
					datasets.append({'name': dataset_name, 'detail': detail, 'n_clusters': n_clusters})
			elif case == 'two_galaxy_clusters_p':
				"""
					two_galaxy_clusters_p
				"""
				ps = [0.05, 0.10, 0.20, 0.35, 0.49]
				# ps =[0.05, 0.49]
				# ps = [v * 0.01 for v in range(1, 50 + 1, 5)]
				print(ps)
				n_clusters = 2
				for p in ps:
					detail = f'p:{p:.2f}|two_galaxy_clusters_p'
					datasets.append({'name': dataset_name, 'detail': detail, 'n_clusters': n_clusters})
			else:
				# raise NotImplementedError
				continue
		elif dataset_name == 'NBAIOT':
			n_clusters = 2
			# case = 'mixed_clusters'  # 'mixed_clusters'
			if case == 'diff_outliers':
				"""
				  r:0.1|mu:-3,0|cov:0.1,0.1|diff_outliers
				"""
				for _x in [-5, -12.5, -25, -50, -100, -200]:#[-5, -12.5, -25, -50, -100, -200]:
					detail = f'r:0.1|mu:{_x},0|cov:0.1,0.1|diff_outliers'
					datasets.append({'name': dataset_name, 'detail': detail, 'n_clusters': n_clusters})
			elif case == 'mixed_clusters':
				"""
					d:2|r:0.4|mixed_clusters
				"""
				for d in [0.1, 0.5, 1, 2.5, 5, 10]: #[0.1, 0.5, 1, 2.5, 5, 10]:  # the distance between two clusters is 2*d.
					detail = f'd:{d}|r:0.4|mixed_clusters'
					datasets.append({'name': dataset_name, 'detail': detail, 'n_clusters': n_clusters})
			else:
				raise NotImplementedError
		else:
			msg = f'{dataset_name}'
			raise NotImplementedError(msg)

	return datasets


def get_algorithms_config_lst(py_names, n_clusters=2):
	algorithms = []
	for py_name in py_names:
		cnt = 0
		name = None
		# init_methods = ['omniscient', 'random', 'kmeans++']		# 'noise'
		init_methods = ['omniscient']
		# init_methods = ['random']
		if py_name == 'kmeans':
			for init_method in init_methods: #['omniscient', 'random', 'kmeans++']:  # ('random', None), ('kmeans++', None)
				algorithms.append({'py_name': py_name, 'name': name, 'n_clusters': n_clusters,
				                   'init_method': init_method})
		elif py_name == 'kmedian':
			for init_method in init_methods:  # ('random', None), ('kmeans++', None)
				algorithms.append({'py_name': py_name, 'name': name, 'n_clusters': n_clusters,
				                   'init_method': init_method})
		elif py_name == 'kmedian_l1':
			for init_method in init_methods:#, 'kmeans++', 'omniscient']:  # ('random', None), ('kmeans++', None)
				algorithms.append({'py_name': py_name, 'name': name, 'n_clusters': n_clusters,
				                   'init_method': init_method})
		elif py_name == 'my_spectralclustering':
			for init_method in [None]:#, 'kmeans++', 'omniscient']:  # ('random', None), ('kmeans++', None)
				algorithms.append({'py_name': py_name, 'name': name, 'n_clusters': n_clusters,
				                   'init_method': init_method})
		elif py_name == 'kmedian_tukey':
			for init_method in ['omniscient']:  # ('random', None), ('kmeans++', None)
				algorithms.append({'py_name': py_name, 'name': name, 'n_clusters': n_clusters,
				                   'init_method': init_method, 'median_method': 'tukey_median'})
		else:
			msg = py_name
			raise NotImplementedError(msg)

	return algorithms


def main(N_REPEATS=1, OVERWRITE=True, IS_DEBUG=False, IS_GEN_DATA=True, VERBOSE=5, CASE='', py_names = ['kmeans'],
		 is_gen_data=True):
	# get default config.yaml
	config_file = 'config.yaml'
	args = config.load(config_file)
	OUT_DIR = args['OUT_DIR']
	SEPERTOR = args['SEPERTOR']
	args['N_REPEATS'] = N_REPEATS
	args['OVERWRITE'] = OVERWRITE
	args['VERBOSE'] = VERBOSE

	tot_cnt = 0
	dataset_names = ['10GAUSSIANS'] #['NBAIOT', 'GALAXY']  #  '3GAUSSIANS', '10GAUSSIANS', '3GAUSSIANS',
	# py_names = [
	# 	'kmeans',
	# 	'kmedian_l1',
	# 	'kmedian',  # our method
	# 	# 'kmedian_tukey',
	# ]

	datasets = get_datasets_config_lst(dataset_names, case=CASE)
	for dataset in datasets:
		algorithms = get_algorithms_config_lst(py_names, dataset['n_clusters'])
		for i_alg, algorithm in enumerate(algorithms):
			# if algorithm['init_method'] != 'omniscient':continue
			if i_alg>0 and is_gen_data: continue
			if VERBOSE > 0: print(f'\n*** {tot_cnt}th experiment ***:', dataset['name'], algorithm['py_name'])
			args_lst = []
			for i_repeat in range(N_REPEATS):
				seed_step = 1000
				# seed_step = 40000 # for testing
				seed = i_repeat * seed_step  # data seed
				seed_step2 = seed_step//1       # repeats 100 times in the inner loop
				seeds2_results = {}
				for seed2 in range(seed, seed+seed_step, seed_step2):
					if VERBOSE > 1: print('***', dataset['name'], i_repeat, seed, seed2)
					args1 = copy.deepcopy(args)
					args1['SEED_1'] = seed
					args1['SEED_DATA'] = seed2
					args1['DATASET']['name'] = dataset['name']
					args1['DATASET']['detail'] = dataset['detail']
					args1['N_CLUSTERS'] = dataset['n_clusters']
					N_REPEATS = args1['N_REPEATS']
					N_CLUSTERS = args1['N_CLUSTERS']
					NORMALIZE_METHOD = args1['NORMALIZE_METHOD']
					args1['DATASET']['detail'] = os.path.join(f'{SEPERTOR}'.join([args1['DATASET']['detail'],
					                                                              NORMALIZE_METHOD, f'K_{N_CLUSTERS}']),
					                                          f'SEED_{seed}', f'SEED2_{seed2}')
					dataset_detail = args1['DATASET']['detail']
					args1['ALGORITHM']['py_name'] = algorithm['py_name']
					args1['ALGORITHM']['init_method'] = algorithm['init_method']
					init_method = args1['ALGORITHM']['init_method']
					NORMALIZE_METHOD = args1['NORMALIZE_METHOD']
					args1['ALGORITHM']['detail'] = f'{SEPERTOR}'.join([f'R_{N_REPEATS}',
					                                                   f'{init_method}',
					                                                   f'{NORMALIZE_METHOD}'])
					args1['OUT_DIR'] = os.path.join(OUT_DIR, f'R_{N_REPEATS}', args1['DATASET']['name'], f'{dataset_detail}',
					                                args1['ALGORITHM']['py_name'], args1['ALGORITHM']['detail'])
					args1['is_gen_data'] = is_gen_data
					if os.path.exists(args1['OUT_DIR']):
						shutil.rmtree(args1['OUT_DIR'])
					# shutil.rmtree(os.path.join(OUT_DIR, args2['DATASET']['name'], f'{dataset_detail}'))
					else:
						os.makedirs(args1['OUT_DIR'])
					new_config_file = os.path.join(args1['OUT_DIR'], 'config_file.yaml')
					if VERBOSE >= 2: pprint(new_config_file)
					config.dump(new_config_file, args1)
					args1['config_file'] = new_config_file
					if VERBOSE >= 5: pprint(args1, sort_dicts=False)
					if IS_DEBUG:
						### run a single experiment for debugging the code
						print(f'{dataset}, {algorithm}, seed:{seed}, seed2:{seed2}')
						single_res = main_single.main(new_config_file)
						if args1['is_gen_data']: continue
						if 'X' in single_res.data.keys():
							del single_res.data['X']
							del single_res.data['y']
						single_res.history['data'] = single_res.data
						seeds2_results[f"SEED2_{seed2}"] = single_res.history
					# return
					args_lst.append(copy.deepcopy(args1))
					tot_cnt += 1
				# save results
				seed_our_dir = single_res.args['OUT_DIR'].replace(f'SEED2_{seed2}/', '')
				out_file = os.path.join(seed_our_dir, f'seeds2.dat')
				dump(seeds2_results, out_file)
			if not IS_GEN_DATA:
				# Be careful that multi-process will operate (generate and remove) the same dataset file
				gen_all_sh(args_lst)

	print(f'*** Total cases: {tot_cnt}')
	# return tot_cnt


def main_call0(kwargs):
	try:
		main(N_REPEATS=kwargs['N_REPEATS'], OVERWRITE=True, IS_DEBUG=True, VERBOSE=1, CASE=kwargs["CASE"],
		 py_names=kwargs["py_names"],is_gen_data=True)
	except Exception as e:
		traceback.print_exc()


def main_call(kwargs):
	try:
		main(N_REPEATS=kwargs['N_REPEATS'], OVERWRITE=False, IS_DEBUG=True, VERBOSE=1, CASE=kwargs["CASE"],
		 py_names=kwargs["py_names"],is_gen_data=False)
	except Exception as e:
		traceback.print_exc()

if __name__ == '__main__':
	# For debugging.
	# main(N_REPEATS=2, OVERWRITE=True, IS_DEBUG=True, VERBOSE=1, CASE='diff3_outliers',
	# is_gen_data=True, py_names=['my_spectralclustering'])
	# main(N_REPEATS=2, OVERWRITE=False, IS_DEBUG=True, VERBOSE=100, CASE='diff2_outliers',
	# 	 is_gen_data=False, py_names=['kmedian'])
	# # Call collect_results.sh to get the plot
	# #./collect_results.sh
	# exit(0)

	from multiprocessing import Pool
	st = time.time()
	N_REPEATS = 1000
	print(f'n_repeats: {N_REPEATS}')
	in_dir = 'datasets/3GAUSSIANS'
	if os.path.exists(in_dir):
		shutil.rmtree(in_dir)
	# 1. recreate all the datasets
	cases = [
			# 'diff_outliers', 'diff2_outliers',
			 # 'diff3_outliers',
			 # 'constructed_3gaussians', 'constructed2_3gaussians',
			 # 'two_galaxy_clusters_n', 'two_galaxy_clusters_p',
			 # 'gaussians10_snr', 'gaussians10_ks'
			'gaussians10_ds',
			# 'gaussians10_random_ds',
			# 'gaussians10_covs',
			 ]
	# cases = ['two_galaxy_clusters_n']
	list_ranges = []
	for CASE in cases:  # , 'mixed_clusters', 'diff_outliers', 'constructed_3gaussians', 'constructed2_3gaussians
		list_ranges.append({'CASE': CASE, 'py_names': ['kmeans'], 'N_REPEATS': N_REPEATS})
	# pool object with number of elements in the list
	pool = Pool(processes=len(list_ranges))
	# map the function to the list and pass
	# function and list_ranges as arguments
	pool.map(main_call0, list_ranges)

	# 2. Get the results
	# No need to create dataset, OVERWRITE=False
	# cases = ['diff_outliers', 'diff2_outliers', 'constructed_3gaussians', 'constructed2_3gaussians']
	py_names = [
		'kmeans',
		'kmedian_l1',
		'kmedian',  # our method
		# # 'kmedian_tukey',
		# 'my_spectralclustering',
	]
	list_ranges = []
	for CASE in cases:  # , 'mixed_clusters', 'diff_outliers', 'constructed_3gaussians', 'constructed2_3gaussians
		for py_name in py_names:
			list_ranges.append({'CASE':CASE, 'py_names': [py_name], 'N_REPEATS': N_REPEATS})

	# pool object with number of elements in the list
	pool = Pool(processes=len(list_ranges))
	# map the function to the list and pass
	# function and list_ranges as arguments
	pool.map(main_call, list_ranges)

	ed = time.time()
	print(f'Total: {ed-st}s.')
