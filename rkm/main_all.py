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

"""
# Email: kun.bj@outllok.com
import copy
import os
import shutil
import traceback
from pprint import pprint

import numpy as np

from rkm import config, main_single

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
			n_clusters = 2
			# case = 'mixed_clusters'  # 'mixed_clusters'
			if case == 'diff_outliers':
				"""
				  r:0.1|mu:-3,0|cov:0.1,0.1|diff_outliers
				"""
				for _x in [-5, -12.5, -25, -50, -100, -200]:
					detail = f'r:0.1|mu:{_x},0|cov:0.1,0.1|diff_outliers'
					datasets.append({'name': dataset_name, 'detail': detail, 'n_clusters': n_clusters})
			elif case == 'mixed_clusters':
				"""
					d:2|r:0.4|mixed_clusters
				"""
				for d in [0.1, 0.5, 1, 2.5, 5, 10]:  # the distance between two clusters is 2*d.
					detail = f'd:{d}|r:0.4|mixed_clusters'
					datasets.append({'name': dataset_name, 'detail': detail, 'n_clusters': n_clusters})
			else:
				raise NotImplementedError

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
		if py_name == 'kmeans':
			for init_method in ['init']:  # ('random', None), ('kmeans++', None)
				algorithms.append({'py_name': py_name, 'name': name, 'n_clusters': n_clusters,
				                   'init_method': init_method})
		elif py_name == 'kmedian':
			for init_method in ['init']:  # ('random', None), ('kmeans++', None)
				algorithms.append({'py_name': py_name, 'name': name, 'n_clusters': n_clusters,
				                   'init_method': init_method})
		elif py_name == 'kmedian_tukey':
			for init_method in ['init']:  # ('random', None), ('kmeans++', None)
				algorithms.append({'py_name': py_name, 'name': name, 'n_clusters': n_clusters,
				                   'init_method': init_method, 'median_method': 'tukey_median'})
		else:
			msg = py_name
			raise NotImplementedError(msg)

	return algorithms


def main(N_REPEATS=1, OVERWRITE=True, IS_DEBUG=False, IS_GEN_DATA=True, VERBOSE=5, CASE=''):
	# get default config.yaml
	config_file = 'config.yaml'
	args = config.load(config_file)
	OUT_DIR = args['OUT_DIR']
	SEPERTOR = args['SEPERTOR']
	args['N_REPEATS'] = N_REPEATS
	args['OVERWRITE'] = OVERWRITE
	args['VERBOSE'] = VERBOSE

	tot_cnt = 0
	dataset_names = ['3GAUSSIANS'] #['NBAIOT']  #  '3GAUSSIANS'
	py_names = [
		'kmeans',
		'kmedian',
		'kmedian_tukey',
	]

	datasets = get_datasets_config_lst(dataset_names, case=CASE)
	for dataset in datasets:
		algorithms = get_algorithms_config_lst(py_names, dataset['n_clusters'])
		for i_alg, algorithm in enumerate(algorithms):
			if VERBOSE > 0: print(f'\n*** {tot_cnt}th experiment ***:', dataset['name'], algorithm['py_name'])
			args_lst = []
			for i_repeat in range(N_REPEATS):
				seed_data = i_repeat * 10  # data seed
				if VERBOSE > 1: print('***', dataset['name'], i_repeat, seed_data)
				args1 = copy.deepcopy(args)
				args1['SEED_DATA'] = seed_data
				args1['DATASET']['name'] = dataset['name']
				args1['DATASET']['detail'] = dataset['detail']
				args1['N_CLUSTERS'] = dataset['n_clusters']
				N_REPEATS = args1['N_REPEATS']
				N_CLUSTERS = args1['N_CLUSTERS']
				NORMALIZE_METHOD = args1['NORMALIZE_METHOD']
				args1['DATASET']['detail'] = os.path.join(f'{SEPERTOR}'.join([args1['DATASET']['detail'],
				                                                              NORMALIZE_METHOD, f'K_{N_CLUSTERS}']),
				                                          f'SEED_DATA_{seed_data}')
				dataset_detail = args1['DATASET']['detail']
				args1['ALGORITHM']['py_name'] = algorithm['py_name']
				args1['ALGORITHM']['init_method'] = algorithm['init_method']
				init_method = args1['ALGORITHM']['init_method']
				NORMALIZE_METHOD = args1['NORMALIZE_METHOD']
				args1['ALGORITHM']['detail'] = f'{SEPERTOR}'.join([f'R_{N_REPEATS}',
				                                                   f'{init_method}',
				                                                   f'{NORMALIZE_METHOD}'])
				args1['OUT_DIR'] = os.path.join(OUT_DIR, args1['DATASET']['name'], f'{dataset_detail}',
				                                args1['ALGORITHM']['py_name'], args1['ALGORITHM']['detail'])
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
					main_single.main(new_config_file)
				# return

				args_lst.append(copy.deepcopy(args1))

			tot_cnt += 1
			if not IS_GEN_DATA:
				gen_all_sh(
					args_lst)  # Be careful that multi-process will operate (generate and remove) the same dataset file

	print(f'*** Total cases: {tot_cnt}')


if __name__ == '__main__':
	for CASE in ['diff_outliers', 'mixed_clusters']:  # , 'mixed_clusters', 'diff_outliers',
		try:
			main(N_REPEATS=10, OVERWRITE=True, IS_DEBUG=True, VERBOSE=1, CASE=CASE)
		except Exception as e:
			traceback.print_exc()
